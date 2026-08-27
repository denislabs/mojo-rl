"""BlockLinear GPU batched-GEMM path — CPU↔GPU parity gate.

The scalar one-thread-per-element BlockLinear kernels were ~40× off GEMM
speed (DreamerV3 Pong profile: the imagination GRU's two BlockLinear forwards
= 21.2 + 14.9 ms/call ≈ the whole AC section at size200m). The GPU path now
gathers to block-major and runs `batched_matmul` over the BLK dim (fwd, dW,
dx). This gates fwd/dx/dW/db CPU↔GPU at:
  1. a Dreamer-like rectangular shape (IN≠OUT, BLOCKS=8) — GEMM path
  2. BLOCKS=1 dense — GEMM path (degenerate batch)
  3. OPB=1 (OUT==BLOCKS) — scalar fallback (bmm N=1 would silently
     miscompute: feedback_max_matmul_n1_gpu_miscompute)
Also checks grad ACCUMULATION (two vjp calls → doubled param grads).

Run:  pixi run -e apple  mojo run -I . tests/nn/test_block_linear_gemm_gpu.mojo
      pixi run -e nvidia mojo run -I . tests/nn/test_block_linear_gemm_gpu.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.block_linear import BlockLinear

# NVIDIA linalg.bmm can dispatch to cutlass TF32 tensor-op kernels
# (s1688gemm) whose 10-bit-mantissa input quantization yields |Δ| ~ 1e-3·|acc|
# vs the fp32 CPU reference, growing with K (observed 7.1e-4 at K=64 while
# K=32 shapes passed 2e-4 and Apple — fp32 GEMMs — was exact). Same lesson as
# feedback_fd_gradcheck_tf32: numeric backend, not an indexing bug (an
# indexing bug shows as O(1) error). Tolerance sized for TF32 at these K.
comptime TOL = Scalar[DT](3e-3)


def _maxdiff(a: Tensor, b: Tensor, n: Int) -> Scalar[DT]:
    var md = Scalar[DT](0.0)
    for i in range(n):
        var d = a.data[i] - b.data[i]
        d = d if d >= Scalar[DT](0.0) else -d
        if d > md:
            md = d
    return md


def _one[IN: Int, OUT: Int, BLK: Int, B: Int](
    ctx: DeviceContext, label: String, accum_check: Bool
) raises:
    print("  BlockLinear[", IN, ",", OUT, ",", BLK, "] B =", B, "(", label, ")")
    comptime BL = BlockLinear[IN, OUT, BLK]
    var octx = Optional[DeviceContext](ctx)
    seed(9)
    var mc = BL.make["cpu", Kaiming](None)
    seed(9)
    var mg = BL.make["gpu", Kaiming](octx)

    seed(77)
    var x = Tensor.alloc(B * IN)
    var xg = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        xg.data[i] = x.data[i]
    xg.upload(ctx)
    var go = Tensor.alloc(B * OUT)
    var gog = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](random_float64() - 0.5)
        gog.data[i] = go.data[i]
    gog.upload(ctx)

    # forward
    var oc = Tensor()
    mc.forward["cpu", B](TensorRefs[1](x), oc, None)
    var og = Tensor()
    mg.forward["gpu", B](TensorRefs[1](xg), og, octx)
    og.download(ctx)
    var md = _maxdiff(oc, og, B * OUT)
    assert_true(md < TOL, "fwd maxdiff " + String(md))

    # vjp (possibly twice → accumulation check on param grads)
    var reps = 2 if accum_check else 1
    var gc = Tensor()
    var gg = Tensor()
    mc.zero_grad["cpu"](None)
    mg.zero_grad["gpu"](octx)
    for _r in range(reps):
        mc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gc), None)
        mg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gg), octx)
    gg.download(ctx)
    md = _maxdiff(gc, gg, B * IN)
    assert_true(md < TOL, "dx maxdiff " + String(md))
    mg.weight.grd.download(ctx)
    md = _maxdiff(mc.weight.grd, mg.weight.grd, IN // BLK * (OUT // BLK) * BLK)
    assert_true(md < TOL, "dW maxdiff " + String(md))
    mg.bias.grd.download(ctx)
    md = _maxdiff(mc.bias.grd, mg.bias.grd, OUT)
    assert_true(md < TOL, "db maxdiff " + String(md))
    print("    ok")


def main() raises:
    print("BlockLinear GPU batched-GEMM parity gates")
    with DeviceContext() as ctx:
        # Dreamer-like rectangular (GEMM path), incl. grad accumulation
        _one[256, 288, 8, 12](ctx, "GEMM rect, accum x2", True)
        # GRU-ish square-per-block (GEMM path)
        _one[256, 768, 8, 5](ctx, "GEMM 1:3", False)
        # dense BLOCKS=1 (GEMM path, degenerate batch)
        _one[64, 48, 1, 7](ctx, "GEMM dense", False)
        # OPB=1 → scalar fallback (bmm N=1 miscompute guard)
        _one[32, 8, 8, 6](ctx, "scalar fallback OPB=1", False)
        # B=1 acting-path shape (GEMM, M=1)
        _one[256, 288, 8, 1](ctx, "GEMM B=1", False)
    print("BLOCK LINEAR GEMM GPU OK")
