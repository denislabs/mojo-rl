"""AMP bf16-flow conv-stack smoke (Flatten / AvgPool2D / ResBlock).

Proves the ACT_DT/ADT threading through the conv stack: a bf16 instantiation of
each leaf/composite reports `ACT_DT == bfloat16` and runs forward+vjp to FINITE
outputs. Apple Metal mis-computes bf16 GEMMs, so this asserts only
compiles+finite (NOT accuracy) — real bf16 numerics are an NVIDIA job. The fp32
parity (ACT_DT == DT byte-identical) lives in test_storage_models_smoke.mojo.

Run:
  pixi run -e apple mojo run -I . tests/nn/test_amp_conv_stack_bf16_smoke.mojo
  pixi run mojo run -I . tests/nn/test_amp_conv_stack_bf16_smoke.mojo
"""

from std.testing import assert_true, assert_equal
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.call import call_forward, call_vjp
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.avg_pool_2d import AvgPool2D
from mojo_rl.nn.models.resnet import ResBlockConv2DBN
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU


comptime BF = DType.bfloat16
comptime B = 2

comptime BTensor = TensorImpl[BF]

# Flatten[8, ADT=bf16] — pure identity copy.
comptime FLAT = Flatten[8, BF]
comptime FLAT_IN = 8
comptime FLAT_OUT = 8

# AvgPool2D[C=2, K=2, S=2, P=0, H=4, W=4, ADT=bf16] → 2×2 out.
comptime POOL = AvgPool2D[2, 2, 2, 0, 4, 4, BF]
comptime POOL_IN = 2 * 4 * 4
comptime POOL_OUT = 2 * 2 * 2

# ResBlockConv2DBN[C=2, K=3, P=1, 4×4, ADT=bf16] (spatial preserved).
comptime RES = ResBlockConv2DBN[2, 3, 1, 4, 4, ADT=BF]
comptime RES_IN = 2 * 4 * 4

# Conv2DBatchNormReLU[1→2, K=3 S=1 P=1, 4×4, ADT=bf16].
comptime CBR = Conv2DBatchNormReLU[1, 2, 3, 1, 1, 4, 4, ADT=BF]
comptime CBR_IN = 1 * 4 * 4
comptime CBR_OUT = 2 * 4 * 4


def _all_finite(t: BTensor, n: Int) -> Bool:
    for i in range(n):
        var v = Float32(t.data[i])
        if v != v:  # NaN
            return False
        # +/-inf check
        if v > 3.0e38 or v < -3.0e38:
            return False
    return True


def _run_cpu[M: Module, IN: Int, OUT: Int](name: String) raises -> Bool:
    comptime assert M.ACT_DT == BF, "expected bf16 ACT_DT (CPU)"
    var m = M.make["cpu", Deterministic]()
    var x = BTensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[BF](Float32((i % 11) - 5) * 0.1)
    var out = BTensor.alloc(B * OUT)
    call_forward["cpu", B](m, TensorRefs[M.ARITY, _, BF](x), out, None)
    var go = BTensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[BF](Float32((i % 7) - 3) * 0.2)
    var gi = BTensor.alloc(B * IN)
    m.zero_grad["cpu"](None)
    call_vjp["cpu", B](
        m, TensorRefs[M.ARITY, _, BF](x), go, TensorRefs[M.ARITY, _, BF](gi),
        None,
    )
    var ok = _all_finite(out, B * OUT) and _all_finite(gi, B * IN)
    print("  ", name, "bf16 CPU fwd+vjp finite:", ok)
    return ok


def _run_gpu[M: Module, IN: Int, OUT: Int](
    name: String, c: DeviceContext
) raises -> Bool:
    comptime assert M.ACT_DT == BF, "expected bf16 ACT_DT (GPU)"
    var m = M.make["gpu", Deterministic](Optional(c))
    var x = BTensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[BF](Float32((i % 11) - 5) * 0.1)
    x.upload(c)
    var out = BTensor.alloc(B * OUT)
    call_forward["gpu", B](
        m, TensorRefs[M.ARITY, _, BF](x), out, Optional(c)
    )
    var go = BTensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[BF](Float32((i % 7) - 3) * 0.2)
    go.upload(c)
    var gi = BTensor.alloc(B * IN)
    m.zero_grad["gpu"](Optional(c))
    call_vjp["gpu", B](
        m, TensorRefs[M.ARITY, _, BF](x), go,
        TensorRefs[M.ARITY, _, BF](gi), Optional(c),
    )
    out.download(c)
    gi.download(c)
    var ok = _all_finite(out, B * OUT) and _all_finite(gi, B * IN)
    print("  ", name, "bf16 GPU fwd+vjp finite:", ok)
    return ok


def main() raises:
    print("=" * 60)
    print("AMP bf16-flow conv-stack smoke")
    print("=" * 60)
    # ACT_DT assertions are comptime (inside the runners) but echo here too.
    assert_equal(Int(FLAT.ACT_DT == BF), 1, "Flatten ACT_DT")
    assert_equal(Int(POOL.ACT_DT == BF), 1, "AvgPool2D ACT_DT")
    assert_equal(Int(RES.ACT_DT == BF), 1, "ResBlock ACT_DT")
    assert_equal(Int(CBR.ACT_DT == BF), 1, "Conv-BN-RL ACT_DT")
    print("ACT_DT == bfloat16 for all four: OK")

    # CPU bf16: only the dtype-transparent / fp32-internal leaves WITHOUT an
    # Elementwise activation (ReLU). bf16-flow Elementwise is GPU-only by an
    # existing constraint in activations.mojo, so the ReLU-bearing composites
    # (ResBlock / Conv-BN-ReLU) are exercised in bf16 on GPU only.
    print("CPU (no-ReLU leaves; bf16 Elementwise is GPU-only):")
    var c1 = _run_cpu[FLAT, FLAT_IN, FLAT_OUT]("Flatten   ")
    var c2 = _run_cpu[POOL, POOL_IN, POOL_OUT]("AvgPool2D ")
    assert_true(c1 and c2, "CPU bf16 smoke")

    print("GPU:")
    var c = DeviceContext()
    var g1 = _run_gpu[FLAT, FLAT_IN, FLAT_OUT]("Flatten   ", c)
    var g2 = _run_gpu[POOL, POOL_IN, POOL_OUT]("AvgPool2D ", c)
    var g3 = _run_gpu[RES, RES_IN, RES_IN]("ResBlock  ", c)
    var g4 = _run_gpu[CBR, CBR_IN, CBR_OUT]("Conv-BN-RL", c)
    assert_true(g1 and g2 and g3 and g4, "GPU bf16 smoke")
    print("ALL PASSED")
