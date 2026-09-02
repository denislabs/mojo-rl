"""Does `LinearAct`'s newly-padded GPU path still compute the same layer?

`LinearAct` used to pad its contraction dim to a multiple of 32 with no floor
and never padded N at all, so `LinearReLU[6, 256]` — the first layer of every
SAC / TD3 / DDPG / PPO / DQN trunk here — ran its forward at K=32 (fails
`multi_gemm_cond`'s `k >= 128`) and its grad_input at N=6 (fails `n % 128`).
Both went to the cuBLAS vendor path, which allocates and memsets 32 MB per
call. Both dims are now padded to 128.

Padding is supposed to be arithmetically inert: the appended rows and columns
are exactly zero, so every dot product is unchanged and only the GEMM's fp32
reduction ORDER moves. This test is that claim.

The reference is the module's own **CPU** path, which this change did not
touch. Comparing GPU-padded against CPU-unpadded checks the padding end to
end — the 2-D weight pad, the sliced bias+activation epilogue, the strided dW
accumulate, and the grad_input slice-back — rather than checking a GEMM in
isolation.

⚠ THIS RUNS ON APPLE. `PADDED` is a property of `IN_`/`OUT_`, not of the
device, so the padded branch is live on Metal too. That makes this the one
part of the split-K/padding work that does NOT need the 5090 — and the shapes
below deliberately include an unpadded control, because a test where every
shape takes the same branch would pass without exercising anything.

    pixi run mojo run -I . tests/nn/test_linear_act_pad_parity.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_linear_act_pad_parity.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.linear_act import LinearAct
from mojo_rl.nn.primitives.ops.relu_op import ReLUOp


def rel_err(
    mut got: Tensor, mut want: Tensor, n: Int, label: String
) raises -> Float64:
    """Relative error, and a VACUITY GUARD on the reference.

    ⚠ `max_abs / mag` returns a perfect 0.0 when `mag` is 0 — i.e. when the
    reference is all zeros and nothing was actually compared. The first run of
    this test printed 0.0 for all four tensors on all seven shapes, which reads
    as a flawless pass and is indistinguishable from measuring nothing. Two
    different implementations (Metal GEMM vs CPU BLAS) agreeing bit-for-bit on
    every element is not a plausible outcome; a zero reference is. So the
    reference magnitude is checked and printed, never assumed."""
    var max_abs = Float64(0)
    var mag = Float64(0)
    var nz = 0
    for i in range(n):
        var a = Float64(got.data[i])
        var b = Float64(want.data[i])
        if abs(b) > mag:
            mag = abs(b)
        if b != 0.0:
            nz += 1
        var d = abs(a - b)
        if d > max_abs:
            max_abs = d
    if mag == 0.0:
        raise Error(
            label + ": the CPU reference is ALL ZEROS over " + String(n)
            + " elements, so this comparison tested nothing"
        )
    if nz * 20 < n:
        raise Error(
            label + ": only " + String(nz) + " of " + String(n)
            + " reference elements are non-zero — too sparse to be a"
            " meaningful comparison"
        )
    return max_abs / mag


def check[
    IN: Int, OUT: Int, B: Int, EXPECT_PAD: Bool
](ctx: DeviceContext, mut n_padded: Int) raises:
    comptime L = LinearAct[IN, OUT, ReLUOp]

    comptime if L.PADDED != EXPECT_PAD:
        comptime assert False, (
            "shape's padding decision is not what the test asserts"
        )
    comptime if L.PADDED:
        n_padded += 1

    var g = L.make["gpu", INIT=Kaiming](ctx=ctx)
    var h = L.make["cpu", INIT=Kaiming]()

    # Same weights in both arms: `make` leaves the host slab populated, so copy
    # the GPU arm's initial values into the CPU arm and re-upload neither.
    g.weight.val.ensure_host(ctx, L.W_SIZE)
    g.bias.val.ensure_host(ctx, L.B_SIZE)
    h.weight.val.ensure(L.W_SIZE)
    h.bias.val.ensure(L.B_SIZE)
    for i in range(L.W_SIZE):
        h.weight.val.data[i] = g.weight.val.data[i]
    for i in range(L.B_SIZE):
        h.bias.val.data[i] = g.bias.val.data[i]

    var xg = Tensor.alloc(B * IN)
    var xh = Tensor.alloc(B * IN)
    for i in range(B * IN):
        # Sign-changing: a padding bug that folds a stray column in shows up
        # as cancellation, not as a uniform scale.
        var v = Scalar[DT](0.01) * Scalar[DT]((i % 37) - 18)
        xg.data[i] = v
        xh.data[i] = v
    xg.upload(ctx)

    var yg = Tensor.alloc_gpu(ctx, B * OUT)
    var yh = Tensor.alloc(B * OUT)
    g.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    h.forward["cpu", B](TensorRefs[1](xh), yh, None)

    var gog = Tensor.alloc(B * OUT)
    var goh = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        var v = Scalar[DT](0.003) * Scalar[DT]((i % 23) - 11)
        gog.data[i] = v
        goh.data[i] = v
    gog.upload(ctx)

    var gig = Tensor.alloc_gpu(ctx, B * IN)
    var gih = Tensor.alloc(B * IN)
    g.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gig), Optional(ctx))
    h.vjp["cpu", B](TensorRefs[1](xh), goh, TensorRefs[1](gih), None)

    yg.download(ctx)
    gig.download(ctx)
    g.weight.grd.download(ctx)
    g.bias.grd.download(ctx)
    ctx.synchronize()

    var e_out = rel_err(yg, yh, B * OUT, "out")
    var e_gi = rel_err(gig, gih, B * IN, "grad_in")
    var e_gw = rel_err(g.weight.grd, h.weight.grd, L.W_SIZE, "grad_w")
    var e_gb = rel_err(g.bias.grd, h.bias.grd, L.B_SIZE, "grad_b")

    print(
        "  IN=", IN, " OUT=", OUT, " B=", B,
        "   K_PAD=", L.K_PAD, " N_PAD=", L.N_PAD,
        "   padded=", L.PADDED,
        "\n      out ", e_out, "   grad_in ", e_gi,
        "   grad_w ", e_gw, "   grad_b ", e_gb,
        sep="",
    )
    var worst = e_out
    if e_gi > worst:
        worst = e_gi
    if e_gw > worst:
        worst = e_gw
    if e_gb > worst:
        worst = e_gb
    if worst > 1e-4:
        raise Error(
            "padded GPU path disagrees with the unpadded CPU reference beyond"
            " fp32 reduction-order noise"
        )


def self_test(ctx: DeviceContext) raises:
    """Prove `rel_err` can FAIL before believing that it passed.

    Every shape below reports exactly 0.0 — the GPU and CPU paths agree bit for
    bit. That is a legitimate outcome for small shapes (Apple's Metal GEMM and
    Accelerate can land on the same fp32 reduction order) but it is also what a
    harness comparing a buffer with ITSELF would print, and the two are
    indistinguishable from the output alone. So: run one real comparison, then
    perturb ONE element of the reference by one ulp-ish amount and require the
    comparison to notice. A test that cannot fail is not evidence.
    """
    comptime L = LinearAct[6, 256, ReLUOp]
    var g = L.make["gpu", INIT=Kaiming](ctx=ctx)
    var x = Tensor.alloc(64 * 6)
    for i in range(64 * 6):
        x.data[i] = Scalar[DT](0.01) * Scalar[DT]((i % 37) - 18)
    x.upload(ctx)
    var y = Tensor.alloc_gpu(ctx, 64 * 256)
    g.forward["gpu", 64](TensorRefs[1](x), y, Optional(ctx))
    y.download(ctx)
    ctx.synchronize()

    var copy = Tensor.alloc(64 * 256)
    for i in range(64 * 256):
        copy.data[i] = y.data[i]
    var clean = rel_err(y, copy, 64 * 256, "self-test clean")
    # Perturb one element by 1% of the tensor's own scale.
    var scale = Float64(0)
    for i in range(64 * 256):
        if abs(Float64(copy.data[i])) > scale:
            scale = abs(Float64(copy.data[i]))
    copy.data[0] = Scalar[DT](Float64(copy.data[0]) + scale * 0.01)
    var dirty = rel_err(y, copy, 64 * 256, "self-test dirty")
    print("harness self-test: clean ", clean, "   perturbed ", dirty, sep="")
    if clean != 0.0:
        raise Error("self-test: a buffer compared with its own copy is not 0")
    if dirty < 1e-3:
        raise Error(
            "SELF-TEST FAILED: a 1% perturbation of one element did not move"
            " rel_err, so this harness has no resolution and every 0.0 below"
            " means nothing"
        )
    print()


def main() raises:
    with DeviceContext() as ctx:
        print("LinearAct GPU (padded) vs its own CPU path (unpadded reference)")
        self_test(ctx)
        var n_padded = 0

        print("== both dims padded (the RL trunk's first layer) ==")
        check[6, 256, 64, True](ctx, n_padded)     # SAC/TD3 obs -> hidden
        check[17, 256, 64, True](ctx, n_padded)    # a wider obs
        check[23, 100, 64, True](ctx, n_padded)    # OUT_ padded too: 100 -> 128

        print("== N padded only (IN_ already a multiple of 128) ==")
        check[256, 100, 64, True](ctx, n_padded)

        print("== K padded only (OUT_ already a multiple of 128) ==")
        check[200, 256, 64, True](ctx, n_padded)

        print("== control: neither padded, must take the OLD branch ==")
        check[256, 256, 64, False](ctx, n_padded)
        check[128, 512, 64, False](ctx, n_padded)

        print()
        print("padded shapes exercised:", n_padded, "of 7")
        if n_padded < 5:
            raise Error(
                "fewer padded shapes than expected — if every shape took the"
                " unpadded branch this run tested none of the new code"
            )
        print("all good")
