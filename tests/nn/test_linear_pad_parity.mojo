"""Does the K/N alignment padding change what `Linear` computes?

The padded columns are exactly 0, so the dot products must be unchanged up to
fp32 reduction-order noise. Checks an UNALIGNED width (518, which now pads to
544) and an ALIGNED one (512, which takes the untouched path) against the CPU
forward, and confirms the aligned case is bit-identical to before.
"""

from std.math import abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.linear import Linear


def check[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises:
    comptime L = Linear[IN, OUT]
    print(
        "  IN=", IN, " OUT=", OUT, " B=", B, "   K_PAD=", L.K_PAD,
        " (", L.NEEDS_PAD, ")  N_PAD=", L.N_PAD, " (", L.NEEDS_N_PAD, ")",
        sep="",
    )

    # Same weights on both devices: build on CPU, copy the slab, upload.
    var lc = L.make["cpu", INIT=Kaiming]()
    var lg = L.make["gpu", INIT=Kaiming](ctx=ctx)
    # Overwrite the GPU module's weights with the CPU module's, host-side, then
    # push them down — the two INITs draw different RNG otherwise.
    lg.weight.val.ensure_host(ctx, L.W_SIZE)
    lg.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        lg.weight.val.data[i] = lc.weight.val.data[i]
    for i in range(L.B_SIZE):
        lg.bias.val.data[i] = lc.bias.val.data[i]
    lg.weight.val.upload_resident(ctx)
    lg.bias.val.upload_resident(ctx)

    var xc = Tensor.alloc(B * IN)
    for i in range(B * IN):
        xc.data[i] = Scalar[DT](0.01) * Scalar[DT]((i % 37) - 18)
    var xg = Tensor.alloc(B * IN)
    for i in range(B * IN):
        xg.data[i] = xc.data[i]
    xg.upload(ctx)

    var yc = Tensor.alloc(B * OUT)
    var yg = Tensor.alloc_gpu(ctx, B * OUT)
    lc.forward["cpu", B](TensorRefs[1](xc), yc, None)
    lg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    yg.download(ctx)
    ctx.synchronize()

    var max_abs = Float64(0)
    var max_rel = Float64(0)
    for i in range(B * OUT):
        var a = Float64(yc.data[i])
        var b = Float64(yg.data[i])
        var d = abs(a - b)
        if d > max_abs:
            max_abs = d
        var denom = abs(a) if abs(a) > 1e-6 else 1e-6
        if d / denom > max_rel:
            max_rel = d / denom
    # ⚠ NON-VACUITY: a comparison of two all-zero buffers also reports 0.0.
    var mag = Float64(0)
    for i in range(B * OUT):
        if abs(Float64(yg.data[i])) > mag:
            mag = abs(Float64(yg.data[i]))
    print(
        "     max_abs=", max_abs, "  max_rel=", max_rel,
        "   |gpu|max=", mag, "  cpu[0]=", yc.data[0], " gpu[0]=", yg.data[0],
        sep="",
    )
    if mag == 0.0:
        raise Error("VACUOUS: the GPU output is all zeros")
    if max_rel > 1e-4:
        raise Error("PADDING CHANGED THE RESULT — max_rel " + String(max_rel))


def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name())
    print()
    print("== unaligned widths (padding ACTIVE) ==")
    check[518, 512, 268](ctx)     # TD-MPC2 za = latent|act
    check[101, 512, 64](ctx)      # BINS as an input width
    check[30, 256, 64](ctx)       # SAC critic obs|act
    check[24, 256, 7](ctx)        # odd batch too
    print()
    print("== narrow/unaligned OUTPUT widths (N padding ACTIVE) ==")
    check[512, 101, 268](ctx)     # TD-MPC2 two-hot head: BINS=101
    check[512, 12, 268](ctx)      # policy head: 2*ACT
    check[512, 1, 268](ctx)       # termination head
    check[518, 101, 64](ctx)      # BOTH dims padded at once
    print()
    print("== aligned widths (padding INACTIVE — must be untouched) ==")
    check[512, 512, 268](ctx)
    check[256, 128, 64](ctx)
    print()
    print("all good")
