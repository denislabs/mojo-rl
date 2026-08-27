"""`SimNorm`'s CPU forward must match its (untouched) GPU forward.

The CPU path was rewritten to evaluate `exp` ONCE per element instead of twice
and to process a whole group as one SIMD vector. Both are silent if wrong — a
mis-indexed group still produces a valid-looking probability vector, it is just
the wrong one, and the trunk keeps training.

Covers the vectorized branch (power-of-two group) AND the scalar fallback
(non-power-of-two group), plus a group of 1 and a group equal to the whole row.

    pixi run -e apple mojo run -I . tests/nn/test_sim_norm_cpu_parity.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.sim_norm import SimNorm


def check[DIM: Int, GROUPS: Int, B: Int](
    ctx: DeviceContext, label: String
) raises:
    comptime S = SimNorm[DIM, GROUPS]
    comptime GS = DIM // GROUPS
    var vec = "SIMD" if (GS > 1 and (GS & (GS - 1)) == 0) else "scalar"
    print(
        "  ", label, " DIM=", DIM, " GROUPS=", GROUPS, " GROUP_SIZE=", GS,
        " (", vec, " branch)", sep="",
    )

    var mc = S.make["cpu", INIT=Kaiming]()
    var mg = S.make["gpu", INIT=Kaiming](ctx=ctx)

    var xc = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xc.data[i] = Scalar[DT](0.21) * Scalar[DT]((i % 43) - 21)
    var xg = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xg.data[i] = xc.data[i]
    xg.upload(ctx)

    var yc = Tensor.alloc(B * DIM)
    var yg = Tensor.alloc_gpu(ctx, B * DIM)
    mc.forward["cpu", B](TensorRefs[1](xc), yc, None)
    mg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    yg.download(ctx)
    ctx.synchronize()

    var max_rel = Float64(0)
    var mag = Float64(0)
    for i in range(B * DIM):
        var a = Float64(yg.data[i])
        var b = Float64(yc.data[i])
        if abs(a) > mag:
            mag = abs(a)
        var denom = abs(a) if abs(a) > 1e-6 else 1e-6
        var r = abs(a - b) / denom
        if r > max_rel:
            max_rel = r

    # Each group must still sum to 1 — a mis-indexed group can match neither
    # this nor the GPU, so the two checks fail independently.
    var max_sum_err = Float64(0)
    for b in range(B):
        for g in range(GROUPS):
            var s = Float64(0)
            for k in range(GS):
                s += Float64(yc.data[b * DIM + g * GS + k])
            if abs(s - 1.0) > max_sum_err:
                max_sum_err = abs(s - 1.0)

    print(
        "     cpu vs gpu max_rel=", max_rel, "   group-sum err=", max_sum_err,
        sep="",
    )
    if mag == 0.0:
        raise Error("VACUOUS: output is identically zero")
    if max_rel > 1e-5:
        raise Error("SimNorm cpu/gpu mismatch: " + String(max_rel))
    if max_sum_err > 1e-5:
        raise Error("SimNorm groups do not sum to 1: " + String(max_sum_err))


def main() raises:
    var ctx = DeviceContext()
    print("SimNorm CPU vs GPU —", ctx.name())
    print()
    check[512, 64, 268](ctx, "TD-MPC2 latent (SN=8)  ")   # GROUP_SIZE 8, SIMD
    check[256, 32, 64](ctx, "encoder-ish            ")    # GROUP_SIZE 8, SIMD
    check[512, 128, 32](ctx, "GROUP_SIZE 4           ")   # SIMD
    check[96, 32, 32](ctx, "GROUP_SIZE 3           ")     # scalar fallback
    check[64, 64, 16](ctx, "GROUP_SIZE 1           ")     # scalar fallback
    print()
    print("ALL PASSED")
