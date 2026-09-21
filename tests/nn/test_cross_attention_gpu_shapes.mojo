# +--------------------------------------------------------------------------+ #
# | CrossAttention GPU vs CPU — at the shapes that USE it, random inputs
# +--------------------------------------------------------------------------+ #
"""The GPU forward, cached weights and backward agree with the CPU leaf at
SigLIP's shape and at every ACT shape.

    pixi run -e apple mojo run -I . tests/nn/test_cross_attention_gpu_shapes.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_cross_attention_gpu_shapes.mojo

`test_cross_attention_gpu.mojo` is the finer gate but reads its inputs from a
torch dump (`/tmp/act_ref`, from the act-ref env), at one toy shape. This one
needs nothing: random inputs, real shapes. It exists because the GPU forward
now materialises Kt contiguous (`_xa_transpose_k_kernel`) into `sk2` — a slab
the BACKWARD writes dV into — so each shape runs forward + backward TWICE, on
different inputs, and checks the second pass: a stale Kt or a stale dV from
the first pass would show there, not in a single pass.

The CPU leaf is independent of the GPU packing/transposing (it indexes the
token-major inputs directly) and is itself gated against torch
(`test_cross_attention_vs_torch.mojo`), so it is not a shared reference.

Differences are in STD UNITS — max|cpu-gpu| / rms(cpu) — so a 1024-token row
of fp32 accumulation is judged on the same scale as a 5-token one.
"""

from std.math import sqrt
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.cross_attention import (
    CrossAttention,
    xa_fused_routes_to_max,
)


comptime TOL_STD: Float64 = 1e-4
"""Fp32 CPU (BLAS, token-major loops) vs fp32 GPU (packed batched matmul):
different reduction orders. The float64-referenced bench measures each side at
~1e-6..1e-5 std units at these shapes; 1e-4 is headroom, not agreement."""
comptime TOL_TF32: Float64 = 2e-2
"""The fused forward's band where it routes to MAX's FA2 kernel
(`xa_fused_routes_to_max`: NVIDIA, unmasked, head dim 64): both matmuls run
as TF32 on the tensor cores, 10 mantissa bits per operand. The float64 bench
measured 3.7e-3 std units at SigLIP's shape on the Orin; 2e-2 is headroom.
⚠ A Metal-written 1e-4 here would be the TF32 trap on CUDA."""


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Scalar[DT]:
        """Uniform in [-1, 1)."""
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Scalar[DT](Float64(self.s >> 11) / 9007199254740992.0 * 2.0 - 1.0)


def std_err(mut cpu: Tensor, mut gpu: Tensor, n: Int) -> Float64:
    var w = Float64(0.0)
    var ss = Float64(0.0)
    for i in range(n):
        var c = Float64(cpu.data[i])
        w = max(w, abs(c - Float64(gpu.data[i])))
        ss += c * c
    var rms = sqrt(ss / Float64(n))
    if rms == 0.0:
        return w
    return w / rms


def check(mut fails: Int, name: String, err: Float64, tol: Float64 = TOL_STD):
    var ok = err < tol
    if not ok:
        fails += 1
    print(
        ("    PASS  " if ok else "    FAIL  ") + name + "  "
        + String(err) + " std u."
    )


def run_shape[
    B: Int, DIM: Int, H: Int, QL: Int, KL: Int, MASKED: Bool
](mut fails: Int, label: String, ctx: DeviceContext) raises:
    comptime XA = CrossAttention[DIM, H, QL, KL, MASKED]
    comptime QN = B * QL * DIM
    comptime KN = B * KL * DIM
    comptime MN = B * KL
    comptime AN = B * H * QL * KL
    comptime N_IN = 4 if MASKED else 3
    print("  " + label)

    var mc = XA.make["cpu", Kaiming]()
    var mg = XA.make["gpu", Kaiming](ctx)
    var rng = Lcg(UInt64(0x5EED) + UInt64(B * 131 + QL * 7 + KL))

    var sizes = List[Int]()
    sizes.append(QN)
    sizes.append(KN)
    sizes.append(KN)
    sizes.append(MN)

    var pc = TensorPack[4]()
    var pg = TensorPack[4]()
    var gc = TensorPack[4]()
    var gg = TensorPack[4]()
    for t in range(N_IN):
        pc[t].ensure(sizes[t])
        pg[t].ensure(sizes[t])
        gc[t].ensure(sizes[t])
        gg[t].ensure_gpu(ctx, sizes[t])

    for rep in range(2):
        for t in range(3):
            for i in range(sizes[t]):
                var x = rng.next()
                pc[t].data[i] = x
                pg[t].data[i] = x
        comptime if MASKED:
            # 1..KL valid keys per sample, shifting with the pass.
            for b in range(B):
                var n_valid = 1 + (b * 37 + rep * 11) % KL
                for j in range(KL):
                    var m = Scalar[DT](1.0) if j < n_valid else Scalar[DT](0.0)
                    pc[3].data[b * KL + j] = m
                    pg[3].data[b * KL + j] = m
        for t in range(N_IN):
            pg[t].upload(ctx)

        var oc = Tensor()
        var og = Tensor()
        var dc = Tensor()
        var dg = Tensor()
        dc.ensure(QN)
        dg.ensure(QN)
        for i in range(QN):
            var x = rng.next()
            dc.data[i] = x
            dg.data[i] = x
        dg.upload(ctx)

        comptime if MASKED:
            mc.forward["cpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[4, MutAnyOrigin](pc[0], pc[1], pc[2], pc[3])
                ), oc,
            )
            mg.forward["gpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[4, MutAnyOrigin](pg[0], pg[1], pg[2], pg[3])
                ), og, ctx,
            )
            mc.vjp["cpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[4, MutAnyOrigin](pc[0], pc[1], pc[2], pc[3])
                ), dc,
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[4, MutAnyOrigin](gc[0], gc[1], gc[2], gc[3])
                ),
            )
            mg.vjp["gpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[4, MutAnyOrigin](pg[0], pg[1], pg[2], pg[3])
                ), dg,
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[4, MutAnyOrigin](gg[0], gg[1], gg[2], gg[3])
                ), ctx,
            )
        else:
            mc.forward["cpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[3, MutAnyOrigin](pc[0], pc[1], pc[2])
                ), oc,
            )
            mg.forward["gpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[3, MutAnyOrigin](pg[0], pg[1], pg[2])
                ), og, ctx,
            )
            mc.vjp["cpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[3, MutAnyOrigin](pc[0], pc[1], pc[2])
                ), dc,
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[3, MutAnyOrigin](gc[0], gc[1], gc[2])
                ),
            )
            mg.vjp["gpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[3, MutAnyOrigin](pg[0], pg[1], pg[2])
                ), dg,
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[3, MutAnyOrigin](gg[0], gg[1], gg[2])
                ), ctx,
            )
        ctx.synchronize()

        if rep == 0:
            continue
        og.download(ctx)
        mg.attn.download(ctx)
        for t in range(3):
            gg[t].download(ctx)
        check(fails, "forward output", std_err(oc, og, QN))
        check(fails, "cached softmax weights", std_err(mc.attn, mg.attn, AN))
        check(fails, "dq", std_err(gc[0], gg[0], QN))
        check(fails, "dk", std_err(gc[1], gg[1], KN))
        check(fails, "dv", std_err(gc[2], gg[2], KN))

        # ── the FUSED inference forward, same inputs ─────────────────────
        # Against the CPU leaf like the rest, plus the two things that would
        # make the check vacuous: the cache must be UNTOUCHED (the fused
        # kernel writes none — a run through the two-pass path would rewrite
        # it) and the vjp must REFUSE (there are no weights to read).
        var of = Tensor()
        of.ensure_gpu(ctx, QN)
        mg.attn.dev.value().enqueue_fill(Scalar[DT](-7.0))
        mg.set_attr["fused_attention"](Scalar[DT](1.0))
        comptime if MASKED:
            mg.forward["gpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[4, MutAnyOrigin](pg[0], pg[1], pg[2], pg[3])
                ), of, ctx,
            )
        else:
            mg.forward["gpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[3, MutAnyOrigin](pg[0], pg[1], pg[2])
                ), of, ctx,
            )
        ctx.synchronize()
        of.download(ctx)
        mg.attn.download(ctx)
        comptime if xa_fused_routes_to_max[MASKED, DIM // H]():
            check(
                fails, "FUSED forward output (MAX FA2, TF32 band)",
                std_err(oc, of, QN), TOL_TF32,
            )
        else:
            check(fails, "FUSED forward output", std_err(oc, of, QN))
        var untouched = 0
        for n in range(AN):
            if mg.attn.data[n] == Scalar[DT](-7.0):
                untouched += 1
        if untouched != AN:
            fails += 1
            print(
                "    FAIL  fused forward wrote the cache ("
                + String(AN - untouched) + " of " + String(AN)
                + " changed) — the two-pass path ran, not the fused one"
            )
        else:
            print("    PASS  fused forward left the cache untouched")
        var refused = False
        try:
            comptime if MASKED:
                mg.vjp["gpu", B](
                    rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                        TensorRefs[4, MutAnyOrigin](pg[0], pg[1], pg[2], pg[3])
                    ), dg,
                    rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                        TensorRefs[4, MutAnyOrigin](gg[0], gg[1], gg[2], gg[3])
                    ), ctx,
                )
            else:
                mg.vjp["gpu", B](
                    rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                        TensorRefs[3, MutAnyOrigin](pg[0], pg[1], pg[2])
                    ), dg,
                    rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                        TensorRefs[3, MutAnyOrigin](gg[0], gg[1], gg[2])
                    ), ctx,
                )
        except:
            refused = True
        if not refused:
            fails += 1
            print("    FAIL  vjp after a fused forward did not refuse")
        else:
            print("    PASS  vjp after a fused forward refuses")
        mg.set_attr["fused_attention"](Scalar[DT](0.0))


def main() raises:
    var fails = 0
    var ctx = DeviceContext()
    print("CrossAttention GPU-vs-CPU at real shapes (second of two passes)")
    print("  device: " + String(ctx.name()))
    run_shape[3, 16, 4, 5, 7, False](fails, "toy B3 5x7", ctx)
    run_shape[3, 16, 4, 5, 7, True](fails, "toy B3 5x7 masked", ctx)
    run_shape[1, 256, 8, 60, 162, False](fails, "ACT decoder cross B1 60x162", ctx)
    run_shape[16, 256, 8, 162, 162, False](fails, "ACT encoder self B16 162", ctx)
    run_shape[16, 256, 8, 62, 62, True](fails, "ACT CVAE masked B16 62", ctx)
    run_shape[1, 768, 12, 1024, 1024, False](fails, "SigLIP B1 1024", ctx)
    print("")
    if fails > 0:
        raise Error(String(fails) + " check(s) failed")
    print("all checks passed")
