# +--------------------------------------------------------------------------+ #
# | M8 gate — CrossAttention GPU vs CPU
# +--------------------------------------------------------------------------+ #
"""The GPU path must agree with the CPU path, which is itself gated against
torch (`test_cross_attention_vs_torch.mojo`).

    pixi run -e apple mojo run -I . tests/nn/test_cross_attention_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_cross_attention_gpu.mojo

Chained rather than re-gating against torch directly: the CPU leaf already
carries that comparison at 1e-7, so CPU-vs-GPU here isolates exactly what the
GPU port could have broken — the repacking index arithmetic, the scratch-slab
aliasing in the backward pass, and the masked softmax kernel.

⚠ The inputs are the SAME ones the torch gate uses, read from the reference
dump. Random inputs would exercise the kernels but would not let a failure here
be lined up against the CPU gate's numbers.

Both `Q_LEN != KV_LEN` and the per-sample mask are covered, and so is the
backward pass — the backward has eight enqueued steps with one deliberately
recycled slab (`sk0`: the packed K becomes dK once K's last read is issued),
which is the part most likely to be wrong and least likely to show up in the
forward.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.cross_attention import CrossAttention


comptime REF_DIR = "/tmp/act_ref"

comptime B = 3
comptime HEADS = 4
comptime DIM = 16
comptime QL = 5
comptime KL = 7

comptime QN = B * QL * DIM
comptime KN = B * KL * DIM
comptime MN = B * KL
comptime AN = B * HEADS * QL * KL

comptime TOL = 1e-5
"""fp32 CPU (BLAS) vs fp32 GPU (batched_matmul) — different reduction orders,
so this is an accumulation bound, not an agreement bound."""


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def load(name: String, n: Int) raises -> List[Scalar[DT]]:
    var path = String(REF_DIR) + "/" + name + ".bin"
    var out = List[Scalar[DT]](unsafe_uninit_length=n)
    with open(path, "r") as f:
        var bytes = f.read_bytes()
        if len(bytes) != n * 4:
            raise Error(
                "gate: " + name + ".bin is " + String(len(bytes))
                + " bytes, expected " + String(n * 4)
                + " — regenerate with tools/act/dump_act_reference.py"
            )
        var p = bytes.unsafe_ptr().unsafe_bitcast[Scalar[DT]]()
        for i in range(n):
            out[i] = p[unsafe_offset=i]
        _ = bytes^
    return out^


def worst(mut a: Tensor, mut b: Tensor, n: Int) -> Float64:
    var w = Float64(0.0)
    for i in range(n):
        w = max(w, abs(Float64(a.data[i]) - Float64(b.data[i])))
    return w


def main() raises:
    var fails = 0
    var ctx = DeviceContext()
    print("CrossAttention GPU-vs-CPU gate")
    print("  device: " + String(ctx.name()))
    print("")

    var q = load(String("xattn_q"), QN)
    var k = load(String("xattn_k"), KN)
    var v = load(String("xattn_v"), KN)
    var valid = load(String("xattn_valid"), MN)
    var gout = load(String("xattn_gout"), QN)

    # ══ unmasked ══════════════════════════════════════════════════════════
    var mc = CrossAttention[DIM, HEADS, QL, KL, False].make["cpu", Kaiming]()
    var mg = CrossAttention[DIM, HEADS, QL, KL, False].make["gpu", Kaiming](
        ctx
    )

    var pc = TensorPack[3]()
    pc[0].ensure(QN)
    pc[1].ensure(KN)
    pc[2].ensure(KN)
    for i in range(QN):
        pc[0].data[i] = q[i]
    for i in range(KN):
        pc[1].data[i] = k[i]
        pc[2].data[i] = v[i]

    var pg = TensorPack[3]()
    pg[0].ensure(QN)
    pg[1].ensure(KN)
    pg[2].ensure(KN)
    for i in range(QN):
        pg[0].data[i] = q[i]
    for i in range(KN):
        pg[1].data[i] = k[i]
        pg[2].data[i] = v[i]
    pg[0].upload(ctx)
    pg[1].upload(ctx)
    pg[2].upload(ctx)

    var oc = Tensor()
    var og = Tensor()
    mc.forward["cpu", B](
        TensorRefs[3, MutAnyOrigin](pc[0], pc[1], pc[2]), oc
    )
    mg.forward["gpu", B](
        TensorRefs[3, MutAnyOrigin](pg[0], pg[1], pg[2]), og, ctx
    )
    ctx.synchronize()
    og.download(ctx)
    check(
        fails,
        "forward (Q_LEN=5 over KV_LEN=7)",
        worst(oc, og, QN) < TOL,
        "max|cpu-gpu| = " + String(worst(oc, og, QN)),
    )

    mg.attn.download(ctx)
    check(
        fails,
        "cached softmax weights",
        worst(mc.attn, mg.attn, AN) < TOL,
        "max|cpu-gpu| = " + String(worst(mc.attn, mg.attn, AN)),
    )

    # backward
    var gc = Tensor()
    var gg = Tensor()
    gc.ensure(QN)
    gg.ensure(QN)
    for i in range(QN):
        gc.data[i] = gout[i]
        gg.data[i] = gout[i]
    gg.upload(ctx)

    var ggc = TensorPack[3]()
    ggc[0].ensure(QN)
    ggc[1].ensure(KN)
    ggc[2].ensure(KN)
    var ggg = TensorPack[3]()
    ggg[0].ensure_gpu(ctx, QN)
    ggg[1].ensure_gpu(ctx, KN)
    ggg[2].ensure_gpu(ctx, KN)

    mc.vjp["cpu", B](
        TensorRefs[3, MutAnyOrigin](pc[0], pc[1], pc[2]),
        gc,
        TensorRefs[3, MutAnyOrigin](ggc[0], ggc[1], ggc[2]),
    )
    mg.vjp["gpu", B](
        TensorRefs[3, MutAnyOrigin](pg[0], pg[1], pg[2]),
        gg,
        TensorRefs[3, MutAnyOrigin](ggg[0], ggg[1], ggg[2]),
        ctx,
    )
    ctx.synchronize()
    ggg[0].download(ctx)
    ggg[1].download(ctx)
    ggg[2].download(ctx)
    check(
        fails,
        "dq",
        worst(ggc[0], ggg[0], QN) < TOL,
        "max|cpu-gpu| = " + String(worst(ggc[0], ggg[0], QN)),
    )
    check(
        fails,
        "dk (the slab recycled for dK in step 8)",
        worst(ggc[1], ggg[1], KN) < TOL,
        "max|cpu-gpu| = " + String(worst(ggc[1], ggg[1], KN)),
    )
    check(
        fails,
        "dv",
        worst(ggc[2], ggg[2], KN) < TOL,
        "max|cpu-gpu| = " + String(worst(ggc[2], ggg[2], KN)),
    )

    # ══ masked ════════════════════════════════════════════════════════════
    var nc = CrossAttention[DIM, HEADS, QL, KL, True].make["cpu", Kaiming]()
    var ng = CrossAttention[DIM, HEADS, QL, KL, True].make["gpu", Kaiming](ctx)

    var qc = TensorPack[4]()
    var qg = TensorPack[4]()
    qc[0].ensure(QN)
    qc[1].ensure(KN)
    qc[2].ensure(KN)
    qc[3].ensure(MN)
    qg[0].ensure(QN)
    qg[1].ensure(KN)
    qg[2].ensure(KN)
    qg[3].ensure(MN)
    for i in range(QN):
        qc[0].data[i] = q[i]
        qg[0].data[i] = q[i]
    for i in range(KN):
        qc[1].data[i] = k[i]
        qc[2].data[i] = v[i]
        qg[1].data[i] = k[i]
        qg[2].data[i] = v[i]
    for i in range(MN):
        qc[3].data[i] = valid[i]
        qg[3].data[i] = valid[i]
    qg[0].upload(ctx)
    qg[1].upload(ctx)
    qg[2].upload(ctx)
    qg[3].upload(ctx)

    var moc = Tensor()
    var mog = Tensor()
    nc.forward["cpu", B](
        TensorRefs[4, MutAnyOrigin](qc[0], qc[1], qc[2], qc[3]), moc
    )
    ng.forward["gpu", B](
        TensorRefs[4, MutAnyOrigin](qg[0], qg[1], qg[2], qg[3]), mog, ctx
    )
    ctx.synchronize()
    mog.download(ctx)
    check(
        fails,
        "masked forward (valid 7/4/1 per sample)",
        worst(moc, mog, QN) < TOL,
        "max|cpu-gpu| = " + String(worst(moc, mog, QN)),
    )

    ng.attn.download(ctx)
    # A masked key must get EXACTLY zero on the GPU too — the additive
    # MASK_NEG plus the explicit post-softmax zeroing, not merely underflow.
    var leak = Float64(0.0)
    for b in range(B):
        for h in range(HEADS):
            for i in range(QL):
                for j in range(KL):
                    if valid[b * KL + j] < Scalar[DT](0.5):
                        leak = max(
                            leak,
                            abs(
                                Float64(
                                    ng.attn.data[
                                        b * HEADS * QL * KL
                                        + h * QL * KL
                                        + i * KL
                                        + j
                                    ]
                                )
                            ),
                        )
    check(
        fails,
        "masked keys get exactly zero weight on GPU",
        leak == 0.0,
        "max weight on a masked key = " + String(leak),
    )

    var mgc = Tensor()
    var mgg = Tensor()
    mgc.ensure(QN)
    mgg.ensure(QN)
    for i in range(QN):
        mgc.data[i] = gout[i]
        mgg.data[i] = gout[i]
    mgg.upload(ctx)

    var mggc = TensorPack[4]()
    mggc[0].ensure(QN)
    mggc[1].ensure(KN)
    mggc[2].ensure(KN)
    mggc[3].ensure(MN)
    var mggg = TensorPack[4]()
    mggg[0].ensure_gpu(ctx, QN)
    mggg[1].ensure_gpu(ctx, KN)
    mggg[2].ensure_gpu(ctx, KN)
    mggg[3].ensure_gpu(ctx, MN)

    nc.vjp["cpu", B](
        TensorRefs[4, MutAnyOrigin](qc[0], qc[1], qc[2], qc[3]),
        mgc,
        TensorRefs[4, MutAnyOrigin](mggc[0], mggc[1], mggc[2], mggc[3]),
    )
    ng.vjp["gpu", B](
        TensorRefs[4, MutAnyOrigin](qg[0], qg[1], qg[2], qg[3]),
        mgg,
        TensorRefs[4, MutAnyOrigin](mggg[0], mggg[1], mggg[2], mggg[3]),
        ctx,
    )
    ctx.synchronize()
    mggg[0].download(ctx)
    mggg[1].download(ctx)
    mggg[2].download(ctx)
    mggg[3].download(ctx)
    check(
        fails,
        "masked dq",
        worst(mggc[0], mggg[0], QN) < TOL,
        "max|cpu-gpu| = " + String(worst(mggc[0], mggg[0], QN)),
    )
    check(
        fails,
        "masked dk",
        worst(mggc[1], mggg[1], KN) < TOL,
        "max|cpu-gpu| = " + String(worst(mggc[1], mggg[1], KN)),
    )
    check(
        fails,
        "masked dv",
        worst(mggc[2], mggg[2], KN) < TOL,
        "max|cpu-gpu| = " + String(worst(mggc[2], mggg[2], KN)),
    )
    var mg_grad = Float64(0.0)
    for i in range(MN):
        mg_grad = max(mg_grad, abs(Float64(mggg[3].data[i])))
    check(
        fails,
        "the mask's gradient is zeroed on GPU",
        mg_grad == 0.0,
        "max|d/dvalid| = " + String(mg_grad),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("cross attention GPU gate failed")
