# +--------------------------------------------------------------------------+ #
# | M8 gate — the whole ACT model on GPU vs CPU
# +--------------------------------------------------------------------------+ #
"""Every piece of ACT, forward and backward, GPU against CPU.

    pixi run -e apple mojo run -I . tests/deep_agents/act/test_act_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/act/test_act_gpu_vs_cpu.mojo

The CPU model is gated against the reference `DETRVAE` end to end
(`test_act_forward_vs_reference.mojo`), so chaining GPU to CPU here covers the
whole model without re-deriving the reference comparison on device. What this
isolates is exactly what porting to GPU could break: kernel index arithmetic,
scratch-slab aliasing, and the host/device transfer points.

## Making the two comparable

Three things are stochastic and must be pinned, or the comparison measures the
RNGs rather than the kernels:

* **weights** — the two models are built independently, so the GPU model's
  parameters and BatchNorm statistics are copied from the CPU model's. Random
  init on both sides with the same `Kaiming` would NOT match: the draws are
  independent.
* **the CVAE latent** — `z.deterministic` on both. The reparameterization's
  device RNG is a Philox stream that cannot agree with the host's Box-Muller.
* **dropout** — `training = 0` on both.

Gradients are compared too, not only the forward: the backward pass has far more
kernels, and a wrong one is invisible in a forward.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.act.trainer import ACTTrainer


# ⚠ A STUB backbone, deliberately. This gate instantiates the whole ACT graph
# TWICE (once per target); with ResNet18 that is 40 conv/BN layers compiled for
# both CPU and GPU, and the build stopped being tractable — it never completed
# on CUDA. What this gate exists to check is the graph, the optimizer and the
# host/device boundary, none of which care WHICH backbone is attached. The real
# backbone's GPU path is gated on its own in `tests/nn/test_resnet18_gpu.mojo`,
# and against torchvision on CPU in `test_act_backbone_vs_reference.mojo`.
comptime FEAT_CH = 8
comptime STUB = Sequential[
    Conv2DBatchNormReLU[3, FEAT_CH, 3, 2, 1, IMG_H, IMG_W],
    Conv2DBatchNormReLU[
        FEAT_CH, FEAT_CH, 3, 2, 1, IMG_H // 2, IMG_W // 2
    ],
]
comptime SOH = IMG_H // 4
comptime SOW = IMG_W // 4


comptime QPOS = 6
comptime ADIM = 6
comptime N_CAM = 2
comptime IMG_H = 64
comptime IMG_W = 64
comptime K = 5
comptime DIM = 16
comptime HEADS = 2
comptime FF = 32
comptime LATENT = 8
comptime N_ENC = 1
comptime N_DEC = 1
comptime BATCH = 2
comptime P = 0.0

comptime TC = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, P, "cpu", FEAT_CH, SOH, SOW, STUB,
]
comptime TG = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, P, "gpu", FEAT_CH, SOH, SOW, STUB,
]
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W

comptime PARITY_ATOL: Float64 = 1e-5
comptime PARITY_RTOL: Float64 = 2e-2
comptime PARITY_ELEM_RTOL: Float64 = 0.1
comptime PARITY_NORM_RTOL: Float64 = 1e-2
"""Elementwise comparisons scale their tolerance by the TENSOR's magnitude, and
are paired with an L2-norm check.

⚠⚠ PER-ELEMENT relative error is the WRONG statistic across precisions. A
gradient is a sum of large cancelling terms, and TF32's error scales with the
TERMS, not the sum — so an element whose true value is ~0 has unbounded relative
error while being perfectly correct. On a 5090 this reported a ratio of 78 for a
discrepancy of ~2e-4 relative to the gradient's own scale, on the same run where
the gradient NORM agreed to 0.13% across every value. The norm is the check with
teeth; the elementwise one only has to catch gross outliers."""
# ⚠ The remaining absolute-tolerance constants above are for SCALAR comparisons
# (a loss, a norm), where per-element relative error is well defined because
# there is only one element. NVIDIA runs fp32 matmuls on TF32 tensor cores — a
# 10-bit mantissa, ~1e-3 relative per matmul, compounding with depth — while
# Apple has no TF32 and sits at ~1e-7. Measured on a 5090, all correct kernels:
#
#     BatchNorm2D alone (no matmul)   6.0e-8
#     ACT eval L1  (2 convs)          rel 3.6e-4
#     ACT eval KL                     rel 1.4e-3
#     ACT gradient norm               rel 1.3e-3
#
# "contains a matmul" vs "does not" is the discriminator, and it is why a
# tolerance calibrated on Metal cannot serve CUDA (`feedback_fd_gradcheck_tf32`,
# which cost three false bug reports, then a fourth here).


def within(a: Float64, b: Float64) -> Bool:
    """numpy-`allclose` semantics: `|a-b| <= atol + rtol*|a|`."""
    return abs(a - b) <= PARITY_ATOL + PARITY_RTOL * abs(a)


def parity_ratio(a: Float64, b: Float64) -> Float64:
    return abs(a - b) / (PARITY_ATOL + PARITY_RTOL * abs(a))


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


# ── weight transfer + gradient comparison visitors ───────────────────────


struct _Collect(ParamVisitor):
    """Snapshot every param (or state) value into a flat host list, in walk
    order."""

    var vals: List[Scalar[DT]]
    var grads: Bool

    def __init__(out self, grads: Bool = False):
        self.vals = List[Scalar[DT]]()
        self.grads = grads

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.grads = move.grads

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target != "cpu":
            if self.grads:
                grad.download(ctx.value())
            else:
                param.download(ctx.value())
        for i in range(N):
            self.vals.append(grad.data[i] if self.grads else param.data[i])


struct _Inject(ParamVisitor):
    """Write a flat host list back over every param (or state), in walk order."""

    var vals: List[Scalar[DT]]
    var pos: Int

    def __init__(out self, var vals: List[Scalar[DT]]):
        self.vals = vals^
        self.pos = 0

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.pos = move.pos

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        param.ensure(N)
        for i in range(N):
            param.data[i] = self.vals[self.pos + i]
        self.pos += N
        comptime if target != "cpu":
            param.upload(ctx.value())


@fieldwise_init
struct Parity(ImplicitlyCopyable):
    """`worst` scale-relative ratio, `nrel` L2-norm relative error, `n_over`
    count past tolerance — the count is what separates a few cancelling
    elements from a systematic disagreement."""

    var worst: Float64
    var nrel: Float64
    var n_over: Int


def worst(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) raises -> Parity:
    """Scale-relative elementwise comparison + an L2-norm aggregate.

    ⚠ NOT `max|a-b|` against `max|a|` (two different elements), and NOT
    per-element relative error (unbounded at a true zero — see the header)."""
    if len(a) != len(b):
        raise Error(
            "gate: walk lengths differ — " + String(len(a)) + " vs "
            + String(len(b))
        )
    var scale = Float64(0.0)
    for i in range(len(a)):
        scale = max(scale, abs(Float64(a[i])))
    var tol = PARITY_ATOL + PARITY_ELEM_RTOL * scale
    var w = Float64(0.0)
    var n_over = 0
    var sa = Float64(0.0)
    var sb = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        var r = abs(x - y) / tol
        w = max(w, r)
        if r > 1.0:
            n_over += 1
        sa += x * x
        sb += y * y
    var na = sa ** 0.5
    return Parity(w, abs(na - (sb ** 0.5)) / (na + 1e-30), n_over)


def main() raises:
    var fails = 0
    var ctx = DeviceContext()
    print("ACT GPU-vs-CPU gate")
    print("  device: " + String(ctx.name()))
    print("")

    var tc = TC.make(kl_weight=Scalar[DT](10.0), max_grad_norm=Scalar[DT](0.0))
    var tg = TG.make(
        kl_weight=Scalar[DT](10.0), max_grad_norm=Scalar[DT](0.0), ctx=ctx
    )

    # Pin the latent on both sides (see the header). ⚠ Via the trainer's sticky
    # setter, NOT `set_node_attr` — `train_mode` rewrites that node attribute on
    # every switch, so a direct write is undone by the first `eval_step` and the
    # train-mode comparison then measures two independent RNG streams.
    tc.set_deterministic_latent(True)
    tg.set_deterministic_latent(True)

    # ── copy CPU weights + BN statistics onto the GPU model ──────────────
    var wp = _Collect()
    tc.graph.for_each_param["cpu"](wp, None, String(""))
    var wi = _Inject(wp.vals.copy())
    tg.graph.for_each_param["gpu"](wi, ctx, String(""))

    var sp = _Collect()
    tc.graph.for_each_state["cpu"](sp, None, String(""))
    var si = _Inject(sp.vals.copy())
    tg.graph.for_each_state["gpu"](si, ctx, String(""))
    check(
        fails,
        "weights + state transferred (walks agree in length)",
        wi.pos > 0 and si.pos > 0,
        String(wi.pos) + " param values, " + String(si.pos) + " state values",
    )

    # ── one fixed batch ──────────────────────────────────────────────────
    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)
    for b in range(BATCH):
        for j in range(QPOS):
            qpos[b * QPOS + j] = Scalar[DT](0.3 * Float64(j) - 0.4 * Float64(b))
        for i in range(IMG_ELEMS):
            images[b * IMG_ELEMS + i] = Scalar[DT](
                0.05 * Float64((i * 11 + b * 5) % 19) - 0.45
            )
        for t in range(K):
            for j in range(ADIM):
                actions[b * K * ADIM + t * ADIM + j] = Scalar[DT](
                    0.3 * Float64(t) - 0.2 * Float64(j) + 0.5 * Float64(b)
                )
            valid[b * K + t] = Scalar[DT](1.0)
    valid[BATCH * K - 1] = Scalar[DT](0.0)  # exercise the mask on both paths

    # ── forward: eval mode (z = 0, BN running stats) ─────────────────────
    var ec = tc.eval_step(qpos, images, actions, valid)
    var eg = tg.eval_step(qpos, images, actions, valid)
    check(
        fails,
        "eval L1",
        within(ec.l1, eg.l1),
        "cpu " + String(ec.l1) + "  gpu " + String(eg.l1) + "  ratio "
        + String(parity_ratio(ec.l1, eg.l1)),
    )
    check(
        fails,
        "eval KL",
        within(ec.kl, eg.kl),
        "cpu " + String(ec.kl) + "  gpu " + String(eg.kl) + "  ratio "
        + String(parity_ratio(ec.kl, eg.kl)),
    )
    check(
        fails,
        "eval total loss",
        within(ec.loss, eg.loss),
        "cpu " + String(ec.loss) + "  gpu " + String(eg.loss) + "  ratio "
        + String(parity_ratio(ec.loss, eg.loss)),
    )
    # A zero loss on both sides would satisfy the checks above and mean nothing.
    check(
        fails,
        "the loss is non-trivial",
        ec.l1 > 0.05,
        "cpu L1 = " + String(ec.l1),
    )

    # ── the inference path, on IDENTICAL weights ─────────────────────────
    # ⚠ BEFORE the training step, deliberately. Run after it, this comparison
    # conflates "is the GPU inference path correct" with "do two fp32 Adam
    # steps land on the same weights" — they do not, quite: one step leaves a
    # ~2e-5 spread over 11.2M parameters, and a 20-layer ResNet plus two
    # transformer stacks amplifies that by ~1e3 into the output. Measured at
    # 0.025 that way, versus round-off here. The optimizer's agreement is
    # already checked directly, on the parameters themselves.
    var ac = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var ag = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    tc.predict(qpos, images, actions, valid, ac)
    tg.predict(qpos, images, actions, valid, ag)
    var aw = worst(ac, ag)
    check(
        fails,
        "predict() action chunk (identical weights)",
        aw.worst < 1.0 and aw.nrel < PARITY_NORM_RTOL,
        "worst " + String(aw.worst) + "  norm-rel " + String(aw.nrel)
        + "  over-tol " + String(aw.n_over),
    )

    # ── one training step: forward, backward, and the resulting weights ──
    # `train_step` runs the optimizer, so comparing the PARAMETERS after it
    # covers the backward AND the Adam walk on device in one comparison.
    var rc = tc.train_step(qpos, images, actions, valid)
    var rg = tg.train_step(qpos, images, actions, valid)
    check(
        fails,
        "train-step L1 (train mode: BN batch stats, latent pinned)",
        within(rc.l1, rg.l1),
        "cpu " + String(rc.l1) + "  gpu " + String(rg.l1) + "  ratio "
        + String(parity_ratio(rc.l1, rg.l1)),
    )
    check(
        fails,
        "gradient norm",
        within(rc.grad_norm, rg.grad_norm),
        "cpu " + String(rc.grad_norm) + "  gpu " + String(rg.grad_norm)
        + "  ratio " + String(parity_ratio(rc.grad_norm, rg.grad_norm)),
    )

    # ⚠ Compare the GRADIENTS, not the post-Adam parameters. `train_step` leaves
    # the gradients populated, and they are what the GPU backward actually
    # produced. Parameters after one Adam step are a BAD parity target: at t=1
    # the update is `lr * m_hat/sqrt(v_hat)`, which is ~`±lr` regardless of
    # gradient MAGNITUDE — so a near-zero gradient that lands on opposite signs
    # between two correct backends yields a full `2*lr` parameter difference.
    # Measured at 2e-5 with `lr = 1e-5`: exactly `2*lr`, and it reads as a 2x
    # tolerance breach while nothing is wrong.
    var pc = _Collect(grads=True)
    tc.graph.for_each_param["cpu"](pc, None, String(""))
    var pg = _Collect(grads=True)
    tg.graph.for_each_param["gpu"](pg, ctx, String(""))
    var pw = worst(pc.vals, pg.vals)
    check(
        fails,
        "gradients agree over every parameter",
        pw.worst < 1.0 and pw.nrel < PARITY_NORM_RTOL,
        "worst " + String(pw.worst) + "  norm-rel " + String(pw.nrel)
        + "  over-tol " + String(pw.n_over) + "/" + String(len(pc.vals)),
    )

    # The parameters too, but with an absolute floor sized to Adam's step —
    # see above. This checks the optimizer walked every parameter on device,
    # which the gradient comparison alone does not.
    var qc = _Collect()
    tc.graph.for_each_param["cpu"](qc, None, String(""))
    var qg = _Collect()
    tg.graph.for_each_param["gpu"](qg, ctx, String(""))
    var qw = Float64(0.0)
    comptime ADAM_STEP_ATOL = 4.0 * 1e-5  # 4 * the trainer's default lr
    for i in range(len(qc.vals)):
        var x = Float64(qc.vals[i])
        qw = max(
            qw,
            abs(x - Float64(qg.vals[i]))
            / (ADAM_STEP_ATOL + PARITY_ELEM_RTOL * abs(x)),
        )
    check(
        fails,
        "parameters after one Adam step (atol = 4*lr)",
        qw < 1.0,
        "worst ratio over " + String(len(qc.vals)) + " values = " + String(qw),
    )

    var sc2 = _Collect()
    tc.graph.for_each_state["cpu"](sc2, None, String(""))
    var sg2 = _Collect()
    tg.graph.for_each_state["gpu"](sg2, ctx, String(""))
    var sw = worst(sc2.vals, sg2.vals)
    check(
        fails,
        "BatchNorm running statistics agree after one step",
        sw.worst < 1.0 and sw.nrel < PARITY_NORM_RTOL,
        "worst " + String(sw.worst) + "  norm-rel " + String(sw.nrel)
        + "  over-tol " + String(sw.n_over) + "/" + String(len(sc2.vals)),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("act GPU-vs-CPU gate failed")
