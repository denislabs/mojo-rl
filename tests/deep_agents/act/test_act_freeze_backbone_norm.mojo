# +--------------------------------------------------------------------------+ #
# | Does `freeze_backbone_norm` actually REACH the backbone's BatchNorms?
# +--------------------------------------------------------------------------+ #
"""Gates the WIRING, not the mechanism.

    pixi run mojo run -I . \\
        tests/deep_agents/act/test_act_freeze_backbone_norm.mojo

`tests/nn/test_batch_norm_2d_frozen_gpu.mojo` already proves that a
`BatchNorm2D` with `frozen = True` behaves like torchvision's
`FrozenBatchNorm2d`. This file proves something different and, in this port,
more fragile: that `ACTTrainer.freeze_backbone_norm` — a `set_attr` broadcast
across a `ComputeGraph`, through `Tokenwise`, through `Sequential`, into
`Conv2DBatchNormReLU` — arrives.

⚠ THAT PROPAGATION HAS FAILED TWICE IN THIS PORT, both times silently:

  * `Dropout` never overrode `set_attr["training"]`. `Module`'s default is
    `pass`, so `train_mode(False)` returned quietly and GPT's validation loss
    and generation both ran at p=0.2 for as long as that went unnoticed.
  * `RepeatConditional` accepted `set_attr` and never FORWARDED it to its
    children. A standalone layer matched the reference at 4.8e-7; the same
    layer stacked read 0.597.

Both were found by a number being wrong, not by an error. A broadcast that
lands nowhere raises nothing, so freezing that silently does not happen would
look exactly like "freezing did not help" — which is a conclusion this port is
otherwise ready to draw.

The stub backbone is `test_act_gpu_vs_cpu.mojo`'s: two `Conv2DBatchNormReLU`,
so the BatchNorms sit behind the same wrappers the real ResNet18 does, and the
graph compiles in seconds instead of minutes. What is under test is the path
the attribute travels, not the layers it ends at.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.act.trainer import ACTTrainer

from max.gpu.host import DeviceContext


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

comptime FEAT_CH = 8
comptime STUB = Sequential[
    Conv2DBatchNormReLU[3, FEAT_CH, 3, 2, 1, IMG_H, IMG_W],
    Conv2DBatchNormReLU[FEAT_CH, FEAT_CH, 3, 2, 1, IMG_H // 2, IMG_W // 2],
]
comptime SOH = IMG_H // 4
comptime SOW = IMG_W // 4

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, P, "cpu", FEAT_CH, SOH, SOW, STUB,
]
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


struct Collect(ParamVisitor):
    """Every visited tensor's values, flattened, in walk order."""

    var vals: List[Scalar[DT]]
    var n: Int

    def __init__(out self):
        self.vals = List[Scalar[DT]]()
        self.n = 0

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.n = move.n

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
        # `for_each_state` passes the state tensor as `param`; `for_each_param`
        # passes the weight. One collector serves both walks.
        for i in range(N):
            self.vals.append(param.data[i])
        self.n += 1


struct CollectGrads(ParamVisitor):
    """The same, over gradients — only meaningful after a backward."""

    var worst: Float64
    var n: Int

    def __init__(out self):
        self.worst = 0.0
        self.n = 0

    def __init__(out self, *, deinit move: Self):
        self.worst = move.worst
        self.n = move.n

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
        # ⚠ THE BACKBONE'S BatchNorm affine, and nothing else. Two filters,
        # both needed:
        #
        #   `feat.` — the backbone's path in the graph. LAYERNORM ALSO HAS
        #   `gamma`/`beta`, and the ACT graph is full of LayerNorms (two per
        #   DETR layer, plus the head). Filtering on the suffix alone matched
        #   20 tensors where the stub backbone has 4, and this gate reported a
        #   working freeze as broken.
        #
        #   the suffix — conv weights ARE still trained in a frozen-BN
        #   backbone; the references fine-tune them. Including their gradients
        #   would fail the check for a correct model.
        if not name.startswith("feat."):
            return
        if not (name.endswith("gamma") or name.endswith("beta")):
            return
        for i in range(N):
            self.worst = max(self.worst, abs(Float64(grad.data[i])))
        self.n += 1


def batch(
    mut qpos: List[Scalar[DT]],
    mut images: List[Scalar[DT]],
    mut actions: List[Scalar[DT]],
    mut valid: List[Scalar[DT]],
):
    """A fixed, non-degenerate batch. Constant images would give BatchNorm zero
    variance and a running-statistic update of nothing, which would pass the
    'did not move' check for the wrong reason."""
    for i in range(BATCH * QPOS):
        qpos[i] = Scalar[DT](0.01 * Float64((i * 7) % 23) - 0.1)
    for i in range(BATCH * IMG_ELEMS):
        images[i] = Scalar[DT](0.02 * Float64((i * 13) % 41) - 0.4)
    for i in range(BATCH * K * ADIM):
        actions[i] = Scalar[DT](0.03 * Float64((i * 5) % 17) - 0.2)
    for i in range(BATCH * K):
        valid[i] = Scalar[DT](1.0)


def state_of(mut tr: T, mut out: List[Scalar[DT]]) raises:
    """Fills `out` rather than returning `c.vals^`: the visitor is borrowed by
    the walk, so moving a field out of it afterwards leaves the rest
    undestroyable."""
    var c = Collect()
    tr.graph.for_each_state["cpu", Collect](c, None)
    out = List[Scalar[DT]]()
    for i in range(len(c.vals)):
        out.append(c.vals[i])


def run(mut fails: Int, freeze: Bool) raises:
    var tr = T.make(lr=Scalar[DT](1e-3), kl_weight=Scalar[DT](10.0))
    tr.train_mode(True)
    if freeze:
        tr.freeze_backbone_norm(True)
        # ⚠ AFTER the freeze, deliberately. `frozen` has to override
        # `training`, or any later `train_mode(True)` — which the training loop
        # does not call again, but `eval_step` does — silently unfreezes.
        tr.train_mode(True)

    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)
    batch(qpos, images, actions, valid)

    var before = List[Scalar[DT]]()
    state_of(tr, before)
    for _ in range(3):
        _ = tr.train_step(qpos, images, actions, valid)
    var after = List[Scalar[DT]]()
    state_of(tr, after)

    if len(before) != len(after) or len(before) == 0:
        raise Error("gate: the state walk returned nothing to compare")

    var moved = Float64(0.0)
    for i in range(len(before)):
        moved = max(moved, abs(Float64(after[i]) - Float64(before[i])))

    var g = CollectGrads()
    tr.graph.for_each_param["cpu", CollectGrads](g, None)
    # The stub is two `Conv2DBatchNormReLU`, so exactly 2 gamma + 2 beta. An
    # exact count, not `> 0`: the filter above is two string predicates over a
    # naming convention, and both of its failure modes — matching nothing, and
    # matching the LayerNorms as well — are silent.
    if g.n != 4:
        raise Error(
            "gate: matched " + String(g.n) + " backbone BatchNorm affine"
            " tensors, expected 4 (2 layers x gamma+beta). The `feat.`"
            " prefix or the parameter naming changed."
        )

    var tag = "frozen" if freeze else "unfrozen"
    if freeze:
        check(
            fails,
            tag + ": BN running statistics did not move over 3 steps",
            moved == 0.0,
            "max|delta| = " + String(moved)
            + " over " + String(len(before)) + " state values",
        )
        check(
            fails,
            tag + ": BN gamma/beta gradients are zero",
            g.worst == 0.0,
            "max|grad| = " + String(g.worst)
            + " over " + String(g.n) + " tensors",
        )
    else:
        check(
            fails,
            tag + ": BN running statistics DO move",
            moved > 1e-4,
            "max|delta| = " + String(moved),
        )
        check(
            fails,
            tag + ": BN gamma/beta DO take gradient",
            g.worst > 1e-9,
            "max|grad| = " + String(g.worst),
        )


def main() raises:
    var fails = 0
    print("ACT freeze_backbone_norm wiring gate")
    print(
        "  stub backbone, " + String(BATCH) + "x" + String(N_CAM)
        + " images at " + String(IMG_H) + "x" + String(IMG_W)
    )
    print("")
    run(fails, True)
    print("")
    # The contrast is the whole gate: "the statistics did not move" is
    # trivially true of a broadcast that reached nothing AND of a model that
    # never ran. Only the unfrozen leg distinguishes those from a real freeze.
    run(fails, False)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("freeze wiring gate failed")
