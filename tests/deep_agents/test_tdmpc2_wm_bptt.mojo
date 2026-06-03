"""TD-MPC2 world-model BPTT validation (CPU).

The decisive P1 gate: the total weighted WM loss must DECREASE over N steps
on a fixed synthetic batch. This can only happen if gradients flow correctly
through the multi-step latent rollout — i.e. the dynamics net learns to
produce latents that stay predictive several steps ahead. The legacy CPU
path (`deep_agents/tdmpc2/tdmpc2.mojo:861-867`) skipped this BPTT and could
not train the dynamics from downstream reward/value losses.

Mirrors the validated `tests/nn2/spike_wm_bptt.mojo` methodology
(loss-decrease smoke, no fixture — this is orchestration + grad-flow, not
new pointwise math).

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_wm_bptt.mojo`
"""

from std.memory import alloc
from std.math import isfinite
from std.testing import assert_true, TestSuite

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.tdmpc2.nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet,
)
from mojo_rl.deep_agents2.tdmpc2.wm_graph import TDMPC2WMGraph
from mojo_rl.deep_agents2.tdmpc2.wm_step import WMStep


comptime OBS = 4
comptime ENC = 16
comptime ACT = 2
comptime LATENT = 16
comptime MLP = 16
comptime BINS = 11
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 4
comptime H = 3


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _fill_pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, sd: Int):
    var s = UInt64(sd * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64((s >> 33)) / Float64(UInt64(1) << 31)
        p[i] = Scalar[DT]((u - 1.0))


def test_wm_bptt_loss_decreases() raises:
    comptime EncT = TDMPC2Encoder[OBS, ENC, LATENT, SN]
    comptime DynT = TDMPC2Dynamics[LATENT, ACT, MLP, SN]
    comptime RewT = TDMPC2Reward[LATENT, ACT, MLP, BINS]
    comptime QNetT = TDMPC2QNet[LATENT, ACT, MLP, BINS]
    comptime GraphT = TDMPC2WMGraph[LATENT, ACT, MLP, BINS, SN, VMIN, VMAX]
    comptime StepT = WMStep[OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H]

    var enc = EncT.make["cpu", INIT=Kaiming]()
    var dyn = DynT.make["cpu", INIT=Kaiming]()
    var rew_net = RewT.make["cpu", INIT=Kaiming]()
    var graph = GraphT.make["cpu", INIT=Kaiming]()

    var lr = Scalar[DT](3e-3)
    var enc_opt = Adam.make["cpu", EncT](enc)
    enc_opt.lr = lr * Scalar[DT](0.3)   # reference enc_lr_scale
    var dyn_opt = Adam.make["cpu", DynT](dyn)
    dyn_opt.lr = lr
    var rew_opt = Adam.make["cpu", RewT](rew_net)
    rew_opt.lr = lr

    var q = List[QNetT]()
    var q_opt = List[Adam]()
    for _ in range(5):
        var qn = QNetT.make["cpu", INIT=Kaiming]()
        var qo = Adam.make["cpu", QNetT](qn)
        qo.lr = lr
        q.append(qn^)
        q_opt.append(qo^)
    var step = StepT.make["cpu"]()

    # Fixed synthetic batch (t-major). td targets are arbitrary stop-grad
    # scalars here — P1 validates grad flow, not TD-target correctness.
    var obs = _alloc((H + 1) * B * OBS)
    var act = _alloc(H * B * ACT)
    var rew = _alloc(H * B)
    var td = _alloc(H * B)
    _fill_pseudo(obs, (H + 1) * B * OBS, 1)
    _fill_pseudo(act, H * B * ACT, 2)
    _fill_pseudo(rew, H * B, 3)
    _fill_pseudo(td, H * B, 4)

    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    comptime ITERS = 40
    for it in range(ITERS):
        var l = step.step["cpu"](
            graph, enc, dyn, rew_net, q,
            enc_opt, dyn_opt, rew_opt, q_opt,
            obs, act, rew, td,
        ).total()
        assert_true(isfinite(l), "WM loss must be finite")
        if it == 0:
            first = l
            print("  iter 0  total WM loss =", l)
        if it == ITERS - 1:
            last = l
            print("  iter", ITERS - 1, " total WM loss =", l)

    print("  decrease:", first, "->", last)
    assert_true(last < first, "WM BPTT loss must decrease over training")
    assert_true(
        last < first * Scalar[DT](0.9),
        "WM BPTT loss must decrease substantially (>10%) — confirms multi-step"
        " gradient flow to dynamics",
    )

    obs.free(); act.free(); rew.free(); td.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
