"""TD-MPC2 TD-target step smoke (CPU).

Forward-only arithmetic: td[t] = reward[t] + γ·(1−done[t])·min(Q_a, Q_b).
Checks: output finite; bootstrap dropped when done=1 (td == reward);
td responds to reward (monotone shift).

Run: `pixi run mojo run -I . tests/deep_agents2/test_tdmpc2_td_target.mojo`
"""

from std.memory import alloc
from std.math import isfinite, abs
from std.testing import assert_true, assert_almost_equal, TestSuite

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.deep_agents2.tdmpc2.nets import (
    TDMPC2Encoder, TDMPC2Policy, TDMPC2QNet,
)
from mojo_rl.deep_agents2.tdmpc2.td_target_step import TDTargetStep


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


def _fill_pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, sd: Int):
    var s = UInt64(sd * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64((s >> 33)) / Float64(UInt64(1) << 31)
        p[i] = Scalar[DT]((u - 1.0))


def test_td_target_smoke() raises:
    comptime EncT = TDMPC2Encoder[OBS, ENC, LATENT, SN]
    comptime PolicyT = TDMPC2Policy[LATENT, ACT, MLP]
    comptime QNetT = TDMPC2QNet[LATENT, ACT, MLP, BINS]
    comptime StepT = TDTargetStep[
        OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H
    ]

    var enc = EncT.make["cpu", INIT=Kaiming]()
    var policy = PolicyT.make["cpu", INIT=Kaiming]()
    var qt = List[QNetT]()
    qt.append(QNetT.make["cpu", INIT=Kaiming]())
    qt.append(QNetT.make["cpu", INIT=Kaiming]())
    var step = StepT.make["cpu"]()

    var obs = alloc[Scalar[DT]]((H + 1) * B * OBS)
    var rew = alloc[Scalar[DT]](H * B)
    var done = alloc[Scalar[DT]](H * B)
    var td = alloc[Scalar[DT]](H * B)
    _fill_pseudo(obs, (H + 1) * B * OBS, 1)
    for i in range(H * B):
        rew[i] = Scalar[DT](0.5)
        done[i] = Scalar[DT](0.0)

    var gamma = Scalar[DT](0.99)
    step.step["cpu"](enc, policy, qt, 0, 1, obs, rew, done, td, gamma)

    var finite = True
    for i in range(H * B):
        if not isfinite(td[i]):
            finite = False
    assert_true(finite, "td finite")

    # done=1 → bootstrap dropped → td == reward.
    for i in range(H * B):
        done[i] = Scalar[DT](1.0)
    var td2 = alloc[Scalar[DT]](H * B)
    step.step["cpu"](enc, policy, qt, 0, 1, obs, rew, done, td2, gamma)
    for i in range(H * B):
        assert_almost_equal(
            td2[i], rew[i], atol=1e-5,
            msg="done=1 must drop the bootstrap (td == reward)",
        )

    # reward shift +1 → td shifts +1 (done=0 again, same RNG re-seeded? no —
    # RSample advances RNG, so compare bootstrap-free with done=1).
    for i in range(H * B):
        rew[i] = Scalar[DT](1.5)
    var td3 = alloc[Scalar[DT]](H * B)
    step.step["cpu"](enc, policy, qt, 0, 1, obs, rew, done, td3, gamma)
    for i in range(H * B):
        assert_almost_equal(
            td3[i], Scalar[DT](1.5), atol=1e-5,
            msg="td tracks reward when bootstrap dropped",
        )

    obs.free(); rew.free(); done.free(); td.free(); td2.free(); td3.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
