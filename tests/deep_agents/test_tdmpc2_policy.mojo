"""TD-MPC2 policy (actor) update validation (CPU).

P2 gate. Two checks:

1. `test_policy_ascends_q`: with two FIXED (untrained) Q heads and the
   entropy term + RunningScale disabled, the policy loss = −0.5·avg_Q(z, π(z))
   must DECREASE — i.e. the policy moves to actions the (decoded two-hot) Q
   prefers. Validates the reparam-sample → Concat → Q(input_only) →
   TwoHotDecode → avg → backprop-into-policy path (the P2 risk).

2. `test_running_scale`: the percentile-range EMA tracks Q magnitude.

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_policy.mojo`
"""

from std.memory import alloc
from std.math import isfinite
from std.testing import assert_true, TestSuite

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.tdmpc2.nets import TDMPC2Policy, TDMPC2QNet
from mojo_rl.deep_agents2.tdmpc2.policy_step import PolicyStep
from mojo_rl.deep_agents2.tdmpc2.running_scale import RunningScale


comptime LATENT = 16
comptime ACT = 2
comptime MLP = 16
comptime BINS = 11
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8


def _fill_pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, sd: Int):
    var s = UInt64(sd * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64((s >> 33)) / Float64(UInt64(1) << 31)
        p[i] = Scalar[DT]((u - 1.0))


def test_policy_ascends_q() raises:
    comptime PolicyT = TDMPC2Policy[LATENT, ACT, MLP]
    comptime QNetT = TDMPC2QNet[LATENT, ACT, MLP, BINS]
    comptime StepT = PolicyStep[LATENT, ACT, MLP, BINS, VMIN, VMAX, B]

    var policy = PolicyT.make["cpu", INIT=Kaiming]()
    var q = List[QNetT]()
    q.append(QNetT.make["cpu", INIT=Kaiming]())
    q.append(QNetT.make["cpu", INIT=Kaiming]())
    var pi_opt = Adam.make["cpu", PolicyT](policy)
    pi_opt.lr = Scalar[DT](1e-3)

    var step = StepT.make["cpu"]()
    # Isolate the Q-ascent path: no entropy term, frozen unit scale.
    step.entropy_coef = Scalar[DT](0.0)
    step.scale.tau = Scalar[DT](0.0)

    var z = alloc[Scalar[DT]](B * LATENT)
    _fill_pseudo(z, B * LATENT, 7)

    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    comptime ITERS = 80
    for it in range(ITERS):
        var l = step.step["cpu"](policy, q, 0, 1, pi_opt, z)
        assert_true(isfinite(l), "policy loss must be finite")
        if it == 0:
            first = l
        if it == ITERS - 1:
            last = l
    print("  policy loss:", first, "->", last)
    assert_true(last < first, "policy loss must decrease (ascends fixed Q)")

    z.free()


def test_running_scale() raises:
    var rs = RunningScale(tau=Scalar[DT](0.01))
    var n = 100
    var x = alloc[Scalar[DT]](n)
    for i in range(n):
        x[i] = Scalar[DT](i)   # 0..99 → (p95−p5) ≈ 89
    rs.update_from(x, n)
    print("  running scale after 1 update:", rs.value)
    # one EMA step from 1.0 toward ~89 with tau=0.01 → ~1.88.
    assert_true(rs.value > Scalar[DT](1.0), "scale must rise toward Q range")
    assert_true(isfinite(rs.value), "scale finite")
    assert_true(rs.inv() > Scalar[DT](0.0), "inv positive")
    x.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
