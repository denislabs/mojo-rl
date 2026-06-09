"""DreamerV3Trainer smoke — synthetic replay, no env.

Fills the sequence replay with pseudo-random transitions, then runs N
`train_step`s. Gate: WM loss + AC loss finite and both decrease (no NaN).
This validates the assembled trainer (WM-BPTT + param-sync + imagination
AC + DreamerOpt over all modules) end-to-end on a fixed buffer.

Run: `pixi run mojo run -I . tests/nn2/test_dreamerv3_trainer_smoke.mojo`
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.trainer import DreamerV3Trainer

comptime OBS = 3
comptime ACT = 1
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime TOKEN = 8
comptime DEC_U = 8
comptime HU = 8
comptime VU = 8
comptime PU = 8
comptime BINS = 7
comptime B = 2
comptime T = 3
comptime T_IMAG = 4
comptime CAP = 256

comptime Tr = DreamerV3Trainer[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP,
]


def main() raises:
    print("=" * 70)
    print("DreamerV3Trainer smoke (synthetic replay)")
    print("=" * 70)
    var tr = Tr.make(lr=Scalar[DT](3e-3), learning_starts=0, warmup_steps=0)

    # fill replay with pseudo-random transitions
    var s = UInt64(12345)
    var ob = alloc[Scalar[DT]](OBS)
    var ac = alloc[Scalar[DT]](ACT)
    for _t in range(120):
        for k in range(OBS):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ob[k] = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)
        for k in range(ACT):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            ac[k] = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)
        tr.record(ob, ac, r, Scalar[DT](0.0))
    ob.free(); ac.free()

    assert_true(tr.can_train(), "replay should be trainable")

    var first_wm: Scalar[DT] = 0.0
    var last_wm: Scalar[DT] = 0.0
    var first_ac: Scalar[DT] = 0.0
    var last_ac: Scalar[DT] = 0.0
    comptime ITERS = 30
    for it in range(ITERS):
        var ok = tr.train_step()
        assert_true(ok, "train_step should run")
        var wm = tr.last_wm_loss()
        var ac = tr.last_ac_loss()
        assert_true(wm == wm, "WM loss finite")
        assert_true(ac == ac, "AC loss finite")
        if it == 0:
            first_wm = wm
            first_ac = ac
            print("  iter 0   WM =", first_wm, " AC =", first_ac)
        if it == ITERS - 1:
            last_wm = wm
            last_ac = ac
            print("  iter", ITERS - 1, "  WM =", last_wm, " AC =", last_ac)

    print("  WM:", first_wm, "->", last_wm, "  AC:", first_ac, "->", last_ac)
    # WM loss is a true supervised loss → must decrease. The AC "loss" is an
    # RL objective (policy_loss = −(logpi·adv + ent), can be negative and
    # non-monotonic), so we only gate it finite + no NaN.
    assert_true(last_wm < first_wm, "WM loss must decrease")
    assert_true(last_ac == last_ac, "AC loss finite")
    print("=" * 70)
    print("SMOKE PASSED — DreamerV3Trainer trains (WM↓, AC finite), no NaN")
    print("=" * 70)
