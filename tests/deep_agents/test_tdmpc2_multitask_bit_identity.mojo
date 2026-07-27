"""TD-MPC2 single-task bit-identity gate after the multi-task (item C) work.

The multi-task feature lives entirely in new `*_mt.mojo` files; the only shared
edit is the additive `task` lane in `SequenceReplay` (new field + `record_task`/
`sample_batch_task` methods — the trait `record`/`sample_batch` are byte-
unchanged). This test re-runs the single-task Pendulum smoke (same seed/config as
`test_tdmpc2_config.mojo`) and asserts the EXACT WM-loss anchor recorded BEFORE
the item-C work, proving no regression leaked into the single-task path.

Anchor (seed 7, 56 train steps): 0.81461537 -> 0.21972585.

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_multitask_bit_identity.mojo`
"""

from std.random import random_float64, seed
from std.math import isfinite, abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.config import TDMPC2
from mojo_rl.envs.pendulum import PendulumV2

comptime OBS = 3
comptime ACT = 1
comptime B = 4
comptime CAP = 4096
comptime ENC = 64
comptime LATENT = 64
comptime MLP = 64
comptime BINS = 51

comptime FIRST_ANCHOR = Scalar[DT](0.81461537)
comptime LAST_ANCHOR = Scalar[DT](0.21972585)


def main() raises:
    print("=" * 70)
    print("TD-MPC2 single-task bit-identity gate (post item-C)")
    print("=" * 70)
    seed(7)
    var ag = TDMPC2[
        "cpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS,
    ](lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=64)

    var env = PendulumV2[DT]()
    var obs = env.reset_obs_list()
    var obsbuf = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var actbuf = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))

    comptime TOTAL = 400
    comptime LEARN_START = 64
    comptime TRAIN_EVERY = 6
    var first_wm: Scalar[DT] = 0.0
    var last_wm: Scalar[DT] = 0.0
    var n_train = 0

    for step in range(TOTAL):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            actbuf[0] = Scalar[DT](random_float64() * 4.0 - 2.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var act_list = List[Scalar[DT]]()
        act_list.append(actbuf[0])
        var res = env.step_continuous_vec[DT](act_list)
        ag.record(
            obsbuf, actbuf, res[1],
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            if ag.train_step():
                var wm = ag.last_wm_loss()
                assert_true(isfinite(wm), "WM finite")
                if n_train == 0:
                    first_wm = wm
                last_wm = wm
                n_train += 1

    print("  WM:", first_wm, "->", last_wm, " (", n_train, "train steps)")
    print("  anchor:", FIRST_ANCHOR, "->", LAST_ANCHOR)
    assert_true(n_train == 56, "train-step count unchanged")
    assert_true(
        abs(first_wm - FIRST_ANCHOR) < Scalar[DT](1e-5),
        "first WM bit-identical to pre-item-C anchor",
    )
    assert_true(
        abs(last_wm - LAST_ANCHOR) < Scalar[DT](1e-5),
        "last WM bit-identical to pre-item-C anchor",
    )
    print("=" * 70)
    print("BIT-IDENTITY PASSED — single-task path unchanged by item C")
    print("=" * 70)
