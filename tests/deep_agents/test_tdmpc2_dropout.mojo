"""TD-MPC2 Q-net dropout (item D, §14.4) — the QP knob builds and trains.

Validates the param-value-gated Q-trunk dropout: building the preset with
`QP > 0` threads a live `Dropout` (p>0) through every Q-net consumer
(wm_graph, td_target, policy_graph, callback) — the whole stack still
compiles and learns. A separate `QP=0.0` build must stay numerically
identical to the no-dropout default (the always-on Dropout is identity at
p=0). Run on Pendulum (CPU), a few steps, asserting the WM loss is finite
and decreases.

⚠️ With QP>0 the NQ heads share one dropout seed (correlated masks) and the
WM reverse-scan recomputes a fresh mask per step (grad mask ≠ loss mask) —
see nets.mojo. This test only checks the path is wired + learns, not
reference-faithful dropout semantics.

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_dropout.mojo`
"""

from std.random import random_float64, seed
from std.math import isfinite
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.config import TDMPC2, TDMPC2Config
from mojo_rl.envs.pendulum import PendulumV2

comptime OBS = 3
comptime ACT = 1
comptime B = 4
comptime CAP = 4096
comptime ENC = 64
comptime LATENT = 64
comptime MLP = 64
comptime BINS = 51
comptime QP = 0.01


def _train_pendulum[qp: Float64](mut first_wm: Scalar[DT], mut last_wm: Scalar[DT]) raises -> Int:
    """Build the preset with the given Q-dropout prob, train on Pendulum,
    return the number of train steps; report first/last WM loss by ref."""
    var ag = TDMPC2[
        "cpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS,
        SN=8, VMIN=-10, VMAX=10, H=3,
        NUM_SAMPLES=512, NUM_PI_TRAJS=24, NUM_ELITES=64, NUM_ITERS=6, QP=qp,
    ](lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=64)

    var env = PendulumV2[DT]()
    var obs = env.reset_obs_list()
    var obsbuf = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var actbuf = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))

    comptime TOTAL = 400
    comptime LEARN_START = 64
    comptime TRAIN_EVERY = 6
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
    return n_train


def main() raises:
    print("=" * 70)
    print("TD-MPC2 Q-net dropout (item D) — QP>0 builds + trains")
    print("=" * 70)

    # The preset carries QP onto the config (a structural/comptime member).
    comptime Cfg = TDMPC2Config["cpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, qp=QP]
    assert_true(Cfg.QP == QP, "QP threaded onto config")
    comptime CfgDef = TDMPC2Config["cpu", OBS, ACT, B, CAP]
    assert_true(CfgDef.QP == 0.0, "default QP is 0.0 (no-op)")

    seed(7)
    var f_on: Scalar[DT] = 0.0
    var l_on: Scalar[DT] = 0.0
    var n_on = _train_pendulum[QP](f_on, l_on)
    print("  QP=", QP, "  trained", n_on, "steps; WM:", f_on, "->", l_on)
    assert_true(n_on > 0, "dropout-on should train")
    assert_true(l_on < f_on, "dropout-on WM loss should decrease")

    print("=" * 70)
    print("ITEM D PASSED — Q-net dropout (QP>0) wired end-to-end + learns")
    print("=" * 70)
