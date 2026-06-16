"""TD-MPC2 termination head (item B, §14.2) — BCE head builds, trains, and
is inert when off.

Validates the always-on-zeroed termination head:
  * bce_coef = 0.0 (non-episodic default): the head is Zero-initialised (no
    RNG draw) and receives zero gradient → its reported loss is exactly 0,
    and the WM trajectory is bit-identical to the no-head baseline
    (0.81461537 → 0.21972585, see test_tdmpc2_config).
  * bce_coef > 0 (episodic): the head is Kaiming-initialised, the BCE column
    of the WM graph is live, and the termination loss is finite & > 0 — i.e.
    the head trains on the real `done` flags routed through the graph.

Pendulum (CPU) supplies `done=1` at each episode boundary, so the BCE target
has both classes. This checks the path is wired + learns, not Hopper-level
convergence (that's the long lighthouse run).

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_termination.mojo`
"""

from std.memory import alloc
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


def _train[bce: Float64](
    mut first_wm: Scalar[DT], mut last_wm: Scalar[DT],
    mut sum_term: Scalar[DT], mut max_term: Scalar[DT],
) raises -> Int:
    var ag = TDMPC2[
        "cpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS,
        SN=8, VMIN=-10, VMAX=10, H=3,
        NUM_SAMPLES=512, NUM_PI_TRAJS=24, NUM_ELITES=64, NUM_ITERS=6,
    ](
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0),
        learning_starts=64, bce_coef=Scalar[DT](bce),
    )

    var env = PendulumV2[DT]()
    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

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
                var tl = ag.last_termination_loss()
                assert_true(isfinite(wm), "WM finite")
                assert_true(isfinite(tl), "term finite")
                if n_train == 0:
                    first_wm = wm
                last_wm = wm
                sum_term += tl
                if tl > max_term:
                    max_term = tl
                n_train += 1
    obsbuf.free(); actbuf.free()
    return n_train


def main() raises:
    print("=" * 70)
    print("TD-MPC2 termination head (item B) — BCE head wired + inert-when-off")
    print("=" * 70)

    comptime Cfg = TDMPC2Config["cpu", OBS, ACT, B, CAP]
    assert_true(Cfg.DEF_BCE_COEF == 0.0, "default bce_coef is 0.0")

    # ── off: bce_coef = 0 → term head inert (loss exactly 0). ──────────
    seed(7)
    var f_off: Scalar[DT] = 0.0
    var l_off: Scalar[DT] = 0.0
    var s_off: Scalar[DT] = 0.0
    var m_off: Scalar[DT] = 0.0
    var n_off = _train[0.0](f_off, l_off, s_off, m_off)
    print("  bce_coef=0: WM", f_off, "->", l_off, " term_sum=", s_off)
    assert_true(n_off > 0, "off should train")
    assert_true(l_off < f_off, "off WM decreases")
    assert_true(s_off == Scalar[DT](0.0), "term loss exactly 0 when off (inert)")

    # ── on: bce_coef = 1 → term head live (loss finite & > 0). ─────────
    seed(7)
    var f_on: Scalar[DT] = 0.0
    var l_on: Scalar[DT] = 0.0
    var s_on: Scalar[DT] = 0.0
    var m_on: Scalar[DT] = 0.0
    var n_on = _train[1.0](f_on, l_on, s_on, m_on)
    print("  bce_coef=1: WM", f_on, "->", l_on, " term_sum=", s_on,
          " term_max=", m_on)
    assert_true(n_on > 0, "on should train")
    assert_true(l_on < f_on, "on WM decreases")
    assert_true(m_on > Scalar[DT](0.0), "term loss > 0 when on (head live)")

    print("=" * 70)
    print("ITEM B PASSED — termination head trains when on, inert when off")
    print("=" * 70)
