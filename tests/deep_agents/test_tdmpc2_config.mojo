"""TD-MPC2 config facade smoke (CPU) — the `TDMPC2[...]` preset builds and
trains the same agent the primitive does.

Validates the Design-F facade (config.mojo): the capitalized preset reads
like a constructor, applies the tuned scalar defaults, threads dim +
planning overrides, and returns a working `TDMPC2Agent`. Runs a few train
steps on Pendulum and checks the WM loss is finite + decreases — i.e. the
facade wires the real learning stack, not a stub.

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_config.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.math import isfinite
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.tdmpc2.config import TDMPC2, TDMPC2Config
from mojo_rl.deep_agents2.tdmpc2.agent import TDMPC2Agent
from mojo_rl.envs.pendulum import PendulumV2

comptime OBS = 3
comptime ACT = 1
comptime B = 4
comptime CAP = 4096
# Small architecture overrides (defaults are the reference latent512 dims).
comptime ENC = 64
comptime LATENT = 64
comptime MLP = 64
comptime BINS = 51


def main() raises:
    print("=" * 70)
    print("TD-MPC2 config facade smoke (CPU) — Pendulum via TDMPC2[...]")
    print("=" * 70)
    seed(7)

    # Build through the preset: only OBS/ACT/B/CAP are mandatory, dims
    # overridden as keyword params, scalars default to the tuned config.
    var ag = TDMPC2[
        "cpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS,
    ](lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=64)

    # The preset must return exactly the primitive agent type — assigning to
    # the explicitly-spelled type proves the param mapping is correct.
    comptime Ag = TDMPC2Agent[
        "cpu", OBS, ENC, ACT, LATENT, MLP, BINS, 8, -10, 10, B, 3, CAP,
        512, 24, 64, 6,
    ]
    var _typed: Ag = ag^

    # Tuned defaults flow from the config (not the agent's own make defaults).
    comptime Cfg = TDMPC2Config["cpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS]
    assert_true(Cfg.DEF_GAMMA == Scalar[DT](0.99), "DEF_GAMMA")
    assert_true(Cfg.DEF_TAU == Scalar[DT](0.01), "DEF_TAU")
    assert_true(Cfg.NUM_SAMPLES == 512, "ref MPPI samples")
    assert_true(Cfg.LATENT == LATENT, "dim override threaded")

    var env = PendulumV2[DT]()
    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

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
            _typed.select_action(obsbuf, actbuf, explore=True)
        var act_list = List[Scalar[DT]]()
        act_list.append(actbuf[0])
        var res = env.step_continuous_vec[DT](act_list)
        _typed.record(
            obsbuf, actbuf, res[1],
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            if _typed.train_step():
                var wm = _typed.last_wm_loss()
                assert_true(isfinite(wm), "WM finite")
                if n_train == 0:
                    first_wm = wm
                last_wm = wm
                n_train += 1

    print("  trained", n_train, "steps; WM:", first_wm, "->", last_wm)
    assert_true(n_train > 0, "should have trained")
    assert_true(last_wm < first_wm, "WM loss should decrease")
    print("=" * 70)
    print("CONFIG FACADE PASSED — TDMPC2[...] preset builds + trains")
    print("=" * 70)
    obsbuf.free(); actbuf.free()
