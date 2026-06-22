"""TD-MPC2 single-task agent CPU smoke (storage framework).

Constructs a small `TDMPC2Agent["cpu", ...]` for a Pendulum-like task
(OBS=3, ACT=1), records random transitions, runs a few hundred train_steps,
and asserts the WM + policy losses stay finite (not NaN/Inf) and that
train_step returns True after learning_starts. Convergence gating is the
parent's job — this only checks finite + runs end to end.

Run:
  pixi run -e apple mojo run -I . tests/nn/test_tdmpc2_pendulum_cpu_smoke.mojo
"""

from std.math import isnan, isinf
from std.random import random_float64
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.agent import TDMPC2Agent


comptime OBS = 3
comptime ENC = 16
comptime ACT = 1
comptime LATENT = 8
comptime MLP = 16
comptime BINS = 5
comptime SN = 4
comptime VMIN = -5
comptime VMAX = 5
comptime B = 32
comptime H = 3
comptime CAP = 2048

comptime AgentT = TDMPC2Agent[
    "cpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP,
]


def _rand_obs() -> List[Scalar[DT]]:
    var o = List[Scalar[DT]](length=OBS, fill=0)
    for i in range(OBS):
        o[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    return o^


def main() raises:
    print("=" * 60)
    print("TD-MPC2 single-task CPU smoke (storage)")
    print("=" * 60)
    var ag = AgentT.make(learning_starts=200)

    # Fill the replay with random transitions.
    var act = List[Scalar[DT]](length=ACT, fill=0)
    for step in range(800):
        var obs = _rand_obs()
        ag.select_action(obs, act, explore=True)
        var r = Scalar[DT](random_float64() * 2.0 - 1.0)
        var done = Scalar[DT](1.0) if (step % 50 == 49) else Scalar[DT](0.0)
        ag.record(obs, act, r, done)

    var n_trained = 0
    for _ in range(300):
        var did = ag.train_step()
        if did:
            n_trained += 1
            var wm = ag.last_wm_loss()
            var pi = ag.last_pi_loss()
            assert_true(
                not (isnan(Float64(wm)) or isinf(Float64(wm))),
                "wm loss finite",
            )
            assert_true(
                not (isnan(Float64(pi)) or isinf(Float64(pi))),
                "pi loss finite",
            )

    print("  trained steps:", n_trained)
    print("  last wm loss:", ag.last_wm_loss())
    print("  last pi loss:", ag.last_pi_loss())
    print("  pi scale:", ag.pi_scale())
    assert_true(n_trained > 0, "train_step returned True after learning_starts")
    print("ALL PASSED")
