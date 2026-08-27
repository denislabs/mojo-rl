"""TD-MPC2 single-task agent GPU smoke (storage framework, Apple Metal).

Constructs a small `TDMPC2Agent["gpu", ...]`, records random transitions, runs
a few train_steps, and asserts the WM + policy losses stay finite. Convergence
gating is the parent's job — this only checks the GPU path builds + runs.

Run:
  pixi run -e apple mojo run -I . tests/nn/test_tdmpc2_pendulum_gpu_smoke.mojo
"""

from std.math import isnan, isinf
from std.random import random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

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
comptime B = 16
comptime H = 3
comptime CAP = 1024

comptime AgentT = TDMPC2Agent[
    "gpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP,
]


def _rand_obs() -> List[Scalar[DT]]:
    var o = List[Scalar[DT]](length=OBS, fill=0)
    for i in range(OBS):
        o[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    return o^


def main() raises:
    print("=" * 60)
    print("TD-MPC2 single-task GPU smoke (storage)")
    print("=" * 60)
    var c = DeviceContext()
    var ag = AgentT.make(learning_starts=100, ctx=Optional(c))

    var act = List[Scalar[DT]](length=ACT, fill=0)
    for step in range(400):
        var obs = _rand_obs()
        ag.select_action(obs, act, explore=True)
        var r = Scalar[DT](random_float64() * 2.0 - 1.0)
        var done = Scalar[DT](1.0) if (step % 50 == 49) else Scalar[DT](0.0)
        ag.record(obs, act, r, done)

    var n_trained = 0
    for _ in range(60):
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
