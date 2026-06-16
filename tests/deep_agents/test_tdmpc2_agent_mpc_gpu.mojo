"""TD-MPC2 agent select_action_mpc smoke (Apple Metal).

Builds a GPU agent (small planning config) and runs one MPC action via the
agent-owned MPPIGPUBatched + a transient GPU rollout callback. Gate: the
action is finite and in [-action_scale, action_scale]. Validates the full
agent MPC wiring (encode → callback → plan_gpu → D2H + persistent planner
warm-start). Launch-bound on Metal (slow); fast on NVIDIA.

Run: `pixi run -e apple mojo run -I . tests/deep_agents/test_tdmpc2_agent_mpc_gpu.mojo`
"""

from std.memory import alloc
from std.random import seed
from std.math import isfinite
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.agent import TDMPC2Agent

comptime OBS = 3
comptime ENC = 32
comptime ACT = 1
comptime LATENT = 32
comptime MLP = 32
comptime BINS = 21
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8
comptime H = 3
comptime CAP = 2000
# small MPC planning config
comptime NS = 16
comptime NPT = 4
comptime NE = 8
comptime NI = 2

comptime Ag = TDMPC2Agent[
    "gpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP,
    NS, NPT, NE, NI,
]


def main() raises:
    print("=" * 70)
    print("TD-MPC2 agent select_action_mpc smoke (Apple)")
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()
    var ag = Ag.make(
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=0,
        ctx=ctx,
    )

    var obs = alloc[Scalar[DT]](OBS)
    obs[0] = Scalar[DT](0.3)
    obs[1] = Scalar[DT](-0.5)
    obs[2] = Scalar[DT](1.2)
    var act = alloc[Scalar[DT]](ACT)

    ag.mpc_start_episode()
    ag.select_action_mpc(obs, act, explore=False)

    print("  MPC action =", act[0])
    assert_true(isfinite(act[0]), "MPC action finite")
    assert_true(
        act[0] >= -2.0001 and act[0] <= 2.0001, "MPC action in [-scale, scale]"
    )
    print("=" * 70)
    print("SMOKE PASSED — agent select_action_mpc works on GPU")
    print("=" * 70)
    obs.free(); act.free()
