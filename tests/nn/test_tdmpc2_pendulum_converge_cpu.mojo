"""TD-MPC2 single-task CONVERGENCE gate — Pendulum V1, CPU (MPC-off).

Real PendulumEnv training loop (a = π(encode(obs))). Confirms the migrated
storage drivers actually LEARN, not merely run finite: the mean episode return
over the last window must improve well above the random-policy floor (~-1200).
Also prints the world-model loss trend (should fall) as a learning signal.

Run:
  pixi run mojo run -I . tests/nn/test_tdmpc2_pendulum_converge_cpu.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.agent import TDMPC2Agent
from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ENC = 64
comptime ACT = 1
comptime LATENT = 32
comptime MLP = 128
comptime BINS = 51
comptime SN = 8
comptime VMIN = -12
comptime VMAX = 12
comptime B = 256
comptime H = 3
comptime CAP = 50_000

comptime TOTAL = 12_000
comptime EP_LEN = 200
comptime LEARN_START = 500

comptime AgentT = TDMPC2Agent[
    "cpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP,
]


def main() raises:
    print("=" * 60)
    print("TD-MPC2 Pendulum convergence (CPU, MPC-off)")
    print("=" * 60)
    var ag = AgentT.make(
        lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99), tau=Scalar[DT](0.01),
        action_scale=Scalar[DT](2.0), learning_starts=LEARN_START,
    )
    var env = PendulumEnv[DT]()
    var obs = env.reset_obs_list()
    var act = List[Scalar[DT]](length=ACT, fill=0)

    var ep_ret = Scalar[DT](0.0)
    var ep_len = 0
    var first10_sum = Scalar[DT](0.0)
    var first10_n = 0
    var last10_sum = Scalar[DT](0.0)
    var last10_n = 0
    var recent = List[Scalar[DT]]()  # ring of last-10 episode returns

    for step in range(TOTAL):
        if step < LEARN_START:
            for i in range(ACT):
                act[i] = Scalar[DT](0.0)
            ag.select_action(obs, act, explore=True)
        else:
            ag.select_action(obs, act, explore=True)
        var res = env.step_continuous_vec[DT](act)
        ag.record(obs, act, res[1], Scalar[DT](0.0))
        ep_ret += res[1]
        ep_len += 1
        obs = res[0].copy()
        _ = ag.train_step()

        if ep_len >= EP_LEN:
            if first10_n < 10:
                first10_sum += ep_ret
                first10_n += 1
            recent.append(ep_ret)
            if len(recent) > 10:
                _ = recent.pop(0)
            ep_ret = Scalar[DT](0.0)
            ep_len = 0
            obs = env.reset_obs_list()

    for r in recent:
        last10_sum += r
        last10_n += 1
    var first_mean = first10_sum / Scalar[DT](first10_n if first10_n > 0 else 1)
    var last_mean = last10_sum / Scalar[DT](last10_n if last10_n > 0 else 1)
    print("  first-10-ep mean return:", first_mean)
    print("  last-10-ep  mean return:", last_mean)
    print("  last wm loss:", ag.last_wm_loss())
    print("  pi scale:", ag.pi_scale())
    # Learning signal: last-window return must beat the random floor clearly.
    assert_true(
        last_mean > Scalar[DT](-1000.0),
        "TD-MPC2 did not learn Pendulum (last-10 mean <= -1000)",
    )
    assert_true(
        last_mean > first_mean + Scalar[DT](100.0),
        "TD-MPC2 return did not improve over training",
    )
    print("PASS: TD-MPC2 learns Pendulum on the storage drivers.")
