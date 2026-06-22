"""DQN storage smoke — compile the full stack (config→agent→trainer→blocks)
and run a short CPU CartPole train + greedy eval. Fast compile-check before the
full convergence gate; asserts learning beats random.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_dqn_smoke.mojo
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dqn.config import DQN
from mojo_rl.envs.cartpole import CartPoleEnv

comptime OBS = 4
comptime ACT = 2
comptime BATCH = 32
comptime CAP = 20_000


def main() raises:
    print("=== DQN storage smoke (CPU CartPole) ===")
    seed(42)
    var agent = DQN["cpu", OBS, ACT, BATCH, CAP, 64](
        lr=Scalar[DT](2.5e-4),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=1_000,
        target_update_freq=500,
    )
    var env = CartPoleEnv[DT]()
    _ = agent.train(env, total_timesteps=12_000, print_every=4_000)

    var eval_env = CartPoleEnv[DT]()
    var ret = agent.eval(eval_env, num_episodes=5, max_steps_per_episode=200)
    print("  eval mean return =", ret, " (random ~22)")
    assert_true(ret > Scalar[DT](60.0), "DQN did not learn (eval <= 60)")
    print("=== PASSED ===")
