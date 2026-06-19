"""SAC AGENT convergence on Pendulum-v1 — storage SACTrainer driver gate.

Builds the SAC agent via the `SAC["cpu", 3, 1, BATCH, CAP]()` facade (storage
trainer + drivers), trains single-env on the real Pendulum env through
`run_offpolicy_train` (via `agent.train_single`), then greedy-evals. Asserts
the policy clearly learns: mean return > -400 (solved ~ -169; the direct-loop
convergence gate hits -145).

This validates the driver-conforming `SACTrainer` assembly end-to-end — the
counterpart to `test_storage_sac_pendulum_convergence.mojo` (direct loop).

Run: pixi run mojo run -I . tests/deep_agents/test_storage_sac_agent_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac.config import SAC
from mojo_rl.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 128
comptime BATCH = 128
comptime CAP = 100_000
comptime ASCALE = Scalar[DT](2.0)   # Pendulum torque in [-2, 2]


def main() raises:
    seed(42)
    print("=" * 60)
    print("SAC AGENT Pendulum-v1 convergence (storage trainer, CPU)")
    print("=" * 60)

    var env = PendulumEnv[DT]()

    var agent = SAC["cpu", OBS, ACT, BATCH, CAP, H](
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=ASCALE,
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-Float64(ACT)),
        learning_starts=500,
    )

    var rand_eval = agent.eval(env, num_episodes=5, max_steps_per_episode=200)
    print("eval @0 (random):", rand_eval)

    _ = agent.train_single(env, total_timesteps=12_000, print_every=3_000)

    var final_eval = agent.eval(
        env, num_episodes=10, max_steps_per_episode=200
    )
    print("FINAL greedy eval(10):", final_eval)
    assert_true(
        final_eval > Scalar[DT](-400.0),
        "SAC agent learns Pendulum (eval return > -400; solved ~ -169)",
    )
    print("SAC AGENT PENDULUM CONVERGENCE OK")
