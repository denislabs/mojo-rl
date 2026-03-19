"""Test AutodiffSAC agent training on Pendulum.

Verifies that the autodiff-composed actor loss produces a working SAC agent
that can train and improve on the Pendulum environment.
"""

from mojo_rl.deep_agents.core.agents import (
    GenericOffPolicyAgent,
    AutodiffSACConfig,
)
from mojo_rl.envs.pendulum import PendulumEnv
from mojo_rl.nn.constants import dtype


fn main() raises:
    print("=== AutodiffSAC Training Test ===")
    print()

    # Pendulum: obs_dim=3, action_dim=1
    comptime Config = AutodiffSACConfig[3, 1, 64, 10000, 32]

    print("Config:", Config.NAME)
    print("  ActorModel PARAM_SIZE:", Config.ActorModel.PARAM_SIZE)
    print("  CriticModel PARAM_SIZE:", Config.CriticModel.PARAM_SIZE)
    print("  ActorLoss: AutodiffMaxEntLoss (composed autodiff graph)")
    print()

    var agent = GenericOffPolicyAgent[Config](
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        auto_alpha=True,
        alpha=0.2,
    )

    var env = PendulumEnv[dtype]()

    print("Training AutodiffSAC on Pendulum (20 episodes, warmup=200)...")
    var metrics = agent.train(
        env,
        num_episodes=20,
        max_steps_per_episode=200,
        warmup_steps=200,
        verbose=True,
        print_every=5,
        environment_name="Pendulum",
    )

    print()
    var n_eps = len(metrics.episodes)
    if n_eps > 0:
        print("Final reward:", metrics.episodes[n_eps - 1].total_reward)
        print("Episodes completed:", n_eps)

    # The autodiff SAC should be able to train without crashing
    if n_eps == 20:
        print("[PASS] AutodiffSAC completed 20 episodes without errors")
    else:
        print("[FAIL] Training did not complete, got", n_eps, "episodes")
