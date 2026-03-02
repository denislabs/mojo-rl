"""Deep A2C and PPO on CartPole.

Run with: pixi run mojo run examples/cartpole_deep_a2c_ppo.mojo

This trains Deep A2C and Deep PPO on the native Mojo CartPole environment.
Compares the two on-policy algorithms:

A2C (Advantage Actor-Critic):
- On-policy learning with single-step updates
- GAE for advantage estimation
- No clipping (simpler but potentially less stable)

PPO (Proximal Policy Optimization):
- Clipped surrogate objective for stability
- Multiple epochs per rollout
- Generally more sample efficient

Expected performance on CartPole:
- Episode 100: ~100-200 avg reward (learning)
- Episode 300: ~400-500 avg reward (near optimal)

CartPole is solved when average reward > 475 over 100 episodes.
"""

from random import seed

from envs import CartPoleEnv
from deep_agents.a2c import DeepA2CAgent
from deep_agents.ppo import DeepPPOAgent


fn main() raises:
    print("=" * 60)
    print("Deep A2C and PPO on CartPole")
    print("=" * 60)
    print()

    # Seed for reproducibility
    seed(42)

    # Training parameters
    var num_episodes = 500
    var max_steps = 500

    print("Training Configuration:")
    print("  Episodes:", num_episodes)
    print("  Max steps per episode:", max_steps)
    print()

    # ========================================
    # Train Deep A2C
    # ========================================
    print("=" * 60)
    print("Training Deep A2C...")
    print("=" * 60)

    var env_a2c = CartPoleEnv[DType.float64]()

    # Create A2C agent
    # CartPole: 4D observations, 2 discrete actions
    var a2c_agent = DeepA2CAgent[
        obs_dim=4,
        num_actions=2,
        hidden_dim=64,
        rollout_len=128,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        actor_lr=0.0003,
        critic_lr=0.001,
        entropy_coef=0.01,
        value_loss_coef=0.5,
    )

    var a2c_metrics = a2c_agent.train(
        env_a2c,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=50,
        environment_name="CartPole",
    )

    print()
    print("A2C Final Evaluation...")
    var a2c_eval = a2c_agent.evaluate(env_a2c, num_episodes=10, verbose=True)
    print("  A2C Average Reward:", a2c_eval)

    # ========================================
    # Train Deep PPO
    # ========================================
    print()
    print("=" * 60)
    print("Training Deep PPO...")
    print("=" * 60)

    var env_ppo = CartPoleEnv[DType.float64]()

    # Create PPO agent
    var ppo_agent = DeepPPOAgent[
        obs_dim=4,
        num_actions=2,
        hidden_dim=64,
        rollout_len=256,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        actor_lr=0.0003,
        critic_lr=0.001,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        num_epochs=4,
        minibatch_size=64,
    )

    var ppo_metrics = ppo_agent.train(
        env_ppo,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=50,
        environment_name="CartPole",
    )

    print()
    print("PPO Final Evaluation...")
    var ppo_eval = ppo_agent.evaluate(env_ppo, num_episodes=10, verbose=True)
    print("  PPO Average Reward:", ppo_eval)

    # ========================================
    # Results Summary
    # ========================================
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()
    print("Mean Training Reward:")
    print("  A2C:", a2c_metrics.mean_reward())
    print("  PPO:", ppo_metrics.mean_reward())
    print()
    print("Final Evaluation (10 episodes, greedy):")
    print("  A2C:", a2c_eval)
    print("  PPO:", ppo_eval)
    print()
    print("Training complete!")
