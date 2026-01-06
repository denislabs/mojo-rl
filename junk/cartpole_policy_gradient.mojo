"""CartPole with Policy Gradient Methods.

Demonstrates training on CartPole using policy gradient algorithms:
- REINFORCE (Monte Carlo Policy Gradient)
- Actor-Critic (online TD-based updates)
- Actor-Critic with eligibility traces
- A2C (Advantage Actor-Critic with n-step returns)

Policy gradient methods directly learn a parameterized policy π(a|s;θ)
without maintaining explicit value estimates for action selection.

Advantages over value-based methods:
1. Can learn stochastic policies (useful for partially observable environments)
2. Smooth policy updates (no sudden policy changes from epsilon-greedy)
3. Natural for continuous action spaces (not demonstrated here)
4. Better convergence properties in some settings

Usage:
    pixi run mojo run examples/cartpole_policy_gradient.mojo
"""

from agents.reinforce import REINFORCEAgent
from agents.actor_critic import ActorCriticAgent, ActorCriticLambdaAgent, A2CAgent


fn main() raises:
    """Run policy gradient comparison on CartPole."""
    print("=" * 60)
    print("CartPole with Policy Gradient Methods")
    print("=" * 60)
    print("")

    # Train REINFORCE
    print("-" * 60)
    print("1. REINFORCE (Monte Carlo Policy Gradient)")
    print("-" * 60)
    var reinforce_agent, reinforce_metrics = REINFORCEAgent.train(
        num_episodes=500,
        learning_rate=0.001,
        use_baseline=True,
        verbose=True,
    )

    print("")

    # Train Actor-Critic
    print("-" * 60)
    print("2. Actor-Critic (Online TD-based)")
    print("-" * 60)
    var ac_agent, ac_metrics = ActorCriticAgent.train(
        num_episodes=500,
        actor_lr=0.001,
        critic_lr=0.01,
        verbose=True,
    )

    print("")

    # Train Actor-Critic(λ)
    print("-" * 60)
    print("3. Actor-Critic(lambda) (Eligibility Traces)")
    print("-" * 60)
    var ac_lambda_agent, ac_lambda_metrics = ActorCriticLambdaAgent.train(
        num_episodes=500,
        actor_lr=0.001,
        critic_lr=0.01,
        lambda_=0.9,
        verbose=True,
    )

    print("")

    # Train A2C
    print("-" * 60)
    print("4. A2C (N-step Returns)")
    print("-" * 60)
    var a2c_agent, a2c_metrics = A2CAgent.train(
        num_episodes=500,
        actor_lr=0.001,
        critic_lr=0.01,
        n_steps=5,
        verbose=True,
    )

    print("")

    # Summary using metrics
    print("=" * 60)
    print("Training Complete - Final Results")
    print("=" * 60)
    print("")
    print("REINFORCE (with baseline):")
    print("  Mean reward:", reinforce_metrics.mean_reward())
    print("  Max reward:", reinforce_metrics.max_reward())
    print("")
    print("Actor-Critic:")
    print("  Mean reward:", ac_metrics.mean_reward())
    print("  Max reward:", ac_metrics.max_reward())
    print("")
    print("Actor-Critic(lambda):")
    print("  Mean reward:", ac_lambda_metrics.mean_reward())
    print("  Max reward:", ac_lambda_metrics.max_reward())
    print("")
    print("A2C:")
    print("  Mean reward:", a2c_metrics.mean_reward())
    print("  Max reward:", a2c_metrics.max_reward())
    print("")
    print("CartPole solved when avg reward >= 475 over 100 episodes")
    print("=" * 60)
