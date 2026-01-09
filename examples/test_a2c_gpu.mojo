from deep_agents.gpu import A2CAgent
from envs import CartPoleEnv
from random import seed


fn main() raises:
    seed(42)
    print("GPU A2C on CartPole")
    print()

    # Create agent with CartPole dimensions (OBS_DIM=4, NUM_ACTIONS=2)
    var agent = A2CAgent[HIDDEN_DIM=256]()

    # Train on GPU - returns TrainingMetrics (DeviceContext created internally)
    var metrics = agent.train[CartPoleEnv, NUM_ENVS=1024](
        num_updates=200,
        verbose=True,
        environment_name="CartPole",
    )

    # Print training summary
    print()
    metrics.print_summary()

    # Evaluate on CPU
    print()
    print("Evaluating trained agent...")
    var env = CartPoleEnv()
    var eval_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps_per_episode=500,
        verbose=True,
    )
    print()
    print("Mean evaluation reward:", eval_reward)

    var _ = agent.evaluate(
        env,
        num_episodes=1,
        max_steps_per_episode=500,
        verbose=True,
        render=True,
    )
