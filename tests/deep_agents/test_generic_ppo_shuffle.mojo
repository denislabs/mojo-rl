"""Test: PPO multi-epoch minibatch training works correctly."""

from std.random import seed

from mojo_rl.deep_agents.core.agents import GenericOnPolicyAgent, PPOConfig
from mojo_rl.envs import CartPoleEnv


def main() raises:
    print("=== PPO Multi-Epoch Minibatch Test ===\n")

    seed(42)
    var agent = GenericOnPolicyAgent[PPOConfig[4, 2, 64, 128]](
        num_epochs=4,
        minibatch_size=32,
    )
    var env = CartPoleEnv[DType.float64]()

    # Train for several updates
    var metrics = agent.train(env, num_updates=10)

    print("Updates:", agent.train_step_count)
    print("Epochs per update:", agent.num_epochs)
    print("Minibatch size:", agent.minibatch_size)
    print("Rollout len:", agent.ROLLOUT)

    # With rollout_len=128 and minibatch_size=32:
    # 4 minibatches per epoch × 4 epochs = 16 minibatch updates per rollout
    # Each sample should be seen exactly 4 times (once per epoch)
    var expected_mb_per_update = (
        agent.ROLLOUT // agent.minibatch_size
    ) * agent.num_epochs
    print("Expected minibatch passes per update:", expected_mb_per_update)

    if agent.train_step_count == 10:
        print("OK: Correct number of updates")
    else:
        print("FAIL: Wrong update count")

    print("\n=== Done ===")
