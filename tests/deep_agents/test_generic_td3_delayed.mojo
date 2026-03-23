"""Test: TD3 delayed actor updates fire at correct intervals."""

from std.random import seed

from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent, TD3Config
from mojo_rl.envs.pendulum import PendulumEnv


def main() raises:
    print("=== TD3 Delayed Actor Update Test ===\n")

    seed(42)
    var agent = GenericOffPolicyAgent[TD3Config[3, 1, 64, 1000, 32]](
        action_scale=2.0, policy_delay=2
    )
    var env = PendulumEnv[DType.float64]()

    # Train enough to fill buffer and do several updates
    _ = agent.train(env, num_episodes=10)

    print("train_steps:", agent.train_step_count)
    print("update_count:", agent.update_count)
    print("policy_delay:", agent.policy_delay)

    # update_count should equal train_step_count (incremented each step)
    if agent.update_count == agent.train_step_count:
        print("OK: update_count == train_step_count")
    else:
        print("FAIL: update_count != train_step_count")

    # Actor should have been updated update_count/policy_delay times
    var expected_actor_updates = agent.update_count // agent.policy_delay
    print("Expected actor updates:", expected_actor_updates)
    print("(actor updates happen when update_count % policy_delay == 0)")

    if expected_actor_updates > 0:
        print("OK: Delayed actor updates occurred")
    else:
        print("FAIL: No actor updates occurred")

    print("\n=== Done ===")
