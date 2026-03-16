"""Test: SAC alpha auto-tuning via unified GenericOffPolicyAgent."""

from std.random import seed
from std.math import abs

from mojo_rl.deep_agents.core.generic import GenericOffPolicyAgent, SACConfig
from mojo_rl.envs.pendulum import PendulumEnv


fn main() raises:
    print("=== SAC Alpha Auto-Tuning Test ===\n")

    seed(42)
    var agent = GenericOffPolicyAgent[SACConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0,
        alpha=0.2,
        auto_alpha=True,
    )

    var initial_alpha = agent.alpha
    print("Initial alpha:", initial_alpha)

    var env = PendulumEnv[DType.float64]()
    _ = agent.train(env, num_episodes=10)

    var final_alpha = agent.alpha
    print("Final alpha:", final_alpha)
    print("Alpha changed:", abs(final_alpha - initial_alpha) > 1e-6)
    print("Alpha positive:", final_alpha > 0.0)
    print("Alpha reasonable (< 10):", final_alpha < 10.0)

    var pass_count = 0
    if abs(final_alpha - initial_alpha) > 1e-6:
        print("OK: Alpha was tuned")
        pass_count += 1
    else:
        print("FAIL: Alpha did not change")

    if final_alpha > 0.0 and final_alpha < 10.0:
        print("OK: Alpha is in reasonable range")
        pass_count += 1
    else:
        print("FAIL: Alpha out of range")

    if agent.alpha_adam_t > 0:
        print("OK: Adam steps taken (" + String(agent.alpha_adam_t) + ")")
        pass_count += 1
    else:
        print("FAIL: No Adam steps for alpha")

    print("\n" + String(pass_count) + "/3 checks passed")
    print("\n=== Done ===")
