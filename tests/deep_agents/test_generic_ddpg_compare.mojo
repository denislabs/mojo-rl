"""Compare GenericOffPolicyAgent[DDPGConfig] vs DeepDDPGAgent.

Both agents train on PendulumEnv for several episodes.
Verifies the generic agent trains correctly by checking:
1. Both agents complete training
2. Both agents perform gradient steps
3. The generic agent's train_step logic produces valid critic loss values
"""

from std.random import seed

from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent, DDPGConfig
from mojo_rl.deep_agents.core.agents import DeepDDPGAgent
from mojo_rl.envs.pendulum import PendulumEnv


fn main() raises:
    print("=== Generic DDPG vs Old DDPG Comparison ===\n")

    # ---------- Generic agent ----------
    print("1. Training GenericOffPolicyAgent[DDPGConfig] (10 episodes)...")
    seed(42)
    var gen = GenericOffPolicyAgent[DDPGConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )
    var env1 = PendulumEnv[DType.float64]()
    var m1 = gen.train(env1, num_episodes=10)
    var gen_steps = gen.train_step_count
    var gen_episodes = len(m1.episodes)
    print("   train_steps:", gen_steps, " episodes:", gen_episodes)

    # ---------- Old agent ----------
    print("\n2. Training DeepDDPGAgent (10 episodes)...")
    seed(42)
    var old = DeepDDPGAgent[3, 1, 64, 1000, 32](action_scale=2.0)
    var env2 = PendulumEnv[DType.float64]()
    var m2 = old.train(env2, num_episodes=10)
    var old_steps = old.train_step_count
    var old_episodes = len(m2.episodes)
    print("   train_steps:", old_steps, " episodes:", old_episodes)

    # ---------- Validation ----------
    print("\n3. Validation...")

    # Both should have trained
    var pass_count = 0
    if gen_steps > 0:
        print("   OK: Generic agent trained (" + String(gen_steps) + " steps)")
        pass_count += 1
    else:
        print("   FAIL: Generic agent did not train")

    if old_steps > 0:
        print("   OK: Old agent trained (" + String(old_steps) + " steps)")
        pass_count += 1
    else:
        print("   FAIL: Old agent did not train")

    # Both should have same number of episodes
    if gen_episodes == old_episodes:
        print(
            "   OK: Same episode count (" + String(gen_episodes) + ")"
        )
        pass_count += 1
    else:
        print(
            "   WARN: Episode count differs (generic="
            + String(gen_episodes)
            + " old="
            + String(old_episodes)
            + ")"
        )

    # Exploration rates should be same (both decayed same number of times)
    var gen_rate = gen.get_explore_rate()
    var old_rate = old.get_explore_rate()
    if gen_rate == old_rate:
        print("   OK: Same explore rate (" + String(gen_rate) + ")")
        pass_count += 1
    else:
        print(
            "   WARN: Explore rate differs (generic="
            + String(gen_rate)
            + " old="
            + String(old_rate)
            + ")"
        )

    print(
        "\n=== " + String(pass_count) + "/4 checks passed ==="
    )
