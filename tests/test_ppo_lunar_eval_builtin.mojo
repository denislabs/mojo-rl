"""Test using the agent's built-in evaluate_gpu method.

This tests whether the built-in GPU evaluation produces the same results
as training, to help diagnose the CPU/GPU forward pass divergence.

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_eval_builtin.mojo
"""

from random import seed

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.lunar_lander import LunarLanderV2, LLConstants
from deep_rl import dtype as gpu_dtype


# =============================================================================
# Constants (must match training)
# =============================================================================

comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous - Built-in evaluate_gpu Test")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # Create agent with same architecture as training
        var agent = DeepPPOContinuousAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=GPU_MINIBATCH_SIZE,
            clip_value=True,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            actor_lr=0.0003,
            critic_lr=0.001,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.02,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_lunar_continuous_gpu.ckpt",
        )

        print("Loading checkpoint...")
        agent.load_checkpoint("ppo_lunar_continuous_gpu.ckpt")
        print("Checkpoint loaded successfully!")
        print()

        # Print first few params to verify
        print("First 10 actor params:", end=" ")
        for i in range(min(10, len(agent.actor.params))):
            print(agent.actor.params[i], end=" ")
        print()
        print("Actor param count:", len(agent.actor.params))
        print()

        # =================================================================
        # Test 1: Built-in evaluate_gpu (uses GPU forward pass like training)
        # =================================================================
        print("-" * 70)
        print("Test 1: Built-in evaluate_gpu (GPU forward pass)")
        print("-" * 70)

        var avg_reward_gpu = agent.evaluate_gpu[LunarLanderV2[gpu_dtype]](
            ctx,
            num_episodes=100,
            max_steps=1000,
            verbose=True,
        )

        print()
        print("Built-in evaluate_gpu result:", avg_reward_gpu)
        print()

        # =================================================================
        # Test 2: CPU evaluate for comparison
        # =================================================================
        print("-" * 70)
        print("Test 2: CPU evaluate (CPU forward pass via select_action)")
        print("-" * 70)

        var env = LunarLanderV2[gpu_dtype]()
        var avg_reward_cpu = agent.evaluate[LunarLanderV2[gpu_dtype]](
            env,
            num_episodes=10,  # Fewer episodes since CPU is slower
            max_steps=1000,
            verbose=True,
        )

        print()
        print("CPU evaluate result:", avg_reward_cpu)
        print()

        # =================================================================
        # Summary
        # =================================================================
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("Built-in evaluate_gpu (100 eps):", avg_reward_gpu)
        print("CPU evaluate (10 eps):", avg_reward_cpu)
        print()

        if avg_reward_gpu > 100 and avg_reward_cpu < 0:
            print("DIAGNOSIS: GPU forward works, CPU forward fails")
            print("  -> There's likely a mismatch between CPU and GPU forward pass")
        elif avg_reward_gpu < 0 and avg_reward_cpu < 0:
            print("DIAGNOSIS: Both fail - the trained mean action is bad")
            print("  -> Training relies on exploration noise for success")
        elif avg_reward_gpu > 100 and avg_reward_cpu > 100:
            print("DIAGNOSIS: Both work - issue was in custom test loop")
        else:
            print("DIAGNOSIS: Unexpected results - need more investigation")

    print()
    print(">>> Test completed <<<")
