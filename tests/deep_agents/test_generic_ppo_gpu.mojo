"""Test: GenericOnPolicyAgent GPU training on CartPole (PPO + A2C).

Tests both PPO and A2C on GPU using the composable strategy infrastructure:
  - PPO: ClippedSurrogate + MultiEpochMinibatch
  - A2C: VanillaPG + SinglePass

Run with:
    pixi run -e apple mojo run -I . tests/deep_agents/test_generic_ppo_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/test_generic_ppo_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import (
    GenericOnPolicyAgent,
    PPOConfig,
    A2CConfig,
)
from mojo_rl.envs import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN_DIM = 64
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 256
comptime GPU_MINIBATCH = 4096

comptime NUM_UPDATES = 200


def main() raises:
    seed(42)
    print("=" * 70)
    print("Generic On-Policy Agent GPU Test on CartPole (PPO + A2C)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # =================================================================
        # Test 1: PPO (GPU)
        # =================================================================

        print(
            "1. GenericOnPolicyAgent[PPOConfig] GPU ("
            + String(NUM_UPDATES)
            + " updates)..."
        )
        print(
            "   Config: ClippedSurrogate + MultiEpochMinibatch"
            + " | n_envs="
            + String(N_ENVS)
            + " | rollout="
            + String(ROLLOUT_LEN)
            + " | mb="
            + String(GPU_MINIBATCH)
        )

        seed(42)
        var ppo = GenericOnPolicyAgent[
            PPOConfig[OBS_DIM, NUM_ACTIONS, HIDDEN_DIM, ROLLOUT_LEN],
            N_ENVS,
            GPU_MINIBATCH,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=4,
            minibatch_size=GPU_MINIBATCH,
            normalize_advantages=True,
            target_kl=0.02,
            max_grad_norm=0.5,
        )

        var t0 = perf_counter_ns()
        var m1 = ppo.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_updates=NUM_UPDATES,
            verbose=True,
            print_every=50,
        )
        var t1 = perf_counter_ns()
        var elapsed = Float64(t1 - t0) / 1e9

        print(
            "   updates: "
            + String(ppo.train_step_count)
            + "  time: "
            + String(elapsed)[byte=:5]
            + "s"
        )
        print("   last-20 avg reward: " + String(m1.mean_reward_last_n(20))[byte=:7])

        if ppo.train_step_count > 0:
            print("   OK: Generic PPO GPU trained")
        else:
            print("   FAIL: Generic PPO GPU did not train")

        # =================================================================
        # Test 2: A2C (GPU)
        # =================================================================

        print()
        print(
            "2. GenericOnPolicyAgent[A2CConfig] GPU ("
            + String(NUM_UPDATES)
            + " updates)..."
        )
        print(
            "   Config: VanillaPG + SinglePass"
            + " | n_envs="
            + String(N_ENVS)
            + " | rollout="
            + String(ROLLOUT_LEN)
        )

        seed(42)
        var a2c = GenericOnPolicyAgent[
            A2CConfig[OBS_DIM, NUM_ACTIONS, 128, ROLLOUT_LEN],
            N_ENVS,
            GPU_MINIBATCH,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            normalize_advantages=True,
            max_grad_norm=0.5,
        )

        t0 = perf_counter_ns()
        var m2 = a2c.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_updates=NUM_UPDATES,
            verbose=True,
            print_every=50,
        )
        t1 = perf_counter_ns()
        elapsed = Float64(t1 - t0) / 1e9

        print(
            "   updates: "
            + String(a2c.train_step_count)
            + "  time: "
            + String(elapsed)[byte=:5]
            + "s"
        )
        print("   last-20 avg reward: " + String(m2.mean_reward_last_n(20))[byte=:7])

        if a2c.train_step_count > 0:
            print("   OK: Generic A2C GPU trained")
        else:
            print("   FAIL: Generic A2C GPU did not train")

        # =================================================================
        # Summary
        # =================================================================

        print()
        print("=" * 70)
        print("Summary:")
        print(
            "  PPO : "
            + String(m1.mean_reward_last_n(20))[byte=:7]
            + " avg (last 20) in "
            + String(ppo.train_step_count)
            + " updates"
        )
        print(
            "  A2C : "
            + String(m2.mean_reward_last_n(20))[byte=:7]
            + " avg (last 20) in "
            + String(a2c.train_step_count)
            + " updates"
        )
        print("=" * 70)
