"""Test: GenericDQNAgent GPU training on CartPole.

Tests all three DQN variants (standard, Double, Dueling) on GPU using
the composable strategy infrastructure.

Run with:
    pixi run -e apple mojo run -I . tests/deep_agents/test_generic_dqn_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/test_generic_dqn_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import (
    GenericDQNAgent,
    DQNConfig,
    DoubleDQNConfig,
    DuelingDQNConfig,
)
from mojo_rl.envs import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN_DIM = 120
comptime HIDDEN_DIM2 = 84
comptime BUFFER_CAPACITY = 10_000
comptime BATCH_SIZE = 128
comptime N_ENVS = 256

comptime NUM_STEPS = 500_000  # CleanRL default: total_timesteps=500000
comptime WARMUP_STEPS = 2_000
comptime GRADIENT_STEPS = 26
comptime SYNC_EVERY = 10_000
comptime TARGET_UPDATE_FREQ = 50


def main() raises:
    seed(42)
    print("=" * 70)
    print("Generic DQN Agent GPU Test on CartPole")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # =================================================================
        # Test 1: Standard DQN (GPU)
        # =================================================================

        print(
            "1. GenericDQNAgent[DQNConfig] GPU ("
            + String(NUM_STEPS)
            + " steps)..."
        )

        seed(42)
        var dqn = GenericDQNAgent[
            DQNConfig[
                OBS_DIM,
                NUM_ACTIONS,
                HIDDEN_DIM,
                HIDDEN_DIM2,
                BUFFER_CAPACITY,
                BATCH_SIZE,
            ],
            N_ENVS,
        ](
            gamma=0.99,
            tau=1.0,
            target_update_freq=TARGET_UPDATE_FREQ,
        )

        var t0 = perf_counter_ns()
        var m1 = dqn.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            gradient_steps=GRADIENT_STEPS,
            sync_every=SYNC_EVERY,
            verbose=True,
            print_every=20_000,
        )
        var t1 = perf_counter_ns()
        var elapsed = Float64(t1 - t0) / 1e9

        print(
            "   steps: "
            + String(dqn.train_step_count)
            + "  episodes: "
            + String(len(m1.episodes))
            + "  time: "
            + String(elapsed)[:5]
            + "s"
        )
        print("   last-20 avg reward: " + String(m1.mean_reward_last_n(20))[:7])

        if dqn.train_step_count > 0:
            print("   OK: Standard DQN GPU trained")
        else:
            print("   FAIL: Standard DQN GPU did not train")

        # =================================================================
        # Test 2: Double DQN (GPU)
        # =================================================================

        print()
        print(
            "2. GenericDQNAgent[DoubleDQNConfig] GPU ("
            + String(NUM_STEPS)
            + " steps)..."
        )

        seed(42)
        var ddqn = GenericDQNAgent[
            DoubleDQNConfig[
                OBS_DIM,
                NUM_ACTIONS,
                HIDDEN_DIM,
                HIDDEN_DIM2,
                BUFFER_CAPACITY,
                BATCH_SIZE,
            ],
            N_ENVS,
        ](
            gamma=0.99,
            tau=1.0,
            target_update_freq=TARGET_UPDATE_FREQ,
        )

        t0 = perf_counter_ns()
        var m2 = ddqn.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            gradient_steps=GRADIENT_STEPS,
            sync_every=SYNC_EVERY,
            verbose=True,
            print_every=20_000,
        )
        t1 = perf_counter_ns()
        elapsed = Float64(t1 - t0) / 1e9

        print(
            "   steps: "
            + String(ddqn.train_step_count)
            + "  episodes: "
            + String(len(m2.episodes))
            + "  time: "
            + String(elapsed)[:5]
            + "s"
        )
        print("   last-20 avg reward: " + String(m2.mean_reward_last_n(20))[:7])

        if ddqn.train_step_count > 0:
            print("   OK: Double DQN GPU trained")
        else:
            print("   FAIL: Double DQN GPU did not train")

        # =================================================================
        # Test 3: Dueling DQN (GPU)
        # =================================================================

        print()
        print(
            "3. GenericDQNAgent[DuelingDQNConfig] GPU ("
            + String(NUM_STEPS)
            + " steps)..."
        )

        seed(42)
        var dueling = GenericDQNAgent[
            DuelingDQNConfig[
                OBS_DIM,
                NUM_ACTIONS,
                HIDDEN_DIM,
                HIDDEN_DIM2,
                BUFFER_CAPACITY,
                BATCH_SIZE,
            ],
            N_ENVS,
        ](
            gamma=0.99,
            tau=1.0,
            target_update_freq=TARGET_UPDATE_FREQ,
        )

        t0 = perf_counter_ns()
        var m3 = dueling.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            gradient_steps=GRADIENT_STEPS,
            sync_every=SYNC_EVERY,
            verbose=True,
            print_every=20_000,
        )
        t1 = perf_counter_ns()
        elapsed = Float64(t1 - t0) / 1e9

        print(
            "   steps: "
            + String(dueling.train_step_count)
            + "  episodes: "
            + String(len(m3.episodes))
            + "  time: "
            + String(elapsed)[:5]
            + "s"
        )
        print("   last-20 avg reward: " + String(m3.mean_reward_last_n(20))[:7])

        if dueling.train_step_count > 0:
            print("   OK: Dueling DQN GPU trained")
        else:
            print("   FAIL: Dueling DQN GPU did not train")

        # =================================================================
        # Summary
        # =================================================================

        print()
        print("=" * 70)
        print("Summary:")
        print(
            "  DQN      : "
            + String(m1.mean_reward_last_n(20))[:7]
            + " avg (last 20)"
        )
        print(
            "  Double   : "
            + String(m2.mean_reward_last_n(20))[:7]
            + " avg (last 20)"
        )
        print(
            "  Dueling  : "
            + String(m3.mean_reward_last_n(20))[:7]
            + " avg (last 20)"
        )
        print("=" * 70)
