"""Test DQN Agent GPU Training on CartPole.

This tests the GPU implementation of DQN using:
- Network wrapper GPU methods (forward_gpu, backward_gpu, update_gpu)
- CPU environment interaction + GPU batch training
- Hyperparameters aligned with CleanRL's dqn.py reference

CleanRL reference: references/RL-Algorithms/cleanrl-master/cleanrl/dqn.py
  - Network: Linear(4,120) → ReLU → Linear(120,84) → ReLU → Linear(84,2)
  - tau=1.0 (hard copy), target_network_frequency=500 env steps
  - train_frequency=10 (1 grad step per 10 env steps)
  - buffer_size=10000, batch_size=128, lr=2.5e-4
  - exploration: linear 1.0 → 0.05 over 50% of training
  - MSE loss, no Double DQN

Run with:
    pixi run -e apple mojo run -I . test_dqn_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . test_dqn_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.deep_agents.core.generic import DQNAgent
from mojo_rl.envs import CartPoleEnv
from mojo_rl.core.logger import RemoteLogger


# =============================================================================
# Constants — aligned with CleanRL dqn.py defaults
# =============================================================================

comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN_DIM = 120  # CleanRL: first hidden layer
comptime HIDDEN_DIM2 = 84  # CleanRL: second hidden layer
comptime BUFFER_CAPACITY = 10_000  # CleanRL: buffer_size=10000
comptime BATCH_SIZE = 128  # CleanRL: batch_size=128
comptime N_ENVS = 256  # Parallel environments for GPU collection

comptime NUM_STEPS = 500_000  # CleanRL: total_timesteps=500000
comptime MAX_STEPS = 500
comptime WARMUP_STEPS = 10_000  # CleanRL: learning_starts=10000
comptime SYNC_EVERY = 10_000  # Sync GPU params to CPU every N transitions

# CleanRL: train_frequency=10 → 1 grad step per 10 env steps
# With N_ENVS=256: 256/10 ≈ 26 gradient steps per collection
comptime GRADIENT_STEPS = 26

# CleanRL: target_network_frequency=500, train_frequency=10
# → 500/10 = 50 gradient steps between target updates
comptime TARGET_UPDATE_FREQ = 50


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DQN Agent GPU Test on CartPole (CleanRL-aligned)")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context, environment and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var env = CartPoleEnv[DType.float32]()
        var agent = DQNAgent[
            OBS_DIM,
            NUM_ACTIONS,
            HIDDEN_DIM,
            HIDDEN_DIM2,
            BUFFER_CAPACITY,
            BATCH_SIZE,
            N_ENVS,
            lr=2.5e-4,
            L=RemoteLogger,
        ](
            gamma=0.99,
            tau=1.0,  # CleanRL: hard copy
            epsilon_min=0.05,
            target_update_freq=TARGET_UPDATE_FREQ,
        )

        print("Environment: CartPole")
        print("Agent: DQN (GPU, CleanRL-aligned)")
        print(
            "  Network: "
            + String(HIDDEN_DIM)
            + " → "
            + String(HIDDEN_DIM2)
            + " → "
            + String(NUM_ACTIONS)
        )
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Gradient steps per collection: " + String(GRADIENT_STEPS))
        print(
            "  Target update: hard copy every "
            + String(TARGET_UPDATE_FREQ)
            + " grad steps"
        )
        print("  Sync every: " + String(SYNC_EVERY) + " transitions")
        print()

        # =====================================================================
        # Setup logger — posts to local RL Monitor worker
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="DQN CartPole GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "DQN")
        logger.set_config("env", "CartPole")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("hidden_dim2", String(HIDDEN_DIM2))
        logger.set_config("lr", "2.5e-4")
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        var metrics = agent.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            gradient_steps=GRADIENT_STEPS,
            sync_every=SYNC_EVERY,
            verbose=True,
            print_every=10_000,
            environment_name="CartPole (GPU)",
            logger=UnsafePointer(to=logger),
            diag_every=50,
        )

        var end_time = perf_counter_ns()
        var elapsed_s = Float64(end_time - start_time) / 1e9

        logger.close()

        print("-" * 70)
        print()

        # =====================================================================
        # Summary
        # =====================================================================

        print("=" * 70)
        print("GPU Training Complete")
        print("=" * 70)
        print()
        print("Total steps: " + String(NUM_STEPS))
        print("Total train steps: " + String(agent.get_train_steps()))
        print("Training time: " + String(elapsed_s)[:6] + " seconds")
        print()

        # Print metrics summary
        print(
            "Final average reward (last 20 episodes): "
            + String(metrics.mean_reward_last_n(20))[:7]
        )
        print("Best episode reward: " + String(metrics.max_reward())[:7])
        print()

        # =====================================================================
        # Evaluation (greedy policy, on CPU with GPU-trained params)
        # =====================================================================

        print("Evaluating greedy policy (10 episodes)...")
        var eval_avg = agent.evaluate(
            env, num_episodes=10, max_steps_per_episode=MAX_STEPS, greedy=True
        )
        print("Evaluation average: " + String(eval_avg)[:7])

        print()
        print("Evaluating with current epsilon (10 episodes)...")
        var eval_eps_avg = agent.evaluate(
            env, num_episodes=10, max_steps_per_episode=MAX_STEPS, greedy=False
        )
        print(
            "Evaluation average (epsilon="
            + String(agent.get_epsilon())[:5]
            + "): "
            + String(eval_eps_avg)[:7]
        )

        print()
        print("=" * 70)
