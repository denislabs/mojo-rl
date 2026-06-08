"""Rainbow DQN CNN GPU Training on Pong — Pixel Observations (deep_agents2).

Trains a Rainbow agent on the native Pong environment using pixel
observations (4×84×84 stacked grayscale frames), stepping `N_ENVS`
environments in parallel on the GPU via `BatchedGpuDiscreteEnv` while the
CNN Q-network trains on the same device.

Rainbow components: C51 + Double + PER + Dueling + Noisy + N-step.

deep_agents2 ships no Rainbow-CNN *preset*, so the network is assembled
inline below from nn2 primitives — a Nature-DQN convolutional backbone
followed by the dueling / noisy / distributional heads, wired into the same
`C51Trainer` the clean-obs script uses:

  Conv2D[4→32, 8×8, s4] → ReLU →
  Conv2D[32→64, 4×4, s2] → ReLU →
  Conv2D[64→64, 3×3, s1] → ReLU →
  Flatten → LinearReLU[3136→512] →
  NoisyLinear[512 → (1+ACT)·NA] →
  DuelingHeadC51[ACT, NA]          # V[NA] + A[ACT,NA] → per-action atom logits

Memory note: deep_agents2's prioritized replay is GPU-resident, so unlike
the legacy host-memory buffer the capacity here is bounded by device memory
(each transition stores obs + next_obs = 2·28224 floats). Keep
BUFFER_CAPACITY modest; raise it on large-VRAM cards.

Run with:
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_pong_pixel_training_gpu.mojo   # compile/smoke
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_pong_pixel_training_gpu.mojo   # training
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn2.constants import DT

from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.conv2d import Conv2D
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.primitives.linear_relu import LinearReLU
from mojo_rl.nn2.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn2.primitives.dueling_head_c51 import DuelingHeadC51

from mojo_rl.deep_agents2.c51.trainer import C51Trainer
from mojo_rl.deep_agents2.training.blocks import NStepSampleStep
from mojo_rl.deep_agents2.data.any_per_replay import AnyPerReplay
from mojo_rl.deep_agents2.training import (
    BatchedGpuDiscreteEnv,
    run_offpolicy_discrete_train_gpu_batched,
)
from mojo_rl.envs.arcade_games.pong import PongPixelEnv


# =============================================================================
# Constants
# =============================================================================

# Pong pixel: 4×84×84 = 28224 observation, 3 discrete actions.
comptime OBS_DIM = PongPixelEnv[DType.float64].OBS_DIM  # 28224
comptime NUM_ACTIONS = PongPixelEnv[DType.float64].NUM_ACTIONS  # 3
comptime FRAMES = 4

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
# N_STEP=1 mirrors the converged clean-obs run (the value-config fix below was
# validated at N_STEP=1). Bump to 3 for full Rainbow once a pixel run confirms
# convergence.
comptime N_STEP = 3

# GPU-resident replay → capacity is VRAM-bound (obs + next_obs per slot).
comptime BUFFER_CAPACITY = 12_000
comptime BATCH_SIZE = 32
comptime N_ENVS = 64  # fewer envs — each owns a pixel render/frame-stack workspace

# Distributional support — must bracket the DISCOUNTED return (≈ ±0.3..±6 with
# γ=0.99 + sparse ±1 rewards), NOT the raw ±21 episode score. [-2, 2] → atom
# spacing 0.08 (vs 0.84 at [-21, 21], too coarse to separate the 3 actions).
# This is the lever that made the clean-obs run converge (-19 → +21 perfect
# game); legacy Rainbow's [-21, 21] never got off the floor.
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)

# Dense ball-return shaping (env `HIT_REWARD`): 0.0 = clean sparse ±1 rewards;
# 0.1 = original shaping (distorts the value scale, worse here since FRAME_SKIP
# accumulates it). Disabled to match the converged clean-obs config.
comptime HIT_REWARD = 0.0

# Replay ratio = GRAD_STEPS / N_ENVS = 16/64 = 0.25 (CleanRL train_freq=4).
comptime GRAD_STEPS = 16
comptime WARMUP = 20_000
comptime NUM_STEPS = 5_000_000
comptime LR = Scalar[DT](6.25e-5)

# Checkpointing. The CNN q-net + optimizer + epsilon are written to CKPT_PATH
# every CKPT_EVERY env-steps (and once at the end); the replay buffer is NOT
# saved. `rainbow_pong_pixel_eval_render.mojo` reconstructs the same trainer
# config and `load_state(CKPT_PATH)`s it to play a live game.
comptime CKPT_EVERY = 250_000
comptime CKPT_PATH = "checkpoints/rainbow_pong_pixel.ckpt"


# Rainbow CNN Q-net: Nature backbone + noisy dueling distributional heads.
# Conv geometry matches NatureDQNNet (84→20→9→7, 64·7·7 = 3136).
comptime RainbowCNNNet = Sequential[
    Conv2D[FRAMES, 32, 8, 4, 0, 84, 84],
    ReLU[32 * 20 * 20],
    Conv2D[32, 64, 4, 2, 0, 20, 20],
    ReLU[64 * 9 * 9],
    Conv2D[64, 64, 3, 1, 0, 9, 9],
    ReLU[64 * 7 * 7],
    Flatten[64 * 7 * 7],
    LinearReLU[64 * 7 * 7, HIDDEN],
    NoisyLinear[HIDDEN, (1 + NUM_ACTIONS) * NUM_ATOMS],
    DuelingHeadC51[NUM_ACTIONS, NUM_ATOMS],
]

comptime SAMPLE = NStepSampleStep[
    N_STEP, AnyPerReplay["gpu", OBS_DIM, 1, BUFFER_CAPACITY], BATCH_SIZE
]
comptime RainbowTrainer = C51Trainer[
    "gpu", SAMPLE, RainbowCNNNet, NUM_ATOMS, NUM_ACTIONS, True
]
comptime PongPixelBatched = BatchedGpuDiscreteEnv[
    PongPixelEnv[DT, HIT_REWARD], N_ENVS, OBS_DIM, 1
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN GPU Training on Pong — Pixel (deep_agents2)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var trainer = RainbowTrainer.make(
            ctx=ctx,
            lr=LR,
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            epsilon=Scalar[DT](0.0),  # Noisy nets supply exploration
            learning_starts=WARMUP,
            target_update_freq=500,
            max_grad_norm=Scalar[DT](10.0),
            per_alpha=Scalar[DT](0.5),
            per_beta=Scalar[DT](0.4),
            per_epsilon=Scalar[DT](1e-6),
            nstep=N_STEP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        var env = PongPixelBatched(ctx)
        # Separate env instance for deterministic (noise-off) greedy eval.
        var eval_env = PongPixelBatched(ctx)

        print("Environment: Pong (GPU-batched Pixel,", N_ENVS, "envs)")
        print("Agent: Rainbow DQN CNN (deep_agents2 C51, GPU)")
        print(
            "  Components: C51 + Double + PER + Dueling + Noisy +",
            N_STEP,
            "-step",
        )
        print("  Observation: 4 × 84 × 84 =", OBS_DIM)
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Network: Nature CNN + Noisy Dueling Distributional heads")
        print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
        print("  Hit-reward shaping:", HIT_REWARD)
        print("  N-step:", N_STEP)
        print("  N envs (parallel):", N_ENVS)
        print("  Buffer capacity:", BUFFER_CAPACITY, "(GPU-resident)")
        print("  Batch size:", BATCH_SIZE)
        print("  Grad steps / iter:", GRAD_STEPS, "(replay ratio 0.25)")
        print("  Learning rate:", LR)
        print("  Warmup:", WARMUP)
        print("  Total transitions:", NUM_STEPS)
        print("  Checkpoint:", CKPT_PATH, "(every", CKPT_EVERY, "steps)")
        print()

        # =====================================================================
        # Logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="Rainbow Pong Pixel GPU (deep_agents2)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Rainbow DQN CNN (deep_agents2)")
        logger.set_config("env", "Pong (Pixel)")
        logger.set_config("obs", "4x84x84")
        logger.set_config("lr", String(LR))
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("n_step", String(N_STEP))
        logger.set_config("num_atoms", String(NUM_ATOMS))
        logger.set_config("v_min", String(V_MIN))
        logger.set_config("v_max", String(V_MAX))
        logger.set_config("hit_reward", String(HIT_REWARD))
        logger.set_config("grad_steps", String(GRAD_STEPS))

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var _ep_returns = run_offpolicy_discrete_train_gpu_batched[
                RainbowTrainer, PongPixelBatched, N_ENVS, N_STEP, RemoteLogger
            ](
                ctx,
                trainer,
                env,
                NUM_STEPS,
                rng_seed=UInt64(42),
                updates_per_step=GRAD_STEPS,
                print_every=20_000,
                verbose=True,
                nstep_gamma=Scalar[DT](0.99),
                logger=UnsafePointer(to=logger),
                diag_every=5_000,
                checkpoint_every=CKPT_EVERY,
                checkpoint_path=String(CKPT_PATH),
                eval_env=UnsafePointer(to=eval_env),
                eval_every=100_000,
                eval_episodes=10,
            )

            var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
            logger.close()

            print("-" * 70)
            print()
            print("=" * 70)
            print("Rainbow CNN GPU Training Complete")
            print("=" * 70)
            print("Total transitions:", NUM_STEPS)
            print("Training time:", String(elapsed_s)[byte=:6], "seconds")
            print(
                "Transitions/second:",
                String(Float64(NUM_STEPS) / elapsed_s)[byte=:9],
            )
            print("Final mean return (last 10):", trainer.mean_return())
            print("Episodes completed:", trainer.ep_count())
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
