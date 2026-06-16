"""Rainbow DQN GPU Training on Pong (deep_agents, GPU-batched envs).

Trains a Rainbow agent (C51 + Double + PER + Dueling + Noisy + N-step) on
the native Pong environment, stepping `N_ENVS` environments in parallel on
the GPU via `BatchedGpuDiscreteEnv` while the Q-network trains on the same
device — the discrete sibling of the SAC/TD3 GPU-batched path.

Pong has 3 discrete actions (NOOP, UP, DOWN) and 6D clean observations
(ball_xy, ball_vxy, paddle_y, cpu_paddle_y — all normalized).

This is the *new* deep_agents Rainbow (`mojo_rl.deep_agents.c51`), NOT the
legacy `deep_agents` agent. The whole agent comes from the `Rainbow` preset
(dueling/noisy distributional net over an N-step-over-PER sample block) and
trains through `agent.train_gpu_batched`, the facade over the GPU-batched
discrete driver. Pong-tuned overrides below: lr 6.25e-5, warmup 20k, and
the V_MIN/V_MAX ±2 support — the lever that made this run converge.

Run with:
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_pong_training_gpu.mojo   # Apple Silicon (compile/smoke)
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_pong_training_gpu.mojo   # NVIDIA GPU (training)
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT

from mojo_rl.deep_agents.c51.config import Rainbow
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.arcade_games.pong import PongEnv


# =============================================================================
# Constants
# =============================================================================

# Pong: 6D clean observation, 3 discrete actions.
comptime OBS_DIM = PongEnv[DType.float64].OBS_DIM  # 6
comptime NUM_ACTIONS = PongEnv[DType.float64].NUM_ACTIONS  # 3

# Rainbow architecture / replay hyperparameters.
comptime HIDDEN_DIM = 128
comptime NUM_ATOMS = 51
# Full Rainbow N-step. N_STEP=1 is CONFIRMED converged at scale (NVIDIA, 256
# envs: eval −18 → +21, loss off ln(51), TD targets propagating) after the
# obs-corruption fix (extract_obs_kernel_gpu normalization + pixel selective-
# reset). N_STEP>1 routes the batched driver through `record_batch_gpu_nstep`
# → the device `GPUNStepBuffer` (per-env n-step reward accumulation + compressed
# transitions), a SEPARATE store path not exercised at N_STEP=1. This run tests
# it: if N_STEP=3 fails to converge while N_STEP=1 did, the GPUNStepBuffer path
# is the suspect (obs is already validated correct). Drop back to 1 to isolate.
comptime N_STEP = 3
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 64
comptime N_ENVS = 256  # parallel GPU environments

# Distributional support. C51's [v_min, v_max] must bracket the achievable
# *discounted* return, NOT the raw episode score. With γ=0.99 + sparse rewards
# the discounted Q lives in roughly ±0.3..±6, so the old [-21, 21] (atom
# spacing 0.84) wasted nearly all resolution on unreachable values and the
# argmax couldn't separate the 3 actions. Narrowed to [-2, 2] (spacing 0.08).
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)

# Dense ball-return shaping reward (env `HIT_REWARD`). 0.0 = clean sparse ±1
# rewards on points only; 0.1 = original shaping (pushes Q positive while the
# agent loses, distorting the value scale). Disabled for this experiment.
comptime HIT_REWARD = 0.0

# Replay ratio. Each iteration collects N_ENVS transitions and performs
# GRAD_STEPS gradient updates → ratio = GRAD_STEPS / N_ENVS. 64/256 = 0.25,
# matching CleanRL's Atari train_frequency=4.
comptime GRAD_STEPS = 64

# Warmup (env-steps of uniform-random action before learning starts).
comptime WARMUP = 20_000

# Training duration (total env transitions, counting all N_ENVS per iter).
comptime NUM_STEPS = 5_000_000

comptime LR = Scalar[DT](6.25e-5)

# Checkpointing. The trainer's q-net + optimizer + epsilon are written to
# CKPT_PATH every CKPT_EVERY env-steps (and once more at the end of training);
# the replay buffer is NOT saved. The render-eval script
# `rainbow_pong_eval_render.mojo` reconstructs the same trainer config and
# `load_state(CKPT_PATH)`s it to play a live game.
comptime CKPT_EVERY = 250_000
comptime CKPT_PATH = "checkpoints/rainbow_pong.ckpt"

comptime PongBatched = BatchedGpuDiscreteEnv[
    PongEnv[DT, HIT_REWARD], N_ENVS, OBS_DIM, 1
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN GPU Training on Pong (deep_agents, GPU-batched)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # Whole agent from the preset — Rainbow == C51 with DOUBLE=True over
        # a (PER + N-step) sample block and a dueling/noisy distributional
        # net. Config-tuned scalars (ε=0 noisy, PER α=0.5/β=0.4,
        # nstep=N_STEP) apply; lr / warmup / value support are Pong-tuned.
        var agent = Rainbow[
            "gpu", OBS_DIM, NUM_ACTIONS, BATCH_SIZE, BUFFER_CAPACITY,
            NUM_ATOMS, HIDDEN_DIM, N_STEP,
        ](
            ctx=ctx,
            lr=LR,
            learning_starts=WARMUP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        var env = PongBatched(ctx)
        # Separate env instance for deterministic (noise-off) greedy eval —
        # never touches the training replay / episode tracker.
        var eval_env = PongBatched(ctx)

        print("Environment: Pong (GPU-batched,", N_ENVS, "envs)")
        print("Agent: Rainbow DQN (deep_agents C51, GPU)")
        print(
            "  Components: C51 + Double + PER + Dueling + Noisy +",
            N_STEP,
            "-step",
        )
        print("  Observation dim:", OBS_DIM)
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Hidden dim:", HIDDEN_DIM)
        print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
        print("  N-step:", N_STEP)
        print("  N envs (parallel):", N_ENVS)
        print("  Buffer capacity:", BUFFER_CAPACITY)
        print("  Batch size:", BATCH_SIZE)
        print("  Grad steps / iter:", GRAD_STEPS, "(replay ratio 0.25)")
        print("  Learning rate:", LR)
        print("  Warmup:", WARMUP)
        print("  Total transitions:", NUM_STEPS)
        print("  Checkpoint:", CKPT_PATH, "(every", CKPT_EVERY, "steps)")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-21 (CPU wins almost every point)")
        print("  - Good policy:   > 0 (beating CPU)")
        print()

        # =====================================================================
        # Logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="Rainbow Pong GPU (deep_agents)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Rainbow DQN (deep_agents)")
        logger.set_config("env", "Pong")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("lr", String(LR))
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("n_step", String(N_STEP))
        logger.set_config("num_atoms", String(NUM_ATOMS))
        logger.set_config("grad_steps", String(GRAD_STEPS))

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var _ep_returns = agent.train_gpu_batched[
                PongBatched, N_ENVS, N_STEP, RemoteLogger
            ](
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
                eval_episodes=20,
            )

            var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
            logger.close()

            print("-" * 70)
            print()
            print("=" * 70)
            print("Rainbow GPU Training Complete")
            print("=" * 70)
            print("Total transitions:", NUM_STEPS)
            print("Training time:", String(elapsed_s)[byte=:6], "seconds")
            print(
                "Transitions/second:",
                String(Float64(NUM_STEPS) / elapsed_s)[byte=:9],
            )
            print("Final mean return (last 10):", agent.mean_return())
            print("Episodes completed:", agent.ep_count())
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
