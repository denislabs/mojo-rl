"""Rainbow DQN CNN GPU Training on Atari 2600 Pong — Pixel Observations.

Trains a Rainbow agent on the real Atari 2600 Pong ROM (6502/TIA/RIOT
emulation) using pixel observations (4×84×84 stacked grayscale frames).

This is the Atari counterpart to `rainbow_pong_pixel_training_gpu.mojo`,
which trains on the *native* GPU Pong physics engine. The crucial
difference: the Atari emulator is **CPU-only** (the 6502 opcode dispatch
diverges on the GPU), so there is no `BatchedGpuDiscreteEnv` path here.
Instead we step a single CPU-emulated env and train the CNN Q-network on
the GPU, via `run_offpolicy_discrete_train` (driver row: cpu env / gpu
train / 1 env). One env step + one train step per iteration (replay
ratio 1.0).

The 4×84×84 pixel pipeline — render → max-pool (sprite-flicker) →
grayscale → box-filter resize 160×210→84×84 → 4-frame ring stack — is
already built into `AtariEnv[PongDef, 1]`; no wrapper needed. Frame skip
is fixed at 4 internally for pixel mode.

Rainbow components: C51 + Double + PER + Dueling + Noisy + N-step.

Network (assembled inline — deep_agents2 ships no Rainbow-CNN preset):

  Conv2D[4→32, 8×8, s4] → ReLU →
  Conv2D[32→64, 4×4, s2] → ReLU →
  Conv2D[64→64, 3×3, s1] → ReLU →
  Flatten → LinearReLU[3136→512] →
  NoisyLinear[512 → (1+ACT)·NA] →
  DuelingHeadC51[ACT, NA]          # V[NA] + A[ACT,NA] → per-action atom logits

Note: Atari Pong exposes 6 ALE actions (NOOP, FIRE, RIGHT, LEFT,
RIGHTFIRE, LEFTFIRE) — FIRE serves the ball — vs the native engine's 3.

Requires the Pong ROM at `roms/pong.bin` (symlink to ale_py/roms/).

Run with:
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_atari_pong_pixel_training_gpu.mojo   # compile/smoke
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_atari_pong_pixel_training_gpu.mojo   # training
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
from mojo_rl.deep_agents2.training import run_offpolicy_discrete_train

from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.pong import PongDef
from mojo_rl.envs.atari.flags import OBS_WIDTH, OBS_HEIGHT


# =============================================================================
# Constants
# =============================================================================

# Atari Pong pixel: 4×84×84 = 28224 observation, 6 ALE actions.
comptime FRAMES = 4
comptime OBS_DIM = FRAMES * OBS_WIDTH * OBS_HEIGHT  # 4 * 84 * 84 = 28224
comptime NUM_ACTIONS = PongDef.NUM_ACTIONS  # 6 (NOOP/FIRE/RIGHT/LEFT/R+F/L+F)

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3

# GPU-resident replay → capacity is VRAM-bound (obs + next_obs per slot,
# 2·28224 floats each). Keep modest; raise on large-VRAM cards.
comptime BUFFER_CAPACITY = 12_000
comptime BATCH_SIZE = 32

# Distributional support — must bracket the DISCOUNTED return (≈ ±0.3..±6
# with γ=0.99 + sparse ±1 rewards), NOT the raw ±21 episode score. [-2, 2]
# → atom spacing 0.08. This is the lever that made the native pixel run
# converge; legacy Rainbow's [-21, 21] support never got off the floor.
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)

comptime WARMUP = 20_000
# Single CPU-emulated env: stepping (not training) is the bottleneck.
# ~minutes/hours depending on hardware; lower for faster local loops.
comptime NUM_STEPS = 2_000_000
comptime LR = Scalar[DT](6.25e-5)

# Checkpointing. The CNN q-net + optimizer + epsilon are written to
# CKPT_PATH every CKPT_EVERY env-steps (and once at the end); the replay
# buffer is NOT saved.
comptime CKPT_EVERY = 250_000
comptime CKPT_PATH = "checkpoints/rainbow_atari_pong_pixel.ckpt"

comptime ROM_PATH = "roms/pong.bin"


# Rainbow CNN Q-net: Nature backbone + noisy dueling distributional heads.
# Conv geometry: 84→20→9→7, 64·7·7 = 3136.
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
comptime AtariPongPixel = AtariEnv[PongDef, 1, DT]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN GPU Training on Atari 2600 Pong — Pixel")
    print("=" * 70)
    print()

    # Load ROM once; both env instances share the read-only buffer.
    print("Loading ROM:", ROM_PATH)
    var rom_data = load_rom(ROM_PATH)
    print("ROM loaded:", rom_data.size, "bytes")
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

        var env = AtariPongPixel(rom_data.data.value(), rom_data.size)
        # Separate env instance for deterministic (noise-off) greedy eval.
        var eval_env = AtariPongPixel(rom_data.data.value(), rom_data.size)

        print("Environment: Atari 2600 Pong (CPU emulation, single env, Pixel)")
        print("Agent: Rainbow DQN CNN (deep_agents2 C51, GPU train)")
        print(
            "  Components: C51 + Double + PER + Dueling + Noisy +",
            N_STEP,
            "-step",
        )
        print("  Observation: 4 × 84 × 84 =", OBS_DIM)
        print(
            "  Actions:",
            NUM_ACTIONS,
            "(NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)",
        )
        print("  Network: Nature CNN + Noisy Dueling Distributional heads")
        print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
        print("  N-step:", N_STEP)
        print("  Buffer capacity:", BUFFER_CAPACITY, "(GPU-resident)")
        print("  Batch size:", BATCH_SIZE)
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
            run_name="Rainbow Atari Pong Pixel GPU (deep_agents2)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Rainbow DQN CNN (deep_agents2)")
        logger.set_config("env", "Atari Pong (Pixel)")
        logger.set_config("obs", "4x84x84")
        logger.set_config("lr", String(LR))
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("n_step", String(N_STEP))
        logger.set_config("num_atoms", String(NUM_ATOMS))
        logger.set_config("v_min", String(V_MIN))
        logger.set_config("v_max", String(V_MAX))
        logger.set_config("num_actions", String(NUM_ACTIONS))

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var _ep_returns = run_offpolicy_discrete_train[
                RainbowTrainer, AtariPongPixel, RemoteLogger
            ](
                trainer,
                env,
                NUM_STEPS,
                ctx=ctx,
                print_every=20_000,
                verbose=True,
                logger=UnsafePointer(to=logger),
                diag_every=5_000,
                checkpoint_every=CKPT_EVERY,
                checkpoint_path=String(CKPT_PATH),
                eval_env=UnsafePointer(to=eval_env),
                eval_every=100_000,
                eval_episodes=3,  # each is a full episode → keep small
            )

            var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
            logger.close()

            print("-" * 70)
            print()
            print("=" * 70)
            print("Rainbow Atari CNN GPU Training Complete")
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
