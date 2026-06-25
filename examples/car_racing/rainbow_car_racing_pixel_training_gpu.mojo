"""Rainbow DQN CNN GPU training on CarRacing — PIXEL observations (multi-body).

The faithful-to-Gymnasium setup: the agent learns from a top-down rendered image
(4x84x84 stacked grayscale frames), so it can actually SEE the road and learn to
follow the track — which the 13-D clean-obs vector cannot convey. Physics is the
Box2D-faithful multi-body model; the env (`CarRacingPixel`) renders a
car-centered, car-aligned view in-kernel and maintains a per-env frame stack.

Rainbow components: C51 + Double + PER + Dueling + Noisy + N-step, Nature-CNN
backbone. Whole agent from the `RainbowCNN` preset; trains via
`agent.train_gpu_batched` over `BatchedGpuDiscreteEnv`.

Run with:
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_pixel_training_gpu.mojo   # compile/smoke
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_pixel_training_gpu.mojo   # training
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT

from mojo_rl.deep_agents.c51.config import RainbowCNN
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.car_racing import CarRacingPixel


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = CarRacingPixel[DType.float64].OBS_DIM  # 4*84*84 = 28224
comptime NUM_ACTIONS = CarRacingPixel[DType.float64].NUM_ACTIONS  # 5
comptime FRAMES = 4

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3

# GPU-resident replay; uint8 obs storage (pixels are exact k/255 → lossless),
# so each slot's pixel payload is 4× smaller than the float ring.
comptime BUFFER_CAPACITY = 48_000
comptime OBS_STORE_DT = DType.uint8
comptime BATCH_SIZE = 32
comptime N_ENVS = 64  # each env owns a pixel render + frame-stack workspace

# Distributional support — sized against the per-step reward, not just the
# return range (the lesson from the clean-obs run). CarRacing's tile reward is
# 1000/N ≈ +3.4; with 51 atoms, [-30, 30] gives spacing 1.2 so a tile reward is
# ~2.8 atoms (visible), while the -100 off-field penalty clamps to -30. The old
# ±100 made spacing 4.0 > a tile reward → tiles invisible → the agent learned to
# park. If it learns to drive but plateaus (good policies' discounted Q ~+50),
# widen to [-60, 60] with NUM_ATOMS=121.
comptime V_MIN = Scalar[DT](-30.0)
comptime V_MAX = Scalar[DT](30.0)

# Replay ratio = GRAD_STEPS / N_ENVS = 16/64 = 0.25 (CleanRL train_freq=4).
comptime GRAD_STEPS = 16
comptime WARMUP = 20_000
comptime NUM_STEPS = 10_000_000
comptime LR = Scalar[DT](6.25e-5)

comptime CKPT_EVERY = 250_000
comptime CKPT_PATH = "checkpoints/rainbow_car_racing_pixel.ckpt"

comptime CarRacingPixelBatched = BatchedGpuDiscreteEnv[
    CarRacingPixel[DT], N_ENVS, OBS_DIM, 1
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN GPU Training on CarRacing — Pixel (multi-body)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = RainbowCNN[
            "gpu", NUM_ACTIONS, BATCH_SIZE, BUFFER_CAPACITY,
            FRAMES, NUM_ATOMS, HIDDEN, N_STEP, OBS_STORE_DT,
        ](
            ctx=ctx,
            lr=LR,
            learning_starts=WARMUP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        var env = CarRacingPixelBatched(ctx)
        var eval_env = CarRacingPixelBatched(ctx)

        print("Environment: CarRacingPixel (GPU-batched,", N_ENVS, "envs)")
        print("Agent: Rainbow DQN CNN (deep_agents C51, GPU)")
        print("  Observation:", OBS_DIM, "= 4x84x84 grayscale frame stack")
        print("  Actions:", NUM_ACTIONS, "(noop, left, right, gas, brake)")
        print("  Hidden:", HIDDEN, " Atoms:", NUM_ATOMS, " support [", V_MIN, ",", V_MAX, "]")
        print("  N-step:", N_STEP, " N envs:", N_ENVS, " Batch:", BATCH_SIZE)
        print("  Buffer:", BUFFER_CAPACITY, "(uint8 obs)  LR:", LR, " Warmup:", WARMUP)
        print("  Checkpoint:", CKPT_PATH, "(every", CKPT_EVERY, "steps)")
        print()

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")
        var logger = RemoteLogger(
            server_url=url,
            run_name="Rainbow CarRacing Pixel GPU (multi-body)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Rainbow DQN CNN (deep_agents)")
        logger.set_config("env", "CarRacingPixel")
        logger.set_config("obs", "4x84x84")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("lr", String(LR))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("n_step", String(N_STEP))
        logger.set_config("num_atoms", String(NUM_ATOMS))

        print("Starting GPU training...")
        print("-" * 70)
        var start_time = perf_counter_ns()

        try:
            var _ep_returns = agent.train_gpu_batched[
                CarRacingPixelBatched, N_ENVS, N_STEP, RemoteLogger
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
            print("Rainbow Pixel GPU Training Complete")
            print("Training time:", String(elapsed_s)[byte=:6], "seconds")
            print("Final mean return (last 10):", agent.mean_return())
            print("Episodes completed:", agent.ep_count())

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)

    print(">>> main() completed normally <<<")
