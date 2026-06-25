"""Rainbow DQN CNN on CarRacing — HYBRID (CPU env stepped on host + GPU train).

The Atari-Pong "Stage-1" pattern: step N parallel CarRacingMB instances on CPU
cores via `BatchedCpuDiscreteEnv`, while the Rainbow CNN Q-network trains on the
GPU. Because the env IS the CPU `CarRacingMB` (faithful Gymnasium track + the
exact multi-body physics + the CPU pixel rasterizer), this:
  - transfers to CPU eval by construction (you train on the eval env), and
  - cannot be cheated — the faithful track is a proper closed loop (unlike the
    GPU embedded `_gen_track`, which is half-width and self-intersecting, which
    let the agent "cut to the end").

Observation: 4x84x84 grayscale (CarRacingMB[DT, PIXEL_OBS=True]). Slower than the
pure-GPU env (the render runs on CPU), so N_ENVS is smaller.

Run with:
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_pixel_hybrid_training.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_pixel_hybrid_training.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT

from mojo_rl.deep_agents.c51.config import RainbowCNN
from mojo_rl.deep_agents.training import BatchedCpuDiscreteEnv
from mojo_rl.envs.car_racing import CarRacingMB


# =============================================================================
# Constants
# =============================================================================

comptime CarRacingPx = CarRacingMB[DT, True]  # PIXEL_OBS=True
comptime OBS_DIM = CarRacingPx.EFF_OBS_DIM  # 4*84*84 = 28224
comptime NUM_ACTIONS = CarRacingPx.NUM_ACTIONS  # 5
comptime FRAMES = 4

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3

comptime BUFFER_CAPACITY = 48_000
comptime OBS_STORE_DT = DType.uint8
comptime BATCH_SIZE = 32
comptime N_ENVS = 8  # CPU-stepped (render on host) → fewer envs than the GPU env

# Same value-support fix as the GPU runs (tile reward ~+3.4 must exceed the atom
# spacing; ±30 / 51 atoms → spacing 1.2). Widen later if it plateaus.
comptime V_MIN = Scalar[DT](-30.0)
comptime V_MAX = Scalar[DT](30.0)

# Replay ratio = UPDATES_PER_STEP / N_ENVS.
comptime UPDATES_PER_STEP = 2
comptime WARMUP = 20_000
comptime NUM_STEPS = 10_000_000
comptime LR = Scalar[DT](6.25e-5)

comptime CKPT_EVERY = 250_000
comptime CKPT_PATH = "checkpoints/rainbow_car_racing_pixel_hybrid.ckpt"

comptime BatchedCarRacing = BatchedCpuDiscreteEnv[CarRacingPx, N_ENVS, OBS_DIM]


def _make_envs() -> List[CarRacingPx]:
    var envs = List[CarRacingPx]()
    for _ in range(N_ENVS):
        envs.append(CarRacingMB[DT, True]())
    return envs^


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow CNN on CarRacing — HYBRID (CPU env + GPU train)")
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

        var env = BatchedCarRacing(_make_envs())
        var eval_env = BatchedCarRacing(_make_envs())

        print("Environment: CarRacingMB pixel (CPU-stepped,", N_ENVS, "envs)")
        print("Agent: Rainbow DQN CNN (GPU)")
        print("  Observation:", OBS_DIM, "= 4x84x84  Actions:", NUM_ACTIONS)
        print("  Hidden:", HIDDEN, " Atoms:", NUM_ATOMS, " support [", V_MIN, ",", V_MAX, "]")
        print("  N-step:", N_STEP, " Buffer:", BUFFER_CAPACITY, "(uint8)")
        print("  Updates/step:", UPDATES_PER_STEP, " LR:", LR, " Warmup:", WARMUP)
        print("  Checkpoint:", CKPT_PATH)
        print()

        var env_vars = load_dotenv()
        var logger = RemoteLogger(
            server_url=env_vars.get("RL_MONITOR_URL", ""),
            run_name="Rainbow CarRacing Pixel HYBRID (cpu env + gpu train)",
            buffer_size=64,
            api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
        )
        logger.set_config("agent", "Rainbow DQN CNN (hybrid)")
        logger.set_config("env", "CarRacingMB pixel (CPU)")
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("n_step", String(N_STEP))

        print("Starting hybrid training...")
        print("-" * 70)
        var start_time = perf_counter_ns()

        try:
            var _ep_returns = agent.train_cpu_batched[
                BatchedCarRacing, N_ENVS, N_STEP, RemoteLogger
            ](
                env,
                NUM_STEPS,
                rng_seed=UInt64(42),
                updates_per_step=UPDATES_PER_STEP,
                print_every=20_000,
                verbose=True,
                nstep_gamma=Scalar[DT](0.99),
                logger=UnsafePointer(to=logger),
                diag_every=5_000,
                checkpoint_every=CKPT_EVERY,
                checkpoint_path=String(CKPT_PATH),
                eval_env=UnsafePointer(to=eval_env),
                eval_every=100_000,
                eval_episodes=N_ENVS,
            )

            var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
            logger.close()
            print("-" * 70)
            print("Hybrid Training Complete")
            print("Training time:", String(elapsed_s)[byte=:6], "seconds")
            print("Final mean return (last 10):", agent.mean_return())

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)

    print(">>> main() completed normally <<<")
