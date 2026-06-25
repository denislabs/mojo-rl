"""Rainbow DQN GPU training on CarRacing (clean obs, multi-body physics).

Trains a Rainbow agent (C51 + Double + PER + Dueling + Noisy + N-step) on
`CarRacingDiscrete` — the Box2D-faithful multi-body CarRacing with a 5-action
discrete interface — stepping `N_ENVS` environments in parallel on the GPU via
`BatchedGpuDiscreteEnv`, exactly the structure of the clean-obs Pong Rainbow run.

CarRacingDiscrete: 13-D normalized clean observation, 5 discrete actions
(do-nothing, steer-left, steer-right, gas, brake). Reward is Gymnasium-faithful
(-0.1/frame + 1000/N per tile, -100 off-playfield).

Run with:
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_training_gpu.mojo   # Apple Silicon (compile/smoke)
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_training_gpu.mojo   # NVIDIA GPU (training)
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
from mojo_rl.envs.car_racing import CarRacingDiscrete


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = CarRacingDiscrete[DT].OBS_DIM  # 13
comptime NUM_ACTIONS = CarRacingDiscrete[DT].NUM_ACTIONS  # 5

# Rainbow architecture / replay hyperparameters.
comptime HIDDEN_DIM = 256  # richer obs than Pong (13D vs 6D)
comptime NUM_ATOMS = 51
comptime N_STEP = 3
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 64
comptime N_ENVS = 256  # parallel GPU environments

# Distributional support. CarRacing rewards are larger than Pong's: +1000/N
# (~+3.3) per tile, -0.1/frame, and a one-off -100 off-playfield. The
# *discounted* Q for a competent on-track policy accumulates into the tens..low
# hundreds, with the -100 floor on the downside. [-100, 100] (atom spacing 4)
# brackets that as a starting point — this support + reward scale is the main
# C51 convergence lever and is the first thing to tune (cf. the Pong run, which
# needed ±2). Narrow it once you observe the realized discounted-return range.
comptime V_MIN = Scalar[DT](-100.0)
comptime V_MAX = Scalar[DT](100.0)

# Replay ratio: GRAD_STEPS / N_ENVS = 64/256 = 0.25 (CleanRL Atari freq 4).
comptime GRAD_STEPS = 64
comptime WARMUP = 20_000
comptime NUM_STEPS = 10_000_000
comptime LR = Scalar[DT](6.25e-5)

comptime CKPT_EVERY = 250_000
comptime CKPT_PATH = "checkpoints/rainbow_car_racing.ckpt"

comptime CarRacingBatched = BatchedGpuDiscreteEnv[
    CarRacingDiscrete[DT], N_ENVS, OBS_DIM, 1
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN GPU Training on CarRacing (clean obs, multi-body)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
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

        var env = CarRacingBatched(ctx)
        var eval_env = CarRacingBatched(ctx)

        print("Environment: CarRacingDiscrete (GPU-batched,", N_ENVS, "envs)")
        print("Agent: Rainbow DQN (deep_agents C51, GPU)")
        print("  Observation dim:", OBS_DIM, "(normalized clean state)")
        print("  Actions:", NUM_ACTIONS, "(noop, left, right, gas, brake)")
        print("  Hidden dim:", HIDDEN_DIM)
        print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
        print("  N-step:", N_STEP, " N envs:", N_ENVS)
        print("  Buffer:", BUFFER_CAPACITY, " Batch:", BATCH_SIZE)
        print("  LR:", LR, " Warmup:", WARMUP, " Total:", NUM_STEPS)
        print("  Checkpoint:", CKPT_PATH, "(every", CKPT_EVERY, "steps)")
        print()

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")
        var logger = RemoteLogger(
            server_url=url,
            run_name="Rainbow CarRacing GPU (multi-body)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Rainbow DQN (deep_agents)")
        logger.set_config("env", "CarRacingDiscrete")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("lr", String(LR))
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("n_step", String(N_STEP))
        logger.set_config("num_atoms", String(NUM_ATOMS))

        print("Starting GPU training...")
        print("-" * 70)
        var start_time = perf_counter_ns()

        try:
            var _ep_returns = agent.train_gpu_batched[
                CarRacingBatched, N_ENVS, N_STEP, RemoteLogger
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
            print("Rainbow GPU Training Complete")
            print("Training time:", String(elapsed_s)[byte=:6], "seconds")
            print("Final mean return (last 10):", agent.mean_return())
            print("Episodes completed:", agent.ep_count())

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)

    print(">>> main() completed normally <<<")
