"""Rainbow DQN (CNN) CPU training on Procgen Bigfish — pixel obs.

Single-env CPU proof that `BigfishGymEnv` learns with the deep_agents Rainbow
stack on a Nature-CNN backbone. Mirrors the maze/chaser/heist CPU trainers.
Bigfish reward is DENSE (+1 per fish eaten, +10 for the 30-fish quota) → expect
chaser-like learnability. Defaults to Easy mode, single level. Move to NVIDIA
GPU-batched for a real run. See `docs/PROCGEN_BIGFISH_SCOPE.md`.

Run:
    pixi run mojo run -I . examples/procgen/bigfish_rainbow_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT, LAYOUT_NCHW

from mojo_rl.deep_agents.c51.trainer import C51Trainer
from mojo_rl.deep_agents.c51.config import RainbowCNNNet
from mojo_rl.deep_agents.training.blocks import NStepSampleStep
from mojo_rl.deep_agents.data.any_per_replay import AnyPerReplay
from mojo_rl.deep_agents.training import run_offpolicy_discrete_train

from mojo_rl.envs.procgen.games import BigfishGymEnv
from mojo_rl.envs.procgen.games.bigfish import DIST_EASY

comptime ASSET_ROOT = String("assets/procgen/")

comptime BigfishCNNEnv = BigfishGymEnv[DT]
comptime OBS_DIM = BigfishCNNEnv.OBS_DIM  # 3 × 84 × 84 = 21168
comptime NUM_ACTIONS = BigfishCNNEnv.NUM_ACTIONS  # 15

comptime FRAMES = 3
comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3
comptime BATCH_SIZE = 32
comptime BUFFER_CAPACITY = 5_000
comptime LAYOUT = LAYOUT_NCHW

comptime WARMUP = 1_000
comptime NUM_STEPS = 200_000

comptime LR = Scalar[DT](6.25e-5)
# Dense reward: up to ~40 (30 fish × 1 + 10 completion). Bracket generously.
comptime V_MIN = Scalar[DT](-1.0)
comptime V_MAX = Scalar[DT](41.0)

comptime SAMPLE = NStepSampleStep[
    N_STEP,
    AnyPerReplay["cpu", OBS_DIM, 1, BUFFER_CAPACITY],
    BATCH_SIZE,
]
comptime QNET = RainbowCNNNet[FRAMES, NUM_ACTIONS, NUM_ATOMS, HIDDEN, LAYOUT]
comptime BigfishTrainer = C51Trainer[
    "cpu", SAMPLE, QNET, NUM_ATOMS, NUM_ACTIONS, True
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN CPU Training on Procgen Bigfish (pixel, single-env)")
    print("=" * 70)

    var trainer = BigfishTrainer.make(
        lr=LR,
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](0.0),
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

    var env = BigfishCNNEnv(
        ASSET_ROOT, rand_seed=0, num_levels=1, start_level=0, dist_mode=DIST_EASY
    )
    var eval_env = BigfishCNNEnv(
        ASSET_ROOT, rand_seed=0, num_levels=1, start_level=0, dist_mode=DIST_EASY
    )

    print("Environment: Procgen Bigfish (CPU, single env, Easy, 1 level)")
    print("Agent: Rainbow DQN CNN (deep_agents C51)")
    print("  Observation:", FRAMES, "× 84 × 84 =", OBS_DIM, "(NCHW)")
    print("  Actions:", NUM_ACTIONS)
    print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
    print("  Buffer:", BUFFER_CAPACITY)
    print("  Batch:", BATCH_SIZE, " N-step:", N_STEP, " lr:", LR)
    print("  Warmup:", WARMUP, " Total steps:", NUM_STEPS)
    print()

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")
    var logger = RemoteLogger(
        server_url=url,
        run_name="Rainbow Bigfish CPU (procgen)",
        buffer_size=64,
        api_key=api_key,
    )
    logger.set_config("agent", "Rainbow DQN CNN (deep_agents, CPU)")
    logger.set_config("env", "procgen-bigfish-easy-1level")
    logger.set_config("obs_dim", String(OBS_DIM))
    logger.set_config("lr", String(LR))
    logger.set_config("n_step", String(N_STEP))

    print("Starting CPU training (slow: CPU convs)...")
    print("-" * 70)
    var start = perf_counter_ns()
    var _ret = run_offpolicy_discrete_train[
        BigfishTrainer, BigfishCNNEnv, RemoteLogger
    ](
        trainer,
        env,
        NUM_STEPS,
        print_every=2_000,
        verbose=True,
        logger=UnsafePointer(to=logger).as_unsafe_any_origin(),
        diag_every=2_000,
        eval_env=UnsafePointer(to=eval_env).as_unsafe_any_origin(),
        eval_every=20_000,
        eval_episodes=3,
    )
    var elapsed = Float64(perf_counter_ns() - start) / 1e9
    logger.close()
    print("-" * 70)
    print("Done in", String(elapsed)[byte=:6], "s")
    print("Final mean return (last 10):", trainer.mean_return())
    print("Episodes:", trainer.ep_count())
