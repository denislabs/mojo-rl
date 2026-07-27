"""Rainbow DQN (CNN) CPU training on Procgen Starpilot — pixel obs.

Single-env CPU proof that `StarpilotGymEnv` learns with the deep_agents Rainbow stack
(C51 + Double + PER + Dueling + Noisy + N-step) on a Nature-CNN backbone over
84×84×3 observations. Mirrors `maze_rainbow_training_cpu.mojo` but on the starpilot
projectile substrate. Reward is DENSE (+1 per enemy shot down, +10 for surviving
to the finish) → expect chaser-like learnability.

Scoped as a *learning* proof, not a speed run: CPU convs are slow, so it defaults
to **Easy mode, a single level** (`num_levels=1`). Scale up (more levels, longer
horizon) and move to NVIDIA GPU-batched for a real run. See
`docs/PROCGEN_STARPILOT_SCOPE.md`.

Run:
    pixi run mojo run -I . examples/procgen/starpilot_rainbow_training_cpu.mojo
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

from mojo_rl.envs.procgen.games import StarpilotGymEnv
from mojo_rl.envs.procgen.games.starpilot import DIST_EASY

comptime ASSET_ROOT = String("assets/procgen/")

comptime StarpilotCNNEnv = StarpilotGymEnv[DT]
comptime OBS_DIM = StarpilotCNNEnv.OBS_DIM  # 3 × 84 × 84 = 21168
comptime NUM_ACTIONS = StarpilotCNNEnv.NUM_ACTIONS  # 15

comptime FRAMES = 3  # RGB channels (no frame stack — starpilot is fully observable)
comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3
comptime BATCH_SIZE = 32
# CPU replay stores obs as float DT (uint8 obs ring is a GPU-only option), so
# each slot is OBS_DIM floats — keep the buffer modest to bound host memory.
comptime BUFFER_CAPACITY = 5_000
comptime LAYOUT = LAYOUT_NCHW

comptime WARMUP = 1_000
comptime NUM_STEPS = 200_000  # proof; extend for a real run

comptime LR = Scalar[DT](6.25e-5)
# Value support: 0 .. ~50 (dense +1/enemy killed over the episode + 10 finish).
comptime V_MIN = Scalar[DT](-1.0)
comptime V_MAX = Scalar[DT](51.0)

comptime SAMPLE = NStepSampleStep[
    N_STEP,
    AnyPerReplay["cpu", OBS_DIM, 1, BUFFER_CAPACITY],
    BATCH_SIZE,
]
comptime QNET = RainbowCNNNet[FRAMES, NUM_ACTIONS, NUM_ATOMS, HIDDEN, LAYOUT]
comptime StarpilotTrainer = C51Trainer[
    "cpu", SAMPLE, QNET, NUM_ATOMS, NUM_ACTIONS, True
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN CPU Training on Procgen Starpilot (pixel, single-env)")
    print("=" * 70)

    var trainer = StarpilotTrainer.make(
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

    var env = StarpilotCNNEnv(
        ASSET_ROOT,
        rand_seed=0,
        num_levels=1,
        start_level=0,
        dist_mode=DIST_EASY,
    )
    var eval_env = StarpilotCNNEnv(
        ASSET_ROOT,
        rand_seed=0,
        num_levels=1,
        start_level=0,
        dist_mode=DIST_EASY,
    )

    print("Environment: Procgen Starpilot (CPU, single env, Easy, 1 level)")
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
        run_name="Rainbow Starpilot CPU (procgen)",
        buffer_size=64,
        api_key=api_key,
    )
    logger.set_config("agent", "Rainbow DQN CNN (deep_agents, CPU)")
    logger.set_config("env", "procgen-starpilot-easy-1level")
    logger.set_config("obs_dim", String(OBS_DIM))
    logger.set_config("lr", String(LR))
    logger.set_config("n_step", String(N_STEP))

    print("Starting CPU training (slow: CPU convs)...")
    print("-" * 70)
    var start = perf_counter_ns()
    var _ret = run_offpolicy_discrete_train[
        StarpilotTrainer, StarpilotCNNEnv, RemoteLogger
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
