"""SAC on dm_control `dog-walk` (GPU, multi-env). The first training run on dog.

dog is the largest dm_control body: 79 dofs, 38 `dyntype="filter"` actuators,
a 223-dim observation, FRAME_SKIP 3 at a 0.005 s timestep. Like every suite
task it runs exactly 1000 steps with a per-step reward in [0, 1] and never
terminates early, so **a return reads against 1000**. The walk reward is
`upright * stand_height * move(speed 1 m/s)`.

The batched env is `DMDogWalkBatched`, gated step-for-step against the
MuJoCo-gated CPU env by `tests/dm_control/test_dog_gpu_vs_cpu.mojo`. The final
greedy eval runs on the CPU env (`DMDogWalk`), which is also a GPU→CPU
transfer check. The two resets differ: the CPU reset draws the 38 actuator
activations uniformly in their ctrlrange, while the GPU reset starts them at
zero (see `noeira/envs/dm_control/dog/dog.mojo`).

NVIDIA only. Apple builds the dog kernels but cannot run them (NV=79 exceeds
Metal's per-thread stack).

⚠ NOT A CONVERGENCE CLAIM. dog is known to be hard for model-free SAC from
states (the TD-MPC / TD-MPC2 papers use it as the task SAC does not solve).
This script exists to find out where noeira's SAC lands on it. The step-stamped
checkpoints make the learning curve renderable after the fact.

Hyper-parameters follow the walker ladder script
(`sac_dm_walker_training_gpu.mojo`: lr 3e-4, init α 0.2, target entropy
−|A|, batch 256, one update per transition) with wider nets (default 1024,
since 223 → 38 is a lot to push through 256).

Targets (comptime):
  -D DOG_N_ENVS=N    batched envs (default 32)
  -D DOG_HIDDEN=H    actor / critic width (default 1024)
  -D DOG_ENV_GRAPH   replay the env step from a CUDA graph (off by default:
                     never validated on dog; HalfCheetah replays bit-exact,
                     `bench_half_cheetah_batch.mojo`)

Args (positional): [seed=1] [env_steps=1000000] [ckpt_every_segments=4]
  A segment is one episode per env (1000 * N_ENVS env steps). The script
  writes `<run>/checkpoints/sac_dm_dog_walk.ckpt.<8-digit step>` every
  `ckpt_every_segments` segments, and always after the last one.

Run (on the NVIDIA box, through pixi so the CUDA interceptor is preloaded):
    pixi run mojo build -I . examples/dm_control/sac_dm_dog_walk_gpu.mojo -o sac_dog_walk
    pixi run ./sac_dog_walk 1 1000000 4
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.sys import argv
from std.sys.defines import get_defined_int, is_defined
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.deep_agents.primitives.stochastic_actor import StochasticActor
from noeira.deep_agents.sac import SACAgent
from noeira.deep_agents.training.blocks import UniformSampleGpuStep
from noeira.envs.dm_control.dog import (
    DMDogWalk,
    DMDogWalkBatched,
    DMDogStandWalkModel,
    DOG_MAX_STEPS,
)


comptime N_ENVS = get_defined_int["DOG_N_ENVS", 32]()
comptime HIDDEN = get_defined_int["DOG_HIDDEN", 1024]()
comptime ENV_GRAPH = is_defined["DOG_ENV_GRAPH"]()

comptime OBS_DIM = DMDogStandWalkModel.OBS_DIM  # 223
comptime ACT_DIM = DMDogStandWalkModel.ACTION_DIM  # 38
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
comptime WARMUP_STEPS = 10_000
comptime DIAG_EVERY = 1000
comptime EVAL_EPISODES = 5

comptime SEGMENT_STEPS = DOG_MAX_STEPS * N_ENVS
comptime MAX_RETURN = 1000.0

comptime BatchedEnvT = DMDogWalkBatched[N_ENVS]
comptime EvalEnvT = DMDogWalk[DT]

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def _stamped(prefix: String, step: Int) raises -> String:
    """`prefix.ckpt.00128000`: zero-padded so the ladder sorts by step."""
    var s = String(step)
    var pad = String("")
    for _ in range(8 - s.byte_length()):
        pad += "0"
    return prefix + ".ckpt." + pad + s


def main() raises:
    var args = argv()
    var run_seed = Int(String(args[1])) if len(args) > 1 else 1
    var env_steps = Int(String(args[2])) if len(args) > 2 else 1_000_000
    var ckpt_every = Int(String(args[3])) if len(args) > 3 else 4
    if ckpt_every < 1:
        raise Error("ckpt_every_segments must be >= 1")
    var n_segments = (env_steps + SEGMENT_STEPS - 1) // SEGMENT_STEPS
    seed(run_seed)

    print("=" * 70)
    print("SAC — dm_control dog-walk (GPU, multi-env)")
    print("=" * 70)
    print("  OBS_DIM / ACT_DIM  =", OBS_DIM, "/", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  N_ENVS             =", N_ENVS)
    print("  env graph          =", ENV_GRAPH)
    print("  env_steps          =", n_segments * SEGMENT_STEPS,
          "(", n_segments, "segments of", SEGMENT_STEPS, ")")
    print("  checkpoint every   =", ckpt_every, "segments")
    print("  seed               =", run_seed)
    print("  max return         = 1000 (dm_control convention)")
    print("=" * 70)

    var run = RunContext(
        project=String("dm-control"),
        driver=String("examples/dm_control/sac_dm_dog_walk_gpu.mojo"),
        slug=String("sac-dm-dog-walk"),
        env=String("builtin:dm_control/dog-walk"),
        seed=run_seed,
    )
    var prefix = run.dir + "/checkpoints/sac_dm_dog_walk"
    print("  Run                =", run.dir)

    with DeviceContext() as ctx:
        print("  device             =", ctx.name())
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "dm_control/dog-walk")
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))
        logger.set_config("env_graph", String(ENV_GRAPH))
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=0.2,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)
        ctx.synchronize()

        print("Starting GPU training ...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        var last_path = String("")
        # One agent across segments (nets, replay and optimizers persist);
        # each segment is one episode per env, so its mean return is fresh.
        for seg in range(n_segments):
            var done_steps = seg * SEGMENT_STEPS
            _ = agent.train[
                BatchedEnvT,
                N_ENVS=N_ENVS,
                USE_TRAIN_CUDA_GRAPH=True,
                USE_ENV_CUDA_GRAPH=ENV_GRAPH,
                L=RunLogger,
            ](
                env,
                SEGMENT_STEPS,
                rng_seed=UInt64(run_seed * 1000 + seg),
                updates_per_step=N_ENVS,
                print_every=SEGMENT_STEPS,
                verbose=True,
                logger=logger_ptr,
                diag_every=DIAG_EVERY,
                episode_sync_every=32,
                base_step=done_steps,
            )
            var at = done_steps + SEGMENT_STEPS
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "  [segment", seg + 1, "/", n_segments, "]  step", at,
                "  mean_ret", agent.mean_return(),
                "  elapsed", Int(elapsed), "s",
                "  sps", Int(Float64(at) / elapsed),
            )
            if (seg + 1) % ckpt_every == 0 or seg + 1 == n_segments:
                last_path = _stamped(prefix, at)
                agent.save(last_path)
                announce_checkpoint(last_path, artifacts, run.dir)
                print("  checkpoint ->", last_path)

        ctx.synchronize()
        var wall_s = Float64(perf_counter_ns() - t_start) / 1e9
        var train_mean = agent.mean_return()

        print("-" * 70)
        print("Greedy eval on the CPU env (", EVAL_EPISODES, "episodes ) ...")
        var eval_env = EvalEnvT()
        var eval_mean = agent.eval(
            eval_env, EVAL_EPISODES, max_steps_per_episode=DOG_MAX_STEPS
        )

        finish_run(
            run, logger, artifacts,
            String("mean_return_100=") + String(train_mean)
            + String(" greedy_eval_cpu=") + String(eval_mean),
        )
        _ = logger  # lifetime extender for logger_ptr

        var total = n_segments * SEGMENT_STEPS
        print("=" * 70)
        print(
            "RESULT env=dm_dog_walk algo=SAC",
            "n_envs=" + String(N_ENVS),
            "hidden=" + String(HIDDEN),
            "env_graph=" + String(ENV_GRAPH),
            "seed=" + String(run_seed),
            "env_steps=" + String(total),
            "wall_s=" + String(wall_s),
            "sps=" + String(Int(Float64(total) / wall_s)),
            "train_return_last100=" + String(train_mean),
            "greedy_eval_cpu_" + String(EVAL_EPISODES) + "ep="
            + String(eval_mean),
        )
        print("  last checkpoint:", last_path)
        var frac = Float64(eval_mean) / MAX_RETURN
        if frac > 0.8:
            print("EXCELLENT — walking at target speed (> 800 / 1000).")
        elif frac > 0.5:
            print("STRONG — sustained walking (> 500 / 1000).")
        elif frac > 0.2:
            print("PROGRESS — standing, some forward motion (> 200 / 1000).")
        else:
            print("WEAK — not walking yet (< 200 / 1000).")
        print("=" * 70)
