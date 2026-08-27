"""SAC SO-ARM101 reach — short run for nsys profiling.

Deliberately the same shape as
`examples/half_cheetah/sac_half_cheetah_profile_graph_nn.mojo` — same knobs,
same sizing style, no logger, no checkpoints — so an nsys side-by-side against
HalfCheetah is apples-to-apples and the ONLY difference is the model.

**What this is for.** SO-ARM101 is the first batched model in the tree with
collidable MESH geometry (26 179 hull vertices; `Phyics3dBatchedEnv` did not
even thread `nmesh_verts` until 2026-08-25). HalfCheetah is capsules only. So
the question this script answers is: **how much does mesh narrow-phase cost
per env-step, against a capsule model of similar DoF?**

⚠ A measurement on Apple Silicon is NOT the answer. A 24k-step run there gave
27.6 env-steps/s, and an extrapolation from it was quoted in conversation
before being retracted — Metal is the slow path for this engine and always
has been. Run this on NVIDIA.

Knobs, same meaning as the HalfCheetah script:
  * USE_TRAIN_CUDA_GRAPH — capture the per-update kernel sequence and replay.
  * USE_ENV_CUDA_GRAPH   — capture the ENV step. ⚠ Left False here: the
    HalfCheetah training script documents that capturing the blocked-Newton
    contact solver illegal-addresses on replay, and this model solves contacts
    through the same path with more geoms. Flip it only with a run to compare.
  * EPISODE_SYNC_EVERY   — batch the reward/done D2H readback.

Run with:
    pixi run -e nvidia nsys profile --stats=true mojo run -I . \\
        examples/so101/sac_so_arm101_reach_profile_gpu.mojo

The number to read out of `--stats=true` is the split between the env-step
kernels (narrow phase, solver, integrator) and the SAC update kernels. If the
env dominates, the next move is primitive collision geoms for the arm — the
visual meshes can stay — rather than anything in the agent.
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig
from mojo_rl.core.fmt import fit


# ─── Profiling knobs ──────────────────────────────────────────────────────
comptime USE_TRAIN_CUDA_GRAPH = False
comptime USE_ENV_CUDA_GRAPH = False
comptime EPISODE_SYNC_EVERY = 32

# ─── Sizing (mirrors the HalfCheetah profile script) ──────────────────────
comptime N_ENVS = 32

comptime BatchedEnvT = Phyics3dBatchedEnv[
    SoArm101Model, SoArm101ReachConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime OBS_DIM = BatchedEnvT.OBS_DIM  # 21
comptime ACT_DIM = 6
comptime HIDDEN = 256
comptime BUFFER_CAPACITY = 100_000
comptime BATCH = 256
comptime NUM_STEPS = 50_000
comptime WARMUP_STEPS = 1_000

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


def main() raises:
    seed(42)
    print("=== SAC SO-ARM101 reach — nsys profile ===")
    print("  Steps:", NUM_STEPS, "| Warmup:", WARMUP_STEPS)
    print("  N_ENVS:", N_ENVS, "| BATCH:", BATCH)
    print("  USE_TRAIN_CUDA_GRAPH:", USE_TRAIN_CUDA_GRAPH)
    print("  USE_ENV_CUDA_GRAPH:", USE_ENV_CUDA_GRAPH)
    print("  EPISODE_SYNC_EVERY:", EPISODE_SYNC_EVERY)
    print("  collidable mesh verts:", SoArm101ReachConfig.NMESH_VERTS)
    print()

    with DeviceContext() as ctx:
        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, BUFFER_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,  # normalized — see the training script
            init_alpha=0.2,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=WARMUP_STEPS,
            window_size=100,
            # ⚠ NOT the -1250.0 default: this task's return is in [0, 500] and
            # the sentinel makes every early reading look negative.
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        var start = perf_counter_ns()
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=USE_TRAIN_CUDA_GRAPH,
            USE_ENV_CUDA_GRAPH=USE_ENV_CUDA_GRAPH,
        ](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=10_000,
            verbose=True,
            episode_sync_every=EPISODE_SYNC_EVERY,
        )

        var elapsed = Float64(perf_counter_ns() - start) / 1e9
        print()
        print("Time:", fit(String(elapsed), 6), "s")
        print("env-steps/s:", fit(String(Float64(NUM_STEPS) / elapsed), 6))
        print("mean ep return (last 100):", agent.mean_return())
        print("episodes:", agent.ep_count())
        print("=== Done ===")
