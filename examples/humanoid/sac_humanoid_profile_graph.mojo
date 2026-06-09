"""SAC Humanoid (legacy DeepSACAgent) — short run for nsys profiling.

Humanoid counterpart of `examples/walker2d/sac_walker2d_profile_graph.mojo`.
Humanoid is the large-NV regime (NV=23, NBODY large, MAX_CONTACTS=50, RK4 +
Newton + PYRAMIDAL cone, STEP_THREADS=NV) — where the MuJoCo-Warp-style
block-parallel solver and the idle-thread RK4-stage parallelization have far
more headroom than walker2d's NV=9. Use this to see where Humanoid's GPU time
actually goes (solver vs RK4 stages vs network) before designing the large-NV
solver work.

Mirrors the walker2d legacy profiler exactly (same N_ENVS / step / warmup /
replay / batch / hidden) so the kernel breakdown is methodologically comparable;
only the env and its dims differ. Dims are taken from the env model (ground
truth) rather than hardcoded.

Run with:
    pixi run -e nvidia nsys profile --trace=cuda --cuda-graph-trace=node \
        --stats=true mojo run -I . \
        examples/humanoid/sac_humanoid_profile_graph.mojo

Profiling knobs (flip and re-profile to isolate cost):
  * USE_CUDA_GRAPH     — capture the per-update device kernel sequence into a
    CUDA graph and replay it. NVIDIA only.
  * USE_ENV_CUDA_GRAPH — capture the deterministic physics step into a graph
    too. NVIDIA only; no-op on Apple/Metal.

Physics-solver flags live in mojo_rl/physics3d/integrator/rk4_integrator.mojo
(RK4_BLOCKED_SOLVER / RK4_PARALLEL_MINV / RK4_PARALLEL_LDL) and apply here too
since Humanoid uses RK4Integrator[NewtonSolver] with STEP_THREADS=NV.
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.humanoid import Humanoid


# ─── Profiling knobs ──────────────────────────────────────────────────────
comptime USE_CUDA_GRAPH = True
comptime USE_ENV_CUDA_GRAPH = False

# ─── Sizing (mirrors sac_walker2d_profile_graph.mojo exactly) ──────────────
comptime dtype = DType.float32
comptime EnvT = Humanoid[dtype, TERMINATE_ON_UNHEALTHY=True]
comptime OBS_DIM = EnvT.OBS_DIM  # 45 (qpos[2:24] + qvel[0:23])
comptime ACTION_DIM = EnvT.ACTION_DIM  # joint torques (NACT)
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4
comptime NUM_STEPS = 50_000
comptime WARMUP_STEPS = 1_000


def main() raises:
    seed(42)
    print("=== SAC Humanoid nsys profile (legacy DeepSACAgent) ===")
    print("  OBS_DIM:", OBS_DIM, "| ACTION_DIM:", ACTION_DIM)
    print("  Steps:", NUM_STEPS, "| Warmup:", WARMUP_STEPS)
    print("  N_ENVS:", MAX_N_ENVS, "| BATCH:", BATCH_SIZE)
    print("  USE_CUDA_GRAPH:", USE_CUDA_GRAPH)
    print("  USE_ENV_CUDA_GRAPH:", USE_ENV_CUDA_GRAPH)
    print()

    with DeviceContext() as ctx:
        var agent = DeepSACAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.001,
            max_n_envs=MAX_N_ENVS,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=0.4,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.001,
            target_entropy=-Float64(ACTION_DIM),
        )

        var start = perf_counter_ns()

        var metrics = agent.train_gpu[
            EnvT,
            USE_CUDA_GRAPH=USE_CUDA_GRAPH,
            USE_ENV_CUDA_GRAPH=USE_ENV_CUDA_GRAPH,
        ](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            verbose=True,
            print_every=10_000,
        )

        var elapsed = Float64(perf_counter_ns() - start) / 1e9
        print()
        print("Total training wall-clock:", String(elapsed)[byte=:8], "s")
        print("Avg reward:", String(metrics.mean_reward_last_n(10))[byte=:8])
        print("=== Done ===")
