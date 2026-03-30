"""Benchmark: CUDA Graph capture on DDPG GPU training step.

Fills a replay buffer with warmup data, then benchmarks:
  1. Direct dispatch: agent.do_gpu_train_step() in a loop
  2. Graph capture: capture one train step, replay N times

The DDPG train step has ~15-20 kernel launches (sample, concat, forward,
backward, optimizer for both critic and actor). With graph replay we
eliminate per-kernel launch overhead.

Note: graph replay uses the same RNG seed for buffer sampling (baked at
capture time), so the same batch is replayed. This is a benchmark for
launch overhead, not training quality.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_cuda_graph_ddpg.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent, DDPGConfig
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.cuda import CUDAGraph


def main() raises:
    print("=== CUDA Graph DDPG Benchmark ===\n")

    seed(42)
    var ctx = DeviceContext()

    # DDPG agent: obs=3, act=1, hidden=64, buffer=10000, batch=64
    comptime CONFIG = DDPGConfig[3, 1, 64, 10000, 64]
    var agent = GenericOffPolicyAgent[CONFIG](action_scale=2.0)

    # Fill the replay buffer via real training (need enough data for sampling)
    print("Warming up: training 5000 steps to fill replay buffer...")
    var metrics = agent.train_gpu[PendulumV2[DType.float32]](
        ctx,
        num_steps=5000,
        warmup_steps=1000,
    )
    print(
        "  Buffer filled. Total steps:",
        agent.total_steps,
        "| Train steps:",
        agent.train_step_count,
    )
    ctx.synchronize()

    # Get the GPU state for direct train step calls
    # We need to re-create it since train_gpu owns it internally
    # Instead, let's use the train_gpu_benchmark function pattern
    comptime GPUState = GenericOffPolicyAgent[CONFIG].GPUStateType
    var gpu_state = GPUState(ctx)

    # Upload current weights to GPU state
    agent.upload_to_gpu(gpu_state, ctx)
    ctx.synchronize()

    # Fill GPU replay buffer with some transitions from the env
    # Run the env + store loop manually
    comptime n_envs = GenericOffPolicyAgent[CONFIG].GPU_N_ENVS
    comptime E = PendulumV2[DType.float32]

    var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Reset envs
    E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=0)
    E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, terminated_buf,
        obs_buf, rng_seed=0,
    )
    ctx.synchronize()

    # Fill GPU buffer with random transitions
    for i in range(200):
        ctx.enqueue_copy(prev_obs_buf, obs_buf)
        agent.select_actions_gpu[n_envs](ctx, gpu_state, obs_buf, actions_buf)
        E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf, terminated_buf,
            obs_buf, rng_seed=UInt64(i + 1),
        )
        gpu_state.gpu_store[n_envs](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, terminated_buf
        )
        E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
            ctx, states_buf, dones_buf, rng_seed=UInt64(i + 1),
        )
        E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
            ctx, states_buf, obs_buf
        )
    ctx.synchronize()
    print("  GPU buffer size:", gpu_state.buffer.size)

    # =================================================================
    # Benchmark: Direct dispatch
    # =================================================================
    comptime ITERS = 1000
    print("\n--- Direct dispatch (" + String(ITERS) + " train steps) ---")

    # Warmup
    for _ in range(50):
        agent.do_gpu_train_step(ctx, gpu_state)
    agent.soft_update_targets_gpu(ctx, gpu_state)
    ctx.synchronize()

    var start = perf_counter_ns()
    for _ in range(ITERS):
        agent.do_gpu_train_step(ctx, gpu_state)
    ctx.synchronize()
    var time_direct = Float64(perf_counter_ns() - start) / 1e6

    print("  Time:", String(time_direct)[byte=:8], "ms")
    print("  Per step:", String(time_direct / Float64(ITERS))[byte=:8], "ms")

    # =================================================================
    # Benchmark: CUDA Graph capture + replay
    # =================================================================
    print("\n--- CUDA Graph (" + String(ITERS) + " replays) ---")

    # Capture one train step
    var graph = CUDAGraph(ctx)
    graph.begin_capture()
    agent.do_gpu_train_step(ctx, gpu_state)
    graph.end_capture()

    print("  Captured graph with", graph.num_nodes(), "nodes")

    # Warmup replay
    for _ in range(50):
        graph.replay()

    start = perf_counter_ns()
    for _ in range(ITERS):
        graph.replay()
    var time_graph = Float64(perf_counter_ns() - start) / 1e6

    print("  Time:", String(time_graph)[byte=:8], "ms")
    print("  Per step:", String(time_graph / Float64(ITERS))[byte=:8], "ms")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 50)
    print("SUMMARY (" + String(ITERS) + " steps)")
    print("=" * 50)
    print("  Direct:  ", String(time_direct)[byte=:8], "ms")
    print("  Graph:   ", String(time_graph)[byte=:8], "ms")
    if time_graph > 0.0:
        print("  Speedup: ", String(time_direct / time_graph)[byte=:6], "x")
    print("=" * 50)
