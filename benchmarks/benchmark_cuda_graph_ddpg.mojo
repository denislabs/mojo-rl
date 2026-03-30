"""Benchmark: CUDA Graph capture on DDPG GPU training step.

Runs DDPG training on PendulumV2 and compares direct dispatch vs
graph-captured train steps. The graph captures one do_gpu_train_step
and replays it, eliminating per-kernel launch overhead.

Note: graph replay uses the same RNG seed (baked at capture time),
so this benchmarks launch overhead, not training quality.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_cuda_graph_ddpg.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent, DDPGConfig
from mojo_rl.deep_agents.core.kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
    uniform_random_actions_kernel,
)
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.cuda import CUDAGraph

from layout import Layout, LayoutTensor


def main() raises:
    print("=== CUDA Graph DDPG Benchmark ===\n")

    seed(42)
    var ctx = DeviceContext()

    # DDPG agent: obs=3, act=1, hidden=64, buffer=10000, batch=64
    comptime CONFIG = DDPGConfig[3, 1, 64, 10000, 64]
    comptime A = GenericOffPolicyAgent[CONFIG]
    var agent = A(action_scale=2.0)

    comptime E = PendulumV2[DType.float32]
    comptime n_envs = A.MAX_N_ENVS
    comptime GPUState = A.GPUStateType

    var gpu_state = GPUState(ctx)
    agent.upload_to_gpu(gpu_state, ctx)

    # --- Env buffers ---
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

    # Fill GPU replay buffer with random transitions
    print("Filling replay buffer...")
    comptime act_blocks = (n_envs * E.ACTION_DIM + TPB - 1) // TPB
    comptime warmup_k = uniform_random_actions_kernel[
        dtype, n_envs, E.ACTION_DIM
    ]
    var action_scale_val = Scalar[dtype](agent.action_scale)

    for i in range(200):
        ctx.enqueue_copy(prev_obs_buf, obs_buf)

        var act_t = LayoutTensor[
            dtype, Layout.row_major(n_envs, E.ACTION_DIM), MutAnyOrigin,
        ](actions_buf.unsafe_ptr())
        ctx.enqueue_function[warmup_k, warmup_k](
            act_t,
            action_scale_val,
            Scalar[DType.uint32](UInt32(i + 1)),
            grid_dim=(act_blocks,),
            block_dim=(TPB,),
        )

        E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf,
            terminated_buf, obs_buf, rng_seed=UInt64(i + 1),
        )
        gpu_state.gpu_store[n_envs](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf,
            terminated_buf,
        )
        E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
            ctx, states_buf, dones_buf, rng_seed=UInt64(i + 1),
        )
        E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
            ctx, states_buf, obs_buf,
        )

    ctx.synchronize()
    print("  Buffer size:", gpu_state.buffer.size)

    # =================================================================
    # Benchmark: Direct dispatch
    # =================================================================
    comptime ITERS = 1000
    print(
        "\n--- Direct dispatch ("
        + String(ITERS)
        + " train steps) ---"
    )

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
    print(
        "  Per step:", String(time_direct / Float64(ITERS))[byte=:8], "ms"
    )

    # =================================================================
    # Benchmark: CUDA Graph capture + replay
    # =================================================================
    print(
        "\n--- CUDA Graph ("
        + String(ITERS)
        + " replays) ---"
    )

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
    print(
        "  Per step:", String(time_graph / Float64(ITERS))[byte=:8], "ms"
    )

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 50)
    print("SUMMARY (" + String(ITERS) + " steps)")
    print("=" * 50)
    print("  Direct:  ", String(time_direct)[byte=:8], "ms")
    print("  Graph:   ", String(time_graph)[byte=:8], "ms")
    if time_graph > 0.0:
        print(
            "  Speedup: ",
            String(time_direct / time_graph)[byte=:6],
            "x",
        )
    print("=" * 50)
