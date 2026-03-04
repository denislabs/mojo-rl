"""Benchmark Group F: Environment data collection kernels.

Kernels for GPU-parallel env stepping and action sampling:
  - tdmpc2_random_actions_kernel   [dtype, N_ENVS, ACT]
  - tdmpc2_sample_actions_kernel   [dtype, N_ENVS, ACT]   <- may be heavy (Philox RNG)
  - accumulate_rewards_kernel      [dtype, N_ENVS]
  - increment_steps_kernel         [dtype, N_ENVS]
  - extract_completed_episodes_kernel [dtype, N_ENVS]
  - selective_reset_tracking_kernel   [dtype, N_ENVS]

Note: these are compiled with n_envs=32 (HalfCheetah default),
      NOT the training batch size.

Run:
    pixi run -e apple mojo build examples/kernel_benchmarks/bench_f_env_collection.mojo -o /tmp/bench_f
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from nn.constants import dtype, TPB
from deep_agents.tdmpc2.kernels import (
    tdmpc2_random_actions_kernel,
    tdmpc2_sample_actions_kernel,
)
from nn.gpu.rl_kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    extract_completed_episodes_kernel,
    selective_reset_tracking_kernel,
)

comptime N_ENVS: Int = 32
comptime ACT: Int = 6
comptime MAX_EP_STEPS: Int = 1000


fn trigger_kernels(ctx: DeviceContext) raises:
    comptime MAX: Int = N_ENVS * ACT * 4
    var scratch = ctx.enqueue_create_buffer[dtype](MAX)
    var p = scratch.unsafe_ptr()

    comptime BLOCKS: Int = (N_ENVS + TPB - 1) // TPB

    # random_actions: actions[N_ENVS, ACT] <- uniform in [-1,1]
    ctx.enqueue_function[
        tdmpc2_random_actions_kernel[dtype, N_ENVS, ACT],
        tdmpc2_random_actions_kernel[dtype, N_ENVS, ACT],
    ](
        LayoutTensor[dtype, Layout.row_major(N_ENVS, ACT), MutAnyOrigin](p),
        Scalar[DType.uint32](42),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # sample_actions: pi_out[N_ENVS, 2*ACT] -> actions[N_ENVS, ACT]  (Philox RNG)
    ctx.enqueue_function[
        tdmpc2_sample_actions_kernel[dtype, N_ENVS, ACT],
        tdmpc2_sample_actions_kernel[dtype, N_ENVS, ACT],
    ](
        LayoutTensor[dtype, Layout.row_major(N_ENVS, ACT * 2), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS, ACT), MutAnyOrigin](p),
        Scalar[DType.uint32](42),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # accumulate_rewards: ep_rews[N_ENVS] += step_rews[N_ENVS]
    ctx.enqueue_function[
        accumulate_rewards_kernel[dtype, N_ENVS],
        accumulate_rewards_kernel[dtype, N_ENVS],
    ](
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # increment_steps: ep_steps[N_ENVS] += 1
    ctx.enqueue_function[
        increment_steps_kernel[dtype, N_ENVS],
        increment_steps_kernel[dtype, N_ENVS],
    ](
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # extract_completed_episodes: gather ep metrics for done envs
    ctx.enqueue_function[
        extract_completed_episodes_kernel[dtype, N_ENVS],
        extract_completed_episodes_kernel[dtype, N_ENVS],
    ](
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # selective_reset_tracking: zero ep_rews/ep_steps for done envs
    ctx.enqueue_function[
        selective_reset_tracking_kernel[dtype, N_ENVS],
        selective_reset_tracking_kernel[dtype, N_ENVS],
    ](
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )


fn main() raises:
    var ctx = DeviceContext()
    trigger_kernels(ctx)
    ctx.synchronize()
    print("Group F kernels compiled and ran OK")
