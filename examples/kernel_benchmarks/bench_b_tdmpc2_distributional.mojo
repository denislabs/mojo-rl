"""Benchmark Group B: Distributional RL kernels (BINS=101).

These involve loops over NUM_BINS=101, expected to be heavier than Group A:
  - tdmpc2_two_hot_loss_grad_kernel  [BATCH, BINS]
  - tdmpc2_q_decode_kernel           [BATCH, BINS]
  - tdmpc2_compute_td_targets_kernel [BATCH, BINS]
  - tdmpc2_decode_and_min_kernel     [BATCH, BINS]

Run:
    pixi run -e apple mojo build examples/kernel_benchmarks/bench_b_tdmpc2_distributional.mojo -o /tmp/bench_b
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from nn.constants import dtype, TPB
from deep_agents.tdmpc2.kernels import (
    tdmpc2_two_hot_loss_grad_kernel,
    tdmpc2_q_decode_kernel,
    tdmpc2_compute_td_targets_kernel,
    tdmpc2_decode_and_min_kernel,
)

comptime BATCH: Int = 256
comptime BINS: Int = 101
comptime NUM_Q: Int = 5


fn trigger_kernels(ctx: DeviceContext) raises:
    comptime MAX: Int = BATCH * BINS * NUM_Q
    var scratch = ctx.enqueue_create_buffer[dtype](MAX)
    var p = scratch.unsafe_ptr()

    comptime BLOCKS: Int = (BATCH + TPB - 1) // TPB

    # two_hot_loss_grad: logits[B,BINS], targets[B,BINS] -> grad[B,BINS]
    ctx.enqueue_function[
        tdmpc2_two_hot_loss_grad_kernel[dtype, BATCH, BINS],
        tdmpc2_two_hot_loss_grad_kernel[dtype, BATCH, BINS],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin](p),
        Scalar[dtype](1.0),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # q_decode: logits[B,BINS], bins[BINS] -> values[B]
    ctx.enqueue_function[
        tdmpc2_q_decode_kernel[dtype, BATCH, BINS],
        tdmpc2_q_decode_kernel[dtype, BATCH, BINS],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # td_targets: rewards[B], dones[B], q_next[B] -> td_targets[B,BINS]
    ctx.enqueue_function[
        tdmpc2_compute_td_targets_kernel[dtype, BATCH, BINS],
        tdmpc2_compute_td_targets_kernel[dtype, BATCH, BINS],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin](p),
        Scalar[dtype](0.99),
        Scalar[dtype](-10.0),
        Scalar[dtype](10.0),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # decode_and_min: logits[B,BINS], bins[BINS], q_min[B] (running min, in-place)
    ctx.enqueue_function[
        tdmpc2_decode_and_min_kernel[dtype, BATCH, BINS],
        tdmpc2_decode_and_min_kernel[dtype, BATCH, BINS],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )


fn main() raises:
    var ctx = DeviceContext()
    trigger_kernels(ctx)
    ctx.synchronize()
    print("Group B kernels compiled and ran OK")
