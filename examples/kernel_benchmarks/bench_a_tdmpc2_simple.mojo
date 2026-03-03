"""Benchmark Group A: Simple TDMPC2 data & gradient kernels.

Kernels tested (all element-wise / scatter-gather, no heavy arithmetic):
  - tdmpc2_build_za_kernel         [BATCH, LATENT, ACT]
  - tdmpc2_extract_obs_step_kernel [BATCH, OBS, H]
  - tdmpc2_extract_act_step_kernel [BATCH, ACT, H]
  - tdmpc2_extract_rew_done_kernel [BATCH, H]
  - tdmpc2_extract_z_from_za_grad_kernel [BATCH, LATENT, ACT]
  - tdmpc2_consistency_loss_grad_kernel  [BATCH, LATENT]
  - tdmpc2_add_two_into_kernel     [B_LATENT]
  - tdmpc2_apply_tanh_kernel       [BATCH, ACT]
  - copy_buffer_kernel             [B_LATENT]
  - tdmpc2_bce_loss_grad_kernel    [BATCH]
  - tdmpc2_policy_grad_kernel      [BATCH, ACT]

Run:
    pixi run -e apple mojo build examples/kernel_benchmarks/bench_a_tdmpc2_simple.mojo -o /tmp/bench_a
"""

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from nn.constants import dtype, TPB
from deep_agents.tdmpc2.kernels import (
    tdmpc2_build_za_kernel,
    tdmpc2_extract_obs_step_kernel,
    tdmpc2_extract_act_step_kernel,
    tdmpc2_extract_rew_done_kernel,
    tdmpc2_extract_z_from_za_grad_kernel,
    tdmpc2_consistency_loss_grad_kernel,
    tdmpc2_add_two_into_kernel,
    tdmpc2_apply_tanh_kernel,
    tdmpc2_bce_loss_grad_kernel,
    tdmpc2_policy_grad_kernel,
)
from nn.gpu.rl_kernels import copy_buffer_kernel

# HalfCheetah TDMPC2 dimensions
comptime BATCH: Int = 256
comptime LATENT: Int = 256
comptime ACT: Int = 6
comptime OBS: Int = 17
comptime H: Int = 3
comptime B_LATENT: Int = BATCH * LATENT


fn trigger_kernels(ctx: DeviceContext) raises:
    """Call each kernel once to force GPU compilation."""
    comptime MAX: Int = BATCH * (H + 1) * LATENT
    var scratch = ctx.enqueue_create_buffer[dtype](MAX)
    var p = scratch.unsafe_ptr()

    comptime BLOCKS: Int = (BATCH + TPB - 1) // TPB

    # build_za: concat z[B,L] and a[B,A] -> za[B,L+A]
    ctx.enqueue_function[
        tdmpc2_build_za_kernel[dtype, BATCH, LATENT, ACT],
        tdmpc2_build_za_kernel[dtype, BATCH, LATENT, ACT],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, LATENT + ACT), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # extract_obs: flat_obs[(H+1)*B*OBS] -> obs_t[B,OBS]
    ctx.enqueue_function[
        tdmpc2_extract_obs_step_kernel[dtype, BATCH, OBS, H],
        tdmpc2_extract_obs_step_kernel[dtype, BATCH, OBS, H],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH * (H + 1) * OBS), MutAnyOrigin](p),
        0,  # step
        LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # extract_act: flat_act[H*B*ACT] -> act_t[B,ACT]
    ctx.enqueue_function[
        tdmpc2_extract_act_step_kernel[dtype, BATCH, ACT, H],
        tdmpc2_extract_act_step_kernel[dtype, BATCH, ACT, H],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH * H * ACT), MutAnyOrigin](p),
        0,  # step
        LayoutTensor[dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # extract_rew_done: flat_rew[H*B], flat_done[H*B] -> rew[B], done[B]
    ctx.enqueue_function[
        tdmpc2_extract_rew_done_kernel[dtype, BATCH, H],
        tdmpc2_extract_rew_done_kernel[dtype, BATCH, H],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH * H), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH * H), MutAnyOrigin](p),
        0,  # step
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # extract_z_from_za_grad: za_grad[B,L+A] -> z_grad[B,L]
    ctx.enqueue_function[
        tdmpc2_extract_z_from_za_grad_kernel[dtype, BATCH, LATENT, ACT],
        tdmpc2_extract_z_from_za_grad_kernel[dtype, BATCH, LATENT, ACT],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH, LATENT + ACT), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # consistency_loss_grad: z_pred[B,L], z_tgt[B,L], grad[B,L], rho_weight
    ctx.enqueue_function[
        tdmpc2_consistency_loss_grad_kernel[dtype, BATCH, LATENT],
        tdmpc2_consistency_loss_grad_kernel[dtype, BATCH, LATENT],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin](p),
        Scalar[dtype](1.0),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # add_two_into: dst[N] += src1[N] + src2[N]
    ctx.enqueue_function[
        tdmpc2_add_two_into_kernel[dtype, B_LATENT],
        tdmpc2_add_two_into_kernel[dtype, B_LATENT],
    ](
        LayoutTensor[dtype, Layout.row_major(B_LATENT), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(B_LATENT), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(B_LATENT), MutAnyOrigin](p),
        grid_dim=((B_LATENT + TPB - 1) // TPB,),
        block_dim=(TPB,),
    )

    # apply_tanh: pi_out[B,2A] -> actions[B,A]
    ctx.enqueue_function[
        tdmpc2_apply_tanh_kernel[dtype, BATCH, ACT],
        tdmpc2_apply_tanh_kernel[dtype, BATCH, ACT],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH, ACT * 2), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # copy_buffer: src -> dst
    ctx.enqueue_function[
        copy_buffer_kernel[dtype, B_LATENT],
        copy_buffer_kernel[dtype, B_LATENT],
    ](
        LayoutTensor[dtype, Layout.row_major(B_LATENT), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(B_LATENT), MutAnyOrigin](p),
        grid_dim=((B_LATENT + TPB - 1) // TPB,),
        block_dim=(TPB,),
    )

    # bce_loss_grad: probs[B], dones[B] -> grad_probs[B]
    ctx.enqueue_function[
        tdmpc2_bce_loss_grad_kernel[dtype, BATCH],
        tdmpc2_bce_loss_grad_kernel[dtype, BATCH],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        Scalar[dtype](1.0),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    # policy_grad: pi_out[B,2A], q_vals[B], grad[B,2A], rho_weight, entropy_coef
    ctx.enqueue_function[
        tdmpc2_policy_grad_kernel[dtype, BATCH, ACT],
        tdmpc2_policy_grad_kernel[dtype, BATCH, ACT],
    ](
        LayoutTensor[dtype, Layout.row_major(BATCH, ACT * 2), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BATCH, ACT * 2), MutAnyOrigin](p),
        Scalar[dtype](1.0),
        Scalar[dtype](1e-4),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )


fn main() raises:
    var ctx = DeviceContext()
    trigger_kernels(ctx)
    ctx.synchronize()
    print("Group A kernels compiled and ran OK")
