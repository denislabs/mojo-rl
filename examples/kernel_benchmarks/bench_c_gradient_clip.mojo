"""Benchmark Group C: Gradient clipping & soft-update kernels.

These are block-reduction kernels, one instantiation per network's PARAM_SIZE:
  - gradient_norm_kernel[dtype, P, BLOCKS, TPB]          × 5 unique P values
  - gradient_reduce_apply_fused_kernel[dtype, P, BLOCKS, TPB]  × 5 unique P values
  - soft_update_kernel[dtype, Q_P]

Network param sizes (HalfCheetah TDMPC2, computed from NormedLinear architecture):
  ENC_P  =  71424  (NormedLinear[17,256] + NormedLinear[256,256])
  DYN_P  = 199936  (NormedLinear[262,256] + NormedLinear[256,256] + Linear[256,256] + SimNorm)
  REW_P  = 160101  (NormedLinear[262,256] + NormedLinear[256,256] + Linear[256,101])
  TERM_P = 132865  (NormedLinear[256,256] × 2 + Linear[256,1])
  POL_P  = 135692  (NormedLinear[256,256] × 2 + Linear[256,12])
  Q_P    = 160101  (same as REW_P -> same kernel!)

Run:
    pixi run -e apple mojo build examples/kernel_benchmarks/bench_c_gradient_clip.mojo -o /tmp/bench_c
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from nn.constants import dtype, TPB
from deep_agents.ppo.kernels import (
    gradient_norm_kernel,
    gradient_reduce_apply_fused_kernel,
)
from deep_agents.core.kernels import soft_update_kernel

# Exact param sizes matching TDMPC2Agent[17, 6, 256, 256, 101, 5, 8] on HalfCheetah
comptime ENC_P: Int = 71424
comptime DYN_P: Int = 199936
comptime REW_P: Int = 160101
comptime TERM_P: Int = 132865
comptime POL_P: Int = 135692
comptime Q_P: Int = 160101  # same as REW_P -> kernel already compiled

comptime ENC_BLOCKS: Int = (ENC_P + TPB - 1) // TPB  # 280
comptime DYN_BLOCKS: Int = (DYN_P + TPB - 1) // TPB  # 782
comptime REW_BLOCKS: Int = (REW_P + TPB - 1) // TPB  # 626
comptime TERM_BLOCKS: Int = (TERM_P + TPB - 1) // TPB  # 520
comptime POL_BLOCKS: Int = (POL_P + TPB - 1) // TPB  # 531


fn trigger_gradient_norm[
    P: Int, BLOCKS: Int
](ctx: DeviceContext, p: UnsafePointer[Scalar[dtype]]) raises:
    """Compile and run gradient_norm + gradient_reduce_apply for param size P.
    """
    ctx.enqueue_function[
        gradient_norm_kernel[dtype, P, BLOCKS, TPB],
        gradient_norm_kernel[dtype, P, BLOCKS, TPB],
    ](
        LayoutTensor[dtype, Layout.row_major(BLOCKS), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(P), MutAnyOrigin](p),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )
    ctx.enqueue_function[
        gradient_reduce_apply_fused_kernel[dtype, P, BLOCKS, TPB],
        gradient_reduce_apply_fused_kernel[dtype, P, BLOCKS, TPB],
    ](
        LayoutTensor[dtype, Layout.row_major(P), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(BLOCKS), MutAnyOrigin](p),
        Scalar[dtype](1.0),  # max_grad_norm
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )


fn trigger_kernels(ctx: DeviceContext) raises:
    comptime MAX: Int = DYN_P  # largest param size
    var scratch = ctx.enqueue_create_buffer[dtype](MAX)
    var p = scratch.unsafe_ptr()

    trigger_gradient_norm[ENC_P, ENC_BLOCKS](ctx, p)
    trigger_gradient_norm[DYN_P, DYN_BLOCKS](ctx, p)
    trigger_gradient_norm[REW_P, REW_BLOCKS](ctx, p)
    trigger_gradient_norm[TERM_P, TERM_BLOCKS](ctx, p)
    trigger_gradient_norm[POL_P, POL_BLOCKS](ctx, p)
    # Q_P == REW_P -> same kernel, already compiled above

    # Soft update: target = tau * source + (1-tau) * target
    ctx.enqueue_function[
        soft_update_kernel[dtype, Q_P],
        soft_update_kernel[dtype, Q_P],
    ](
        LayoutTensor[dtype, Layout.row_major(Q_P), MutAnyOrigin](p),
        LayoutTensor[dtype, Layout.row_major(Q_P), MutAnyOrigin](p),
        Scalar[dtype](0.01),
        grid_dim=((Q_P + TPB - 1) // TPB,),
        block_dim=(TPB,),
    )


fn main() raises:
    var ctx = DeviceContext()
    trigger_kernels(ctx)
    ctx.synchronize()
    print("Group C kernels compiled and ran OK")
