"""GPU PCN training step + workspace (Phase D).

`pc_module_train_one_batch_gpu` is the GPU mirror of
`pc_module_train_one_batch`: the settling loop runs on-device via
`PCTrainer.compute_grads_only_gpu`, fills `net.weights.grd` (device), then an
nn2 `Optimizer.step['gpu']` updates `net.weights.val`.

All matmuls in the GPU settling go through `linalg.matmul` (`max_matmul`),
matching nn2's convention — PCN's custom 2×2 register-tiled MMA fallback is
dead on the default path (`PCBlock.USE_MAX_KERNELS=True`). See
`pc_block.mojo`'s `predict_gpu` / `pull_back_gpu` / `weight_grad_gpu`.

`PCGpuWorkspace` owns the settling's device working buffers (latents, μ/ε,
a_below, z_below, dx). Allocate it ONCE and reuse it across steps so the
training loop does no per-step device allocation (and stays CUDA-graph
capturable). This is the GPU analogue of moving the working buffers onto the
holder as persistent storage.
"""

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.optimizer import Optimizer

from .predictive_model import PCBlockTrait
from .pc_sequential import PCSequential
from .pc_trainer import PCTrainer
from .pc_module import PCModule


@fieldwise_init
struct PCGpuWorkspace[BATCH: Int, *BLOCKS: PCBlockTrait](Movable):
    """Persistent device working buffers for the GPU settling loop. Sized
    once from the net's compile-time dims; reused every training step."""

    comptime NET = PCSequential[*Self.BLOCKS]

    var latents_b: DeviceBuffer[DT]
    var mu_eps_b: DeviceBuffer[DT]
    var a_below_b: DeviceBuffer[DT]
    var z_below_b: DeviceBuffer[DT]
    var dx_b: DeviceBuffer[DT]

    @staticmethod
    def make(ctx: DeviceContext) raises -> Self:
        return Self(
            latents_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.LATENT_DIM
            ),
            mu_eps_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.SCRATCH_OUT_DIM
            ),
            a_below_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.SCRATCH_IN_DIM
            ),
            z_below_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.SCRATCH_IN_DIM
            ),
            dx_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.LATENT_DIM
            ),
        )


def pc_module_train_one_batch_gpu[
    BATCH: Int, OPT: Optimizer, *BLOCKS: PCBlockTrait
](
    ctx: DeviceContext,
    mut net: PCModule[*BLOCKS],
    mut opt: OPT,
    mut ws: PCGpuWorkspace[BATCH, *BLOCKS],
    x_in: LayoutTensor[
        DT, Layout.row_major(BATCH, PCSequential[*BLOCKS].IN_DIM), MutAnyOrigin
    ],
    y_target: LayoutTensor[
        DT, Layout.row_major(BATCH, PCSequential[*BLOCKS].OUT_DIM), MutAnyOrigin
    ],
    T_infer: Int,
    lr_x: Scalar[DT],
) raises:
    comptime NET = PCSequential[*BLOCKS]

    var params = LayoutTensor[
        DT, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](net.weights.val.dev.value().unsafe_ptr())
    var grads = LayoutTensor[
        DT, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](net.weights.grd.dev.value().unsafe_ptr())

    var latents = LayoutTensor[
        DT, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](ws.latents_b.unsafe_ptr())
    var mu_eps = LayoutTensor[
        DT, Layout.row_major(BATCH, NET.SCRATCH_OUT_DIM), MutAnyOrigin
    ](ws.mu_eps_b.unsafe_ptr())
    var a_below = LayoutTensor[
        DT, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](ws.a_below_b.unsafe_ptr())
    var z_below = LayoutTensor[
        DT, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](ws.z_below_b.unsafe_ptr())
    var dx = LayoutTensor[
        DT, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](ws.dx_b.unsafe_ptr())

    # 1. GPU settling fills `grads` (= net.weights.grd) with +∂E/∂W.
    PCTrainer[*BLOCKS].compute_grads_only_gpu[BATCH](
        ctx,
        params,
        grads,
        latents,
        mu_eps,
        a_below,
        z_below,
        dx,
        x_in,
        y_target,
        T_infer,
        lr_x,
    )

    # 2. nn2 optimizer consumes net.weights.grd, updates net.weights.val.
    opt.step["gpu", PCModule[*BLOCKS]](net)


def pc_module_train_one_batch_gpu[
    BATCH: Int, OPT: Optimizer, *BLOCKS: PCBlockTrait
](
    ctx: DeviceContext,
    mut net: PCModule[*BLOCKS],
    mut opt: OPT,
    x_in: LayoutTensor[
        DT, Layout.row_major(BATCH, PCSequential[*BLOCKS].IN_DIM), MutAnyOrigin
    ],
    y_target: LayoutTensor[
        DT, Layout.row_major(BATCH, PCSequential[*BLOCKS].OUT_DIM), MutAnyOrigin
    ],
    T_infer: Int,
    lr_x: Scalar[DT],
) raises:
    """Convenience overload: allocate a one-shot workspace per call. Prefer
    the workspace-taking overload in a training loop to avoid per-step
    device allocations."""
    var ws = PCGpuWorkspace[BATCH, *BLOCKS].make(ctx)
    pc_module_train_one_batch_gpu[BATCH](
        ctx, net, opt, ws, x_in, y_target, T_infer, lr_x
    )
    ctx.synchronize()
