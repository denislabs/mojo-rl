"""GPUNetworkState: GPU-side counterpart of NetworkState.

Holds DeviceBuffers for params, grads, and optimizer state, plus pinned
HostBuffers for efficient CPU↔GPU transfers. Provides the same zero-copy
LayoutTensor view interface as NetworkState but over device memory.

The workspace buffer is intentionally NOT included here — it is batch-size
dependent (BATCH * MODEL.WORKSPACE_SIZE_PER_SAMPLE) and is owned by the
training loop or RL agent, which knows the batch size.

Usage:
    # At agent / trainer init
    var gpu = GPUNetworkState[model_type, Adam](ctx)
    gpu.upload_from(network.state, ctx)

    # Training loop
    gpu.zero_grads(ctx)
    Network.forward_gpu[BATCH](ctx, input_t, output_t, gpu.params_view(), ws_buf)
    Network.backward_gpu[BATCH](ctx, grad_out_t, grad_in_t,
                                gpu.params_view(), cache_t, gpu.grads_view(), ws_buf)
    gpu.optimizer_step(ctx)

    # Periodic CPU sync
    gpu.download_to(network.state, ctx)
    target.soft_update_from(network, tau=0.005)
"""

from ..model import Model
from ..optimizer import Optimizer
from ..constants import dtype, TPB
from .network_state import NetworkState

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu import block_dim, block_idx, thread_idx


@always_inline
fn soft_update_kernel[
    dtype: DType,
    SIZE: Int,
](
    target: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    tau: Scalar[dtype],
):
    """Soft update: target = tau * source + (1 - tau) * target.

    Used for target network updates in DQN, DDPG, TD3, SAC.

    Args:
        target: Target network parameters (updated in-place).
        source: Source (online) network parameters.
        tau: Blending factor (typically 0.001 - 0.01).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return

    var src_val = source[i]
    var tgt_val = target[i]
    target[i] = tau * src_val + (Scalar[dtype](1.0) - tau) * tgt_val


struct GPUNetworkState[MODEL: Model, OPTIMIZER: Optimizer](
    ImplicitlyCopyable, Movable
):
    """GPU-side network state using pre-allocated DeviceBuffers.

    Mirrors NetworkState but lives on the GPU device. LayoutTensor views
    over device memory let GPU kernels access params/grads/state directly.

    Parameters:
        MODEL: The model architecture (implements Model trait).
        OPTIMIZER: The optimizer (implements Optimizer trait).
    """

    comptime PARAM_SIZE: Int = Self.MODEL.PARAM_SIZE
    comptime STATE_SIZE: Int = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM

    var params_buf: DeviceBuffer[dtype]  # device: model weights
    var grads_buf: DeviceBuffer[dtype]  # device: parameter gradients
    var state_buf: DeviceBuffer[
        dtype
    ]  # device: optimizer state (e.g. Adam m/v)
    var params_host: HostBuffer[
        dtype
    ]  # pinned host mirror — fast DMA for params
    var state_host: HostBuffer[dtype]  # pinned host mirror — fast DMA for state
    var step_num: Int
    var lr_scale: Float64

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all device and pinned host buffers.

        Args:
            ctx: GPU device context (determines which device owns the buffers).
        """
        self.step_num = 0
        self.lr_scale = 1.0
        self.params_buf = ctx.enqueue_create_buffer[dtype](Self.PARAM_SIZE)
        self.grads_buf = ctx.enqueue_create_buffer[dtype](Self.PARAM_SIZE)
        self.state_buf = ctx.enqueue_create_buffer[dtype](Self.STATE_SIZE)
        self.params_host = ctx.enqueue_create_host_buffer[dtype](
            Self.PARAM_SIZE
        )
        self.state_host = ctx.enqueue_create_host_buffer[dtype](Self.STATE_SIZE)
        # Zero-initialize device buffers
        ctx.enqueue_memset(self.params_buf, 0)
        ctx.enqueue_memset(self.grads_buf, 0)
        ctx.enqueue_memset(self.state_buf, 0)

    fn __init__(out self, *, copy: Self):
        self.step_num = copy.step_num
        self.lr_scale = copy.lr_scale
        self.params_buf = copy.params_buf.copy()
        self.grads_buf = copy.grads_buf.copy()
        self.state_buf = copy.state_buf.copy()
        self.params_host = copy.params_host.copy()
        self.state_host = copy.state_host.copy()

    fn __init__(out self, *, deinit take: Self):
        self.step_num = take.step_num
        self.lr_scale = take.lr_scale
        self.params_buf = take.params_buf^
        self.grads_buf = take.grads_buf^
        self.state_buf = take.state_buf^
        self.params_host = take.params_host^
        self.state_host = take.state_host^

    # =========================================================================
    # LayoutTensor Views over device memory (zero-copy)
    # =========================================================================

    fn params_view(
        self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """LayoutTensor view over device params buffer."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ](self.params_buf.unsafe_ptr())

    fn grads_view(
        self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """LayoutTensor view over device grads buffer."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ](self.grads_buf.unsafe_ptr())

    fn state_view(
        self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM),
        MutAnyOrigin,
    ]:
        """LayoutTensor view over device optimizer state buffer."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM),
            MutAnyOrigin,
        ](self.state_buf.unsafe_ptr())

    # =========================================================================
    # Core GPU Operations
    # =========================================================================

    fn zero_grads(self, ctx: DeviceContext) raises:
        """Zero the gradients buffer on device (async)."""
        ctx.enqueue_memset(self.grads_buf, 0)

    fn set_lr_scale(mut self, scale: Float64):
        """Set the LR multiplier applied at each optimizer step.

        Args:
            scale: Multiplier applied to the compile-time base LR (default 1.0).
        """
        self.lr_scale = scale

    fn optimizer_step(mut self, ctx: DeviceContext) raises:
        """One GPU optimizer step + increment step_num.

        Applies self.lr_scale to the base LR (set via set_lr_scale()).
        Creates lvalue views (params, state are mut in step_gpu).

        Args:
            ctx: GPU device context.
        """
        self.step_num += 1
        var p = self.params_view()
        var s = self.state_view()
        Self.OPTIMIZER.step_gpu[Self.PARAM_SIZE](
            ctx, p, self.grads_view(), s, self.step_num, self.lr_scale
        )

    fn soft_update_from_gpu(
        self,
        source: Self,
        tau: Float64,
        ctx: DeviceContext,
    ) raises:
        """Soft update on GPU: self.params = tau * source.params + (1 - tau) * self.params.

        Runs entirely on device — no CPU synchronization required.

        Args:
            source: Source (online) network state on GPU.
            tau: Blending factor (typically 0.001 – 0.01).
            ctx: GPU device context.
        """
        comptime PARAM_SIZE = Self.PARAM_SIZE
        comptime PARAM_BLOCKS = (PARAM_SIZE + TPB - 1) // TPB
        var target_t = self.params_view()
        var source_t = source.params_view()
        var tau_s = Scalar[dtype](tau)

        @always_inline
        fn soft_update_wrapper(
            tgt: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            t: Scalar[dtype],
        ):
            soft_update_kernel[dtype, PARAM_SIZE](tgt, src, t)

        ctx.enqueue_function[soft_update_wrapper, soft_update_wrapper](
            target_t,
            source_t,
            tau_s,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # CPU ↔ GPU Sync
    # =========================================================================

    fn upload_from(
        mut self,
        cpu: NetworkState[Self.MODEL, Self.OPTIMIZER],
        ctx: DeviceContext,
    ) raises:
        """Upload CPU params and optimizer state to device (async).

        Copies via pinned host buffers for optimal DMA throughput.
        No synchronize() needed — GPU commands are sequenced by the queue.

        Args:
            cpu: Source CPU network state.
            ctx: GPU device context.
        """
        for i in range(Self.PARAM_SIZE):
            self.params_host[i] = cpu.params[i]
        ctx.enqueue_copy(self.params_buf, self.params_host)

        for i in range(Self.STATE_SIZE):
            self.state_host[i] = cpu.optimizer_state[i]
        ctx.enqueue_copy(self.state_buf, self.state_host)

        self.step_num = cpu.step_num

    fn download_to(
        mut self,
        mut cpu: NetworkState[Self.MODEL, Self.OPTIMIZER],
        ctx: DeviceContext,
    ) raises:
        """Download device params and optimizer state to CPU (synchronizes).

        Args:
            cpu: Destination CPU network state (modified in-place).
            ctx: GPU device context.
        """
        ctx.enqueue_copy(self.params_host, self.params_buf)
        ctx.enqueue_copy(self.state_host, self.state_buf)
        ctx.synchronize()

        for i in range(Self.PARAM_SIZE):
            cpu.params[i] = self.params_host[i]
        for i in range(Self.STATE_SIZE):
            cpu.optimizer_state[i] = self.state_host[i]
        cpu.step_num = self.step_num
