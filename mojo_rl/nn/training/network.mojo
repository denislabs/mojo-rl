"""Network: all-static namespace for neural network forward/backward ops.

All CPU and GPU methods are @staticmethod — no stored state.
Use NetworkState to manage params/grads/optimizer state, and
GPUNetworkState for device-side training.

Usage:
    from mojo_rl.nn import seq, Linear, ReLU, Adam
    from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState

    alias M = typeof(seq(Linear[4, 64](), ReLU[64](), Linear[64, 2]()))

    # Create and initialize state separately
    var state = NetworkState[M, Adam]()
    state.initialize[Kaiming[]]()

    # CPU inference (batch=1)
    var p = state.params_view()
    Network[M, Adam].forward[1](obs_t, q_t, p)

    # CPU training step
    var p = state.params_view()
    var g = state.grads_view()
    Network[M, Adam].forward_with_cache[B](input_t, output_t, p, cache_t)
    state.zero_grads()
    Network[M, Adam].backward[B](grad_out_t, grad_in_t, p, cache_t, g)
    state.optimizer_step()

    # Target network soft update
    target_state.soft_update_from(online_state, tau=0.005)

    # GPU training with GPUNetworkState
    var gpu = GPUNetworkState[M, Adam](ctx)
    gpu.upload_from(state, ctx)
    var p = gpu.params_view()
    var g = gpu.grads_view()
    Network[M, Adam].forward_gpu[B](ctx, input_t, output_t, p, ws_buf)
    Network[M, Adam].backward_gpu[B](ctx, dout_t, din_t, p, cache_t, g, ws_buf)
    gpu.optimizer_step(ctx)
    gpu.download_to(state, ctx)
"""

from ..model import Model
from ..model.model import PerfTimerPtr, NULL_PERF
from ..optimizer import Optimizer
from ..constants import dtype

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


struct Network[MODEL: Model, OPTIMIZER: Optimizer]:
    """All-static namespace for network forward/backward operations.

    CPU and GPU methods are @staticmethod — no stored state.
    Target network ops and checkpointing live on NetworkState.

    Parameters:
        MODEL: The model architecture (implements Model trait).
        OPTIMIZER: The optimizer (implements Optimizer trait).
    """

    comptime IN_DIM: Int = Self.MODEL.IN_DIM
    comptime OUT_DIM: Int = Self.MODEL.OUT_DIM
    comptime PARAM_SIZE: Int = Self.MODEL.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.MODEL.CACHE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.MODEL.WORKSPACE_SIZE_PER_SAMPLE

    # =========================================================================
    # CPU Forward Pass
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass without caching (inference / action selection).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [PARAM_SIZE] (e.g. state.params_view()).
        """
        Self.MODEL.forward[BATCH](input, output, params)

    @staticmethod
    def forward_with_cache[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass with caching (training — cache needed for backward).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [PARAM_SIZE].
            cache: Cache tensor [BATCH, CACHE_SIZE] (written).
        """
        Self.MODEL.forward[BATCH](input, output, params, cache)

    # =========================================================================
    # CPU Backward Pass
    # =========================================================================

    @staticmethod
    def backward[
        BATCH: Int
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass: accumulate param grads and compute grad_input.

        Call state.zero_grads() before if you want fresh gradients.

        Args:
            grad_output: Gradient of the loss w.r.t. output [BATCH, OUT_DIM].
            grad_input: Gradient of the loss w.r.t. input [BATCH, IN_DIM] (written).
            params: Model parameters [PARAM_SIZE].
            cache: Cache from forward_with_cache [BATCH, CACHE_SIZE].
            grads: Gradient accumulator [PARAM_SIZE] (e.g. state.grads_view()).
        """
        Self.MODEL.backward[BATCH](
            grad_output, grad_input, params, cache, grads
        )

    # =========================================================================
    # GPU Forward / Backward
    # All accept LayoutTensors over device memory (from std.gpuNetworkState views).
    # workspace_buf stays as DeviceBuffer (untyped, variable size).
    # =========================================================================

    @staticmethod
    def forward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        workspace_buf: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward pass without caching.

        Args:
            ctx: GPU device context.
            input: Device input tensor [BATCH, IN_DIM].
            output: Device output tensor [BATCH, OUT_DIM] (written).
            params: Device params tensor [PARAM_SIZE] (e.g. gpu.params_view()).
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
            perf: Optional profiling timer pointer (null = no profiling).
            perf_slot: Base slot index for per-layer timing.
        """
        Self.MODEL.forward_gpu_no_cache[BATCH](
            ctx, output, input, params, workspace_buf, perf, perf_slot
        )

    @staticmethod
    def forward_gpu_with_cache[
        BATCH: Int
    ](
        ctx: DeviceContext,
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ],
        workspace_buf: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward pass with caching (needed for backward).

        Args:
            ctx: GPU device context.
            input: Device input tensor [BATCH, IN_DIM].
            output: Device output tensor [BATCH, OUT_DIM] (written).
            params: Device params tensor [PARAM_SIZE].
            cache: Device cache tensor [BATCH, CACHE_SIZE] (written).
            workspace_buf: Pre-allocated workspace.
            perf: Optional profiling timer pointer (null = no profiling).
            perf_slot: Base slot index for per-layer timing.
        """
        Self.MODEL.forward_gpu[BATCH](
            ctx, output, input, params, cache, workspace_buf, perf, perf_slot
        )

    @staticmethod
    def forward_gpu_with_cache_on_stream[
        BATCH: Int
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass with caching on stream.

        Args:
            ctx: GPU device context.
            stream: Device stream for kernel dispatch.
            input: Device input tensor [BATCH, IN_DIM].
            output: Device output tensor [BATCH, OUT_DIM] (written).
            params: Device params tensor [PARAM_SIZE].
            cache: Device cache tensor [BATCH, CACHE_SIZE] (written).
            workspace_buf: Pre-allocated workspace.
        """
        Self.MODEL.forward_gpu_on_stream[BATCH](
            ctx, stream, output, input, params, cache, workspace_buf
        )

    @staticmethod
    def backward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        workspace_buf: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU backward pass (accumulates into grads).

        Args:
            ctx: GPU device context.
            grad_output: Gradient of the loss w.r.t. output [BATCH, OUT_DIM].
            grad_input: Gradient of the loss w.r.t. input [BATCH, IN_DIM] (written).
            params: Device params tensor [PARAM_SIZE].
            cache: Cache from forward_gpu_with_cache [BATCH, CACHE_SIZE].
            grads: Device grads tensor [PARAM_SIZE] (e.g. gpu.grads_view()).
            workspace_buf: Pre-allocated workspace.
            perf: Optional profiling timer pointer (null = no profiling).
            perf_slot: Base slot index for per-layer timing.
        """
        Self.MODEL.backward_gpu[BATCH](
            ctx,
            grad_input,
            grad_output,
            params,
            cache,
            grads,
            workspace_buf,
            perf,
            perf_slot,
        )

    @staticmethod
    def backward_gpu_on_stream[
        BATCH: Int
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward using stream dispatch."""
        Self.MODEL.backward_gpu_on_stream[BATCH](
            ctx,
            stream,
            grad_input,
            grad_output,
            params,
            cache,
            grads,
            workspace_buf,
        )
