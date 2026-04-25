from ..constants import dtype as default_dtype
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.memory import UnsafePointer

# Opaque pointer for optional per-layer profiling timer.
# Null (address=0) = no profiling. Non-null = points to a PerfTimer[True].
# Sequential uses this to inject sync+timing around each layer.
comptime PerfTimerPtr = UnsafePointer[NoneType, MutAnyOrigin]
comptime NULL_PERF = PerfTimerPtr(unsafe_from_address=0)


trait Model(Movable & ImplicitlyCopyable):
    """Base trait for neural network modules (stateless).

    Models are stateless - they describe the computation graph but don't store
    weights or gradients. All state (params, grads, cache) is managed externally
    as LayoutTensor views for zero-copy composition.

    Compile-time constants:
    - IN_DIM: Input dimension per sample
    - OUT_DIM: Output dimension per sample
    - PARAM_SIZE: Total number of parameters (e.g., W + b for Linear)
    - CACHE_SIZE: Elements cached per sample during forward (for backward pass)
    - WORKSPACE_SIZE_PER_SAMPLE: GPU workspace needed per sample for intermediate buffers.
      For leaf layers (Linear, ReLU, etc.) this is 0.
      For Sequential, this includes intermediate activation buffers.
      Total workspace = BATCH * WORKSPACE_SIZE_PER_SAMPLE.

    All CPU and GPU tensors use LayoutTensor for consistent zero-copy views:
    - input/output: [BATCH, DIM] layout
    - params/grads: [PARAM_SIZE] layout (1D)
    - cache: [BATCH, CACHE_SIZE] layout

    GPU workspace is kept as DeviceBuffer (size may be 0 for leaf layers,
    allocated as max(1, BATCH * WORKSPACE_SIZE_PER_SAMPLE) by the caller).
    """

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int
    comptime CACHE_SIZE: Int
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int

    # Persistent non-trainable state (BN running stats, RNG counters, etc.).
    # Lives on GPU between forward/backward calls. Default 0 — most layers
    # don't need it. See docs/STATE_SIZE_DESIGN.md.
    comptime STATE_SIZE: Int = 0

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize parameters with per-layer fan dimensions.

        Leaf layers call INIT.init with their own IN_DIM/OUT_DIM.
        Sequential overrides to iterate through layers.
        Activation-only layers (PARAM_SIZE=0) are no-ops.
        """
        ...

    @staticmethod
    def zero_biases[dtype: DType = DType.float32](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Opt-in bias zero-init — overwrite BiasAdd slots with zeros.

        Default = no-op (for activations, normalizations, and any Model that
        doesn't expose a MatMul+BiasAdd pattern). Composites (Sequential,
        Parallel, AutoFused) override to recurse or zero their BiasAdd slots.
        Intended for use AFTER `initialize_params` to match Keras/TF
        `bias_initializer='zeros'` convention. Currently invoked only by
        MBPO — other agents keep their default non-zero biases.
        """
        pass

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize persistent non-trainable state (BN running stats, RNG
        counters, etc.).

        Default = no-op: layers with STATE_SIZE=0 get a zero-length tensor
        and don't need to do anything. Stateful layers (BN, Dropout,
        NoisyLinear) override; composites (Sequential, Parallel, etc.)
        override to recurse.
        """
        pass

    # =========================================================================
    # Forward passes
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int, dtype: DType = DType.float32
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass with caching (for training).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [PARAM_SIZE].
            state: Persistent non-trainable state [STATE_SIZE] (BN running
                stats, RNG counters; zero-length for most layers).
            cache: Cache buffer [BATCH, CACHE_SIZE] for backward pass (written).
        """
        ...

    @staticmethod
    def forward[
        BATCH: Int, dtype: DType = DType.float32
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass without caching (for inference).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [PARAM_SIZE].
            state: Persistent non-trainable state [STATE_SIZE] (read-only in
                inference; zero-length for most layers).
        """
        ...

    # =========================================================================
    # Backward pass
    # =========================================================================

    @staticmethod
    def backward[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass: compute grad_input and accumulate parameter gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output [BATCH, OUT_DIM].
            grad_input: Gradient of loss w.r.t. input [BATCH, IN_DIM] (written).
            params: Model parameters [PARAM_SIZE].
            state: Persistent non-trainable state [STATE_SIZE] (read-only;
                zero-length for most layers).
            cache: Cache from forward pass [BATCH, CACHE_SIZE].
            grads: Parameter gradients [PARAM_SIZE] (accumulated, not overwritten).
        """
        ...

    # =========================================================================
    # GPU forward passes
    # =========================================================================
    # Shaped tensors (input, output, params, cache) are passed as LayoutTensor
    # for a uniform API matching the CPU interface.
    # workspace is kept as DeviceBuffer because its size may be 0 for leaf
    # layers (allocated as max(1, BATCH * WORKSPACE_SIZE_PER_SAMPLE) by caller).
    # =========================================================================

    @staticmethod
    def forward_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward pass with caching (for training).

        Args:
            ctx: GPU device context.
            output: Output [BATCH, OUT_DIM] (written).
            input: Input [BATCH, IN_DIM].
            params: Parameters [PARAM_SIZE].
            state: Persistent non-trainable state [STATE_SIZE] (BN running
                stats, RNG counters; zero-length for most layers).
            cache: Cache [BATCH, CACHE_SIZE] (written).
            workspace: Pre-allocated workspace for Sequential intermediate buffers.
            perf: Optional profiling timer pointer (null = no profiling).
            perf_slot: Base slot index in the timer for per-layer timing.
        """
        ...

    @staticmethod
    def forward_gpu_no_cache[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward pass without caching (for inference).

        Args:
            ctx: GPU device context.
            output: Output [BATCH, OUT_DIM] (written).
            input: Input [BATCH, IN_DIM].
            params: Parameters [PARAM_SIZE].
            state: Persistent non-trainable state [STATE_SIZE] (read-only;
                zero-length for most layers).
            workspace: Pre-allocated workspace for Sequential intermediate buffers.
            perf: Optional profiling timer pointer (null = no profiling).
            perf_slot: Base slot index in the timer for per-layer timing.
        """
        ...

    # =========================================================================
    # GPU forward pass (no cache) — on DeviceStream
    # =========================================================================

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass without caching, enqueued on a DeviceStream.

        Default: delegates to forward_gpu_no_cache (default stream).
        Override for actual stream parallelism.
        """
        ...

    # =========================================================================
    # GPU backward pass
    # =========================================================================

    @staticmethod
    def backward_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU backward pass.

        Args:
            ctx: GPU device context.
            grad_input: Gradient w.r.t. input [BATCH, IN_DIM] (written).
            grad_output: Gradient w.r.t. output [BATCH, OUT_DIM].
            params: Parameters [PARAM_SIZE].
            state: Persistent non-trainable state [STATE_SIZE] (read-only in
                training-mode backward; zero-length for most layers).
            cache: Cache from forward pass [BATCH, CACHE_SIZE].
            grads: Parameter gradients [PARAM_SIZE] (accumulated).
            workspace: Pre-allocated workspace for Sequential intermediate buffers.
            perf: Optional profiling timer pointer (null = no profiling).
            perf_slot: Base slot index in the timer for per-layer timing.
        """
        ...

    # =========================================================================
    # GPU forward + backward (inference-mode with cache)
    # =========================================================================
    # Used when running an evaluation/RL forward through a model containing BN
    # but we want BN to use its frozen running stats instead of batch stats —
    # the paper-faithful behavior for OFENet inside REDQ-OFE updates. Default
    # implementations delegate to the training-mode kernels so non-BN layers
    # need no override; BN variants (and their fused composites) override to
    # use running stats and skip EMA updates / param-grad writes.
    # =========================================================================

    @staticmethod
    def forward_gpu_inference_with_cache[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU inference-mode forward, populates `cache` for inference-mode backward.

        Default = training-mode `forward_gpu`. BatchNorm1D / BatchNorm2D
        override to use running stats from `state` (no batch-stat reduction,
        no EMA update on `state`). Non-BN leaf layers (Linear, ReLU,
        LayerNorm, ...) inherit the default since their training kernel has
        no batch-stat dependency. Sequential and combinators override to
        recurse into children's inference variants.

        Caveat — fused BN composites (`Conv2DBatchNormReLU`,
        `LinearBatchNormReLU`, `ResBlockConv2DBN`) currently inherit this
        default, so calling inference-mode through them silently uses
        batch-stat reductions and EMA updates inside their fused BN kernel.
        REDQ-OFE doesn't hit this path (it uses raw `BatchNorm1D`); a proper
        inference-mode override for the fused composites is a follow-up
        once a use case requires it.
        """
        Self.forward_gpu[BATCH, dtype](
            ctx, output, input, params, state, cache, workspace, perf, perf_slot
        )

    @staticmethod
    def backward_gpu_inference[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU inference-mode backward (consumes the inference-mode cache).

        Default = training-mode `backward_gpu`. BN variants override to apply
        the simpler `dx = γ·inv_std_r·dy` formula and skip writes to
        `grad_params` (BN params are conceptually frozen in inference mode;
        the caller — e.g. REDQ-OFE — zeros their gradient slots).
        """
        Self.backward_gpu[BATCH, dtype](
            ctx, grad_input, grad_output, params, state, cache, grads, workspace, perf, perf_slot
        )
