from ..constants import dtype
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.builtin.variadics import Variadic
from mojo_rl.deep_agents.core.perf_timer import PerfTimer


# GPU matmul requires 16-byte alignment = 4 float32 elements
@always_inline
def _seq_align4(x: Int) -> Int:
    """Round up to next multiple of 4 for GPU alignment."""
    return (x + 3) & ~3


# =============================================================================
# Variadic Sequential Container
# =============================================================================
#
# Sequential[*LAYERS: Model] handles any number of layers directly using
# Variadic.types + comptime for with index-based access for compile-time
# iteration over heterogeneous type parameter packs.
#
# Usage:
#     var model = seq(Linear[2, 64](), ReLU[64](), Linear[64, 1]())
#     # Creates Sequential[Linear[2, 64], ReLU[64], Linear[64, 1]]
#
# Cache layout: [L0 cache | L1 cache | ... | L_{N-1} cache]
# GPU workspace layout: [inter_bufs | L0 ws | L1 ws | ... | L_{N-1} ws]
# =============================================================================


@fieldwise_init
struct Sequential[*LAYERS: Model](Model):
    """Variadic sequential container for N layers.

    Composes N layers where layer[i].OUT_DIM == layer[i+1].IN_DIM.
    Uses Variadic.types + comptime for to iterate at compile time.
    """

    comptime model_types = Variadic.types[T=Model, *Self.LAYERS]
    comptime N = Variadic.size(Self.model_types)

    comptime IN_DIM: Int = Self.model_types[0].IN_DIM
    comptime OUT_DIM: Int = Self.model_types[Self.N - 1].OUT_DIM

    # --- Sum helpers ---

    @staticmethod
    def _sum_param_size() -> Int:
        """Total param size with alignment padding between layers.

        Each layer except the last is padded to 4-element alignment so that
        the next layer's params start at a GPU-aligned address. This prevents
        CUDA_ERROR_MISALIGNED_ADDRESS in matmul when Sequential composes
        models with odd PARAM_SIZE (e.g., Linear[256,6] has PARAM_SIZE=1542).
        """
        var total = 0

        comptime for i in range(Self.N - 1):
            total += _seq_align4(Self.model_types[i].PARAM_SIZE)
        # Last layer: no padding needed after it
        total += Self.model_types[Self.N - 1].PARAM_SIZE
        return total

    @staticmethod
    def _sum_cache_size() -> Int:
        var total = 0

        comptime for i in range(Self.N):
            total += Self.model_types[i].CACHE_SIZE
        return total

    @staticmethod
    def _total_inter() -> Int:
        """Per-sample intermediate buffer size (sum of OUT_DIM for layers 0..N-2).
        """
        var total = 0

        comptime for i in range(Self.N - 1):
            total += Self.model_types[i].OUT_DIM
        return total

    @staticmethod
    def _sum_ws() -> Int:
        var total = 0

        comptime for i in range(Self.N):
            total += Self.model_types[i].WORKSPACE_SIZE_PER_SAMPLE
        return total

    comptime PARAM_SIZE: Int = Self._sum_param_size()
    comptime CACHE_SIZE: Int = Self._sum_cache_size()
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self._total_inter() + Self._sum_ws()

    # --- Offset helpers (all per-sample) ---

    @staticmethod
    def _param_offset[idx: Int]() -> Int:
        """Aligned param offset for layer idx.

        Each preceding layer's PARAM_SIZE is rounded up to 4-element alignment.
        """
        var total = 0

        comptime for j in range(idx):
            total += _seq_align4(Self.model_types[j].PARAM_SIZE)
        return total

    @staticmethod
    def _cache_offset[idx: Int]() -> Int:
        var total = 0

        comptime for j in range(idx):
            total += Self.model_types[j].CACHE_SIZE
        return total

    @staticmethod
    def _inter_offset[idx: Int]() -> Int:
        """Offset of intermediate slot idx (per sample)."""
        var total = 0

        comptime for j in range(idx):
            total += Self.model_types[j].OUT_DIM
        return total

    @staticmethod
    def _ws_layer_offset[idx: Int]() -> Int:
        """Offset for layer idx's workspace (per sample), after all inter buffers.
        """
        var total = Self._total_inter()

        comptime for j in range(idx):
            total += Self.model_types[j].WORKSPACE_SIZE_PER_SAMPLE
        return total

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize each layer with its own fan_in/fan_out dimensions.

        Zeros alignment padding regions between layers.
        """
        # Zero the entire buffer first (covers padding between layers)
        for i in range(Self.PARAM_SIZE):
            params.ptr[i] = Scalar[dtype](0.0)

        comptime for i in range(Self.N):
            comptime if Self.model_types[i].PARAM_SIZE > 0:
                var layer_params = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                Self.model_types[i].initialize_params[INIT, dtype](layer_params)

    # =========================================================================
    # CPU Forward (with cache)
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
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        comptime if Self.N == 1:
            var in_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].IN_DIM),
                MutAnyOrigin,
            ](input.ptr)
            var out_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].OUT_DIM),
                MutAnyOrigin,
            ](output.ptr)
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            Self.model_types[0].forward[BATCH, dtype](in_v, out_v, p_v, c_v)
        else:
            # Flat intermediate buffer for all N-1 inter-layer activations
            var inter_storage = List[Scalar[dtype]](
                capacity=BATCH * Self._total_inter()
            )
            for _ in range(BATCH * Self._total_inter()):
                inter_storage.append(0)
            var inter_ptr = inter_storage.unsafe_ptr()

            comptime for i in range(Self.N):
                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[i].CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())

                comptime if i == 0:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](input.ptr)
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr)
                    Self.model_types[i].forward[BATCH, dtype](
                        li_in, li_out, li_p, li_c
                    )
                elif i == Self.N - 1:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](output.ptr)
                    Self.model_types[i].forward[BATCH, dtype](
                        li_in, li_out, li_p, li_c
                    )
                else:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i]())
                    Self.model_types[i].forward[BATCH, dtype](
                        li_in, li_out, li_p, li_c
                    )

    # =========================================================================
    # CPU Forward (no cache)
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
    ):
        comptime if Self.N == 1:
            var in_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].IN_DIM),
                MutAnyOrigin,
            ](input.ptr)
            var out_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].OUT_DIM),
                MutAnyOrigin,
            ](output.ptr)
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            Self.model_types[0].forward[BATCH, dtype](in_v, out_v, p_v)
        else:
            var inter_storage = List[Scalar[dtype]](
                capacity=BATCH * Self._total_inter()
            )
            for _ in range(BATCH * Self._total_inter()):
                inter_storage.append(0)
            var inter_ptr = inter_storage.unsafe_ptr()

            comptime for i in range(Self.N):
                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())

                comptime if i == 0:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](input.ptr)
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr)
                    Self.model_types[i].forward[BATCH, dtype](li_in, li_out, li_p)
                elif i == Self.N - 1:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](output.ptr)
                    Self.model_types[i].forward[BATCH, dtype](li_in, li_out, li_p)
                else:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i]())
                    Self.model_types[i].forward[BATCH, dtype](li_in, li_out, li_p)

    # =========================================================================
    # CPU Backward
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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        comptime if Self.N == 1:
            var go_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].OUT_DIM),
                MutAnyOrigin,
            ](grad_output.ptr)
            var gi_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].IN_DIM),
                MutAnyOrigin,
            ](grad_input.ptr)
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var g_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr)
            Self.model_types[0].backward[BATCH, dtype](go_v, gi_v, p_v, c_v, g_v)
        else:
            # Gradient intermediate buffer (same layout as forward inter)
            var grad_inter_storage = List[Scalar[dtype]](
                capacity=BATCH * Self._total_inter()
            )
            for _ in range(BATCH * Self._total_inter()):
                grad_inter_storage.append(0)
            var gi_ptr = grad_inter_storage.unsafe_ptr()

            # Reverse iteration
            comptime for _ri in range(Self.N):
                comptime i = Self.N - 1 - _ri

                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[i].CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var li_g = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](grads.ptr + Self._param_offset[i]())

                comptime if i == Self.N - 1:
                    # Last layer: Sequential grad_output -> grad_inter[i-1]
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](grad_output.ptr)
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.model_types[i].backward[BATCH, dtype](
                        li_go, li_gi, li_p, li_c, li_g
                    )
                elif i == 0:
                    # First layer: grad_inter[0] -> Sequential grad_input
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](gi_ptr)
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](grad_input.ptr)
                    Self.model_types[i].backward[BATCH, dtype](
                        li_go, li_gi, li_p, li_c, li_g
                    )
                else:
                    # Middle: grad_inter[i] -> grad_inter[i-1]
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i]())
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.model_types[i].backward[BATCH, dtype](
                        li_go, li_gi, li_p, li_c, li_g
                    )

    # =========================================================================
    # GPU Forward (with cache)
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
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward pass using pre-allocated workspace.

        Workspace layout: [inter_bufs (N-1 buffers) | L0 ws | L1 ws | ... | L_{N-1} ws]

        When perf is non-null, injects ctx.synchronize() + timing around each
        layer using perf_slot + i as the slot index for layer i.
        """

        # Save caller's _mark so L3 timing doesn't clobber L2's mark
        var saved_mark: UInt = 0
        if perf:
            saved_mark = perf.bitcast[PerfTimer[True]]()[]._mark

        comptime if Self.N == 1:
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var out_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[0].OUT_DIM),
                    MutAnyOrigin,
                ]
            ](output)
            var in_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[0].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            if perf:
                perf.bitcast[PerfTimer[True]]()[].sync_and_mark(ctx)
            Self.model_types[0].forward_gpu[BATCH, dtype](
                ctx, out_rb, in_rb, p_v, c_v, workspace
            )
            if perf:
                perf.bitcast[PerfTimer[True]]()[].sync_and_accumulate(
                    perf_slot, ctx
                )
        else:
            var ws_ptr = workspace.unsafe_ptr()

            comptime for i in range(Self.N):
                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[i].CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var li_ws_size = (
                    BATCH * Self.model_types[i].WORKSPACE_SIZE_PER_SAMPLE
                )
                var li_ws = DeviceBuffer[dtype](
                    ctx,
                    ws_ptr + BATCH * Self._ws_layer_offset[i](),
                    li_ws_size if li_ws_size > 0 else 1,
                    owning=False,
                )

                if perf:
                    perf.bitcast[PerfTimer[True]]()[].sync_and_mark(ctx)

                comptime if i == 0:
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr)
                    var in_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                            MutAnyOrigin,
                        ]
                    ](input)
                    Self.model_types[i].forward_gpu[BATCH, dtype](
                        ctx, inter_out, in_rb, li_p, li_c, li_ws
                    )
                elif i == Self.N - 1:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var out_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(
                                BATCH, Self.model_types[i].OUT_DIM
                            ),
                            MutAnyOrigin,
                        ]
                    ](output)
                    Self.model_types[i].forward_gpu[BATCH, dtype](
                        ctx, out_rb, inter_in, li_p, li_c, li_ws
                    )
                else:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i]())
                    Self.model_types[i].forward_gpu[BATCH, dtype](
                        ctx, inter_out, inter_in, li_p, li_c, li_ws
                    )

                if perf:
                    perf.bitcast[PerfTimer[True]]()[].sync_and_accumulate(
                        perf_slot + i, ctx
                    )

        # Restore caller's _mark so L2 timing measures the full span
        if perf:
            perf.bitcast[PerfTimer[True]]()[]._mark = saved_mark

    # =========================================================================
    # GPU Forward (no cache)
    # =========================================================================

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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        # Save caller's _mark so L3 timing doesn't clobber L2's mark
        var saved_mark: UInt = 0
        if perf:
            saved_mark = perf.bitcast[PerfTimer[True]]()[]._mark

        comptime if Self.N == 1:
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var out_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[0].OUT_DIM),
                    MutAnyOrigin,
                ]
            ](output)
            var in_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[0].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            if perf:
                perf.bitcast[PerfTimer[True]]()[].sync_and_mark(ctx)
            Self.model_types[0].forward_gpu_no_cache[BATCH, dtype](
                ctx, out_rb, in_rb, p_v, workspace
            )
            if perf:
                perf.bitcast[PerfTimer[True]]()[].sync_and_accumulate(
                    perf_slot, ctx
                )
        else:
            var ws_ptr = workspace.unsafe_ptr()

            comptime for i in range(Self.N):
                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_ws_size = (
                    BATCH * Self.model_types[i].WORKSPACE_SIZE_PER_SAMPLE
                )
                var li_ws = DeviceBuffer[dtype](
                    ctx,
                    ws_ptr + BATCH * Self._ws_layer_offset[i](),
                    li_ws_size if li_ws_size > 0 else 1,
                    owning=False,
                )

                if perf:
                    perf.bitcast[PerfTimer[True]]()[].sync_and_mark(ctx)

                comptime if i == 0:
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr)
                    var in_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                            MutAnyOrigin,
                        ]
                    ](input)
                    Self.model_types[i].forward_gpu_no_cache[BATCH, dtype](
                        ctx, inter_out, in_rb, li_p, li_ws
                    )
                elif i == Self.N - 1:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var out_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(
                                BATCH, Self.model_types[i].OUT_DIM
                            ),
                            MutAnyOrigin,
                        ]
                    ](output)
                    Self.model_types[i].forward_gpu_no_cache[BATCH, dtype](
                        ctx, out_rb, inter_in, li_p, li_ws
                    )
                else:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i]())
                    Self.model_types[i].forward_gpu_no_cache[BATCH, dtype](
                        ctx, inter_out, inter_in, li_p, li_ws
                    )

                if perf:
                    perf.bitcast[PerfTimer[True]]()[].sync_and_accumulate(
                        perf_slot + i, ctx
                    )

        # Restore caller's _mark so L2 timing measures the full span
        if perf:
            perf.bitcast[PerfTimer[True]]()[]._mark = saved_mark

    # =========================================================================
    # GPU Forward (no cache) — on DeviceStream
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
        workspace: DeviceBuffer[dtype],
    ) raises:
        comptime if Self.N == 1:
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var out_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[0].OUT_DIM),
                    MutAnyOrigin,
                ]
            ](output)
            var in_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[0].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            Self.model_types[0].forward_gpu_no_cache_on_stream[BATCH, dtype](
                ctx, stream, out_rb, in_rb, p_v, workspace
            )
        else:
            var ws_ptr = workspace.unsafe_ptr()

            comptime for i in range(Self.N):
                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_ws_size = (
                    BATCH * Self.model_types[i].WORKSPACE_SIZE_PER_SAMPLE
                )
                var li_ws = DeviceBuffer[dtype](
                    ctx,
                    ws_ptr + BATCH * Self._ws_layer_offset[i](),
                    li_ws_size if li_ws_size > 0 else 1,
                    owning=False,
                )

                comptime if i == 0:
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr)
                    var in_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                            MutAnyOrigin,
                        ]
                    ](input)
                    Self.model_types[i].forward_gpu_no_cache_on_stream[BATCH, dtype](
                        ctx, stream, inter_out, in_rb, li_p, li_ws
                    )
                elif i == Self.N - 1:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var out_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(
                                BATCH, Self.model_types[i].OUT_DIM
                            ),
                            MutAnyOrigin,
                        ]
                    ](output)
                    Self.model_types[i].forward_gpu_no_cache_on_stream[BATCH, dtype](
                        ctx, stream, out_rb, inter_in, li_p, li_ws
                    )
                else:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i]())
                    Self.model_types[i].forward_gpu_no_cache_on_stream[BATCH, dtype](
                        ctx, stream, inter_out, inter_in, li_p, li_ws
                    )

    # =========================================================================
    # GPU Backward
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
        """GPU backward pass. Workspace inter region reused for gradient intermediates.

        When perf is non-null, uses perf_slot + _ri (reverse iteration index)
        as slot for each layer, matching register_backward_slots order.
        """

        # Save caller's _mark so L3 timing doesn't clobber L2's mark
        var saved_mark: UInt = 0
        if perf:
            saved_mark = perf.bitcast[PerfTimer[True]]()[]._mark

        comptime if Self.N == 1:
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.model_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var g_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.model_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr)
            var gi_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[0].IN_DIM),
                    MutAnyOrigin,
                ]
            ](grad_input)
            var go_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[0].OUT_DIM),
                    MutAnyOrigin,
                ]
            ](grad_output)
            if perf:
                perf.bitcast[PerfTimer[True]]()[].sync_and_mark(ctx)
            Self.model_types[0].backward_gpu[BATCH, dtype](
                ctx, gi_rb, go_rb, p_v, c_v, g_v, workspace
            )
            if perf:
                perf.bitcast[PerfTimer[True]]()[].sync_and_accumulate(
                    perf_slot, ctx
                )
        else:
            var ws_ptr = workspace.unsafe_ptr()

            # Reverse iteration
            comptime for _ri in range(Self.N):
                comptime i = Self.N - 1 - _ri

                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.model_types[i].CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var li_g = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](grads.ptr + Self._param_offset[i]())
                var li_ws_size = (
                    BATCH * Self.model_types[i].WORKSPACE_SIZE_PER_SAMPLE
                )
                var li_ws = DeviceBuffer[dtype](
                    ctx,
                    ws_ptr + BATCH * Self._ws_layer_offset[i](),
                    li_ws_size if li_ws_size > 0 else 1,
                    owning=False,
                )

                if perf:
                    perf.bitcast[PerfTimer[True]]()[].sync_and_mark(ctx)

                comptime if i == Self.N - 1:
                    # Last layer: grad_output -> grad_inter[i-1]
                    var gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var go_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(
                                BATCH, Self.model_types[i].OUT_DIM
                            ),
                            MutAnyOrigin,
                        ]
                    ](grad_output)
                    Self.model_types[i].backward_gpu[BATCH, dtype](
                        ctx, gi, go_rb, li_p, li_c, li_g, li_ws
                    )
                elif i == 0:
                    # First layer: grad_inter[0] -> grad_input
                    var go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr)
                    var gi_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                            MutAnyOrigin,
                        ]
                    ](grad_input)
                    Self.model_types[i].backward_gpu[BATCH, dtype](
                        ctx, gi_rb, go, li_p, li_c, li_g, li_ws
                    )
                else:
                    # Middle: grad_inter[i] -> grad_inter[i-1]
                    var go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i]())
                    var gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.model_types[i].backward_gpu[BATCH, dtype](
                        ctx, gi, go, li_p, li_c, li_g, li_ws
                    )

                if perf:
                    perf.bitcast[PerfTimer[True]]()[].sync_and_accumulate(
                        perf_slot + _ri, ctx
                    )

        # Restore caller's _mark so L2 timing measures the full span
        if perf:
            perf.bitcast[PerfTimer[True]]()[]._mark = saved_mark

    # =========================================================================
    # Slot Registration for L3 Profiling
    # =========================================================================

    @staticmethod
    def register_forward_slots[
        ENABLED: Bool
    ](mut timer: PerfTimer[ENABLED], parent: Int = -1) -> Int:
        """Add N slots for forward-pass layers. Returns base slot index."""
        var base = len(timer.accum_ns)
        comptime for i in range(Self.N):
            _ = timer.add_slot(
                "L"
                + String(i)
                + "["
                + String(Self.model_types[i].IN_DIM)
                + "→"
                + String(Self.model_types[i].OUT_DIM)
                + "]",
                parent=parent,
            )
        return base

    @staticmethod
    def register_backward_slots[
        ENABLED: Bool
    ](mut timer: PerfTimer[ENABLED], parent: Int = -1) -> Int:
        """Add N slots for backward-pass layers (reverse order). Returns base slot index.
        """
        var base = len(timer.accum_ns)
        comptime for _ri in range(Self.N):
            comptime i = Self.N - 1 - _ri
            _ = timer.add_slot(
                "L"
                + String(i)
                + "["
                + String(Self.model_types[i].OUT_DIM)
                + "←"
                + String(Self.model_types[i].IN_DIM)
                + "]",
                parent=parent,
            )
        return base
