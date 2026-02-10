from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from std.builtin.variadics import Variadic

# =============================================================================
# Variadic Sequential Container
# =============================================================================
#
# Sequential[*LAYERS: Model] handles any number of layers directly using
# Variadic.types + @parameter for with index-based access for compile-time
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
    Uses Variadic.types + @parameter for to iterate at compile time.
    """

    comptime model_types = Variadic.types[T=Model, *Self.LAYERS]
    comptime N = Variadic.size(Self.model_types)

    comptime IN_DIM: Int = Self.model_types[0].IN_DIM
    comptime OUT_DIM: Int = Self.model_types[Self.N - 1].OUT_DIM

    # --- Sum helpers ---

    @staticmethod
    fn _sum_param_size() -> Int:
        var total = 0
        @parameter
        for i in range(Self.N):
            total += Self.model_types[i].PARAM_SIZE
        return total

    @staticmethod
    fn _sum_cache_size() -> Int:
        var total = 0
        @parameter
        for i in range(Self.N):
            total += Self.model_types[i].CACHE_SIZE
        return total

    @staticmethod
    fn _total_inter() -> Int:
        """Per-sample intermediate buffer size (sum of OUT_DIM for layers 0..N-2)."""
        var total = 0
        @parameter
        for i in range(Self.N - 1):
            total += Self.model_types[i].OUT_DIM
        return total

    @staticmethod
    fn _sum_ws() -> Int:
        var total = 0
        @parameter
        for i in range(Self.N):
            total += Self.model_types[i].WORKSPACE_SIZE_PER_SAMPLE
        return total

    comptime PARAM_SIZE: Int = Self._sum_param_size()
    comptime CACHE_SIZE: Int = Self._sum_cache_size()
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self._total_inter() + Self._sum_ws()

    # --- Offset helpers (all per-sample) ---

    @staticmethod
    fn _param_offset[idx: Int]() -> Int:
        var total = 0
        @parameter
        for j in range(idx):
            total += Self.model_types[j].PARAM_SIZE
        return total

    @staticmethod
    fn _cache_offset[idx: Int]() -> Int:
        var total = 0
        @parameter
        for j in range(idx):
            total += Self.model_types[j].CACHE_SIZE
        return total

    @staticmethod
    fn _inter_offset[idx: Int]() -> Int:
        """Offset of intermediate slot idx (per sample)."""
        var total = 0
        @parameter
        for j in range(idx):
            total += Self.model_types[j].OUT_DIM
        return total

    @staticmethod
    fn _ws_layer_offset[idx: Int]() -> Int:
        """Offset for layer idx's workspace (per sample), after all inter buffers."""
        var total = Self._total_inter()
        @parameter
        for j in range(idx):
            total += Self.model_types[j].WORKSPACE_SIZE_PER_SAMPLE
        return total

    # =========================================================================
    # CPU Forward (with cache)
    # =========================================================================

    @staticmethod
    fn forward[
        BATCH: Int
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
        @parameter
        if Self.N == 1:
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
            Self.model_types[0].forward[BATCH](in_v, out_v, p_v, c_v)
        else:
            # Flat intermediate buffer for all N-1 inter-layer activations
            var inter_storage = List[Scalar[dtype]](
                capacity=BATCH * Self._total_inter()
            )
            for _ in range(BATCH * Self._total_inter()):
                inter_storage.append(0)
            var inter_ptr = inter_storage.unsafe_ptr()

            @parameter
            for i in range(Self.N):
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

                @parameter
                if i == 0:
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
                    Self.model_types[i].forward[BATCH](
                        li_in, li_out, li_p, li_c
                    )
                elif i == Self.N - 1:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](
                        inter_ptr
                        + BATCH * Self._inter_offset[i - 1]()
                    )
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](output.ptr)
                    Self.model_types[i].forward[BATCH](
                        li_in, li_out, li_p, li_c
                    )
                else:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](
                        inter_ptr
                        + BATCH * Self._inter_offset[i - 1]()
                    )
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i]())
                    Self.model_types[i].forward[BATCH](
                        li_in, li_out, li_p, li_c
                    )

    # =========================================================================
    # CPU Forward (no cache)
    # =========================================================================

    @staticmethod
    fn forward[
        BATCH: Int
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
        @parameter
        if Self.N == 1:
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
            Self.model_types[0].forward[BATCH](in_v, out_v, p_v)
        else:
            var inter_storage = List[Scalar[dtype]](
                capacity=BATCH * Self._total_inter()
            )
            for _ in range(BATCH * Self._total_inter()):
                inter_storage.append(0)
            var inter_ptr = inter_storage.unsafe_ptr()

            @parameter
            for i in range(Self.N):
                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.model_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())

                @parameter
                if i == 0:
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
                    Self.model_types[i].forward[BATCH](
                        li_in, li_out, li_p
                    )
                elif i == Self.N - 1:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](
                        inter_ptr
                        + BATCH * Self._inter_offset[i - 1]()
                    )
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](output.ptr)
                    Self.model_types[i].forward[BATCH](
                        li_in, li_out, li_p
                    )
                else:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](
                        inter_ptr
                        + BATCH * Self._inter_offset[i - 1]()
                    )
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.model_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i]())
                    Self.model_types[i].forward[BATCH](
                        li_in, li_out, li_p
                    )

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    fn backward[
        BATCH: Int
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
        @parameter
        if Self.N == 1:
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
            Self.model_types[0].backward[BATCH](
                go_v, gi_v, p_v, c_v, g_v
            )
        else:
            # Gradient intermediate buffer (same layout as forward inter)
            var grad_inter_storage = List[Scalar[dtype]](
                capacity=BATCH * Self._total_inter()
            )
            for _ in range(BATCH * Self._total_inter()):
                grad_inter_storage.append(0)
            var gi_ptr = grad_inter_storage.unsafe_ptr()

            # Reverse iteration
            @parameter
            for _ri in range(Self.N):
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

                @parameter
                if i == Self.N - 1:
                    # Last layer: Sequential grad_output -> grad_inter[i-1]
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.model_types[i].OUT_DIM
                        ),
                        MutAnyOrigin,
                    ](grad_output.ptr)
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.model_types[i].IN_DIM
                        ),
                        MutAnyOrigin,
                    ](
                        gi_ptr
                        + BATCH * Self._inter_offset[i - 1]()
                    )
                    Self.model_types[i].backward[BATCH](
                        li_go, li_gi, li_p, li_c, li_g
                    )
                elif i == 0:
                    # First layer: grad_inter[0] -> Sequential grad_input
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.model_types[i].OUT_DIM
                        ),
                        MutAnyOrigin,
                    ](gi_ptr)
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.model_types[i].IN_DIM
                        ),
                        MutAnyOrigin,
                    ](grad_input.ptr)
                    Self.model_types[i].backward[BATCH](
                        li_go, li_gi, li_p, li_c, li_g
                    )
                else:
                    # Middle: grad_inter[i] -> grad_inter[i-1]
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.model_types[i].OUT_DIM
                        ),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i]())
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.model_types[i].IN_DIM
                        ),
                        MutAnyOrigin,
                    ](
                        gi_ptr
                        + BATCH * Self._inter_offset[i - 1]()
                    )
                    Self.model_types[i].backward[BATCH](
                        li_go, li_gi, li_p, li_c, li_g
                    )

    # =========================================================================
    # GPU Forward (with cache)
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass using pre-allocated workspace.

        Workspace layout: [inter_bufs (N-1 buffers) | L0 ws | L1 ws | ... | L_{N-1} ws]
        """
        @parameter
        if Self.N == 1:
            Self.model_types[0].forward_gpu[BATCH](
                ctx,
                output_buf,
                input_buf,
                params_buf,
                cache_buf,
                workspace_buf,
            )
        else:
            var ws_ptr = workspace_buf.unsafe_ptr()
            var p_ptr = params_buf.unsafe_ptr()
            var c_ptr = cache_buf.unsafe_ptr()

            @parameter
            for i in range(Self.N):
                # Params view
                var li_params = DeviceBuffer[dtype](
                    ctx,
                    p_ptr + Self._param_offset[i](),
                    Self.model_types[i].PARAM_SIZE,
                    owning=False,
                )
                # Cache view
                var li_cache = DeviceBuffer[dtype](
                    ctx,
                    c_ptr + BATCH * Self._cache_offset[i](),
                    BATCH * Self.model_types[i].CACHE_SIZE,
                    owning=False,
                )
                # Layer workspace view
                var li_ws_size = BATCH * Self.model_types[
                    i
                ].WORKSPACE_SIZE_PER_SAMPLE
                var li_ws = DeviceBuffer[dtype](
                    ctx,
                    ws_ptr + BATCH * Self._ws_layer_offset[i](),
                    li_ws_size if li_ws_size > 0 else 1,
                    owning=False,
                )

                @parameter
                if i == 0:
                    var inter_out = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr,
                        BATCH * Self.model_types[i].OUT_DIM,
                        owning=False,
                    )
                    Self.model_types[i].forward_gpu[BATCH](
                        ctx,
                        inter_out,
                        input_buf,
                        li_params,
                        li_cache,
                        li_ws,
                    )
                elif i == Self.N - 1:
                    var inter_in = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr
                        + BATCH * Self._inter_offset[i - 1](),
                        BATCH * Self.model_types[i].IN_DIM,
                        owning=False,
                    )
                    Self.model_types[i].forward_gpu[BATCH](
                        ctx,
                        output_buf,
                        inter_in,
                        li_params,
                        li_cache,
                        li_ws,
                    )
                else:
                    var inter_in = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr
                        + BATCH * Self._inter_offset[i - 1](),
                        BATCH * Self.model_types[i].IN_DIM,
                        owning=False,
                    )
                    var inter_out = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr + BATCH * Self._inter_offset[i](),
                        BATCH * Self.model_types[i].OUT_DIM,
                        owning=False,
                    )
                    Self.model_types[i].forward_gpu[BATCH](
                        ctx,
                        inter_out,
                        inter_in,
                        li_params,
                        li_cache,
                        li_ws,
                    )

    # =========================================================================
    # GPU Forward (no cache)
    # =========================================================================

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        @parameter
        if Self.N == 1:
            Self.model_types[0].forward_gpu_no_cache[BATCH](
                ctx, output_buf, input_buf, params_buf, workspace_buf
            )
        else:
            var ws_ptr = workspace_buf.unsafe_ptr()
            var p_ptr = params_buf.unsafe_ptr()

            @parameter
            for i in range(Self.N):
                var li_params = DeviceBuffer[dtype](
                    ctx,
                    p_ptr + Self._param_offset[i](),
                    Self.model_types[i].PARAM_SIZE,
                    owning=False,
                )
                var li_ws_size = BATCH * Self.model_types[
                    i
                ].WORKSPACE_SIZE_PER_SAMPLE
                var li_ws = DeviceBuffer[dtype](
                    ctx,
                    ws_ptr + BATCH * Self._ws_layer_offset[i](),
                    li_ws_size if li_ws_size > 0 else 1,
                    owning=False,
                )

                @parameter
                if i == 0:
                    var inter_out = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr,
                        BATCH * Self.model_types[i].OUT_DIM,
                        owning=False,
                    )
                    Self.model_types[i].forward_gpu_no_cache[BATCH](
                        ctx, inter_out, input_buf, li_params, li_ws
                    )
                elif i == Self.N - 1:
                    var inter_in = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr
                        + BATCH * Self._inter_offset[i - 1](),
                        BATCH * Self.model_types[i].IN_DIM,
                        owning=False,
                    )
                    Self.model_types[i].forward_gpu_no_cache[BATCH](
                        ctx, output_buf, inter_in, li_params, li_ws
                    )
                else:
                    var inter_in = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr
                        + BATCH * Self._inter_offset[i - 1](),
                        BATCH * Self.model_types[i].IN_DIM,
                        owning=False,
                    )
                    var inter_out = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr + BATCH * Self._inter_offset[i](),
                        BATCH * Self.model_types[i].OUT_DIM,
                        owning=False,
                    )
                    Self.model_types[i].forward_gpu_no_cache[BATCH](
                        ctx, inter_out, inter_in, li_params, li_ws
                    )

    # =========================================================================
    # GPU Backward
    # =========================================================================

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward pass. Workspace inter region reused for gradient intermediates."""
        @parameter
        if Self.N == 1:
            Self.model_types[0].backward_gpu[BATCH](
                ctx,
                grad_input_buf,
                grad_output_buf,
                params_buf,
                cache_buf,
                grads_buf,
                workspace_buf,
            )
        else:
            var ws_ptr = workspace_buf.unsafe_ptr()
            var p_ptr = params_buf.unsafe_ptr()
            var c_ptr = cache_buf.unsafe_ptr()
            var g_ptr = grads_buf.unsafe_ptr()

            # Reverse iteration
            @parameter
            for _ri in range(Self.N):
                comptime i = Self.N - 1 - _ri

                var li_params = DeviceBuffer[dtype](
                    ctx,
                    p_ptr + Self._param_offset[i](),
                    Self.model_types[i].PARAM_SIZE,
                    owning=False,
                )
                var li_cache = DeviceBuffer[dtype](
                    ctx,
                    c_ptr + BATCH * Self._cache_offset[i](),
                    BATCH * Self.model_types[i].CACHE_SIZE,
                    owning=False,
                )
                var li_grads = DeviceBuffer[dtype](
                    ctx,
                    g_ptr + Self._param_offset[i](),
                    Self.model_types[i].PARAM_SIZE,
                    owning=False,
                )
                var li_ws_size = BATCH * Self.model_types[
                    i
                ].WORKSPACE_SIZE_PER_SAMPLE
                var li_ws = DeviceBuffer[dtype](
                    ctx,
                    ws_ptr + BATCH * Self._ws_layer_offset[i](),
                    li_ws_size if li_ws_size > 0 else 1,
                    owning=False,
                )

                @parameter
                if i == Self.N - 1:
                    # Last layer: grad_output -> grad_inter[i-1]
                    var gi_buf = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr
                        + BATCH * Self._inter_offset[i - 1](),
                        BATCH * Self.model_types[i].IN_DIM,
                        owning=False,
                    )
                    Self.model_types[i].backward_gpu[BATCH](
                        ctx,
                        gi_buf,
                        grad_output_buf,
                        li_params,
                        li_cache,
                        li_grads,
                        li_ws,
                    )
                elif i == 0:
                    # First layer: grad_inter[0] -> grad_input
                    var go_buf = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr,
                        BATCH * Self.model_types[i].OUT_DIM,
                        owning=False,
                    )
                    Self.model_types[i].backward_gpu[BATCH](
                        ctx,
                        grad_input_buf,
                        go_buf,
                        li_params,
                        li_cache,
                        li_grads,
                        li_ws,
                    )
                else:
                    # Middle: grad_inter[i] -> grad_inter[i-1]
                    var go_buf = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr + BATCH * Self._inter_offset[i](),
                        BATCH * Self.model_types[i].OUT_DIM,
                        owning=False,
                    )
                    var gi_buf = DeviceBuffer[dtype](
                        ctx,
                        ws_ptr
                        + BATCH * Self._inter_offset[i - 1](),
                        BATCH * Self.model_types[i].IN_DIM,
                        owning=False,
                    )
                    Self.model_types[i].backward_gpu[BATCH](
                        ctx,
                        gi_buf,
                        go_buf,
                        li_params,
                        li_cache,
                        li_grads,
                        li_ws,
                    )


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

comptime Seq2[L0: Model, L1: Model] = Sequential[L0, L1]
comptime Seq3[L0: Model, L1: Model, L2: Model] = Sequential[L0, L1, L2]
comptime Seq4[L0: Model, L1: Model, L2: Model, L3: Model] = Sequential[
    L0, L1, L2, L3
]
comptime Seq5[
    L0: Model, L1: Model, L2: Model, L3: Model, L4: Model
] = Sequential[L0, L1, L2, L3, L4]
comptime Seq6[
    L0: Model, L1: Model, L2: Model, L3: Model, L4: Model, L5: Model
] = Sequential[L0, L1, L2, L3, L4, L5]
comptime Seq7[
    L0: Model,
    L1: Model,
    L2: Model,
    L3: Model,
    L4: Model,
    L5: Model,
    L6: Model,
] = Sequential[L0, L1, L2, L3, L4, L5, L6]
comptime Seq8[
    L0: Model,
    L1: Model,
    L2: Model,
    L3: Model,
    L4: Model,
    L5: Model,
    L6: Model,
    L7: Model,
] = Sequential[L0, L1, L2, L3, L4, L5, L6, L7]


# =============================================================================
# Helper Functions
# =============================================================================


fn seq[L0: Model, L1: Model](l0: L0, l1: L1) -> Sequential[L0, L1]:
    """Create a 2-layer sequential model."""
    _ = l0
    _ = l1
    return Sequential[L0, L1]()


fn seq[
    L0: Model, L1: Model, L2: Model
](l0: L0, l1: L1, l2: L2) -> Sequential[L0, L1, L2]:
    """Create a 3-layer sequential model."""
    _ = l0
    _ = l1
    _ = l2
    return Sequential[L0, L1, L2]()


fn seq[
    L0: Model, L1: Model, L2: Model, L3: Model
](l0: L0, l1: L1, l2: L2, l3: L3) -> Sequential[L0, L1, L2, L3]:
    """Create a 4-layer sequential model."""
    _ = l0
    _ = l1
    _ = l2
    _ = l3
    return Sequential[L0, L1, L2, L3]()


fn seq[
    L0: Model, L1: Model, L2: Model, L3: Model, L4: Model
](
    l0: L0, l1: L1, l2: L2, l3: L3, l4: L4
) -> Sequential[L0, L1, L2, L3, L4]:
    """Create a 5-layer sequential model."""
    _ = l0
    _ = l1
    _ = l2
    _ = l3
    _ = l4
    return Sequential[L0, L1, L2, L3, L4]()


fn seq[
    L0: Model, L1: Model, L2: Model, L3: Model, L4: Model, L5: Model
](
    l0: L0, l1: L1, l2: L2, l3: L3, l4: L4, l5: L5
) -> Sequential[L0, L1, L2, L3, L4, L5]:
    """Create a 6-layer sequential model."""
    _ = l0
    _ = l1
    _ = l2
    _ = l3
    _ = l4
    _ = l5
    return Sequential[L0, L1, L2, L3, L4, L5]()


fn seq[
    L0: Model,
    L1: Model,
    L2: Model,
    L3: Model,
    L4: Model,
    L5: Model,
    L6: Model,
](
    l0: L0, l1: L1, l2: L2, l3: L3, l4: L4, l5: L5, l6: L6
) -> Sequential[L0, L1, L2, L3, L4, L5, L6]:
    """Create a 7-layer sequential model."""
    _ = l0
    _ = l1
    _ = l2
    _ = l3
    _ = l4
    _ = l5
    _ = l6
    return Sequential[L0, L1, L2, L3, L4, L5, L6]()


fn seq[
    L0: Model,
    L1: Model,
    L2: Model,
    L3: Model,
    L4: Model,
    L5: Model,
    L6: Model,
    L7: Model,
]() -> Sequential[L0, L1, L2, L3, L4, L5, L6, L7]:
    """Create an 8-layer sequential model."""
    return Sequential[L0, L1, L2, L3, L4, L5, L6, L7]()
