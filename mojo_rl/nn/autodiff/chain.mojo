from ..constants import dtype
from ..model.model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from .op import DiffOp
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.builtin.variadics import Variadic

# =============================================================================
# AutoDiffChain — compose DiffOp primitives into a Model
# =============================================================================
#
# AutoDiffChain[*OPS: DiffOp] chains N differentiable operations into a single
# Model-conforming struct. Forward calls eval() on each op sequentially;
# backward calls vjp() in reverse order.
#
# Unlike Sequential (which composes Model types), AutoDiffChain composes
# fine-grained DiffOp primitives. DiffOp GPU methods do not use a workspace
# parameter, so WORKSPACE_SIZE_PER_SAMPLE is just the intermediate buffers.
#
# Usage:
#     comptime LinearAD = AutoDiffChain[MatMul[2, 64], BiasAdd[64]]
#     comptime LinearReLUAD = AutoDiffChain[MatMul[2, 64], BiasAdd[64], ReLUOp[64]]
#
# Cache layout: [op0 cache | op1 cache | ... | op_{N-1} cache]
# GPU workspace layout: [inter_buf_0 | inter_buf_1 | ... | inter_buf_{N-2}]
# =============================================================================


@fieldwise_init
struct AutoDiffChain[*OPS: DiffOp](Model):
    """Variadic chain of DiffOp primitives conforming to Model.

    Composes N DiffOps where op[i].OUT_DIM == op[i+1].IN_DIM.
    Uses Variadic.types + comptime for to iterate at compile time.
    """

    comptime op_types = Variadic.types[T=DiffOp, *Self.OPS]
    comptime N = Variadic.size(Self.op_types)

    comptime IN_DIM: Int = Self.op_types[0].IN_DIM
    comptime OUT_DIM: Int = Self.op_types[Self.N - 1].OUT_DIM

    # --- Sum helpers ---

    @staticmethod
    fn _sum_param_size() -> Int:
        var total = 0

        comptime for i in range(Self.N):
            total += Self.op_types[i].PARAM_SIZE
        return total

    @staticmethod
    fn _sum_cache_size() -> Int:
        var total = 0

        comptime for i in range(Self.N):
            total += Self.op_types[i].CACHE_SIZE
        return total

    @staticmethod
    fn _total_inter() -> Int:
        """Per-sample intermediate buffer size (sum of OUT_DIM for ops 0..N-2).
        """
        var total = 0

        comptime for i in range(Self.N - 1):
            total += Self.op_types[i].OUT_DIM
        return total

    @staticmethod
    fn _max_op_workspace() -> Int:
        """Max per-sample op workspace across all ops (reused sequentially)."""
        var m = 0

        comptime for i in range(Self.N):
            if Self.op_types[i].OP_WORKSPACE_PER_SAMPLE > m:
                m = Self.op_types[i].OP_WORKSPACE_PER_SAMPLE
        return m

    comptime PARAM_SIZE: Int = Self._sum_param_size()
    comptime CACHE_SIZE: Int = Self._sum_cache_size()
    comptime INTER_SIZE_PER_SAMPLE: Int = Self._total_inter()
    comptime MAX_OP_WORKSPACE_PER_SAMPLE: Int = Self._max_op_workspace()
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.INTER_SIZE_PER_SAMPLE + Self.CACHE_SIZE + Self.MAX_OP_WORKSPACE_PER_SAMPLE

    # --- Offset helpers ---

    @staticmethod
    fn _param_offset[idx: Int]() -> Int:
        var total = 0

        comptime for j in range(idx):
            total += Self.op_types[j].PARAM_SIZE
        return total

    @staticmethod
    fn _cache_offset[idx: Int]() -> Int:
        var total = 0

        comptime for j in range(idx):
            total += Self.op_types[j].CACHE_SIZE
        return total

    @staticmethod
    fn _inter_offset[idx: Int]() -> Int:
        """Offset of intermediate slot idx (per sample)."""
        var total = 0

        comptime for j in range(idx):
            total += Self.op_types[j].OUT_DIM
        return total

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize each DiffOp's params with its own fan dimensions."""
        comptime for i in range(Self.N):
            comptime if Self.op_types[i].PARAM_SIZE > 0:
                var op_params = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.op_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                INIT.init[
                    Self.op_types[i].PARAM_SIZE,
                    Self.op_types[i].IN_DIM,
                    Self.op_types[i].OUT_DIM,
                ](op_params)

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
        comptime if Self.N == 1:
            var in_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].IN_DIM),
                MutAnyOrigin,
            ](input.ptr)
            var out_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].OUT_DIM),
                MutAnyOrigin,
            ](output.ptr)
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.op_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            Self.op_types[0].eval[BATCH](in_v, out_v, p_v, c_v)
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
                    Layout.row_major(Self.op_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[i].CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())

                comptime if i == 0:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](input.ptr)
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr)
                    Self.op_types[i].eval[BATCH](li_in, li_out, li_p, li_c)
                elif i == Self.N - 1:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](output.ptr)
                    Self.op_types[i].eval[BATCH](li_in, li_out, li_p, li_c)
                else:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i]())
                    Self.op_types[i].eval[BATCH](li_in, li_out, li_p, li_c)

    # =========================================================================
    # CPU Forward (no cache — inference)
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
        # DiffOp.eval always takes a cache param, so allocate a dummy cache.
        var dummy_cache = List[Scalar[dtype]](
            capacity=BATCH * Self.CACHE_SIZE if Self.CACHE_SIZE > 0 else 1
        )
        var cap = BATCH * Self.CACHE_SIZE if Self.CACHE_SIZE > 0 else 1
        for _ in range(cap):
            dummy_cache.append(0)
        var c = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](dummy_cache.unsafe_ptr())
        Self.forward[BATCH](input, output, params, c)

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
        comptime if Self.N == 1:
            var go_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].OUT_DIM),
                MutAnyOrigin,
            ](grad_output.ptr)
            var gi_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].IN_DIM),
                MutAnyOrigin,
            ](grad_input.ptr)
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.op_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var g_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.op_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr)
            Self.op_types[0].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
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
                    Layout.row_major(Self.op_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[i].CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var li_g = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.op_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](grads.ptr + Self._param_offset[i]())

                comptime if i == Self.N - 1:
                    # Last op: chain grad_output -> grad_inter[i-1]
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](grad_output.ptr)
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.op_types[i].vjp[BATCH](
                        li_go, li_gi, li_p, li_c, li_g
                    )
                elif i == 0:
                    # First op: grad_inter[0] -> chain grad_input
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](gi_ptr)
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](grad_input.ptr)
                    Self.op_types[i].vjp[BATCH](
                        li_go, li_gi, li_p, li_c, li_g
                    )
                else:
                    # Middle: grad_inter[i] -> grad_inter[i-1]
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i]())
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.op_types[i].vjp[BATCH](
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
        """GPU forward pass using pre-allocated workspace for intermediates.

        Workspace layout:
          [inter_buf_0 | ... | inter_buf_{N-2} | cache (unused) | op_workspace]
        Each inter_buf_i has size BATCH * op_types[i].OUT_DIM.
        op_workspace has size BATCH * MAX_OP_WORKSPACE_PER_SAMPLE.
        """

        # Op workspace pointer: past inter + cache region
        var op_ws_ptr = workspace.unsafe_ptr() + BATCH * (
            Self.INTER_SIZE_PER_SAMPLE + Self.CACHE_SIZE
        )

        comptime if Self.N == 1:
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.op_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var out_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[0].OUT_DIM),
                    MutAnyOrigin,
                ]
            ](output)
            var in_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[0].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            Self.op_types[0].eval_gpu[BATCH](
                ctx, out_rb, in_rb, p_v, c_v, op_ws_ptr
            )
        else:
            var ws_ptr = workspace.unsafe_ptr()

            comptime for i in range(Self.N):
                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.op_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[i].CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())

                comptime if i == 0:
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr)
                    var in_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                            MutAnyOrigin,
                        ]
                    ](input)
                    Self.op_types[i].eval_gpu[BATCH](
                        ctx, inter_out, in_rb, li_p, li_c, op_ws_ptr
                    )
                elif i == Self.N - 1:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var out_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(
                                BATCH, Self.op_types[i].OUT_DIM
                            ),
                            MutAnyOrigin,
                        ]
                    ](output)
                    Self.op_types[i].eval_gpu[BATCH](
                        ctx, out_rb, inter_in, li_p, li_c, op_ws_ptr
                    )
                else:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i]())
                    Self.op_types[i].eval_gpu[BATCH](
                        ctx, inter_out, inter_in, li_p, li_c, op_ws_ptr
                    )

    # =========================================================================
    # GPU Forward (no cache — inference)
    # =========================================================================

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
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
        """GPU inference forward. Dummy cache carved from workspace — no allocation."""

        # Op workspace pointer: past inter + cache region
        var op_ws_ptr = workspace.unsafe_ptr() + BATCH * (
            Self.INTER_SIZE_PER_SAMPLE + Self.CACHE_SIZE
        )

        comptime if Self.N == 1:
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.op_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            # Dummy cache from workspace (after inter region)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](workspace.unsafe_ptr() + BATCH * Self.INTER_SIZE_PER_SAMPLE)
            var out_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[0].OUT_DIM),
                    MutAnyOrigin,
                ]
            ](output)
            var in_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[0].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            Self.op_types[0].eval_gpu[BATCH](
                ctx, out_rb, in_rb, p_v, c_v, op_ws_ptr
            )
        else:
            # Dummy cache from workspace (after inter region)
            var cache_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                MutAnyOrigin,
            ](workspace.unsafe_ptr() + BATCH * Self.INTER_SIZE_PER_SAMPLE)
            # Delegate to the caching forward
            Self.forward_gpu[BATCH](
                ctx, output, input, params, cache_v, workspace
            )

    @staticmethod
    fn forward_gpu_no_cache_on_stream[
        BATCH: Int,
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
        """GPU forward on stream — delegates to default stream."""
        Self.forward_gpu_no_cache[BATCH](ctx, output, input, params, workspace)

    # =========================================================================
    # GPU Backward
    # =========================================================================

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
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
        """

        var op_ws_ptr = workspace.unsafe_ptr() + BATCH * (
            Self.INTER_SIZE_PER_SAMPLE + Self.CACHE_SIZE
        )

        comptime if Self.N == 1:
            var p_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.op_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var c_v = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.op_types[0].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var g_v = LayoutTensor[
                dtype,
                Layout.row_major(Self.op_types[0].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr)
            var gi_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[0].IN_DIM),
                    MutAnyOrigin,
                ]
            ](grad_input)
            var go_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[0].OUT_DIM),
                    MutAnyOrigin,
                ]
            ](grad_output)
            Self.op_types[0].vjp_gpu[BATCH](
                ctx, go_rb, gi_rb, p_v, c_v, g_v, op_ws_ptr
            )
        else:
            var ws_ptr = workspace.unsafe_ptr()

            # Reverse iteration
            comptime for _ri in range(Self.N):
                comptime i = Self.N - 1 - _ri

                var li_p = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.op_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var li_c = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.op_types[i].CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var li_g = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.op_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](grads.ptr + Self._param_offset[i]())

                comptime if i == Self.N - 1:
                    # Last op: grad_output -> grad_inter[i-1]
                    var gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var go_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(
                                BATCH, Self.op_types[i].OUT_DIM
                            ),
                            MutAnyOrigin,
                        ]
                    ](grad_output)
                    Self.op_types[i].vjp_gpu[BATCH](
                        ctx, go_rb, gi, li_p, li_c, li_g, op_ws_ptr
                    )
                elif i == 0:
                    # First op: grad_inter[0] -> grad_input
                    var go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr)
                    var gi_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                            MutAnyOrigin,
                        ]
                    ](grad_input)
                    Self.op_types[i].vjp_gpu[BATCH](
                        ctx, go, gi_rb, li_p, li_c, li_g, op_ws_ptr
                    )
                else:
                    # Middle: grad_inter[i] -> grad_inter[i-1]
                    var go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i]())
                    var gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.op_types[i].IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.op_types[i].vjp_gpu[BATCH](
                        ctx, go, gi, li_p, li_c, li_g, op_ws_ptr
                    )
