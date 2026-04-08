"""Parallel branch combinator (variadic).

Parallel[*BRANCHES: Model] runs all N branches on the same input
and concatenates their outputs: y = concat(B0(x), B1(x), ..., B_{N-1}(x)).

All branches must share the same IN_DIM.
Output dimension is sum(branch_types[i].OUT_DIM).

Forward:  output = [B0(input), B1(input), ..., B_{N-1}(input)]  (per-row concat)
Backward: grad_input = sum_i(B_i.backward(grad_i))
          where grad_i = grad_output[:, out_offset[i]:out_offset[i+1]]
"""

from ...constants import dtype, TPB
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.builtin.variadics import Variadic


@fieldwise_init
struct Parallel[*BRANCHES: Model](Model):
    """N-branch parallel: y = concat(B0(x), B1(x), ..., B_{N-1}(x)).

    All branches receive the same input. Their outputs are concatenated
    along the feature dimension (per row).
    """

    comptime branch_types = Variadic.types[T=Model, *Self.BRANCHES]
    comptime N = TypeList[*Self.branch_types].size

    comptime IN_DIM: Int = Self.branch_types[0].IN_DIM

    # --- Sum helpers ---

    @staticmethod
    def _sum_out_dim() -> Int:
        var total = 0

        comptime for i in range(Self.N):
            total += Self.branch_types[i].OUT_DIM
        return total

    @staticmethod
    def _sum_param_size() -> Int:
        var total = 0

        comptime for i in range(Self.N):
            total += Self.branch_types[i].PARAM_SIZE
        return total

    @staticmethod
    def _sum_cache_size() -> Int:
        var total = 0

        comptime for i in range(Self.N):
            total += Self.branch_types[i].CACHE_SIZE
        return total

    @staticmethod
    def _sum_ws() -> Int:
        var total = 0

        comptime for i in range(Self.N):
            total += Self.branch_types[i].WORKSPACE_SIZE_PER_SAMPLE
        return total

    comptime OUT_DIM: Int = Self._sum_out_dim()
    comptime PARAM_SIZE: Int = Self._sum_param_size()
    comptime CACHE_SIZE: Int = Self._sum_cache_size()
    # Own scratch: N * IN_DIM for per-branch grad_input buffers in backward
    comptime _OWN_WS: Int = Self.N * Self.IN_DIM
    # Workspace: own scratch + branch output buffers + per-branch workspace
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self._OWN_WS + Self._sum_out_dim() + Self._sum_ws()

    # --- Offset helpers ---

    @staticmethod
    def _out_offset[idx: Int]() -> Int:
        """Sum of OUT_DIM for branches 0..idx-1."""
        var total = 0

        comptime for j in range(idx):
            total += Self.branch_types[j].OUT_DIM
        return total

    @staticmethod
    def _param_offset[idx: Int]() -> Int:
        var total = 0

        comptime for j in range(idx):
            total += Self.branch_types[j].PARAM_SIZE
        return total

    @staticmethod
    def _cache_offset[idx: Int]() -> Int:
        var total = 0

        comptime for j in range(idx):
            total += Self.branch_types[j].CACHE_SIZE
        return total

    @staticmethod
    def _ws_branch_offset[idx: Int]() -> Int:
        """Workspace offset for branch idx, after own scratch + output buffers.
        """
        var total = Self._OWN_WS + Self._sum_out_dim()

        comptime for j in range(idx):
            total += Self.branch_types[j].WORKSPACE_SIZE_PER_SAMPLE
        return total

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
        """Initialize each branch with its own fan dimensions."""
        comptime for i in range(Self.N):
            comptime if Self.branch_types[i].PARAM_SIZE > 0:
                var branch_params = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                Self.branch_types[i].initialize_params[INIT, dtype](branch_params)

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
        # Flat buffer for all branch outputs
        var buf_storage = List[Scalar[dtype]](capacity=BATCH * Self.OUT_DIM)
        for _ in range(BATCH * Self.OUT_DIM):
            buf_storage.append(0)
        var buf_ptr = buf_storage.unsafe_ptr()

        comptime for i in range(Self.N):
            var buf_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                MutAnyOrigin,
            ](buf_ptr + BATCH * Self._out_offset[i]())
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._cache_offset[i]())

            var inp_i = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            Self.branch_types[i].forward[BATCH, dtype](inp_i, buf_i, pi, ci)

        # Interleave: for each row, copy each branch's output into correct columns
        for b in range(BATCH):
            comptime for i in range(Self.N):
                for j in range(Self.branch_types[i].OUT_DIM):
                    output.ptr[
                        b * Self.OUT_DIM + Self._out_offset[i]() + j
                    ] = buf_ptr[
                        BATCH * Self._out_offset[i]()
                        + b * Self.branch_types[i].OUT_DIM
                        + j
                    ]

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
        var buf_storage = List[Scalar[dtype]](capacity=BATCH * Self.OUT_DIM)
        for _ in range(BATCH * Self.OUT_DIM):
            buf_storage.append(0)
        var buf_ptr = buf_storage.unsafe_ptr()

        comptime for i in range(Self.N):
            var buf_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                MutAnyOrigin,
            ](buf_ptr + BATCH * Self._out_offset[i]())
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())

            var inp_i = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            Self.branch_types[i].forward[BATCH, dtype](inp_i, buf_i, pi)

        for b in range(BATCH):
            comptime for i in range(Self.N):
                for j in range(Self.branch_types[i].OUT_DIM):
                    output.ptr[
                        b * Self.OUT_DIM + Self._out_offset[i]() + j
                    ] = buf_ptr[
                        BATCH * Self._out_offset[i]()
                        + b * Self.branch_types[i].OUT_DIM
                        + j
                    ]

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
        # Split grad_output into per-branch grad buffers (de-interleave)
        var grad_branch_storage = List[Scalar[dtype]](
            capacity=BATCH * Self.OUT_DIM
        )
        for _ in range(BATCH * Self.OUT_DIM):
            grad_branch_storage.append(0)
        var gb_ptr = grad_branch_storage.unsafe_ptr()

        for b in range(BATCH):
            comptime for i in range(Self.N):
                for j in range(Self.branch_types[i].OUT_DIM):
                    (gb_ptr + BATCH * Self._out_offset[i]())[
                        b * Self.branch_types[i].OUT_DIM + j
                    ] = grad_output.ptr[
                        b * Self.OUT_DIM + Self._out_offset[i]() + j
                    ]

        # Flat buffer for N grad_input contributions: N * BATCH * IN_DIM
        var gi_all_storage = List[Scalar[dtype]](
            capacity=Self.N * BATCH * Self.IN_DIM
        )
        for _ in range(Self.N * BATCH * Self.IN_DIM):
            gi_all_storage.append(0)
        var gi_all_ptr = gi_all_storage.unsafe_ptr()

        comptime for i in range(Self.N):
            var grad_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                MutAnyOrigin,
            ](gb_ptr + BATCH * Self._out_offset[i]())
            var gi_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.IN_DIM),
                MutAnyOrigin,
            ](gi_all_ptr + i * BATCH * Self.IN_DIM)
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._cache_offset[i]())
            var gp_i = LayoutTensor[
                dtype,
                Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr + Self._param_offset[i]())

            # Rebind gi_i for branches with IN_DIM type match
            var gi_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].IN_DIM),
                    MutAnyOrigin,
                ]
            ](gi_i)
            Self.branch_types[i].backward[BATCH, dtype](grad_i, gi_rb, pi, ci, gp_i)

        # Sum all grad_input contributions
        for k in range(BATCH * Self.IN_DIM):
            var s = Scalar[dtype](0)
            comptime for i in range(Self.N):
                s += gi_all_ptr[i * BATCH * Self.IN_DIM + k]
            grad_input.ptr[k] = s

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
        var ws_ptr = workspace.unsafe_ptr()

        # Run each branch forward into its workspace output buffer
        comptime for i in range(Self.N):
            var buf_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                MutAnyOrigin,
            ](ws_ptr + BATCH * (Self._OWN_WS + Self._out_offset[i]()))
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._cache_offset[i]())

            var ws_i_size = (
                BATCH * Self.branch_types[i].WORKSPACE_SIZE_PER_SAMPLE
            )
            var ws_i = DeviceBuffer[dtype](
                ctx,
                ws_ptr + BATCH * Self._ws_branch_offset[i](),
                ws_i_size if ws_i_size > 0 else 1,
                owning=False,
            )

            var inp_i = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            Self.branch_types[i].forward_gpu[BATCH, dtype](
                ctx, buf_i, inp_i, pi, ci, ws_i
            )

        # Interleave all branch outputs into final output via per-branch copy kernels
        comptime for i in range(Self.N):
            comptime BRANCH_OUT = Self.branch_types[i].OUT_DIM
            comptime BRANCH_OFF = Self._out_offset[i]()
            comptime TOTAL_OUT = Self.OUT_DIM

            var buf_i_immut = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                ImmutAnyOrigin,
            ](ws_ptr + BATCH * (Self._OWN_WS + Self._out_offset[i]()))

            @always_inline
            def copy_branch_fwd(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OUT_DIM),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                    ImmutAnyOrigin,
                ],
            ):
                var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
                if tid >= BATCH * BRANCH_OUT:
                    return
                var row = tid // BRANCH_OUT
                var col = tid % BRANCH_OUT
                dst.ptr[row * TOTAL_OUT + BRANCH_OFF + col] = src.ptr[tid]

            var grid_x = (BATCH * BRANCH_OUT + TPB - 1) // TPB
            ctx.enqueue_function[copy_branch_fwd, copy_branch_fwd](
                output,
                buf_i_immut,
                grid_dim=(grid_x,),
                block_dim=(TPB,),
            )

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
        var ws_ptr = workspace.unsafe_ptr()

        comptime for i in range(Self.N):
            var buf_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                MutAnyOrigin,
            ](ws_ptr + BATCH * (Self._OWN_WS + Self._out_offset[i]()))
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())

            var ws_i_size = (
                BATCH * Self.branch_types[i].WORKSPACE_SIZE_PER_SAMPLE
            )
            var ws_i = DeviceBuffer[dtype](
                ctx,
                ws_ptr + BATCH * Self._ws_branch_offset[i](),
                ws_i_size if ws_i_size > 0 else 1,
                owning=False,
            )

            var inp_i = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            Self.branch_types[i].forward_gpu_no_cache[BATCH, dtype](
                ctx, buf_i, inp_i, pi, ws_i
            )

        comptime for i in range(Self.N):
            comptime BRANCH_OUT = Self.branch_types[i].OUT_DIM
            comptime BRANCH_OFF = Self._out_offset[i]()
            comptime TOTAL_OUT = Self.OUT_DIM

            var buf_i_immut = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                ImmutAnyOrigin,
            ](ws_ptr + BATCH * (Self._OWN_WS + Self._out_offset[i]()))

            @always_inline
            def copy_branch_fwd_nc(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OUT_DIM),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                    ImmutAnyOrigin,
                ],
            ):
                var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
                if tid >= BATCH * BRANCH_OUT:
                    return
                var row = tid // BRANCH_OUT
                var col = tid % BRANCH_OUT
                dst.ptr[row * TOTAL_OUT + BRANCH_OFF + col] = src.ptr[tid]

            var grid_x = (BATCH * BRANCH_OUT + TPB - 1) // TPB
            ctx.enqueue_function[copy_branch_fwd_nc, copy_branch_fwd_nc](
                output,
                buf_i_immut,
                grid_dim=(grid_x,),
                block_dim=(TPB,),
            )

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
        """GPU forward on stream — delegates to default stream."""
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, workspace)

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
        var ws_ptr = workspace.unsafe_ptr()

        # De-interleave grad_output into per-branch grad buffers (reuse workspace output region)
        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        comptime for i in range(Self.N):
            comptime BRANCH_OUT = Self.branch_types[i].OUT_DIM
            comptime BRANCH_OFF = Self._out_offset[i]()
            comptime TOTAL_OUT = Self.OUT_DIM

            var grad_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                MutAnyOrigin,
            ](ws_ptr + BATCH * (Self._OWN_WS + Self._out_offset[i]()))

            @always_inline
            def split_branch(
                gi: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                    MutAnyOrigin,
                ],
                go: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OUT_DIM),
                    ImmutAnyOrigin,
                ],
            ):
                var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
                if tid >= BATCH * BRANCH_OUT:
                    return
                var row = tid // BRANCH_OUT
                var col = tid % BRANCH_OUT
                gi.ptr[tid] = go.ptr[row * TOTAL_OUT + BRANCH_OFF + col]

            var grid_x = (BATCH * BRANCH_OUT + TPB - 1) // TPB
            ctx.enqueue_function[split_branch, split_branch](
                grad_i,
                go_immut,
                grid_dim=(grid_x,),
                block_dim=(TPB,),
            )

        # Slice N grad_input buffers from workspace (at offset 0)
        var gi_buf_ptr = ws_ptr

        # Run backward for each branch
        comptime for i in range(Self.N):
            var grad_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].OUT_DIM),
                MutAnyOrigin,
            ](ws_ptr + BATCH * (Self._OWN_WS + Self._out_offset[i]()))
            var gi_i = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.IN_DIM),
                MutAnyOrigin,
            ](gi_buf_ptr + i * BATCH * Self.IN_DIM)
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.branch_types[i].CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._cache_offset[i]())
            var gp_i = LayoutTensor[
                dtype,
                Layout.row_major(Self.branch_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr + Self._param_offset[i]())

            var ws_i_size = (
                BATCH * Self.branch_types[i].WORKSPACE_SIZE_PER_SAMPLE
            )
            var ws_i = DeviceBuffer[dtype](
                ctx,
                ws_ptr + BATCH * Self._ws_branch_offset[i](),
                ws_i_size if ws_i_size > 0 else 1,
                owning=False,
            )

            var gi_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.branch_types[i].IN_DIM),
                    MutAnyOrigin,
                ]
            ](gi_i)
            Self.branch_types[i].backward_gpu[BATCH, dtype](
                ctx, gi_rb, grad_i, pi, ci, gp_i, ws_i
            )

        # Sum all N grad_input contributions
        var gi_immut = LayoutTensor[
            dtype,
            Layout.row_major(Self.N * BATCH, Self.IN_DIM),
            ImmutAnyOrigin,
        ](gi_buf_ptr)

        comptime GI_TOTAL = BATCH * Self.IN_DIM
        comptime N_BRANCHES = Self.N

        @always_inline
        def sum_gi_wrapper(
            dst: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.IN_DIM),
                MutAnyOrigin,
            ],
            all_gi: LayoutTensor[
                dtype,
                Layout.row_major(Self.N * BATCH, Self.IN_DIM),
                ImmutAnyOrigin,
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= GI_TOTAL:
                return
            var s = all_gi.ptr[tid]
            for b in range(1, N_BRANCHES):
                s += all_gi.ptr[b * GI_TOTAL + tid]
            dst.ptr[tid] = s

        var grid_sum = (BATCH * Self.IN_DIM + TPB - 1) // TPB
        ctx.enqueue_function[sum_gi_wrapper, sum_gi_wrapper](
            grad_input,
            gi_immut,
            grid_dim=(grid_sum,),
            block_dim=(TPB,),
        )
