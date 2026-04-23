"""Repeat combinator — apply the same model n times in sequence.

Repeat[n, Inner, shared] applies Inner n times sequentially.

  shared=True  (default): All iterations share the same weights.
                          PARAM_SIZE = Inner.PARAM_SIZE.
                          Gradients accumulate into one buffer.

  shared=False: Each iteration has independent weights.
                PARAM_SIZE = n * Inner.PARAM_SIZE.
                Like Sequential with n identical layer types.

Forward:  y = f_n(f_{n-1}(...f_1(x)...))
Backward: Reverse n iterations, each using its own cache slot.

Requires Inner.IN_DIM == Inner.OUT_DIM (same shape for chaining).
"""

from ...constants import dtype, TPB
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


@fieldwise_init
struct Repeat[n: Int, Inner: Model, shared: Bool = True](Model):
    """Repeated application: y = Inner^n(x).

    Parameters:
        n: Number of repetitions.
        Inner: Model to repeat (must have IN_DIM == OUT_DIM).
        shared: If True, all iterations share weights. If False, independent.
    """

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.Inner.OUT_DIM
    comptime PARAM_SIZE: Int = Self.Inner.PARAM_SIZE if Self.shared else Self.n * Self.Inner.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.Inner.CACHE_SIZE * Self.n  # one per iter
    # Workspace: (n-1) intermediate buffers + Inner's own workspace + cache for no_cache inference
    comptime INTER_SIZE_PER_SAMPLE: Int = (
        Self.Inner.WORKSPACE_SIZE_PER_SAMPLE + (Self.n - 1) * Self.Inner.OUT_DIM
    )
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.INTER_SIZE_PER_SAMPLE + Self.CACHE_SIZE

    # --- Offset helpers ---

    @staticmethod
    def _cache_offset[idx: Int]() -> Int:
        """Cache offset for iteration idx."""
        return idx * Self.Inner.CACHE_SIZE

    @staticmethod
    def _inter_offset[idx: Int]() -> Int:
        """Intermediate buffer offset for iteration idx (per sample)."""
        return idx * Self.Inner.OUT_DIM

    @staticmethod
    def _param_offset[idx: Int]() -> Int:
        """Param offset for iteration idx. 0 for all if shared."""
        comptime if Self.shared:
            return 0
        else:
            return idx * Self.Inner.PARAM_SIZE

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
        """Initialize params. If shared, one init. If independent, init each block."""
        comptime if Self.shared:
            var p0 = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            Self.Inner.initialize_params[INIT, dtype](p0)
        else:
            comptime for i in range(Self.n):
                var pi = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                Self.Inner.initialize_params[INIT, dtype](pi)

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
        comptime if Self.n == 1:
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            Self.Inner.forward[BATCH, dtype](input, output, pi, ci)
        else:
            # Intermediate buffers for n-1 activations
            var inter_storage = List[Scalar[dtype]](
                capacity=BATCH * (Self.n - 1) * Self.Inner.OUT_DIM
            )
            for _ in range(BATCH * (Self.n - 1) * Self.Inner.OUT_DIM):
                inter_storage.append(0)
            var inter_ptr = inter_storage.unsafe_ptr()

            comptime for i in range(Self.n):
                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var pi = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())

                comptime if i == 0:
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr)
                    Self.Inner.forward[BATCH, dtype](input, li_out, pi, ci)
                elif i == Self.n - 1:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.Inner.forward[BATCH, dtype](li_in, output, pi, ci)
                else:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i]())
                    Self.Inner.forward[BATCH, dtype](li_in, li_out, pi, ci)

    # =========================================================================
    # CPU Forward (no cache — inference)
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
        # Allocate dummy cache and delegate
        var cap = BATCH * Self.CACHE_SIZE if Self.CACHE_SIZE > 0 else 1
        var dummy_cache = List[Scalar[dtype]](capacity=cap)
        for _ in range(cap):
            dummy_cache.append(0)
        var c = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](dummy_cache.unsafe_ptr())
        Self.forward[BATCH, dtype](input, output, params, c)

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
        comptime if Self.n == 1:
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var gi = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr)
            Self.Inner.backward[BATCH, dtype](
                grad_output, grad_input, pi, ci, gi
            )
        else:
            # Gradient intermediate buffer
            var grad_inter_storage = List[Scalar[dtype]](
                capacity=BATCH * (Self.n - 1) * Self.Inner.OUT_DIM
            )
            for _ in range(BATCH * (Self.n - 1) * Self.Inner.OUT_DIM):
                grad_inter_storage.append(0)
            var gi_ptr = grad_inter_storage.unsafe_ptr()

            # Reverse iteration
            comptime for _ri in range(Self.n):
                comptime i = Self.n - 1 - _ri

                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var pi = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                var gp = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](grads.ptr + Self._param_offset[i]())

                comptime if i == Self.n - 1:
                    # Last: grad_output -> grad_inter[i-1]
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.Inner.backward[BATCH, dtype](
                        grad_output, li_gi, pi, ci, gp
                    )
                elif i == 0:
                    # First: grad_inter[0] -> grad_input
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](gi_ptr)
                    Self.Inner.backward[BATCH, dtype](
                        li_go, grad_input, pi, ci, gp
                    )
                else:
                    # Middle: grad_inter[i] -> grad_inter[i-1]
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i]())
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.Inner.backward[BATCH, dtype](li_go, li_gi, pi, ci, gp)

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
        # Workspace: [inter_buf_0 | ... | inter_buf_{n-2} | Inner ws]
        comptime if Self.n == 1:
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var out_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                    MutAnyOrigin,
                ]
            ](output)
            var in_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.IN_DIM),
                    MutAnyOrigin,
                ]
            ](input)
            Self.Inner.forward_gpu[BATCH, dtype](
                ctx, out_rb, in_rb, pi, ci, workspace
            )
        else:
            var ws_ptr = workspace.unsafe_ptr()
            comptime INNER_WS_OFF = (Self.n - 1) * Self.Inner.OUT_DIM
            var inner_ws_size = BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
            var inner_ws = DeviceBuffer[dtype](
                ctx,
                ws_ptr + BATCH * INNER_WS_OFF,
                inner_ws_size if inner_ws_size > 0 else 1,
                owning=False,
            )

            comptime for i in range(Self.n):
                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var pi = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())

                comptime if i == 0:
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr)
                    var in_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.Inner.IN_DIM),
                            MutAnyOrigin,
                        ]
                    ](input)
                    Self.Inner.forward_gpu[BATCH, dtype](
                        ctx, inter_out, in_rb, pi, ci, inner_ws
                    )
                elif i == Self.n - 1:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var out_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                            MutAnyOrigin,
                        ]
                    ](output)
                    Self.Inner.forward_gpu[BATCH, dtype](
                        ctx, out_rb, inter_in, pi, ci, inner_ws
                    )
                else:
                    var inter_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var inter_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i]())
                    Self.Inner.forward_gpu[BATCH, dtype](
                        ctx, inter_out, inter_in, pi, ci, inner_ws
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
        # Dummy cache carved from workspace (after inter region) — no allocation.
        var cache_v = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](workspace.unsafe_ptr() + BATCH * Self.INTER_SIZE_PER_SAMPLE)
        Self.forward_gpu[BATCH, dtype](ctx, output, input, params, cache_v, workspace)

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
        comptime if Self.n == 1:
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr)
            var gp = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr)
            var gi_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.IN_DIM),
                    MutAnyOrigin,
                ]
            ](grad_input)
            var go_rb = rebind[
                LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                    MutAnyOrigin,
                ]
            ](grad_output)
            Self.Inner.backward_gpu[BATCH, dtype](
                ctx, gi_rb, go_rb, pi, ci, gp, workspace
            )
        else:
            var ws_ptr = workspace.unsafe_ptr()
            comptime INNER_WS_OFF = (Self.n - 1) * Self.Inner.OUT_DIM
            var inner_ws_size = BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
            var inner_ws = DeviceBuffer[dtype](
                ctx,
                ws_ptr + BATCH * INNER_WS_OFF,
                inner_ws_size if inner_ws_size > 0 else 1,
                owning=False,
            )

            # For shared weights, Inner.backward_gpu may overwrite (not
            # accumulate) grads. Use a temp buffer per iteration + manual
            # accumulation into the main grads to guarantee correctness.
            comptime TEMP_PS = Self.Inner.PARAM_SIZE if Self.shared and Self.Inner.PARAM_SIZE > 0 else 1
            var temp_grads_buf = ctx.enqueue_create_buffer[dtype](TEMP_PS)

            # Accumulate kernel: dst[i] += src[i]
            @parameter
            @always_inline
            def _accum_kernel(
                dst: LayoutTensor[dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin],
                src: LayoutTensor[dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= Self.Inner.PARAM_SIZE:
                    return
                dst[idx] = dst[idx] + src[idx]

            # Reverse iteration
            comptime for _ri in range(Self.n):
                comptime i = Self.n - 1 - _ri

                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())
                var pi = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())

                # For shared weights: zero temp, backward into temp
                # For independent weights: backward into main grads directly
                var gp_base = grads.ptr + Self._param_offset[i]()
                comptime if Self.shared:
                    ctx.enqueue_memset(temp_grads_buf, 0)
                    gp_base = temp_grads_buf.unsafe_ptr()
                var gp = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](gp_base)

                comptime if i == Self.n - 1:
                    var gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var go_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                            MutAnyOrigin,
                        ]
                    ](grad_output)
                    Self.Inner.backward_gpu[BATCH, dtype](
                        ctx, gi, go_rb, pi, ci, gp, inner_ws
                    )
                elif i == 0:
                    var go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr)
                    var gi_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.Inner.IN_DIM),
                            MutAnyOrigin,
                        ]
                    ](grad_input)
                    Self.Inner.backward_gpu[BATCH, dtype](
                        ctx, gi_rb, go, pi, ci, gp, inner_ws
                    )
                else:
                    var go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i]())
                    var gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.Inner.backward_gpu[BATCH, dtype](
                        ctx, gi, go, pi, ci, gp, inner_ws
                    )

                # Accumulate temp grads into main grads for shared weights
                comptime if Self.shared and Self.Inner.PARAM_SIZE > 0:
                    var main_grads_v = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.Inner.PARAM_SIZE),
                        MutAnyOrigin,
                    ](grads.ptr)
                    var temp_grads_v = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.Inner.PARAM_SIZE),
                        MutAnyOrigin,
                    ](temp_grads_buf.unsafe_ptr())
                    comptime ACCUM_GRID = (Self.Inner.PARAM_SIZE + TPB - 1) // TPB
                    ctx.enqueue_function[_accum_kernel, _accum_kernel](
                        main_grads_v, temp_grads_v,
                        grid_dim=(ACCUM_GRID,), block_dim=(TPB,),
                    )
