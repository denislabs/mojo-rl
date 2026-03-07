"""Repeat combinator with weight sharing.

Repeat[n: Int, Inner: Model] applies the same Inner model n times in sequence,
sharing weights across all applications.

Forward:  y = f(f(...f(x)...))  (n times)
Backward: Reverse n iterations, all accumulating into the same grads buffer.
          This correctly computes the gradient for shared weights.

Requires Inner.IN_DIM == Inner.OUT_DIM (same shape for chaining).
"""

from ...constants import dtype, TPB
from ...model.model import Model
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer


@fieldwise_init
struct Repeat[n: Int, Inner: Model](Model):
    """Weight-shared repetition: y = Inner^n(x).

    Applies Inner n times sequentially, sharing the same parameters.
    Each application has its own cache slot for correct backprop.
    """

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.Inner.OUT_DIM
    comptime PARAM_SIZE: Int = Self.Inner.PARAM_SIZE  # shared weights!
    comptime CACHE_SIZE: Int = Self.Inner.CACHE_SIZE * Self.n  # one per iter
    # Workspace: (n-1) intermediate buffers + Inner's own workspace
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
        + (Self.n - 1) * Self.Inner.OUT_DIM
    )

    # --- Offset helpers ---

    @staticmethod
    fn _cache_offset[idx: Int]() -> Int:
        """Cache offset for iteration idx."""
        return idx * Self.Inner.CACHE_SIZE

    @staticmethod
    fn _inter_offset[idx: Int]() -> Int:
        """Intermediate buffer offset for iteration idx (per sample)."""
        return idx * Self.Inner.OUT_DIM

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
        comptime if Self.n == 1:
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            Self.Inner.forward[BATCH](input, output, params, ci)
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

                comptime if i == 0:
                    var li_out = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](inter_ptr)
                    Self.Inner.forward[BATCH](input, li_out, params, ci)
                elif i == Self.n - 1:
                    var li_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](inter_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.Inner.forward[BATCH](li_in, output, params, ci)
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
                    Self.Inner.forward[BATCH](li_in, li_out, params, ci)

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
        comptime if Self.n == 1:
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
            Self.Inner.backward[BATCH](
                grad_output, grad_input, params, ci, grads
            )
        else:
            # Gradient intermediate buffer
            var grad_inter_storage = List[Scalar[dtype]](
                capacity=BATCH * (Self.n - 1) * Self.Inner.OUT_DIM
            )
            for _ in range(BATCH * (Self.n - 1) * Self.Inner.OUT_DIM):
                grad_inter_storage.append(0)
            var gi_ptr = grad_inter_storage.unsafe_ptr()

            # Reverse iteration — all use the SAME params and grads pointers
            comptime for _ri in range(Self.n):
                comptime i = Self.n - 1 - _ri

                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())

                comptime if i == Self.n - 1:
                    # Last: grad_output -> grad_inter[i-1]
                    var li_gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](gi_ptr + BATCH * Self._inter_offset[i - 1]())
                    Self.Inner.backward[BATCH](
                        grad_output, li_gi, params, ci, grads
                    )
                elif i == 0:
                    # First: grad_inter[0] -> grad_input
                    var li_go = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.OUT_DIM),
                        MutAnyOrigin,
                    ](gi_ptr)
                    Self.Inner.backward[BATCH](
                        li_go, grad_input, params, ci, grads
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
                    Self.Inner.backward[BATCH](
                        li_go, li_gi, params, ci, grads
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
    ) raises:
        # Workspace: [inter_buf_0 | ... | inter_buf_{n-2} | Inner ws]
        comptime if Self.n == 1:
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
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
            Self.Inner.forward_gpu[BATCH](
                ctx, out_rb, in_rb, params, ci, workspace
            )
        else:
            var ws_ptr = workspace.unsafe_ptr()
            comptime INNER_WS_OFF = (Self.n - 1) * Self.Inner.OUT_DIM
            var inner_ws_size = (
                BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
            )
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
                    Self.Inner.forward_gpu[BATCH](
                        ctx, inter_out, in_rb, params, ci, inner_ws
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
                    Self.Inner.forward_gpu[BATCH](
                        ctx, out_rb, inter_in, params, ci, inner_ws
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
                    Self.Inner.forward_gpu[BATCH](
                        ctx, inter_out, inter_in, params, ci, inner_ws
                    )

    # =========================================================================
    # GPU Forward (no cache)
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
    ) raises:
        # Allocate dummy cache on device
        var total_cache = BATCH * Self.CACHE_SIZE
        var dummy_cache_buf = ctx.enqueue_create_buffer[dtype](
            total_cache if total_cache > 0 else 1
        )
        var cache_v = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](dummy_cache_buf.unsafe_ptr())
        Self.forward_gpu[BATCH](ctx, output, input, params, cache_v, workspace)

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
    ) raises:
        comptime if Self.n == 1:
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr)
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
            Self.Inner.backward_gpu[BATCH](
                ctx, gi_rb, go_rb, params, ci, grads, workspace
            )
        else:
            var ws_ptr = workspace.unsafe_ptr()
            comptime INNER_WS_OFF = (Self.n - 1) * Self.Inner.OUT_DIM
            var inner_ws_size = (
                BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
            )
            var inner_ws = DeviceBuffer[dtype](
                ctx,
                ws_ptr + BATCH * INNER_WS_OFF,
                inner_ws_size if inner_ws_size > 0 else 1,
                owning=False,
            )

            # Reverse iteration
            comptime for _ri in range(Self.n):
                comptime i = Self.n - 1 - _ri

                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr + BATCH * Self._cache_offset[i]())

                comptime if i == Self.n - 1:
                    var gi = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.Inner.IN_DIM),
                        MutAnyOrigin,
                    ](ws_ptr + BATCH * Self._inter_offset[i - 1]())
                    var go_rb = rebind[
                        LayoutTensor[
                            dtype,
                            Layout.row_major(
                                BATCH, Self.Inner.OUT_DIM
                            ),
                            MutAnyOrigin,
                        ]
                    ](grad_output)
                    Self.Inner.backward_gpu[BATCH](
                        ctx, gi, go_rb, params, ci, grads, inner_ws
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
                    Self.Inner.backward_gpu[BATCH](
                        ctx, gi_rb, go, params, ci, grads, inner_ws
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
                    Self.Inner.backward_gpu[BATCH](
                        ctx, gi, go, params, ci, grads, inner_ws
                    )
