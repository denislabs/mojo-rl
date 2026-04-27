"""Tokenwise combinator — apply the same Model to each of seq_len tokens.

Tokenwise[seq_len, Inner] applies Inner to each `Inner.IN_DIM`-sized token
in parallel with shared weights. Equivalent to reshaping
  (BATCH, seq_len * Inner.IN_DIM)  →  (BATCH * seq_len, Inner.IN_DIM)
calling Inner once at the expanded batch size, then reshaping back.

This is the standard transformer pattern for QKV projections, output
projections, FFN layers, and per-token LayerNorm: weights are shared
across positions, each position is processed independently.

Forward:  for each batch and each token t, output[batch, t] = Inner(input[batch, t])
Backward: gradients accumulate into the shared parameter buffer across all
          (batch, token) pairs — same as Inner running at BATCH * seq_len.

Compile-time:
    PARAM_SIZE = Inner.PARAM_SIZE                  (shared)
    CACHE_SIZE = seq_len * Inner.CACHE_SIZE        (per-token)
    STATE_SIZE = Inner.STATE_SIZE                  (shared)
    WORKSPACE_SIZE_PER_SAMPLE = seq_len * Inner.WORKSPACE_SIZE_PER_SAMPLE
"""

from ...constants import dtype, TPB
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


@fieldwise_init
struct Tokenwise[seq_len: Int, Inner: Model](Model):
    """Apply Inner to each of seq_len tokens with shared weights.

    Input/output shapes (per sample):
        input:  seq_len * Inner.IN_DIM
        output: seq_len * Inner.OUT_DIM
    """

    comptime IN_DIM: Int = Self.seq_len * Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.seq_len * Self.Inner.OUT_DIM
    comptime PARAM_SIZE: Int = Self.Inner.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.seq_len * Self.Inner.CACHE_SIZE
    comptime STATE_SIZE: Int = Self.Inner.STATE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.seq_len * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
    )

    # =========================================================================
    # Initialization (delegate to Inner — params are shared across tokens)
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        var inner_p = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.Inner.initialize_params[INIT, dtype](inner_p)

    @staticmethod
    def zero_biases[dtype: DType = DType.float32](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        var inner_p = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.Inner.zero_biases[dtype](inner_p)

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        comptime if Self.Inner.STATE_SIZE > 0:
            var inner_s = LayoutTensor[
                dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
            ](state.ptr)
            Self.Inner.initialize_state[dtype](inner_s)

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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        # Reinterpret (BATCH, seq_len * Inner.IN_DIM) as
        # (BATCH * seq_len, Inner.IN_DIM). Memory layout is identical:
        # row-major over outer-batch-then-token already matches what Inner
        # expects when we just expand the batch axis.
        var inner_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.IN_DIM),
            MutAnyOrigin,
        ](input.ptr)
        var inner_out = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.OUT_DIM),
            MutAnyOrigin,
        ](output.ptr)
        var inner_cache = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.CACHE_SIZE),
            MutAnyOrigin,
        ](cache.ptr)
        var inner_p = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var inner_s = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        Self.Inner.forward[BATCH * Self.seq_len, dtype](
            inner_in, inner_out, inner_p, inner_s, inner_cache
        )

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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        var inner_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.IN_DIM),
            MutAnyOrigin,
        ](input.ptr)
        var inner_out = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.OUT_DIM),
            MutAnyOrigin,
        ](output.ptr)
        var inner_p = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var inner_s = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        Self.Inner.forward[BATCH * Self.seq_len, dtype](
            inner_in, inner_out, inner_p, inner_s
        )

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
        var inner_go = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.OUT_DIM),
            MutAnyOrigin,
        ](grad_output.ptr)
        var inner_gi = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.IN_DIM),
            MutAnyOrigin,
        ](grad_input.ptr)
        var inner_cache = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.CACHE_SIZE),
            MutAnyOrigin,
        ](cache.ptr)
        var inner_p = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var inner_s = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var inner_gp = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.Inner.backward[BATCH * Self.seq_len, dtype](
            inner_go, inner_gi, inner_p, inner_s, inner_cache, inner_gp
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
        var inner_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.IN_DIM),
            MutAnyOrigin,
        ](input.ptr)
        var inner_out = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.OUT_DIM),
            MutAnyOrigin,
        ](output.ptr)
        var inner_cache = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.CACHE_SIZE),
            MutAnyOrigin,
        ](cache.ptr)
        var inner_p = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var inner_s = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        Self.Inner.forward_gpu[BATCH * Self.seq_len, dtype](
            ctx, inner_out, inner_in, inner_p, inner_s, inner_cache, workspace
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var inner_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.IN_DIM),
            MutAnyOrigin,
        ](input.ptr)
        var inner_out = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.OUT_DIM),
            MutAnyOrigin,
        ](output.ptr)
        var inner_p = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var inner_s = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        Self.Inner.forward_gpu_no_cache[BATCH * Self.seq_len, dtype](
            ctx, inner_out, inner_in, inner_p, inner_s, workspace
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.forward_gpu_no_cache[BATCH, dtype](
            ctx, output, input, params, state, workspace
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
        var inner_go = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.OUT_DIM),
            MutAnyOrigin,
        ](grad_output.ptr)
        var inner_gi = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.IN_DIM),
            MutAnyOrigin,
        ](grad_input.ptr)
        var inner_cache = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * Self.seq_len, Self.Inner.CACHE_SIZE),
            MutAnyOrigin,
        ](cache.ptr)
        var inner_p = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var inner_s = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var inner_gp = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.Inner.backward_gpu[BATCH * Self.seq_len, dtype](
            ctx,
            inner_gi,
            inner_go,
            inner_p,
            inner_s,
            inner_cache,
            inner_gp,
            workspace,
        )
