"""Scaled Dot-Product Attention primitive.

ScaledDotProductAttention[dim, n_heads, seq_len]:
  Input:  (BATCH, seq_len * dim * 3)  — concatenated Q, K, V projections
  Output: (BATCH, seq_len * dim)      — attended values

Each sample contains a full sequence of tokens. Q, K, V are pre-projected
(use MatMul + BiasAdd before this op for the projection).

The input layout per sample is:
  [Q_0, Q_1, ..., Q_{seq-1}, K_0, K_1, ..., K_{seq-1}, V_0, V_1, ..., V_{seq-1}]
where each Q_t, K_t, V_t is a dim-dimensional vector.

For multi-head attention, dim must be divisible by n_heads. Each head
operates on head_dim = dim // n_heads dimensions independently.

Forward:
  For each head h:
    scores = Q_h @ K_h^T / sqrt(head_dim)   — (seq_len, seq_len)
    attn_weights = softmax(scores, dim=-1)    — (seq_len, seq_len)
    output_h = attn_weights @ V_h            — (seq_len, head_dim)
  Concatenate heads → output (seq_len, dim)

Backward:
  Standard attention backward through softmax and matmuls.
  Requires cached Q, K, V, and attention weights.

Cache layout per sample:
  [Q | K | V | attn_weights]
  Q, K, V: seq_len * dim each
  attn_weights: n_heads * seq_len * seq_len
  Total: 3 * seq_len * dim + n_heads * seq_len * seq_len
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import exp, sqrt


struct ScaledDotProductAttention[dim: Int, n_heads: Int, seq_len: Int](
    DiffOp
):
    """Scaled dot-product attention with multi-head support.

    Input: concatenated Q, K, V as (BATCH, seq_len * dim * 3)
    Output: attended values as (BATCH, seq_len * dim)

    No learnable parameters — projections should be done by MatMul ops
    before this op in the chain.
    """

    comptime head_dim: Int = Self.dim // Self.n_heads
    comptime OP_ID: Int = OpID.SCALED_DOT_PRODUCT_ATTENTION._value
    comptime IN_DIM: Int = Self.seq_len * Self.dim * 3
    comptime OUT_DIM: Int = Self.seq_len * Self.dim
    comptime PARAM_SIZE: Int = 0
    # Cache: Q + K + V + attention weights
    comptime CACHE_SIZE: Int = (
        3 * Self.seq_len * Self.dim
        + Self.n_heads * Self.seq_len * Self.seq_len
    )

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # Helper: index into flattened per-sample data
    # =========================================================================
    # Q offset in input: 0
    # K offset in input: seq_len * dim
    # V offset in input: 2 * seq_len * dim
    # Attn weights offset in cache (after Q,K,V): 3 * seq_len * dim

    @always_inline
    @staticmethod
    fn _q_offset() -> Int:
        return 0

    @always_inline
    @staticmethod
    fn _k_offset() -> Int:
        return Self.seq_len * Self.dim

    @always_inline
    @staticmethod
    fn _v_offset() -> Int:
        return 2 * Self.seq_len * Self.dim

    @always_inline
    @staticmethod
    fn _attn_cache_offset() -> Int:
        return 3 * Self.seq_len * Self.dim

    # =========================================================================
    # CPU eval
    # =========================================================================

    @staticmethod
    fn eval[
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
        var scale = 1.0 / sqrt(Float64(Self.head_dim))

        for b in range(BATCH):
            # Cache Q, K, V from input
            for i in range(Self.seq_len * Self.dim):
                cache.ptr[b * Self.CACHE_SIZE + i] = input.ptr[
                    b * Self.IN_DIM + i
                ]  # Q
                cache.ptr[
                    b * Self.CACHE_SIZE + Self.seq_len * Self.dim + i
                ] = input.ptr[
                    b * Self.IN_DIM + Self.seq_len * Self.dim + i
                ]  # K
                cache.ptr[
                    b * Self.CACHE_SIZE + 2 * Self.seq_len * Self.dim + i
                ] = input.ptr[
                    b * Self.IN_DIM + 2 * Self.seq_len * Self.dim + i
                ]  # V

            # For each head
            for h in range(Self.n_heads):
                var h_off = h * Self.head_dim

                # Compute attention scores: Q_h @ K_h^T / sqrt(head_dim)
                # scores[i, j] = sum_d Q[i, h*hd + d] * K[j, h*hd + d] / sqrt(hd)
                for i in range(Self.seq_len):
                    # Find max for numerical stability
                    var max_score: Float64 = -1e30
                    for j in range(Self.seq_len):
                        var score: Float64 = 0.0
                        for d in range(Self.head_dim):
                            var q_val = Float64(
                                input.ptr[
                                    b * Self.IN_DIM
                                    + i * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            var k_val = Float64(
                                input.ptr[
                                    b * Self.IN_DIM
                                    + Self.seq_len * Self.dim
                                    + j * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            score += q_val * k_val
                        score *= scale
                        # Store raw score temporarily in cache attn_weights slot
                        var attn_idx = (
                            b * Self.CACHE_SIZE
                            + Self._attn_cache_offset()
                            + h * Self.seq_len * Self.seq_len
                            + i * Self.seq_len
                            + j
                        )
                        cache.ptr[attn_idx] = Scalar[dtype](score)
                        if score > max_score:
                            max_score = score

                    # Softmax over j dimension
                    var sum_exp: Float64 = 0.0
                    for j in range(Self.seq_len):
                        var attn_idx = (
                            b * Self.CACHE_SIZE
                            + Self._attn_cache_offset()
                            + h * Self.seq_len * Self.seq_len
                            + i * Self.seq_len
                            + j
                        )
                        var e = exp(
                            Float64(cache.ptr[attn_idx]) - max_score
                        )
                        cache.ptr[attn_idx] = Scalar[dtype](e)
                        sum_exp += e

                    var inv_sum = 1.0 / sum_exp
                    for j in range(Self.seq_len):
                        var attn_idx = (
                            b * Self.CACHE_SIZE
                            + Self._attn_cache_offset()
                            + h * Self.seq_len * Self.seq_len
                            + i * Self.seq_len
                            + j
                        )
                        cache.ptr[attn_idx] = Scalar[dtype](
                            Float64(cache.ptr[attn_idx]) * inv_sum
                        )

                    # Compute output: attn_weights @ V_h
                    for d in range(Self.head_dim):
                        var acc: Float64 = 0.0
                        for j in range(Self.seq_len):
                            var attn_idx = (
                                b * Self.CACHE_SIZE
                                + Self._attn_cache_offset()
                                + h * Self.seq_len * Self.seq_len
                                + i * Self.seq_len
                                + j
                            )
                            var v_val = Float64(
                                input.ptr[
                                    b * Self.IN_DIM
                                    + 2 * Self.seq_len * Self.dim
                                    + j * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            acc += Float64(cache.ptr[attn_idx]) * v_val
                        output.ptr[
                            b * Self.OUT_DIM + i * Self.dim + h_off + d
                        ] = Scalar[dtype](acc)

    # =========================================================================
    # CPU vjp
    # =========================================================================

    @staticmethod
    fn vjp[
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass for scaled dot-product attention.

        grad_input layout matches input: [dQ | dK | dV]
        Uses cached Q, K, V, and attention weights.
        """
        var scale = 1.0 / sqrt(Float64(Self.head_dim))

        # Zero grad_input
        for i in range(BATCH * Self.IN_DIM):
            grad_input.ptr[i] = 0

        for b in range(BATCH):
            for h in range(Self.n_heads):
                var h_off = h * Self.head_dim

                # Step 1: dV and d(attn_weights) from output gradient
                # output = attn @ V  →  dV = attn^T @ grad_out,  d_attn = grad_out @ V^T
                for i in range(Self.seq_len):
                    for j in range(Self.seq_len):
                        var attn_idx = (
                            b * Self.CACHE_SIZE
                            + Self._attn_cache_offset()
                            + h * Self.seq_len * Self.seq_len
                            + i * Self.seq_len
                            + j
                        )
                        var attn_w = Float64(cache.ptr[attn_idx])

                        for d in range(Self.head_dim):
                            var go = Float64(
                                grad_output.ptr[
                                    b * Self.OUT_DIM
                                    + i * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            # dV[j, h_off+d] += attn[i,j] * grad_out[i, h_off+d]
                            var dv_idx = (
                                b * Self.IN_DIM
                                + 2 * Self.seq_len * Self.dim
                                + j * Self.dim
                                + h_off
                                + d
                            )
                            grad_input.ptr[dv_idx] = grad_input.ptr[
                                dv_idx
                            ] + Scalar[dtype](attn_w * go)

                # Step 2: d(attn_weights) — compute grad through softmax
                # d_attn_pre_softmax[i,j] = grad_out[i] @ V[j] (dot product over head_dim)
                # Then softmax backward: d_score[i,j] = attn[i,j] * (d_attn[i,j] - dot_j)
                for i in range(Self.seq_len):
                    # Compute d_attn[i, j] = sum_d grad_out[i, h_off+d] * V[j, h_off+d]
                    # and the softmax backward dot product
                    var dot_sum: Float64 = 0.0
                    for j in range(Self.seq_len):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.head_dim):
                            var go = Float64(
                                grad_output.ptr[
                                    b * Self.OUT_DIM
                                    + i * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            var v_val = Float64(
                                cache.ptr[
                                    b * Self.CACHE_SIZE
                                    + 2 * Self.seq_len * Self.dim
                                    + j * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            d_attn += go * v_val

                        var attn_idx = (
                            b * Self.CACHE_SIZE
                            + Self._attn_cache_offset()
                            + h * Self.seq_len * Self.seq_len
                            + i * Self.seq_len
                            + j
                        )
                        var attn_w = Float64(cache.ptr[attn_idx])
                        dot_sum += d_attn * attn_w

                        # Temporarily store d_attn in a local computation
                        # We need both d_attn and dot_sum, so we do two passes

                    # Second pass: compute d_score and propagate to dQ, dK
                    for j in range(Self.seq_len):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.head_dim):
                            var go = Float64(
                                grad_output.ptr[
                                    b * Self.OUT_DIM
                                    + i * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            var v_val = Float64(
                                cache.ptr[
                                    b * Self.CACHE_SIZE
                                    + 2 * Self.seq_len * Self.dim
                                    + j * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            d_attn += go * v_val

                        var attn_idx = (
                            b * Self.CACHE_SIZE
                            + Self._attn_cache_offset()
                            + h * Self.seq_len * Self.seq_len
                            + i * Self.seq_len
                            + j
                        )
                        var attn_w = Float64(cache.ptr[attn_idx])

                        # Softmax backward: d_score = attn * (d_attn - dot_sum)
                        var d_score = attn_w * (d_attn - dot_sum) * scale

                        # d_score = dL/d(Q[i] . K[j] / sqrt(hd))
                        # dQ[i, d] += d_score * K[j, d]
                        # dK[j, d] += d_score * Q[i, d]
                        for d in range(Self.head_dim):
                            var q_val = Float64(
                                cache.ptr[
                                    b * Self.CACHE_SIZE
                                    + i * Self.dim
                                    + h_off
                                    + d
                                ]
                            )
                            var k_val = Float64(
                                cache.ptr[
                                    b * Self.CACHE_SIZE
                                    + Self.seq_len * Self.dim
                                    + j * Self.dim
                                    + h_off
                                    + d
                                ]
                            )

                            # dQ[i, h_off+d]
                            var dq_idx = (
                                b * Self.IN_DIM
                                + i * Self.dim
                                + h_off
                                + d
                            )
                            grad_input.ptr[dq_idx] = grad_input.ptr[
                                dq_idx
                            ] + Scalar[dtype](d_score * k_val)

                            # dK[j, h_off+d]
                            var dk_idx = (
                                b * Self.IN_DIM
                                + Self.seq_len * Self.dim
                                + j * Self.dim
                                + h_off
                                + d
                            )
                            grad_input.ptr[dk_idx] = grad_input.ptr[
                                dk_idx
                            ] + Scalar[dtype](d_score * q_val)

    # =========================================================================
    # GPU eval
    # =========================================================================

    @staticmethod
    fn eval_gpu[
        BATCH: Int
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
    ) raises:
        # CPU fallback for now — GPU attention kernel is a future optimization
        Self.eval[BATCH](input, output, params, cache)

    # =========================================================================
    # GPU vjp
    # =========================================================================

    @staticmethod
    fn vjp_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ) raises:
        # CPU fallback for now — GPU attention kernel is a future optimization
        Self.vjp[BATCH](grad_output, grad_input, params, cache, grad_params)
