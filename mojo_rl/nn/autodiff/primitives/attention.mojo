"""Scaled Dot-Product Attention primitive.

ScaledDotProductAttention[dim, n_heads, seq_len, causal=False]:
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

Causal mode (causal=True):
  Position i only attends to positions j ≤ i (decoder/GPT-style). Implemented
  by bounding the inner j-loop to range(i + 1) — masked entries are skipped
  rather than computed-then-masked. The cache shape is unchanged; upper-
  triangle slots are never written and never read. ~2× speedup over a naive
  zero-mask in both forward and backward.

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
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.primitives import block
from std.math import exp, sqrt


struct ScaledDotProductAttention[
    dim: Int, n_heads: Int, seq_len: Int, causal: Bool = False
](DiffOp):
    """Scaled dot-product attention with multi-head support.

    Input: concatenated Q, K, V as (BATCH, seq_len * dim * 3)
    Output: attended values as (BATCH, seq_len * dim)

    Set causal=True for decoder/GPT-style attention (position i attends only
    to j ≤ i). Defaults to False for bidirectional/encoder/ViT attention.

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
        3 * Self.seq_len * Self.dim + Self.n_heads * Self.seq_len * Self.seq_len
    )
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
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
    def _q_offset() -> Int:
        return 0

    @always_inline
    @staticmethod
    def _k_offset() -> Int:
        return Self.seq_len * Self.dim

    @always_inline
    @staticmethod
    def _v_offset() -> Int:
        return 2 * Self.seq_len * Self.dim

    @always_inline
    @staticmethod
    def _attn_cache_offset() -> Int:
        return 3 * Self.seq_len * Self.dim

    # =========================================================================
    # CPU eval
    # =========================================================================

    @staticmethod
    def eval[
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
                    # j_end: in causal mode, position i attends only to j ≤ i.
                    # Comptime branch — Self.causal is a comptime parameter.
                    var j_end = Self.seq_len
                    comptime if Self.causal:
                        j_end = i + 1

                    # Find max for numerical stability
                    var max_score: Float64 = -1e30
                    for j in range(j_end):
                        var score: Float64 = 0.0
                        for d in range(Self.head_dim):
                            var q_val = Float64(
                                input.ptr[
                                    b * Self.IN_DIM + i * Self.dim + h_off + d
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
                    for j in range(j_end):
                        var attn_idx = (
                            b * Self.CACHE_SIZE
                            + Self._attn_cache_offset()
                            + h * Self.seq_len * Self.seq_len
                            + i * Self.seq_len
                            + j
                        )
                        var e = exp(Float64(cache.ptr[attn_idx]) - max_score)
                        cache.ptr[attn_idx] = Scalar[dtype](e)
                        sum_exp += e

                    var inv_sum = 1.0 / sum_exp
                    for j in range(j_end):
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
                        for j in range(j_end):
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
    def vjp[
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
                # Causal mode: attn[i,j]=0 for j>i, so dV[j] only accumulates
                # contributions from i ≥ j (equivalently j ≤ i).
                for i in range(Self.seq_len):
                    var j_end = Self.seq_len
                    comptime if Self.causal:
                        j_end = i + 1
                    for j in range(j_end):
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
                                    b * Self.OUT_DIM + i * Self.dim + h_off + d
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
                    var j_end = Self.seq_len
                    comptime if Self.causal:
                        j_end = i + 1

                    # Compute d_attn[i, j] = sum_d grad_out[i, h_off+d] * V[j, h_off+d]
                    # and the softmax backward dot product
                    var dot_sum: Float64 = 0.0
                    for j in range(j_end):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.head_dim):
                            var go = Float64(
                                grad_output.ptr[
                                    b * Self.OUT_DIM + i * Self.dim + h_off + d
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
                    for j in range(j_end):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.head_dim):
                            var go = Float64(
                                grad_output.ptr[
                                    b * Self.OUT_DIM + i * Self.dim + h_off + d
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
                                b * Self.IN_DIM + i * Self.dim + h_off + d
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

    # =========================================================================
    # GPU eval kernel — one block per (batch, head); threads stride over rows.
    # =========================================================================
    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Per-(batch, head) attention kernel.

        Grid:  (BATCH * n_heads,)
        Block: (TPB,)

        Each block handles one (b, h) pair. Threads stride over query rows i;
        for each row i the thread serially:
          1) caches Q/K/V slice for its head (per-d work fully parallel across
             threads since d ranges over head_dim and threads stride over rows
             ⇒ no cross-thread overlap in cache writes within a block);
          2) computes scores Q[i] · K[j]^T * scale into cache.attn;
          3) numerically-stable softmax over j;
          4) computes output[i,d] = Σ_j attn[i,j] * V[j,d].

        Causal mode: j-loop bounded by `i + 1`. Cache slots for j > i are
        never written and never read (backward respects the same bound).
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var blk = Int(block_idx.x)
        var b = blk // Self.n_heads
        var h = blk % Self.n_heads
        if b >= BATCH:
            return

        var h_off = h * Self.head_dim
        var tid = Int(thread_idx.x)
        var bs = Int(block_dim.x)

        # Metal does not support Float64 — keep scale in Float32.
        var scale = Scalar[dtype](Float32(1.0) / sqrt(Float32(Self.head_dim)))

        # Step 1: cache this head's Q/K/V slice (inputs are (b, t, h_off..h_off+head_dim)).
        # Threads stride over (i, d) flat indices in [seq_len * head_dim).
        var n_qkv = Self.seq_len * Self.head_dim
        var idx0 = tid
        while idx0 < n_qkv:
            var i = idx0 // Self.head_dim
            var d = idx0 % Self.head_dim
            var inp_q = b * Self.IN_DIM + i * Self.dim + h_off + d
            var inp_k = b * Self.IN_DIM + Self._k_offset() + i * Self.dim + h_off + d
            var inp_v = b * Self.IN_DIM + Self._v_offset() + i * Self.dim + h_off + d
            var c_q = b * Self.CACHE_SIZE + i * Self.dim + h_off + d
            var c_k = (
                b * Self.CACHE_SIZE + Self._k_offset() + i * Self.dim + h_off + d
            )
            var c_v = (
                b * Self.CACHE_SIZE + Self._v_offset() + i * Self.dim + h_off + d
            )
            cache.ptr[c_q] = rebind[Scalar[dtype]](input.ptr[inp_q])
            cache.ptr[c_k] = rebind[Scalar[dtype]](input.ptr[inp_k])
            cache.ptr[c_v] = rebind[Scalar[dtype]](input.ptr[inp_v])
            idx0 += bs

        # Step 2: per-row attention. Each thread strides over rows.
        var i = tid
        while i < Self.seq_len:
            var j_end = Self.seq_len
            comptime if Self.causal:
                j_end = i + 1

            # Compute raw scores into cache.attn.
            var max_score = Scalar[dtype](-1e30)
            for j in range(j_end):
                var s = Scalar[dtype](0)
                for d in range(Self.head_dim):
                    var q = rebind[Scalar[dtype]](
                        input.ptr[b * Self.IN_DIM + i * Self.dim + h_off + d]
                    )
                    var k = rebind[Scalar[dtype]](
                        input.ptr[
                            b * Self.IN_DIM
                            + Self._k_offset()
                            + j * Self.dim
                            + h_off
                            + d
                        ]
                    )
                    s += q * k
                s *= scale
                var aidx = (
                    b * Self.CACHE_SIZE
                    + Self._attn_cache_offset()
                    + h * Self.seq_len * Self.seq_len
                    + i * Self.seq_len
                    + j
                )
                cache.ptr[aidx] = s
                if s > max_score:
                    max_score = s

            # Softmax (numerically stable).
            var sum_exp = Scalar[dtype](0)
            for j in range(j_end):
                var aidx = (
                    b * Self.CACHE_SIZE
                    + Self._attn_cache_offset()
                    + h * Self.seq_len * Self.seq_len
                    + i * Self.seq_len
                    + j
                )
                var e = exp(rebind[Scalar[dtype]](cache.ptr[aidx]) - max_score)
                cache.ptr[aidx] = e
                sum_exp += e
            var inv_sum = Scalar[dtype](1) / sum_exp
            for j in range(j_end):
                var aidx = (
                    b * Self.CACHE_SIZE
                    + Self._attn_cache_offset()
                    + h * Self.seq_len * Self.seq_len
                    + i * Self.seq_len
                    + j
                )
                cache.ptr[aidx] = (
                    rebind[Scalar[dtype]](cache.ptr[aidx]) * inv_sum
                )

            # Output: attn @ V_h.
            for d in range(Self.head_dim):
                var acc = Scalar[dtype](0)
                for j in range(j_end):
                    var aidx = (
                        b * Self.CACHE_SIZE
                        + Self._attn_cache_offset()
                        + h * Self.seq_len * Self.seq_len
                        + i * Self.seq_len
                        + j
                    )
                    var v = rebind[Scalar[dtype]](
                        input.ptr[
                            b * Self.IN_DIM
                            + Self._v_offset()
                            + j * Self.dim
                            + h_off
                            + d
                        ]
                    )
                    acc += rebind[Scalar[dtype]](cache.ptr[aidx]) * v
                output.ptr[
                    b * Self.OUT_DIM + i * Self.dim + h_off + d
                ] = acc

            i += bs

    @staticmethod
    def eval_gpu[
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
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        @parameter
        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(BATCH * Self.n_heads,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU vjp
    # =========================================================================

    # =========================================================================
    # GPU vjp kernels — 4 stages:
    #   B0: zero grad_input
    #   B1: dV          (1 block per (b, h); threads stride over (j, d))
    #   B2: d_score + dQ (1 block per (b, h); threads stride over rows i;
    #                     overwrites cache.attn with d_score)
    #   B3: dK          (1 block per (b, h); threads stride over (j, d);
    #                     reads d_score from cache.attn)
    # =========================================================================

    @always_inline
    @staticmethod
    def vjp_zero_grad_input_kernel[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.IN_DIM:
            return
        grad_input.ptr[idx] = Scalar[dtype](0)

    @always_inline
    @staticmethod
    def vjp_dV_kernel[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """dV[j, h_off+d] = Σ_i attn[i,j] * grad_out[i, h_off+d].

        Causal: only i ≥ j contribute (attn[i,j]=0 for j>i, equivalently for
        i<j). So the inner loop runs i in [j, seq_len).
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var blk = Int(block_idx.x)
        var b = blk // Self.n_heads
        var h = blk % Self.n_heads
        if b >= BATCH:
            return

        var h_off = h * Self.head_dim
        var tid = Int(thread_idx.x)
        var bs = Int(block_dim.x)
        var n_jd = Self.seq_len * Self.head_dim

        var idx0 = tid
        while idx0 < n_jd:
            var j = idx0 // Self.head_dim
            var d = idx0 % Self.head_dim
            var i_start = 0
            comptime if Self.causal:
                i_start = j

            var acc = Scalar[dtype](0)
            for i in range(i_start, Self.seq_len):
                var aidx = (
                    b * Self.CACHE_SIZE
                    + Self._attn_cache_offset()
                    + h * Self.seq_len * Self.seq_len
                    + i * Self.seq_len
                    + j
                )
                var go = rebind[Scalar[dtype]](
                    grad_output.ptr[
                        b * Self.OUT_DIM + i * Self.dim + h_off + d
                    ]
                )
                acc += rebind[Scalar[dtype]](cache.ptr[aidx]) * go
            var dv_idx = (
                b * Self.IN_DIM
                + Self._v_offset()
                + j * Self.dim
                + h_off
                + d
            )
            grad_input.ptr[dv_idx] = grad_input.ptr[dv_idx] + acc
            idx0 += bs

    @always_inline
    @staticmethod
    def vjp_dscore_dQ_kernel[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """For each row i:
          1. dot_sum_i = Σ_j attn[i,j] * d_attn[i,j],  d_attn[i,j] = grad_out[i] · V[j]
          2. d_score[i,j] = attn[i,j] * (d_attn[i,j] - dot_sum_i) * scale
             (overwritten into cache.attn so dK can read it)
          3. dQ[i, h_off+d] += Σ_j d_score[i,j] * K[j, h_off+d]

        Per-row d_attn[j] is recomputed (CPU vjp does the same two-pass to
        avoid a per-row scratch buffer).
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var blk = Int(block_idx.x)
        var b = blk // Self.n_heads
        var h = blk % Self.n_heads
        if b >= BATCH:
            return

        var h_off = h * Self.head_dim
        var tid = Int(thread_idx.x)
        var bs = Int(block_dim.x)

        var scale = Scalar[dtype](Float32(1.0) / sqrt(Float32(Self.head_dim)))

        var i = tid
        while i < Self.seq_len:
            var j_end = Self.seq_len
            comptime if Self.causal:
                j_end = i + 1

            # Pass 1: compute dot_sum_i.
            var dot_sum = Scalar[dtype](0)
            for j in range(j_end):
                var d_attn = Scalar[dtype](0)
                for d in range(Self.head_dim):
                    var go = rebind[Scalar[dtype]](
                        grad_output.ptr[
                            b * Self.OUT_DIM + i * Self.dim + h_off + d
                        ]
                    )
                    var v = rebind[Scalar[dtype]](
                        cache.ptr[
                            b * Self.CACHE_SIZE
                            + Self._v_offset()
                            + j * Self.dim
                            + h_off
                            + d
                        ]
                    )
                    d_attn += go * v
                var aidx = (
                    b * Self.CACHE_SIZE
                    + Self._attn_cache_offset()
                    + h * Self.seq_len * Self.seq_len
                    + i * Self.seq_len
                    + j
                )
                var attn_w = rebind[Scalar[dtype]](cache.ptr[aidx])
                dot_sum += attn_w * d_attn

            # Pass 2: write d_score[i,j] into cache.attn (overwrites).
            for j in range(j_end):
                var d_attn = Scalar[dtype](0)
                for d in range(Self.head_dim):
                    var go = rebind[Scalar[dtype]](
                        grad_output.ptr[
                            b * Self.OUT_DIM + i * Self.dim + h_off + d
                        ]
                    )
                    var v = rebind[Scalar[dtype]](
                        cache.ptr[
                            b * Self.CACHE_SIZE
                            + Self._v_offset()
                            + j * Self.dim
                            + h_off
                            + d
                        ]
                    )
                    d_attn += go * v
                var aidx = (
                    b * Self.CACHE_SIZE
                    + Self._attn_cache_offset()
                    + h * Self.seq_len * Self.seq_len
                    + i * Self.seq_len
                    + j
                )
                var attn_w = rebind[Scalar[dtype]](cache.ptr[aidx])
                var d_score = attn_w * (d_attn - dot_sum) * scale
                cache.ptr[aidx] = d_score

            # Pass 3: dQ[i, h_off+d] = Σ_j d_score[i,j] * K[j, h_off+d].
            for d in range(Self.head_dim):
                var acc = Scalar[dtype](0)
                for j in range(j_end):
                    var aidx = (
                        b * Self.CACHE_SIZE
                        + Self._attn_cache_offset()
                        + h * Self.seq_len * Self.seq_len
                        + i * Self.seq_len
                        + j
                    )
                    var d_score = rebind[Scalar[dtype]](cache.ptr[aidx])
                    var k = rebind[Scalar[dtype]](
                        cache.ptr[
                            b * Self.CACHE_SIZE
                            + Self._k_offset()
                            + j * Self.dim
                            + h_off
                            + d
                        ]
                    )
                    acc += d_score * k
                var dq_idx = (
                    b * Self.IN_DIM + i * Self.dim + h_off + d
                )
                grad_input.ptr[dq_idx] = grad_input.ptr[dq_idx] + acc

            i += bs

    @always_inline
    @staticmethod
    def vjp_dK_kernel[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """dK[j, h_off+d] = Σ_i d_score[i,j] * Q[i, h_off+d].

        Reads d_score from cache.attn (vjp_dscore_dQ_kernel overwrote it).
        Causal: only i ≥ j contribute.
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var blk = Int(block_idx.x)
        var b = blk // Self.n_heads
        var h = blk % Self.n_heads
        if b >= BATCH:
            return

        var h_off = h * Self.head_dim
        var tid = Int(thread_idx.x)
        var bs = Int(block_dim.x)
        var n_jd = Self.seq_len * Self.head_dim

        var idx0 = tid
        while idx0 < n_jd:
            var j = idx0 // Self.head_dim
            var d = idx0 % Self.head_dim
            var i_start = 0
            comptime if Self.causal:
                i_start = j

            var acc = Scalar[dtype](0)
            for i in range(i_start, Self.seq_len):
                var aidx = (
                    b * Self.CACHE_SIZE
                    + Self._attn_cache_offset()
                    + h * Self.seq_len * Self.seq_len
                    + i * Self.seq_len
                    + j
                )
                var d_score = rebind[Scalar[dtype]](cache.ptr[aidx])
                var q = rebind[Scalar[dtype]](
                    cache.ptr[b * Self.CACHE_SIZE + i * Self.dim + h_off + d]
                )
                acc += d_score * q
            var dk_idx = (
                b * Self.IN_DIM
                + Self._k_offset()
                + j * Self.dim
                + h_off
                + d
            )
            grad_input.ptr[dk_idx] = grad_input.ptr[dk_idx] + acc
            idx0 += bs

    @staticmethod
    def vjp_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        # B0: zero grad_input.
        @parameter
        @always_inline
        def zero_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
        ):
            Self.vjp_zero_grad_input_kernel[BATCH, dtype](grad_input)

        comptime ZERO_BLOCKS = (
            BATCH * Self.IN_DIM + TPB - 1
        ) // TPB
        ctx.enqueue_function[zero_wrapper, zero_wrapper](
            grad_input,
            grid_dim=(ZERO_BLOCKS,),
            block_dim=(TPB,),
        )

        # B1: dV.
        @parameter
        @always_inline
        def dV_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.vjp_dV_kernel[BATCH, dtype](grad_input, grad_output, cache)

        ctx.enqueue_function[dV_wrapper, dV_wrapper](
            grad_input,
            go_immut,
            cache_immut,
            grid_dim=(BATCH * Self.n_heads,),
            block_dim=(TPB,),
        )

        # B2: d_score + dQ (mutates cache.attn → d_score; reads cache.K/V).
        @parameter
        @always_inline
        def dscore_dQ_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.vjp_dscore_dQ_kernel[BATCH, dtype](
                grad_input, grad_output, cache
            )

        ctx.enqueue_function[dscore_dQ_wrapper, dscore_dQ_wrapper](
            grad_input,
            go_immut,
            cache,
            grid_dim=(BATCH * Self.n_heads,),
            block_dim=(TPB,),
        )

        # B3: dK (reads d_score from cache.attn; reads cache.Q).
        @parameter
        @always_inline
        def dK_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.vjp_dK_kernel[BATCH, dtype](grad_input, cache)

        ctx.enqueue_function[dK_wrapper, dK_wrapper](
            grad_input,
            cache_immut,
            grid_dim=(BATCH * Self.n_heads,),
            block_dim=(TPB,),
        )
