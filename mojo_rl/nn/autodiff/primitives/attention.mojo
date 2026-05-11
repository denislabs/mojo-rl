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
from std.runtime.asyncrt import DeviceContextPtr
from std.gpu.primitives import block
from std.math import exp, sqrt
from std.sys import has_nvidia_gpu_accelerator
from linalg.bmm import batched_matmul
from layout.tile_tensor import TileTensor, lt_to_tt
from layout import row_major


struct ScaledDotProductAttention[
    dim: Int,
    n_heads: Int,
    seq_len: Int,
    causal: Bool = False,
    USE_MAX_KERNELS: Bool = True,
](DiffOp):
    """Scaled dot-product attention with multi-head support.

    Input: concatenated Q, K, V as (BATCH, seq_len * dim * 3)
    Output: attended values as (BATCH, seq_len * dim)

    Set causal=True for decoder/GPT-style attention (position i attends only
    to j ≤ i). Defaults to False for bidirectional/encoder/ViT attention.

    USE_MAX_KERNELS (NVIDIA only): when True, route the QK^T and AV matmuls
    through `linalg.bmm.batched_matmul` (single batched GEMM call covering
    all BATCH*n_heads matmuls each), with softmax kept as a custom kernel.
    When False, use the existing single-kernel-per-(b,h) implementation that
    does scalar Q·K dot products inline. Apple is unaffected — always uses
    the existing per-(b,h) kernel regardless of this flag.

    Backward (vjp_gpu) is currently always on the per-(b,h) custom path
    regardless of this flag — phase 4c will add bmm support there.

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

        comptime if Self.USE_MAX_KERNELS and has_nvidia_gpu_accelerator():
            Self._eval_gpu_bmm[BATCH, dtype](
                ctx, output, input_immut, cache
            )
        else:

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

            ctx.enqueue_function[wrapper](
                output,
                input_immut,
                cache,
                grid_dim=(BATCH * Self.n_heads,),
                block_dim=(TPB,),
            )

    # =========================================================================
    # GPU eval — bmm path (NVIDIA, USE_MAX_KERNELS=True)
    # =========================================================================
    # Pipeline:
    #   1. pack_qkv:  (BATCH, seq, n_heads*head_dim) → (BATCH*n_heads, seq, head_dim)
    #                 contiguous packed buffers (also writes Q/K/V to cache for backward).
    #   2. bmm[transpose_b]: scratch_scores = packed_Q @ packed_K^T,
    #                        shape (BATCH*n_heads, seq, seq).
    #   3. softmax_kernel: scale + (causal mask) + softmax in scratch_scores,
    #                      and gather-write attention weights into cache.attn (with
    #                      per-sample stride CACHE_SIZE — needed for backward).
    #   4. bmm: packed_out = scratch_scores @ packed_V,
    #           shape (BATCH*n_heads, seq, head_dim).
    #   5. unpack_out: (BATCH*n_heads, seq, head_dim) → (BATCH, seq, n_heads*head_dim).
    # Total launches: 5, regardless of BATCH/n_heads.
    @staticmethod
    def _eval_gpu_bmm[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ) raises:
        comptime BH = BATCH * Self.n_heads
        comptime PACKED_QKV_SIZE = BATCH * Self.seq_len * Self.dim
        comptime SCORES_SIZE = BH * Self.seq_len * Self.seq_len

        # Scratch buffers — released after the forward returns.
        var packed_q_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var packed_k_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var packed_v_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var packed_out_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var scores_buf = ctx.enqueue_create_buffer[dtype](SCORES_SIZE)

        # ── 1. Pack Q/K/V from (B, seq, n_heads*head_dim) → (B*H, seq, head_dim).
        # Also writes Q/K/V to cache (cache layout unchanged: per-sample
        # [Q | K | V | attn], with seq*dim each for QKV).
        comptime pack_elems = BATCH * Self.seq_len * Self.dim
        comptime pack_blocks = (pack_elems + TPB - 1) // TPB

        @parameter
        @always_inline
        def pack_qkv_wrapper(
            packed_q: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
            ],
            packed_k: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
            ],
            packed_v: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= pack_elems:
                return
            # idx unrolls to (b, t, h, d) over BATCH x seq x n_heads x head_dim
            var hd = Self.head_dim
            var d = idx % hd
            var rem = idx // hd
            var h = rem % Self.n_heads
            var rem2 = rem // Self.n_heads
            var t = rem2 % Self.seq_len
            var b = rem2 // Self.seq_len

            var col_in_dim = h * hd + d  # head_dim chunk inside dim

            # Source positions in input (input layout: per-sample
            # [Q_0..Q_seq | K_0..K_seq | V_0..V_seq], each Q_t/K_t/V_t is dim-wide)
            var inp_q = (
                b * Self.IN_DIM + t * Self.dim + col_in_dim
            )
            var inp_k = (
                b * Self.IN_DIM
                + Self._k_offset()
                + t * Self.dim
                + col_in_dim
            )
            var inp_v = (
                b * Self.IN_DIM
                + Self._v_offset()
                + t * Self.dim
                + col_in_dim
            )

            # Cache positions (same layout as before — backward reads here)
            var c_q = b * Self.CACHE_SIZE + t * Self.dim + col_in_dim
            var c_k = (
                b * Self.CACHE_SIZE
                + Self._k_offset()
                + t * Self.dim
                + col_in_dim
            )
            var c_v = (
                b * Self.CACHE_SIZE
                + Self._v_offset()
                + t * Self.dim
                + col_in_dim
            )

            # Packed positions: (BATCH*n_heads, seq, head_dim) flat
            var bh = b * Self.n_heads + h
            var packed_idx = bh * Self.seq_len * hd + t * hd + d

            var q_val = input.ptr[inp_q]
            var k_val = input.ptr[inp_k]
            var v_val = input.ptr[inp_v]

            cache.ptr[c_q] = q_val
            cache.ptr[c_k] = k_val
            cache.ptr[c_v] = v_val
            packed_q.ptr[packed_idx] = q_val
            packed_k.ptr[packed_idx] = k_val
            packed_v.ptr[packed_idx] = v_val

        var packed_q_lt = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
        ](packed_q_buf.unsafe_ptr())
        var packed_k_lt = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
        ](packed_k_buf.unsafe_ptr())
        var packed_v_lt = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
        ](packed_v_buf.unsafe_ptr())

        ctx.enqueue_function[pack_qkv_wrapper](
            packed_q_lt,
            packed_k_lt,
            packed_v_lt,
            cache,
            input,
            grid_dim=(pack_blocks,),
            block_dim=(TPB,),
        )

        # ── 2. bmm[transpose_b]: scratch_scores = packed_Q @ packed_K^T.
        var scores_tt = TileTensor(
            scores_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.seq_len](),
        )
        var packed_q_tt = TileTensor(
            packed_q_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        var packed_k_tt = TileTensor(
            packed_k_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        batched_matmul[transpose_b=True, target="gpu"](
            scores_tt,
            packed_q_tt,
            packed_k_tt,
            context=DeviceContextPtr(ctx),
        )

        # ── 3. Softmax: scale + (optional causal) + numerically-stable softmax
        # in-place on scores_buf. Also gather-writes the resulting attention
        # weights into cache.attn (per-sample-strided) so backward can read
        # them from the existing cache layout.
        var scale = Scalar[dtype](Float32(1.0) / sqrt(Float32(Self.head_dim)))

        @parameter
        @always_inline
        def softmax_wrapper(
            scores: LayoutTensor[
                dtype, Layout.row_major(SCORES_SIZE), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
            scale_v: Scalar[dtype],
        ):
            comptime assert dtype.is_floating_point(), "dtype must be floating point"
            # 1 block per (b, h) — threads stride over rows i.
            var blk = Int(block_idx.x)
            if blk >= BH:
                return
            var b = blk // Self.n_heads
            var h = blk % Self.n_heads
            var tid = Int(thread_idx.x)
            var bs = Int(block_dim.x)

            var bh_off = blk * Self.seq_len * Self.seq_len
            var cache_attn_base = (
                b * Self.CACHE_SIZE
                + Self._attn_cache_offset()
                + h * Self.seq_len * Self.seq_len
            )

            var i = tid
            while i < Self.seq_len:
                var j_end = Self.seq_len
                comptime if Self.causal:
                    j_end = i + 1

                var row_off = bh_off + i * Self.seq_len
                var cache_row_off = cache_attn_base + i * Self.seq_len

                # Apply scale + find max (causal: ignore j > i).
                var max_score = Scalar[dtype](-1e30)
                for j in range(j_end):
                    var s = (
                        rebind[Scalar[dtype]](scores.ptr[row_off + j])
                        * scale_v
                    )
                    scores.ptr[row_off + j] = s
                    if s > max_score:
                        max_score = s

                # Exponentiate (zero out masked region first so the second
                # bmm still produces correct output in the causal case).
                var sum_exp = Scalar[dtype](0)
                for j in range(j_end):
                    var e = exp(
                        rebind[Scalar[dtype]](scores.ptr[row_off + j])
                        - max_score
                    )
                    scores.ptr[row_off + j] = e
                    sum_exp += e
                comptime if Self.causal:
                    for j in range(i + 1, Self.seq_len):
                        scores.ptr[row_off + j] = Scalar[dtype](0)

                var inv_sum = Scalar[dtype](1) / sum_exp
                for j in range(j_end):
                    var w = (
                        rebind[Scalar[dtype]](scores.ptr[row_off + j])
                        * inv_sum
                    )
                    scores.ptr[row_off + j] = w
                    cache.ptr[cache_row_off + j] = w

                # Causal: also zero cache.attn[i, j > i] so the backward
                # softmax_jvp reads zeros (not uninitialized memory) for
                # the masked positions. The custom backward avoids these
                # slots entirely via bounded loops, so the OFF path doesn't
                # need this — but the bmm backward computes a full (seq,
                # seq) dscore and relies on a[i, j > i] = 0 to mask out.
                comptime if Self.causal:
                    for j in range(i + 1, Self.seq_len):
                        cache.ptr[cache_row_off + j] = Scalar[dtype](0)

                i += bs

        var scores_lt = LayoutTensor[
            dtype, Layout.row_major(SCORES_SIZE), MutAnyOrigin
        ](scores_buf.unsafe_ptr())

        ctx.enqueue_function[softmax_wrapper](
            scores_lt,
            cache,
            scale,
            grid_dim=(BH,),
            block_dim=(TPB,),
        )

        # ── 4. bmm: packed_out = attn @ V.
        var packed_out_tt = TileTensor(
            packed_out_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        var packed_v_tt = TileTensor(
            packed_v_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        batched_matmul[target="gpu"](
            packed_out_tt,
            scores_tt,
            packed_v_tt,
            context=DeviceContextPtr(ctx),
        )

        # ── 5. Unpack output: (BH, seq, head_dim) → (BATCH, seq, n_heads*head_dim).
        comptime unpack_elems = BATCH * Self.seq_len * Self.dim
        comptime unpack_blocks = (unpack_elems + TPB - 1) // TPB

        @parameter
        @always_inline
        def unpack_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            packed_out: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= unpack_elems:
                return
            var hd = Self.head_dim
            var d = idx % hd
            var rem = idx // hd
            var h = rem % Self.n_heads
            var rem2 = rem // Self.n_heads
            var t = rem2 % Self.seq_len
            var b = rem2 // Self.seq_len

            var bh = b * Self.n_heads + h
            var packed_idx = bh * Self.seq_len * hd + t * hd + d
            var out_idx = (
                b * Self.OUT_DIM + t * Self.dim + h * hd + d
            )
            output.ptr[out_idx] = packed_out.ptr[packed_idx]

        var packed_out_lt = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
        ](packed_out_buf.unsafe_ptr())

        ctx.enqueue_function[unpack_wrapper](
            output,
            packed_out_lt,
            grid_dim=(unpack_blocks,),
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

        comptime if Self.USE_MAX_KERNELS and has_nvidia_gpu_accelerator():
            Self._vjp_gpu_bmm[BATCH, dtype](
                ctx, go_immut, grad_input, cache_immut
            )
            return

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
        ctx.enqueue_function[zero_wrapper](
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

        ctx.enqueue_function[dV_wrapper](
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

        ctx.enqueue_function[dscore_dQ_wrapper](
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

        ctx.enqueue_function[dK_wrapper](
            grad_input,
            cache_immut,
            grid_dim=(BATCH * Self.n_heads,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU vjp — bmm path (NVIDIA, USE_MAX_KERNELS=True)
    # =========================================================================
    # Pipeline (9 launches, regardless of BATCH/n_heads):
    #   1. pack_in:    grad_output + cache.{Q,K,V} → packed dout/Q/K/V (BH, seq, d)
    #   2. bmm[t_b]:   dattn = dout @ V^T,                shape (BH, seq, seq)
    #   3. softmax_jvp: dscore = scale * a * (dattn - sum_k a_k * dattn_k);
    #                   reads cache.attn (per-sample-strided), writes contiguous.
    #   4. transpose:  attn_T[bh, j, i] = cache.attn[bh, i, j]
    #   5. bmm:        dV = attn_T @ dout,                shape (BH, seq, d)
    #   6. transpose:  dscore_T[bh, j, i] = dscore[bh, i, j]
    #   7. bmm:        dK = dscore_T @ Q,                 shape (BH, seq, d)
    #   8. bmm:        dQ = dscore @ K,                   shape (BH, seq, d)
    #   9. unpack_out: dQ/dK/dV → grad_input ([Q_grad | K_grad | V_grad] per
    #                  sample, with per-head reshape).
    # Causal: handled implicitly because cache.attn[i, j] = 0 for j > i, so
    # the softmax_jvp kernel naturally produces dscore[i, j] = 0 there.
    @staticmethod
    def _vjp_gpu_bmm[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ) raises:
        comptime BH = BATCH * Self.n_heads
        comptime PACKED_QKV_SIZE = BATCH * Self.seq_len * Self.dim
        comptime SCORES_SIZE = BH * Self.seq_len * Self.seq_len

        # Scratch buffers (10 of them — released after the backward returns).
        var packed_dout_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var packed_q_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var packed_k_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var packed_v_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var dattn_buf = ctx.enqueue_create_buffer[dtype](SCORES_SIZE)
        var dscore_buf = ctx.enqueue_create_buffer[dtype](SCORES_SIZE)
        # attn_T then dscore_T — buffer reused after step 5.
        var attn_T_buf = ctx.enqueue_create_buffer[dtype](SCORES_SIZE)
        var dQ_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var dK_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)
        var dV_buf = ctx.enqueue_create_buffer[dtype](PACKED_QKV_SIZE)

        # ── 1. Pack: grad_output → dout_packed; cache.{Q,K,V} → packed_Q/K/V.
        comptime pack_elems = BATCH * Self.seq_len * Self.dim
        comptime pack_blocks = (pack_elems + TPB - 1) // TPB

        @parameter
        @always_inline
        def pack_in_wrapper(
            packed_dout: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
            ],
            packed_q: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
            ],
            packed_k: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
            ],
            packed_v: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= pack_elems:
                return
            var hd = Self.head_dim
            var d = idx % hd
            var rem = idx // hd
            var h = rem % Self.n_heads
            var rem2 = rem // Self.n_heads
            var t = rem2 % Self.seq_len
            var b = rem2 // Self.seq_len

            var col_in_dim = h * hd + d

            var go_idx = b * Self.OUT_DIM + t * Self.dim + col_in_dim
            var c_q = b * Self.CACHE_SIZE + t * Self.dim + col_in_dim
            var c_k = (
                b * Self.CACHE_SIZE
                + Self._k_offset()
                + t * Self.dim
                + col_in_dim
            )
            var c_v = (
                b * Self.CACHE_SIZE
                + Self._v_offset()
                + t * Self.dim
                + col_in_dim
            )

            var bh = b * Self.n_heads + h
            var packed_idx = bh * Self.seq_len * hd + t * hd + d

            packed_dout.ptr[packed_idx] = grad_output.ptr[go_idx]
            packed_q.ptr[packed_idx] = rebind[Scalar[dtype]](cache.ptr[c_q])
            packed_k.ptr[packed_idx] = rebind[Scalar[dtype]](cache.ptr[c_k])
            packed_v.ptr[packed_idx] = rebind[Scalar[dtype]](cache.ptr[c_v])

        var packed_dout_lt = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
        ](packed_dout_buf.unsafe_ptr())
        var packed_q_lt = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
        ](packed_q_buf.unsafe_ptr())
        var packed_k_lt = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
        ](packed_k_buf.unsafe_ptr())
        var packed_v_lt = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), MutAnyOrigin
        ](packed_v_buf.unsafe_ptr())

        ctx.enqueue_function[pack_in_wrapper](
            packed_dout_lt,
            packed_q_lt,
            packed_k_lt,
            packed_v_lt,
            grad_output,
            cache,
            grid_dim=(pack_blocks,),
            block_dim=(TPB,),
        )

        # ── 2. bmm[transpose_b]: dattn = dout @ V^T.
        var packed_dout_tt = TileTensor(
            packed_dout_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        var packed_q_tt = TileTensor(
            packed_q_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        var packed_k_tt = TileTensor(
            packed_k_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        var packed_v_tt = TileTensor(
            packed_v_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        var dattn_tt = TileTensor(
            dattn_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.seq_len](),
        )
        batched_matmul[transpose_b=True, target="gpu"](
            dattn_tt,
            packed_dout_tt,
            packed_v_tt,
            context=DeviceContextPtr(ctx),
        )

        # ── 3. Softmax JVP: dscore = scale * a * (dattn - sum_k a_k * dattn_k).
        # Reads cache.attn (per-sample-strided), reads dattn (contiguous),
        # writes dscore (contiguous). For causal: cache.attn[i, j] = 0 for j > i,
        # so dscore[i, j] is naturally 0 there — no explicit masking needed.
        var scale = Scalar[dtype](Float32(1.0) / sqrt(Float32(Self.head_dim)))

        @parameter
        @always_inline
        def softmax_jvp_wrapper(
            dscore: LayoutTensor[
                dtype, Layout.row_major(SCORES_SIZE), MutAnyOrigin
            ],
            dattn: LayoutTensor[
                dtype, Layout.row_major(SCORES_SIZE), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
            scale_v: Scalar[dtype],
        ):
            comptime assert dtype.is_floating_point(), "dtype must be floating point"
            # 1 block per (b, h); threads stride over rows i.
            var blk = Int(block_idx.x)
            if blk >= BH:
                return
            var b = blk // Self.n_heads
            var h = blk % Self.n_heads
            var tid = Int(thread_idx.x)
            var bs = Int(block_dim.x)

            var bh_off = blk * Self.seq_len * Self.seq_len
            var cache_attn_base = (
                b * Self.CACHE_SIZE
                + Self._attn_cache_offset()
                + h * Self.seq_len * Self.seq_len
            )

            var i = tid
            while i < Self.seq_len:
                var row_off = bh_off + i * Self.seq_len
                var cache_row_off = cache_attn_base + i * Self.seq_len

                # First pass: compute sum_k a_k * dattn_k for this row.
                var s = Scalar[dtype](0)
                for j in range(Self.seq_len):
                    var a = rebind[Scalar[dtype]](
                        cache.ptr[cache_row_off + j]
                    )
                    var d_a = rebind[Scalar[dtype]](dattn.ptr[row_off + j])
                    s += a * d_a

                # Second pass: dscore[j] = scale * a[j] * (dattn[j] - s).
                for j in range(Self.seq_len):
                    var a = rebind[Scalar[dtype]](
                        cache.ptr[cache_row_off + j]
                    )
                    var d_a = rebind[Scalar[dtype]](dattn.ptr[row_off + j])
                    dscore.ptr[row_off + j] = scale_v * a * (d_a - s)

                i += bs

        var dattn_lt = LayoutTensor[
            dtype, Layout.row_major(SCORES_SIZE), ImmutAnyOrigin
        ](dattn_buf.unsafe_ptr())
        var dscore_lt = LayoutTensor[
            dtype, Layout.row_major(SCORES_SIZE), MutAnyOrigin
        ](dscore_buf.unsafe_ptr())

        ctx.enqueue_function[softmax_jvp_wrapper](
            dscore_lt,
            dattn_lt,
            cache,
            scale,
            grid_dim=(BH,),
            block_dim=(TPB,),
        )

        # ── 4. Transpose attn from cache → attn_T_packed (BH, seq, seq).
        comptime transpose_elems = BH * Self.seq_len * Self.seq_len
        comptime transpose_blocks = (transpose_elems + TPB - 1) // TPB

        @parameter
        @always_inline
        def transpose_attn_wrapper(
            attn_T: LayoutTensor[
                dtype, Layout.row_major(SCORES_SIZE), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= transpose_elems:
                return
            # idx unrolls (bh, j, i) over BH × seq × seq → write attn_T[bh, j, i]
            var i = idx % Self.seq_len
            var rem = idx // Self.seq_len
            var j = rem % Self.seq_len
            var bh = rem // Self.seq_len
            var b = bh // Self.n_heads
            var h = bh % Self.n_heads
            var src = (
                b * Self.CACHE_SIZE
                + Self._attn_cache_offset()
                + h * Self.seq_len * Self.seq_len
                + i * Self.seq_len
                + j
            )
            var dst = bh * Self.seq_len * Self.seq_len + j * Self.seq_len + i
            attn_T.ptr[dst] = rebind[Scalar[dtype]](cache.ptr[src])

        var attn_T_lt = LayoutTensor[
            dtype, Layout.row_major(SCORES_SIZE), MutAnyOrigin
        ](attn_T_buf.unsafe_ptr())

        ctx.enqueue_function[transpose_attn_wrapper](
            attn_T_lt,
            cache,
            grid_dim=(transpose_blocks,),
            block_dim=(TPB,),
        )

        # ── 5. bmm: dV = attn_T @ dout.
        var attn_T_tt = TileTensor(
            attn_T_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.seq_len](),
        )
        var dV_tt = TileTensor(
            dV_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        batched_matmul[target="gpu"](
            dV_tt,
            attn_T_tt,
            packed_dout_tt,
            context=DeviceContextPtr(ctx),
        )

        # ── 6. Transpose dscore in-place into the (now-free) attn_T buffer.
        @parameter
        @always_inline
        def transpose_dscore_wrapper(
            dscore_T: LayoutTensor[
                dtype, Layout.row_major(SCORES_SIZE), MutAnyOrigin
            ],
            dscore: LayoutTensor[
                dtype, Layout.row_major(SCORES_SIZE), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= transpose_elems:
                return
            var i = idx % Self.seq_len
            var rem = idx // Self.seq_len
            var j = rem % Self.seq_len
            var bh = rem // Self.seq_len
            var src = bh * Self.seq_len * Self.seq_len + i * Self.seq_len + j
            var dst = bh * Self.seq_len * Self.seq_len + j * Self.seq_len + i
            dscore_T.ptr[dst] = rebind[Scalar[dtype]](dscore.ptr[src])

        var dscore_immut = LayoutTensor[
            dtype, Layout.row_major(SCORES_SIZE), ImmutAnyOrigin
        ](dscore_buf.unsafe_ptr())

        ctx.enqueue_function[transpose_dscore_wrapper](
            attn_T_lt,  # reused for dscore_T
            dscore_immut,
            grid_dim=(transpose_blocks,),
            block_dim=(TPB,),
        )

        # ── 7. bmm: dK = dscore_T @ Q.
        var dscore_T_tt = TileTensor(
            attn_T_buf.unsafe_ptr(),  # same buffer
            row_major[BH, Self.seq_len, Self.seq_len](),
        )
        var dK_tt = TileTensor(
            dK_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        batched_matmul[target="gpu"](
            dK_tt,
            dscore_T_tt,
            packed_q_tt,
            context=DeviceContextPtr(ctx),
        )

        # ── 8. bmm: dQ = dscore @ K.
        var dscore_tt = TileTensor(
            dscore_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.seq_len](),
        )
        var dQ_tt = TileTensor(
            dQ_buf.unsafe_ptr(),
            row_major[BH, Self.seq_len, Self.head_dim](),
        )
        batched_matmul[target="gpu"](
            dQ_tt,
            dscore_tt,
            packed_k_tt,
            context=DeviceContextPtr(ctx),
        )

        # ── 9. Unpack: dQ/dK/dV → grad_input ([Q_grad | K_grad | V_grad] per
        # sample, with per-head reshape).
        comptime unpack_blocks = (pack_elems + TPB - 1) // TPB

        @parameter
        @always_inline
        def unpack_grad_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            dQ: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), ImmutAnyOrigin
            ],
            dK: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), ImmutAnyOrigin
            ],
            dV: LayoutTensor[
                dtype, Layout.row_major(PACKED_QKV_SIZE), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= pack_elems:
                return
            var hd = Self.head_dim
            var d = idx % hd
            var rem = idx // hd
            var h = rem % Self.n_heads
            var rem2 = rem // Self.n_heads
            var t = rem2 % Self.seq_len
            var b = rem2 // Self.seq_len

            var col_in_dim = h * hd + d
            var bh = b * Self.n_heads + h
            var packed_idx = bh * Self.seq_len * hd + t * hd + d

            var gi_q = b * Self.IN_DIM + t * Self.dim + col_in_dim
            var gi_k = (
                b * Self.IN_DIM
                + Self._k_offset()
                + t * Self.dim
                + col_in_dim
            )
            var gi_v = (
                b * Self.IN_DIM
                + Self._v_offset()
                + t * Self.dim
                + col_in_dim
            )

            grad_input.ptr[gi_q] = rebind[Scalar[dtype]](dQ.ptr[packed_idx])
            grad_input.ptr[gi_k] = rebind[Scalar[dtype]](dK.ptr[packed_idx])
            grad_input.ptr[gi_v] = rebind[Scalar[dtype]](dV.ptr[packed_idx])

        var dQ_immut = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), ImmutAnyOrigin
        ](dQ_buf.unsafe_ptr())
        var dK_immut = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), ImmutAnyOrigin
        ](dK_buf.unsafe_ptr())
        var dV_immut = LayoutTensor[
            dtype, Layout.row_major(PACKED_QKV_SIZE), ImmutAnyOrigin
        ](dV_buf.unsafe_ptr())

        ctx.enqueue_function[unpack_grad_wrapper](
            grad_input,
            dQ_immut,
            dK_immut,
            dV_immut,
            grid_dim=(unpack_blocks,),
            block_dim=(TPB,),
        )
