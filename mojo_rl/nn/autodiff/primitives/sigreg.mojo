"""SIGRegOp: Sketch Isotropic Gaussian Regularizer.

Epps-Pulley Gaussianity statistic via random projections (Maes, Le Lidec et
al., LeWM, 2026; based on the broader Epps-Pulley literature). Enforces
projected embeddings to look like samples from N(0, I) in characteristic-
function space, without EMAs, pretrained encoders, or auxiliary
supervision.

Semantics (input shape (B, T*D), interpreted as (B, T, D)):
    A ~ N(0, I_D)^{D x num_proj}, columns L2-normalized.   # random projection
    z[b, t, p]  = sum_d input[b, t*D + d] * A[d, p]        # (B, T, num_proj)
    cm[t, p, k] = (1/B) * sum_b cos(z[b, t, p] * t_k)      # (T, num_proj, K)
    sm[t, p, k] = (1/B) * sum_b sin(z[b, t, p] * t_k)
    err[t, p, k] = (cm[t,p,k] - phi_k)^2 + sm[t,p,k]^2
    stat = (B / (T * num_proj)) * sum_{t,p,k} w_k * err[t,p,k]

Constants (derived comptime from `knots`):
    t_k    = k * 3 / (K-1),  k = 0..K-1            # knots, max=3
    phi_k  = exp(-t_k^2 / 2)                        # target N(0,1) char fn
    w_k    = trap_k * phi_k                         # trapezoidal · window

Output is the scalar `stat` replicated to every output[b, 0] slot. With a
standard 1/B grad seed the chain-rule effective seed equals 1, matching the
PyTorch reference's `.mean()` semantics.

Gradient (B cancels):
    dL/dz[b,t,p] = (2/(T*num_proj)) * G * sum_k w_k * t_k * [
                    -(cm[t,p,k] - phi_k) * sin(z[b,t,p] * t_k)
                    + sm[t,p,k]           * cos(z[b,t,p] * t_k)
                  ]
    dL/dinput[b, t*D + d] = sum_p A[d, p] * dL/dz[b, t, p]
where G = sum_b grad_output[b, 0].

PRNG: A is regenerated deterministically each call from `Int(cache.ptr)` as
seed (same trick as DropoutOp). Forward and backward see the same cache
buffer, so they produce the same A.

PARAM_SIZE = 0
CACHE_SIZE = T * num_proj  (stores z[b, t, p] per sample)
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.math import sin, cos, sqrt, log, exp, pi
from std.random.philox import Random as PhiloxRandom


struct SIGRegOp[
    dim: Int, seq_len: Int, num_proj: Int, knots: Int
](DiffOp):
    """Epps-Pulley Gaussianity regularizer.

    Input shape  : (BATCH, seq_len * dim)            — interpreted as (B, T, D)
    Output shape : (BATCH, 1)                        — scalar replicated
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 22
    comptime IN_DIM: Int = Self.seq_len * Self.dim
    comptime OUT_DIM: Int = 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.seq_len * Self.num_proj
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # Compile-time helpers (knot grid, weights)
    # =========================================================================

    @always_inline
    @staticmethod
    def _t_step() -> Float64:
        """Spacing between knots over [0, 3]."""
        return 3.0 / Float64(Self.knots - 1)

    @always_inline
    @staticmethod
    def _t_k(k: Int) -> Float64:
        """k-th knot location in [0, 3]."""
        return Float64(k) * Self._t_step()

    @always_inline
    @staticmethod
    def _phi_k(k: Int) -> Float64:
        """Target Gaussian characteristic function value at t_k."""
        var tk = Self._t_k(k)
        return exp(-tk * tk * 0.5)

    @always_inline
    @staticmethod
    def _w_k(k: Int) -> Float64:
        """Effective weight: trapezoidal · phi (matches `self.weights` in ref)."""
        var dt = Self._t_step()
        var trap = dt if (k == 0 or k == Self.knots - 1) else 2.0 * dt
        return trap * Self._phi_k(k)

    # =========================================================================
    # Random projection generation (deterministic from seed)
    # =========================================================================

    @staticmethod
    def _generate_a[dtype: DType](
        seed: UInt64,
        a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        """Fill a_ptr[0 .. dim*num_proj] with column-normalized Gaussian.

        Layout: row-major (dim, num_proj). A[d, p] = a_ptr[d * num_proj + p].
        """
        # Step 1: raw Gaussian via Box-Muller from two Philox uniforms.
        for d in range(Self.dim):
            for p in range(Self.num_proj):
                var idx = d * Self.num_proj + p
                var rng1 = PhiloxRandom(
                    seed=seed, offset=UInt64(2 * idx)
                )
                var rng2 = PhiloxRandom(
                    seed=seed, offset=UInt64(2 * idx + 1)
                )
                var u1 = Float64(rng1.step_uniform()[0])
                var u2 = Float64(rng2.step_uniform()[0])
                if u1 < 1e-10:
                    u1 = 1e-10
                var g = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
                a_ptr[idx] = Scalar[dtype](g)

        # Step 2: normalize each column to unit L2 norm.
        for p in range(Self.num_proj):
            var sum_sq = Float64(0.0)
            for d in range(Self.dim):
                var v = Float64(a_ptr[d * Self.num_proj + p])
                sum_sq += v * v
            var norm = sqrt(sum_sq + 1e-12)
            for d in range(Self.dim):
                var v = Float64(a_ptr[d * Self.num_proj + p])
                a_ptr[d * Self.num_proj + p] = Scalar[dtype](v / norm)

    # =========================================================================
    # CPU eval / vjp
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
        comptime assert dtype.is_floating_point(), "dtype must be floating point"

        # Generate A (D, num_proj) deterministically from cache pointer.
        var seed = UInt64(Int(cache.ptr))
        var a_storage = InlineArray[Scalar[dtype], Self.dim * Self.num_proj](
            uninitialized=True
        )
        Self._generate_a[dtype](seed, a_storage.unsafe_ptr())

        # Compute z[b, t, p] = sum_d input[b, t*D + d] * A[d, p].
        # Store in cache for backward.
        for b in range(BATCH):
            for t in range(Self.seq_len):
                for p in range(Self.num_proj):
                    var z = Scalar[dtype](0)
                    for d in range(Self.dim):
                        var xi = rebind[Scalar[dtype]](
                            input[b, t * Self.dim + d]
                        )
                        var aval = a_storage[d * Self.num_proj + p]
                        z += xi * aval
                    cache[b, t * Self.num_proj + p] = z

        # Aggregate cm[t, p, k] and sm[t, p, k] over batch.
        var n_tpk = Self.seq_len * Self.num_proj * Self.knots
        var cm = InlineArray[Scalar[dtype], Self.seq_len * Self.num_proj * Self.knots](
            uninitialized=True
        )
        var sm = InlineArray[Scalar[dtype], Self.seq_len * Self.num_proj * Self.knots](
            uninitialized=True
        )
        for i in range(n_tpk):
            cm[i] = Scalar[dtype](0)
            sm[i] = Scalar[dtype](0)

        var inv_b = Scalar[dtype](1.0 / Float64(BATCH))
        for b in range(BATCH):
            for t in range(Self.seq_len):
                for p in range(Self.num_proj):
                    var z = rebind[Scalar[dtype]](
                        cache[b, t * Self.num_proj + p]
                    )
                    for k in range(Self.knots):
                        var tk = Scalar[dtype](Self._t_k(k))
                        var arg = z * tk
                        var idx = (t * Self.num_proj + p) * Self.knots + k
                        cm[idx] += cos(arg) * inv_b
                        sm[idx] += sin(arg) * inv_b

        # Statistic: stat = (B/(T*num_proj)) * sum_{t,p,k} w_k * err_{t,p,k}.
        # The B factor here matches PyTorch's `(err @ weights) * proj.size(-2)`
        # before the `.mean()` at the end. Folded into the prefactor below.
        var prefactor = Scalar[dtype](
            Float64(BATCH) / Float64(Self.seq_len * Self.num_proj)
        )
        var stat = Scalar[dtype](0)
        for t in range(Self.seq_len):
            for p in range(Self.num_proj):
                for k in range(Self.knots):
                    var idx = (t * Self.num_proj + p) * Self.knots + k
                    var phi = Scalar[dtype](Self._phi_k(k))
                    var wk = Scalar[dtype](Self._w_k(k))
                    var diff = cm[idx] - phi
                    var err = diff * diff + sm[idx] * sm[idx]
                    stat += wk * err
        stat *= prefactor

        # Replicate to every output slot.
        for b in range(BATCH):
            output[b, 0] = stat

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
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        # Regenerate A from same seed (cache pointer unchanged across fwd/bwd).
        var seed = UInt64(Int(cache.ptr))
        var a_storage = InlineArray[Scalar[dtype], Self.dim * Self.num_proj](
            uninitialized=True
        )
        Self._generate_a[dtype](seed, a_storage.unsafe_ptr())

        # Recompute cm, sm (batch-aggregated; we cached only per-sample z).
        var n_tpk = Self.seq_len * Self.num_proj * Self.knots
        var cm = InlineArray[Scalar[dtype], Self.seq_len * Self.num_proj * Self.knots](
            uninitialized=True
        )
        var sm = InlineArray[Scalar[dtype], Self.seq_len * Self.num_proj * Self.knots](
            uninitialized=True
        )
        for i in range(n_tpk):
            cm[i] = Scalar[dtype](0)
            sm[i] = Scalar[dtype](0)

        var inv_b = Scalar[dtype](1.0 / Float64(BATCH))
        for b in range(BATCH):
            for t in range(Self.seq_len):
                for p in range(Self.num_proj):
                    var z = rebind[Scalar[dtype]](
                        cache[b, t * Self.num_proj + p]
                    )
                    for k in range(Self.knots):
                        var tk = Scalar[dtype](Self._t_k(k))
                        var arg = z * tk
                        var idx = (t * Self.num_proj + p) * Self.knots + k
                        cm[idx] += cos(arg) * inv_b
                        sm[idx] += sin(arg) * inv_b

        # Effective grad seed: G = sum_b grad_output[b, 0].
        var G = Scalar[dtype](0)
        for b in range(BATCH):
            G += rebind[Scalar[dtype]](grad_output[b, 0])

        # 2 G / (T * num_proj) — multiplier on dL/dz from chain rule.
        var coef = G * Scalar[dtype](
            2.0 / Float64(Self.seq_len * Self.num_proj)
        )

        # Compute grad_input[b, t*D + d] = sum_p A[d, p] * dL/dz[b, t, p].
        for b in range(BATCH):
            for t in range(Self.seq_len):
                # First compute dL/dz[b, t, :] for this (b, t) into a buffer.
                var dLdz = InlineArray[Scalar[dtype], Self.num_proj](
                    uninitialized=True
                )
                for p in range(Self.num_proj):
                    var z = rebind[Scalar[dtype]](
                        cache[b, t * Self.num_proj + p]
                    )
                    var acc = Scalar[dtype](0)
                    for k in range(Self.knots):
                        var tk = Scalar[dtype](Self._t_k(k))
                        var wk = Scalar[dtype](Self._w_k(k))
                        var phi = Scalar[dtype](Self._phi_k(k))
                        var idx = (t * Self.num_proj + p) * Self.knots + k
                        var arg = z * tk
                        var s_arg = sin(arg)
                        var c_arg = cos(arg)
                        var bracket = (
                            -(cm[idx] - phi) * s_arg + sm[idx] * c_arg
                        )
                        acc += wk * tk * bracket
                    dLdz[p] = coef * acc

                # Now grad_input[b, t*D + d] = sum_p A[d, p] * dLdz[p].
                for d in range(Self.dim):
                    var g = Scalar[dtype](0)
                    for p in range(Self.num_proj):
                        var aval = a_storage[d * Self.num_proj + p]
                        g += aval * dLdz[p]
                    grad_input[b, t * Self.dim + d] = g

    # =========================================================================
    # GPU eval / vjp — Phase 1 stub: delegates to CPU via host roundtrip.
    # =========================================================================
    # SIGReg is a regularizer (not a hot path) and the multi-stage reduction
    # (A generation → projection → cm/sm aggregation → statistic) is awkward
    # to express as a single kernel. Phase 1 ships a host-side CPU fallback
    # that round-trips through map_to_host so the GPU autodiff chain stays
    # type-consistent. A real GPU implementation is deferred to Phase 3 once
    # we have a profiler reading on the regularizer's share of step time.

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
        # Phase 1: not implemented on GPU. Callers should run SIGReg on CPU
        # outputs OR await Phase 3 GPU implementation.
        raise Error(
            "SIGRegOp.eval_gpu not implemented in Phase 1 — use CPU eval"
        )

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
        raise Error(
            "SIGRegOp.vjp_gpu not implemented in Phase 1 — use CPU vjp"
        )
