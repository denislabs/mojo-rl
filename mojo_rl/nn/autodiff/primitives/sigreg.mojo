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
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from std.gpu.primitives import block
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
    # Per-sample workspace is awkward here — most scratch (A, cm, sm,
    # partials) is batch-independent. Callers use `workspace_size_for[BATCH]`
    # to allocate the total workspace once and pass its pointer; setting
    # OP_WORKSPACE_PER_SAMPLE to 0 keeps Sequential/AutoFused compositions
    # unaffected (they never use SIGReg directly).
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    # Workspace layout (eval and vjp share the prefix; vjp uses the tail
    # too). Offsets in elements (Scalar[dtype]):
    #   [0,                       D*P)                A (D × num_proj)
    #   [D*P,                     D*P + T*P*K)        cm (T × P × K)
    #   [D*P + T*P*K,             D*P + 2*T*P*K)      sm (T × P × K)
    #   [D*P + 2*T*P*K,           D*P + 2*T*P*K + Np) partials (#blocks)
    #   [+ Np,                    + Np + 1)           stat scalar (eval) /
    #                                                  g scalar (vjp)
    #   [+ 1, + 1 + BATCH*T*P)                        dLdz (vjp only)

    @always_inline
    @staticmethod
    def _n_partials() -> Int:
        return (Self.seq_len * Self.num_proj * Self.knots + TPB - 1) // TPB

    @always_inline
    @staticmethod
    def _ws_off_a() -> Int:
        return 0

    @always_inline
    @staticmethod
    def _ws_off_cm() -> Int:
        return Self.dim * Self.num_proj

    @always_inline
    @staticmethod
    def _ws_off_sm() -> Int:
        return Self._ws_off_cm() + Self.seq_len * Self.num_proj * Self.knots

    @always_inline
    @staticmethod
    def _ws_off_partials() -> Int:
        return Self._ws_off_sm() + Self.seq_len * Self.num_proj * Self.knots

    @always_inline
    @staticmethod
    def _ws_off_scalar() -> Int:
        return Self._ws_off_partials() + Self._n_partials()

    @always_inline
    @staticmethod
    def _ws_off_dLdz() -> Int:
        return Self._ws_off_scalar() + 1

    @always_inline
    @staticmethod
    def workspace_size_for[BATCH: Int]() -> Int:
        """Total workspace size (in elements) the caller must allocate.

        Sized for vjp (the larger consumer); reusing for eval is safe.
        """
        return Self._ws_off_dLdz() + BATCH * Self.seq_len * Self.num_proj

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
    # GPU eval / vjp
    # =========================================================================
    # Strategy:
    #   1. Generate A (D, P) inside scratch DeviceBuffer (deterministic from
    #      cache.ptr seed — same as CPU): Box-Muller in one kernel, column
    #      L2-normalize in second kernel.
    #   2. Project: z[b,t,p] = sum_d input[b,t*D+d] * A[d,p] → cache.
    #   3. cm/sm reduction over B → cm_t, sm_t. Per-block stat partial via
    #      block.sum, then a single-block final-reduce kernel collapses
    #      partials into the scalar stat.
    #   4. Broadcast scaled stat to output[b, 0].
    # vjp is symmetric: regen A, recompute cm/sm, reduce G via block.sum
    # (BATCH ≤ TPB so single block), compute dLdz, matmul with A.
    #
    # Numeric note: kernels accumulate in Scalar[dtype] (Float32 by default,
    # Metal-compat). CPU uses Float64. Expect ~1e-3 relative tolerance, not
    # bit-exact.

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
        """Workspace must be at least `Self.workspace_size_for[BATCH]()`."""
        comptime D = Self.dim
        comptime T = Self.seq_len
        comptime P = Self.num_proj
        comptime K = Self.knots
        comptime N_PARTIALS = Self._n_partials()

        # Carve workspace.
        var a_ptr = workspace + Self._ws_off_a()
        var cm_ptr = workspace + Self._ws_off_cm()
        var sm_ptr = workspace + Self._ws_off_sm()
        var partials_ptr = workspace + Self._ws_off_partials()
        var stat_ptr = workspace + Self._ws_off_scalar()

        var a_t = LayoutTensor[dtype, Layout.row_major(D, P), MutAnyOrigin](
            a_ptr
        )
        var cm_t = LayoutTensor[
            dtype, Layout.row_major(T, P * K), MutAnyOrigin
        ](cm_ptr)
        var sm_t = LayoutTensor[
            dtype, Layout.row_major(T, P * K), MutAnyOrigin
        ](sm_ptr)

        var seed = UInt64(Int(cache.ptr))

        # 1. Box-Muller into A.
        var grid_a = (D * P + TPB - 1) // TPB
        ctx.enqueue_function[gen_a_unnorm_kernel[D, P, dtype]](
            a_t, seed,
            grid_dim=(grid_a,), block_dim=(TPB,),
        )
        # 2. Column L2-normalize.
        var grid_norm = (P + TPB - 1) // TPB
        ctx.enqueue_function[norm_a_kernel[D, P, dtype]](
            a_t,
            grid_dim=(grid_norm,), block_dim=(TPB,),
        )
        # 3. Project: z = input @ A.
        var grid_proj = (BATCH * T * P + TPB - 1) // TPB
        ctx.enqueue_function[project_kernel[BATCH, T, D, P, dtype]](
            input, a_t, cache,
            grid_dim=(grid_proj,), block_dim=(TPB,),
        )
        # 4. cm/sm + per-block partial stat via block.sum.
        ctx.enqueue_function[
            cm_sm_kernel[BATCH, T, P, K, dtype, True]
        ](
            cache, cm_t, sm_t, partials_ptr,
            grid_dim=(N_PARTIALS,), block_dim=(TPB,),
        )
        # 5. Final reduce: collapse N_PARTIALS scalars into stat
        #    using a single block (grid-stride loop inside the kernel).
        ctx.enqueue_function[final_reduce_kernel[N_PARTIALS, dtype]](
            partials_ptr, stat_ptr,
            grid_dim=(1,), block_dim=(TPB,),
        )
        # 6. Broadcast scaled stat to all output[b, 0].
        var grid_bcast = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[broadcast_stat_kernel[BATCH, T, P, dtype]](
            stat_ptr, output,
            grid_dim=(grid_bcast,), block_dim=(TPB,),
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
        """Workspace must be at least `Self.workspace_size_for[BATCH]()`."""
        comptime D = Self.dim
        comptime T = Self.seq_len
        comptime P = Self.num_proj
        comptime K = Self.knots
        comptime N_PARTIALS = Self._n_partials()

        # Carve workspace (same layout as eval; reuses scalar slot for G).
        var a_ptr = workspace + Self._ws_off_a()
        var cm_ptr = workspace + Self._ws_off_cm()
        var sm_ptr = workspace + Self._ws_off_sm()
        var partials_ptr = workspace + Self._ws_off_partials()
        var g_ptr = workspace + Self._ws_off_scalar()
        var dLdz_ptr = workspace + Self._ws_off_dLdz()

        var a_t = LayoutTensor[dtype, Layout.row_major(D, P), MutAnyOrigin](
            a_ptr
        )
        var cm_t = LayoutTensor[
            dtype, Layout.row_major(T, P * K), MutAnyOrigin
        ](cm_ptr)
        var sm_t = LayoutTensor[
            dtype, Layout.row_major(T, P * K), MutAnyOrigin
        ](sm_ptr)
        var dLdz_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, T * P), MutAnyOrigin
        ](dLdz_ptr)

        var seed = UInt64(Int(cache.ptr))

        # 1. Regen A from same seed.
        var grid_a = (D * P + TPB - 1) // TPB
        ctx.enqueue_function[gen_a_unnorm_kernel[D, P, dtype]](
            a_t, seed,
            grid_dim=(grid_a,), block_dim=(TPB,),
        )
        var grid_norm = (P + TPB - 1) // TPB
        ctx.enqueue_function[norm_a_kernel[D, P, dtype]](
            a_t,
            grid_dim=(grid_norm,), block_dim=(TPB,),
        )
        # 2. Recompute cm/sm (no stat needed in backward).
        ctx.enqueue_function[
            cm_sm_kernel[BATCH, T, P, K, dtype, False]
        ](
            cache, cm_t, sm_t, partials_ptr,
            grid_dim=(N_PARTIALS,), block_dim=(TPB,),
        )
        # 3. Reduce G = sum_b grad_output[b, 0]. BATCH typically ≤ TPB
        #    so this fits in a single block via block.sum.
        ctx.enqueue_function[reduce_g_kernel[BATCH, dtype]](
            grad_output, g_ptr,
            grid_dim=(1,), block_dim=(TPB,),
        )
        # 4. dLdz[b,t,p] using cached z + cm/sm.
        var grid_dLdz = (BATCH * T * P + TPB - 1) // TPB
        ctx.enqueue_function[dLdz_kernel[BATCH, T, P, K, dtype]](
            cache, cm_t, sm_t, dLdz_t, g_ptr,
            grid_dim=(grid_dLdz,), block_dim=(TPB,),
        )
        # 5. grad_input = dLdz @ A^T.
        var grid_mm = (BATCH * T * D + TPB - 1) // TPB
        ctx.enqueue_function[matmul_a_kernel[BATCH, T, D, P, dtype]](
            dLdz_t, a_t, grad_input,
            grid_dim=(grid_mm,), block_dim=(TPB,),
        )


# ============================================================================
# Module-level GPU kernels (parameterised on dimensions + dtype).
#
# Reductions use block.sum (no Atomic — atomics on GPU floats are awkward and
# the codebase already standardises on block ops for reductions).
# Box-Muller in `gen_a_unnorm_kernel` uses Float32 directly because Philox's
# `step_uniform()` returns Float32 natively; every other kernel arithmetic
# happens in Scalar[dtype] so swapping the project-wide dtype Just Works.
# ============================================================================


def gen_a_unnorm_kernel[
    D: Int, P: Int, dtype: DType
](
    a_t: LayoutTensor[dtype, Layout.row_major(D, P), MutAnyOrigin],
    seed: UInt64,
):
    """One thread per A[d, p]; Box-Muller from Philox(seed, idx)."""
    var idx = Int(global_idx.x)
    if idx >= D * P:
        return
    var d_idx = idx // P
    var p_idx = idx % P
    var philox = PhiloxRandom(seed=seed, offset=UInt64(idx))
    var rand_vals = philox.step_uniform()
    # Philox returns Float32 natively — kept as-is here.
    var u1 = Float32(rand_vals[0]) + Float32(1e-8)
    var u2 = Float32(rand_vals[1])
    var mag = sqrt(Float32(-2.0) * log(u1))
    var g = mag * cos(u2 * Float32(6.283185307179586))
    a_t[d_idx, p_idx] = Scalar[dtype](g)


def norm_a_kernel[
    D: Int, P: Int, dtype: DType
](
    a_t: LayoutTensor[dtype, Layout.row_major(D, P), MutAnyOrigin],
):
    """One thread per column; L2-normalise A[:, p]."""
    comptime assert dtype.is_floating_point(), "dtype must be FP"
    var p_idx = Int(global_idx.x)
    if p_idx >= P:
        return
    var sum_sq = Scalar[dtype](0)
    for d in range(D):
        var v = rebind[Scalar[dtype]](a_t[d, p_idx])
        sum_sq += v * v
    var inv_norm = Scalar[dtype](1) / (sqrt(sum_sq) + Scalar[dtype](1e-12))
    for d in range(D):
        var v = rebind[Scalar[dtype]](a_t[d, p_idx])
        a_t[d, p_idx] = v * inv_norm


def project_kernel[
    BATCH: Int, T: Int, D: Int, P: Int, dtype: DType
](
    input_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ],
    a_t: LayoutTensor[dtype, Layout.row_major(D, P), MutAnyOrigin],
    cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * P), MutAnyOrigin
    ],
):
    """One thread per (b, t, p); z = sum_d input[b, t*D+d] * A[d, p]."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * P:
        return
    var p_idx = idx % P
    var t_idx = (idx // P) % T
    var b = idx // (T * P)
    var z = Scalar[dtype](0)
    for d in range(D):
        var xi = rebind[Scalar[dtype]](input_t[b, t_idx * D + d])
        var aval = rebind[Scalar[dtype]](a_t[d, p_idx])
        z += xi * aval
    cache_t[b, t_idx * P + p_idx] = z


def cm_sm_kernel[
    BATCH: Int, T: Int, P: Int, K: Int, dtype: DType, INCLUDE_STAT: Bool
](
    cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * P), MutAnyOrigin
    ],
    cm_t: LayoutTensor[dtype, Layout.row_major(T, P * K), MutAnyOrigin],
    sm_t: LayoutTensor[dtype, Layout.row_major(T, P * K), MutAnyOrigin],
    partials_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    """One thread per (t, p, k); reduce over B → cm/sm; block.sum partial."""
    comptime assert dtype.is_floating_point(), "dtype must be FP"
    var idx = Int(global_idx.x)

    # Knot grid + window weights (comptime-constant deltas, runtime k).
    var dt = Scalar[dtype](3.0 / Float64(K - 1))
    var inv_b = Scalar[dtype](1.0 / Float64(BATCH))

    var contrib = Scalar[dtype](0)
    if idx < T * P * K:
        var k_idx = idx % K
        var p_idx = (idx // K) % P
        var t_idx = idx // (K * P)
        var tk = dt * Scalar[dtype](k_idx)
        var phi = exp(Scalar[dtype](-0.5) * tk * tk)

        # Reduce over B (accumulate raw, divide at end for stability).
        var cm_sum = Scalar[dtype](0)
        var sm_sum = Scalar[dtype](0)
        for b in range(BATCH):
            var z = rebind[Scalar[dtype]](cache_t[b, t_idx * P + p_idx])
            var arg = z * tk
            cm_sum += cos(arg)
            sm_sum += sin(arg)
        var cm = cm_sum * inv_b
        var sm = sm_sum * inv_b

        cm_t[t_idx, p_idx * K + k_idx] = cm
        sm_t[t_idx, p_idx * K + k_idx] = sm

        comptime if INCLUDE_STAT:
            var trap = (
                dt if (k_idx == 0 or k_idx == K - 1)
                else dt * Scalar[dtype](2.0)
            )
            var wk = trap * phi
            var diff = cm - phi
            var err = diff * diff + sm * sm
            contrib = wk * err

    # All threads (including OOB) must participate in block.sum.
    comptime if INCLUDE_STAT:
        var partial = block.sum[block_size=TPB, broadcast=False](
            val=SIMD[dtype, 1](contrib)
        )
        if thread_idx.x == 0:
            partials_ptr[Int(block_idx.x)] = partial[0]


def final_reduce_kernel[
    N_PARTIALS: Int, dtype: DType
](
    partials_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    """Single block; grid-stride loop summing all partials into out_ptr[0]."""
    var tid = Int(thread_idx.x)
    var v = Scalar[dtype](0)
    var i = tid
    while i < N_PARTIALS:
        v += partials_ptr[i]
        i += TPB
    var total = block.sum[block_size=TPB, broadcast=False](
        val=SIMD[dtype, 1](v)
    )
    if tid == 0:
        out_ptr[0] = total[0]


def broadcast_stat_kernel[
    BATCH: Int, T: Int, P: Int, dtype: DType
](
    stat_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output_t: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    """Replicate scaled stat to every output[b, 0]."""
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var prefactor = Scalar[dtype](Float64(BATCH) / Float64(T * P))
    var stat = stat_ptr[0]
    output_t[b, 0] = stat * prefactor


def reduce_g_kernel[
    BATCH: Int, dtype: DType
](
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ],
    g_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    """Single block; block.sum of grad_output[b, 0] into g_ptr[0].

    Assumes BATCH ≤ TPB. Each thread either owns one b or contributes 0.
    """
    var b = Int(thread_idx.x)
    var v = Scalar[dtype](0)
    if b < BATCH:
        v = rebind[Scalar[dtype]](grad_output[b, 0])
    var total = block.sum[block_size=TPB, broadcast=False](
        val=SIMD[dtype, 1](v)
    )
    if b == 0:
        g_ptr[0] = total[0]


def dLdz_kernel[
    BATCH: Int, T: Int, P: Int, K: Int, dtype: DType
](
    cache_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * P), MutAnyOrigin
    ],
    cm_t: LayoutTensor[dtype, Layout.row_major(T, P * K), MutAnyOrigin],
    sm_t: LayoutTensor[dtype, Layout.row_major(T, P * K), MutAnyOrigin],
    dLdz_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * P), MutAnyOrigin
    ],
    g_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    """One thread per (b, t, p); chain rule through cos/sin to dLdz."""
    comptime assert dtype.is_floating_point(), "dtype must be FP"
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * P:
        return
    var p_idx = idx % P
    var t_idx = (idx // P) % T
    var b = idx // (T * P)
    var z = rebind[Scalar[dtype]](cache_t[b, t_idx * P + p_idx])
    var dt = Scalar[dtype](3.0 / Float64(K - 1))

    var acc = Scalar[dtype](0)
    for k in range(K):
        var tk = dt * Scalar[dtype](k)
        var phi = exp(Scalar[dtype](-0.5) * tk * tk)
        var trap = (
            dt if (k == 0 or k == K - 1) else dt * Scalar[dtype](2.0)
        )
        var wk = trap * phi
        var cm = rebind[Scalar[dtype]](cm_t[t_idx, p_idx * K + k])
        var sm = rebind[Scalar[dtype]](sm_t[t_idx, p_idx * K + k])
        var arg = z * tk
        var s_arg = sin(arg)
        var c_arg = cos(arg)
        var bracket = -(cm - phi) * s_arg + sm * c_arg
        acc += wk * tk * bracket

    var G = g_ptr[0]
    var coef = G * Scalar[dtype](2.0 / Float64(T * P))
    dLdz_t[b, t_idx * P + p_idx] = coef * acc


def matmul_a_kernel[
    BATCH: Int, T: Int, D: Int, P: Int, dtype: DType
](
    dLdz_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * P), MutAnyOrigin
    ],
    a_t: LayoutTensor[dtype, Layout.row_major(D, P), MutAnyOrigin],
    grad_input_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * D), MutAnyOrigin
    ],
):
    """grad_input[b, t*D+d] = sum_p A[d, p] * dLdz[b, t, p]."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * D:
        return
    var d_idx = idx % D
    var t_idx = (idx // D) % T
    var b = idx // (T * D)
    var acc = Scalar[dtype](0)
    for p in range(P):
        var dL = rebind[Scalar[dtype]](dLdz_t[b, t_idx * P + p])
        var aval = rebind[Scalar[dtype]](a_t[d_idx, p])
        acc += aval * dL
    grad_input_t[b, t_idx * D + d_idx] = acc
