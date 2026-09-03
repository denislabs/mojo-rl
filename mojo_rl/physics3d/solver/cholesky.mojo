"""Dense Cholesky utilities for small NV×NV matrices.

Used by primal Newton solver for Hessian factorization and solve.
These operate on `Scratch` for register-friendly small matrices
(NV is typically 6-30 for robotics models).

Functions:
- chol_factor: In-place Cholesky L*L^T = H (lower triangular) [CPU, uses List]
- chol_solve: Solve H*x = b given Cholesky factor L [CPU, uses List]
- chol_factor_inline: Same as chol_factor but uses Scratch [GPU-compatible]
- chol_solve_inline: Same as chol_solve but uses Scratch [GPU-compatible]
- chol_rank1_update: Rank-1 update H ← H ± v*v^T with Cholesky factor update

## 2b.2: `nv` is a RUNTIME argument here, and that is the whole point

Every `NV` below was a comptime parameter, and every call site bound it to
`D.CAP_NV`. Inside this file NV is used only two ways — as a loop bound and as
the row stride of `L[i * NV + j]` — and it is a *cap* at the call site. On the
static leg `CAP_NV == NV`, so the two agree and no gate that runs today can
tell them apart; on a dynamic provider the cap is not the model's NV and every
one of these routines would factor the wrong matrix, silently, because every
offset it produces still lands inside the array.

So NV became the runtime `nv` and the comptime parameters that remain
(`M_CAP`, `V_CAP`) size containers and nothing else. See
`fields/scratch.mojo` for why a cap is 0 rather than -1 on the dynamic leg.
"""

from std.math import sqrt
from max.gpu.memory import AddressSpace
from ..fields.scratch import Scratch

# MuJoCo's `mjMINVAL` (`mjmodel.h`), which is what `mj_solNewton` passes as
# `mindiag` to `mju_cholFactor` (`engine_solver.c:2038`, `:2068`).
comptime _MJMINVAL: Float64 = 1e-15


@always_inline
def chol_factor[
    DTYPE: DType,
](H: List[Scalar[DTYPE]], mut L: List[Scalar[DTYPE]], nv: Int) -> Bool:
    """In-place Cholesky factorization: L*L^T = H (lower triangular).

    H must be symmetric positive definite. L is output lower triangular.
    Both are nv×nv row-major.

    Returns True if successful, False if rank-deficient (diagonal < threshold).
    When False, L still contains a usable factorization (with clamped diagonals),
    but the caller should add regularization and retry.
    """
    var rank_ok = True

    # Zero L
    for i in range(nv * nv):
        L[i] = Scalar[DTYPE](0)

    for i in range(nv):
        for j in range(i + 1):
            var s: Scalar[DTYPE] = 0
            for k in range(j):
                s += L[i * nv + k] * L[j * nv + k]
            if i == j:
                var diag = H[i * nv + i] - s
                # ⚠⚠ THE THRESHOLD IS `mjMINVAL`, AND `1e-10` WAS OURS, NOT
                # MuJoCo'S. `mju_cholFactor` takes `mindiag` as an ARGUMENT
                # (`engine_util_solve.c:33`) and the Newton solver passes
                # `mjMINVAL` at both of its call sites
                # (`engine_solver.c:2038` and `:2068`). The `1e-10` that used
                # to be here is the value MuJoCo passes from a DIFFERENT
                # routine (`engine_util_solve.c:1284`), five orders looser, and
                # it arrived here without that provenance.
                #
                # ⚠ IT IS ALSO A RANK DETECTOR, NOT A CONDITIONER. MuJoCo
                # floors the pivot and decrements `rank` so the factorization
                # cannot produce NaN on a singular matrix; it does not try to
                # improve conditioning, and the Newton path IGNORES the
                # returned rank entirely.
                #
                # MEASURED, because a threshold nobody reaches is not a fix:
                # instrumented at 1e-6 — four orders ABOVE the guard — it fired
                # ZERO times across all 85 Menagerie scenes and zero times over
                # `reassemble_5`'s twelve towers at BOTH dtypes. So this change
                # is a provable no-op today and is here to remove an invented
                # constant, not to alter behaviour.
                #
                # ⚠ `PHYSICS3D_CONTACT_FIDELITY_REASSEMBLE5.md` §5.1 filed this
                # as "an absolute guard against a float32 noise floor of
                # 4.8e-06". That framing does not survive the measurement: the
                # 4.8e-06 is a RELATIVE pivot on a Hessian whose diagonal spans
                # 1.1e6, so the absolute pivot is ~2 and nowhere near any
                # threshold. MuJoCo has no relative guard, and inventing one
                # (Jacobi equilibration) was already tried and refuted at
                # 2435x worse.
                if diag < Scalar[DTYPE](_MJMINVAL):
                    rank_ok = False
                    diag = Scalar[DTYPE](_MJMINVAL)
                L[i * nv + j] = sqrt(diag)
            else:
                L[i * nv + j] = (H[i * nv + j] - s) / L[j * nv + j]

    return rank_ok


@always_inline
def chol_solve[
    DTYPE: DType,
    V_CAP: Int,
](
    L: List[Scalar[DTYPE]],
    b: List[Scalar[DTYPE]],
    mut x: List[Scalar[DTYPE]],
    nv: Int,
):
    """Solve H*x = b given Cholesky factor L (where H = L*L^T).

    Two-phase: forward substitution L*y = b, then back substitution L^T*x = y.
    """
    # Forward substitution: L*y = b
    var y = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    for i in range(nv):
        var s: Scalar[DTYPE] = 0
        for j in range(i):
            s += L[i * nv + j] * y[j]
        y[i] = (b[i] - s) / L[i * nv + i]

    # Back substitution: L^T*x = y
    for i_rev in range(nv):
        var i = nv - 1 - i_rev
        var s: Scalar[DTYPE] = 0
        for j in range(i + 1, nv):
            s += L[j * nv + i] * x[j]
        x[i] = (y[i] - s) / L[i * nv + i]


@always_inline
def chol_factor_inline[
    DTYPE: DType,
    M_CAP: Int,
](
    H: Scratch[Scalar[DTYPE], M_CAP],
    mut L: Scratch[Scalar[DTYPE], M_CAP],
    nv: Int,
) -> Bool:
    """In-place Cholesky factorization: L*L^T = H (lower triangular), GPU-compatible.

    Returns True if successful, False if rank-deficient.

    ⚠ THE WHOLE MATRIX IS ONE SEGMENT — this is `chol_factor_seg` over
    `[0, nv)` and nothing else, so the two cannot drift. See that function for
    why a segment exists at all.
    """
    for i in range(nv * nv):
        L[i] = Scalar[DTYPE](0)
    return chol_factor_seg[DTYPE, M_CAP](H, L, nv, 0, nv)


@always_inline
def chol_factor_seg[
    DTYPE: DType,
    M_CAP: Int,
](
    H: Scratch[Scalar[DTYPE], M_CAP],
    mut L: Scratch[Scalar[DTYPE], M_CAP],
    nv: Int,
    s0: Int,
    s1: Int,
) -> Bool:
    """Factor the DIAGONAL SUB-BLOCK `[s0, s1)` of an `nv x nv` `H` into `L`.

    ⚠⚠ IT DOES NOT ZERO `L`. Segments are factored one after another into the
    same buffer, so a per-call zeroing would wipe the block before it. The
    caller zeroes once — `chol_factor_inline` above does exactly that.

    WHY. `H = M + sum D*J^T J` is block-diagonal over the kinematic trees,
    merged by whichever trees a constraint row couples
    (`solver/newton_blocks.build_dof_segments`). P0 measured the dense
    factorisation at 70% of GPU time on `so101_park_k9`, where nine of ten
    trees carry no constraint row at all: one 6^3 plus nine diagonals is 270
    operations against 216,000.

    ⚠ RESTRICTING THE LOOPS IS BIT-EXACT, NOT AN APPROXIMATION. Every entry
    this skips is `L[i*nv+k] * L[j*nv+k]` with `k` outside the block, and `L`
    there is exactly `0` — zeroed by the caller and never written, because no
    segment owns those columns. A sequential accumulation that drops exact
    zeros returns the identical bit pattern, which is why
    `chol_factor_inline` can delegate here and stay byte-for-byte what it was.
    """
    var rank_ok = True

    for i in range(s0, s1):
        for j in range(s0, i + 1):
            var s: Scalar[DTYPE] = 0
            for k in range(s0, j):
                s += L[i * nv + k] * L[j * nv + k]
            if i == j:
                var diag = H[i * nv + i] - s
                # ⚠⚠ THE THRESHOLD IS `mjMINVAL`, AND `1e-10` WAS OURS, NOT
                # MuJoCo'S. `mju_cholFactor` takes `mindiag` as an ARGUMENT
                # (`engine_util_solve.c:33`) and the Newton solver passes
                # `mjMINVAL` at both of its call sites
                # (`engine_solver.c:2038` and `:2068`). The `1e-10` that used
                # to be here is the value MuJoCo passes from a DIFFERENT
                # routine (`engine_util_solve.c:1284`), five orders looser, and
                # it arrived here without that provenance.
                #
                # ⚠ IT IS ALSO A RANK DETECTOR, NOT A CONDITIONER. MuJoCo
                # floors the pivot and decrements `rank` so the factorization
                # cannot produce NaN on a singular matrix; it does not try to
                # improve conditioning, and the Newton path IGNORES the
                # returned rank entirely.
                #
                # MEASURED, because a threshold nobody reaches is not a fix:
                # instrumented at 1e-6 — four orders ABOVE the guard — it fired
                # ZERO times across all 85 Menagerie scenes and zero times over
                # `reassemble_5`'s twelve towers at BOTH dtypes. So this change
                # is a provable no-op today and is here to remove an invented
                # constant, not to alter behaviour.
                #
                # ⚠ `PHYSICS3D_CONTACT_FIDELITY_REASSEMBLE5.md` §5.1 filed this
                # as "an absolute guard against a float32 noise floor of
                # 4.8e-06". That framing does not survive the measurement: the
                # 4.8e-06 is a RELATIVE pivot on a Hessian whose diagonal spans
                # 1.1e6, so the absolute pivot is ~2 and nowhere near any
                # threshold. MuJoCo has no relative guard, and inventing one
                # (Jacobi equilibration) was already tried and refuted at
                # 2435x worse.
                if diag < Scalar[DTYPE](_MJMINVAL):
                    rank_ok = False
                    diag = Scalar[DTYPE](_MJMINVAL)
                L[i * nv + j] = sqrt(diag)
            else:
                L[i * nv + j] = (H[i * nv + j] - s) / L[j * nv + j]

    return rank_ok


@always_inline
def chol_solve_inline[
    DTYPE: DType,
    M_CAP: Int,
    V_CAP: Int,
](
    # `mut` for the same reason as `chol_solve_seg` below — taking a pointer
    # out of a `Scratch` needs a mutable borrow; neither is written.
    mut L: Scratch[Scalar[DTYPE], M_CAP],
    mut b: Scratch[Scalar[DTYPE], V_CAP],
    mut x: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
):
    """Solve H*x = b given Cholesky factor L (where H = L*L^T), GPU-compatible.

    Same algorithm as chol_solve but operates on `Scratch` so it can be
    used inside @always_inline GPU kernels without heap allocation.
    L is nv×nv in an M_CAP array, b/x are nv in V_CAP arrays.
    Two-phase: forward substitution L*y = b, then back substitution L^T*x = y.

    ⚠ THE WHOLE VECTOR IS ONE SEGMENT — `chol_solve_seg` over `[0, nv)`.
    """
    chol_solve_seg[DTYPE, M_CAP, V_CAP](L, b, x, nv, 0, nv)


@always_inline
def chol_solve_seg[
    DTYPE: DType,
    M_CAP: Int,
    V_CAP: Int,
](
    # ⚠ `mut` ON `L` AND `b`, WHICH ARE READ-ONLY TO THE ALGORITHM. `Scratch`
    # hands out a pointer only through `unsafe_ptr[SO: MutOrigin](ref [SO]
    # self)`, so taking the address at all requires a mutable borrow. The body
    # below never writes through either — see its docstring.
    mut L: Scratch[Scalar[DTYPE], M_CAP],
    mut b: Scratch[Scalar[DTYPE], V_CAP],
    mut x: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
    s0: Int,
    s1: Int,
):
    """`Scratch` adapter over `chol_solve_seg_p` — see it for the algorithm.

    Every per-thread caller (the per-env solvers, `chol_solve_inline`, the
    tests) holds its operands as `Scratch`, so this spelling stays. It owns no
    arithmetic: it takes the three pointers and delegates.
    """
    chol_solve_seg_p[DTYPE, V_CAP](
        L.unsafe_ptr(), b.unsafe_ptr(), x.unsafe_ptr(), nv, s0, s1
    )


@always_inline
def chol_solve_seg_p[
    LO: MutOrigin,
    BO: MutOrigin,
    XO: MutOrigin, //,
    DTYPE: DType,
    V_CAP: Int,
    # GENERIC is the per-thread caller (`Scratch`, via the adapter above). The
    # blocked Newton kernel passes `L_sh.ptr` — SHARED — so the factor is read
    # WHERE IT WAS WRITTEN and there is no copy. See `noslip_pyramidal`, which
    # takes its row storage the same way and for the same reason.
    L_AS: AddressSpace = AddressSpace.GENERIC,
    B_AS: AddressSpace = AddressSpace.GENERIC,
    X_AS: AddressSpace = AddressSpace.GENERIC,
](
    L: Pointer[Scalar[DTYPE], LO, address_space=L_AS],
    b: Pointer[Scalar[DTYPE], BO, address_space=B_AS],
    x: Pointer[Scalar[DTYPE], XO, address_space=X_AS],
    nv: Int,
    s0: Int,
    s1: Int,
):
    """Solve the `[s0, s1)` sub-system of `H*x = b` given `L` from
    `chol_factor_seg`.

    ⚠ THE SEGMENTS ARE INDEPENDENT SYSTEMS, which is the whole point: `L` has
    no entry linking two of them, so solving each in turn over its own range
    gives the same `x` as one solve over `[0, nv)` — bit for bit, by the same
    exact-zero argument as the factorisation. Entries of `x` outside every
    segment are never written, and with a partition that tiles `[0, nv)` there
    are none.

    ⚠ IT READS `L` AND WRITES ONLY `x` (and its own `y`), which is what lets
    the blocked kernel hand it SHARED memory that other threads are also
    reading. A caller that runs several segments CONCURRENTLY is therefore
    safe iff the segments are disjoint — they are, by construction — but it
    must still barrier before the first call, because `L` is only complete
    after the cooperative factorisation's last barrier.
    """
    # Forward substitution: L*y = b
    var y = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    for i in range(s0, s1):
        var s: Scalar[DTYPE] = 0
        for j in range(s0, i):
            s += L[i * nv + j] * y[j]
        y[i] = (b[i] - s) / L[i * nv + i]

    # Back substitution: L^T*x = y
    for i_rev in range(s1 - s0):
        var i = s1 - 1 - i_rev
        var s: Scalar[DTYPE] = 0
        for j in range(i + 1, s1):
            s += L[j * nv + i] * x[j]
        x[i] = (y[i] - s) / L[i * nv + i]


@always_inline
def chol_rank1_update[
    DTYPE: DType,
    M_CAP: Int,
    V_CAP: Int,
](
    mut L: Scratch[Scalar[DTYPE], M_CAP],
    v: Scratch[Scalar[DTYPE], V_CAP],
    sign: Scalar[DTYPE],
    nv: Int,
):
    """Rank-1 Cholesky update: H ← H + sign * v * v^T.

    sign = +1 for update (adding), sign = -1 for downdate (removing).
    Modifies L in-place. Uses the standard rank-1 Cholesky update algorithm.

    For downdate (sign=-1), the result may not be PD if v is too large.
    In that case, diagonal elements are clamped to a small positive value.
    """
    # Work on a copy of v that gets modified
    var w = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    for i in range(nv):
        w[i] = v[i]

    for i in range(nv):
        var L_ii = L[i * nv + i]
        var w_i = w[i]

        var r_sq = L_ii * L_ii + sign * w_i * w_i
        if r_sq < Scalar[DTYPE](1e-14):
            r_sq = Scalar[DTYPE](1e-14)
        var r = sqrt(r_sq)

        var c = r / L_ii
        var s_val = w_i / L_ii

        L[i * nv + i] = r

        # Update remaining elements in column i
        for j in range(i + 1, nv):
            L[j * nv + i] = (L[j * nv + i] + sign * s_val * w[j]) / c
            w[j] = c * w[j] - s_val * L[j * nv + i]
