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
from std.sys import simd_width_of
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


# =============================================================================
# SIMD dot / axpy over contiguous rows — the CPU `VEC` path
# =============================================================================
#
# ⚠ NOT BIT-EXACT AGAINST THE SCALAR LOOPS: a `W`-wide accumulator reassociates
# the sum. Mojo does not autovectorise (PERFORMANCE.md §7.1), so this is the
# only way the Cholesky's inner product — 51% of humanoid_CMU's solve, one
# 62x62 factor per iteration at ~1.3 GFLOP/s scalar — gets wider than one
# lane. `VEC` is off by default and set only by the CPU Newton; the GPU legs
# compile the scalar loops they always did. Gated like every other numerics
# change here: the MuJoCo trajectory gates and `test_newton_float32_tracks_float64`.


@always_inline
def _dot_seg[
    AO: MutOrigin,
    BO: MutOrigin, //,
    DTYPE: DType,
    A_AS: AddressSpace = AddressSpace.GENERIC,
    B_AS: AddressSpace = AddressSpace.GENERIC,
](
    a: Pointer[Scalar[DTYPE], AO, address_space=A_AS],
    ao: Int,
    b: Pointer[Scalar[DTYPE], BO, address_space=B_AS],
    bo: Int,
    n: Int,
) -> Scalar[DTYPE]:
    """`sum_k a[ao+k] * b[bo+k]`, `W` lanes at a time, scalar tail."""
    comptime W = 2 * simd_width_of[DTYPE]()
    var acc = SIMD[DTYPE, W](0)
    var k = 0
    while k + W <= n:
        acc += a.load[width=W](ao + k) * b.load[width=W](bo + k)
        k += W
    var s = acc.reduce_add()
    while k < n:
        s += a[ao + k] * b[bo + k]
        k += 1
    return s


@always_inline
def _dot_rows[
    AO: MutOrigin, //,
    DTYPE: DType,
    A_AS: AddressSpace = AddressSpace.GENERIC,
](
    a: Pointer[Scalar[DTYPE], AO, address_space=A_AS],
    ao: Int,
    bo: Int,
    n: Int,
) -> Scalar[DTYPE]:
    """`_dot_seg` over two rows of ONE buffer. The exclusivity checker refuses
    the same mutable pointer in two arguments, so the self-dot the Cholesky
    needs (row i against row j of `L`) takes one pointer and two offsets."""
    comptime W = 2 * simd_width_of[DTYPE]()
    var acc = SIMD[DTYPE, W](0)
    var k = 0
    while k + W <= n:
        acc += a.load[width=W](ao + k) * a.load[width=W](bo + k)
        k += W
    var s = acc.reduce_add()
    while k < n:
        s += a[ao + k] * a[bo + k]
        k += 1
    return s


@always_inline
def _axpy_seg[
    XO: MutOrigin,
    AO: MutOrigin, //,
    DTYPE: DType,
    X_AS: AddressSpace = AddressSpace.GENERIC,
    A_AS: AddressSpace = AddressSpace.GENERIC,
](
    x: Pointer[Scalar[DTYPE], XO, address_space=X_AS],
    xo: Int,
    a: Pointer[Scalar[DTYPE], AO, address_space=A_AS],
    ao: Int,
    alpha: Scalar[DTYPE],
    n: Int,
):
    """`x[xo+k] += alpha * a[ao+k]` for `k < n`, `W` lanes at a time."""
    comptime W = 2 * simd_width_of[DTYPE]()
    var av = SIMD[DTYPE, W](alpha)
    var k = 0
    while k + W <= n:
        x.store(xo + k, x.load[width=W](xo + k) + av * a.load[width=W](ao + k))
        k += W
    while k < n:
        x[xo + k] = x[xo + k] + alpha * a[ao + k]
        k += 1


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
    VEC: Bool = False,
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
    var Lp = L.unsafe_ptr()

    comptime if VEC:
        # COLUMN-OUTER, as `mju_cholFactor` (engine_util_solve.c). Row-outer
        # (`for i: for j <= i`) is a SERIAL CHAIN along each row: the dot for
        # `(i, j+1)` ends on `L[i, j]`, which is a subtract and a divide
        # away, so every pair waits ~18 cycles on the one before it — 25 µs
        # for dog's 79 dofs, latency-bound, not the 82k flops. With the
        # column outermost the `s1 - j` dots of column `j` read rows that
        # are already complete and are independent of each other, so the
        # core overlaps them. THE SAME OPERATIONS IN THE SAME ORDER PER
        # ENTRY — only the order of the entries changes — so the factor is
        # bit-exact with the row-outer form below.
        for j in range(s0, s1):
            var sd = _dot_rows[DTYPE](Lp, j * nv + s0, j * nv + s0, j - s0)
            var diag = H[j * nv + j] - sd
            if diag < Scalar[DTYPE](_MJMINVAL):
                rank_ok = False
                diag = Scalar[DTYPE](_MJMINVAL)
            var d = sqrt(diag)
            L[j * nv + j] = d
            for i in range(j + 1, s1):
                var si = _dot_rows[DTYPE](Lp, i * nv + s0, j * nv + s0, j - s0)
                L[i * nv + j] = (H[i * nv + j] - si) / d
        return rank_ok

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
def chol_update_seg[
    DTYPE: DType,
    M_CAP: Int,
    V_CAP: Int,
](
    mut L: Scratch[Scalar[DTYPE], M_CAP],
    mut x: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
    s0: Int,
    s1: Int,
    plus: Bool,
) -> Bool:
    """Rank-1 update of a Cholesky factor in place over the segment `[s0, s1)`:
    `L Lᵀ ± x xᵀ`, MuJoCo's `mju_cholUpdate` (engine_util_solve.c:96). `x` is
    destroyed. Returns False if a diagonal fell below `mjMINVAL` — the caller
    then recomputes `H` and refactors, as the reference does on rank loss.

    WHY. `mj_solPrimal` factors the Hessian ONCE per solve and, after each
    iteration, updates it with `J_i·√D_i` for every row that entered
    (`+`) or left (`-`) the quadratic zone (engine_solver.c:2120). We were
    rebuilding `H` and refactoring from scratch on every iteration — 3.3
    factorisations of a 62x62 on humanoid_CMU where the reference does one
    plus a handful of O(nv²) updates (PERFORMANCE.md §13.14, item 1).

    ⚠ SEGMENT-RESTRICTED, AND THAT IS EXACT: a row's `x` is nonzero inside one
    segment only, and `L` is zero across segments, so every skipped entry
    would be `0 ± 0`.
    """
    var rank_ok = True
    var Lp = L.unsafe_ptr()
    var xp = x.unsafe_ptr()
    for k in range(s0, s1):
        var xk = x[k]
        if xk == Scalar[DTYPE](0):
            continue
        var Lkk = L[k * nv + k]
        var tmp = Lkk * Lkk + (xk * xk if plus else -(xk * xk))
        if tmp < Scalar[DTYPE](_MJMINVAL):
            tmp = Scalar[DTYPE](_MJMINVAL)
            rank_ok = False
        var r = sqrt(tmp)
        var c = r / Lkk
        var cinv = Scalar[DTYPE](1) / c
        var sc = xk / Lkk
        L[k * nv + k] = r
        # ONE walk down column `k` (stride `nv`), not two: the `x` update
        # reads the entry the `L` update just wrote, so both go in the same
        # pass. Same operations per entry — bit-exact with the two-pass form.
        if plus:
            for i in range(k + 1, s1):
                var l = (Lp[i * nv + k] + sc * xp[i]) * cinv
                Lp[i * nv + k] = l
                xp[i] = c * xp[i] - sc * l
        else:
            for i in range(k + 1, s1):
                var l = (Lp[i * nv + k] - sc * xp[i]) * cinv
                Lp[i * nv + k] = l
                xp[i] = c * xp[i] - sc * l
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
    VEC: Bool = False,
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
    chol_solve_seg_p[DTYPE, V_CAP, VEC=VEC](
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
    VEC: Bool = False,
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
    comptime if VEC:
        # Forward: row i of L against y, both contiguous over [s0, i).
        var yp = y.unsafe_ptr()
        for i in range(s0, s1):
            var s = _dot_seg[DTYPE](L, i * nv + s0, yp, s0, i - s0)
            y[i] = (b[i] - s) / L[i * nv + i]
        # Backward, in AXPY form so the walk stays along rows: the scalar
        # loop gathers a COLUMN of L (stride nv); here x[i] is finished first
        # and then subtracted from every earlier entry along row i.
        for i in range(s0, s1):
            x[i] = y[i]
        for i_rev in range(s1 - s0):
            var i = s1 - 1 - i_rev
            var xi = x[i] / L[i * nv + i]
            x[i] = xi
            _axpy_seg[DTYPE](x, s0, L, i * nv + s0, -xi, i - s0)
        return
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
