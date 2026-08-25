"""The ELLIPTIC friction cone at any condim: state, force, Hessian, linesearch.

Transcribed from MuJoCo's `mjCNSTR_CONTACT_ELLIPTIC` branches:

    state + force + cone Hessian   engine_core_constraint.c  mj_constraintUpdate_impl
    linesearch quadratics          engine_solver.c           PrimalPrepare / PrimalEval

⚠⚠ WRITTEN IN MuJoCo's U-SPACE, WHICH IS WHY IT GENERALIZES. The solver used to
carry the same arithmetic with the friction coefficient FACTORED OUT: one
`mu` per contact, `T = |jar_t|` unscaled, two hard-coded tangents. That
factoring is algebraically identical to MuJoCo — but ONLY when every
tangential row shares one coefficient, which is exactly the condim-3
isotropic-slide case. The moment a contact has a torsional row
(`friction = "1 1 0.005"`) the cone stops being circular in `jar` and the
factored form has no way to say so. So the rows are kept RAW here and mapped
the way MuJoCo maps them:

    U[0] = jar_n * mu            N = U[0]
    U[t] = jar_t[t] * fr[t]      T = |U[1..]|

with `mu = con->mu` the REGULARIZED coefficient and `fr[t] = con->friction[t]`
the raw per-direction one. The three zones are then `N >= mu*T` (top, no
force), `mu*N + T <= 0` (bottom, unconstrained quadratic) and the cone surface
between them. At condim 3 with `fr[0] == fr[1]` this reduces algebraically to
the old expressions — verified term by term against them before the switch, so
condim-3 gates do not move.

⚠ THE `nt` ARGUMENT IS PER CONTACT AND RUNTIME. `NT` sizes the arrays for the
model's worst condim; `nt` is THIS contact's `dim-1`, because condim is a
property of the geom pair. Rows `t >= nt` are zeroed by the producer, but every
loop here is bounded by `nt` rather than relying on that — a zero row would
still contribute a `0*0` term to `T` and change nothing, whereas a zero
`fr[t]` in a denominator would not be so forgiving.

`nt == 0` is a FRICTIONLESS contact (`condim="1"`). It is not a special case:
`T` is 0, so the zone test collapses to `N >= 0` versus `N < 0`, which is the
one-sided normal constraint MuJoCo emits as `mjCNSTR_CONTACT_FRICTIONLESS`.
"""

from std.math import sqrt

from ..fields.scratch import Scratch


# Constraint states, matching `mjCNSTRSTATE_*` for the values this cone can
# produce. The solver stores these per contact in `cs_arr`.
comptime ELL_SATISFIED: Int = 0
comptime ELL_QUADRATIC: Int = 1
comptime ELL_CONE: Int = 2

# `mjMINVAL`. Used only where MuJoCo's own guarantee (`T > 0` strictly in the
# middle zone) is one float32 rounding away from failing.
comptime ELL_MINVAL: Float64 = 1e-12


@always_inline
def ell_state_force[
    DTYPE: DType, NT: Int, T_CAP: Int
](
    nt: Int,
    base: Int,
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    mu: Scalar[DTYPE],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    fr: Scratch[Scalar[DTYPE], T_CAP],
    mut f_n: Scalar[DTYPE],
    mut f_t: Scratch[Scalar[DTYPE], T_CAP],
) -> Int:
    """Zone, normal force and tangential forces for one contact.

    `base` indexes the contact's row block (`c*NT`) in the flat arrays;
    `jar_t[base+t]` is row `t`. Returns one of `ELL_SATISFIED` /
    `ELL_QUADRATIC` / `ELL_CONE` and writes `f_n` / `f_t[base..base+nt)`.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime ONE = Scalar[DTYPE](1)
    comptime MINVAL = Scalar[DTYPE](ELL_MINVAL)

    var N = jar_n * mu
    var T_sq = ZERO
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        T_sq += u * u
    var T = sqrt(T_sq)

    # top zone: no force at all
    if N >= mu * T or (T <= ZERO and N >= ZERO):
        f_n = ZERO
        for t in range(nt):
            f_t[base + t] = ZERO
        return ELL_SATISFIED

    # bottom zone: the unconstrained quadratic, one independent row each
    if mu * N + T <= ZERO or (T <= ZERO and N < ZERO):
        f_n = -D_n * jar_n
        for t in range(nt):
            f_t[base + t] = -D_t[base + t] * jar_t[base + t]
        return ELL_QUADRATIC

    # middle zone: on the cone surface.
    # `T > 0` STRICTLY here — `T == 0` forces `mu*N + T = mu*N <= 0` given
    # `N < mu*T = 0`, which is the bottom zone. The floor is float32 paranoia,
    # not a behaviour, and MuJoCo does not carry one.
    var T_s = T if T > MINVAL else MINVAL
    var Dm = D_n / (mu * mu * (ONE + mu * mu))
    var NmT = N - mu * T
    f_n = -Dm * NmT * mu
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        f_t[base + t] = -f_n / T_s * u * fr[base + t]
    return ELL_CONE


@always_inline
def ell_hessian_block[
    DTYPE: DType, NT: Int, T_CAP: Int, HN: Int
](
    state: Int,
    nt: Int,
    base: Int,
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    mu: Scalar[DTYPE],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    fr: Scratch[Scalar[DTYPE], T_CAP],
    mut Hb: InlineArray[Scalar[DTYPE], HN],
):
    """The contact's `dim x dim` Hessian block in ROW space, row-major over
    `(n, t_0, ..., t_{nt-1})` with stride `NT+1`.

    The caller turns this into the `nv x nv` contribution as
    `sum_{k,j} Hb[k,j] * J_k J_j^T`. Splitting it out is what lets QUADRATIC
    and CONE share one accumulation loop — they used to be two hand-fused
    copies of the same six outer products, written out twice more inside the
    Newton loop for the state-change rebuild.

    ⚠ `Hb` IS ZEROED HERE FOR THE FULL `(NT+1)^2`, not just the live `dim`
    rows. The caller loops to `nt+1`, but a stale entry from a contact with
    more rows would otherwise be read if that ever changed.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime ONE = Scalar[DTYPE](1)
    comptime MINVAL = Scalar[DTYPE](ELL_MINVAL)
    comptime DIM = NT + 1

    for k in range(HN):
        Hb[k] = ZERO
    if state == ELL_SATISFIED:
        return

    if state == ELL_QUADRATIC:
        # Independent rows: diag(D_n, D_t...).
        Hb[0] = D_n
        for t in range(nt):
            Hb[(t + 1) * DIM + (t + 1)] = D_t[base + t]
        return

    # CONE. Verbatim from `mj_constraintUpdate_impl`'s `flg_coneHessian`
    # block, including the order of operations: build in U-space, then pre-
    # and post-multiply by `diag(mu, friction)` and scale by `Dm`.
    var N = jar_n * mu
    var T_sq = ZERO
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        T_sq += u * u
    var T = sqrt(T_sq)
    var T_s = T if T > MINVAL else MINVAL
    var Dm = D_n / (mu * mu * (ONE + mu * mu))

    # first row: (1, -mu/T * U)
    Hb[0] = ONE
    var scl = -mu / T_s
    for t in range(nt):
        Hb[t + 1] = scl * (jar_t[base + t] * fr[base + t])

    # upper block: mu*N/T^3 * U U'
    scl = mu * N / (T_s * T_s * T_s)
    for k in range(nt):
        var uk = jar_t[base + k] * fr[base + k]
        for j in range(k, nt):
            var uj = jar_t[base + j] * fr[base + j]
            Hb[(k + 1) * DIM + (j + 1)] = scl * uj * uk

    # diagonal: += (mu^2 - mu*N/T)
    scl = mu * mu - mu * N / T_s
    for t in range(nt):
        Hb[(t + 1) * DIM + (t + 1)] += scl

    # pre/post multiply by diag(mu, friction), scale by Dm
    for k in range(nt + 1):
        var sk = Dm * (mu if k == 0 else fr[base + k - 1])
        for j in range(k, nt + 1):
            Hb[k * DIM + j] *= sk * (mu if j == 0 else fr[base + j - 1])

    # symmetrize
    for k in range(nt + 1):
        for j in range(k + 1, nt + 1):
            Hb[j * DIM + k] = Hb[k * DIM + j]

    # ── PSD PROJECTION — `HessianCone` (engine_solver.c:2052) ───────────────
    #
    # ⚠⚠ THE CONE HESSIAN IS INDEFINITE AND MuJoCo NEVER ADDS IT RAW. Its
    # middle-zone cost is `0.5*Dm*(N - mu*T)^2`, whose Hessian carries the term
    # `Dm*(N - mu*T)*Hess(N - mu*T)` — and `N - mu*T < 0` is the very condition
    # that DEFINES the middle zone, so that term is negative-curvature by
    # construction. `HessianCone` factors the block with
    # `mju_cholFactor(local, dim, mjMINVAL)`, whose diagonal CLAMP turns it
    # into a PSD matrix, and then applies `dim` rank-1 `mju_cholUpdate`s of
    # `L' J`. The Hessian the reference's Newton direction is computed against
    # is therefore `J' (L L') J`, not `J' H J`.
    #
    # ⚠ THE SYMPTOM OF ADDING IT RAW IS A ZIG-ZAG, NOT A BLOW-UP. An indefinite
    # Hessian gives a direction the linesearch has to cut, and the solver
    # enters a period-2 cycle: measured on `unitree_go1` at `impratio="100"`,
    # `alpha` alternated 0.0895 / 0.2317 for hundreds of iterations while
    # `scale*|grad|` decayed ~5% per PAIR of steps. MuJoCo converges that pose
    # in **6** iterations and we needed ~800 — the residual on board row
    # `unitree_go1` was this, not the cone algebra.
    #
    # `L L'` is reconstructed here rather than threaded out as a factor,
    # because the accumulation below already forms `J' Hb J` and the two are
    # the same matrix.
    var n = nt + 1
    for j in range(n):
        var tj = Hb[j * DIM + j]
        for k in range(j):
            tj -= Hb[j * DIM + k] * Hb[j * DIM + k]
        # `mjMINVAL`, and the clamp IS the projection — a pivot at or below it
        # is a direction of non-positive curvature, and MuJoCo keeps the
        # matrix usable rather than declaring the factorization failed.
        if tj < Scalar[DTYPE](1e-15):
            tj = Scalar[DTYPE](1e-15)
        Hb[j * DIM + j] = sqrt(tj)
        var inv = ONE / Hb[j * DIM + j]
        for i in range(j + 1, n):
            var v = Hb[i * DIM + j]
            for k in range(j):
                v -= Hb[i * DIM + k] * Hb[j * DIM + k]
            Hb[i * DIM + j] = v * inv
    # `Hb <- L L'`, lower factor read in place. Walk k from the LAST column
    # backwards so a row's own factor entries are still intact when read.
    for i in range(n - 1, -1, -1):
        for j in range(i, -1, -1):
            var acc = ZERO
            for k in range(j + 1):
                acc += Hb[i * DIM + k] * Hb[j * DIM + k]
            Hb[i * DIM + j] = acc
    for k in range(n):
        for j in range(k + 1, n):
            Hb[k * DIM + j] = Hb[j * DIM + k]


@always_inline
def ell_add_contact_hessian[
    DTYPE: DType,
    MC_CAP: Int,
    NT: Int,
    T_CAP: Int,
    V_CAP: Int,
    M_CAP: Int,
    HN: Int,
](
    nc: Int,
    cs_arr: Scratch[Int, MC_CAP],
    nt_cache: Scratch[Int, MC_CAP],
    Jn_c: Scratch[Scalar[DTYPE], MC_CAP * V_CAP],
    Jt_c: Scratch[Scalar[DTYPE], T_CAP * V_CAP],
    jar_n_arr: Scratch[Scalar[DTYPE], MC_CAP],
    jar_t_arr: Scratch[Scalar[DTYPE], T_CAP],
    mu_cache: Scratch[Scalar[DTYPE], MC_CAP],
    D_n_cache: Scratch[Scalar[DTYPE], MC_CAP],
    D_t_cache: Scratch[Scalar[DTYPE], T_CAP],
    fr_cache: Scratch[Scalar[DTYPE], T_CAP],
    mut H: Scratch[Scalar[DTYPE], M_CAP],
    nv: Int,
):
    """Add every contact's `J^T Hb J` to the `nv x nv` Newton Hessian.

    Two-stage — `JH[k] = sum_j Hb[k,j] J_j`, then `H += sum_k J_k JH[k]^T` —
    which is `O(dim^2 nv + dim nv^2)` rather than the `O(dim^2 nv^2)` a naive
    double loop would cost. At condim 3 that is THREE rank-1 outer products
    where the hand-fused two-tangent version it replaces did six, so the
    generalization is not a slowdown even before the extra rows.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime DIM = NT + 1
    # `Hb` is (NT+1)^2 -- CONDIM-derived, so it stays a real InlineArray and
    # keeps its comptime bound. `JH` is DIM rows of `nv`: the row COUNT is
    # condim, the row LENGTH is the dof count, and only the latter goes
    # dynamic. Not every comptime size in this file is a model dimension.
    var Hb = InlineArray[Scalar[DTYPE], HN](fill=ZERO)
    var JH = Scratch[Scalar[DTYPE], DIM * V_CAP](DIM * nv, fill=ZERO)

    for c in range(nc):
        var cs = cs_arr[c]
        if cs == ELL_SATISFIED:
            continue
        var nt_c = nt_cache[c]
        ell_hessian_block[DTYPE, NT, T_CAP, HN](
            cs, nt_c, c * NT, jar_n_arr[c], jar_t_arr,
            mu_cache[c], D_n_cache[c], D_t_cache, fr_cache, Hb,
        )

        for k in range(nt_c + 1):
            for i in range(nv):
                JH[k * nv + i] = ZERO
            for j in range(nt_c + 1):
                var h = Hb[k * DIM + j]
                if h == ZERO:
                    continue
                if j == 0:
                    for i in range(nv):
                        JH[k * nv + i] += h * Jn_c[c * nv + i]
                else:
                    var jb = (c * NT + j - 1) * nv
                    for i in range(nv):
                        JH[k * nv + i] += h * Jt_c[jb + i]

        for k in range(nt_c + 1):
            var kb = c * nv if k == 0 else (c * NT + k - 1) * nv
            for i in range(nv):
                var jki = Jn_c[kb + i] if k == 0 else Jt_c[kb + i]
                if jki == ZERO:
                    continue
                for j in range(nv):
                    H[i * nv + j] += jki * JH[k * nv + j]


@always_inline
def ell_line_deriv[
    DTYPE: DType, NT: Int, T_CAP: Int
](
    nt: Int,
    base: Int,
    alpha: Scalar[DTYPE],
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    Js_n: Scalar[DTYPE],
    Js_t: Scratch[Scalar[DTYPE], T_CAP],
    mu: Scalar[DTYPE],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    fr: Scratch[Scalar[DTYPE], T_CAP],
    mut d1: Scalar[DTYPE],
    mut d2: Scalar[DTYPE],
):
    """Add this contact's contribution to the linesearch derivatives at
    `alpha` — first into `d1`, second into `d2`.

    Port of `PrimalEval`'s elliptic branch fused with the `quad` terms
    `PrimalPrepare` builds for it. MuJoCo accumulates the bottom-zone
    quadratic into `quadTotal` and differentiates once at the end; per row
    that is `d1 += 2*alpha*q2 + q1` and `d2 += 2*q2`, which is the form used
    here so no cross-row state is needed.

    The caller evaluates this at four different `alpha` (the trial point, the
    initial Newton step, each one-sided pursuit step, each bisection midpoint).
    Those were four hand-inlined copies of the two-tangent expressions; making
    them one call is the only reason generalizing the tangent count is a
    tractable edit rather than a fourfold one.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime ONE = Scalar[DTYPE](1)
    comptime MINVAL = Scalar[DTYPE](ELL_MINVAL)

    # U/V: the ray `jar + alpha*Js` mapped into the space where the cone is
    # circular. `UU/UV/VV` are `PrimalPrepare`'s quad[5..7].
    var U0 = jar_n * mu
    var V0 = Js_n * mu
    var UU = ZERO
    var UV = ZERO
    var VV = ZERO
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        var v = Js_t[base + t] * fr[base + t]
        UU += u * u
        UV += u * v
        VV += v * v

    var N = U0 + alpha * V0
    var T_sq = UU + alpha * (Scalar[DTYPE](2) * UV + alpha * VV)

    # No tangential force anywhere along the ray: top or bottom by sign of N.
    if T_sq <= ZERO:
        if N < ZERO:
            _ell_quad_deriv[DTYPE, NT, T_CAP](
                nt, base, alpha, jar_n, jar_t, Js_n, Js_t, D_n, D_t, d1, d2
            )
        return

    var T = sqrt(T_sq)
    if N >= mu * T:
        return  # top zone: no cost
    if mu * N + T <= ZERO:
        _ell_quad_deriv[DTYPE, NT, T_CAP](
            nt, base, alpha, jar_n, jar_t, Js_n, Js_t, D_n, D_t, d1, d2
        )
        return

    # middle zone
    var T_s = T if T > MINVAL else MINVAL
    var Dm = D_n / (mu * mu * (ONE + mu * mu))
    var N1 = V0
    var T1 = (UV + alpha * VV) / T_s
    var T2 = VV / T_s - (UV + alpha * VV) * T1 / (T_s * T_s)
    var NmT = N - mu * T
    var dN = N1 - mu * T1
    d1 += Dm * NmT * dN
    d2 += Dm * (dN * dN + NmT * (-mu * T2))


@always_inline
def _ell_quad_deriv[
    DTYPE: DType, NT: Int, T_CAP: Int
](
    nt: Int,
    base: Int,
    alpha: Scalar[DTYPE],
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    Js_n: Scalar[DTYPE],
    Js_t: Scratch[Scalar[DTYPE], T_CAP],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    mut d1: Scalar[DTYPE],
    mut d2: Scalar[DTYPE],
):
    """Bottom zone: every row is an independent quadratic `0.5*D*jar^2`."""
    var tN = jar_n + alpha * Js_n
    d1 += D_n * tN * Js_n
    d2 += D_n * Js_n * Js_n
    for t in range(nt):
        var d = D_t[base + t]
        var js = Js_t[base + t]
        d1 += d * (jar_t[base + t] + alpha * js) * js
        d2 += d * js * js
