"""`mj_solNoSlip` — the friction-only post-solver sweep (PYRAMIDAL cone).

WIRED IN as of 2026-08-03: `solver/newton_solve.mojo`'s PYRAMIDAL per-env path
calls this after the primal solve, under `comptime if NOSLIP_ITER > 0`, with
`NOSLIP_ITER` threaded from `<option noslip_iterations>` via
`ModelDefFromXML` -> `Phyics3dEnv` -> `EulerIntegrator` -> `solve_newton`.
Gated by `tests/physics3d/test_noslip_vs_mujoco.mojo` (compiles, runs, and does
not perturb an already-converged solve) and, for the case where it actually
bites, by dm_control's dog.

⚠ THE ELLIPTIC PATH DOES NOT CALL THIS, so an elliptic model with
`noslip_iterations` set would skip the pass in silence. `ModelDefFromXML`
refuses to build that combination — see the `allow_missing_noslip` assert
there. That refusal IS the dispatch guarantee this module depends on.

WHAT IT IS

After the primal solve converges, MuJoCo optionally runs a Gauss-Seidel sweep
over the FRICTION dimensions only, with the normal forces held fixed, to
remove residual slip. dm_control's dog is the one suite model that asks for it
(`<option timestep="0.005" noslip_iterations="4"/>`).

It is NOT a refinement that can be skipped. Measured MuJoCo-against-MuJoCo with
only the option changed: `max|d(qvel)|` is **2.9e-2 on the FIRST contacting
step**. (Over 200 steps it reaches 2.7, which proves nothing on its own — a
contact-rich rollout is chaotic and any perturbation grows. The first-step
number is the one that settles it, because there is nothing yet to amplify.)

`mj_solNoSlip` is byte-identical across MuJoCo 3.3.6, 3.6.0 and main, so for
once there is no version-drift question — see `feedback_reference_tree_version
_drift`. Transcribed from `engine_solver.c:537`.

THE TWO SIMPLIFICATIONS THAT MAKE THIS TRACTABLE

MuJoCo works in the DUAL formulation and needs `efc_AR = J M^-1 J^T + R`, a
dense `nefc x nefc` matrix we do not build. Both of its uses collapse:

  * `residual(..., flg_subR=1)` at row j expands to

        b_j + (AR f)_j - R_j f_j
      = (J qacc_smooth - aref)_j + (J M^-1 J^T f)_j
      = J_j (qacc_smooth + M^-1 J^T f) - aref_j
      = J_j qacc - aref_j

    which is EXACTLY the primal solver's `jar[j]`. No dual vector needed.

  * `extractBlock(..., flg_subR=1)` is `A` with `R` subtracted off the diagonal
    and the diagonal clamped to 1e-10 — i.e. plain `J_j M^-1 J_k^T`, with NO
    `R` term at all.

So the only genuinely new quantity is `M^-1 J_j^T` per row, for the in-place
`qacc` update that keeps `jar` current between blocks. `m_inv` is already a
dense NV x NV on `Data`, so that is a matvec, not a solve.

WHY THE NORMAL FORCE IS PRESERVED EXACTLY

For a pyramidal contact the friction is carried by opposing edge PAIRS
`(f_j, f_{j+1})`, and the normal contribution of a pair is
`mid = (f_j + f_{j+1})/2`. Every branch below writes the pair as
`(mid + y, mid - y)` for some `y`, so `mid` — and with it the normal force —
is invariant by construction. That is what "with the normal forces frozen"
means operationally, and it is why this cannot destabilise a contact.

SCOPE, STATED RATHER THAN IMPLIED

  * PYRAMIDAL only, BY CONSTRUCTION — the pair arithmetic below IS the
    pyramidal branch, so there is nothing here to check at runtime and this
    function does not take a cone type. The elliptic branch of `mj_solNoSlip`
    is a different algorithm (`mju_QCQP2/3` over the dual block; `solver/
    qcqp.mojo` has those). No in-scope model combines `cone="elliptic"` with
    `noslip_iterations`, so writing it now would be unmeasured code.
    ⚠ THE OBLIGATION IS ON THE CALLER: the wiring must dispatch on
    `CONE_TYPE` and must NOT route an elliptic model here, because doing so
    would silently apply the wrong friction law rather than fail. When the
    wiring lands, that dispatch is the thing to gate.
  * `scale` and `tolerance` are CALLER-SUPPLIED, and both must be MuJoCo's or
    the iteration count diverges. `tolerance` is `m->opt.noslip_tolerance`
    (default 1e-6). `scale` is `1 / (stat.meaninertia * max(1, nv))`, and
    `stat.meaninertia` is the mean of the mass matrix DIAGONAL —
    `engine_setconst.c:1139-1146`:

        meaninertia = (1/nv) * sum_i qM[dof_Madr[i]]

    evaluated once at model-build time. Getting this wrong does not corrupt
    the sweep, it changes WHEN the loop stops, which is a subtler and more
    annoying divergence — hence spelling it out here.

⚠⚠ EVERY SCALAR HERE IS `Scalar[DTYPE]`. IT USED TO BE `Float64` (2026-08-10)

This module originally widened to `Float64` internally — the natural choice,
since the arithmetic below is where MuJoCo's own solver is most sensitive to
rounding. That made the whole pass UNCOMPILABLE ON GPU:

    Function 'air.convert.f.f64.f.f32' has Metal-unsupported instructions
    Function 'mojo_rl_physics3d_solver_nosl...' has Metal-unsupported ...
    LLVM ERROR: Failed to verify LLVM IR for Metal

and `Float64` is off the table on the NVIDIA path too. Since dog is the ONLY
model in the tree that asks for this pass (`dog_xml`, `dog_fetch_xml`), that
made "dog on GPU" and "noslip in Float64" mutually exclusive. The port is the
resolution. The CPU path is UNAFFECTED — it instantiates at
`DTYPE = float64`, so its arithmetic is bit-identical to before.

⚠ WHAT THE GPU PATH GIVES UP, STATED RATHER THAN DISCOVERED LATER. At
float32 three things below get noisier, and all three change WHEN the sweep
stops rather than what it converges to:

  * `improvement` accumulates `0.5*d*d*a_ii + d*res` over every row, then is
    compared against `tolerance` (1e-6) after scaling. A float32 sum over
    hundreds of rows carries ~1e-7 relative noise, which is the same order as
    the threshold — so the GPU may run one more or one fewer iteration than
    the CPU on the same state.
  * `_COST_REJECT` (1e-10) gates a `change` built from differences of
    products. That is catastrophic cancellation at float32, so a block the
    CPU accepts may be rejected on GPU and vice versa.
  * `k1 = a00 + a11 - a01 - a10` is a four-way cancellation feeding a
    division. When it lands under `_MINVAL` the pair is left at `mid`.

None of that can destabilise a contact — the `(mid + y, mid - y)` invariant
below is structural, not numerical, so the NORMAL force is preserved exactly
whatever the arithmetic does. It does mean a CPU-vs-GPU gate on a CONTACTING
dog should expect iteration-count divergence, which is the same regime
`test_quadruped_gpu_vs_cpu` already declines to bound.
"""

from std.math import sqrt
from std.collections import InlineArray
from layout import Layout, LayoutTensor

from ..constraints.scalar_rows import SROW_FRICTION
from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_CONDIM,
)


# ⚠ These stay `Float64` DELIBERATELY: they are `comptime`, so they never
# reach the emitted IR — `Scalar[DTYPE](_MINVAL)` folds at compile time. What
# Metal rejects is a `double` VALUE or CONVERSION in the kernel body, not a
# compile-time constant that happens to be spelled in double precision.
# All three are representable at float32 (min normal ~1.2e-38).
# `mjMINVAL`.
comptime _MINVAL: Float64 = 1e-15
# `extractBlock`'s diagonal floor when `flg_subR` is set.
comptime _DIAG_FLOOR: Float64 = 1e-10
# `costChange`'s "this made things worse" threshold.
comptime _COST_REJECT: Float64 = 1e-10


@always_inline
def _minv_jt[
    DTYPE: DType, NV: Int, ME: Int, V_SIZE: Int, BATCH: Int
](
    env: Int,
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    Je: InlineArray[Scalar[DTYPE], ME * V_SIZE],
    row: Int,
    mut out_v: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """`out_v = M^-1 J_row^T`.

    Recomputed per use rather than cached for every row. The cache would be
    `num_edges * NV` doubles — ~95 kB on dog — which is fine on the CPU path
    and far too much for per-env GPU local memory, and this module is meant to
    serve both. If the CPU path ever needs the speed, hoist it: `J` and `M` do
    not change during the sweep, so the result is loop-invariant.
    """
    for i in range(NV):
        var acc = Scalar[DTYPE](0)
        for k in range(NV):
            acc += rebind[Scalar[DTYPE]](m_inv[env, i * NV + k]) * Je[
                row * V_SIZE + k
            ]
        out_v[i] = acc


@always_inline
def _dot_row[
    DTYPE: DType, NV: Int, ME: Int, V_SIZE: Int
](
    Je: InlineArray[Scalar[DTYPE], ME * V_SIZE],
    row: Int,
    v: InlineArray[Scalar[DTYPE], V_SIZE],
) -> Scalar[DTYPE]:
    """`J_row . v`."""
    var acc = Scalar[DTYPE](0)
    for i in range(NV):
        acc += Je[row * V_SIZE + i] * v[i]
    return acc


@always_inline
def _refresh_jar[
    DTYPE: DType, NV: Int, ME: Int, V_SIZE: Int
](
    num_edges: Int,
    Je: InlineArray[Scalar[DTYPE], ME * V_SIZE],
    bias_e: InlineArray[Scalar[DTYPE], ME],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    mut jar: InlineArray[Scalar[DTYPE], ME],
):
    """`jar[e] = J_e . qacc + bias_e` for every row.

    Recomputed in full after each block rather than updated incrementally.
    Both cost `ME * NV`; the full recompute cannot drift, and this pass runs
    at most `noslip_iterations` times per step.
    """
    for e in range(num_edges):
        jar[e] = _dot_row[DTYPE, NV, ME, V_SIZE](Je, e, qacc) + bias_e[e]


@always_inline
def _cost_change[
    DTYPE: DType
](
    a00: Scalar[DTYPE], a01: Scalar[DTYPE],
    a10: Scalar[DTYPE], a11: Scalar[DTYPE],
    f0: Scalar[DTYPE], f1: Scalar[DTYPE],
    old0: Scalar[DTYPE], old1: Scalar[DTYPE],
    r0: Scalar[DTYPE], r1: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """`costChange` for a 2x2 block: `0.5 d^T A d + d . res`.

    The caller must REJECT the update when this is positive — MuJoCo restores
    the old forces and counts zero improvement. Returned unclamped so the
    caller can do exactly that.

    ⚠ This is the cancellation-sensitive expression of the module: `f0 - old0`
    and `f1 - old1` are small differences of comparable numbers, then squared
    against `A`. See the float32 note in the module docstring.
    """
    var d0 = f0 - old0
    var d1 = f1 - old1
    var quad = (
        d0 * (a00 * d0 + a01 * d1) + d1 * (a10 * d0 + a11 * d1)
    ) * Scalar[DTYPE](0.5)
    return quad + d0 * r0 + d1 * r1


def noslip_pyramidal[
    DTYPE: DType,
    NV: Int,
    ME: Int,
    V_SIZE: Int,
    MC: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
    BATCH: Int,
    MAX_ITER: Int,
](
    env: Int,
    nc: Int,
    num_edges: Int,
    # ⚠ Keyed on MAX_CONTACTS, not MC. `MC = _max_one[MAX_CONTACTS]()` and the
    # two agree everywhere except MAX_CONTACTS == 0, but a LayoutTensor
    # parameter is part of the TYPE — passing the caller's tensor with the
    # wrong one is a compile error, not a coercion.
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    Je: InlineArray[Scalar[DTYPE], ME * V_SIZE],
    bias_e: InlineArray[Scalar[DTYPE], ME],
    kind_e: InlineArray[Int, ME],
    R_e: InlineArray[Scalar[DTYPE], ME],
    floss_e: InlineArray[Scalar[DTYPE], ME],
    qacc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    # ⚠ `Scalar[DTYPE]`, not `Float64` — the caller must build these in DTYPE
    # too, or the conversion reappears at the CALL SITE and Metal rejects it
    # there instead. `newton_solve` computes `1/(meaninertia*max(1,nv))` from
    # a DTYPE tensor read, with `max(1,nv)` folded to a comptime constant.
    scale: Scalar[DTYPE],
    tolerance: Scalar[DTYPE],
    mut qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    mut jar: InlineArray[Scalar[DTYPE], ME],
    mut force: InlineArray[Scalar[DTYPE], ME],
    mut qfrc: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """One `mj_solNoSlip` call: up to `MAX_ITER` friction-only sweeps.

    On entry `qacc`, `jar` and `force` are the primal solver's converged
    output. On exit `force` has been redistributed within each friction pair,
    and `qacc` / `qfrc` have been recomputed from the new forces exactly as
    `dualFinish` does — from the FINAL forces, not accumulated incrementally,
    so the in-sweep `qacc` updates cannot leave a residue.
    """
    comptime NE_PYR = 2 * (MAX_CONDIM - 1)

    var mj_a = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))
    var mj_b = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))

    comptime ZERO = Scalar[DTYPE](0)
    comptime HALF = Scalar[DTYPE](0.5)
    comptime TWO = Scalar[DTYPE](2.0)
    comptime MINVAL = Scalar[DTYPE](_MINVAL)
    comptime DIAG_FLOOR = Scalar[DTYPE](_DIAG_FLOOR)
    comptime COST_REJECT = Scalar[DTYPE](_COST_REJECT)

    for it in range(MAX_ITER):
        var improvement = ZERO

        # `iter == 0` correction: MuJoCo folds in the R-weighted force energy
        # once, so the first iteration's improvement is comparable with the
        # primal solver's own.
        if it == 0:
            for i in range(num_edges):
                var f = force[i]
                improvement += HALF * f * f * R_e[i]

        # ── sweep 1: dry-friction dof rows, box-clamped to +-frictionloss ──
        for i in range(num_edges):
            if kind_e[i] != SROW_FRICTION:
                continue
            _minv_jt[DTYPE, NV, ME, V_SIZE, BATCH](env, m_inv, Je, i, mj_a)
            var a_ii = _dot_row[DTYPE, NV, ME, V_SIZE](Je, i, mj_a)
            var arinv = Scalar[DTYPE](1.0) / (
                a_ii if a_ii > MINVAL else MINVAL
            )

            var res = jar[i]
            var old = force[i]
            var f = old - res * arinv
            var lim = floss_e[i]
            if f < -lim:
                f = -lim
            elif f > lim:
                f = lim

            var d = f - old
            if d != ZERO:
                force[i] = f
                for k in range(NV):
                    qacc[k] += d * mj_a[k]
                _refresh_jar[DTYPE, NV, ME, V_SIZE](
                    num_edges, Je, bias_e, qacc, jar
                )
            # `0.5*d^2/ARinv` — and `1/ARinv` is `a_ii`, so no division here.
            improvement -= HALF * d * d * a_ii + d * res

        # ── sweep 2: contact friction, one opposing pyramid pair at a time ──
        for c in range(nc):
            var dim = Int(contacts[env, c * CONTACT_SIZE + CONTACT_IDX_CONDIM])
            if dim < 3:
                continue  # frictionless contact: no friction rows to sweep
            var base = c * NE_PYR
            for k in range(dim - 1):
                var j0 = base + 2 * k
                var j1 = j0 + 1
                if j1 >= num_edges:
                    break

                _minv_jt[DTYPE, NV, ME, V_SIZE, BATCH](env, m_inv, Je, j0, mj_a)
                _minv_jt[DTYPE, NV, ME, V_SIZE, BATCH](env, m_inv, Je, j1, mj_b)

                # `Ac` = A submatrix, diagonal clamped (flg_subR semantics:
                # R is NOT part of it).
                var a00 = _dot_row[DTYPE, NV, ME, V_SIZE](Je, j0, mj_a)
                var a01 = _dot_row[DTYPE, NV, ME, V_SIZE](Je, j0, mj_b)
                var a10 = _dot_row[DTYPE, NV, ME, V_SIZE](Je, j1, mj_a)
                var a11 = _dot_row[DTYPE, NV, ME, V_SIZE](Je, j1, mj_b)
                if a00 < DIAG_FLOOR:
                    a00 = DIAG_FLOOR
                if a11 < DIAG_FLOOR:
                    a11 = DIAG_FLOOR

                var r0 = jar[j0]
                var r1 = jar[j1]
                var old0 = force[j0]
                var old1 = force[j1]

                # `bc = res - Ac * oldforce`
                var b0 = r0 - (a00 * old0 + a01 * old1)
                var b1 = r1 - (a10 * old0 + a11 * old1)

                # The pair is written as (mid + y, mid - y), so `mid` — the
                # NORMAL contribution — is invariant no matter which branch
                # fires. That is the whole "normal forces frozen" property,
                # and it is STRUCTURAL: it survives the float32 port intact,
                # because every branch below is symmetric about `mid` by
                # construction rather than by cancellation.
                var mid = HALF * (old0 + old1)
                var k1 = a00 + a11 - a01 - a10
                var k0 = mid * (a00 - a11) + b0 - b1

                var f0 = mid
                var f1 = mid
                if k1 >= MINVAL:
                    var y = -k0 / k1
                    if y < -mid:
                        f0 = ZERO
                        f1 = TWO * mid
                    elif y > mid:
                        f0 = TWO * mid
                        f1 = ZERO
                    else:
                        f0 = mid + y
                        f1 = mid - y

                var change = _cost_change[DTYPE](
                    a00, a01, a10, a11, f0, f1, old0, old1, r0, r1
                )
                if change > COST_REJECT:
                    # Made it worse — MuJoCo restores and counts nothing.
                    continue

                var d0 = f0 - old0
                var d1 = f1 - old1
                if d0 != ZERO or d1 != ZERO:
                    force[j0] = f0
                    force[j1] = f1
                    for q in range(NV):
                        qacc[q] += d0 * mj_a[q] + d1 * mj_b[q]
                    _refresh_jar[DTYPE, NV, ME, V_SIZE](
                        num_edges, Je, bias_e, qacc, jar
                    )
                improvement -= change

        improvement *= scale
        if improvement < tolerance:
            break

    # ── dualFinish ─────────────────────────────────────────────────────────
    # `qfrc_constraint = J^T f`, then `qacc = M^-1 qfrc + qacc_smooth`.
    #
    # ⚠ RECOMPUTED FROM THE FINAL FORCES, not carried over from the in-sweep
    # `qacc +=` updates. Those exist only to keep `jar` current between blocks;
    # MuJoCo recomputes here and so does this, so any drift they accumulated is
    # discarded rather than integrated.
    for i in range(NV):
        qfrc[i] = Scalar[DTYPE](0)
    for e in range(num_edges):
        var f = force[e]
        if f == Scalar[DTYPE](0):
            continue
        for i in range(NV):
            qfrc[i] += Je[e * V_SIZE + i] * f
    for i in range(NV):
        var acc = Scalar[DTYPE](0)
        for k in range(NV):
            acc += rebind[Scalar[DTYPE]](m_inv[env, i * NV + k]) * qfrc[k]
        qacc[i] = acc + qacc_smooth[i]
