"""`qpos_from_site_pose` — dm_control's damped-least-squares site IK.

Step 3 of the dm_control Phase 7 reset path. Transcribed from
`dm_control/utils/inverse_kinematics.py`, which is the reference here — this
is dm_control's own algorithm, not a MuJoCo engine routine, so the spec is
that Python file rather than anything in `references/mujoco-*`.

The caller of record is `entities/manipulators/base.py::set_site_to_xpos`, and
several of its arguments are NOT the defaults of the function it calls. Taking
the defaults from `qpos_from_site_pose`'s own signature gives the wrong solver:

  * `rot_weight = 2`, not the signature default of 1.
  * `joint_names = arm_joint_names` — ⚠ THE SOLVE IS RESTRICTED TO THE ARM'S
    DOFs. This is the single most load-bearing argument in the whole routine;
    see the note on rank below.
  * `inplace = True`.

⚠ WHY THE DOF RESTRICTION DECIDES THE NUMERICS. `nullspace_method` builds
`H = J^T J` and, when unregularised, solves it with
`np.linalg.lstsq(..., rcond=-1)` — a MINIMUM-NORM least-squares solve, which
in general needs an SVD. Over ALL of Jaco's 9 DOFs, `H` is 9x9 of rank at most
6, so that branch would be genuinely rank-deficient and the min-norm solution
would matter. Restricted to the arm's 6 DOFs, `H` is 6x6 and generically FULL
RANK: measured over 300 random poses of the real `reach_site_features` model,
the rank was 6 in 300 of 300, worst eigenvalue ratio 7.6e-09 — poorly
conditioned in places, but nowhere near the ~1e-15 cutoff a rank test uses.

So `lstsq` degenerates to an ordinary solve here, and this port does an
ordinary SPD solve. That equivalence was MEASURED on the real model, not
argued: running the full loop both ways over 60 randomised targets, the
success flag agreed 60/60, the step count differed by at most 1, and the worst
`|d qpos|` was 1.2e-13.

⚠ AND IT IS NOT REPRODUCIBLE ANY OTHER WAY, so do not "improve" this into a
pseudoinverse to chase bit-parity on the rank-deficient case. LAPACK's own
cutoff keeps numerical-noise directions: on a rank-6 9x9 `H`, `lstsq` reported
rank 7 and divided by a singular value of 2.7e-15, giving a solution whose
norm exceeded the true minimum-norm one. That happened in 41 of 300 trials,
and neither `eps` nor `eps/2` as a cutoff reproduces which side of the line
LAPACK lands on. If a future model does make `H` rank-deficient, the honest
outcome is the explicit failure below, not a silently different answer.

⚠ `rot_weight` SCALES THE CONVERGENCE TEST ONLY, NOT THE STEP. It enters
`err_norm`, which drives the tolerance check, the regularisation switch and
the progress guard — but the vector handed to the solver is the UNWEIGHTED
`[err_pos, err_rot]`. Weighting the rotational rows of the residual (the
obvious "fix") changes the search direction and is a different algorithm.

⚠ A FAILED SOLVE STILL RETURNS A PLAUSIBLE qpos. The progress guard
(`err_norm / update_norm > progress_thresh`) breaks out with `success` still
False, leaving `qpos` wherever the last accepted step put it. Callers MUST
branch on `success`; `set_site_to_xpos` does, retrying with a randomised arm
pose up to `max_ik_attempts` times.

WHAT IS DELIBERATELY NOT RUN. The reference calls `mj_fwdPosition`, which also
does broadphase and collision. The IK loop never reads a contact, so this runs
only the position pipeline it does read — forward kinematics, `subtree_com`,
`cdof`. Collisions matter to the CALLER (the rejection sampler in
`ToolCenterPointInitializer`), after IK returns.
"""

from std.math import sqrt
from std.collections import InlineArray
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl

from ..fields import Data, Model, DynamicsScratch, Dims
from ..kinematics.forward_kinematics import forward_kinematics
from ..kinematics.integrate_pos import integrate_pos
from ..kinematics.quat_math import quat_mul, quat_conjugate, quat2vel
from ..kinematics.site_frame import site_world_quat_list
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
)
from .subtree_com import compute_subtree_com
from .cdof import compute_cdof
from .jac_point import jac_site


@fieldwise_init
struct IKResult(Copyable, Movable):
    """`inverse_kinematics.IKResult`, minus the qpos (which is written into
    `Data` in place, since the caller is always `inplace=True`)."""

    var err_norm: Float64
    var steps: Int
    var success: Bool
    var rank_deficient: Bool
    """⚠ Set when the SPD solve hit a non-positive pivot — i.e. the case the
    reference would have handled with a min-norm `lstsq` and this port cannot
    reproduce. Never observed on `reach_site_features`; surfaced rather than
    swallowed so that a model which does trigger it is not silently given a
    different algorithm."""


@always_inline
def _solve_spd[
    DTYPE: DType, N: Int
](
    a: InlineArray[Scalar[DTYPE], N * N],
    b: InlineArray[Scalar[DTYPE], N],
    mut x: InlineArray[Scalar[DTYPE], N],
) -> Bool:
    """Cholesky solve of a symmetric positive-definite `N x N` system.

    Returns False, without writing a solution, if a pivot is not positive —
    which for `J^T J (+ lambda I)` means rank deficiency. Reporting that is
    the point: see the module docstring on why this port must not quietly
    substitute a pseudoinverse.
    """
    var l = InlineArray[Scalar[DTYPE], N * N](fill=Scalar[DTYPE](0))
    for i in range(N):
        for j in range(i + 1):
            var s = a[i * N + j]
            for k in range(j):
                s -= l[i * N + k] * l[j * N + k]
            if i == j:
                if s <= Scalar[DTYPE](0):
                    return False
                l[i * N + j] = sqrt(s)
            else:
                l[i * N + j] = s / l[j * N + j]

    # forward substitution: L y = b
    var y = InlineArray[Scalar[DTYPE], N](fill=Scalar[DTYPE](0))
    for i in range(N):
        var s = b[i]
        for k in range(i):
            s -= l[i * N + k] * y[k]
        y[i] = s / l[i * N + i]

    # back substitution: L^T x = y
    for ii in range(N):
        var i = N - 1 - ii
        var s = y[i]
        for k in range(i + 1, N):
            s -= l[k * N + i] * x[k]
        x[i] = s / l[i * N + i]
    return True


def qpos_from_site_pose[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    # ⚠ NPAIR WAS A HARDCODED `0` HERE, AND THAT IS NOT THE SAME AS "no pairs".
    # A model def built by `parse_xml` carries NPAIR as the SYMBOLIC
    # `parse_xml(XML).NPAIR`, which the compiler will not unify with the
    # literal `Int(0)` even when the model declares no `<contact><pair>` at
    # all. So the literal never restricted the FEATURE, it restricted which
    # CALLERS could compile — and it locked out every env going through
    # `Phyics3dEnv`, whose `Model` type comes from the model def. The gate
    # that exercised this code passed `0` explicitly and could not see it.
    NPAIR: Int,
    MAXC: Int,
    NDOF: Int,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    site: Int,
    target_pos: InlineArray[Scalar[DTYPE], 3],
    target_quat: InlineArray[Scalar[DTYPE], 4],
    dof_idx: InlineArray[Int, NDOF],
    use_pos: Bool = True,
    use_quat: Bool = True,
    tol: Float64 = 1e-14,
    rot_weight: Float64 = 2.0,
    regularization_threshold: Float64 = 0.1,
    regularization_strength: Float64 = 3e-2,
    max_update_norm: Float64 = 2.0,
    progress_thresh: Float64 = 20.0,
    max_steps: Int = 100,
) raises -> IKResult:
    """Drive `site` to `target_pos` / `target_quat`, writing `d.qpos` in place.

    `target_quat` is `(x, y, z, w)`, this tree's convention — NOT MuJoCo's
    `(w, x, y, z)`. dm_control's `DOWN_QUATERNION` is `(0, 0.7071, 0.7071, 0)`
    in MuJoCo order, so it becomes `(0.7071, 0.7071, 0, 0)` here.

    `dof_idx` lists the velocity-space DOFs the solver may move; everything
    else is held. See the module docstring — this is not an optimisation, it
    is what keeps the normal matrix full rank.
    """
    comptime L_QPOS = Layout.row_major(1, NQ)
    comptime L_NV = Layout.row_major(1, NV)
    comptime L_NB3 = Layout.row_major(1, NBODY * 3)
    comptime L_JNT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_BOD = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_MET = Layout.row_major(MODEL_META_SIZE)
    comptime L_CDOF = Layout.row_major(1, NV * 6)
    comptime L_SITE = Layout.row_major(NSITE, MODEL_SITE_SIZE)
    comptime L_SX = Layout.row_major(1, NSITE * 3)

    var joints_v = mf.joints.lt["cpu", L_JNT]()
    var bodies_v = mf.bodies.lt["cpu", L_BOD]()
    var sites_v = mf.sites.lt["cpu", L_SITE]()
    var mmeta_v = mf.meta.lt["cpu", L_MET]()
    var qpos_v = d.qpos.lt["cpu", L_QPOS]()

    var scratch = DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], 1]()
    # ⚠ Its own buffer, NOT a borrowed `Data` field. `d.qacc`/`d.qvel` would
    # have done the job and silently clobbered a physics output that the
    # caller has every reason to expect IK left alone.
    var update_t = TensorImpl[DTYPE].alloc(NV)
    var update_nv = update_t.lt["cpu", L_NV]()

    var site_body = Int(rebind[Scalar[DTYPE]](sites_v[site, SITE_IDX_BODY]))

    var err = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    var jp = InlineArray[Scalar[DTYPE], 3 * NV](fill=Scalar[DTYPE](0))
    var jr = InlineArray[Scalar[DTYPE], 3 * NV](fill=Scalar[DTYPE](0))
    var hess = InlineArray[Scalar[DTYPE], NDOF * NDOF](fill=Scalar[DTYPE](0))
    var grad = InlineArray[Scalar[DTYPE], NDOF](fill=Scalar[DTYPE](0))
    var upd = InlineArray[Scalar[DTYPE], NDOF](fill=Scalar[DTYPE](0))

    var err_norm = 0.0
    var steps = 0
    var success = False
    var rank_deficient = False

    # "Ensure that the Cartesian position of the site is up to date."
    forward_kinematics["cpu"](d, mf)

    for s in range(max_steps):
        steps = s
        err_norm = 0.0

        if use_pos:
            for k in range(3):
                err[k] = target_pos[k] - rebind[Scalar[DTYPE]](
                    d.site_xpos.data[site * 3 + k]
                )
            err_norm += sqrt(
                Float64(err[0] * err[0] + err[1] * err[1] + err[2] * err[2])
            )

        if use_quat:
            # ⚠ The reference reads `site_xmat` and runs `mju_mat2Quat`. We
            # compose the site's world quaternion directly. That is safe
            # because the two can differ only by an overall sign, and
            # `quat2vel` is sign-invariant — measured, and asserted by
            # `tests/physics3d/test_ik_primitives_vs_mujoco.mojo`.
            var sq = site_world_quat_list[DTYPE](
                mf.sites.data, d.xquat.data, site_body, site
            )
            var nq_ = quat_conjugate[DTYPE](
                Scalar[DTYPE](sq[0]),
                Scalar[DTYPE](sq[1]),
                Scalar[DTYPE](sq[2]),
                Scalar[DTYPE](sq[3]),
            )
            var eq = quat_mul[DTYPE](
                target_quat[0],
                target_quat[1],
                target_quat[2],
                target_quat[3],
                nq_[0],
                nq_[1],
                nq_[2],
                nq_[3],
            )
            var ev = quat2vel[DTYPE](
                eq[0], eq[1], eq[2], eq[3], Scalar[DTYPE](1)
            )
            err[3] = ev[0]
            err[4] = ev[1]
            err[5] = ev[2]
            # ⚠ rot_weight scales the NORM only; `err` itself stays unweighted.
            err_norm += rot_weight * sqrt(
                Float64(err[3] * err[3] + err[4] * err[4] + err[5] * err[5])
            )

        if err_norm < tol:
            success = True
            break

        compute_subtree_com["cpu"](d, mf)
        compute_cdof["cpu"](d, mf, scratch)
        var subtree_v = d.subtree_com.lt["cpu", L_NB3]()
        var cdof_v = scratch.cdof.lt["cpu", L_CDOF]()
        var sxpos_v = d.site_xpos.lt["cpu", L_SX]()
        jac_site[DTYPE, NV, NBODY, NJOINT, NSITE, 1](
            0, subtree_v, joints_v, bodies_v, mmeta_v, cdof_v,
            sites_v, sxpos_v, site, jp, jr,
        )

        # `jac_joints` — the 6 x NDOF slice. Rows 0-2 translational, 3-5
        # rotational, matching how the reference stacks `err`.
        var reg = 0.0
        if err_norm > regularization_threshold:
            reg = regularization_strength

        for a in range(NDOF):
            var ca = dof_idx[a]
            var g = Scalar[DTYPE](0)
            for r in range(3):
                if use_pos:
                    g += jp[r * NV + ca] * err[r]
                if use_quat:
                    g += jr[r * NV + ca] * err[3 + r]
            grad[a] = g
            for b in range(NDOF):
                var cb = dof_idx[b]
                var h = Scalar[DTYPE](0)
                for r in range(3):
                    if use_pos:
                        h += jp[r * NV + ca] * jp[r * NV + cb]
                    if use_quat:
                        h += jr[r * NV + ca] * jr[r * NV + cb]
                hess[a * NDOF + b] = h
            hess[a * NDOF + a] += Scalar[DTYPE](reg)

        if not _solve_spd[DTYPE, NDOF](hess, grad, upd):
            rank_deficient = True
            break

        var un2 = Scalar[DTYPE](0)
        for a in range(NDOF):
            un2 += upd[a] * upd[a]
        var update_norm = sqrt(Float64(un2))

        if err_norm / update_norm > progress_thresh:
            break

        if update_norm > max_update_norm:
            var sc = Scalar[DTYPE](max_update_norm / update_norm)
            for a in range(NDOF):
                upd[a] *= sc

        for i in range(NV):
            update_nv[0, i] = Scalar[DTYPE](0)
        for a in range(NDOF):
            update_nv[0, dof_idx[a]] = upd[a]

        integrate_pos[DTYPE, NQ, NV, NJOINT, 1](
            0, qpos_v, update_nv, joints_v, Scalar[DTYPE](1)
        )
        forward_kinematics["cpu"](d, mf)

    return IKResult(err_norm, steps, success, rank_deficient)


@fieldwise_init
struct SetSiteResult(Copyable, Movable):
    """Outcome of `set_site_to_xpos`."""

    var success: Bool
    var attempts: Int
    """How many IK attempts were consumed, 1-based. ⚠ Reported so a gate can
    prove the RETRY path ran: if every trial happens to succeed first time,
    the re-randomisation and the injected pose sequence are never exercised
    and a green test says nothing about them."""


def canonicalize_arm_joints[
    DTYPE: DType, NQ: Int, NDOF: Int
](
    mut qpos: List[Scalar[DTYPE]],
    qpos_adr: InlineArray[Int, NDOF],
    lower: InlineArray[Float64, NDOF],
    upper: InlineArray[Float64, NDOF],
) -> Bool:
    """`set_site_to_xpos`'s "canonicalise the angle to [0, 2*pi]" block.

    Returns False if any joint could not be brought inside its bounds.

    ⚠ TWO THINGS HERE ARE EASY TO TRANSCRIBE WRONG.

    1. The reference's `break` on failure exits the INNER `while` only — the
       `for` over joints carries on and keeps canonicalising the REST. So a
       failed joint does not stop the others from being rewritten, and this
       must return False while still having mutated everything after it.
    2. It runs for UNLIMITED hinges too, which `_get_joint_pos_sampling_bounds`
       gives the bounds `[0, 2*pi]`. Those joints are wrapped into that window
       even though nothing physical constrains them.

    Wrapping a hinge by a multiple of 2*pi does not move the arm, so the site
    pose IK just solved for is preserved exactly.
    """
    var ok = True
    comptime TWO_PI = 6.283185307179586
    for a in range(NDOF):
        var p = qpos_adr[a]
        var v = Float64(qpos[p])
        while v >= upper[a]:
            v -= TWO_PI
        while v < lower[a]:
            v += TWO_PI
            if v > upper[a]:
                ok = False
                break
        qpos[p] = Scalar[DTYPE](v)
    return ok


def set_site_to_xpos[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    # ⚠ NPAIR WAS A HARDCODED `0` HERE, AND THAT IS NOT THE SAME AS "no pairs".
    # A model def built by `parse_xml` carries NPAIR as the SYMBOLIC
    # `parse_xml(XML).NPAIR`, which the compiler will not unify with the
    # literal `Int(0)` even when the model declares no `<contact><pair>` at
    # all. So the literal never restricted the FEATURE, it restricted which
    # CALLERS could compile — and it locked out every env going through
    # `Phyics3dEnv`, whose `Model` type comes from the model def. The gate
    # that exercised this code passed `0` explicitly and could not see it.
    NPAIR: Int,
    MAXC: Int,
    NDOF: Int,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAXC, NSITE, 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    site: Int,
    target_pos: InlineArray[Scalar[DTYPE], 3],
    target_quat: InlineArray[Scalar[DTYPE], 4],
    dof_idx: InlineArray[Int, NDOF],
    qpos_adr: InlineArray[Int, NDOF],
    lower: InlineArray[Float64, NDOF],
    upper: InlineArray[Float64, NDOF],
    retry_poses: List[Scalar[DTYPE]],
    max_ik_attempts: Int = 10,
    retry_offset: Int = 0,
) raises -> SetSiteResult:
    """`entities/manipulators/base.py::set_site_to_xpos`.

    IK, then canonicalise; on failure re-randomise the arm and try again, up
    to `max_ik_attempts`.

    ⚠ THE RETRY POSES ARE INJECTED, NOT DRAWN. The reference calls
    `randomize_arm_joints`, i.e. `random_state.uniform(lower, upper)` on a
    numpy `RandomState`. Reproducing that bit stream in Mojo is neither
    possible nor desirable, so the caller supplies the sequence:
    `retry_poses` is `(max_ik_attempts - 1) * NDOF` values, consumed one
    NDOF-block per failed attempt, in order. A gate can therefore drive both
    implementations down IDENTICAL trajectories by precomputing the same
    draws with the same seed — which is exactly what
    `test_set_site_to_xpos_vs_dm_control` does. Production callers can pass
    whatever sampler they like.

    `retry_offset` indexes into that same flat list, so a CALLER THAT LOOPS —
    `tool_center_point_initializer` draws up to `max_rejection_samples` target
    poses and each gets its own attempt budget — can hand over one contiguous
    draw sequence and advance a cursor, rather than slicing a fresh `List` per
    sample. The reference's draws come off one `RandomState` in exactly that
    order, so a single flat list IS the faithful shape.

    ⚠ `rot_weight = 2` is not this function's choice to make; it is what the
    reference passes, and `qpos_from_site_pose` defaults to it here.
    """
    var success = False
    var attempts = 0
    for attempt in range(max_ik_attempts):
        attempts = attempt + 1
        var res = qpos_from_site_pose[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAXC, NDOF,
        ](d, mf, site, target_pos, target_quat, dof_idx)
        success = res.success

        if success:
            success = canonicalize_arm_joints[DTYPE, NQ, NDOF](
                d.qpos.data, qpos_adr, lower, upper
            )

        # "If succeeded or only one attempt, break and do not randomize."
        if success or max_ik_attempts <= 1:
            break

        var base = retry_offset + attempt * NDOF
        if base + NDOF > len(retry_poses):
            # Out of injected poses — stop rather than silently repeat the
            # same attempt, which would look like convergence failure.
            break
        for a in range(NDOF):
            d.qpos.data[qpos_adr[a]] = retry_poses[base + a]

    return SetSiteResult(success, attempts)
