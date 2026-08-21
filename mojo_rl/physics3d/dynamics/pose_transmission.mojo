"""Actuator forces whose transmission depends on the POSE, not just `qpos`.

`apply_actions_fields` walks a transmission stored as `(qadr, dadr, coef)`
triples: an actuator's length is `gear * sum_k coef_k * qpos[qadr_k]` and its
moment is `gear * coef_k` on dof `dadr_k`. That covers a `joint=` transmission
(one triple, coef 1) and a `tendon=` transmission naming a **FIXED** tendon
(one triple per (joint, coef) pair). Both are functions of `qpos` alone, which
is why they can be evaluated before forward kinematics has run.

A **SPATIAL** tendon is not. Its length is the length of a polyline through
sites — possibly wrapping geoms — and its moment arm is
`d(length)/dq`, a dense `nv` vector that changes with the pose. MuJoCo gets
both from `mj_tendon`, which runs inside `mj_fwdPosition`, and reads them back
in `mj_fwdActuation` as `ten_length[id]` and `ten_moment[id]`.

WHAT THIS FILE IS FOR. `_fill_actuator_transmission` resolves a `tendon=`
actuator by copying the tendon's `(joint, coef)` list — the FIXED
representation. A spatial tendon has no such list, so the actuator came out
with `trn_n = 0`: it kept its slot in `nact`, consumed its control, and
applied **zero force**. `tetheria_aero_hand_open` drives six of its seven
actuators that way (`<position tendon="if_tendon0" kp="10000"/>` and five
more), so the hand had no tendons pulling it at all — it was the worst scene
in the Menagerie sweep at 2.590e-01.

The same is true of a spatial tendon SPRING (`<spatial stiffness=...>`), whose
deadband force needs the same length and the same moment arm.
`tetheria`'s eight springs are at their rest length at `qpos0` — which is why
the step-1 sweep saw only the actuators — and stretch as soon as it moves.

⚠⚠ THE CALLER'S OBLIGATION IS FRESH KINEMATICS, AND THAT IS WHY THIS IS A
SEPARATE FUNCTION. `apply_actions_fields` runs BEFORE the integrator's
`forward_kinematics`, so `d.xpos` / `d.xquat` / `d.subtree_com` describe the
pose one substep back and `sc.cdof` has not been rebuilt at all. For a joint
transmission that does not matter — it reads `qpos` directly. For this one it
is the difference between a moment arm and a stale moment arm, so this
function refreshes FK, the subtree CoM and `cdof` itself before reading them.
Running them again costs three passes on the models that need it and nothing
on the models that do not (the scan below exits before any of it).

⚠ IT ACCUMULATES, AND IT MUST RUN AFTER `apply_actions_fields`. That function
ZEROES `d.qfrc` and `d.dof_actdamp` before its own loop; this one adds to
both.

⚠ ONE ORDERING DIFFERENCE, DELIBERATELY LEFT. `apply_actions_fields` applies
MuJoCo's per-joint `jnt_actfrcrange` clamp at the end of its actuator loop,
i.e. BEFORE these forces are added. MuJoCo clamps the total once. The two
disagree only for a joint that is BOTH `actfrclimited` AND driven by a
pose transmission, where our answer is `clamp(a) + b` against MuJoCo's
`clamp(a + b)`. No model in this tree is: `jnt_actfrclimited` is 0 on all
three models with a pose transmission (tetheria, skydio_x2,
bitcraze_crazyflie_2). Re-clamping here would be wrong in the other
direction — it would clamp the tendon SPRING, which is `qfrc_passive` and
which MuJoCo never clamps.

⚠⚠ ONE PER CONTROL SUBSTEP, WHICH IS EXACT UNDER EULER AND NOT UNDER RK4.
The caller computes these forces once and then integrates; `mj_Euler`
evaluates the derivative once too, so the two agree exactly. `mj_RungeKutta`
evaluates it FOUR TIMES, at four different poses, and recomputes the
transmission at each — while ours stays frozen at the stage-0 moment. The
error is the rotation of the site (or the stretch of the tendon) across one
step, so it is small but real:

    bitcraze_crazyflie_2, one step, worst |d(qpos)| vs MuJoCo
        as shipped (`integrator="RK4"`)            9.200e-06
        the same model rewritten to Euler          5.294e-23

That is the whole of that scene's residual — ablating `density`/`viscosity`
changes nothing, and its `qfrc_actuator` matches MuJoCo to 0.0 exactly at
stage 0. A pose-INDEPENDENT transmission does not care: a `<motor joint=>`
has the same force at all four stages, which is why RK4 models have been
exact until a transmission started depending on the pose. Fixing it means
evaluating actuation inside the RK4 stage loop, i.e. giving the integrator
`sf` and the controls — the refactor this file was written to avoid.

⚠ CPU ONLY, AND THE GPU PATH IS NOT SILENTLY WRONG. `actions` is a host
`List`, so there is nothing to launch; `apply_actions_kernel_gpu` keeps the
`(qadr, dadr, coef)` walk and therefore keeps giving a spatial-tendon
actuator zero force. The parser prints a count at model build for exactly
this reason. No batched env in this tree drives a spatial tendon.

`site=` and `body=` transmissions (`mjTRN_SITE`, `mjTRN_BODY`) belong here
too and are NOT implemented yet — both Menagerie quadrotors drive every
rotor through `<motor site="thrust1" gear="0 0 1 0 0 -.0201"/>`. The
kinematics this file refreshes are exactly what `mj_jacSite` needs, so that
is an addition to the loop below rather than another ordering problem.
"""

from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext

from ..fields import (
    Data, Model, DimsLike, DynamicsScratch, SpecFields,
    Scratch, cap, DYN1, DYN2, rl1, rl2,
)
from ..kinematics.forward_kinematics import forward_kinematics
from ..dynamics.subtree_com import compute_subtree_com
from ..dynamics.cdof import compute_cdof
from ..dynamics.tendon import spatial_tendon_length_jac
from ..dynamics.jac_point import jac_point
from ..kinematics.quat_math import gpu_quat_rotate
from ..parser.flat_model import ACT_KIND_POSITION, ACT_KIND_VELOCITY
from ..gpu.constants import (
    ACTTEN_IDX_SPRING_HI,
    ACTTEN_IDX_SPRING_LO,
    ACTTEN_IDX_STIFFNESS,
    ACT_IDX_ACT_ADR,
    ACT_IDX_CTRL_LIMITED,
    ACT_IDX_CTRL_MAX,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_DYN_TAU,
    ACT_IDX_FORCE_LIMITED,
    ACT_IDX_FORCE_MAX,
    ACT_IDX_FORCE_MIN,
    ACT_IDX_GEAR,
    ACT_IDX_KIND,
    ACT_IDX_KP,
    ACT_IDX_KV,
    ACT_IDX_TENDON_ID,
    ACT_IDX_SITE_ID,
    ACT_IDX_GEAR_1,
    ACT_IDX_GEAR_2,
    ACT_IDX_GEAR_3,
    ACT_IDX_GEAR_4,
    ACT_IDX_GEAR_5,
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
    ACT_IDX_TRN_N,
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    MODEL_BODY_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_SITE_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_META_IDX_NTENDON,
    TENDON_IDX_KIND,
    TENDON_KIND_SPATIAL,
)


@always_inline
def _is_spatial[
    DTYPE: DType, D2: DimsLike
](m: Model[DTYPE, D2], t: Int) -> Bool:
    return (
        Int(Float64(m.tendons.data[t * MODEL_TENDON_SIZE + TENDON_IDX_KIND]))
        == TENDON_KIND_SPATIAL
    )


def model_has_pose_transmission[
    DTYPE: DType, D: DimsLike, D2: DimsLike
](sf: SpecFields[DTYPE, D], m: Model[DTYPE, D2]) -> Bool:
    """True when some actuator or tendon spring needs the pose to be read.

    ⚠ A SCAN, NOT A STORED FLAG. It walks `nact + ntendon` records, which is
    tens of entries on the largest model here, and it is what lets the
    refresh below be skipped entirely on the 82 Menagerie scenes that have
    no spatial tendon. A flag in `Model.meta` would have to be written by
    two parsers and kept in step with them; this cannot go stale.
    """
    var n_act = sf.dims.get_nact()
    var n_ten = m.dims.get_ntendon()
    var nten_live = Int(Float64(m.meta.data[MODEL_META_IDX_NTENDON]))
    if nten_live < n_ten:
        n_ten = nten_live
    for i in range(n_act):
        var o = i * MODEL_ACTUATOR_SIZE
        if Int(sf.actuators.data[o + ACT_IDX_TRN_N]) != 0:
            continue
        if Int(sf.actuators.data[o + ACT_IDX_SITE_ID]) >= 0:
            return True
        var tid = Int(sf.actuators.data[o + ACT_IDX_TENDON_ID])
        if tid >= 0 and tid < n_ten and _is_spatial(m, tid):
            return True
    for t in range(n_ten):
        if not _is_spatial(m, t):
            continue
        var to = t * MODEL_ACT_TENDON_SIZE
        if sf.act_tendons.data[to + ACTTEN_IDX_STIFFNESS] != 0:
            return True
    return False


def apply_pose_transmission[
    DTYPE: DType, D: DimsLike, D2: DimsLike, BATCH: Int = 1
](
    sf: SpecFields[DTYPE, D],
    mut m: Model[DTYPE, D2],
    mut d: Data[DTYPE, D2, BATCH],
    mut sc: DynamicsScratch[DTYPE, D2, BATCH],
    actions: List[Float64],
    mut act: List[Scalar[DTYPE]],
    timestep: Float64,
) raises:
    """Spatial-tendon actuator and spring forces, added to `d.qfrc`.

    Call AFTER `apply_actions_fields` and BEFORE the integrator's `step`.
    Returns immediately on a model with nothing to do.
    """
    if not model_has_pose_transmission(sf, m):
        return

    # ── the pose, at THIS `qpos` ─────────────────────────────────────────
    # See the header: the products in `d`/`sc` at entry belong to the
    # previous substep. `forward_kinematics` is a pure function of `qpos`,
    # so re-running it is idempotent and the integrator's own call a moment
    # later recomputes the same numbers.
    forward_kinematics["cpu", DTYPE, D2, BATCH](d, m)
    compute_subtree_com["cpu", DTYPE, D2, BATCH](d, m)
    compute_cdof["cpu", DTYPE, D2, BATCH](d, m, sc)

    var nv = d.dims.get_nv()
    var n_act = sf.dims.get_nact()
    var dm = d.dims
    var mdm = m.dims

    comptime V_CAP = cap[D2.CAP_NV]()
    var tJ = Scratch[Scalar[DTYPE], V_CAP](nv, fill=Scalar[DTYPE](0))
    # The site branch's two Jacobian blocks, ROW-MAJOR 3 x nv as `jac_point`
    # writes them (`jacp[k*nv + i]`). Allocated once for the whole loop.
    var jacp = Scratch[Scalar[DTYPE], 3 * V_CAP](
        3 * nv, fill=Scalar[DTYPE](0)
    )
    var jacr = Scratch[Scalar[DTYPE], 3 * V_CAP](
        3 * nv, fill=Scalar[DTYPE](0)
    )
    var nb3 = dm.get_nbody() * 3
    var nb4 = dm.get_nbody() * 4

    # LayoutTensor views, the same idiom every `dynamics/` dispatcher uses.
    var rl_TEN = rl2(mdm.get_ntendon(), MODEL_TENDON_SIZE)
    var rl_SITE = rl2(mdm.get_nsite(), MODEL_SITE_SIZE)
    var rl_GEOM = rl2(mdm.get_ngeom(), MODEL_GEOM_SIZE)
    var rl_BODY = rl2(mdm.get_nbody(), MODEL_BODY_SIZE)
    var rl_JOINT = rl2(mdm.get_njoint(), MODEL_JOINT_SIZE)
    var rl_MMETA = rl1(MODEL_META_SIZE)
    var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
    var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
    var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
    var tendons_v = m.tendons.lt_dyn["cpu", DYN2](rl_TEN)
    var sites_v = m.sites.lt_dyn["cpu", DYN2](rl_SITE)
    var geoms_v = m.geoms.lt_dyn["cpu", DYN2](rl_GEOM)
    var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BODY)
    var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
    var mmeta_v = m.meta.lt_dyn["cpu", DYN1](rl_MMETA)
    var stcom_v = d.subtree_com.lt_dyn["cpu", DYN2](rl_B3)
    var xpos_v = d.xpos.lt_dyn["cpu", DYN2](rl_B3)
    var xquat_v = d.xquat.lt_dyn["cpu", DYN2](rl_B4)
    var cdof_v = sc.cdof.lt_dyn["cpu", DYN2](rl_CDOF)

    var n_ten = mdm.get_ntendon()
    var nten_live = Int(Float64(m.meta.data[MODEL_META_IDX_NTENDON]))
    if nten_live < n_ten:
        n_ten = nten_live

    for e in range(BATCH):
        # ── actuators through a spatial tendon (`mjTRN_TENDON`) ──────────
        for i in range(n_act):
            if i >= len(actions):
                break
            var o = i * MODEL_ACTUATOR_SIZE
            # ⚠ `trn_n != 0` MEANS `apply_actions_fields` ALREADY PAID IT.
            # A joint or fixed-tendon actuator is complete by the time this
            # runs; touching it here would double its force.
            if Int(sf.actuators.data[o + ACT_IDX_TRN_N]) != 0:
                continue
            var tid = Int(sf.actuators.data[o + ACT_IDX_TENDON_ID])
            if tid < 0 or tid >= n_ten:
                continue
            if not _is_spatial(m, tid):
                continue

            var ten_len = Float64(
                spatial_tendon_length_jac[DTYPE, V_CAP, BATCH](
                    e, tid, dm, tendons_v, sites_v, geoms_v, bodies_v,
                    joints_v, mmeta_v, stcom_v, cdof_v, xpos_v, xquat_v, tJ,
                )
            )

            # The force law below is `apply_actions_fields`' term for term;
            # only `length`, `vel` and the moment come from the tendon
            # instead of from the `(qadr, dadr, coef)` triples.
            var ctrl = actions[i]
            if sf.actuators.data[o + ACT_IDX_CTRL_LIMITED] != 0:
                var c_max = Float64(sf.actuators.data[o + ACT_IDX_CTRL_MAX])
                var c_min = Float64(sf.actuators.data[o + ACT_IDX_CTRL_MIN])
                if ctrl > c_max:
                    ctrl = c_max
                elif ctrl < c_min:
                    ctrl = c_min

            var gear = Float64(sf.actuators.data[o + ACT_IDX_GEAR])
            var adr = Int(sf.actuators.data[o + ACT_IDX_ACT_ADR])
            var u = ctrl
            if adr >= 0 and adr < len(act):
                u = Float64(act[adr])

            var kp = Float64(sf.actuators.data[o + ACT_IDX_KP])
            var force = kp * u
            var kind = Int(sf.actuators.data[o + ACT_IDX_KIND])
            comptime _POS = ACT_KIND_POSITION
            comptime _VEL = ACT_KIND_VELOCITY
            var vel = Float64(0)
            if kind == _POS or kind == _VEL:
                for a in range(nv):
                    vel += Float64(tJ[a]) * Float64(d.qvel.data[e * nv + a])
                var length = gear * ten_len
                vel *= gear
                var setpoint = u - length if kind == _POS else u
                var kv = Float64(sf.actuators.data[o + ACT_IDX_KV])
                force = kp * setpoint - kv * vel

            var saturated = False
            if sf.actuators.data[o + ACT_IDX_FORCE_LIMITED] != 0:
                var f_hi = Float64(sf.actuators.data[o + ACT_IDX_FORCE_MAX])
                var f_lo = Float64(sf.actuators.data[o + ACT_IDX_FORCE_MIN])
                if force > f_hi:
                    force = f_hi
                elif force < f_lo:
                    force = f_lo
                saturated = force <= f_lo or force >= f_hi

            # `-d force / d qvel`, diagonal only — the same approximation
            # `apply_actions_fields` documents for a multi-dof transmission,
            # and a spatial tendon is always multi-dof.
            if (kind == _POS or kind == _VEL) and not saturated:
                var kv_d = Float64(sf.actuators.data[o + ACT_IDX_KV])
                if kv_d != 0.0:
                    for a in range(nv):
                        var gc = gear * Float64(tJ[a])
                        d.dof_actdamp.data[e * nv + a] += Scalar[DTYPE](
                            kv_d * gc * gc
                        )

            for a in range(nv):
                if tJ[a] == Scalar[DTYPE](0):
                    continue
                d.qfrc.data[e * nv + a] += Scalar[DTYPE](
                    gear * Float64(tJ[a]) * force
                )

            # mjDYN_FILTER, as `apply_actions_fields` integrates it. These
            # actuators never reach that loop, so this is the only place
            # their activation advances.
            if adr >= 0 and adr < len(act):
                var tau = Float64(sf.actuators.data[o + ACT_IDX_DYN_TAU])
                if tau < 1e-10:
                    tau = 1e-10
                act[adr] = Scalar[DTYPE](u + (ctrl - u) / tau * timestep)

        # ── actuators through a SITE (`mjTRN_SITE`) ──────────────────────
        # `mj_jacSite` at the site's world point, then the gear rotated into
        # the world by the site's frame and contracted with the two Jacobian
        # blocks. `length` is 0 by definition, so a `<position site=>` servos
        # toward 0 — MuJoCo's rule, not an omission.
        for i in range(n_act):
            if i >= len(actions):
                break
            var o = i * MODEL_ACTUATOR_SIZE
            if Int(sf.actuators.data[o + ACT_IDX_TRN_N]) != 0:
                continue
            var sid = Int(sf.actuators.data[o + ACT_IDX_SITE_ID])
            if sid < 0 or sid >= mdm.get_nsite():
                continue

            # The site's world pose, composed rather than read. `Data` has no
            # `site_xmat`; it has the body's `xquat` and the site's LOCAL
            # quat. Rotating by `R_body · R_site` is two `gpu_quat_rotate`
            # calls, and it avoids binding `site_xpos`, which is EMPTY on
            # every site-less model — the operand that crashed three solver
            # tests when `tendon.mojo` tried it (see `_site_world` there).
            var s_body = Int(
                Float64(m.sites.data[sid * MODEL_SITE_SIZE + SITE_IDX_BODY])
            )
            var slx = Float64(
                m.sites.data[sid * MODEL_SITE_SIZE + SITE_IDX_POS_X]
            )
            var sly = Float64(
                m.sites.data[sid * MODEL_SITE_SIZE + SITE_IDX_POS_Y]
            )
            var slz = Float64(
                m.sites.data[sid * MODEL_SITE_SIZE + SITE_IDX_POS_Z]
            )
            var sqx = Scalar[DTYPE](
                m.sites.data[sid * MODEL_SITE_SIZE + SITE_IDX_QUAT_X]
            )
            var sqy = Scalar[DTYPE](
                m.sites.data[sid * MODEL_SITE_SIZE + SITE_IDX_QUAT_Y]
            )
            var sqz = Scalar[DTYPE](
                m.sites.data[sid * MODEL_SITE_SIZE + SITE_IDX_QUAT_Z]
            )
            var sqw = Scalar[DTYPE](
                m.sites.data[sid * MODEL_SITE_SIZE + SITE_IDX_QUAT_W]
            )
            # ⚠ `xquat` IS STORED (x, y, z, w) — w LAST — while a free
            # joint's `qpos` stores it w FIRST. The two conventions live one
            # file apart; `_site_world` in `tendon.mojo` reads it the same
            # way this does.
            var bqx = Scalar[DTYPE](d.xquat.data[e * nb4 + s_body * 4 + 0])
            var bqy = Scalar[DTYPE](d.xquat.data[e * nb4 + s_body * 4 + 1])
            var bqz = Scalar[DTYPE](d.xquat.data[e * nb4 + s_body * 4 + 2])
            var bqw = Scalar[DTYPE](d.xquat.data[e * nb4 + s_body * 4 + 3])
            var srot = gpu_quat_rotate[DTYPE](
                bqx, bqy, bqz, bqw,
                Scalar[DTYPE](slx), Scalar[DTYPE](sly), Scalar[DTYPE](slz),
            )
            var spx = Scalar[DTYPE](
                d.xpos.data[e * nb3 + s_body * 3 + 0]
            ) + srot[0]
            var spy = Scalar[DTYPE](
                d.xpos.data[e * nb3 + s_body * 3 + 1]
            ) + srot[1]
            var spz = Scalar[DTYPE](
                d.xpos.data[e * nb3 + s_body * 3 + 2]
            ) + srot[2]

            for k in range(3 * nv):
                jacp[k] = Scalar[DTYPE](0)
                jacr[k] = Scalar[DTYPE](0)
            jac_point[DTYPE, V_CAP](
                e, stcom_v, joints_v, bodies_v, mmeta_v, cdof_v,
                s_body, spx, spy, spz, jacp, jacr, nv,
            )

            # gear -> a world-frame wrench. `R_body · (R_site · gear)`.
            var g0 = Scalar[DTYPE](sf.actuators.data[o + ACT_IDX_GEAR])
            var g1 = Scalar[DTYPE](sf.actuators.data[o + ACT_IDX_GEAR_1])
            var g2 = Scalar[DTYPE](sf.actuators.data[o + ACT_IDX_GEAR_2])
            var g3 = Scalar[DTYPE](sf.actuators.data[o + ACT_IDX_GEAR_3])
            var g4 = Scalar[DTYPE](sf.actuators.data[o + ACT_IDX_GEAR_4])
            var g5 = Scalar[DTYPE](sf.actuators.data[o + ACT_IDX_GEAR_5])
            var wl = gpu_quat_rotate[DTYPE](sqx, sqy, sqz, sqw, g0, g1, g2)
            var w = gpu_quat_rotate[DTYPE](
                bqx, bqy, bqz, bqw, wl[0], wl[1], wl[2]
            )
            var w2l = gpu_quat_rotate[DTYPE](sqx, sqy, sqz, sqw, g3, g4, g5)
            var w2 = gpu_quat_rotate[DTYPE](
                bqx, bqy, bqz, bqw, w2l[0], w2l[1], w2l[2]
            )

            # moment = jacp^T w + jacr^T w2, into `tJ` (free here — the
            # tendon loops above have finished with it).
            for a in range(nv):
                tJ[a] = (
                    jacp[0 * nv + a] * w[0]
                    + jacp[1 * nv + a] * w[1]
                    + jacp[2 * nv + a] * w[2]
                    + jacr[0 * nv + a] * w2[0]
                    + jacr[1 * nv + a] * w2[1]
                    + jacr[2 * nv + a] * w2[2]
                )

            var ctrl_s = actions[i]
            if sf.actuators.data[o + ACT_IDX_CTRL_LIMITED] != 0:
                var cx = Float64(sf.actuators.data[o + ACT_IDX_CTRL_MAX])
                var cn = Float64(sf.actuators.data[o + ACT_IDX_CTRL_MIN])
                if ctrl_s > cx:
                    ctrl_s = cx
                elif ctrl_s < cn:
                    ctrl_s = cn
            var adr_s = Int(sf.actuators.data[o + ACT_IDX_ACT_ADR])
            var u_s = ctrl_s
            if adr_s >= 0 and adr_s < len(act):
                u_s = Float64(act[adr_s])

            # ⚠ NO `gear` FACTOR HERE. The whole six-vector is already in the
            # moment; multiplying by `ACT_IDX_GEAR` as the joint and
            # fixed-tendon paths do would square its first component.
            var kp_s = Float64(sf.actuators.data[o + ACT_IDX_KP])
            var force_s = kp_s * u_s
            var kind_s = Int(sf.actuators.data[o + ACT_IDX_KIND])
            var vel_s = Float64(0)
            if kind_s == ACT_KIND_POSITION or kind_s == ACT_KIND_VELOCITY:
                for a in range(nv):
                    vel_s += Float64(tJ[a]) * Float64(
                        d.qvel.data[e * nv + a]
                    )
                # `length` is 0 for a site transmission, so a POSITION servo
                # reads `u - 0`.
                var kv_s = Float64(sf.actuators.data[o + ACT_IDX_KV])
                force_s = kp_s * u_s - kv_s * vel_s

            var sat_s = False
            if sf.actuators.data[o + ACT_IDX_FORCE_LIMITED] != 0:
                var fh = Float64(sf.actuators.data[o + ACT_IDX_FORCE_MAX])
                var fl = Float64(sf.actuators.data[o + ACT_IDX_FORCE_MIN])
                if force_s > fh:
                    force_s = fh
                elif force_s < fl:
                    force_s = fl
                sat_s = force_s <= fl or force_s >= fh

            if (
                kind_s == ACT_KIND_POSITION or kind_s == ACT_KIND_VELOCITY
            ) and not sat_s:
                var kvd_s = Float64(sf.actuators.data[o + ACT_IDX_KV])
                if kvd_s != 0.0:
                    for a in range(nv):
                        var mj_a = Float64(tJ[a])
                        d.dof_actdamp.data[e * nv + a] += Scalar[DTYPE](
                            kvd_s * mj_a * mj_a
                        )

            for a in range(nv):
                if tJ[a] == Scalar[DTYPE](0):
                    continue
                d.qfrc.data[e * nv + a] += Scalar[DTYPE](
                    Float64(tJ[a]) * force_s
                )

            if adr_s >= 0 and adr_s < len(act):
                var tau_s = Float64(sf.actuators.data[o + ACT_IDX_DYN_TAU])
                if tau_s < 1e-10:
                    tau_s = 1e-10
                act[adr_s] = Scalar[DTYPE](
                    u_s + (ctrl_s - u_s) / tau_s * timestep
                )

        # ── spatial-tendon SPRINGS (`qfrc_passive`) ──────────────────────
        # Deadband on `tendon_lengthspring`, zero inside the band — the same
        # law `apply_actions_fields` runs for a fixed tendon, over the
        # polyline length instead of `sum(coef * qpos)`.
        for t in range(n_ten):
            if not _is_spatial(m, t):
                continue
            var to = t * MODEL_ACT_TENDON_SIZE
            var k_spring = Float64(
                sf.act_tendons.data[to + ACTTEN_IDX_STIFFNESS]
            )
            if k_spring == 0.0:
                continue
            var length = Float64(
                spatial_tendon_length_jac[DTYPE, V_CAP, BATCH](
                    e, t, dm, tendons_v, sites_v, geoms_v, bodies_v,
                    joints_v, mmeta_v, stcom_v, cdof_v, xpos_v, xquat_v, tJ,
                )
            )
            var lo = Float64(sf.act_tendons.data[to + ACTTEN_IDX_SPRING_LO])
            var hi = Float64(sf.act_tendons.data[to + ACTTEN_IDX_SPRING_HI])
            var frc = Float64(0)
            if length > hi:
                frc = k_spring * (hi - length)
            elif length < lo:
                frc = k_spring * (lo - length)
            if frc == 0.0:
                continue
            for a in range(nv):
                if tJ[a] == Scalar[DTYPE](0):
                    continue
                d.qfrc.data[e * nv + a] += Scalar[DTYPE](
                    Float64(tJ[a]) * frc
                )
