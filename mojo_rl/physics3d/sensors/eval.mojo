"""The `<sensor>` evaluation passes — `mj_sensorPos` / `Vel` / `Acc` (AUD-23).

The six kernels beside this file each compute one sensor, addressed by
`(body, site)` or `(root)`. This is the front end over them: it walks the
model's sensor table, dispatches by `mjtSensor`, writes the values into
`d.sensordata` at each sensor's own `adr`, and applies `cutoff`.

⚠ TWO SENSORS ARE COMPUTED HERE RATHER THAN IN A KERNEL FILE, and that is
deliberate. `jointpos` and `jointvel` are `d->qpos[m->jnt_qposadr[objid]]` and
`d->qvel[m->jnt_dofadr[objid]]` — the whole of the reference's case arms
(`engine_sensor.c:644, 873`). A file per sensor would wrap a subscript in a
call boundary. Their `objid` is a JOINT index, not a site or a body, which is
the only place in this pass where that is true.

⚠⚠ THREE PASSES, BECAUSE MuJoCo HAS THREE. `mj_sensorPos` runs after forward
kinematics, `mj_sensorVel` after the velocity stage, `mj_sensorAcc` after the
constraint solve (`engine_forward.c:1797, 1814, 1832`). A sensor's stage is not
a scheduling detail — it is what decides whether the quantity it reads exists
yet. Collapsing them into one pass would compute the velocimeter from last
step's velocities on every integrator that separates the stages, which is all
of them.

⚠ THE ACCELERATION PASS READS THE `*_acc` SNAPSHOT, NOT THE LIVE FK PRODUCTS.
`site_xpos_acc` / `xquat_acc` are the pre-integration pose (`fields/data.mojo`,
"defect 19"); the live `site_xpos` has already moved by the time the step ends.
Every acceleration-stage kernel here takes the snapshot, matching what the
hand-written env hooks already do.

⚠ CPU, BATCH=1. Every existing CPU sensor caller in the tree is BATCH=1 and
these kernels take host `List`s; the `_gpu` twins take `LayoutTensor`s and want
a kernel of their own rather than a loop over envs. `sensordata` is allocated
`[BATCH, NSENSORDATA]` so the batched pass has somewhere to write when it
lands.

⚠ AN UNSERVED SENSOR IS SKIPPED, AND ITS SLICE STAYS ZERO. It still holds its
`adr`, so every sensor after it lands in the right place — that is the whole
reason `_fill_sensors` gives it a row. Reading one by name raises
(`FlatModelDef._require_served`) rather than handing back zeros that look like
a measurement.
"""

from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import (
    Data, Model, DimsLike, DYN2, rl2,
)
from mojo_rl.physics3d.constants import (
    SENS_TOUCH,
    SENS_ACCELEROMETER,
    SENS_VELOCIMETER,
    SENS_GYRO,
    SENS_FORCE,
    SENS_TORQUE,
    SENS_RANGEFINDER,
    SENS_JOINTPOS,
    SENS_JOINTVEL,
    SENS_TENDONPOS,
    SENS_ACTUATORPOS,
    SENS_JOINTACTFRC,
    SENS_FRAMEPOS,
    SENS_FRAMEQUAT,
    SENS_FRAMEXAXIS,
    SENS_FRAMEYAXIS,
    SENS_FRAMEZAXIS,
    SENS_FRAMELINVEL,
    SENS_FRAMEANGVEL,
    SENS_SUBTREECOM,
    SENSOBJ_UNKNOWN,
    SENS_SUBTREELINVEL,
    SENSDATA_REAL,
    SENSDATA_POSITIVE,
    SENSSTAGE_POS,
    SENSSTAGE_VEL,
    SENSSTAGE_ACC,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    CONTACT_SIZE,
    METADATA_SIZE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    MODEL_SENSOR_SIZE,
    SENSOR_IDX_TYPE,
    SENSOR_IDX_OBJTYPE,
    SENSOR_IDX_OBJID,
    SENSOR_IDX_DIM,
    SENSOR_IDX_ADR,
    SENSOR_IDX_DATATYPE,
    SENSOR_IDX_NEEDSTAGE,
    SENSOR_IDX_CUTOFF,
    SENSOR_IDX_BODY,
    SENSOR_IDX_SERVED,
    SENSOR_IDX_REFTYPE,
    SENSOR_IDX_REFID,
)
from .frame import (
    frame_object_pose,
    frame_object_body,
    frame_pos_sensor,
    frame_axis_sensor,
    frame_quat_sensor,
    frame_vel_sensor,
)
from .frame_vel import site_frame_velocity_gpu
from .site_acc import site_accelerometer_gpu, site_force_torque_gpu
from .subtree import subtree_linvel_gpu
from .touch import touch_sphere_site_gpu
from .rangefinder import rangefinder_site


@always_inline
def _atleast1(n: Int) -> Int:
    """A zero-extent LayoutTensor cannot be bound; an optional table floors to
    one row exactly as `Data` allocates it."""
    return n if n > 0 else 1


@always_inline
def _apply_cutoff[
    DTYPE: DType, L_SD: Layout
](
    sensordata: LayoutTensor[DTYPE, L_SD, MutAnyOrigin],
    env: Int,
    adr: Int,
    dim: Int,
    datatype: Int,
    cutoff: Float64,
):
    """`apply_cutoff` (engine_sensor.c:198-224), verbatim.

    ⚠ `cutoff <= 0` DISABLES CLAMPING; it does not clamp to zero. That is the
    reference's first line and the reason `SensorData.cutoff` is a plain value
    with no `has_cutoff` flag beside it.

    REAL clips to `[-cutoff, +cutoff]`; POSITIVE takes `min(cutoff, x)` and
    leaves the negative side alone. ⚠ The rangefinder is REAL, not POSITIVE —
    the audit's AUD-47 said otherwise and was wrong; only touch and insidesite
    are positive in 3.10, 3.11 and 3.12 alike.
    """
    if cutoff <= 0.0:
        return
    var c = Scalar[DTYPE](cutoff)
    for k in range(dim):
        var v = rebind[Scalar[DTYPE]](sensordata[env, adr + k])
        if datatype == SENSDATA_REAL:
            if v > c:
                v = c
            elif v < -c:
                v = -c
        elif datatype == SENSDATA_POSITIVE:
            if v > c:
                v = c
        sensordata[env, adr + k] = v


def _eval_stage[
    DTYPE: DType, D: DimsLike, BATCH: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    stage: Int,
    have_rne_post: Bool = True,
) raises:
    """Evaluate every SERVED sensor whose `needstage` is `stage`.

    One function rather than three near-identical ones: the stage is a filter
    over the same table, and the three public entry points below name the
    stages so a caller reads like `mj_sensorPos` and cannot pass a number.

    ⚠⚠ `have_rne_post` IS NOT THE SAME AS "THE ACCELERATION STAGE RUNS", and
    conflating them was a real bug. MuJoCo classes TOUCH as an acceleration
    sensor, so the first version of this gated the whole stage on `RNE_POST`
    and hopper — which declares two touch sensors and runs `RNE_POST=False` —
    stopped stepping. Touch reads CONTACTS and the live `site_xpos`; it never
    touches `cacc` or `cfrc_int`. Only the accelerometer and the force/torque
    pair need the post-constraint RNE, and only those are skipped when it has
    not run.
    """
    var nsensor = m.dims.get_nsensor()
    if nsensor == 0:
        return

    var nbody = m.dims.get_nbody()
    var dm = d.dims
    var mdm = m.dims

    # ── the batched views, bound ONCE for the whole pass ──────────────────
    #
    # ⚠⚠ EVERY KERNEL BELOW IS THE `_gpu` TWIN, EVEN ON THE CPU, AND THAT IS
    # WHAT MAKES `BATCH > 1` WORK AT ALL (AUD-53). The `List` forms beside
    # them index from 0 and cannot see an env; their `_gpu` twins take
    # `(tensor..., env)` and were written for the device. Calling those with
    # `lt_dyn["cpu", ...]` views is the idiom every `dynamics/` dispatcher
    # already uses, and it means the single-env and batched legs run ONE
    # implementation rather than two. The `List` forms stay for the env config
    # hooks that predate the framework and call them directly.
    var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
    var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
    var rl_B6 = rl2(BATCH, dm.get_nbody() * 6)
    var rl_S3 = rl2(BATCH, _atleast1(dm.get_nsite() * 3))
    var rl_NQ = rl2(BATCH, dm.get_nq())
    var rl_NV = rl2(BATCH, dm.get_nv())
    var rl_SD = rl2(BATCH, _atleast1(mdm.get_nsensordata()))
    var rl_TEN = rl2(BATCH, _atleast1(mdm.get_ntendon()))
    var rl_ACT = rl2(BATCH, _atleast1(mdm.get_nact()))
    var rl_CON = rl2(BATCH, dm.get_max_contacts() * CONTACT_SIZE)
    var rl_META = rl2(BATCH, METADATA_SIZE)
    var rl_BODY = rl2(mdm.get_nbody(), MODEL_BODY_SIZE)
    var rl_SITE = rl2(_atleast1(mdm.get_nsite()), MODEL_SITE_SIZE)
    var rl_GEOM = rl2(_atleast1(mdm.get_ngeom()), MODEL_GEOM_SIZE)
    var rl_JOINT = rl2(mdm.get_njoint(), MODEL_JOINT_SIZE)

    var xpos_v = d.xpos.lt_dyn["cpu", DYN2](rl_B3)
    var xquat_v = d.xquat.lt_dyn["cpu", DYN2](rl_B4)
    var xipos_v = d.xipos.lt_dyn["cpu", DYN2](rl_B3)
    var xvel_v = d.xvel.lt_dyn["cpu", DYN2](rl_B3)
    var xangvel_v = d.xangvel.lt_dyn["cpu", DYN2](rl_B3)
    var site_xpos_v = d.site_xpos.lt_dyn["cpu", DYN2](rl_S3)
    var site_xpos_acc_v = d.site_xpos_acc.lt_dyn["cpu", DYN2](rl_S3)
    var xquat_acc_v = d.xquat_acc.lt_dyn["cpu", DYN2](rl_B4)
    var cvel_v = d.cvel.lt_dyn["cpu", DYN2](rl_B6)
    var cacc_v = d.cacc.lt_dyn["cpu", DYN2](rl_B6)
    var cfrc_int_v = d.cfrc_int.lt_dyn["cpu", DYN2](rl_B6)
    var stcom_v = d.subtree_com.lt_dyn["cpu", DYN2](rl_B3)
    var qpos_v = d.qpos.lt_dyn["cpu", DYN2](rl_NQ)
    var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_NV)
    var qfrc_v = d.qfrc.lt_dyn["cpu", DYN2](rl_NV)
    var tenlen_v = d.ten_length.lt_dyn["cpu", DYN2](rl_TEN)
    var actlen_v = d.actuator_length.lt_dyn["cpu", DYN2](rl_ACT)
    var sdata_v = d.sensordata.lt_dyn["cpu", DYN2](rl_SD)
    var con_v = d.contacts.lt_dyn["cpu", DYN2](rl_CON)
    var dmeta_v = d.meta.lt_dyn["cpu", DYN2](rl_META)
    var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BODY)
    var sites_v = m.sites.lt_dyn["cpu", DYN2](rl_SITE)
    var geoms_v = m.geoms.lt_dyn["cpu", DYN2](rl_GEOM)
    var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)

    for env in range(BATCH):
        for i in range(nsensor):
            var o = i * MODEL_SENSOR_SIZE
            if Int(m.sensors.data[o + SENSOR_IDX_SERVED]) != 1:
                continue
            if Int(m.sensors.data[o + SENSOR_IDX_NEEDSTAGE]) != stage:
                continue

            var st = Int(m.sensors.data[o + SENSOR_IDX_TYPE])
            # The three that genuinely read `cacc` / `cfrc_int`. Without the
            # post-constraint RNE they would report whatever those buffers hold,
            # so they are skipped and their slots keep the NaN `Data` filled them
            # with — loud to a reader, inert to the seven manipulation envs that
            # declare force/torque sensors and never read `sensordata`.
            if not have_rne_post and _needs_rne_post(st):
                continue
            var objid = Int(m.sensors.data[o + SENSOR_IDX_OBJID])
            var body = Int(m.sensors.data[o + SENSOR_IDX_BODY])
            var adr = Int(m.sensors.data[o + SENSOR_IDX_ADR])
            var dim = Int(m.sensors.data[o + SENSOR_IDX_DIM])

            if st == SENS_RANGEFINDER:
                # ⚠⚠ A COMPTIME DType SPLIT, AND NOT BY PREFERENCE. Only
                # `rangefinder_site` among the six carries
                # `where DTYPE.is_floating_point()` (it reaches `ray_model`), and
                # this pass cannot: it is called from `EulerIntegrator.step`,
                # whose `DTYPE` is unconstrained because the env-config trait that
                # reaches it is. Adding the constraint upward makes the compiler
                # reject the whole conformance — the same wall `touch.mojo`'s GPU
                # twin hit, and the same fix: name the two concrete float types
                # here rather than widen a trait every environment implements.
                comptime if DTYPE == DType.float32:
                    sdata_v[env, adr] = Scalar[DTYPE](
                        rangefinder_site[DType.float32, D, BATCH](
                            rebind[Data[DType.float32, D, BATCH]](d),
                            rebind[Model[DType.float32, D]](m),
                            objid, env,
                        )
                    )
                else:
                    comptime if DTYPE == DType.float64:
                        sdata_v[env, adr] = Scalar[DTYPE](
                            rangefinder_site[DType.float64, D, BATCH](
                                rebind[Data[DType.float64, D, BATCH]](d),
                                rebind[Model[DType.float64, D]](m),
                                objid, env,
                            )
                        )

            elif st == SENS_VELOCIMETER or st == SENS_GYRO:
                # ⚠ ONE KERNEL, TWO SENSORS, AND IT RETURNS LINEAR FIRST.
                # `site_frame_velocity` gives (lin, ang) — the opposite order to
                # MuJoCo's packed `res` — so the velocimeter takes [0..2] and the
                # gyro [3..5]. Getting this backwards is silent: both are 3-vectors
                # of plausible magnitude.
                var fv = site_frame_velocity_gpu[DTYPE](
                    xvel_v, xangvel_v, xipos_v, xquat_v, site_xpos_v,
                    sites_v, env, body, objid,
                )
                # ⚠ UNROLLED, AND NOT BY PREFERENCE: `site_frame_velocity`
                # returns a TUPLE, whose index must be a compile-time constant.
                # `fv[base + k]` with a runtime `base` does not compile.
                var vbase = 0 if st == SENS_VELOCIMETER else 3
                for k in range(3):
                    sdata_v[env, adr + k] = fv[vbase + k]

            elif st == SENS_JOINTPOS:
                # `mjSENS_JOINTPOS` (engine_sensor.c:644), verbatim:
                # `sensordata[0] = d->qpos[m->jnt_qposadr[objid]]`.
                #
                # ⚠ `objid` IS A JOINT INDEX HERE, NOT A SITE. The other seven
                # served types resolve to a site or a body; these two are the only
                # ones whose `objtype` is `mjOBJ_JOINT`, and `_fill_sensors` is
                # where that is decided. ⚠ The joint is guaranteed slide or hinge
                # — the loader refuses anything else, as MuJoCo's compiler does —
                # so `qpos_adr` names exactly one scalar.
                sdata_v[env, adr] = rebind[Scalar[DTYPE]](qpos_v[
                    env,
                    Int(rebind[Scalar[DTYPE]](
                        joints_v[objid, JOINT_IDX_QPOS_ADR]))
                ])

            elif st == SENS_JOINTVEL:
                # `mjSENS_JOINTVEL` (engine_sensor.c:873):
                # `sensordata[0] = d->qvel[m->jnt_dofadr[objid]]`.
                #
                # ⚠ `dof_adr`, NOT `qpos_adr`. They coincide on a model whose
                # joints are all scalar and diverge the moment a free or ball
                # joint appears anywhere BEFORE this one — which is most models
                # that declare these sensors at all. Reading the wrong table gives
                # a plausible number from the wrong joint.
                sdata_v[env, adr] = rebind[Scalar[DTYPE]](qvel_v[
                    env,
                    Int(rebind[Scalar[DTYPE]](
                        joints_v[objid, JOINT_IDX_DOF_ADR]))
                ])

            elif (
                st == SENS_FRAMEPOS
                or st == SENS_FRAMEQUAT
                or st == SENS_FRAMEXAXIS
                or st == SENS_FRAMEYAXIS
                or st == SENS_FRAMEZAXIS
                or st == SENS_FRAMELINVEL
                or st == SENS_FRAMEANGVEL
            ):
                # ⚠ FIVE SENSORS, ONE POSE LOOKUP, AND THE REFERENCE FRAME IS
                # PART OF IT. `objtype` selects among body / xbody / geom / site
                # (the four this loader resolves); `reftype`/`refid` are MuJoCo's
                # optional relative form and `-1` is its own "absent". A frame
                # sensor whose reference were ignored would report the GLOBAL
                # quantity — right units, right magnitude, wrong frame — so the
                # branch is here rather than in the parser.
                var otype = Int(m.sensors.data[o + SENSOR_IDX_OBJTYPE])
                var op = frame_object_pose[DTYPE](
                    xpos_v, xquat_v, xipos_v, site_xpos_v,
                    bodies_v, geoms_v, sites_v, env, otype, objid,
                )
                var rtype = Int(m.sensors.data[o + SENSOR_IDX_REFTYPE])
                var refid = Int(m.sensors.data[o + SENSOR_IDX_REFID])
                var has_ref = refid >= 0 and rtype != SENSOBJ_UNKNOWN
                # ⚠ THE REFERENCE POSE IS FETCHED UNCONDITIONALLY and ignored
                # when absent. `frame_object_pose` is total (its own docstring
                # says so) and a `Tuple` cannot be declared and filled later in
                # two branches without the compiler losing track of it; the cost
                # is one pose lookup on models that declare no reference, which
                # is every model in this tree today.
                var rp = frame_object_pose[DTYPE](
                    xpos_v, xquat_v, xipos_v, site_xpos_v,
                    bodies_v, geoms_v, sites_v, env,
                    rtype if has_ref else SENSOBJ_UNKNOWN,
                    refid if has_ref else 0,
                )

                if st == SENS_FRAMEPOS:
                    var v = frame_pos_sensor(
                        op[0], op[1], op[2], has_ref,
                        rp[0], rp[1], rp[2], rp[3], rp[4], rp[5], rp[6],
                    )
                    # ⚠ UNROLLED: `frame_pos_sensor` / `frame_axis_sensor`
                    # return a TUPLE, whose index must be a compile-time
                    # constant. The `_gpu` kernels return `Array` and can be
                    # indexed by a loop variable; these cannot.
                    sdata_v[env, adr + 0] = Scalar[DTYPE](v[0])
                    sdata_v[env, adr + 1] = Scalar[DTYPE](v[1])
                    sdata_v[env, adr + 2] = Scalar[DTYPE](v[2])
                elif st == SENS_FRAMEQUAT:
                    # ⚠ (w, x, y, z) COMES BACK, because that is what
                    # `sensordata` holds. See `frame.mojo`'s module note.
                    var q = frame_quat_sensor(
                        op[3], op[4], op[5], op[6], has_ref,
                        rp[3], rp[4], rp[5], rp[6],
                    )
                    sdata_v[env, adr + 0] = Scalar[DTYPE](q[0])
                    sdata_v[env, adr + 1] = Scalar[DTYPE](q[1])
                    sdata_v[env, adr + 2] = Scalar[DTYPE](q[2])
                    sdata_v[env, adr + 3] = Scalar[DTYPE](q[3])
                elif st == SENS_FRAMELINVEL or st == SENS_FRAMEANGVEL:
                    # ⚠ THE VELOCITY STAGE, AND `needstage` IS WHAT PUTS IT
                    # THERE. These two are the only frame sensors MuJoCo
                    # evaluates in `mj_sensorVel`; the stage filter at the top of
                    # this loop is what keeps them from reading `xvel`/`xangvel`
                    # a stage too early, when they still describe the previous
                    # step.
                    var fb = frame_object_body[DTYPE](
                        geoms_v, sites_v, otype, objid
                    )
                    var rb = frame_object_body[DTYPE](
                        geoms_v, sites_v,
                        rtype if has_ref else SENSOBJ_UNKNOWN,
                        refid if has_ref else 0,
                    )
                    var fv = frame_vel_sensor[DTYPE](
                        xvel_v, xangvel_v, xipos_v, env, fb,
                        op[0], op[1], op[2],
                        has_ref, rb, rp[0], rp[1], rp[2],
                        rp[3], rp[4], rp[5], rp[6],
                    )
                    # ⚠ ANGULAR FIRST, THEN LINEAR — `frame_vel_sensor` returns
                    # MuJoCo's packed order, not `site_frame_velocity`'s. Both
                    # halves are three plausible floats, so a swap is silent.
                    if st == SENS_FRAMELINVEL:
                        sdata_v[env, adr + 0] = Scalar[DTYPE](fv[3])
                        sdata_v[env, adr + 1] = Scalar[DTYPE](fv[4])
                        sdata_v[env, adr + 2] = Scalar[DTYPE](fv[5])
                    else:
                        sdata_v[env, adr + 0] = Scalar[DTYPE](fv[0])
                        sdata_v[env, adr + 1] = Scalar[DTYPE](fv[1])
                        sdata_v[env, adr + 2] = Scalar[DTYPE](fv[2])
                else:
                    # `mjSENS_FRAMEXAXIS` is 28 and the three are consecutive, so
                    # the axis index is the type offset — MuJoCo's own
                    # `type - mjSENS_FRAMEXAXIS` (engine_sensor.c:694).
                    var v = frame_axis_sensor(
                        op[3], op[4], op[5], op[6], st - SENS_FRAMEXAXIS,
                        has_ref, rp[3], rp[4], rp[5], rp[6],
                    )
                    # ⚠ UNROLLED: `frame_pos_sensor` / `frame_axis_sensor`
                    # return a TUPLE, whose index must be a compile-time
                    # constant. The `_gpu` kernels return `Array` and can be
                    # indexed by a loop variable; these cannot.
                    sdata_v[env, adr + 0] = Scalar[DTYPE](v[0])
                    sdata_v[env, adr + 1] = Scalar[DTYPE](v[1])
                    sdata_v[env, adr + 2] = Scalar[DTYPE](v[2])

            elif st == SENS_ACTUATORPOS:
                # `mjSENS_ACTUATORPOS` (engine_sensor.c:652):
                # `d->actuator_length[actuator_outadr[objid]]`, `sensor_dim`
                # values. Every transmission MuJoCo currently ships has an output
                # block of ONE row (`mj_transmission`'s own comment at :1299), so
                # `outadr[i] == i` and the dim is 1.
                #
                # ⚠ FILLED BY `compute_actuator_lengths`, which is guarded on THIS
                # row existing — and which leaves the slot NaN for a transmission
                # it cannot express. Such a row is marked unserved at load, so the
                # pass above skips it and this line is never reached for one.
                sdata_v[env, adr] = rebind[Scalar[DTYPE]](
                    actlen_v[env, objid])

            elif st == SENS_TENDONPOS:
                # `mjSENS_TENDONPOS` (engine_sensor.c:648):
                # `sensordata[0] = d->ten_length[objid]`.
                #
                # ⚠ `d.ten_length` IS FILLED ONLY WHEN A SENSOR ASKS, by
                # `dynamics/tendon_lengths.compute_tendon_lengths`, which runs at
                # MuJoCo's `mj_tendon` point in the step. Its guard is THIS row's
                # existence, so the two cannot drift: no `<tendonpos>`, no pass,
                # and the array keeps the NaN `Data` allocated it with.
                sdata_v[env, adr] = rebind[Scalar[DTYPE]](
                    tenlen_v[env, objid])

            elif st == SENS_JOINTACTFRC:
                # `mjSENS_JOINTACTFRC` (engine_sensor.c:1309):
                # `sensordata[0] = d->qfrc_actuator[m->jnt_dofadr[objid]]`.
                #
                # ⚠⚠ `d.qfrc` IS THIS TREE'S `qfrc_actuator`, AND `d.qfrc_actuator`
                # IS NOT. The field of that name in `Data` is allocated, uploaded,
                # downloaded and never written by anything in `physics3d`. What
                # `apply_actions_fields` fills — and what
                # `_clamp_joint_actfrc` clamps against `jnt_actfrcrange`, exactly
                # as `mj_fwdActuation` clamps `qfrc_actuator`
                # (engine_forward.c:722-738) — is `d.qfrc`. Reading the
                # same-named buffer would have returned zeros forever.
                #
                # ⚠ IT IS SHORT `actuatorgravcomp` (AUD-30). MuJoCo routes a
                # gravcomp body's share into `qfrc_actuator` when the actuator
                # asks for it; this engine leaves it in the passive term. No model
                # in this tree declares both, and the audit tracks it separately.
                sdata_v[env, adr] = rebind[Scalar[DTYPE]](qfrc_v[
                    env,
                    Int(rebind[Scalar[DTYPE]](
                        joints_v[objid, JOINT_IDX_DOF_ADR]))
                ])

            elif st == SENS_SUBTREECOM:
                # `mjSENS_SUBTREECOM` (engine_sensor.c:737):
                # `mju_copy3(sensordata, d->subtree_com + 3*objid)`.
                #
                # ⚠ THIS READ IS WHY `compute_subtree_com` MOVED AHEAD OF
                # `sensor_pos` IN THE STEP. It is `mj_comPos`, upstream of
                # `mj_sensorPos` in the reference; evaluated where it used to sit
                # this would have reported LAST step's centre of mass.
                for k in range(3):
                    sdata_v[env, adr + k] = rebind[Scalar[DTYPE]](
                        stcom_v[env, objid * 3 + k])

            elif st == SENS_SUBTREELINVEL:
                var svx = Scalar[DTYPE](0)
                var svy = Scalar[DTYPE](0)
                var svz = Scalar[DTYPE](0)
                subtree_linvel_gpu[DTYPE](
                    mdm, xvel_v, bodies_v, env, objid, svx, svy, svz
                )
                sdata_v[env, adr + 0] = svx
                sdata_v[env, adr + 1] = svy
                sdata_v[env, adr + 2] = svz

            elif st == SENS_ACCELEROMETER:
                # `*_acc`, not the live FK products — see the module note.
                var a = site_accelerometer_gpu[DTYPE](
                    cvel_v, cacc_v, stcom_v, site_xpos_acc_v, xquat_acc_v,
                    bodies_v, sites_v, env, body, objid,
                )
                for k in range(3):
                    sdata_v[env, adr + k] = a[k]

            elif st == SENS_FORCE or st == SENS_TORQUE:
                # ⚠ ONE KERNEL, TWO SENSORS AGAIN: force first, then torque.
                var ft = site_force_torque_gpu[DTYPE](
                    cfrc_int_v, stcom_v, site_xpos_acc_v, xquat_acc_v,
                    bodies_v, sites_v, env, body, objid,
                )
                # Tuple again — see the velocimeter note above.
                var fbase = 0 if st == SENS_FORCE else 3
                for k in range(3):
                    sdata_v[env, adr + k] = ft[fbase + k]

            elif st == SENS_TOUCH:
                # `scale` 1.0 is the MuJoCo-matching value; the parameter exists
                # for callers wanting impulses, and a sensor is not one of them.
                sdata_v[env, adr] = touch_sphere_site_gpu[DTYPE](
                    mdm, con_v, site_xpos_v, sites_v, dmeta_v, xquat_v,
                    env, objid, Scalar[DTYPE](1.0),
                )

            _apply_cutoff[DTYPE](
                sdata_v,
                env,
                adr,
                dim,
                Int(m.sensors.data[o + SENSOR_IDX_DATATYPE]),
                Float64(m.sensors.data[o + SENSOR_IDX_CUTOFF]),
            )


@always_inline
def _needs_rne_post(sensor_type: Int) -> Bool:
    """Does this sensor read `cacc` / `cfrc_int`?

    ⚠ TOUCH IS AN ACCELERATION-STAGE SENSOR AND IS NOT ONE OF THESE. It sums
    contact normal forces over a zone — contacts and `site_xpos`, both valid
    without the post-constraint RNE. Putting it in this set is what broke
    hopper.
    """
    return (
        sensor_type == SENS_ACCELEROMETER
        or sensor_type == SENS_FORCE
        or sensor_type == SENS_TORQUE
    )


@always_inline
def sensor_pos[
    DTYPE: DType, D: DimsLike, BATCH: Int = 1
](mut d: Data[DTYPE, D, BATCH], mut m: Model[DTYPE, D]) raises:
    """`mj_sensorPos` — the position stage. Call after forward kinematics."""
    _eval_stage[DTYPE, D, BATCH](d, m, SENSSTAGE_POS)


@always_inline
def sensor_vel[
    DTYPE: DType, D: DimsLike, BATCH: Int = 1
](mut d: Data[DTYPE, D, BATCH], mut m: Model[DTYPE, D]) raises:
    """`mj_sensorVel` — the velocity stage. Call after body velocities."""
    _eval_stage[DTYPE, D, BATCH](d, m, SENSSTAGE_VEL)


@always_inline
def sensor_acc[
    DTYPE: DType, D: DimsLike, BATCH: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    have_rne_post: Bool = True,
) raises:
    """`mj_sensorAcc` — the acceleration stage.

    Call after the constraint solve and after `rne_post` has filled `cacc` /
    `cfrc_int`, at the point `EulerIntegrator.step` already evaluates the
    hand-written acceleration-stage hooks.
    """
    _eval_stage[DTYPE, D, BATCH](d, m, SENSSTAGE_ACC, have_rne_post)
