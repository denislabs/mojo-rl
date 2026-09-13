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

from mojo_rl.physics3d.fields import Data, Model, DimsLike
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
from .frame_vel import site_frame_velocity
from .site_acc import site_accelerometer, site_force_torque
from .subtree import subtree_linvel
from .touch import touch_sphere_site
from .rangefinder import rangefinder_site


@always_inline
def _apply_cutoff[
    DTYPE: DType
](
    mut d_sensordata: List[Scalar[DTYPE]],
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
        var v = d_sensordata[adr + k]
        if datatype == SENSDATA_REAL:
            if v > c:
                v = c
            elif v < -c:
                v = -c
        elif datatype == SENSDATA_POSITIVE:
            if v > c:
                v = c
        d_sensordata[adr + k] = v


def _eval_stage[
    DTYPE: DType, D: DimsLike
](
    mut d: Data[DTYPE, D, 1],
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

    # ⚠ `m.sites.data` IS PASSED AT EACH CALL, NOT HOISTED INTO A LOCAL.
    # `List` is not `ImplicitlyCopyable` here, so binding it would need a
    # `.copy()` — a full copy of the record slab, per pass, to save a field
    # access that costs nothing. (The first draft hoisted it with a comment
    # about "rebuilding the list per sensor"; there is no rebuild. The
    # compiler caught it.)
    var nbody = m.dims.get_nbody()

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
                d.sensordata.data[adr] = Scalar[DTYPE](
                    rangefinder_site[DType.float32, D, 1](
                        rebind[Data[DType.float32, D, 1]](d),
                        rebind[Model[DType.float32, D]](m),
                        objid, 0,
                    )
                )
            else:
                comptime if DTYPE == DType.float64:
                    d.sensordata.data[adr] = Scalar[DTYPE](
                        rangefinder_site[DType.float64, D, 1](
                            rebind[Data[DType.float64, D, 1]](d),
                            rebind[Model[DType.float64, D]](m),
                            objid, 0,
                        )
                    )

        elif st == SENS_VELOCIMETER or st == SENS_GYRO:
            # ⚠ ONE KERNEL, TWO SENSORS, AND IT RETURNS LINEAR FIRST.
            # `site_frame_velocity` gives (lin, ang) — the opposite order to
            # MuJoCo's packed `res` — so the velocimeter takes [0..2] and the
            # gyro [3..5]. Getting this backwards is silent: both are 3-vectors
            # of plausible magnitude.
            var fv = site_frame_velocity[DTYPE](
                d.xvel.data, d.xangvel.data, d.xipos.data, d.xquat.data,
                d.site_xpos.data, m.sites.data, body, objid,
            )
            # ⚠ UNROLLED, AND NOT BY PREFERENCE: `site_frame_velocity`
            # returns a TUPLE, whose index must be a compile-time constant.
            # `fv[base + k]` with a runtime `base` does not compile.
            if st == SENS_VELOCIMETER:
                d.sensordata.data[adr + 0] = Scalar[DTYPE](fv[0])
                d.sensordata.data[adr + 1] = Scalar[DTYPE](fv[1])
                d.sensordata.data[adr + 2] = Scalar[DTYPE](fv[2])
            else:
                d.sensordata.data[adr + 0] = Scalar[DTYPE](fv[3])
                d.sensordata.data[adr + 1] = Scalar[DTYPE](fv[4])
                d.sensordata.data[adr + 2] = Scalar[DTYPE](fv[5])

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
            d.sensordata.data[adr] = d.qpos.data[
                Int(m.joints.data[objid * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
            ]

        elif st == SENS_JOINTVEL:
            # `mjSENS_JOINTVEL` (engine_sensor.c:873):
            # `sensordata[0] = d->qvel[m->jnt_dofadr[objid]]`.
            #
            # ⚠ `dof_adr`, NOT `qpos_adr`. They coincide on a model whose
            # joints are all scalar and diverge the moment a free or ball
            # joint appears anywhere BEFORE this one — which is most models
            # that declare these sensors at all. Reading the wrong table gives
            # a plausible number from the wrong joint.
            d.sensordata.data[adr] = d.qvel.data[
                Int(m.joints.data[objid * MODEL_JOINT_SIZE + JOINT_IDX_DOF_ADR])
            ]

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
                d.xpos.data, d.xquat.data, d.xipos.data, d.site_xpos.data,
                m.bodies.data, m.geoms.data, m.sites.data, otype, objid,
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
                d.xpos.data, d.xquat.data, d.xipos.data, d.site_xpos.data,
                m.bodies.data, m.geoms.data, m.sites.data,
                rtype if has_ref else SENSOBJ_UNKNOWN,
                refid if has_ref else 0,
            )

            if st == SENS_FRAMEPOS:
                var v = frame_pos_sensor(
                    op[0], op[1], op[2], has_ref,
                    rp[0], rp[1], rp[2], rp[3], rp[4], rp[5], rp[6],
                )
                d.sensordata.data[adr + 0] = Scalar[DTYPE](v[0])
                d.sensordata.data[adr + 1] = Scalar[DTYPE](v[1])
                d.sensordata.data[adr + 2] = Scalar[DTYPE](v[2])
            elif st == SENS_FRAMEQUAT:
                # ⚠ (w, x, y, z) COMES BACK, because that is what
                # `sensordata` holds. See `frame.mojo`'s module note.
                var q = frame_quat_sensor(
                    op[3], op[4], op[5], op[6], has_ref,
                    rp[3], rp[4], rp[5], rp[6],
                )
                d.sensordata.data[adr + 0] = Scalar[DTYPE](q[0])
                d.sensordata.data[adr + 1] = Scalar[DTYPE](q[1])
                d.sensordata.data[adr + 2] = Scalar[DTYPE](q[2])
                d.sensordata.data[adr + 3] = Scalar[DTYPE](q[3])
            elif st == SENS_FRAMELINVEL or st == SENS_FRAMEANGVEL:
                # ⚠ THE VELOCITY STAGE, AND `needstage` IS WHAT PUTS IT
                # THERE. These two are the only frame sensors MuJoCo
                # evaluates in `mj_sensorVel`; the stage filter at the top of
                # this loop is what keeps them from reading `xvel`/`xangvel`
                # a stage too early, when they still describe the previous
                # step.
                var fb = frame_object_body[DTYPE](
                    m.geoms.data, m.sites.data, otype, objid
                )
                var rb = frame_object_body[DTYPE](
                    m.geoms.data, m.sites.data,
                    rtype if has_ref else SENSOBJ_UNKNOWN,
                    refid if has_ref else 0,
                )
                var fv = frame_vel_sensor[DTYPE](
                    d.xvel.data, d.xangvel.data, d.xipos.data, fb,
                    op[0], op[1], op[2],
                    has_ref, rb, rp[0], rp[1], rp[2],
                    rp[3], rp[4], rp[5], rp[6],
                )
                # ⚠ ANGULAR FIRST, THEN LINEAR — `frame_vel_sensor` returns
                # MuJoCo's packed order, not `site_frame_velocity`'s. Both
                # halves are three plausible floats, so a swap is silent.
                if st == SENS_FRAMELINVEL:
                    d.sensordata.data[adr + 0] = Scalar[DTYPE](fv[3])
                    d.sensordata.data[adr + 1] = Scalar[DTYPE](fv[4])
                    d.sensordata.data[adr + 2] = Scalar[DTYPE](fv[5])
                else:
                    d.sensordata.data[adr + 0] = Scalar[DTYPE](fv[0])
                    d.sensordata.data[adr + 1] = Scalar[DTYPE](fv[1])
                    d.sensordata.data[adr + 2] = Scalar[DTYPE](fv[2])
            else:
                # `mjSENS_FRAMEXAXIS` is 28 and the three are consecutive, so
                # the axis index is the type offset — MuJoCo's own
                # `type - mjSENS_FRAMEXAXIS` (engine_sensor.c:694).
                var v = frame_axis_sensor(
                    op[3], op[4], op[5], op[6], st - SENS_FRAMEXAXIS,
                    has_ref, rp[3], rp[4], rp[5], rp[6],
                )
                d.sensordata.data[adr + 0] = Scalar[DTYPE](v[0])
                d.sensordata.data[adr + 1] = Scalar[DTYPE](v[1])
                d.sensordata.data[adr + 2] = Scalar[DTYPE](v[2])

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
            d.sensordata.data[adr] = d.actuator_length.data[objid]

        elif st == SENS_TENDONPOS:
            # `mjSENS_TENDONPOS` (engine_sensor.c:648):
            # `sensordata[0] = d->ten_length[objid]`.
            #
            # ⚠ `d.ten_length` IS FILLED ONLY WHEN A SENSOR ASKS, by
            # `dynamics/tendon_lengths.compute_tendon_lengths`, which runs at
            # MuJoCo's `mj_tendon` point in the step. Its guard is THIS row's
            # existence, so the two cannot drift: no `<tendonpos>`, no pass,
            # and the array keeps the NaN `Data` allocated it with.
            d.sensordata.data[adr] = d.ten_length.data[objid]

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
            d.sensordata.data[adr] = d.qfrc.data[
                Int(m.joints.data[objid * MODEL_JOINT_SIZE + JOINT_IDX_DOF_ADR])
            ]

        elif st == SENS_SUBTREECOM:
            # `mjSENS_SUBTREECOM` (engine_sensor.c:737):
            # `mju_copy3(sensordata, d->subtree_com + 3*objid)`.
            #
            # ⚠ THIS READ IS WHY `compute_subtree_com` MOVED AHEAD OF
            # `sensor_pos` IN THE STEP. It is `mj_comPos`, upstream of
            # `mj_sensorPos` in the reference; evaluated where it used to sit
            # this would have reported LAST step's centre of mass.
            d.sensordata.data[adr + 0] = d.subtree_com.data[objid * 3 + 0]
            d.sensordata.data[adr + 1] = d.subtree_com.data[objid * 3 + 1]
            d.sensordata.data[adr + 2] = d.subtree_com.data[objid * 3 + 2]

        elif st == SENS_SUBTREELINVEL:
            var vx = 0.0
            var vy = 0.0
            var vz = 0.0
            subtree_linvel[DTYPE](d.xvel.data, m.bodies.data, nbody, objid,
                                  vx, vy, vz)
            d.sensordata.data[adr + 0] = Scalar[DTYPE](vx)
            d.sensordata.data[adr + 1] = Scalar[DTYPE](vy)
            d.sensordata.data[adr + 2] = Scalar[DTYPE](vz)

        elif st == SENS_ACCELEROMETER:
            # `*_acc`, not the live FK products — see the module note.
            var a = site_accelerometer[DTYPE](
                d.cvel.data, d.cacc.data, d.subtree_com.data,
                d.site_xpos_acc.data, d.xquat_acc.data, m.bodies.data, m.sites.data,
                body, objid,
            )
            d.sensordata.data[adr + 0] = Scalar[DTYPE](a[0])
            d.sensordata.data[adr + 1] = Scalar[DTYPE](a[1])
            d.sensordata.data[adr + 2] = Scalar[DTYPE](a[2])

        elif st == SENS_FORCE or st == SENS_TORQUE:
            # ⚠ ONE KERNEL, TWO SENSORS AGAIN: force first, then torque.
            var ft = site_force_torque[DTYPE](
                d.cfrc_int.data, d.subtree_com.data,
                d.site_xpos_acc.data, d.xquat_acc.data, m.bodies.data, m.sites.data,
                body, objid,
            )
            # Tuple again — see the velocimeter note above.
            if st == SENS_FORCE:
                d.sensordata.data[adr + 0] = Scalar[DTYPE](ft[0])
                d.sensordata.data[adr + 1] = Scalar[DTYPE](ft[1])
                d.sensordata.data[adr + 2] = Scalar[DTYPE](ft[2])
            else:
                d.sensordata.data[adr + 0] = Scalar[DTYPE](ft[3])
                d.sensordata.data[adr + 1] = Scalar[DTYPE](ft[4])
                d.sensordata.data[adr + 2] = Scalar[DTYPE](ft[5])

        elif st == SENS_TOUCH:
            # `scale` 1.0 is the MuJoCo-matching value; the parameter exists
            # for callers wanting impulses, and a sensor is not one of them.
            d.sensordata.data[adr] = Scalar[DTYPE](
                touch_sphere_site[DTYPE, D](d, m.sites.data, objid, 1.0)
            )

        _apply_cutoff[DTYPE](
            d.sensordata.data,
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
    DTYPE: DType, D: DimsLike
](mut d: Data[DTYPE, D, 1], mut m: Model[DTYPE, D]) raises:
    """`mj_sensorPos` — the position stage. Call after forward kinematics."""
    _eval_stage[DTYPE, D](d, m, SENSSTAGE_POS)


@always_inline
def sensor_vel[
    DTYPE: DType, D: DimsLike
](mut d: Data[DTYPE, D, 1], mut m: Model[DTYPE, D]) raises:
    """`mj_sensorVel` — the velocity stage. Call after body velocities."""
    _eval_stage[DTYPE, D](d, m, SENSSTAGE_VEL)


@always_inline
def sensor_acc[
    DTYPE: DType, D: DimsLike
](
    mut d: Data[DTYPE, D, 1],
    mut m: Model[DTYPE, D],
    have_rne_post: Bool = True,
) raises:
    """`mj_sensorAcc` — the acceleration stage.

    Call after the constraint solve and after `rne_post` has filled `cacc` /
    `cfrc_int`, at the point `EulerIntegrator.step` already evaluates the
    hand-written acceleration-stage hooks.
    """
    _eval_stage[DTYPE, D](d, m, SENSSTAGE_ACC, have_rne_post)
