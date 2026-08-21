"""Flat data structs and FlatModelDef for XML-driven model construction.

Promotes BodyData/JointData from test_flat_model_def.mojo prototype and adds
GeomData, ActuatorData, DefaultsData.  FlatModelDef[NBODY,NJOINT,NQ,NV,NGEOM,NACT]
holds the fully-populated data parsed from MJCF XML and can populate a Model
struct via setup_model().
"""

from ..types import ConeType, SolverType, IntegratorType
from std.collections import InlineArray
from mojo_rl.physics3d.joint_types import (
    JNT_HINGE,
    JNT_SLIDE,
    JNT_BALL,
    JNT_FREE,
)
# The single source for how many joints/sites one tendon may wrap. `TendonData`
# and the packed field layout must agree, so both read it from here.
from mojo_rl.physics3d.gpu.constants import (
    TENDON_MAX_WRAPS,
    TENDON_MAX_SPATIAL_WRAPS,
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
)


# =============================================================================
# BodyData
# =============================================================================


struct BodyData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime body data — replaces a compile-time BodySpec type."""

    var parent: Int
    var mass: Float64
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var quat_x: Float64
    var quat_y: Float64
    var quat_z: Float64
    var quat_w: Float64
    var ipos_x: Float64
    var ipos_y: Float64
    var ipos_z: Float64
    var iquat_x: Float64
    var iquat_y: Float64
    var iquat_z: Float64
    var iquat_w: Float64
    var ixx: Float64
    var iyy: Float64
    var izz: Float64
    var is_mocap: Bool  # True when <body mocap="true">
    var has_explicit_inertia: Bool  # True when body has mass/diaginertia/inertial

    def __init__(
        out self,
        parent: Int = 0,
        mass: Float64 = 1.0,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        quat_x: Float64 = 0.0,
        quat_y: Float64 = 0.0,
        quat_z: Float64 = 0.0,
        quat_w: Float64 = 1.0,
        ipos_x: Float64 = 0.0,
        ipos_y: Float64 = 0.0,
        ipos_z: Float64 = 0.0,
        iquat_x: Float64 = 0.0,
        iquat_y: Float64 = 0.0,
        iquat_z: Float64 = 0.0,
        iquat_w: Float64 = 1.0,
        ixx: Float64 = 0.01,
        iyy: Float64 = 0.01,
        izz: Float64 = 0.01,
        is_mocap: Bool = False,
        has_explicit_inertia: Bool = False,
    ):
        self.parent = parent
        self.mass = mass
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.quat_x = quat_x
        self.quat_y = quat_y
        self.quat_z = quat_z
        self.quat_w = quat_w
        self.ipos_x = ipos_x
        self.ipos_y = ipos_y
        self.ipos_z = ipos_z
        self.iquat_x = iquat_x
        self.iquat_y = iquat_y
        self.iquat_z = iquat_z
        self.iquat_w = iquat_w
        self.ixx = ixx
        self.iyy = iyy
        self.izz = izz
        self.is_mocap = is_mocap
        self.has_explicit_inertia = has_explicit_inertia


# =============================================================================
# JointData
# =============================================================================


struct JointData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime joint data — replaces a compile-time JointSpec type."""

    var jnt_type: Int  # JNT_HINGE=3, JNT_SLIDE=2, JNT_BALL=1, JNT_FREE=0
    var body_id: Int
    var nq: Int
    var nv: Int
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var axis_x: Float64
    var axis_y: Float64
    var axis_z: Float64
    var range_min: Float64
    var range_max: Float64
    var is_limited: Bool
    # `<joint actuatorfrcrange>` — MuJoCo's `jnt_actfrcrange`, the clamp
    # `mj_fwdActuation` applies to the ACCUMULATED `qfrc_actuator` at this
    # joint's dof address. A DIFFERENT limit from the actuator's own
    # `forcerange`, and on most models that declare it the only one.
    var actfrc_min: Float64
    var actfrc_max: Float64
    var is_actfrc_limited: Bool
    var armature: Float64
    var damping: Float64
    var stiffness: Float64
    # `<joint springdamper="timeconst dampratio">`. Both <= 0 means "absent",
    # which is MuJoCo's own test (`AutoSpringDamper` skips the joint unless
    # BOTH are strictly positive). When present, MuJoCo DERIVES `stiffness`
    # and `damping` from the body's own inertia and OVERWRITES whatever the
    # XML or the class said — see `_apply_auto_spring_damper`.
    var springdamper_0: Float64
    var springdamper_1: Float64
    var springref: Float64
    var frictionloss: Float64
    var ref_val: Float64  # MuJoCo joint ref (zero-position offset for qpos0)
    var solref_limit_0: Float64  # -1.0 = use model default
    var solref_limit_1: Float64  # -1.0 = use model default
    var solimp_limit_0: Float64  # -1.0 = use model default
    var solimp_limit_1: Float64  # -1.0 = use model default
    var solimp_limit_2: Float64  # -1.0 = use model default
    var solimp_limit_3: Float64  # -1.0 = use model default
    var solimp_limit_4: Float64  # -1.0 = use model default
    # MJCF `solreffriction` / `solimpfriction` — the dof-FRICTION solver
    # parameters, a DIFFERENT pair from the limit ones above (MuJoCo keeps
    # them in dof_solref/dof_solimp). Not plumbed; `friction_dof.mojo` uses
    # MuJoCo's defaults, which is exact for every model in the repo. This flag
    # lets `ModelDefFromXML.init_fields` raise if an XML ever sets them, so it
    # is loud rather than a silently wrong friction force.
    var has_friction_solparams: Bool

    def __init__(
        out self,
        jnt_type: Int = JNT_HINGE,
        body_id: Int = 1,
        nq: Int = 1,
        nv: Int = 1,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        # ⚠ MuJoCo's default joint axis is Z, not Y — `mjCJoint::mjCJoint`
        # sets `axis[0] = axis[1] = 0; axis[2] = 1` (user_objects.cc:3247),
        # confirmed on the 3.10.0 runtime with an axis-less `<joint
        # type="hinge"/>`, which compiles to `jnt_axis = [0, 0, 1]`.
        #
        # This was Y until 2026-08-13, and `full_parser` only assigns an axis
        # when the element OR its default class supplies one — so ANY joint
        # relying on the MuJoCo default silently became a hinge about the
        # WRONG AXIS. It hid because almost every hand-written MJCF spells
        # `axis` out; Jaco's `<joint name="jaco_arm/joint_1"
        # class="jaco_arm/large_joint"/>` does not, and neither does its
        # class. The visible symptom was forward kinematics: body b_1 exact,
        # every body below it wrong by up to 0.154 m, because a body's own
        # orientation only reaches its CHILDREN.
        axis_x: Float64 = 0.0,
        axis_y: Float64 = 0.0,
        axis_z: Float64 = 1.0,
        range_min: Float64 = -1e10,
        range_max: Float64 = 1e10,
        is_limited: Bool = False,
        actfrc_min: Float64 = 0.0,
        actfrc_max: Float64 = 0.0,
        is_actfrc_limited: Bool = False,
        armature: Float64 = 0.0,
        damping: Float64 = 0.0,
        stiffness: Float64 = 0.0,
        springdamper_0: Float64 = 0.0,
        springdamper_1: Float64 = 0.0,
        springref: Float64 = 0.0,
        frictionloss: Float64 = 0.0,
        ref_val: Float64 = 0.0,
        solref_limit_0: Float64 = -1.0,
        solref_limit_1: Float64 = -1.0,
        solimp_limit_0: Float64 = -1.0,
        solimp_limit_1: Float64 = -1.0,
        solimp_limit_2: Float64 = -1.0,
        solimp_limit_3: Float64 = -1.0,
        solimp_limit_4: Float64 = -1.0,
        has_friction_solparams: Bool = False,
    ):
        self.jnt_type = jnt_type
        self.body_id = body_id
        self.nq = nq
        self.nv = nv
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z
        self.range_min = range_min
        self.range_max = range_max
        self.is_limited = is_limited
        self.actfrc_min = actfrc_min
        self.actfrc_max = actfrc_max
        self.is_actfrc_limited = is_actfrc_limited
        self.armature = armature
        self.damping = damping
        self.stiffness = stiffness
        self.springdamper_0 = springdamper_0
        self.springdamper_1 = springdamper_1
        self.springref = springref
        self.frictionloss = frictionloss
        self.ref_val = ref_val
        self.solref_limit_0 = solref_limit_0
        self.solref_limit_1 = solref_limit_1
        self.solimp_limit_0 = solimp_limit_0
        self.solimp_limit_1 = solimp_limit_1
        self.solimp_limit_2 = solimp_limit_2
        self.solimp_limit_3 = solimp_limit_3
        self.solimp_limit_4 = solimp_limit_4
        self.has_friction_solparams = has_friction_solparams


# =============================================================================
# GeomData
# =============================================================================

# Geometry type constants (matches physics3d/constants.mojo)
comptime _GEOM_PLANE: Int = 0
comptime _GEOM_SPHERE: Int = 1
comptime _GEOM_CAPSULE: Int = 2
comptime _GEOM_BOX: Int = 3
comptime _GEOM_CYLINDER: Int = 4
comptime _GEOM_MESH: Int = 5
comptime _GEOM_ELLIPSOID: Int = 6


struct GeomData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime geom data."""

    var body_id: Int  # 0 = worldbody/static, >=1 = body-attached
    var geom_type: Int  # _GEOM_PLANE/SPHERE/CAPSULE/BOX/CYLINDER
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var quat_x: Float64
    var quat_y: Float64
    var quat_z: Float64
    var quat_w: Float64
    var radius: Float64
    var half_length: Float64  # capsule half-length along axis
    var half_x: Float64  # box half-extents
    var half_y: Float64
    var half_z: Float64
    var friction: Float64
    var friction_spin: Float64
    var friction_roll: Float64
    var contype: Int
    var conaffinity: Int
    var condim: Int
    var solref_0: Float64
    var solref_1: Float64
    var solimp_0: Float64
    var solimp_1: Float64
    var solimp_2: Float64
    var solimp_3: Float64
    var solimp_4: Float64
    var margin: Float64
    var density: Float64  # kg/m³; used when mass=-1 to compute mass from volume
    var mass: Float64  # -1.0 = use density (not specified explicitly)
    var rgba_r: Float64  # visual colour (r component, 0..1)
    var rgba_g: Float64
    var rgba_b: Float64
    var rgba_a: Float64
    var material_id: Int  # index into FlatModelDef.materials[], -1 if none
    var group: Int  # geom group (0-5), used for inertiagrouprange filtering
    var priority: Int  # `<geom priority>`; higher wins ALL contact params
    # `<geom mesh="...">` on a geom whose TYPE is a primitive. MuJoCo fits
    # the primitive to that mesh's inertia box and then drops the mesh
    # reference, so the result is a sphere/capsule/box sized from the mesh —
    # NOT a mesh geom, and NOT a default-sized primitive. See the fit in
    # `fields_build`.
    var fit_from_mesh: Bool
    var mesh_id: Int  # index into mesh hull data (-1 if not mesh geom)
    var mesh_filename: String  # STL filename for mesh geoms ("" if not mesh)
    var mesh_inertia_shell: Bool
    """`<mesh inertia="shell">` on the asset this geom names."""
    var mesh_ref_pos_x: Float64
    var mesh_ref_pos_y: Float64
    var mesh_ref_pos_z: Float64
    var mesh_ref_quat_w: Float64
    var mesh_ref_quat_x: Float64
    var mesh_ref_quat_y: Float64
    var mesh_ref_quat_z: Float64
    """`<mesh refpos>` / `<mesh refquat>` for the asset this geom names — see
    `FlatModelDef.mesh_asset_refpos`. Identity when the asset declares
    neither, which is 84 of Menagerie's 85 scenes."""
    var mesh_scale_x: Float64
    var mesh_scale_y: Float64
    var mesh_scale_z: Float64
    """`<mesh scale>` of the ASSET this geom names — 1,1,1 when unset.

    ⚠ CARRIED ON THE GEOM, NOT LOOKED UP AT LOAD TIME, because the loader is
    handed a FILENAME and nothing else, and two geoms may name two assets that
    share one file at different scales (a mirrored left/right pair is exactly
    that). Resolving it here, where `mesh_filename` is resolved, keeps the two
    from disagreeing."""
    # `material="..."` AFTER the `<default>`/`childclass` chain is applied,
    # and whether the geom (or its class) stated a colour of its own. Both are
    # parse-time bookkeeping for `_resolve_geom_materials`, which turns the
    # name into `material_id` and applies the material's colour only where
    # `has_own_rgba` is False. They live on the record because the class chain
    # is only in scope during the worldbody walk, and the post-pass that needs
    # the answer runs long after it. See `_parse_one_geom` for what reading
    # the tag alone cost.
    var material_name: String
    var has_own_rgba: Bool
    var has_explicit_mass: Bool
    """True when the SOURCE wrote `mass=`, rather than mass being density x
    volume.

    ⚠ THE STUDIO NEEDS THIS TO WRITE A MASS EDIT BACK. On a model with
    `<compiler inertiafromgeom="true">` a body's mass comes from its geoms and
    an `<inertial>` is IGNORED — by MuJoCo too — so the only expressible
    override is on the geoms. Which attribute to write depends on which one
    the file used: `density=` is silently overridden by an existing `mass=`.
    Guessing from the numbers would need the volume and would be wrong on a
    mesh."""

    def __init__(
        out self,
        body_id: Int = 0,
        geom_type: Int = _GEOM_SPHERE,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        quat_x: Float64 = 0.0,
        quat_y: Float64 = 0.0,
        quat_z: Float64 = 0.0,
        quat_w: Float64 = 1.0,
        radius: Float64 = 0.5,
        half_length: Float64 = 0.5,
        half_x: Float64 = 0.5,
        half_y: Float64 = 0.5,
        half_z: Float64 = 0.5,
        friction: Float64 = 0.5,
        friction_spin: Float64 = 0.005,
        friction_roll: Float64 = 0.0001,
        contype: Int = 1,
        conaffinity: Int = 1,
        condim: Int = 3,
        solref_0: Float64 = 0.02,
        solref_1: Float64 = 1.0,
        solimp_0: Float64 = 0.9,
        solimp_1: Float64 = 0.95,
        solimp_2: Float64 = 0.001,
        solimp_3: Float64 = 0.5,
        solimp_4: Float64 = 2.0,
        margin: Float64 = 0.0,
        density: Float64 = 1000.0,
        mass: Float64 = -1.0,
        # ⚠ MuJoCo's geom `rgba` default is "0.5 0.5 0.5 1"
        # (XMLreference.rst:2632), NOT 0.7 grey. Ours was 0.7, so every geom
        # carrying neither an `rgba` nor a `material` drew lighter than
        # MuJoCo draws it. Found by the MuJoCo parity gate in 1a.5c; the
        # consistency gate could not see it because BOTH our parsers used
        # 0.7 and agreed.
        #
        # ⚠⚠ THIS VALUE IS ALSO A SENTINEL, not just a fallback. MuJoCo
        # applies a material's colour unless the geom's own rgba DIFFERS FROM
        # THIS DEFAULT (XMLreference.rst:2623), so `_resolve_geom_materials`
        # tests against it. Changing it silently changes that rule.
        rgba_r: Float64 = 0.5,
        rgba_g: Float64 = 0.5,
        rgba_b: Float64 = 0.5,
        rgba_a: Float64 = 1.0,
        material_id: Int = -1,
        group: Int = 0,
        mesh_id: Int = -1,
        mesh_filename: String = "",
        material_name: String = "",
        has_own_rgba: Bool = False,
        has_explicit_mass: Bool = False,
    ):
        self.material_name = material_name
        self.has_own_rgba = has_own_rgba
        self.has_explicit_mass = has_explicit_mass
        self.body_id = body_id
        self.geom_type = geom_type
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.quat_x = quat_x
        self.quat_y = quat_y
        self.quat_z = quat_z
        self.quat_w = quat_w
        self.radius = radius
        self.half_length = half_length
        self.half_x = half_x
        self.half_y = half_y
        self.half_z = half_z
        self.friction = friction
        self.friction_spin = friction_spin
        self.friction_roll = friction_roll
        self.contype = contype
        self.conaffinity = conaffinity
        self.condim = condim
        self.solref_0 = solref_0
        self.solref_1 = solref_1
        self.solimp_0 = solimp_0
        self.solimp_1 = solimp_1
        self.solimp_2 = solimp_2
        self.solimp_3 = solimp_3
        self.solimp_4 = solimp_4
        self.margin = margin
        self.density = density
        self.mass = mass
        self.rgba_r = rgba_r
        self.rgba_g = rgba_g
        self.rgba_b = rgba_b
        self.rgba_a = rgba_a
        self.material_id = material_id
        self.group = group
        self.priority = 0  # MuJoCo default; set by the parser when declared
        self.fit_from_mesh = False
        self.mesh_id = mesh_id
        self.mesh_filename = mesh_filename
        self.mesh_inertia_shell = False
        self.mesh_ref_pos_x = 0.0
        self.mesh_ref_pos_y = 0.0
        self.mesh_ref_pos_z = 0.0
        self.mesh_ref_quat_w = 1.0
        self.mesh_ref_quat_x = 0.0
        self.mesh_ref_quat_y = 0.0
        self.mesh_ref_quat_z = 0.0
        self.mesh_scale_x = 1.0
        self.mesh_scale_y = 1.0
        self.mesh_scale_z = 1.0


# =============================================================================
# ActuatorData
# =============================================================================


# Actuator transmission/gain kinds. Only ACT_KIND_MOTOR (direct force =
# gear * ctrl) is implemented by the engine; the rest are RECOGNIZED by the
# parser purely so the model build can reject them loudly instead of
# simulating a servo as a torque motor. See `_fill_actuators`.
comptime ACT_KIND_MOTOR: Int = 0
comptime ACT_KIND_POSITION: Int = 1
comptime ACT_KIND_VELOCITY: Int = 2
comptime ACT_KIND_GENERAL: Int = 3


def act_kind_name(kind: Int) -> String:
    """Human-readable MJCF tag for an ACT_KIND_* value (error messages)."""
    if kind == ACT_KIND_POSITION:
        return "<position>"
    if kind == ACT_KIND_VELOCITY:
        return "<velocity>"
    if kind == ACT_KIND_GENERAL:
        return "<general>"
    return "<motor>"


struct ActuatorData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime actuator data parsed from <motor/position/velocity> tags."""

    var joint_id: Int  # 0-based joint index this actuator drives
    var gear: Float64  # Force/torque scaling
    var ctrl_min: Float64
    var ctrl_max: Float64
    var is_ctrl_limited: Bool
    # ACT_KIND_*. This struct carries no gainprm/biasprm — the gains live in
    # `xml_parser`'s comptime `ComptimeActData` (`motor_kp`/`motor_kv`), which
    # is what `apply_actions` reads. The kind is here so `init_fields` can
    # refuse a transmission neither parser models rather than silently run it
    # as a torque motor.
    var kind: Int
    # forcerange / forcelimited. Phase 1a.1 — previously these lived ONLY on
    # the comptime `ComptimeActData`. Semantics mirrored from it exactly:
    # `forcelimited` defaults to "auto" = limited iff the range is defined,
    # and `"0 0"` IS the undefined marker (measured — an explicit
    # `forcerange="0 0"` still reports forcelimited 0).
    var force_limited: Bool
    var force_min: Float64
    var force_max: Float64
    # mjDYN_FILTER: act_dot = (ctrl - act) / dynprm[0]. `dyn_tau > 0` means
    # this actuator owns ONE activation variable and `act_adr` is its index;
    # otherwise dyn_tau is 0 and act_adr is -1.
    # ⚠ ONLY `<general>` HONOURS `dyntype` HERE, mirroring the comptime twin
    # (`xml_parser.mojo:4292` `elif is_general:` encloses the whole block).
    # MJCF permits `dyntype` on <position>/<velocity> too; if that ever needs
    # supporting it must change on BOTH parsers at once, or this record and
    # `_acd` silently disagree.
    var dyn_tau: Float64
    var act_adr: Int
    # Transmission. Mirrors `ComptimeActData.motor_trn_*`
    # (`xml_parser.mojo:4381`): a `joint=` actuator is ONE (qadr, dadr, 1.0)
    # triple with `trn_n = 1`; a `tendon=` actuator copies the named tendon's
    # whole wrap list and takes `dof_adr` from its FIRST wrap. `dof_adr` is
    # the single dof the actuator is reported against.
    #
    # ⚠ FILLED BY A SECOND PASS, `_fill_actuator_transmission`, because
    # `_fill_actuators` runs BEFORE `_fill_tendons` and the tendon branch
    # needs `result.tendons` populated. `tendon_id` is resolved in the first
    # pass (off the `<tendon>` SECTION TEXT, which does exist by then) so the
    # second pass needs no re-scan of the actuator tags.
    # ⚠ THE WRAP ARRAYS LIVE ON `FlatModelDef`, NOT HERE. `ActuatorData` is
    # `ImplicitlyCopyable` and `InlineArray` is not, so inline wrap arrays
    # cannot synthesize a copy constructor. Flat `motor_trn_*` lists on the
    # parent, indexed `ai * TENDON_MAX_WRAPS + k`, also match the comptime
    # twin's own layout (`motor_trn_qadr[act_count * WRAPS + k]`) exactly.
    var tendon_id: Int
    var dof_adr: Int
    var trn_n: Int
    # Servo gains. MuJoCo `gainprm[0]` and `-biasprm[2]`.
    #   MOTOR     force = kp * u                 (kp is the bare gain)
    #   POSITION  force = kp * (u - length) - kv * vel
    #   VELOCITY  force = kp * u            - kv * vel
    # ⚠ kp and kv are INDEPENDENT. `<velocity>` happens to set both to K, but
    # `gainprm="5 0 0" biasprm="0 0 -3"` is legal and means
    # `force = 5*u - 3*vel`. Do not collapse them.
    var unsupported_transmission: Bool
    """True when the actuator names a transmission this engine does not model.

    ⚠⚠ MuJoCo drives actuators through SITES, BODIES and slider-cranks as well
    as joints and tendons; this engine implements the last two. Such an
    actuator resolves to `joint_id = -1, tendon_id = -1` — identical to an
    actuator with NO transmission, which MuJoCo refuses outright. Without this
    flag `studio.validate` reported `bitcraze_crazyflie_2`'s four rotor
    actuators as "MuJoCo refuses this model", on a model MuJoCo loads. The two
    cases need different words: one is a broken file, the other is a gap here.
    """

    var kp: Float64
    var kv: Float64
    var dampratio: Float64
    """`<position dampratio="X">` — a kv MuJoCo DERIVES, 0 when absent.

    ⚠ NOT A kv AND NOT INTERCHANGEABLE WITH ONE. MuJoCo carries it in the
    same slot as kv under a SIGN convention (`user_api.cc:1211`: negative is a
    literal kv, positive is a pending dampratio) and converts it in
    `mj_setConst` once the mass matrix at qpos0 exists:

        mass = sum over the transmission dofs of dof_M0[dof] / gear^2
        kv   = dampratio * 2 * sqrt(kp * mass)

    so it cannot be resolved by the parser — the reflected inertia is not
    known until the model is built. `build_actuator_damping` does it."""

    def __init__(
        out self,
        joint_id: Int = -1,
        gear: Float64 = 1.0,
        ctrl_min: Float64 = -1.0,
        ctrl_max: Float64 = 1.0,
        is_ctrl_limited: Bool = False,
        kind: Int = ACT_KIND_MOTOR,
    ):
        self.joint_id = joint_id
        self.gear = gear
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max
        self.is_ctrl_limited = is_ctrl_limited
        self.kind = kind
        # Not ctor params — set by `_fill_actuators`, like the structural
        # attrs on `DefaultsData`, so no existing call site changes.
        self.force_limited = False
        self.force_min = 0.0
        self.force_max = 0.0
        self.dyn_tau = 0.0
        self.act_adr = -1
        self.tendon_id = -1
        self.dof_adr = -1
        self.trn_n = 0
        self.unsupported_transmission = False
        # ⚠ kp DEFAULTS TO 1.0, NOT 0.0 — MuJoCo's `gainprm[0]` default, and
        # the comptime twin inits the same way (`fill=1.0`, xml_parser:3204).
        # `apply_actions` computes `force = kp * u` for EVERY kind, including
        # MOTOR, so a 0.0 default silently produces ZERO FORCE on every plain
        # `<motor>` — cartpole, half_cheetah, ant, the lot. There is no
        # `<motor>` branch to set it; the default IS the value.
        self.kp = 1.0
        self.kv = 0.0
        self.dampratio = 0.0


# =============================================================================
# TextureData
# =============================================================================

# Texture type constants
comptime TEX_SKYBOX: Int = 0
comptime TEX_2D: Int = 1
comptime TEX_CUBE: Int = 2

# Texture builtin constants
comptime TEX_BUILTIN_NONE: Int = 0
comptime TEX_BUILTIN_GRADIENT: Int = 1
comptime TEX_BUILTIN_CHECKER: Int = 2
comptime TEX_BUILTIN_FLAT: Int = 3

# Texture mark constants
comptime TEX_MARK_NONE: Int = 0
comptime TEX_MARK_EDGE: Int = 1
comptime TEX_MARK_CROSS: Int = 2
comptime TEX_MARK_RANDOM: Int = 3


struct TextureData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime texture data parsed from <texture> in <asset>."""

    var tex_type: Int  # TEX_SKYBOX / TEX_2D / TEX_CUBE
    var builtin: Int  # TEX_BUILTIN_* — procedural texture pattern
    var mark: Int  # TEX_MARK_* — overlay mark type
    # `name` / `file` — the ASSET IDENTITY, added in phase 1a.5 because the
    # render path needs both and neither was recorded here. `<material
    # texture="grid">` is resolved to a `tex_id` by name at parse time, but
    # `ModelDefFromXML.get_skybox_colors` and the PNG loader look the texture
    # up by NAME at render time, and a `file=` texture is loaded from disk —
    # so dropping these two makes the runtime record unable to drive the
    # renderer even though every colour field is present. `_rcd.tex_names` /
    # `tex_files` are what they replace.
    var name: String
    var file: String
    var rgb1_r: Float64
    var rgb1_g: Float64
    var rgb1_b: Float64  # primary colour (background / gradient start)
    var rgb2_r: Float64
    var rgb2_g: Float64
    var rgb2_b: Float64  # secondary colour (gradient end / checker colour 2)
    var markrgb_r: Float64
    var markrgb_g: Float64
    var markrgb_b: Float64  # mark colour
    var width: Int
    var height: Int
    var random: Float64  # random mark density (0..1)

    def __init__(
        out self,
        tex_type: Int = TEX_2D,
        builtin: Int = TEX_BUILTIN_NONE,
        mark: Int = TEX_MARK_NONE,
        rgb1_r: Float64 = 0.8,
        rgb1_g: Float64 = 0.8,
        rgb1_b: Float64 = 0.8,
        rgb2_r: Float64 = 0.5,
        rgb2_g: Float64 = 0.5,
        rgb2_b: Float64 = 0.5,
        markrgb_r: Float64 = 0.0,
        markrgb_g: Float64 = 0.0,
        markrgb_b: Float64 = 0.0,
        width: Int = 512,
        height: Int = 512,
        random: Float64 = 0.01,
        name: String = "",
        file: String = "",
    ):
        self.tex_type = tex_type
        self.builtin = builtin
        self.mark = mark
        self.name = name
        self.file = file
        self.rgb1_r = rgb1_r
        self.rgb1_g = rgb1_g
        self.rgb1_b = rgb1_b
        self.rgb2_r = rgb2_r
        self.rgb2_g = rgb2_g
        self.rgb2_b = rgb2_b
        self.markrgb_r = markrgb_r
        self.markrgb_g = markrgb_g
        self.markrgb_b = markrgb_b
        self.width = width
        self.height = height
        self.random = random


# =============================================================================
# MaterialData
# =============================================================================


struct MaterialData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime material data parsed from <material> in <asset>."""

    var tex_id: Int  # index into textures[], -1 if no texture
    var rgba_r: Float64
    var rgba_g: Float64
    var rgba_b: Float64
    var rgba_a: Float64
    var shininess: Float64
    var specular: Float64
    var reflectance: Float64
    var texrepeat_u: Float64
    var texrepeat_v: Float64
    var texuniform: Bool  # tile texture uniformly across surface

    def __init__(
        out self,
        tex_id: Int = -1,
        rgba_r: Float64 = 1.0,
        rgba_g: Float64 = 1.0,
        rgba_b: Float64 = 1.0,
        rgba_a: Float64 = 1.0,
        shininess: Float64 = 0.5,
        specular: Float64 = 0.5,
        reflectance: Float64 = 0.0,
        texrepeat_u: Float64 = 1.0,
        texrepeat_v: Float64 = 1.0,
        texuniform: Bool = False,
    ):
        self.tex_id = tex_id
        self.rgba_r = rgba_r
        self.rgba_g = rgba_g
        self.rgba_b = rgba_b
        self.rgba_a = rgba_a
        self.shininess = shininess
        self.specular = specular
        self.reflectance = reflectance
        self.texrepeat_u = texrepeat_u
        self.texrepeat_v = texrepeat_v
        self.texuniform = texuniform


# =============================================================================
# LightData
# =============================================================================

# Light mode constants
comptime LIGHT_MODE_FIXED: Int = 0
comptime LIGHT_MODE_TRACK: Int = 1
comptime LIGHT_MODE_TRACKCOM: Int = 2
comptime LIGHT_MODE_TARGETBODY: Int = 3
comptime LIGHT_MODE_TARGETBODYCOM: Int = 4


struct LightData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime light data parsed from <light> in <worldbody>."""

    var body_id: Int  # body this light is attached to (0 = worldbody)
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var dir_x: Float64
    var dir_y: Float64
    var dir_z: Float64
    var diffuse_r: Float64
    var diffuse_g: Float64
    var diffuse_b: Float64
    var specular_r: Float64
    var specular_g: Float64
    var specular_b: Float64
    var ambient_r: Float64
    var ambient_g: Float64
    var ambient_b: Float64
    var directional: Bool  # true = directional (infinite), false = point/spot
    var castshadow: Bool
    var cutoff: Float64  # spot cone half-angle in degrees (100 = point light)
    var exponent: Float64  # spot exponent
    var mode: Int  # LIGHT_MODE_*

    def __init__(
        out self,
        body_id: Int = 0,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        dir_x: Float64 = 0.0,
        dir_y: Float64 = 0.0,
        dir_z: Float64 = -1.0,
        diffuse_r: Float64 = 0.7,
        diffuse_g: Float64 = 0.7,
        diffuse_b: Float64 = 0.7,
        specular_r: Float64 = 0.3,
        specular_g: Float64 = 0.3,
        specular_b: Float64 = 0.3,
        ambient_r: Float64 = 0.0,
        ambient_g: Float64 = 0.0,
        ambient_b: Float64 = 0.0,
        directional: Bool = False,
        castshadow: Bool = True,
        cutoff: Float64 = 45.0,
        exponent: Float64 = 10.0,
        mode: Int = LIGHT_MODE_FIXED,
    ):
        self.body_id = body_id
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z
        self.diffuse_r = diffuse_r
        self.diffuse_g = diffuse_g
        self.diffuse_b = diffuse_b
        self.specular_r = specular_r
        self.specular_g = specular_g
        self.specular_b = specular_b
        self.ambient_r = ambient_r
        self.ambient_g = ambient_g
        self.ambient_b = ambient_b
        self.directional = directional
        self.castshadow = castshadow
        self.cutoff = cutoff
        self.exponent = exponent
        self.mode = mode


# =============================================================================
# CameraData
# =============================================================================

# Camera mode constants
comptime CAM_MODE_FIXED: Int = 0
comptime CAM_MODE_TRACK: Int = 1
comptime CAM_MODE_TRACKCOM: Int = 2
comptime CAM_MODE_TARGETBODY: Int = 3
comptime CAM_MODE_TARGETBODYCOM: Int = 4


struct CameraData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime camera data parsed from <camera> in <worldbody>."""

    var body_id: Int  # body this camera is attached to (0 = worldbody)
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var quat_x: Float64
    var quat_y: Float64
    var quat_z: Float64
    var quat_w: Float64
    var fovy: Float64  # vertical field of view in degrees
    var ipd: Float64  # interpupillary distance (stereo)
    var mode: Int  # CAM_MODE_*
    # Body a `mode="targetbody"` camera aims at, resolved from `target="..."`
    # at parse time; -1 when it has none. Resolved HERE and not at render time
    # because the re-aim runs every frame and must not do string work — the
    # same reason `_rcd.cam_target_body` exists, which this replaces. Without
    # it a `targetbody` camera has a mode and nothing to point at.
    var target_body: Int

    def __init__(
        out self,
        body_id: Int = 0,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        quat_x: Float64 = 0.0,
        quat_y: Float64 = 0.0,
        quat_z: Float64 = 0.0,
        quat_w: Float64 = 1.0,
        fovy: Float64 = 45.0,
        ipd: Float64 = 0.068,
        mode: Int = CAM_MODE_FIXED,
        target_body: Int = -1,
    ):
        self.body_id = body_id
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.quat_x = quat_x
        self.quat_y = quat_y
        self.quat_z = quat_z
        self.quat_w = quat_w
        self.fovy = fovy
        self.ipd = ipd
        self.mode = mode
        self.target_body = target_body


# =============================================================================
# SiteData
# =============================================================================


struct SiteData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime site data parsed from <site> in <worldbody>."""

    var body_id: Int
    var site_type: Int  # geom-like type (_GEOM_SPHERE / _GEOM_CAPSULE / etc.)
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var quat_x: Float64
    var quat_y: Float64
    var quat_z: Float64
    var quat_w: Float64
    var size_0: Float64  # radius (sphere/capsule/cylinder) or half-x (box)
    var size_1: Float64  # half-length (capsule/cylinder) or half-y (box)
    var size_2: Float64  # half-z (box only)

    def __init__(
        out self,
        body_id: Int = 0,
        site_type: Int = 1,  # _GEOM_SPHERE
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        quat_x: Float64 = 0.0,
        quat_y: Float64 = 0.0,
        quat_z: Float64 = 0.0,
        quat_w: Float64 = 1.0,
        size_0: Float64 = 0.005,
        size_1: Float64 = 0.0,
        size_2: Float64 = 0.0,
    ):
        self.body_id = body_id
        self.site_type = site_type
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.quat_x = quat_x
        self.quat_y = quat_y
        self.quat_z = quat_z
        self.quat_w = quat_w
        self.size_0 = size_0
        self.size_1 = size_1
        self.size_2 = size_2


# =============================================================================
# TendonData
# =============================================================================


# Mirror of TENDON_KIND_* in physics3d/gpu/constants.mojo, kept local so the
# parser does not depend on the GPU record layout (same pattern as _EQ_*).
comptime _TENDON_KIND_FIXED: Int = 0
comptime _TENDON_KIND_SPATIAL: Int = 1


struct TendonData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime tendon data parsed from <tendon><fixed> / <spatial>.

    A FIXED tendon is a linear combination of joint coordinates
    (`length = sum coef_i * qpos[joint_i]`); a SPATIAL one is a polyline
    through sites (`length = sum |p_{k+1} - p_k|`). `kind` selects which half
    of this record is meaningful — they are never both populated.

    A spatial tendon routes through a SEQUENCE of waypoints, each either a
    site or a wrap geom (sphere/cylinder) with an optional sidesite — MuJoCo's
    `wrap_type`/`wrap_objid`/`wrap_prm` triple. `<pulley>` is still rejected
    by the parser rather than skipped, because silently dropping a waypoint
    would shorten the tendon and surface only as a physics divergence.
    """

    var kind: Int  # _TENDON_KIND_FIXED / _TENDON_KIND_SPATIAL
    var is_equality: Int  # 1 only when <equality><tendon> names it

    # fixed
    var num_joints: Int
    var joint_ids: InlineArray[Int, TENDON_MAX_WRAPS]
    var coefs: InlineArray[Float64, TENDON_MAX_WRAPS]
    var length_ref: Float64
    # `<fixed stiffness=>` and `<fixed springlength=>`. Mirrors
    # `ComptimeActData.tendon_stiffness` / `_spring_lo` / `_spring_hi`
    # (`xml_parser.mojo:3994`, `:4045-4057`).
    # ⚠ WHEN `springlength` IS ABSENT BOTH BOUNDS DEFAULT TO `length0`, the
    # tendon's rest length `sum(coef * joint.ref)` — NOT to zero. A zero
    # default would make the deadband spring pull toward a length the tendon
    # never has. fish is the only model in the tree that declares a tendon
    # spring at all (two `<fixed stiffness="1e-4">`).
    var stiffness: Float64
    var spring_lo: Float64
    var spring_hi: Float64

    # spatial — the routing sequence, MuJoCo's wrap_type/wrap_objid/wrap_prm
    var num_wraps: Int
    # site id when `wrap_types[k] == WRAP_SITE`, geom id for a wrap object.
    var wrap_objs: InlineArray[Int, TENDON_MAX_SPATIAL_WRAPS]
    # `wrap.WRAP_SITE` / `WRAP_SPHERE` / `WRAP_CYLINDER`.
    var wrap_types: InlineArray[Int, TENDON_MAX_SPATIAL_WRAPS]
    # sidesite id for a wrap geom, -1 when there is none.
    var wrap_sides: InlineArray[Int, TENDON_MAX_SPATIAL_WRAPS]
    # `<spatial width= rgba=>` — RENDER ONLY, and here because the viewer's
    # `render_spatial_tendons` read them off the comptime `ComptimeRenderData`
    # (phase 1a.5). MuJoCo's defaults: width 0.003, rgba .5 .5 .5 1.
    var render_width: Float64
    var rgba_r: Float64
    var rgba_g: Float64
    var rgba_b: Float64
    var rgba_a: Float64

    # Wraps the XML declared beyond `TENDON_MAX_WRAPS`. Non-zero makes the
    # model build RAISE rather than run a silently truncated tendon — the
    # lesson of defect 17, where a bare `while n < 4` drove a third of dog's
    # tail joints and nothing said so.
    var wrap_overflow: Int

    # limit
    var limited: Int
    var range_min: Float64
    var range_max: Float64
    var margin: Float64
    var solref_lim_0: Float64
    var solref_lim_1: Float64
    var solimp_lim_0: Float64
    var solimp_lim_1: Float64
    var solimp_lim_2: Float64
    var solimp_lim_3: Float64
    var solimp_lim_4: Float64

    # equality (<equality><tendon tendon1="..."/>), distinct from the LIMIT
    # pair above — MuJoCo keeps eq_solref/eq_solimp on the equality, not on
    # the tendon. Only meaningful when `is_equality`; defaults are MuJoCo's
    # own (solref 0.02 1, solimp 0.9 0.95 0.001 0.5 2).
    var solref_eq_0: Float64
    var solref_eq_1: Float64
    var solimp_eq_0: Float64
    var solimp_eq_1: Float64
    var solimp_eq_2: Float64
    var solimp_eq_3: Float64
    var solimp_eq_4: Float64

    def __init__(out self, *, copy: Self):
        # Mojo 1.0: `Array` is no longer `ImplicitlyCopyable`, so the compiler
        # can no longer synthesise this struct's copy constructor. Written out
        # explicitly to keep the previous (implicitly copyable) semantics.
        self.kind = copy.kind
        self.is_equality = copy.is_equality
        self.num_joints = copy.num_joints
        self.joint_ids = copy.joint_ids.copy()
        self.coefs = copy.coefs.copy()
        self.length_ref = copy.length_ref
        self.stiffness = copy.stiffness
        self.spring_lo = copy.spring_lo
        self.spring_hi = copy.spring_hi
        self.num_wraps = copy.num_wraps
        self.render_width = copy.render_width
        self.rgba_r = copy.rgba_r
        self.rgba_g = copy.rgba_g
        self.rgba_b = copy.rgba_b
        self.rgba_a = copy.rgba_a
        self.wrap_objs = copy.wrap_objs.copy()
        self.wrap_types = copy.wrap_types.copy()
        self.wrap_sides = copy.wrap_sides.copy()
        self.wrap_overflow = copy.wrap_overflow
        self.limited = copy.limited
        self.range_min = copy.range_min
        self.range_max = copy.range_max
        self.margin = copy.margin
        self.solref_lim_0 = copy.solref_lim_0
        self.solref_lim_1 = copy.solref_lim_1
        self.solimp_lim_0 = copy.solimp_lim_0
        self.solimp_lim_1 = copy.solimp_lim_1
        self.solimp_lim_2 = copy.solimp_lim_2
        self.solimp_lim_3 = copy.solimp_lim_3
        self.solimp_lim_4 = copy.solimp_lim_4
        self.solref_eq_0 = copy.solref_eq_0
        self.solref_eq_1 = copy.solref_eq_1
        self.solimp_eq_0 = copy.solimp_eq_0
        self.solimp_eq_1 = copy.solimp_eq_1
        self.solimp_eq_2 = copy.solimp_eq_2
        self.solimp_eq_3 = copy.solimp_eq_3
        self.solimp_eq_4 = copy.solimp_eq_4

    def __init__(out self):
        self.kind = _TENDON_KIND_FIXED
        self.is_equality = 0
        self.num_joints = 0
        self.joint_ids = InlineArray[Int, TENDON_MAX_WRAPS](fill=-1)
        self.coefs = InlineArray[Float64, TENDON_MAX_WRAPS](fill=0.0)
        self.length_ref = 0.0
        self.stiffness = 0.0
        self.spring_lo = 0.0
        self.spring_hi = 0.0
        self.num_wraps = 0
        self.render_width = 0.003
        self.rgba_r = 0.5
        self.rgba_g = 0.5
        self.rgba_b = 0.5
        self.rgba_a = 1.0
        self.wrap_objs = InlineArray[Int, TENDON_MAX_SPATIAL_WRAPS](fill=-1)
        self.wrap_types = InlineArray[Int, TENDON_MAX_SPATIAL_WRAPS](fill=0)
        self.wrap_sides = InlineArray[Int, TENDON_MAX_SPATIAL_WRAPS](fill=-1)
        self.wrap_overflow = 0
        self.limited = 0
        self.range_min = 0.0
        self.range_max = 0.0
        self.margin = 0.0
        # MuJoCo model defaults (mjModel tendon_solref_lim / tendon_solimp_lim).
        self.solref_lim_0 = 0.02
        self.solref_lim_1 = 1.0
        self.solimp_lim_0 = 0.9
        self.solimp_lim_1 = 0.95
        self.solimp_lim_2 = 0.001
        self.solimp_lim_3 = 0.5
        self.solimp_lim_4 = 2.0
        # MuJoCo model defaults (mjModel eq_solref / eq_solimp).
        self.solref_eq_0 = 0.02
        self.solref_eq_1 = 1.0
        self.solimp_eq_0 = 0.9
        self.solimp_eq_1 = 0.95
        self.solimp_eq_2 = 0.001
        self.solimp_eq_3 = 0.5
        self.solimp_eq_4 = 2.0


# =============================================================================
# EqualityData
# =============================================================================


# Equality constraint type constants (matches physics3d/types.mojo)
comptime _EQ_CONNECT: Int = 0
comptime _EQ_WELD: Int = 1
comptime _EQ_JOINT: Int = 2  # mjEQ_JOINT — objects are JOINTS, data is polycoef

# Equality object semantics (matches EQ_OBJ_* in physics3d/types.mojo).
comptime _EQ_OBJ_BODY: Int = 0
comptime _EQ_OBJ_SITE: Int = 1


struct EqualityData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime equality constraint data parsed from <equality> section."""

    var eq_type: Int  # _EQ_CONNECT or _EQ_WELD
    # `_EQ_OBJ_BODY` or `_EQ_OBJ_SITE` — MuJoCo's `eq_objtype`.
    #
    # A SITE reference is stored REDUCED to the body form: `body_*` is the
    # site's body and `anchor_*` is the site's body-local `pos`, which is
    # exactly what `site_xpos` expands to. The flag is still carried because
    # the qpos0 anchor derivation in `compute_invweight0` must skip the site
    # form — MuJoCo zeroes `eq_data` for it and never derives an anchor.
    var objtype: Int
    var body_a: Int  # first body index
    var body_b: Int  # second body index (0 = worldbody)
    var anchor_a_x: Float64
    var anchor_a_y: Float64
    var anchor_a_z: Float64
    var anchor_b_x: Float64
    var anchor_b_y: Float64
    var anchor_b_z: Float64
    # Weld only: target relative orientation, (x, y, z, w).
    #
    # ⚠ AN ALL-ZERO QUATERNION IS THE "UNSET" MARKER, NOT A VALUE — it is
    # MJCF's own default (`relpose="0 0 0 0 0 0 0"`) and means "derive from the
    # qpos0 relative pose". `compute_invweight0` fills it in; anything reading
    # this before that runs sees a degenerate quaternion. `w` therefore
    # defaults to 0, NOT 1: an identity default is a real relative pose and
    # silently welds the bodies coincident. That is what it did until
    # 2026-08-12, and it was invisible because sawyer's mocap and hand ARE
    # coincident at qpos0.
    var relpose_x: Float64
    var relpose_y: Float64
    var relpose_z: Float64
    var relpose_w: Float64
    var solref_0: Float64
    var solref_1: Float64
    var solimp_0: Float64
    var solimp_1: Float64
    var solimp_2: Float64
    var solimp_3: Float64
    var solimp_4: Float64
    # Weld only: MuJoCo's `eq_data[10]`, scaling the three orientation rows.
    # MJCF's default is 1; MetaWorld's `reset_mocap_welds` sets 5.
    var torquescale: Float64

    def __init__(
        out self,
        eq_type: Int = _EQ_WELD,
        objtype: Int = _EQ_OBJ_BODY,
        body_a: Int = 0,
        body_b: Int = 0,
        anchor_a_x: Float64 = 0.0,
        anchor_a_y: Float64 = 0.0,
        anchor_a_z: Float64 = 0.0,
        anchor_b_x: Float64 = 0.0,
        anchor_b_y: Float64 = 0.0,
        anchor_b_z: Float64 = 0.0,
        relpose_x: Float64 = 0.0,
        relpose_y: Float64 = 0.0,
        relpose_z: Float64 = 0.0,
        relpose_w: Float64 = 0.0,  # all-zero = UNSET, see the field comment
        solref_0: Float64 = 0.02,
        solref_1: Float64 = 1.0,
        solimp_0: Float64 = 0.9,
        solimp_1: Float64 = 0.95,
        solimp_2: Float64 = 0.001,
        solimp_3: Float64 = 0.5,
        solimp_4: Float64 = 2.0,
        torquescale: Float64 = 1.0,
    ):
        self.eq_type = eq_type
        self.objtype = objtype
        self.body_a = body_a
        self.body_b = body_b
        self.anchor_a_x = anchor_a_x
        self.anchor_a_y = anchor_a_y
        self.anchor_a_z = anchor_a_z
        self.anchor_b_x = anchor_b_x
        self.anchor_b_y = anchor_b_y
        self.anchor_b_z = anchor_b_z
        self.relpose_x = relpose_x
        self.relpose_y = relpose_y
        self.relpose_z = relpose_z
        self.relpose_w = relpose_w
        self.solref_0 = solref_0
        self.solref_1 = solref_1
        self.solimp_0 = solimp_0
        self.solimp_1 = solimp_1
        self.solimp_2 = solimp_2
        self.solimp_3 = solimp_3
        self.solimp_4 = solimp_4
        self.torquescale = torquescale


# =============================================================================
# DefaultsData
# =============================================================================


struct DefaultsData(Copyable, ImplicitlyCopyable, Movable):
    """Parsed <default> block — applied when specific attrs are absent.

    Two kinds of field live here:

    * Tuning attributes (armature, damping, friction, solref, ...) are parsed
      into typed values, because the element parsers want them as numbers.
    * STRUCTURAL attributes (joint type/axis/range, geom type/fromto/size/
      mass/material/...) are kept as the RAW attribute string, "" meaning
      "not set by this class". The element parsers substitute the string
      before their own parsing runs, so unit conversion (deg->rad), `fromto`
      -> pos/quat decomposition and mesh resolution all keep working
      untouched, and inheritance from a parent class comes for free via the
      existing parent-merge in `_parse_one_default_block`.

    Structural inheritance was added 2026-07-29 for the dm_control suite,
    where a class can supply an element's ENTIRE definition — cartpole's
    `<geom name="pole_1"/>` gets type/fromto/size/material/mass from
    `<default class="pole">`. Before that, only tuning attributes inherited.
    """

    var joint_armature: Float64
    var joint_damping: Float64
    var joint_stiffness: Float64
    var joint_springdamper_0: Float64
    var joint_springdamper_1: Float64
    var joint_limited: Bool
    # `<joint actuatorfrcrange/actuatorfrclimited>` from a default
    # CLASS. Same three-field shape as `motor_force_*`, and resolved
    # by the same shared helper (`_apply_actfrcrange`).
    var joint_actfrc_min: Float64
    var joint_actfrc_max: Float64
    var joint_actfrc_limited: Bool
    var joint_frictionloss: Float64
    var joint_springref: Float64
    var joint_solref_limit_0: Float64
    var joint_solref_limit_1: Float64
    var joint_solimp_limit_0: Float64
    var joint_solimp_limit_1: Float64
    var joint_solimp_limit_2: Float64
    var joint_solimp_limit_3: Float64
    var joint_solimp_limit_4: Float64
    var geom_density: Float64
    var geom_friction: Float64
    var geom_friction_spin: Float64
    var geom_friction_roll: Float64
    var geom_contype: Int
    var geom_conaffinity: Int
    var geom_condim: Int
    # `<geom priority>` from a default CLASS. Read from the element
    # itself since quadruped (whose ball spells it inline), but the
    # class path was missing until dog — whose 42 teeth carry it
    # only via `<default class="tooth_primitive">`, so every one of
    # them resolved to priority 0 and lost its condim/friction/solref
    # override against the floor.
    var geom_priority: Int
    var geom_solref_0: Float64
    var geom_solref_1: Float64
    var geom_solimp_0: Float64
    var geom_solimp_1: Float64
    var geom_solimp_2: Float64
    var geom_solimp_3: Float64
    var geom_solimp_4: Float64
    var geom_margin: Float64
    var geom_rgba_r: Float64  # default geom colour (r, 0..1); -1.0 = not set
    var geom_rgba_g: Float64
    var geom_rgba_b: Float64
    var geom_rgba_a: Float64
    var motor_ctrl_limited: Bool
    var motor_ctrl_min: Float64
    var motor_ctrl_max: Float64
    var motor_gear: Float64
    # Phase 1a.1. Assignment-initialised like the structural attrs below.
    var motor_force_limited: Bool
    var motor_force_min: Float64
    var motor_force_max: Float64
    # Raw, like the structural attrs below ("" = not set by this class). dog
    # and quadruped both declare `dyntype="filter"` in a <default> block
    # rather than on the element, so the class path is the one that matters.
    var motor_dyntype_s: String
    var motor_dynprm_s: String
    # `<general>`/`<position>`/`<velocity>` gain attributes, raw. dog and
    # quadruped both declare gainprm/biasprm/biastype in a <default> block,
    # so the class path is the one that carries them.
    # ── The merged actuator default's GAIN and DAMPING BIAS ─────────────
    #
    # ⚠⚠ MuJoCo LAYERS EVERY ACTUATOR TAG IN A `<default>` BLOCK ONTO ONE
    # RECORD, IN DOCUMENT ORDER — it does NOT keep a separate default per tag
    # kind. `<position kp>` writes `gainprm[0] = kp` and `biasprm[2] = -kv`;
    # `<velocity kv>` writes BOTH `gainprm[0] = kv` and `biasprm[2] = -kv`;
    # `<general gainprm/biasprm>` writes them directly. Whichever tag comes
    # LAST wins per field, and the element that later inherits does not care
    # which tag kind supplied it.
    #
    # ⚠ THAT PRODUCES GENUINELY SURPRISING VALUES, and rby1 is the proof:
    # its block is `<motor .../><velocity ctrllimited="true"/><position
    # kp="4000" kv="400"/>`, so the merged gain is 4000 — and its two
    # `<velocity>` WHEEL actuators, which state no `kv`, compile to
    # `kv = 4000`, inherited from a `<position>` tag's `kp`. Measured on the
    # 3.10.0 runtime: `left_wheel_act` gainprm[0] 4000, biasprm [0, 0, -4000].
    #
    # `*_set` is "did any tag state it", which is what lets an element fall
    # back to MuJoCo's own base default (gain 1, bias2 0) rather than to a
    # value no tag wrote.
    var motor_gain: Float64
    var motor_gain_set: Bool
    var motor_bias2: Float64
    var motor_bias2_set: Bool
    var motor_kp_s: String
    # `<position inheritrange>` — see `_fill_actuators`. Carried as raw text so
    # an absent attribute inherits the parent class rather than resetting it,
    # exactly like `motor_kp_s`.
    var motor_inheritrange_s: String
    var motor_kv_s: String
    var motor_dampratio_s: String
    var mesh_scale_s: String
    """`<default><mesh scale="x y z"/></default>` — raw, "" when unset.

    ⚠ THE ASSET TABLE IS WHERE THIS IS CONSUMED, not a body. `<mesh>` is an
    ASSET, so unlike every other field here it is resolved in `_fill_assets`
    against the top-level default (and a named class when the asset carries
    `class=`), not while walking the body tree."""
    var motor_gaintype_s: String
    var motor_biastype_s: String
    var motor_gainprm_s: String
    var motor_biasprm_s: String

    # Structural attributes, kept as raw strings ("" = not set by this class).
    # Set by `_parse_one_default_block`, consumed by the joint/geom element
    # parsers as a fallback when the element itself omits the attribute.
    var joint_type_s: String
    var joint_axis_s: String
    var joint_range_s: String
    var joint_pos_s: String
    var geom_type_s: String
    var geom_fromto_s: String
    var geom_size_s: String
    var geom_mass_s: String
    # ⚠ `mesh=` FROM A CLASS. Jaco's finger geoms are bare `<geom name="..."/>`
    # elements whose type, mass AND mesh all come from a `childclass`; without
    # this the type resolved to mesh and then no mesh was ever attached
    # (`mesh_id` -1), so six geoms became massless nothings. Every model ported
    # before Jaco spells `mesh=` on the element.
    var geom_mesh_s: String
    # NOTE: captured but not yet consumed. Materials are resolved in a second
    # pass over the worldbody that has no class/childclass context, so a
    # class-supplied `material` currently does not reach the geom. Cosmetic
    # only (rgba fallback), no physics effect — wiring it means threading the
    # childclass stack into that pass.
    var geom_material_s: String
    var geom_pos_s: String
    var geom_quat_s: String
    var geom_group_s: String
    # Site structural attrs, same raw-string treatment as the geom ones.
    # dm_control's hopper declares its two touch sites entirely by class
    # (`<default class="hopper"><site type="sphere" size="0.05"/>`), so the
    # touch sensor sees nothing without these.
    # ── `<default><tendon .../></default>` ────────────────────────────────
    # ⚠⚠ THE SPRING TENDONS OF AN ENTIRE HAND LIVE HERE. `_fill_tendons` read
    # every tendon attribute off the ELEMENT's own tag, and
    # tetheria_aero_hand_open puts `stiffness` and `springlength` in
    # `<default class="distal_spring"><tendon .../></default>` — so its eight
    # spring tendons had stiffness 0 and pulled on nothing. That is the same
    # `<default>`-chain trap this parser has been bitten by for geom `type`,
    # geom `material`, actuator tags and joint ranges; the cure is the same,
    # which is to come through THIS struct (whose classes already inherit
    # from their parent) rather than through a one-level tag lookup.
    #
    # Stored as RAW STRINGS, like the site block below: the consumer already
    # knows how to parse each one (`springlength` takes one value or two,
    # `solreflimit` has the partial-value rule) and a second parse site is a
    # second place for those rules to drift.
    var tendon_stiffness_s: String
    var tendon_springlength_s: String
    var tendon_limited_s: String
    var tendon_range_s: String
    var tendon_margin_s: String
    var tendon_solreflimit_s: String
    var tendon_solimplimit_s: String
    var tendon_width_s: String
    var tendon_rgba_s: String
    var site_type_s: String
    var site_size_s: String
    # POSE from a default class. `type`/`size` were enough until manipulator,
    # whose five touch zones get BOTH their offset and their orientation from
    # `<default class="hand"><site pos=".022 0 -.002" euler="0 15 0"/>` — the
    # site tags themselves carry only `name` and `group`. Without these the
    # zones sit at the body origin, axis-aligned.
    var site_pos_s: String
    var site_quat_s: String
    var site_axisangle_s: String
    var site_xyaxes_s: String
    var site_zaxis_s: String
    var site_euler_s: String

    def __init__(
        out self,
        joint_armature: Float64 = 0.0,
        joint_damping: Float64 = 0.0,
        joint_stiffness: Float64 = 0.0,
        joint_springdamper_0: Float64 = 0.0,
        joint_springdamper_1: Float64 = 0.0,
        joint_limited: Bool = False,
        joint_actfrc_min: Float64 = 0.0,
        joint_actfrc_max: Float64 = 0.0,
        joint_actfrc_limited: Bool = False,
        joint_frictionloss: Float64 = 0.0,
        joint_springref: Float64 = 0.0,
        joint_solref_limit_0: Float64 = 0.02,
        joint_solref_limit_1: Float64 = 1.0,
        joint_solimp_limit_0: Float64 = 0.9,
        joint_solimp_limit_1: Float64 = 0.95,
        joint_solimp_limit_2: Float64 = 0.001,
        joint_solimp_limit_3: Float64 = 0.5,
        joint_solimp_limit_4: Float64 = 2.0,
        geom_density: Float64 = 1000.0,
        geom_friction: Float64 = 1.0,
        geom_friction_spin: Float64 = 0.005,
        geom_friction_roll: Float64 = 0.0001,
        geom_contype: Int = 1,
        geom_conaffinity: Int = 1,
        geom_condim: Int = 3,
        geom_priority: Int = 0,
        geom_solref_0: Float64 = 0.02,
        geom_solref_1: Float64 = 1.0,
        geom_solimp_0: Float64 = 0.9,
        geom_solimp_1: Float64 = 0.95,
        geom_solimp_2: Float64 = 0.001,
        geom_solimp_3: Float64 = 0.5,
        geom_solimp_4: Float64 = 2.0,
        geom_margin: Float64 = 0.0,
        geom_rgba_r: Float64 = -1.0,
        geom_rgba_g: Float64 = -1.0,
        geom_rgba_b: Float64 = -1.0,
        geom_rgba_a: Float64 = -1.0,
        motor_ctrl_limited: Bool = False,
        motor_ctrl_min: Float64 = -1.0,
        motor_ctrl_max: Float64 = 1.0,
        motor_gear: Float64 = 1.0,
    ):
        self.joint_armature = joint_armature
        self.joint_damping = joint_damping
        self.joint_stiffness = joint_stiffness
        self.joint_springdamper_0 = joint_springdamper_0
        self.joint_springdamper_1 = joint_springdamper_1
        self.joint_limited = joint_limited
        self.joint_actfrc_min = joint_actfrc_min
        self.joint_actfrc_max = joint_actfrc_max
        self.joint_actfrc_limited = joint_actfrc_limited
        self.joint_frictionloss = joint_frictionloss
        self.joint_springref = joint_springref
        self.joint_solref_limit_0 = joint_solref_limit_0
        self.joint_solref_limit_1 = joint_solref_limit_1
        self.joint_solimp_limit_0 = joint_solimp_limit_0
        self.joint_solimp_limit_1 = joint_solimp_limit_1
        self.joint_solimp_limit_2 = joint_solimp_limit_2
        self.joint_solimp_limit_3 = joint_solimp_limit_3
        self.joint_solimp_limit_4 = joint_solimp_limit_4
        self.geom_density = geom_density
        self.geom_friction = geom_friction
        self.geom_friction_spin = geom_friction_spin
        self.geom_friction_roll = geom_friction_roll
        self.geom_contype = geom_contype
        self.geom_conaffinity = geom_conaffinity
        self.geom_condim = geom_condim
        self.geom_priority = geom_priority
        self.geom_solref_0 = geom_solref_0
        self.geom_solref_1 = geom_solref_1
        self.geom_solimp_0 = geom_solimp_0
        self.geom_solimp_1 = geom_solimp_1
        self.geom_solimp_2 = geom_solimp_2
        self.geom_solimp_3 = geom_solimp_3
        self.geom_solimp_4 = geom_solimp_4
        self.geom_margin = geom_margin
        self.geom_rgba_r = geom_rgba_r
        self.geom_rgba_g = geom_rgba_g
        self.geom_rgba_b = geom_rgba_b
        self.geom_rgba_a = geom_rgba_a
        self.motor_ctrl_limited = motor_ctrl_limited
        self.motor_ctrl_min = motor_ctrl_min
        self.motor_ctrl_max = motor_ctrl_max
        self.motor_gear = motor_gear
        # Structural attrs are never passed positionally — a <default> block
        # sets them by assignment in `_parse_one_default_block`.
        self.motor_force_limited = False
        self.motor_force_min = 0.0
        self.motor_force_max = 0.0
        self.motor_dyntype_s = ""
        self.motor_dynprm_s = ""
        self.motor_gain = 1.0
        self.motor_gain_set = False
        self.motor_bias2 = 0.0
        self.motor_bias2_set = False
        self.motor_kp_s = ""
        self.motor_inheritrange_s = ""
        self.motor_kv_s = ""
        self.motor_dampratio_s = ""
        self.mesh_scale_s = ""
        self.motor_gaintype_s = ""
        self.motor_biastype_s = ""
        self.motor_gainprm_s = ""
        self.motor_biasprm_s = ""
        self.joint_type_s = ""
        self.joint_axis_s = ""
        self.joint_range_s = ""
        self.joint_pos_s = ""
        self.geom_type_s = ""
        self.geom_fromto_s = ""
        self.geom_size_s = ""
        self.geom_mass_s = ""
        self.geom_mesh_s = ""
        self.geom_material_s = ""
        self.geom_pos_s = ""
        self.geom_quat_s = ""
        self.geom_group_s = ""
        self.tendon_stiffness_s = ""
        self.tendon_springlength_s = ""
        self.tendon_limited_s = ""
        self.tendon_range_s = ""
        self.tendon_margin_s = ""
        self.tendon_solreflimit_s = ""
        self.tendon_solimplimit_s = ""
        self.tendon_width_s = ""
        self.tendon_rgba_s = ""
        self.site_type_s = ""
        self.site_size_s = ""
        self.site_pos_s = ""
        self.site_quat_s = ""
        self.site_axisangle_s = ""
        self.site_xyaxes_s = ""
        self.site_zaxis_s = ""
        self.site_euler_s = ""


# =============================================================================
# NamedDefaultsList — named <default class="..."> blocks
# =============================================================================

# Maximum number of named default classes supported.
#
# Raised 16 -> 128 on 2026-08-03 for dm_control's dog, which declares 42.
#
# ⚠ THE OLD BOUND FAILED SILENTLY, AND IN THE WORST POSSIBLE WAY. `add()`
# dropped anything past the cap without a word, and `find()` returns
# `DefaultsData()` — MuJoCo's GLOBAL defaults — for a class it cannot find. So
# an element naming a dropped class did not error and did not keep its
# neighbours' values; it silently got the compiler defaults.
#
# On dog that meant, measured against MuJoCo:
#
#   geom condim   ours {1: 48, 3: 80, 6:  0}   MuJoCo {1: 81, 3: 5, 6: 42}
#
# i.e. 80 geoms sitting at the default of 3, including all 42 teeth, which
# lost the `condim="6" priority="2" friction="0.5 0.01 0.01"` that
# `<default class="tooth_primitive">` exists to give them. The same drop hit
# the joint classes: `max|d(armature)| = 0.01` and `max|d(stiffness)| = 45`,
# and a wrong armature propagates into `dof_invweight0` (3445x relative).
#
# ⚠ AND IT IS NOT SIMPLY "THE FIRST 16 SURVIVE". `_collect_named_defaults`
# walks BREADTH-FIRST over a worklist, so every top-level class is registered
# before any nested one. dog has more than 16 at the top level, so
# `tooth_primitive` — nested one level inside `collision_primitive` — never
# got in even though it is the sixth class in document order. A cap that
# interacts with traversal order is not something to reason about; it is
# something to make impossible, hence the raise in `add()` below.
comptime MAX_NAMED_DEFAULTS: Int = 128


struct NamedDefault(Copyable, ImplicitlyCopyable, Movable):
    """A named default class: pairs a class name with its DefaultsData."""

    var class_name: String
    var defaults: DefaultsData

    def __init__(out self):
        self.class_name = ""
        self.defaults = DefaultsData()

    def __init__(out self, class_name: String, defaults: DefaultsData):
        self.class_name = class_name
        self.defaults = defaults


struct NamedDefaultsList(Copyable, ImplicitlyCopyable, Movable):
    """List of named default classes for <default class="..."> resolution."""

    # ⚠ HEAP-BACKED, AND THE REASON IS COMPILE TIME, NOT MEMORY.
    #
    # This was `InlineArray[NamedDefault, MAX_NAMED_DEFAULTS]`. A `NamedDefault`
    # is a `String` plus `DefaultsData`'s 59 fields (20 of them `String`), so at
    # MAX_NAMED_DEFAULTS=128 the struct carried ~7,700 fields INLINE — and
    # `parse_xml_full` holds it live across ~14 raising calls, each of which
    # needs a cleanup path that destroys the whole thing.
    #
    # Measured 2026-08-11 by truncating `parse_xml_full` one statement at a
    # time: cutting just BEFORE `_parse_defaults` built in 2 s, cutting just
    # AFTER it built in 284 s. Three lines, 142x. Every other function in this
    # 2935-line parser is free by comparison — gutting 70% of them changed the
    # build by 0%.
    #
    # A `List` puts one pointer in the struct instead, so the destructor the
    # compiler has to emit at every cleanup path is a single call. Same move
    # `FlatModelDef` got on 2026-08-05, for a related reason.
    #
    # Bonus: the cap is gone, and with it the overflow error path. A `List`
    # cannot truncate, so the "more than N named <default> blocks" failure mode
    # this class used to raise on simply cannot occur.
    var items: List[NamedDefault]
    var count: Int

    def __init__(out self, *, copy: Self):
        self.items = copy.items.copy()
        self.count = copy.count

    def __init__(out self):
        self.items = List[NamedDefault]()
        self.count = 0

    def add(mut self, class_name: String, defaults: DefaultsData) raises:
        """Add a named default class.

        No cap: a `List` grows. The previous `InlineArray` version raised once
        it hit `MAX_NAMED_DEFAULTS`, and before that it dropped the class
        SILENTLY — every element naming it then took MuJoCo's global defaults
        instead, with no diagnostic anywhere.
        """
        self.items.append(NamedDefault(class_name, defaults))
        self.count += 1

    def find(self, class_name: String) -> DefaultsData:
        """Find defaults for a class name. Returns top-level defaults if not found.
        """
        for i in range(self.count):
            if self.items[i].class_name == class_name:
                return self.items[i].defaults
        return DefaultsData()


# =============================================================================
# ExcludeData — contact exclusion pair parsed from <contact><exclude>
# =============================================================================


struct ExcludeData(Copyable, ImplicitlyCopyable, Movable):
    """A body pair to exclude from contact detection."""

    var body1: Int  # first body index
    var body2: Int  # second body index

    def __init__(out self, body1: Int = 0, body2: Int = 0):
        self.body1 = body1
        self.body2 = body2


# =============================================================================
# PairData — predefined contact pair parsed from <contact><pair>
# =============================================================================


struct PairData(Copyable, ImplicitlyCopyable, Movable):
    """A geom pair that collides unconditionally, with its own parameters.

    The defaults below are MuJoCo's `mjs_defaultPair`, NOT values derived from
    the two geoms — see the long note on `MODEL_PAIR_SIZE` in
    `physics3d/gpu/constants.mojo` for why the derivation in `mjCPair::Compile`
    is dead code on the XML path, and what the 3.10.0 runtime actually returns.
    """

    var geom1: Int  # geom index, sorted so geom1 < geom2 (as MuJoCo sorts)
    var geom2: Int
    var condim: Int
    var friction: Float64  # sliding
    var friction_spin: Float64  # torsional
    var friction_roll: Float64  # rolling
    var solref_0: Float64
    var solref_1: Float64
    var solimp_0: Float64
    var solimp_1: Float64
    var solimp_2: Float64
    var solimp_3: Float64
    var solimp_4: Float64
    var margin: Float64

    def __init__(out self, geom1: Int = 0, geom2: Int = 0):
        self.geom1 = geom1
        self.geom2 = geom2
        self.condim = 3
        self.friction = 1.0
        self.friction_spin = 0.005
        self.friction_roll = 0.0001
        self.solref_0 = 0.02
        self.solref_1 = 1.0
        self.solimp_0 = 0.9
        self.solimp_1 = 0.95
        self.solimp_2 = 0.001
        self.solimp_3 = 0.5
        self.solimp_4 = 2.0
        self.margin = 0.0


# =============================================================================
# FlatModelDef
# =============================================================================


struct FlatModelDef(Movable):
    """Parsed MJCF model, in flat `List`s — the build-time intermediate.

    ⚠ NON-GENERIC BY DESIGN, as of 2026-08-05. This struct used to carry the
    fourteen dimension parameters (`NBODY`, `NJOINT`, `NGEOM`, …) and store its
    contents in `InlineArray`s sized by them, which forced `parse_xml_full` and
    its eleven helpers to be generic too.

    THAT GENERICITY WAS INCIDENTAL AND IT WAS THE SINGLE LARGEST BUILD COST IN
    THE TREE. The parser is a text scanner; none of its logic depends on a
    dimension. But because the OUTPUT type was dimension-parameterized, every
    distinct model instantiated a fresh copy of a ~2900-line function. Measured
    (`docs/DM_CONTROL_PORT_PHASE2.md` §15):

        parse_xml_full alone, at dm_control dog's dimensions   1961 s
        init_fields total, same model                          2085 s
        init_fields on a 2-body / 2-geom model                  344 s

    i.e. 94% of a 35-minute build, and a ~6-minute floor paid by EVERY model in
    the codebase. With `List` storage the parser compiles ONCE for the whole
    program.

    ⚠ THIS DOES NOT MAKE THE ENGINE RUNTIME-DIMENSIONED, and the two must not
    be conflated. `FlatModelDef` is consumed exactly once, by
    `fields_build.build_model_fields_from_flat`, which copies into the
    COMPTIME-sized `fields.Model`. `Model`, `Data`, the integrators and the
    solvers keep their comptime dimensions — that is the hot path, and its
    genericity is deliberate (zero-cost indexing, GPU batching). See
    `project_physics3d_runtime_dims_assessment` for that separate question.

    ⚠ CAPACITY GUARDS ARE GONE, AND THAT IS AN IMPROVEMENT. The old code wrote
    `if joint_count < NJOINT: result.joints[joint_count] = jd` and then
    incremented the counter REGARDLESS, so a model with more elements than its
    declared dimension silently dropped the overflow — the same shape as
    `MAX_COMPTIME_TENDONS` and `MAX_NAMED_DEFAULTS`. A `List` cannot truncate;
    instead `ModelDefFromXML.init_fields` now checks the resulting lengths
    against the declared dimensions and RAISES on a mismatch, so a
    `parse_xml` / `full_parser` disagreement is loud rather than silent.

    Counts live in the `List`s themselves (`len(fmd.joints)`). `bodies` holds
    NBODY-1 entries — the worldbody is body 0 in `Model` and is not stored here,
    so a body with `Model` index `i` is `bodies[i - 1]`.
    """

    var bodies: List[BodyData]
    var joints: List[JointData]
    var geoms: List[GeomData]
    var actuators: List[ActuatorData]
    var textures: List[TextureData]
    var materials: List[MaterialData]
    var lights: List[LightData]
    var cameras: List[CameraData]
    var sites: List[SiteData]
    var equalities: List[EqualityData]
    var excludes: List[ExcludeData]
    var pairs: List[PairData]
    var tendons: List[TendonData]

    # ── NAMES ─────────────────────────────────────────────────────────────
    # ⚠⚠ THE PARSER RESOLVES NAMES INTO INDICES AND USED TO DROP THE STRINGS,
    # and that breaks four things at once for anything that EDITS a model
    # rather than simulating one (`docs/PHYSICS3D_STUDIO_PLAN.md` §1.3):
    #
    #   * an outliner can only say "body 7";
    #   * selection cannot survive an insert or delete, because identity IS
    #     the index;
    #   * a state remap across a rebuild has no key;
    #   * an MJCF writer would have to synthesise `body0`/`geom3`, and a
    #     flattened export is acceptable while a NAMELESS one is not —
    #     keyframes, sensors, `<contact>` pairs and user code all key on names.
    #
    # ⚠ PARALLEL LISTS, NOT A `name` FIELD ON EACH RECORD, and the reason is
    # mechanical: `BodyData` & co. are `ImplicitlyCopyable` trivial structs
    # held in `List`s and copied freely through the parser's hot path. Adding
    # a `String` makes them non-trivial — implicit deep copies everywhere, and
    # this tree has a measured compile cliff for non-trivial structs in
    # `InlineArray` (282 s -> 5 s for removing ONE such field). `RenderFields`
    # already stores `mesh_names` / `tex_names` this way, so this is the local
    # convention as well as the cheap option.
    #
    # ⚠ INDEXED IN MuJoCo ELEMENT ORDER, WHICH IS NOT DOCUMENT ORDER for
    # joints, sites and geoms — they are grouped by body. The tables come from
    # `names_in_element_order`, which IS the walk `_index_by_name_grouped`
    # now looks up, so an index read out of a record and an index into these
    # tables cannot drift apart.
    #
    # ⚠ AN UNNAMED ELEMENT IS "", not a synthesised name. Most geoms in this
    # tree have no `name=`; inventing one here would make an export claim a
    # name the source never had.
    var body_names: List[String]
    """Body names by `Model` body id. **Index 0 is the worldbody, "world"** —
    unlike `bodies`, which omits it, so `body_names[i]` names `bodies[i - 1]`."""

    var joint_names: List[String]
    var geom_names: List[String]
    var site_names: List[String]
    var actuator_names: List[String]

    var gravity_x: Float64
    var gravity_y: Float64
    var gravity_z: Float64
    var timestep: Float64
    var opt_density: Float64  # Fluid density (kg/m³), 0 = disabled
    var opt_viscosity: Float64  # Fluid dynamic viscosity (Pa·s), 0 = disabled
    # `<option noslip_tolerance>` — the improvement threshold `mj_solNoSlip`
    # stops on, NOT the primal solver's `tolerance`. MuJoCo's default is 1e-6;
    # dm_control's manipulation models set 0 ("run every iteration"). See
    # `_parse_option` for the measurement that made this worth parsing.
    var noslip_tolerance: Float64
    # `<option ccd_tolerance= ccd_iterations=>` — EPA's stopping rule. MuJoCo's
    # defaults are 1e-6 and 35; ours were hardcoded at 1e-8 and 64 and a model
    # setting either was ignored. See `_scan_ccd_tolerance` in `xml_parser` for
    # why tighter is not safer.
    var ccd_tolerance: Float64
    var ccd_iterations: Int
    # `<option impratio>` — the ratio of frictional to normal constraint
    # IMPEDANCE, MuJoCo default 1. See `_parse_option` in `full_parser` for the
    # measurement; the short version is that five solvers read
    # `MODEL_META_IDX_IMPRATIO` and `fields_build` hardcoded 1.0 into it, so
    # this attribute was parsed by nothing and every model was simulated at 1.
    var impratio: Float64
    # `<option cone/solver/integrator>` — which solver the MODEL asks for, as
    # opposed to the one the caller happened to build. Parsed 2026-08-19; see
    # `_parse_option` in `full_parser`. Defaults are MuJoCo's: PYRAMIDAL and
    # NEWTON, both the opposite of what this tree habitually builds.
    var cone: Int
    var solver: Int
    var integrator: Int
    # Largest `condim` any geom asks for, floored at 3 — what a caller must
    # build `MAX_CONDIM` at to avoid `contact_solve`'s silent clamp.
    var max_condim: Int
    # `<option><flag multiccd="disable" nativeccd="disable"/></option>` —
    # `mjDSBL_MULTICCD` / `mjDSBL_NATIVECCD`, both DISABLE bits that are off by
    # default on the 3.10.0 runtime.
    #
    # ⚠⚠ `multiccd_disabled` IS LOAD-BEARING AND `nativeccd_disabled` IS NOT.
    # The first switches off `multi_ccd`'s perturbation loop, which is what
    # took `manipulation/reassemble5` from 437 contacts to MuJoCo's 111. The
    # second is parsed and stored so the value is INSPECTABLE rather than
    # silently absent, but nothing reads it: see the note beside
    # `MODEL_META_IDX_MULTICCD_DISABLED` for what honouring it would change and
    # why that is a separate, smaller job.
    # `<option><flag eulerdamp="disable"/></option>` — mjDSBL_EULERDAMP.
    # Only the EULER integrator reads it; `mj_implicit` has its own path and
    # MuJoCo does not consult this flag there. One model in the Menagerie
    # tree sets it (tetheria_aero_hand_open) and it is worth 61.5% of that
    # model's velocity — see `MODEL_META_IDX_EULERDAMP_DISABLED`.
    var eulerdamp_disabled: Bool
    var multiccd_disabled: Bool
    var nativeccd_disabled: Bool
    # `<compiler boundmass= boundinertia=>`. MuJoCo clamps EVERY body (id > 0)
    # after its inertial frame is set: `mass = max(mass, boundmass)` and the
    # same per principal moment. Default 0, i.e. no bound. Load-bearing on
    # composer models — 3 of Jaco's 17 bodies (two attachment frames carrying
    # no geoms at all, plus one whose only geom has mass 1e-9) take their
    # ENTIRE mass and inertia from these.
    var boundmass: Float64
    var boundinertia: Float64
    # `<compiler inertiafromgeom= inertiagrouprange= settotalmass=>`.
    #
    # ⚠ THESE WERE READ AT COMPTIME OFF THE RAW XML until phase 1b, by
    # `_xml_compiler_inertiafromgeom` / `_inertiagrouprange` / `_settotalmass`,
    # and reached `build_model_fields_from_flat` as compile-time PARAMETERS.
    # They are here now because 1b's whole point is that nothing may read the
    # MJCF at compile time — the comptime interpreter cannot `open()` a file,
    # so every comptime reader pins the model to a `String` in Mojo source.
    #
    # ⚠ `inertiafromgeom` DEFAULTS TO AUTO (2), NOT off. MuJoCo derives a
    # body's mass/inertia from its geoms unless the body carries an explicit
    # `<inertial>`. Defaulting to 0 gave pendulum ~1/21 of its true inertia
    # and went unnoticed for months because every Gym-derived XML states the
    # attribute explicitly and the dm_control suite states nothing
    # (fixed 2026-07-29 on the comptime side; the same default is kept here).
    #
    # ⚠ `settotalmass` is -1.0 when ABSENT, not 0.0 — 0 is a legal request.
    # The ROOT `<default>`'s motor ctrlrange, i.e. what `ModelDefLike`'s
    # CTRL_MIN / CTRL_MAX advertise as the env's scalar action bounds.
    #
    # ⚠⚠ A SUMMARY, NOT THE CLAMP, AND IT LIES ON SOME MODELS — see the note
    # on `ModelDefLike.CTRL_MIN`. `apply_actions` clamps each actuator to its
    # OWN range; this pair only sizes the box a policy samples from. It is
    # reproduced here EXACTLY, wrong models included, because redefining it
    # changes the action scaling of every shipped env and that is a behaviour
    # change owed its own before/after — not a ride-along on phase 1b.
    #
    # ⚠ Defaults to (-1, 1) when the root default has no actuator tag, which
    # is what `_xml_default_motor_ctrlrange` returns in the same case.
    var default_motor_ctrl_min: Float64
    var default_motor_ctrl_max: Float64
    var inertiafromgeom: Int  # 0=false, 1=true, 2=auto
    var inertiagrouprange_min: Int
    var inertiagrouprange_max: Int
    var settotalmass: Float64  # -1.0 = absent
    # MuJoCo `m->na` — ACTIVATION variables, not `nu`. Derived by
    # `_fill_actuators` as it walks the actuators, exactly as the comptime twin
    # does. Phase 1a.4 turns `na` into a comptime PARAMETER of
    # `ModelDefFromXML` (a dimension cannot come from a runtime record — it
    # sizes the `act` tensor); this field is what that parameter gets asserted
    # against at construction so a wrong value is loud instead of silent.
    var na: Int
    # Actuator transmission wraps, `ai * TENDON_MAX_WRAPS + k`. Sized by
    # `_fill_actuator_transmission` once the actuator count is known.
    # First actuator whose gaintype/biastype shape we do not model, and why.
    # Mirrors `ComptimeActData.bad_actuator` / `_code`, which
    # `ModelDefFromXML` already refuses at BUILD time (`:1122`). Ported here so
    # the runtime path can make the same refusal in 1a.3; nothing reads it yet.
    #   0 gaintype != fixed        2 biasprm[0] != 0
    #   1 biastype not none/affine 3 biasprm[1] not in {-gain, 0}
    var bad_actuator: Int
    var bad_actuator_code: Int
    # How many `<actuator>` children this parser SKIPPED because it does not
    # model their element type — `<adhesion>`, `<plugin>`, `<muscle>`, ... —
    # so that `len(actuators) + this == MuJoCo's nu` for every model.
    #
    # ⚠⚠ WITHOUT IT A SHORT `nact` IS INDISTINGUISHABLE FROM A SHORT MODEL. A
    # skipped actuator does not fail, it shifts: every control index past the
    # first one missing lands on the wrong actuator. flybody reports nu 78 to
    # MuJoCo and 70 here (eight `<adhesion>`) and shadow_dexee 12 and 0
    # (twelve `<plugin plugin="mujoco.pid">`); both were found by diffing
    # counts against the runtime, not by anything the parser said.
    #
    # The invariant is what to assert, not the value: a model whose skipped
    # types get implemented should send this to 0 and `nact` up by the same
    # amount, and the sum stays right either way.
    var unmodelled_actuators: Int
    # How many actuators PARSED fine and then found no transmission this
    # engine can express — a `site=`, `body=`, `slidersite=` or `cranksite=`
    # motor. They occupy a slot in `nact`, consume their control, and produce
    # ZERO FORCE.
    #
    # ⚠⚠ THIS IS A WHOLE ROBOT CLASS, NOT A ROUNDING ERROR. Both quadrotors in
    # Menagerie — skydio_x2 and bitcraze_crazyflie_2 — drive EVERY one of their
    # four rotors through `<motor site="thrust1" gear="0 0 1 0 0 -.0201"/>`, so
    # in this engine neither aircraft has any thrust at all. MuJoCo answers
    # `qfrc_actuator = [0, 0, 0.378896, 0.01744, -0.053045, -0.001947]` on
    # skydio's first step; we answered six zeros.
    var zero_transmission_actuators: Int
    # Actuators whose transmission needs the POSE, not just `qpos` — today
    # that is `<position tendon="...">` on a SPATIAL tendon. They keep
    # `trn_n = 0` like the unmodelled ones above, and are deliberately NOT
    # in that count: they ARE applied, by
    # `dynamics/pose_transmission.apply_pose_transmission`, which runs after
    # forward kinematics. See `_fill_actuator_transmission`.
    var pose_transmission_actuators: Int
    # ── qpos0 / initial pose ─────────────────────────────────────────────
    # Three sources, in this order (`xml_parser.mojo:4504`, `:4520`, `:4554`):
    #   1. each joint's `ref`, already deg-converted, at its qpos address
    #   2. a free joint's enclosing body `pos` into adr..adr+2, and qw=1 at
    #      adr+3 when no explicit init_qpos overrides
    #   3. `<custom><numeric name="init_qpos" data=...>` OVERRIDING both
    # `qpos0_nq` mirrors `ComptimeActData.nq`: how many entries are
    # meaningful, NOT the model's nq. `reset_data` applies them only when > 0.
    var qpos0: List[Float64]
    var qpos0_nq: Int
    var free_joint_qpos_adr: Int
    # ── <keyframe> ───────────────────────────────────────────────────────
    # ⚠ RECORDED, NOT APPLIED — `mj_resetData` ignores keyframes and so must
    # `reset_data` (`feedback_a_keyframe_is_not_a_reset_pose`).
    # Row strides: `qpos` AND `qvel` both stride by nq (mirroring the twin,
    # which uses NQ0 for both); `ctrl` strides by nact.
    # ⚠ A wrong-length attribute is REJECTED, not padded — MuJoCo pads from
    # the RAW pre-unit-conversion attribute, which is not worth reproducing,
    # and 145 of 145 Menagerie keyframe attributes are exactly full length.
    var nkey: Int
    var key_time: List[Float64]
    var key_nqpos: List[Int]
    var key_nqvel: List[Int]
    var key_nctrl: List[Int]
    var key_qpos: List[Float64]
    var key_qvel: List[Float64]
    var key_ctrl: List[Float64]
    # 2 = a key carried act/mpos/mquat, which are refused rather than dropped.
    # ⚠ The twin's code 1 (over MAX_COMPTIME_KEYFRAMES) CANNOT occur here: a
    # `List` does not truncate. Same reason `MAX_NAMED_DEFAULTS` stopped being
    # a failure mode.
    var bad_keyframe_code: Int
    var motor_trn_qadr: List[Int]
    var motor_trn_dadr: List[Int]
    var motor_trn_coef: List[Float64]

    # Mesh assets: name → file path mapping.
    var mesh_asset_names: List[String]
    var mesh_asset_files: List[String]
    var mesh_asset_scale: List[Float64]
    # `<mesh refpos="x y z" refquat="w x y z">` — a rigid transform MuJoCo
    # applies to the RAW vertices before everything else
    # (`mjCMesh::ApplyTransformations`, user_mesh.cc:1257):
    #
    #     v -= refpos                    ; then
    #     v  = R(normalize(refquat))^T v ; then
    #     v *= scale
    #
    # ⚠ THE ROTATION IS THE INVERSE OF THE QUATERNION. `mjuu_mulvecmatT` is
    # `M^T v`, so `refquat="1 -1 0 0"` (a -90 deg turn about x) rotates the
    # mesh +90 deg. Reading it as a forward rotation lands 180 deg away.
    #
    # ⚠ IT COMES BEFORE `scale`, so it cannot be folded in afterwards unless
    # the scale is uniform.
    var mesh_asset_refpos: List[Float64]   # 3 per asset
    var mesh_asset_refquat: List[Float64]  # 4 per asset, w x y z
    # `<mesh inertia="shell">` — 1 per asset, 0 for MuJoCo's default "legacy".
    #
    # ⚠ A DIFFERENT PHYSICAL MODEL, NOT A FALLBACK: the mass sits on the
    # SURFACE rather than through the volume. Three Menagerie models declare
    # it — hello_robot_stretch_3 (11 meshes), hello_robot_stretch (8),
    # pndbotics_adam_lite (4).
    var mesh_asset_inertia_shell: List[Int]
    # How many `maxhullvert` declarations the document carries that this
    # engine does not honour. MuJoCo decimates each convex hull to that many
    # vertices; we keep all of them, so ours CONTAINS MuJoCo's and contacts on
    # the decimated faces sit slightly differently. Recorded so the gate can
    # assert the count rather than watching for a print.
    var unhonoured_maxhullvert: Int
    """Three per asset, parallel to `mesh_asset_names` — `<mesh scale>`.

    ⚠⚠ NOT COSMETIC AND NOT USUALLY 1. 19 Menagerie robots set it: 38
    declarations are `0.001 0.001 0.001` (the STL is in MILLIMETRES) and 44
    are a MIRROR like `1 -1 1` (one mesh serving a left and a right part).
    Ignoring it made robotis_op3's collision hulls 1000x oversized — every
    hull spanning metres, overlapping the floor and each other, which the
    solver answered by launching the robot to 77 m."""
    # ── `<visual>` (phase 1a.5) ──────────────────────────────────────────
    #
    # ⚠ RENDER-ONLY, and absent from this record until 1a.5: the viewer read
    # them off the comptime `ComptimeRenderData`. Defaults are MuJoCo's, and
    # `vis_has_headlight` is a PRESENCE FLAG rather than a colour test —
    # `<headlight ambient="0 0 0"/>` is a real declaration and equals the
    # unset value.
    var vis_znear: Float64
    var vis_fogstart: Float64
    var vis_fogend: Float64
    var vis_shadowsize: Int
    var vis_headlight_ambient_r: Float64
    var vis_headlight_ambient_g: Float64
    var vis_headlight_ambient_b: Float64
    var vis_has_headlight: Bool

    var num_mesh_assets: Int

    def __init__(out self):
        self.boundmass = 0.0
        self.boundinertia = 0.0
        self.default_motor_ctrl_min = -1.0
        self.default_motor_ctrl_max = 1.0
        # MuJoCo's defaults, not zeroes — see the field declarations.
        self.inertiafromgeom = 2
        self.inertiagrouprange_min = 0
        self.inertiagrouprange_max = 5
        self.settotalmass = -1.0
        self.bodies = List[BodyData]()
        self.joints = List[JointData]()
        self.geoms = List[GeomData]()
        self.actuators = List[ActuatorData]()
        self.na = 0
        self.bad_actuator = -1
        self.bad_actuator_code = -1
        self.unmodelled_actuators = 0
        self.zero_transmission_actuators = 0
        self.pose_transmission_actuators = 0
        self.qpos0 = List[Float64]()
        self.qpos0_nq = 0
        self.free_joint_qpos_adr = -1
        self.nkey = 0
        self.key_time = List[Float64]()
        self.key_nqpos = List[Int]()
        self.key_nqvel = List[Int]()
        self.key_nctrl = List[Int]()
        self.key_qpos = List[Float64]()
        self.key_qvel = List[Float64]()
        self.key_ctrl = List[Float64]()
        self.bad_keyframe_code = 0
        self.motor_trn_qadr = List[Int]()
        self.motor_trn_dadr = List[Int]()
        self.motor_trn_coef = List[Float64]()
        self.textures = List[TextureData]()
        self.materials = List[MaterialData]()
        self.lights = List[LightData]()
        self.cameras = List[CameraData]()
        self.sites = List[SiteData]()
        self.equalities = List[EqualityData]()
        self.excludes = List[ExcludeData]()
        self.pairs = List[PairData]()
        self.tendons = List[TendonData]()
        self.body_names = List[String]()
        self.joint_names = List[String]()
        self.geom_names = List[String]()
        self.site_names = List[String]()
        self.actuator_names = List[String]()
        self.gravity_x = Float64(0)
        self.gravity_y = Float64(0)
        self.gravity_z = Float64(-9.81)
        self.timestep = Float64(0.01)
        self.opt_density = Float64(0)
        self.opt_viscosity = Float64(0)
        self.noslip_tolerance = Float64(1e-6)
        self.ccd_tolerance = Float64(MJ_CCD_TOLERANCE)
        self.ccd_iterations = MJ_CCD_ITERATIONS
        self.impratio = 1.0
        self.cone = ConeType.PYRAMIDAL
        self.solver = SolverType.NEWTON
        self.integrator = IntegratorType.EULER
        self.max_condim = 3
        # Both default OFF, matching MuJoCo: the features are ON unless a model
        # disables them.
        self.eulerdamp_disabled = False
        self.multiccd_disabled = False
        self.nativeccd_disabled = False
        self.mesh_asset_names = List[String]()
        self.mesh_asset_files = List[String]()
        self.mesh_asset_scale = List[Float64]()
        self.mesh_asset_refpos = List[Float64]()
        self.mesh_asset_refquat = List[Float64]()
        self.mesh_asset_inertia_shell = List[Int]()
        self.unhonoured_maxhullvert = 0
        self.vis_znear = 0.01
        self.vis_fogstart = 3.0
        self.vis_fogend = 10.0
        self.vis_shadowsize = 4096
        self.vis_headlight_ambient_r = 0.1
        self.vis_headlight_ambient_g = 0.1
        self.vis_headlight_ambient_b = 0.1
        self.vis_has_headlight = False
        self.num_mesh_assets = 0

    # `setup_model` (FlatModelDef -> legacy CPU `Model`) was deleted at the
    # G4 fields sunset — the spec-direct build is
    # `fields_build.build_model_fields_from_flat`.
