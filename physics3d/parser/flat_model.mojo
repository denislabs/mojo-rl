"""Flat data structs and FlatModelDef for XML-driven model construction.

Promotes BodyData/JointData from test_flat_model_def.mojo prototype and adds
GeomData, ActuatorData, DefaultsData.  FlatModelDef[NBODY,NJOINT,NQ,NV,NGEOM,NACT]
holds the fully-populated data parsed from MJCF XML and can populate a Model
struct via setup_model().
"""

from collections import InlineArray
from physics3d.types import Model, ConeType
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE


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

    fn __init__(
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
    var armature: Float64
    var damping: Float64
    var stiffness: Float64
    var springref: Float64
    var frictionloss: Float64

    fn __init__(
        out self,
        jnt_type: Int = JNT_HINGE,
        body_id: Int = 1,
        nq: Int = 1,
        nv: Int = 1,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        axis_x: Float64 = 0.0,
        axis_y: Float64 = 1.0,
        axis_z: Float64 = 0.0,
        range_min: Float64 = -1e10,
        range_max: Float64 = 1e10,
        is_limited: Bool = False,
        armature: Float64 = 0.0,
        damping: Float64 = 0.0,
        stiffness: Float64 = 0.0,
        springref: Float64 = 0.0,
        frictionloss: Float64 = 0.0,
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
        self.armature = armature
        self.damping = damping
        self.stiffness = stiffness
        self.springref = springref
        self.frictionloss = frictionloss


# =============================================================================
# GeomData
# =============================================================================

# Geometry type constants (matches physics3d/constants.mojo)
comptime _GEOM_PLANE: Int = 0
comptime _GEOM_SPHERE: Int = 1
comptime _GEOM_CAPSULE: Int = 2
comptime _GEOM_BOX: Int = 3
comptime _GEOM_CYLINDER: Int = 4


struct GeomData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime geom data."""

    var body_id: Int        # 0 = worldbody/static, >=1 = body-attached
    var geom_type: Int      # _GEOM_PLANE/SPHERE/CAPSULE/BOX/CYLINDER
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var quat_x: Float64
    var quat_y: Float64
    var quat_z: Float64
    var quat_w: Float64
    var radius: Float64
    var half_length: Float64  # capsule half-length along axis
    var half_x: Float64       # box half-extents
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
    var mass: Float64     # -1.0 = use density (not specified explicitly)
    var rgba_r: Float64   # visual colour (r component, 0..1)
    var rgba_g: Float64
    var rgba_b: Float64
    var rgba_a: Float64
    var material_id: Int  # index into FlatModelDef.materials[], -1 if none

    fn __init__(
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
        solimp_0: Float64 = 0.0,
        solimp_1: Float64 = 0.8,
        solimp_2: Float64 = 0.01,
        solimp_3: Float64 = 0.5,
        solimp_4: Float64 = 2.0,
        margin: Float64 = 0.0,
        mass: Float64 = -1.0,
        rgba_r: Float64 = 0.7,
        rgba_g: Float64 = 0.7,
        rgba_b: Float64 = 0.7,
        rgba_a: Float64 = 1.0,
        material_id: Int = -1,
    ):
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
        self.mass = mass
        self.rgba_r = rgba_r
        self.rgba_g = rgba_g
        self.rgba_b = rgba_b
        self.rgba_a = rgba_a
        self.material_id = material_id


# =============================================================================
# ActuatorData
# =============================================================================


struct ActuatorData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime actuator data parsed from <motor/position/velocity> tags."""

    var joint_id: Int        # 0-based joint index this actuator drives
    var gear: Float64        # Force/torque scaling
    var ctrl_min: Float64
    var ctrl_max: Float64
    var is_ctrl_limited: Bool

    fn __init__(
        out self,
        joint_id: Int = -1,
        gear: Float64 = 1.0,
        ctrl_min: Float64 = -1.0,
        ctrl_max: Float64 = 1.0,
        is_ctrl_limited: Bool = False,
    ):
        self.joint_id = joint_id
        self.gear = gear
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max
        self.is_ctrl_limited = is_ctrl_limited


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

    var tex_type: Int    # TEX_SKYBOX / TEX_2D / TEX_CUBE
    var builtin: Int     # TEX_BUILTIN_* — procedural texture pattern
    var mark: Int        # TEX_MARK_* — overlay mark type
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

    fn __init__(
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
    ):
        self.tex_type = tex_type
        self.builtin = builtin
        self.mark = mark
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

    var tex_id: Int         # index into textures[], -1 if no texture
    var rgba_r: Float64
    var rgba_g: Float64
    var rgba_b: Float64
    var rgba_a: Float64
    var shininess: Float64
    var specular: Float64
    var reflectance: Float64
    var texrepeat_u: Float64
    var texrepeat_v: Float64
    var texuniform: Bool    # tile texture uniformly across surface

    fn __init__(
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

    var body_id: Int        # body this light is attached to (0 = worldbody)
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
    var directional: Bool   # true = directional (infinite), false = point/spot
    var castshadow: Bool
    var cutoff: Float64     # spot cone half-angle in degrees (100 = point light)
    var exponent: Float64   # spot exponent
    var mode: Int           # LIGHT_MODE_*

    fn __init__(
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

    var body_id: Int        # body this camera is attached to (0 = worldbody)
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var quat_x: Float64
    var quat_y: Float64
    var quat_z: Float64
    var quat_w: Float64
    var fovy: Float64       # vertical field of view in degrees
    var ipd: Float64        # interpupillary distance (stereo)
    var mode: Int           # CAM_MODE_*

    fn __init__(
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


# =============================================================================
# SiteData
# =============================================================================


struct SiteData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime site data parsed from <site> in <worldbody>."""

    var body_id: Int
    var site_type: Int      # geom-like type (_GEOM_SPHERE / _GEOM_CAPSULE / etc.)
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var quat_x: Float64
    var quat_y: Float64
    var quat_z: Float64
    var quat_w: Float64
    var size_0: Float64     # radius (sphere/capsule/cylinder) or half-x (box)
    var size_1: Float64     # half-length (capsule/cylinder) or half-y (box)
    var size_2: Float64     # half-z (box only)

    fn __init__(
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
# DefaultsData
# =============================================================================


struct DefaultsData(Copyable, ImplicitlyCopyable, Movable):
    """Parsed <default> block — applied when specific attrs are absent."""

    var joint_armature: Float64
    var joint_damping: Float64
    var joint_stiffness: Float64
    var joint_limited: Bool
    var joint_frictionloss: Float64
    var joint_springref: Float64
    var geom_friction: Float64
    var geom_friction_spin: Float64
    var geom_friction_roll: Float64
    var geom_contype: Int
    var geom_conaffinity: Int
    var geom_condim: Int
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

    fn __init__(
        out self,
        joint_armature: Float64 = 0.0,
        joint_damping: Float64 = 0.0,
        joint_stiffness: Float64 = 0.0,
        joint_limited: Bool = False,
        joint_frictionloss: Float64 = 0.0,
        joint_springref: Float64 = 0.0,
        geom_friction: Float64 = 0.5,
        geom_friction_spin: Float64 = 0.005,
        geom_friction_roll: Float64 = 0.0001,
        geom_contype: Int = 1,
        geom_conaffinity: Int = 1,
        geom_condim: Int = 3,
        geom_solref_0: Float64 = 0.02,
        geom_solref_1: Float64 = 1.0,
        geom_solimp_0: Float64 = 0.0,
        geom_solimp_1: Float64 = 0.8,
        geom_solimp_2: Float64 = 0.01,
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
    ):
        self.joint_armature = joint_armature
        self.joint_damping = joint_damping
        self.joint_stiffness = joint_stiffness
        self.joint_limited = joint_limited
        self.joint_frictionloss = joint_frictionloss
        self.joint_springref = joint_springref
        self.geom_friction = geom_friction
        self.geom_friction_spin = geom_friction_spin
        self.geom_friction_roll = geom_friction_roll
        self.geom_contype = geom_contype
        self.geom_conaffinity = geom_conaffinity
        self.geom_condim = geom_condim
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


# =============================================================================
# FlatModelDef
# =============================================================================


struct FlatModelDef[
    NBODY: Int,
    NJOINT: Int,
    NQ: Int,
    NV: Int,
    NGEOM: Int,
    NACT: Int,
    NTEX: Int = 0,
    NMAT: Int = 0,
    NLIGHT: Int = 0,
    NCAM: Int = 0,
    NSITE: Int = 0,
](Movable):
    """Model definition using flat InlineArrays — driven entirely from XML.

    Dimensions are supplied as comptime parameters (from parse_xml() output).
    Data is stored in InlineArray[BodyData/JointData/GeomData/ActuatorData/...].
    setup_model() uses regular for loops — no @parameter needed.

    Optional visual-element arrays (NTEX, NMAT, NLIGHT, NCAM, NSITE) default to
    0; the underlying InlineArray uses size+1 to satisfy the >0 requirement.
    Access indices 0..N-1 only; index N is a sentinel and should be ignored.
    """

    var bodies: InlineArray[BodyData, Self.NBODY]
    var joints: InlineArray[JointData, Self.NJOINT]
    var geoms: InlineArray[GeomData, Self.NGEOM]
    var actuators: InlineArray[ActuatorData, Self.NACT]
    var textures: InlineArray[TextureData, Self.NTEX + 1]
    var materials: InlineArray[MaterialData, Self.NMAT + 1]
    var lights: InlineArray[LightData, Self.NLIGHT + 1]
    var cameras: InlineArray[CameraData, Self.NCAM + 1]
    var sites: InlineArray[SiteData, Self.NSITE + 1]
    var gravity_x: Float64
    var gravity_y: Float64
    var gravity_z: Float64
    var timestep: Float64

    fn __init__(out self):
        self.bodies = InlineArray[BodyData, Self.NBODY](fill=BodyData())
        self.joints = InlineArray[JointData, Self.NJOINT](fill=JointData())
        self.geoms = InlineArray[GeomData, Self.NGEOM](fill=GeomData())
        self.actuators = InlineArray[ActuatorData, Self.NACT](fill=ActuatorData())
        self.textures = InlineArray[TextureData, Self.NTEX + 1](fill=TextureData())
        self.materials = InlineArray[MaterialData, Self.NMAT + 1](fill=MaterialData())
        self.lights = InlineArray[LightData, Self.NLIGHT + 1](fill=LightData())
        self.cameras = InlineArray[CameraData, Self.NCAM + 1](fill=CameraData())
        self.sites = InlineArray[SiteData, Self.NSITE + 1](fill=SiteData())
        self.gravity_x = Float64(0)
        self.gravity_y = Float64(0)
        self.gravity_z = Float64(-9.81)
        self.timestep = Float64(0.01)

    fn setup_model[
        DTYPE: DType,
        MAX_CONTACTS: Int,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        MODEL_NSITE: Int = 0,
    ](
        self,
        mut model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
            Self.NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            MODEL_NSITE,
        ],
    ):
        """Write body, joint, and geom data from InlineArrays into the Model struct.

        Uses regular for loops — no @parameter needed since we write
        scalar values, not instantiate per-type specialisations.
        """
        # Global physics parameters
        model.gravity = SIMD[DTYPE, 4](
            Scalar[DTYPE](self.gravity_x),
            Scalar[DTYPE](self.gravity_y),
            Scalar[DTYPE](self.gravity_z),
            Scalar[DTYPE](0),
        )
        model.timestep = Scalar[DTYPE](self.timestep)

        # Bodies (index 1..NBODY-1; worldbody=0 is pre-initialised by Model)
        for i in range(Self.NBODY - 1):
            var b = self.bodies[i]
            var body_idx = i + 1
            model.set_body(
                body_idx,
                name="body_" + String(i),
                mass=Scalar[DTYPE](b.mass),
                inertia=(
                    Scalar[DTYPE](b.ixx),
                    Scalar[DTYPE](b.iyy),
                    Scalar[DTYPE](b.izz),
                ),
            )
            model.set_body_parent(body_idx, b.parent)
            model.set_body_local_frame(
                body_idx,
                pos=(
                    Scalar[DTYPE](b.pos_x),
                    Scalar[DTYPE](b.pos_y),
                    Scalar[DTYPE](b.pos_z),
                ),
                quat=(
                    Scalar[DTYPE](b.quat_x),
                    Scalar[DTYPE](b.quat_y),
                    Scalar[DTYPE](b.quat_z),
                    Scalar[DTYPE](b.quat_w),
                ),
            )
            model.set_body_ipos_iquat(
                body_idx,
                ipos=(
                    Scalar[DTYPE](b.ipos_x),
                    Scalar[DTYPE](b.ipos_y),
                    Scalar[DTYPE](b.ipos_z),
                ),
                iquat=(
                    Scalar[DTYPE](b.iquat_x),
                    Scalar[DTYPE](b.iquat_y),
                    Scalar[DTYPE](b.iquat_z),
                    Scalar[DTYPE](b.iquat_w),
                ),
            )

        # Joints — use Model.add_hinge_joint / add_slide_joint API
        for j in range(Self.NJOINT):
            var jd = self.joints[j]
            if jd.jnt_type == JNT_HINGE:
                _ = model.add_hinge_joint(
                    jd.body_id,
                    pos=(
                        Scalar[DTYPE](jd.pos_x),
                        Scalar[DTYPE](jd.pos_y),
                        Scalar[DTYPE](jd.pos_z),
                    ),
                    axis=(
                        Scalar[DTYPE](jd.axis_x),
                        Scalar[DTYPE](jd.axis_y),
                        Scalar[DTYPE](jd.axis_z),
                    ),
                    range_min=Scalar[DTYPE](jd.range_min),
                    range_max=Scalar[DTYPE](jd.range_max),
                    armature=Scalar[DTYPE](jd.armature),
                    damping=Scalar[DTYPE](jd.damping),
                    stiffness=Scalar[DTYPE](jd.stiffness),
                    springref=Scalar[DTYPE](jd.springref),
                    frictionloss=Scalar[DTYPE](jd.frictionloss),
                )
            elif jd.jnt_type == JNT_SLIDE:
                _ = model.add_slide_joint(
                    jd.body_id,
                    pos=(
                        Scalar[DTYPE](jd.pos_x),
                        Scalar[DTYPE](jd.pos_y),
                        Scalar[DTYPE](jd.pos_z),
                    ),
                    axis=(
                        Scalar[DTYPE](jd.axis_x),
                        Scalar[DTYPE](jd.axis_y),
                        Scalar[DTYPE](jd.axis_z),
                    ),
                    armature=Scalar[DTYPE](jd.armature),
                    damping=Scalar[DTYPE](jd.damping),
                    stiffness=Scalar[DTYPE](jd.stiffness),
                    springref=Scalar[DTYPE](jd.springref),
                    frictionloss=Scalar[DTYPE](jd.frictionloss),
                )
            # JNT_BALL and JNT_FREE not yet wired — add when needed

        # Geoms — populate model.geom_* arrays directly
        for i in range(Self.NGEOM):
            var gd = self.geoms[i]
            model.geom_type[i] = gd.geom_type
            model.geom_body[i] = gd.body_id
            model.geom_pos[i * 3 + 0] = Scalar[DTYPE](gd.pos_x)
            model.geom_pos[i * 3 + 1] = Scalar[DTYPE](gd.pos_y)
            model.geom_pos[i * 3 + 2] = Scalar[DTYPE](gd.pos_z)
            model.geom_quat[i * 4 + 0] = Scalar[DTYPE](gd.quat_x)
            model.geom_quat[i * 4 + 1] = Scalar[DTYPE](gd.quat_y)
            model.geom_quat[i * 4 + 2] = Scalar[DTYPE](gd.quat_z)
            model.geom_quat[i * 4 + 3] = Scalar[DTYPE](gd.quat_w)
            model.geom_radius[i] = Scalar[DTYPE](gd.radius)
            model.geom_half_length[i] = Scalar[DTYPE](gd.half_length)
            model.geom_half_x[i] = Scalar[DTYPE](gd.half_x)
            model.geom_half_y[i] = Scalar[DTYPE](gd.half_y)
            model.geom_half_z[i] = Scalar[DTYPE](gd.half_z)
            model.geom_friction[i] = Scalar[DTYPE](gd.friction)
            model.geom_condim[i] = gd.condim
            model.geom_friction_spin[i] = Scalar[DTYPE](gd.friction_spin)
            model.geom_friction_roll[i] = Scalar[DTYPE](gd.friction_roll)
            model.geom_contype[i] = gd.contype
            model.geom_conaffinity[i] = gd.conaffinity
            model.geom_solref[i * 2 + 0] = Scalar[DTYPE](gd.solref_0)
            model.geom_solref[i * 2 + 1] = Scalar[DTYPE](gd.solref_1)
            model.geom_solimp[i * 5 + 0] = Scalar[DTYPE](gd.solimp_0)
            model.geom_solimp[i * 5 + 1] = Scalar[DTYPE](gd.solimp_1)
            model.geom_solimp[i * 5 + 2] = Scalar[DTYPE](gd.solimp_2)
            model.geom_solimp[i * 5 + 3] = Scalar[DTYPE](gd.solimp_3)
            model.geom_solimp[i * 5 + 4] = Scalar[DTYPE](gd.solimp_4)
            model.geom_margin[i] = Scalar[DTYPE](gd.margin)
            model.geom_mass[i] = Scalar[DTYPE](gd.mass)

        # Actuators: Model has no actuators[] array — stored only in FlatModelDef.
        # Use fmd.actuators[i] directly when constructing Actuators[...] at call site.
