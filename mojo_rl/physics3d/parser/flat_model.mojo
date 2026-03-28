"""Flat data structs and FlatModelDef for XML-driven model construction.

Promotes BodyData/JointData from test_flat_model_def.mojo prototype and adds
GeomData, ActuatorData, DefaultsData.  FlatModelDef[NBODY,NJOINT,NQ,NV,NGEOM,NACT]
holds the fully-populated data parsed from MJCF XML and can populate a Model
struct via setup_model().
"""

from std.collections import InlineArray
from std.math import sqrt
from mojo_rl.physics3d.types import Model, ConeType
from mojo_rl.physics3d.joint_types import (
    JNT_HINGE,
    JNT_SLIDE,
    JNT_BALL,
    JNT_FREE,
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
    var armature: Float64
    var damping: Float64
    var stiffness: Float64
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

    def __init__(
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
        ref_val: Float64 = 0.0,
        solref_limit_0: Float64 = -1.0,
        solref_limit_1: Float64 = -1.0,
        solimp_limit_0: Float64 = -1.0,
        solimp_limit_1: Float64 = -1.0,
        solimp_limit_2: Float64 = -1.0,
        solimp_limit_3: Float64 = -1.0,
        solimp_limit_4: Float64 = -1.0,
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
        self.ref_val = ref_val
        self.solref_limit_0 = solref_limit_0
        self.solref_limit_1 = solref_limit_1
        self.solimp_limit_0 = solimp_limit_0
        self.solimp_limit_1 = solimp_limit_1
        self.solimp_limit_2 = solimp_limit_2
        self.solimp_limit_3 = solimp_limit_3
        self.solimp_limit_4 = solimp_limit_4


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
        solimp_0: Float64 = 0.0,
        solimp_1: Float64 = 0.8,
        solimp_2: Float64 = 0.01,
        solimp_3: Float64 = 0.5,
        solimp_4: Float64 = 2.0,
        margin: Float64 = 0.0,
        density: Float64 = 1000.0,
        mass: Float64 = -1.0,
        rgba_r: Float64 = 0.7,
        rgba_g: Float64 = 0.7,
        rgba_b: Float64 = 0.7,
        rgba_a: Float64 = 1.0,
        material_id: Int = -1,
        group: Int = 0,
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
        self.density = density
        self.mass = mass
        self.rgba_r = rgba_r
        self.rgba_g = rgba_g
        self.rgba_b = rgba_b
        self.rgba_a = rgba_a
        self.material_id = material_id
        self.group = group


# =============================================================================
# ActuatorData
# =============================================================================


struct ActuatorData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime actuator data parsed from <motor/position/velocity> tags."""

    var joint_id: Int  # 0-based joint index this actuator drives
    var gear: Float64  # Force/torque scaling
    var ctrl_min: Float64
    var ctrl_max: Float64
    var is_ctrl_limited: Bool

    def __init__(
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

    var tex_type: Int  # TEX_SKYBOX / TEX_2D / TEX_CUBE
    var builtin: Int  # TEX_BUILTIN_* — procedural texture pattern
    var mark: Int  # TEX_MARK_* — overlay mark type
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
# EqualityData
# =============================================================================


# Equality constraint type constants (matches physics3d/types.mojo)
comptime _EQ_CONNECT: Int = 0
comptime _EQ_WELD: Int = 1


struct EqualityData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime equality constraint data parsed from <equality> section."""

    var eq_type: Int  # _EQ_CONNECT or _EQ_WELD
    var body_a: Int  # first body index
    var body_b: Int  # second body index (0 = worldbody)
    var anchor_a_x: Float64
    var anchor_a_y: Float64
    var anchor_a_z: Float64
    var anchor_b_x: Float64
    var anchor_b_y: Float64
    var anchor_b_z: Float64
    var relpose_x: Float64  # weld only: relative orientation quaternion
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

    def __init__(
        out self,
        eq_type: Int = _EQ_WELD,
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
        relpose_w: Float64 = 1.0,
        solref_0: Float64 = 0.02,
        solref_1: Float64 = 1.0,
        solimp_0: Float64 = 0.9,
        solimp_1: Float64 = 0.95,
        solimp_2: Float64 = 0.001,
        solimp_3: Float64 = 0.5,
        solimp_4: Float64 = 2.0,
    ):
        self.eq_type = eq_type
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

    def __init__(
        out self,
        joint_armature: Float64 = 0.0,
        joint_damping: Float64 = 0.0,
        joint_stiffness: Float64 = 0.0,
        joint_limited: Bool = False,
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
        geom_friction: Float64 = 0.5,
        geom_friction_spin: Float64 = 0.005,
        geom_friction_roll: Float64 = 0.0001,
        geom_contype: Int = 1,
        geom_conaffinity: Int = 1,
        geom_condim: Int = 3,
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
    ):
        self.joint_armature = joint_armature
        self.joint_damping = joint_damping
        self.joint_stiffness = joint_stiffness
        self.joint_limited = joint_limited
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
# NamedDefaultsList — named <default class="..."> blocks
# =============================================================================

# Maximum number of named default classes supported
comptime MAX_NAMED_DEFAULTS: Int = 16


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

    var items: InlineArray[NamedDefault, MAX_NAMED_DEFAULTS]
    var count: Int

    def __init__(out self):
        self.items = InlineArray[NamedDefault, MAX_NAMED_DEFAULTS](
            fill=NamedDefault()
        )
        self.count = 0

    def add(mut self, class_name: String, defaults: DefaultsData):
        """Add a named default class."""
        if self.count < MAX_NAMED_DEFAULTS:
            self.items[self.count] = NamedDefault(class_name, defaults)
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
    NEQ: Int = 0,
    NEXCLUDE: Int = 0,
](Movable):
    """Model definition using flat InlineArrays — driven entirely from XML.

    Dimensions are supplied as comptime parameters (from parse_xml() output).
    Data is stored in InlineArray[BodyData/JointData/GeomData/ActuatorData/...].
    setup_model() uses regular for loops — no comptime if needed.

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
    var equalities: InlineArray[EqualityData, Self.NEQ + 1]
    var excludes: InlineArray[ExcludeData, Self.NEXCLUDE + 1]
    var gravity_x: Float64
    var gravity_y: Float64
    var gravity_z: Float64
    var timestep: Float64

    def __init__(out self):
        self.bodies = InlineArray[BodyData, Self.NBODY](fill=BodyData())
        self.joints = InlineArray[JointData, Self.NJOINT](fill=JointData())
        self.geoms = InlineArray[GeomData, Self.NGEOM](fill=GeomData())
        self.actuators = InlineArray[ActuatorData, Self.NACT](
            fill=ActuatorData()
        )
        self.textures = InlineArray[TextureData, Self.NTEX + 1](
            fill=TextureData()
        )
        self.materials = InlineArray[MaterialData, Self.NMAT + 1](
            fill=MaterialData()
        )
        self.lights = InlineArray[LightData, Self.NLIGHT + 1](fill=LightData())
        self.cameras = InlineArray[CameraData, Self.NCAM + 1](fill=CameraData())
        self.sites = InlineArray[SiteData, Self.NSITE + 1](fill=SiteData())
        self.equalities = InlineArray[EqualityData, Self.NEQ + 1](
            fill=EqualityData()
        )
        self.excludes = InlineArray[ExcludeData, Self.NEXCLUDE + 1](
            fill=ExcludeData()
        )
        self.gravity_x = Float64(0)
        self.gravity_y = Float64(0)
        self.gravity_z = Float64(-9.81)
        self.timestep = Float64(0.01)

    def setup_model[
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

        Uses regular for loops — no comptime if needed since we write
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

        # Contact solimp/solref from first geom's parsed values (geom[0] =
        # floor / first worldbody geom, which inherits the <default><geom> solimp).
        if Self.NGEOM > 0:
            var g0 = self.geoms[0]
            model.solref_contact[0] = Scalar[DTYPE](g0.solref_0)
            model.solref_contact[1] = Scalar[DTYPE](g0.solref_1)
            model.solimp_contact[0] = Scalar[DTYPE](g0.solimp_0)
            model.solimp_contact[1] = Scalar[DTYPE](g0.solimp_1)
            model.solimp_contact[2] = Scalar[DTYPE](g0.solimp_2)
            model.solimp_contact[3] = Scalar[DTYPE](g0.solimp_3)
            model.solimp_contact[4] = Scalar[DTYPE](g0.solimp_4)

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
            # Mocap body flag
            if b.is_mocap:
                model.body_mocap[body_idx] = True
            # Explicit inertia flag (for inertiafromgeom="auto")
            if b.has_explicit_inertia:
                model.body_has_explicit_inertia[body_idx] = True

        # Compute body_rootid from body_parent (root = child of worldbody)
        model.body_rootid[0] = 0
        for bi in range(1, Self.NBODY):
            var p = model.body_parent[bi]
            if p == 0:
                model.body_rootid[bi] = bi
            else:
                model.body_rootid[bi] = model.body_rootid[p]

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
            elif jd.jnt_type == JNT_FREE:
                _ = model.add_free_joint(
                    jd.body_id,
                    armature=Scalar[DTYPE](jd.armature),
                    damping=Scalar[DTYPE](jd.damping),
                )
            # JNT_BALL not yet wired — add when needed

            # Set qpos0 from joint ref attribute (MuJoCo: displacement = qpos - qpos0)
            var qpos_adr_j = model.joints[j].qpos_adr
            model.qpos0[qpos_adr_j] = Scalar[DTYPE](jd.ref_val)

            # Set per-joint solimp/solref for limits (use parsed value if >= 0, else model default)
            var jr0: Float64 = (
                jd.solref_limit_0 if jd.solref_limit_0
                >= 0.0 else Float64(model.solref_limit[0])
            )
            var jr1: Float64 = (
                jd.solref_limit_1 if jd.solref_limit_1
                >= 0.0 else Float64(model.solref_limit[1])
            )
            var ji0: Float64 = (
                jd.solimp_limit_0 if jd.solimp_limit_0
                >= 0.0 else Float64(model.solimp_limit[0])
            )
            var ji1: Float64 = (
                jd.solimp_limit_1 if jd.solimp_limit_1
                >= 0.0 else Float64(model.solimp_limit[1])
            )
            var ji2: Float64 = (
                jd.solimp_limit_2 if jd.solimp_limit_2
                >= 0.0 else Float64(model.solimp_limit[2])
            )
            var ji3: Float64 = (
                jd.solimp_limit_3 if jd.solimp_limit_3
                >= 0.0 else Float64(model.solimp_limit[3])
            )
            var ji4: Float64 = (
                jd.solimp_limit_4 if jd.solimp_limit_4
                >= 0.0 else Float64(model.solimp_limit[4])
            )
            model.joint_solref_limit[j * 2 + 0] = Scalar[DTYPE](jr0)
            model.joint_solref_limit[j * 2 + 1] = Scalar[DTYPE](jr1)
            model.joint_solimp_limit[j * 5 + 0] = Scalar[DTYPE](ji0)
            model.joint_solimp_limit[j * 5 + 1] = Scalar[DTYPE](ji1)
            model.joint_solimp_limit[j * 5 + 2] = Scalar[DTYPE](ji2)
            model.joint_solimp_limit[j * 5 + 3] = Scalar[DTYPE](ji3)
            model.joint_solimp_limit[j * 5 + 4] = Scalar[DTYPE](ji4)

        # Sync model-level solimp_limit/solref_limit from joint[0] values.
        # GPU constraint builder reads MODEL_META_IDX_SOLIMP_LIMIT_* (model-level meta)
        # while CPU reads per-joint values. For models with uniform joint solimp
        # (all current models), this ensures CPU/GPU consistency.
        comptime if Self.NJOINT > 0:
            model.solimp_limit[0] = model.joint_solimp_limit[0]
            model.solimp_limit[1] = model.joint_solimp_limit[1]
            model.solimp_limit[2] = model.joint_solimp_limit[2]
            model.solimp_limit[3] = model.joint_solimp_limit[3]
            model.solimp_limit[4] = model.joint_solimp_limit[4]
            model.solref_limit[0] = model.joint_solref_limit[0]
            model.solref_limit[1] = model.joint_solref_limit[1]

        # Compute body_weldid — bodies with joints get their own ID,
        # bodies without joints inherit parent's weldid (MuJoCo convention)
        var body_has_joint = List[Bool](capacity=Self.NBODY)
        for _ in range(Self.NBODY):
            body_has_joint.append(False)
        for j in range(Self.NJOINT):
            body_has_joint[model.joints[j].body_id] = True
        model.body_weldid[0] = 0  # worldbody welds to itself
        for bi in range(1, Self.NBODY):
            if body_has_joint[bi]:
                model.body_weldid[bi] = bi
            else:
                model.body_weldid[bi] = model.body_weldid[model.body_parent[bi]]

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
            model.geom_group[i] = gd.group
            # Bounding sphere radius for broad-phase collision detection
            if gd.geom_type == _GEOM_PLANE:
                model.geom_rbound[i] = Scalar[DTYPE](
                    1e10
                )  # planes are infinite
            elif gd.geom_type == _GEOM_SPHERE:
                model.geom_rbound[i] = Scalar[DTYPE](gd.radius)
            elif gd.geom_type == _GEOM_CAPSULE:
                model.geom_rbound[i] = Scalar[DTYPE](gd.radius + gd.half_length)
            elif gd.geom_type == _GEOM_CYLINDER:
                model.geom_rbound[i] = Scalar[DTYPE](
                    sqrt(
                        gd.half_length * gd.half_length + gd.radius * gd.radius
                    )
                )
            elif gd.geom_type == _GEOM_BOX:
                model.geom_rbound[i] = Scalar[DTYPE](
                    sqrt(
                        gd.half_x * gd.half_x
                        + gd.half_y * gd.half_y
                        + gd.half_z * gd.half_z
                    )
                )
            else:
                model.geom_rbound[i] = Scalar[DTYPE](gd.radius)

        # Sites — populate model.site_body and model.site_pos for FK
        for i in range(Self.NSITE):
            var sd = self.sites[i]
            model.site_body[i] = sd.body_id
            model.site_pos[i * 3 + 0] = Scalar[DTYPE](sd.pos_x)
            model.site_pos[i * 3 + 1] = Scalar[DTYPE](sd.pos_y)
            model.site_pos[i * 3 + 2] = Scalar[DTYPE](sd.pos_z)

        # Actuators: Model has no actuators[] array — stored only in FlatModelDef.
        # Use fmd.actuators[i] directly when constructing Actuators[...] at call site.

        # Equality constraints — populate model.equality_constraints[] via add API
        for i in range(Self.NEQ):
            var ed = self.equalities[i]
            if ed.eq_type == _EQ_CONNECT:
                _ = model.add_connect_constraint(
                    ed.body_a,
                    ed.body_b,
                    anchor_a=(
                        Scalar[DTYPE](ed.anchor_a_x),
                        Scalar[DTYPE](ed.anchor_a_y),
                        Scalar[DTYPE](ed.anchor_a_z),
                    ),
                    anchor_b=(
                        Scalar[DTYPE](ed.anchor_b_x),
                        Scalar[DTYPE](ed.anchor_b_y),
                        Scalar[DTYPE](ed.anchor_b_z),
                    ),
                    solref=(
                        Scalar[DTYPE](ed.solref_0),
                        Scalar[DTYPE](ed.solref_1),
                    ),
                    solimp=(
                        Scalar[DTYPE](ed.solimp_0),
                        Scalar[DTYPE](ed.solimp_1),
                        Scalar[DTYPE](ed.solimp_2),
                    ),
                )
            elif ed.eq_type == _EQ_WELD:
                _ = model.add_weld_constraint(
                    ed.body_a,
                    ed.body_b,
                    anchor_a=(
                        Scalar[DTYPE](ed.anchor_a_x),
                        Scalar[DTYPE](ed.anchor_a_y),
                        Scalar[DTYPE](ed.anchor_a_z),
                    ),
                    anchor_b=(
                        Scalar[DTYPE](ed.anchor_b_x),
                        Scalar[DTYPE](ed.anchor_b_y),
                        Scalar[DTYPE](ed.anchor_b_z),
                    ),
                    relpose=(
                        Scalar[DTYPE](ed.relpose_x),
                        Scalar[DTYPE](ed.relpose_y),
                        Scalar[DTYPE](ed.relpose_z),
                        Scalar[DTYPE](ed.relpose_w),
                    ),
                    solref=(
                        Scalar[DTYPE](ed.solref_0),
                        Scalar[DTYPE](ed.solref_1),
                    ),
                    solimp=(
                        Scalar[DTYPE](ed.solimp_0),
                        Scalar[DTYPE](ed.solimp_1),
                        Scalar[DTYPE](ed.solimp_2),
                    ),
                )

        # Contact exclusion pairs
        for i in range(Self.NEXCLUDE):
            var ex = self.excludes[i]
            model.exclude_body1.append(ex.body1)
            model.exclude_body2.append(ex.body2)
        model.num_excludes = Self.NEXCLUDE
