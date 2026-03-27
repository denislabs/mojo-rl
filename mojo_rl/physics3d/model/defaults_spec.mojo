# =============================================================================
# Sentinel value for "use model default"
# =============================================================================

# Float64 fields that are always non-negative use -1.0 as "unset"
comptime UNSET_F64: Float64 = -1.0
# Int fields that are always non-negative use -1 as "unset"
comptime UNSET_INT: Int = -1


def _resolve_f64[val: Float64, default: Float64]() -> Float64:
    """Resolve a compile-time Float64: use val if set (>= 0), else default."""

    comptime if val >= 0.0:
        return val
    else:
        return default
    return default


def _resolve_int[val: Int, default: Int]() -> Int:
    """Resolve a compile-time Int: use val if set (>= 0), else default."""

    comptime if val >= 0:
        return val
    else:
        return default


# =============================================================================
# ModelDefaults — MuJoCo-style <default> block
# =============================================================================


trait ModelDefaultsLike(TrivialRegisterPassable):
    """Trait for compile-time model defaults (MuJoCo-style <default> + <option>).

    Allows different specializations of ModelDefaults to be passed as
    type parameters to setup_model functions.
    """

    comptime GEOM_FRICTION: Float64
    comptime GEOM_FRICTION_SPIN: Float64
    comptime GEOM_FRICTION_ROLL: Float64
    comptime GEOM_CONDIM: Int
    comptime GEOM_CONTYPE: Int
    comptime GEOM_CONAFFINITY: Int
    comptime GEOM_SOLREF_0: Float64
    comptime GEOM_SOLREF_1: Float64
    comptime GEOM_SOLIMP_0: Float64
    comptime GEOM_SOLIMP_1: Float64
    comptime GEOM_SOLIMP_2: Float64
    comptime GEOM_SOLIMP_3: Float64
    comptime GEOM_SOLIMP_4: Float64
    comptime GEOM_MARGIN: Float64
    comptime JOINT_ARMATURE: Float64
    comptime JOINT_DAMPING: Float64
    comptime JOINT_STIFFNESS: Float64
    comptime JOINT_FRICTIONLOSS: Float64
    comptime JOINT_SOLREF_LIMIT_0: Float64
    comptime JOINT_SOLREF_LIMIT_1: Float64
    comptime JOINT_SOLIMP_LIMIT_0: Float64
    comptime JOINT_SOLIMP_LIMIT_1: Float64
    comptime JOINT_SOLIMP_LIMIT_2: Float64
    comptime JOINT_SOLIMP_LIMIT_3: Float64
    comptime JOINT_SOLIMP_LIMIT_4: Float64
    comptime IMPRATIO: Float64
    # Geom density default (kg/m³)
    comptime GEOM_DENSITY: Float64
    # MuJoCo <compiler> inertiafromgeom
    comptime INERTIAFROMGEOM: Bool
    # MuJoCo <option> block
    comptime GRAVITY_X: Float64
    comptime GRAVITY_Y: Float64
    comptime GRAVITY_Z: Float64
    comptime TIMESTEP: Float64
    # MuJoCo <compiler> block
    comptime SETTOTALMASS: Float64
    # MuJoCo <option> fluid parameters
    comptime OPT_DENSITY: Float64  # Fluid density (kg/m³), 0 = disabled
    comptime OPT_VISCOSITY: Float64  # Fluid dynamic viscosity (Pa·s), 0 = disabled


@fieldwise_init
struct ModelDefaults[
    # Geom defaults (MuJoCo <default><geom .../>)
    geom_friction: Float64 = 0.5,
    geom_friction_spin: Float64 = 0.005,
    geom_friction_roll: Float64 = 0.0001,
    geom_condim: Int = 3,
    geom_contype: Int = 1,
    geom_conaffinity: Int = 1,
    geom_solref_0: Float64 = 0.02,
    geom_solref_1: Float64 = 1.0,
    geom_solimp_0: Float64 = 0.9,
    geom_solimp_1: Float64 = 0.95,
    geom_solimp_2: Float64 = 0.001,
    geom_solimp_3: Float64 = 0.5,
    geom_solimp_4: Float64 = 2.0,
    geom_margin: Float64 = 0.0,
    # Joint defaults (MuJoCo <default><joint .../>)
    joint_armature: Float64 = 0.1,
    joint_damping: Float64 = 0.0,
    joint_stiffness: Float64 = 0.0,
    joint_frictionloss: Float64 = 0.0,
    joint_solref_limit_0: Float64 = 0.02,
    joint_solref_limit_1: Float64 = 1.0,
    joint_solimp_limit_0: Float64 = 0.9,
    joint_solimp_limit_1: Float64 = 0.95,
    joint_solimp_limit_2: Float64 = 0.001,
    joint_solimp_limit_3: Float64 = 0.5,
    joint_solimp_limit_4: Float64 = 2.0,
    # Motor defaults (MuJoCo <default><motor .../>)
    motor_ctrl_min: Float64 = -1.0,
    motor_ctrl_max: Float64 = 1.0,
    # Model-level (MuJoCo <option>)
    impratio: Float64 = 1.0,
    # Geom density default (MuJoCo default = 1000 kg/m³)
    geom_density: Float64 = 1000.0,
    # MuJoCo <compiler> inertiafromgeom (default True, matching MuJoCo compiler default)
    inertiafromgeom: Bool = True,
    gravity_x: Float64 = 0.0,
    gravity_y: Float64 = 0.0,
    gravity_z: Float64 = -9.81,
    timestep: Float64 = 0.01,
    # Compiler directive (MuJoCo <compiler>)
    settotalmass: Float64 = -1.0,
    # Fluid dynamics options (MuJoCo <option density="..." viscosity="..."/>)
    opt_density: Float64 = 0.0,
    opt_viscosity: Float64 = 0.0,
](ModelDefaultsLike):
    """MuJoCo-style model defaults block.

    Components that don't specify a value (sentinel = -1.0/-1) inherit
    from these defaults. Resolution happens at Geoms/Joints.setup_model time.

    Default values match MuJoCo's built-in defaults for geom/joint elements.
    """

    # Explicit trait member mapping (Mojo struct params don't auto-satisfy traits)
    comptime GEOM_FRICTION: Float64 = Self.geom_friction
    comptime GEOM_FRICTION_SPIN: Float64 = Self.geom_friction_spin
    comptime GEOM_FRICTION_ROLL: Float64 = Self.geom_friction_roll
    comptime GEOM_CONDIM: Int = Self.geom_condim
    comptime GEOM_CONTYPE: Int = Self.geom_contype
    comptime GEOM_CONAFFINITY: Int = Self.geom_conaffinity
    comptime GEOM_SOLREF_0: Float64 = Self.geom_solref_0
    comptime GEOM_SOLREF_1: Float64 = Self.geom_solref_1
    comptime GEOM_SOLIMP_0: Float64 = Self.geom_solimp_0
    comptime GEOM_SOLIMP_1: Float64 = Self.geom_solimp_1
    comptime GEOM_SOLIMP_2: Float64 = Self.geom_solimp_2
    comptime GEOM_SOLIMP_3: Float64 = Self.geom_solimp_3
    comptime GEOM_SOLIMP_4: Float64 = Self.geom_solimp_4
    comptime GEOM_MARGIN: Float64 = Self.geom_margin
    comptime JOINT_ARMATURE: Float64 = Self.joint_armature
    comptime JOINT_DAMPING: Float64 = Self.joint_damping
    comptime JOINT_STIFFNESS: Float64 = Self.joint_stiffness
    comptime JOINT_FRICTIONLOSS: Float64 = Self.joint_frictionloss
    comptime JOINT_SOLREF_LIMIT_0: Float64 = Self.joint_solref_limit_0
    comptime JOINT_SOLREF_LIMIT_1: Float64 = Self.joint_solref_limit_1
    comptime JOINT_SOLIMP_LIMIT_0: Float64 = Self.joint_solimp_limit_0
    comptime JOINT_SOLIMP_LIMIT_1: Float64 = Self.joint_solimp_limit_1
    comptime JOINT_SOLIMP_LIMIT_2: Float64 = Self.joint_solimp_limit_2
    comptime JOINT_SOLIMP_LIMIT_3: Float64 = Self.joint_solimp_limit_3
    comptime JOINT_SOLIMP_LIMIT_4: Float64 = Self.joint_solimp_limit_4
    comptime IMPRATIO: Float64 = Self.impratio
    comptime GEOM_DENSITY: Float64 = Self.geom_density
    comptime INERTIAFROMGEOM: Bool = Self.inertiafromgeom
    comptime GRAVITY_X: Float64 = Self.gravity_x
    comptime GRAVITY_Y: Float64 = Self.gravity_y
    comptime GRAVITY_Z: Float64 = Self.gravity_z
    comptime TIMESTEP: Float64 = Self.timestep
    comptime SETTOTALMASS: Float64 = Self.settotalmass
    comptime OPT_DENSITY: Float64 = Self.opt_density
    comptime OPT_VISCOSITY: Float64 = Self.opt_viscosity
