"""RobotDef compositor for compile-time robot definitions.

Composes Bodies and Joints into a RobotDef with auto-computed dimensions.
Uses Variadic.types + @parameter for to iterate at compile time, following
the same pattern as Sequential[*LAYERS: Model] in deep_rl/model/sequential.mojo.

Note: Bodies and Joints are standalone variadic containers. RobotDef takes
concrete Int parameters because Mojo cannot resolve variadic type packs
through multiple levels of nesting (accessing RobotDef.NQ would fail with
"unbound parameter" if RobotDef contained Bodies/Joints directly).

Usage:
    comptime HalfCheetahBodies = Bodies[Torso, BThigh, ...]
    comptime HalfCheetahJoints = Joints[RootX, RootZ, ...]
    comptime HalfCheetahRobot = RobotDef[
        HalfCheetahBodies.N,
        HalfCheetahJoints.N,
        HalfCheetahJoints._sum_nq(),
        HalfCheetahJoints._sum_nv(),
    ]
"""

from std.builtin.variadics import Variadic
from .body_spec import BodySpec
from .joint_spec import JointSpec
from ..types import Model
from ..joint_types import JNT_HINGE, JNT_SLIDE


# =============================================================================
# Bodies — variadic body list
# =============================================================================


@fieldwise_init
struct Bodies[*B: BodySpec]:
    """Compile-time list of body specifications.

    Provides N (body count) and type-level access to each body via body_types[i].
    """

    comptime body_types = Variadic.types[T=BodySpec, *Self.B]
    comptime N: Int = Variadic.size(Self.body_types)

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](mut model: Model[DTYPE, NQ, NV, Self.N, NJOINT, MAX_CONTACTS]):
        """Populate model body properties from compile-time BodySpec list.

        Iterates over all body specs and sets mass, inertia, geometry, parent,
        local frame, and collision filtering on the model.
        """

        @parameter
        for i in range(Self.N):
            comptime B = Self.body_types[i]

            # Mass, inertia, radius
            model.set_body(
                i,
                mass=Scalar[DTYPE](B.MASS),
                inertia=(
                    Scalar[DTYPE](B.ixx()),
                    Scalar[DTYPE](B.iyy()),
                    Scalar[DTYPE](B.izz()),
                ),
                radius=Scalar[DTYPE](B.RADIUS),
            )

            # Kinematic tree
            model.set_body_parent(i, B.PARENT)

            # Geometry
            model.body_geom_type[i] = B.GEOM_TYPE
            model.body_half_length[i] = Scalar[DTYPE](B.HALF_LENGTH)
            model.body_half_x[i] = Scalar[DTYPE](B.HALF_X)
            model.body_half_y[i] = Scalar[DTYPE](B.HALF_Y)
            model.body_half_z[i] = Scalar[DTYPE](B.HALF_Z)

            # Local frame in parent
            model.set_body_local_frame(
                i,
                pos=(
                    Scalar[DTYPE](B.POS_X),
                    Scalar[DTYPE](B.POS_Y),
                    Scalar[DTYPE](B.POS_Z),
                ),
                quat=(
                    Scalar[DTYPE](B.QUAT_X),
                    Scalar[DTYPE](B.QUAT_Y),
                    Scalar[DTYPE](B.QUAT_Z),
                    Scalar[DTYPE](B.QUAT_W),
                ),
            )

            # Collision filtering
            model.body_contype[i] = B.CONTYPE
            model.body_conaffinity[i] = B.CONAFFINITY


# =============================================================================
# Joints — variadic joint list with sum helpers
# =============================================================================


@fieldwise_init
struct Joints[*J: JointSpec]:
    """Compile-time list of joint specifications.

    Provides N (joint count), sum helpers for total NQ/NV, and offset helpers
    for computing qpos/qvel addresses of each joint.
    """

    comptime joint_types = Variadic.types[T=JointSpec, *Self.J]
    comptime N: Int = Variadic.size(Self.joint_types)

    @staticmethod
    fn _sum_nq() -> Int:
        """Sum NQ across all joints (total qpos dimension)."""
        var total = 0

        @parameter
        for i in range(Self.N):
            total += Self.joint_types[i].NQ
        return total

    @staticmethod
    fn _sum_nv() -> Int:
        """Sum NV across all joints (total qvel dimension)."""
        var total = 0

        @parameter
        for i in range(Self.N):
            total += Self.joint_types[i].NV
        return total

    @staticmethod
    fn _qpos_offset[idx: Int]() -> Int:
        """Compute qpos address for joint idx (sum of NQ for joints 0..idx-1).
        """
        var total = 0

        @parameter
        for j in range(idx):
            total += Self.joint_types[j].NQ
        return total

    @staticmethod
    fn _qvel_offset[idx: Int]() -> Int:
        """Compute qvel/dof address for joint idx (sum of NV for joints 0..idx-1).
        """
        var total = 0

        @parameter
        for j in range(idx):
            total += Self.joint_types[j].NV
        return total

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut model: Model[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        """Populate model joints from compile-time JointSpec list.

        Iterates over all joint specs and calls add_hinge_joint or
        add_slide_joint with correct qpos/qvel offsets.
        """

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.JNT_TYPE == JNT_HINGE:
                _ = model.add_hinge_joint(
                    body_id=J.BODY_IDX,
                    pos=(
                        Scalar[DTYPE](J.POS_X),
                        Scalar[DTYPE](J.POS_Y),
                        Scalar[DTYPE](J.POS_Z),
                    ),
                    axis=(
                        Scalar[DTYPE](J.AXIS_X),
                        Scalar[DTYPE](J.AXIS_Y),
                        Scalar[DTYPE](J.AXIS_Z),
                    ),
                    tau_limit=Scalar[DTYPE](J.TAU_LIMIT),
                    range_min=Scalar[DTYPE](J.RANGE_MIN),
                    range_max=Scalar[DTYPE](J.RANGE_MAX),
                    armature=Scalar[DTYPE](J.ARMATURE),
                    damping=Scalar[DTYPE](J.DAMPING),
                    stiffness=Scalar[DTYPE](J.STIFFNESS),
                    springref=Scalar[DTYPE](J.SPRINGREF),
                    frictionloss=Scalar[DTYPE](J.FRICTIONLOSS),
                )
            elif J.JNT_TYPE == JNT_SLIDE:
                _ = model.add_slide_joint(
                    body_id=J.BODY_IDX,
                    pos=(
                        Scalar[DTYPE](J.POS_X),
                        Scalar[DTYPE](J.POS_Y),
                        Scalar[DTYPE](J.POS_Z),
                    ),
                    axis=(
                        Scalar[DTYPE](J.AXIS_X),
                        Scalar[DTYPE](J.AXIS_Y),
                        Scalar[DTYPE](J.AXIS_Z),
                    ),
                    force_limit=Scalar[DTYPE](J.TAU_LIMIT),
                    range_min=Scalar[DTYPE](J.RANGE_MIN),
                    range_max=Scalar[DTYPE](J.RANGE_MAX),
                    armature=Scalar[DTYPE](J.ARMATURE),
                    damping=Scalar[DTYPE](J.DAMPING),
                    stiffness=Scalar[DTYPE](J.STIFFNESS),
                    springref=Scalar[DTYPE](J.SPRINGREF),
                    frictionloss=Scalar[DTYPE](J.FRICTIONLOSS),
                )


# =============================================================================
# RobotDef — full robot compositor (concrete Int parameters)
# =============================================================================


@fieldwise_init
struct RobotDef[nbody: Int, njoint: Int, nq: Int, nv: Int]:
    """Compile-time robot definition with pre-computed dimensions.

    Takes concrete Int parameters rather than Bodies/Joints directly,
    because Mojo cannot resolve variadic type packs through nesting.

    Usage:
        comptime MyBodies = Bodies[...]
        comptime MyJoints = Joints[...]
        comptime MyRobot = RobotDef[
            MyBodies.N, MyJoints.N,
            MyJoints._sum_nq(), MyJoints._sum_nv(),
        ]
    """

    comptime NBODY: Int = Self.nbody
    comptime NJOINT: Int = Self.njoint
    comptime NQ: Int = Self.nq
    comptime NV: Int = Self.nv
