"""TendonSpec trait and concrete fixed tendon types.

A fixed tendon is a linear combination of joint positions:
  ten_length = Σ(coef_i * qpos[joint_qposadr_i])

An equality constraint enforces: ten_length - length_ref = 0
producing 1 bilateral constraint row with trivial Jacobian: J[dof_adr_i] = coef_i.

Usage:
    from physics3d.model.tendon_spec import TendonSpec, FixedTendon, Tendons

    # Couple joint 3 and joint 5 with equal coefficients
    comptime MyTendon = FixedTendon[num_joints=2, joint_0=3, coef_0=1.0, joint_1=5, coef_1=-1.0]
"""

from ..types import TendonDef, ConeType, Model
from std.builtin.variadics import Variadic


trait TendonSpec:
    """Compile-time specification for a fixed tendon."""

    comptime NUM_JOINTS: Int  # Number of joints (1..4)
    comptime JOINT_0: Int  # Joint index 0 (-1 if unused)
    comptime JOINT_1: Int  # Joint index 1 (-1 if unused)
    comptime JOINT_2: Int  # Joint index 2 (-1 if unused)
    comptime JOINT_3: Int  # Joint index 3 (-1 if unused)
    comptime COEF_0: Float64  # Coefficient for joint 0
    comptime COEF_1: Float64  # Coefficient for joint 1
    comptime COEF_2: Float64  # Coefficient for joint 2
    comptime COEF_3: Float64  # Coefficient for joint 3
    comptime LENGTH_REF: Float64  # Reference length (0 = compute from initial qpos)
    # Impedance parameters
    comptime SOLREF_0: Float64  # timeconst
    comptime SOLREF_1: Float64  # dampratio
    comptime SOLIMP_0: Float64  # dmin
    comptime SOLIMP_1: Float64  # dmax
    comptime SOLIMP_2: Float64  # width
    comptime SOLIMP_3: Float64  # midpoint
    comptime SOLIMP_4: Float64  # power


@fieldwise_init
struct FixedTendon[
    num_joints: Int = 2,
    joint_0: Int = -1,
    coef_0: Float64 = 0.0,
    joint_1: Int = -1,
    coef_1: Float64 = 0.0,
    joint_2: Int = -1,
    coef_2: Float64 = 0.0,
    joint_3: Int = -1,
    coef_3: Float64 = 0.0,
    length_ref: Float64 = 0.0,
    solref_0: Float64 = 0.02,
    solref_1: Float64 = 1.0,
    solimp_0: Float64 = 0.9,
    solimp_1: Float64 = 0.95,
    solimp_2: Float64 = 0.001,
    solimp_3: Float64 = 0.5,
    solimp_4: Float64 = 2.0,
](TendonSpec):
    """Fixed tendon — 1 bilateral constraint row."""

    comptime NUM_JOINTS: Int = Self.num_joints
    comptime JOINT_0: Int = Self.joint_0
    comptime JOINT_1: Int = Self.joint_1
    comptime JOINT_2: Int = Self.joint_2
    comptime JOINT_3: Int = Self.joint_3
    comptime COEF_0: Float64 = Self.coef_0
    comptime COEF_1: Float64 = Self.coef_1
    comptime COEF_2: Float64 = Self.coef_2
    comptime COEF_3: Float64 = Self.coef_3
    comptime LENGTH_REF: Float64 = Self.length_ref
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4


# =============================================================================
# Tendons — variadic tendon list
# =============================================================================


@fieldwise_init
struct Tendons[*T: TendonSpec]:
    """Compile-time list of tendon specifications.

    Provides N (tendon count) for MAX_TENDON sizing.
    """

    comptime tendon_types = Variadic.types[T=TendonSpec, *Self.T]
    comptime N: Int = Variadic.size(Self.tendon_types)

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        """Populate model tendons from compile-time specs."""

        comptime for i in range(Self.N):
            comptime T_item = Self.tendon_types[i]

            model.tendons[i] = TendonDef[DTYPE](
                num_joints=T_item.NUM_JOINTS,
                joint_idx_0=T_item.JOINT_0,
                joint_idx_1=T_item.JOINT_1,
                joint_idx_2=T_item.JOINT_2,
                joint_idx_3=T_item.JOINT_3,
                coef_0=Scalar[DTYPE](T_item.COEF_0),
                coef_1=Scalar[DTYPE](T_item.COEF_1),
                coef_2=Scalar[DTYPE](T_item.COEF_2),
                coef_3=Scalar[DTYPE](T_item.COEF_3),
                length_ref=Scalar[DTYPE](T_item.LENGTH_REF),
                solref_0=Scalar[DTYPE](T_item.SOLREF_0),
                solref_1=Scalar[DTYPE](T_item.SOLREF_1),
                solimp_0=Scalar[DTYPE](T_item.SOLIMP_0),
                solimp_1=Scalar[DTYPE](T_item.SOLIMP_1),
                solimp_2=Scalar[DTYPE](T_item.SOLIMP_2),
                solimp_3=Scalar[DTYPE](T_item.SOLIMP_3),
                solimp_4=Scalar[DTYPE](T_item.SOLIMP_4),
            )
        model.num_tendons = Self.N
