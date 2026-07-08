"""EqualitySpec trait and concrete equality constraint types.

Supports two equality constraint types:
  - ConnectConstraint: Point-to-point ball joint (3 DOF, position only)
  - WeldConstraint: Rigid attachment (6 DOF, position + orientation)

Usage:
    from mojo_rl.physics3d.model.equality_spec import EqualitySpec, ConnectConstraint, WeldConstraint

    # Ball joint connecting body 0 and body 1 at their origins
    comptime MyConnect = ConnectConstraint[body_a=0, body_b=1]

    # Weld body 2 to body 3 at specified anchors
    comptime MyWeld = WeldConstraint[body_a=2, body_b=3, anchor_a_z=0.1]
"""

from ..types import EQ_CONNECT, EQ_WELD, EqualityConstraintDef, ConeType


trait EqualitySpec:
    """Compile-time specification for an equality constraint."""

    comptime EQ_TYPE: Int  # EQ_CONNECT or EQ_WELD
    comptime BODY_A: Int  # First body index
    comptime BODY_B: Int  # Second body index (0 for worldbody)
    # Anchor point in body_a frame
    comptime ANCHOR_A_X: Float64
    comptime ANCHOR_A_Y: Float64
    comptime ANCHOR_A_Z: Float64
    # Anchor point in body_b frame (or world frame if BODY_B == 0)
    comptime ANCHOR_B_X: Float64
    comptime ANCHOR_B_Y: Float64
    comptime ANCHOR_B_Z: Float64
    # Relative orientation quaternion [x, y, z, w] (weld only)
    comptime RELPOSE_X: Float64
    comptime RELPOSE_Y: Float64
    comptime RELPOSE_Z: Float64
    comptime RELPOSE_W: Float64
    # Impedance parameters
    comptime SOLREF_0: Float64  # timeconst
    comptime SOLREF_1: Float64  # dampratio
    comptime SOLIMP_0: Float64  # dmin
    comptime SOLIMP_1: Float64  # dmax
    comptime SOLIMP_2: Float64  # width
    comptime SOLIMP_3: Float64  # midpoint
    comptime SOLIMP_4: Float64  # power
    # Number of constraint rows (3 for connect, 6 for weld)
    comptime NUM_ROWS: Int


@fieldwise_init
struct ConnectConstraint[
    body_a: Int,
    body_b: Int = 0,
    anchor_a_x: Float64 = 0.0,
    anchor_a_y: Float64 = 0.0,
    anchor_a_z: Float64 = 0.0,
    anchor_b_x: Float64 = 0.0,
    anchor_b_y: Float64 = 0.0,
    anchor_b_z: Float64 = 0.0,
    solref_0: Float64 = 0.02,
    solref_1: Float64 = 1.0,
    solimp_0: Float64 = 0.9,
    solimp_1: Float64 = 0.95,
    solimp_2: Float64 = 0.001,
    solimp_3: Float64 = 0.5,
    solimp_4: Float64 = 2.0,
](EqualitySpec):
    """Connect (ball joint) equality constraint — 3 position rows."""

    comptime EQ_TYPE: Int = EQ_CONNECT
    comptime BODY_A: Int = Self.body_a
    comptime BODY_B: Int = Self.body_b
    comptime ANCHOR_A_X: Float64 = Self.anchor_a_x
    comptime ANCHOR_A_Y: Float64 = Self.anchor_a_y
    comptime ANCHOR_A_Z: Float64 = Self.anchor_a_z
    comptime ANCHOR_B_X: Float64 = Self.anchor_b_x
    comptime ANCHOR_B_Y: Float64 = Self.anchor_b_y
    comptime ANCHOR_B_Z: Float64 = Self.anchor_b_z
    comptime RELPOSE_X: Float64 = 0.0
    comptime RELPOSE_Y: Float64 = 0.0
    comptime RELPOSE_Z: Float64 = 0.0
    comptime RELPOSE_W: Float64 = 1.0
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime NUM_ROWS: Int = 3


@fieldwise_init
struct WeldConstraint[
    body_a: Int,
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
](EqualitySpec):
    """Weld (rigid attachment) equality constraint — 6 rows (3 position + 3 orientation).
    """

    comptime EQ_TYPE: Int = EQ_WELD
    comptime BODY_A: Int = Self.body_a
    comptime BODY_B: Int = Self.body_b
    comptime ANCHOR_A_X: Float64 = Self.anchor_a_x
    comptime ANCHOR_A_Y: Float64 = Self.anchor_a_y
    comptime ANCHOR_A_Z: Float64 = Self.anchor_a_z
    comptime ANCHOR_B_X: Float64 = Self.anchor_b_x
    comptime ANCHOR_B_Y: Float64 = Self.anchor_b_y
    comptime ANCHOR_B_Z: Float64 = Self.anchor_b_z
    comptime RELPOSE_X: Float64 = Self.relpose_x
    comptime RELPOSE_Y: Float64 = Self.relpose_y
    comptime RELPOSE_Z: Float64 = Self.relpose_z
    comptime RELPOSE_W: Float64 = Self.relpose_w
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime NUM_ROWS: Int = 6


# =============================================================================
# Equalities — variadic equality constraint list
# =============================================================================


@fieldwise_init
struct Equalities[*E: EqualitySpec]:
    """Compile-time list of equality constraint specifications.

    Provides N (constraint count) and _sum_rows() for total row count.
    """

    comptime eq_types = Self.E
    comptime N: Int = Self.eq_types.size

    @staticmethod
    def _sum_rows() -> Int:
        """Sum NUM_ROWS across all equality constraints (total constraint rows).
        """
        var total = 0

        comptime for i in range(Self.N):
            total += Self.eq_types[i].NUM_ROWS
        return total

    @staticmethod
    def setup_model[
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
            Self.N,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        """Populate model equality constraints from compile-time specs."""

        comptime for i in range(Self.N):
            comptime E_item = Self.eq_types[i]

            model.equality_constraints[i] = EqualityConstraintDef[DTYPE](
                eq_type=E_item.EQ_TYPE,
                body_a=E_item.BODY_A,
                body_b=E_item.BODY_B,
                anchor_a_x=Scalar[DTYPE](E_item.ANCHOR_A_X),
                anchor_a_y=Scalar[DTYPE](E_item.ANCHOR_A_Y),
                anchor_a_z=Scalar[DTYPE](E_item.ANCHOR_A_Z),
                anchor_b_x=Scalar[DTYPE](E_item.ANCHOR_B_X),
                anchor_b_y=Scalar[DTYPE](E_item.ANCHOR_B_Y),
                anchor_b_z=Scalar[DTYPE](E_item.ANCHOR_B_Z),
                relpose_x=Scalar[DTYPE](E_item.RELPOSE_X),
                relpose_y=Scalar[DTYPE](E_item.RELPOSE_Y),
                relpose_z=Scalar[DTYPE](E_item.RELPOSE_Z),
                relpose_w=Scalar[DTYPE](E_item.RELPOSE_W),
                solref_0=Scalar[DTYPE](E_item.SOLREF_0),
                solref_1=Scalar[DTYPE](E_item.SOLREF_1),
                solimp_0=Scalar[DTYPE](E_item.SOLIMP_0),
                solimp_1=Scalar[DTYPE](E_item.SOLIMP_1),
                solimp_2=Scalar[DTYPE](E_item.SOLIMP_2),
                solimp_3=Scalar[DTYPE](E_item.SOLIMP_3),
                solimp_4=Scalar[DTYPE](E_item.SOLIMP_4),
            )
        model.num_equality = Self.N
