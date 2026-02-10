"""Unified constraint representation for physics3d solvers.

ConstraintRow and ConstraintData provide a solver-agnostic representation
of all constraint types (contact normals, friction, joint limits). The
constraint builder populates this data, and solvers consume it as pure
iterative algorithms.

Constraint types:
- CNSTR_NORMAL (0): Contact normal constraint (lambda >= 0)
- CNSTR_FRICTION_T1 (1): Friction tangent 1 (Coulomb cone)
- CNSTR_FRICTION_T2 (2): Friction tangent 2 (Coulomb cone)
- CNSTR_LIMIT (3): Joint limit constraint (lambda >= 0)
"""

from ..types import _max_one

# Constraint type constants
comptime CNSTR_NORMAL: Int = 0
comptime CNSTR_FRICTION_T1: Int = 1
comptime CNSTR_FRICTION_T2: Int = 2
comptime CNSTR_LIMIT: Int = 3


@fieldwise_init
struct ConstraintRow[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """A single constraint row.

    Stores all per-constraint data needed by solvers: Jacobian index,
    effective mass, impedance bias, bounds, impulse, and metadata for
    friction coupling and impulse writeback.
    """

    var K: Scalar[Self.DTYPE]  # Effective mass J @ M_inv @ J^T
    var bias: Scalar[
        Self.DTYPE
    ]  # RHS (impedance position + velocity correction)
    var inv_K_imp: Scalar[
        Self.DTYPE
    ]  # imp / K (for PGS: delta = -(v+bias) * inv_K_imp)
    var lo: Scalar[Self.DTYPE]  # Impulse lower bound (0 for contacts/limits)
    var hi: Scalar[Self.DTYPE]  # Impulse upper bound (1e20)
    var lambda_val: Scalar[
        Self.DTYPE
    ]  # Current impulse (warm-started, modified by solver)
    var constraint_type: Int  # CNSTR_NORMAL, CNSTR_FRICTION_T1/T2, CNSTR_LIMIT
    var friction_parent: Int  # Index of normal row for friction coupling (-1 if N/A)
    var friction_coef: Scalar[Self.DTYPE]  # mu for Coulomb cone
    var source_contact_idx: Int  # For impulse writeback (-1 for limits)
    var source_dof: Int  # For limit constraints (-1 for contacts)
    var limit_sign: Scalar[
        Self.DTYPE
    ]  # +1 for lower limit, -1 for upper limit (0 for contacts)

    fn __init__(out self):
        self.K = Scalar[Self.DTYPE](1)
        self.bias = Scalar[Self.DTYPE](0)
        self.inv_K_imp = Scalar[Self.DTYPE](0)
        self.lo = Scalar[Self.DTYPE](0)
        self.hi = Scalar[Self.DTYPE](1e20)
        self.lambda_val = Scalar[Self.DTYPE](0)
        self.constraint_type = CNSTR_NORMAL
        self.friction_parent = -1
        self.friction_coef = Scalar[Self.DTYPE](0)
        self.source_contact_idx = -1
        self.source_dof = -1
        self.limit_sign = Scalar[Self.DTYPE](0)


struct ConstraintData[DTYPE: DType, MAX_ROWS: Int, NV: Int]:
    """Pre-built constraint data consumed by solvers.

    Contains all constraint rows plus dense Jacobian and MinvJT matrices.
    The builder fills this data, and solvers iterate over it.

    Layout:
    - rows[0..num_normals): contact normal constraints
    - rows[num_normals..num_normals+num_friction): friction constraints (paired t1/t2)
    - rows[num_normals+num_friction..num_rows): joint limit constraints

    J and MinvJT are stored row-major: row r spans [r*NV .. (r+1)*NV).
    """

    var rows: InlineArray[ConstraintRow[Self.DTYPE], _max_one[Self.MAX_ROWS]()]
    var J: InlineArray[Scalar[Self.DTYPE], _max_one[Self.MAX_ROWS * Self.NV]()]
    var MinvJT: InlineArray[
        Scalar[Self.DTYPE], _max_one[Self.MAX_ROWS * Self.NV]()
    ]
    var num_rows: Int
    var num_normals: Int  # Normal contact constraints [0..num_normals)
    var num_friction: Int  # Friction rows [num_normals..num_normals+num_friction)
    var num_limits: Int  # Limit rows [num_normals+num_friction..)

    fn __init__(out self):
        comptime MR = _max_one[Self.MAX_ROWS]()
        comptime JSize = _max_one[Self.MAX_ROWS * Self.NV]()
        self.rows = InlineArray[ConstraintRow[Self.DTYPE], MR](
            ConstraintRow[Self.DTYPE]()
        )
        self.J = InlineArray[Scalar[Self.DTYPE], JSize](Scalar[Self.DTYPE](0))
        self.MinvJT = InlineArray[Scalar[Self.DTYPE], JSize](
            Scalar[Self.DTYPE](0)
        )
        self.num_rows = 0
        self.num_normals = 0
        self.num_friction = 0
        self.num_limits = 0
