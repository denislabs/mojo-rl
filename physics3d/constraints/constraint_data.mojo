"""Unified constraint representation for physics3d solvers.

ConstraintRow and ConstraintData provide a solver-agnostic representation
of all constraint types (contact normals, friction, joint limits, equality).
The constraint builder populates this data, and solvers consume it as pure
iterative algorithms.

Constraint types:
- CNSTR_NORMAL (0): Contact normal constraint (lambda >= 0)
- CNSTR_FRICTION_T1 (1): Friction tangent 1 (Coulomb cone)
- CNSTR_FRICTION_T2 (2): Friction tangent 2 (Coulomb cone)
- CNSTR_LIMIT (3): Joint limit constraint (lambda >= 0)
- CNSTR_FRICTION_TORSION (4): Torsional friction (condim >= 4)
- CNSTR_FRICTION_ROLL1 (5): Rolling friction 1 (condim == 6)
- CNSTR_FRICTION_ROLL2 (6): Rolling friction 2 (condim == 6)
- CNSTR_PYRAMID_EDGE (7): Pyramidal cone edge constraint (lambda >= 0)
- CNSTR_EQUALITY_CONNECT (8): Equality connect constraint (bilateral, 3 rows)
- CNSTR_EQUALITY_WELD (9): Equality weld constraint (bilateral, 6 rows)
- CNSTR_EQUALITY_TENDON (10): Fixed tendon equality constraint (bilateral, 1 row)
"""

from ..types import _max_one

# Constraint type constants
comptime CNSTR_NORMAL: Int = 0
comptime CNSTR_FRICTION_T1: Int = 1
comptime CNSTR_FRICTION_T2: Int = 2
comptime CNSTR_LIMIT: Int = 3
comptime CNSTR_FRICTION_TORSION: Int = 4
comptime CNSTR_FRICTION_ROLL1: Int = 5
comptime CNSTR_FRICTION_ROLL2: Int = 6
comptime CNSTR_PYRAMID_EDGE: Int = 7
comptime CNSTR_EQUALITY_CONNECT: Int = 8
comptime CNSTR_EQUALITY_WELD: Int = 9
comptime CNSTR_EQUALITY_TENDON: Int = 10


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
    var diagApprox: Scalar[
        Self.DTYPE
    ]  # MuJoCo body_invweight0 diagonal approximation for D/R

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
        self.diagApprox = Scalar[Self.DTYPE](0)


struct ConstraintData[DTYPE: DType, MAX_ROWS: Int, NV: Int]:
    """Pre-built constraint data consumed by solvers.

    Contains all constraint rows plus dense Jacobian and MinvJT matrices.
    The builder fills this data, and solvers iterate over it.

    Layout:
    - rows[0..num_normals): contact normal constraints
    - rows[num_normals..num_normals+num_friction): friction constraints (paired t1/t2)
    - rows[num_normals+num_friction..+num_limits): joint limit constraints
    - rows[..+num_equality): equality constraints (connect/weld, bilateral)

    J and MinvJT are stored row-major: row r spans [r*NV .. (r+1)*NV).

    M_hat and qfrc_smooth are filled by the integrator before calling solve().
    They are used by primal solvers (NewtonSolver, CGSolver) which
    operate in qacc space rather than dual (force) space.

    rows, J, MinvJT are heap-allocated (List) to avoid stack overflow when
    MAX_ROWS is large (e.g. 232 for Hopper with MAX_CONTACTS=20).
    M_hat and qfrc_smooth stay on the stack (small: NV*NV and NV).
    """

    var rows: List[ConstraintRow[Self.DTYPE]]
    var J: List[Scalar[Self.DTYPE]]
    var MinvJT: List[Scalar[Self.DTYPE]]
    # Mass matrix (with armature + implicit damping) — for primal solvers
    var M_hat: List[Scalar[Self.DTYPE]]
    # Net unconstrained force (qfrc - bias - passive) — for primal solvers
    var qfrc_smooth: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()]
    var num_rows: Int
    var num_normals: Int  # Normal contact constraints [0..num_normals)
    var num_friction: Int  # Friction rows [num_normals..num_normals+num_friction)
    var num_limits: Int  # Limit rows [num_normals+num_friction..)
    var num_equality: Int  # Equality rows [after limits..)

    fn __init__(out self):
        comptime MR = _max_one[Self.MAX_ROWS]()
        comptime JSize = _max_one[Self.MAX_ROWS * Self.NV]()
        comptime MSize = _max_one[Self.NV * Self.NV]()
        comptime VSize = _max_one[Self.NV]()
        # Heap-allocate rows, J, MinvJT to avoid stack overflow for large models
        self.rows = List[ConstraintRow[Self.DTYPE]](capacity=MR)
        for _ in range(MR):
            self.rows.append(ConstraintRow[Self.DTYPE]())
        self.J = List[Scalar[Self.DTYPE]](capacity=JSize)
        for _ in range(JSize):
            self.J.append(Scalar[Self.DTYPE](0))
        self.MinvJT = List[Scalar[Self.DTYPE]](capacity=JSize)
        for _ in range(JSize):
            self.MinvJT.append(Scalar[Self.DTYPE](0))
        self.M_hat = List[Scalar[Self.DTYPE]](capacity=MSize)
        for _ in range(MSize):
            self.M_hat.append(Scalar[Self.DTYPE](0))

        self.qfrc_smooth = InlineArray[Scalar[Self.DTYPE], VSize](
            fill=Scalar[Self.DTYPE](0)
        )
        self.num_rows = 0
        self.num_normals = 0
        self.num_friction = 0
        self.num_limits = 0
        self.num_equality = 0
