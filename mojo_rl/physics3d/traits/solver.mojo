"""Constraint solver trait for Generalized Coordinates (GC) engine.

ConstraintSolver defines the interface for constraint-based contact solving
in joint space. Implementations receive pre-built ConstraintData and modify
the acceleration vector in-place to satisfy constraints (acceleration-level solving).

The constraint builder populates ConstraintData (normals, friction, limits)
before the solver is called. Solvers are pure iterative algorithms.
"""

from layout import LayoutTensor, Layout

from ..types import Model, Data
from ..constraints.constraint_data import ConstraintData


trait ConstraintSolver(Movable & ImplicitlyCopyable):
    """Trait for constraint-based contact solvers in GC engine.

    Solvers modify the acceleration vector in-place to satisfy contact
    constraints (non-penetration + friction) and joint limits.

    The solve() method receives:
    - model/data: physics model and current state (for force writeback)
    - M_inv: full dense inverse mass matrix (NV×NV)
    - constraints: pre-built ConstraintData with all constraints
    - qacc: acceleration vector (modified in-place by solver)
    - dt: timestep
    """

    # Whether this solver needs pre-computed M_inv (full NV×NV inverse).
    # Newton solver: False (uses LDL solve directly).
    # PGS/CG/Island solvers: True (read M_inv for constraint projection).
    comptime NEEDS_M_INV: Bool

    @staticmethod
    def solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Solver-specific workspace size in floats per environment.

        This workspace is allocated after M_inv (NV*NV) in the per-environment
        workspace buffer. Each solver declares how much device memory it needs
        for its MC-sized arrays (moved out of registers to reduce spilling).
        """
        ...

    @staticmethod
    def solver_threads[
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ]() -> Int:
        """Number of threads for parallelization.

        This is the maximum number of threads that can be used for parallelization.
        """
        ...

    @staticmethod
    def solve[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        MAX_ROWS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        model: Model[
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
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        M_inv: List[Scalar[DTYPE]],
        mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
        mut qacc: List[Scalar[DTYPE]],
        dt: Scalar[DTYPE],
    ):
        """Solve constraints on CPU.

        Args:
            model: Static model configuration.
            data: Mutable simulation state. Forces written back for warm-starting.
            M_inv: Full dense inverse mass matrix (NV×NV, row-major).
            constraints: Pre-built constraint data (normals, friction, limits).
            qacc: Acceleration vector, modified in-place to satisfy constraints.
            dt: Timestep.
        """
        ...

    @staticmethod
    def solve_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        V_SIZE: Int,
        BATCH: Int,
        WS_SIZE: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
    ):
        """Solve contact constraints on GPU (per-environment).

        Args:
            state: GPU state buffer for all environments.
            model: GPU model buffer.
            workspace: Device memory workspace [BATCH, WS_SIZE] containing
                integrator temps, M_inv, and solver arrays per environment.
                Reads cdof and qacc from workspace, modifies qacc
                in-place to satisfy constraints (acceleration-level).
        """
        ...
