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

    def __init__(out self):
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

    def __init__(out self):
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


# =============================================================================
# solref -> (K, B) — the ONE place this conversion happens
# =============================================================================


@always_inline
def solref_spring_damper[
    DTYPE: DType
](
    ref_tc_raw: Scalar[DTYPE],
    ref_dr: Scalar[DTYPE],
    d_width: Scalar[DTYPE],
    timestep: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE]]:
    """MuJoCo's `solref` -> constraint spring/damper, INCLUDING the direct form.

    Port of `engine_core_constraint.c:1845-1862`. Returns `(K, B)` for
    `aref = -B*vel - K*imp*pos`.

        ref[0] = max(ref[0], 2*timestep)   if ref[0] > 0   (REFSAFE)
        K = ref[0] > 0 : 1 / (d_width^2 * ref[0]^2 * ref[1]^2)   standard
                       : -ref[0] / d_width^2                     direct
        B = ref[1] > 0 : 2 / (d_width * ref[0])                  standard
                       : -ref[1] / d_width                       direct

    ⚠ **A NEGATIVE `solref` IS NOT AN ERROR, IT SELECTS A DIFFERENT MEANING.**
    With `solref[0] >= 0` the pair is `(timeconst, dampratio)`; with it negative
    MuJoCo reads `(-stiffness, -damping)` and uses them directly. Until
    2026-08-03 this codebase implemented only the standard form, in TWELVE
    copy-pasted sites, so a negative solref produced
      * `K` about 1e-8 of its correct value — effectively no stiffness; and
      * `B` NEGATIVE, i.e. a damper that INJECTS energy rather than removing it.
    dm_control's quadruped `fetch` is the first model in the tree to use one
    (`<geom name="ball" solref="-10000 -30"/>`), which is why it never bit.

    ⚠ **THE TWO BRANCHES ARE SELECTED INDEPENDENTLY** — `K` on `ref[0]`, `B` on
    `ref[1]` — which is what the C does, so this transcribes it. It is NOT
    because a mixed-sign solref can reach here: MuJoCo's COMPILER rejects one
    outright ("WARNING: mixed solref format, replacing with default", measured),
    so both components always share a sign by the time the solver sees them.
    An earlier draft of this docstring claimed a model could legitimately mix
    them and justified the two branches on that; the branches are right and the
    justification was wrong.

    ⚠ **THE STANDARD `B` BRANCH IS SELECTED BY `ref[1]` BUT COMPUTED FROM
    `ref[0]`.** That is not a transcription slip; it is what MuJoCo does.

    THIS FUNCTION EXISTS BECAUSE THE FORMULA WAS COPY-PASTED TWELVE TIMES —
    `contact_solve`, `equality_tendon` (x2), `limits`, `scalar_rows`,
    `friction_dof`, `cg_solve`, `island_pgs_solve` and `newton_solve` (x4). That
    is the same shape as bug 21, where a missing `solimp` clamp lived in six
    copied sites and survived three investigations. Adding a thirteenth copy is
    how the next one gets missed: call this instead.

    ⚠ FRICTION ROWS TAKE `K = 0` (`mjCNSTR_FRICTION_*` and elliptic friction,
    same source lines) — that branch is the CALLER's, not this function's, and
    `friction_dof.mojo` already implements it. Do not route a friction row's K
    through here expecting a zero.

    `d_width` is `solimp[1]` (dmax), already clamped to [mjMINIMP, mjMAXIMP] by
    the caller — this function does not clamp it, because the callers do it in
    the same breath as clamping dmin and power.

    VERIFIED against `mjData.efc_KBIP`, which exposes MuJoCo's own K and B:

        contact solref   dmax    MuJoCo K, B            ours
        (0.02, 1)        0.95     2770.0831 105.2632    identical
        (0.0125, 0.75)   0.95    12606.9560 168.4211    identical
        (0.005, 0.5)     0.95   177285.3186 421.0526    identical
        (-10000, -30)    0.95    11080.3324  31.5789    identical   <- direct

    ⚠ **THIS FUNCTION IS NECESSARY BUT NOT SUFFICIENT FOR A PER-GEOM `solref`.**
    Contact rows currently read ONE MODEL-LEVEL solref
    (`mmeta[MODEL_META_IDX_SOLREF_CONTACT_0/1]`), so a geom's own `solref` never
    reaches here at all — the contact record carries per-contact friction,
    condim and margin but no solref/solimp. Supporting
    `<geom solref="-10000 -30"/>` needs those two added to the record and mixed
    in the narrow phase by MuJoCo's rule, measured as:
      * priorities differ  -> the higher-priority geom's parameters, wholesale;
      * priorities equal, BOTH solrefs positive -> elementwise MEAN;
      * priorities equal, EITHER negative       -> elementwise MIN.
    (That last rule is why a direct solref wins over a standard one even at
    equal priority.)
    """
    comptime MINVAL = Scalar[DTYPE](1e-15)

    # ⚠ REFSAFE: `timeconst` is raised to 2*timestep before anything uses it
    # (`engine_core_constraint.c:2028`, "integrator safety", active unless
    # mjDSBL_REFSAFE — and that flag is OFF by default, so this ALWAYS applies).
    # It is the DIRECT form (`ref_tc <= 0`) that is exempt: MuJoCo guards the
    # clamp with `solref[0] > 0`, so a negative solref passes through as the
    # stiffness it literally is.
    #
    # Missing this made quadruped 4x too stiff on its four LIVE equality rows:
    # eq_solref 0.005 against dt 0.005, where MuJoCo's efc_KBIP reads 40812.16
    # (the clamped 1/(0.99^2 * 0.01^2)) and we computed 163248.65.
    #
    # ⚠ `timestep` IS REQUIRED, not defaulted. A default of 0 would make this
    # a silent no-op at any call site that forgot to pass it, which is the
    # half-fix shape that produced defect 22 and the twelve-way copy-paste this
    # function exists to end. The compiler enumerates the call sites instead.
    var ref_tc = ref_tc_raw
    if ref_tc > Scalar[DTYPE](0):
        var two_dt = Scalar[DTYPE](2.0) * timestep
        if ref_tc < two_dt:
            ref_tc = two_dt

    var k_den = d_width * d_width * ref_tc * ref_tc * ref_dr * ref_dr
    var k_out: Scalar[DTYPE]
    if ref_tc > Scalar[DTYPE](0):
        k_out = Scalar[DTYPE](1.0) / (k_den if k_den > MINVAL else MINVAL)
    else:
        var dw2 = d_width * d_width
        k_out = -ref_tc / (dw2 if dw2 > MINVAL else MINVAL)

    var b_out: Scalar[DTYPE]
    if ref_dr > Scalar[DTYPE](0):
        var b_den = d_width * ref_tc
        b_out = Scalar[DTYPE](2.0) / (b_den if b_den > MINVAL else MINVAL)
    else:
        b_out = -ref_dr / (d_width if d_width > MINVAL else MINVAL)

    return (k_out, b_out)


@always_inline
def refsafe_timeconst[
    DTYPE: DType
](ref_tc: Scalar[DTYPE], timestep: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """MuJoCo's REFSAFE clamp on its own — `max(ref_tc, 2*timestep)`.

    `solref_spring_damper` applies this internally for constraint rows. The dry
    FRICTION rows do not go through it — their `K` is identically 0, so they
    compute only `B = 2/(dmax*timeconst)` inline from the HARDCODED default
    `DOF_SOLREF_TIMECONST = 0.02` — and MuJoCo clamps that default exactly the
    same way (`engine_core_constraint.c:2039`, the `solreffriction` twin of the
    :2028 clamp). This exists so those three sites can apply the rule without a
    fourth copy of the arithmetic.

    ⚠ INERT ON EVERY MODEL IN THE REPO TODAY, and that is measured, not
    assumed: `finger` is the only suite model with `frictionloss > 0`, and its
    timestep is 0.01, so `2*dt` is exactly 0.02 and the clamp changes nothing.
    It bites the first model that combines `frictionloss` with a timestep
    above 0.01 — MuJoCo would then use `2*dt` where we would have used 0.02,
    and `B` scales as 1/timeconst.

    ⚠ ONLY THE STANDARD FORM IS CLAMPED. A non-positive `ref_tc` is the direct
    (stiffness) form and passes through, matching MuJoCo's `solref[0] > 0`
    guard.
    """
    if ref_tc <= Scalar[DTYPE](0):
        return ref_tc
    var two_dt = Scalar[DTYPE](2.0) * timestep
    return two_dt if ref_tc < two_dt else ref_tc
