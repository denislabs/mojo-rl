"""Equality (connect/weld) + fixed-tendon constraints over per-field tensors
(migration P4, single-source).

Per-field ports of `build_and_solve_equality_gpu`
(constraints/constraint_builder_gpu.mojo:1030) and
`build_and_solve_tendon_gpu` (:1573), plus the weld/angular Jacobian rows
they call (dynamics/jacobian.mojo: `compute_weld_jacobian_row_gpu`,
`compute_angular_jacobian_row_gpu`) — arithmetic verbatim. Both are per-env
functions called by the fields PGS contact solve
(constraints/contact_solve.mojo) at the exact legacy position: after
the joint-limits pass, before the friction phase, with the legacy
PGS_ITERATIONS iteration count.

Legacy invweight0-offset quirk (reproduced bit-exactly — port the CODE, not
the algorithm): the legacy builders compute their diagApprox offsets with
NTENDON/NSITE left at their 0 defaults —
`model_body_invweight0_offset[NBODY, NJOINT, NGEOM, MAX_EQUALITY]()` in the
equality builder and `model_dof_invweight0_offset[NBODY, NJOINT, NGEOM,
MAX_EQUALITY]()` in the tendon builder — so on any model with tendons
and/or sites the reads land NTENDON*MODEL_TENDON_SIZE +
NSITE*MODEL_SITE_SIZE BEFORE the true invweight0 records, i.e. inside the
tendon / site records (Humanoid's tendon builder, for example, reads its
diag from body_invweight0 entries of unrelated bodies). `_legacy_invw_read`
reproduces that addressing exactly by mapping the legacy slab offset
(relative to the tendon-records start, which is what the legacy shift-free
base equals) onto the concatenated
[tendons | sites | body_invweight0 | dof_invweight0] record tensors."""

from std.math import abs, pow
from layout import Layout, LayoutTensor

from ..types import _max_one, EQ_WELD
from ..joint_types import JNT_FREE, JNT_BALL
from ..kinematics.quat_math import quat_mul, quat_conjugate, quat_rotate
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_IS_EQUALITY,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
    TENDON_MAX_WRAPS,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    TENDON_IDX_INVWEIGHT0,
    TENDON_IDX_KIND,
    TENDON_KIND_SPATIAL,
)


from .constraint_data import solref_spring_damper


@always_inline
def _legacy_tendon_col(c: Int) -> Int:
    """Legacy tendon-record column -> its column in the CURRENT record.

    The legacy record was 17 wide and laid out

        0        num_joints
        1..4     joint_0..3
        5..8     coef_0..3
        9        length_ref
        10..11   solref_0..1
        12..16   solimp_0..4

    `_legacy_invw_read` reproduces a historical misread BIT-EXACTLY, so it must
    keep naming those same quantities however the live record is arranged. It
    used the column index raw, which worked only while every layout change
    APPENDED. `TENDON_MAX_WRAPS` 4 -> 16 widened the joint and coef runs in
    place instead, so columns 5..16 changed meaning underneath it.

    Written as a mapping rather than a second frozen copy of the values
    because the point is to track the live record: if a wrap run moves again,
    this moves with it, and only a genuine reordering of the SCALAR fields
    would need an edit here.
    """
    if c <= 0:
        return TENDON_IDX_NUM_JOINTS
    if c <= 4:
        return TENDON_IDX_JOINT_0 + (c - 1)
    if c <= 8:
        return TENDON_IDX_COEF_0 + (c - 5)
    if c == 9:
        return TENDON_IDX_LENGTH_REF
    if c <= 11:
        return TENDON_IDX_SOLREF_0 + (c - 10)
    return TENDON_IDX_SOLIMP_0 + (c - 12)


@always_inline
def _legacy_invw_read[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NTENDON: Int,
    NSITE: Int,
](
    delta: Int,
    tendons: LayoutTensor[
        DTYPE, Layout.row_major(NTENDON, MODEL_TENDON_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    body_invweight0: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, 2), MutAnyOrigin
    ],
    dof_invweight0: LayoutTensor[DTYPE, Layout.row_major(NV), MutAnyOrigin],
) -> Scalar[DTYPE]:
    """Read the slab element the LEGACY equality/tendon builders address.

    The legacy diagApprox offsets omit NTENDON/NSITE, so their base equals
    the true tendon-records start; `delta` is the legacy offset from that
    base (`body*2` / `body*2+1` for the equality builder, `NBODY*2 +
    dof_adr` for the tendon builder). Mapped address-faithfully onto the
    record tensors so the value read is bit-identical to the legacy slab
    read (including the misreads on models with tendons/sites).

    ⚠ THE SLAB GEOMETRY IS PINNED TO THE HISTORICAL RECORD WIDTHS, not to the
    current `MODEL_TENDON_SIZE`. This function's entire contract is "reproduce
    the legacy addressing exactly", so it must not move when a record grows.
    It did move on 2026-07-31, when `MODEL_TENDON_SIZE` went 17 -> 36 for
    spatial tendons: `T_END` doubled, every `delta` landed in a different
    record, and `test_equality_tendon_fields`'s golden shifted 0.26% for a
    reason that had nothing to do with the change under test.

    Pinning is safe precisely because both growths APPENDED columns:
    `tendons[r, c]` for `c < 17` and `sites[r, c]` for `c < 8` still return
    exactly what the legacy slab held. (`MODEL_SITE_SIZE` went 8 -> 12 on
    2026-08-01 for the site quaternion; `LEGACY_SITE_SIZE` stayed 8, so this
    read did not move.) A record growth that REORDERS or INSERTS columns would
    break the contract silently — the values would still be finite and
    plausible, and only a golden would notice."""
    comptime LEGACY_TENDON_SIZE = 17
    comptime LEGACY_SITE_SIZE = 8
    comptime T_END = NTENDON * LEGACY_TENDON_SIZE
    comptime S_END = T_END + NSITE * LEGACY_SITE_SIZE
    comptime B_END = S_END + NBODY * 2
    if delta < T_END:
        # ⚠ THE COLUMN IS REMAPPED, NOT USED RAW — the paragraph above called
        # this exactly. `TENDON_MAX_WRAPS` 4 -> 16 did not append, it WIDENED
        # two runs in the middle: COEF_0 went 5 -> 17 and LENGTH_REF 9 -> 33,
        # so a raw `delta % 17` stopped naming the legacy quantity.
        #
        # ⚠ THIS WAS NOT THE CAUSE OF THE GOLDEN THAT MOVED, though it was
        # confidently reported as such at the time. Adding the remap left
        # `test_equality_tendon_fields`'s Part A fingerprint bit-identical
        # (-664336.7153001489) because this reader is reached only from the
        # weld/connect builders, not from the tendon path Part A exercises;
        # the real cause was 16-wide per-thread arrays in the kernel below.
        # The remap is kept because it is independently correct — the weld
        # path WOULD have read the wrong columns at width 16 — but it is a
        # latent fix, not the explanation. Recorded because a plausible
        # mechanism read off a docstring is not a measurement.
        return rebind[Scalar[DTYPE]](
            tendons[
                delta // LEGACY_TENDON_SIZE,
                _legacy_tendon_col(delta % LEGACY_TENDON_SIZE),
            ]
        )
    if delta < S_END:
        return rebind[Scalar[DTYPE]](
            sites[
                (delta - T_END) // LEGACY_SITE_SIZE,
                (delta - T_END) % LEGACY_SITE_SIZE,
            ]
        )
    if delta < B_END:
        return rebind[Scalar[DTYPE]](
            body_invweight0[(delta - S_END) // 2, (delta - S_END) % 2]
        )
    return rebind[Scalar[DTYPE]](dof_invweight0[delta - B_END])


# =============================================================================
# Weld / angular Jacobian rows (ports of dynamics/jacobian.mojo GPU rows)
# =============================================================================


@always_inline
def _weld_jacobian_row[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    V_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    body_a: Int,
    body_b: Int,
    pos_a_x: Scalar[DTYPE],
    pos_a_y: Scalar[DTYPE],
    pos_a_z: Scalar[DTYPE],
    pos_b_x: Scalar[DTYPE],
    pos_b_y: Scalar[DTYPE],
    pos_b_z: Scalar[DTYPE],
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Weld/connect Jacobian row: J = J_a(at pos_a) - J_b(at pos_b)
    (verbatim from compute_weld_jacobian_row_gpu).

    Each body's Jacobian uses its OWN anchor position, unlike the contact
    Jacobian which uses a single shared contact point.
    """
    for i in range(V_SIZE):
        J_row[i] = 0

    var num_joints = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT])
    )

    for j_idx in range(num_joints):
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_TYPE])
        )
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_DOF_ADR])
        )

        # Check if this joint affects body_a
        var affects_a = False
        if body_a == joint_body:
            affects_a = True
        else:
            var current = body_a
            while current > 0:
                var current_parent = Int(
                    rebind[Scalar[DTYPE]](bodies[current, BODY_IDX_PARENT])
                )
                if current_parent == joint_body:
                    affects_a = True
                    break
                current = current_parent

        # Check if this joint affects body_b
        var affects_b = False
        if body_b > 0:
            if body_b == joint_body:
                affects_b = True
            else:
                var current_b = body_b
                while current_b > 0:
                    var current_parent_b = Int(
                        rebind[Scalar[DTYPE]](
                            bodies[current_b, BODY_IDX_PARENT]
                        )
                    )
                    if current_parent_b == joint_body:
                        affects_b = True
                        break
                    current_b = current_parent_b

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        # Reference point for cdof cross product
        var jb_rootid = Int(
            rebind[Scalar[DTYPE]](bodies[joint_body, BODY_IDX_ROOTID])
        )
        var ref_x = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 0]
        )
        var ref_y = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 1]
        )
        var ref_z = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 2]
        )

        for d in range(num_dof):
            var dof_idx = dof_adr + d
            var ang_x = cdof[env, dof_idx * 6 + 0]
            var ang_y = cdof[env, dof_idx * 6 + 1]
            var ang_z = cdof[env, dof_idx * 6 + 2]
            var lin_x = cdof[env, dof_idx * 6 + 3]
            var lin_y = cdof[env, dof_idx * 6 + 4]
            var lin_z = cdof[env, dof_idx * 6 + 5]

            if affects_a:
                # Jacobian at body_a's anchor point
                var ra_x = pos_a_x - ref_x
                var ra_y = pos_a_y - ref_y
                var ra_z = pos_a_z - ref_z
                var cx = ang_y * ra_z - ang_z * ra_y
                var cy = ang_z * ra_x - ang_x * ra_z
                var cz = ang_x * ra_y - ang_y * ra_x
                var val = (lin_x + cx) * dir_x + (lin_y + cy) * dir_y + (lin_z + cz) * dir_z
                J_row[dof_idx] += rebind[Scalar[DTYPE]](val)

            if affects_b:
                # Jacobian at body_b's anchor point (separate!)
                var rb_x = pos_b_x - ref_x
                var rb_y = pos_b_y - ref_y
                var rb_z = pos_b_z - ref_z
                var cx = ang_y * rb_z - ang_z * rb_y
                var cy = ang_z * rb_x - ang_x * rb_z
                var cz = ang_x * rb_y - ang_y * rb_x
                var val = (lin_x + cx) * dir_x + (lin_y + cy) * dir_y + (lin_z + cz) * dir_z
                J_row[dof_idx] -= rebind[Scalar[DTYPE]](val)


@always_inline
def _angular_jacobian_row_eq[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    V_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    contact_body_a: Int,
    contact_body_b: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Angular-only Jacobian row (verbatim from
    compute_angular_jacobian_row_gpu; duplicate of
    contact_solve._angular_jacobian_row — that module imports
    THIS one, so importing back would be circular).

    J[dof] = cdof_angular[dof] . dir (bilateral: body_a - body_b).
    """
    for i in range(V_SIZE):
        J_row[i] = 0

    var num_joints = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT])
    )

    for j_idx in range(num_joints):
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_TYPE])
        )
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_DOF_ADR])
        )

        # Check if this joint affects body_a
        var affects_a = False
        if contact_body_a == joint_body:
            affects_a = True
        else:
            var current = contact_body_a
            while current > 0:
                var current_parent = Int(
                    rebind[Scalar[DTYPE]](bodies[current, BODY_IDX_PARENT])
                )
                if current_parent == joint_body:
                    affects_a = True
                    break
                current = current_parent

        # Check if this joint affects body_b (only if body_b > 0, i.e. not ground)
        var affects_b = False
        if contact_body_b > 0:
            if contact_body_b == joint_body:
                affects_b = True
            else:
                var current_b = contact_body_b
                while current_b > 0:
                    var current_parent_b = Int(
                        rebind[Scalar[DTYPE]](
                            bodies[current_b, BODY_IDX_PARENT]
                        )
                    )
                    if current_parent_b == joint_body:
                        affects_b = True
                        break
                    current_b = current_parent_b

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var dof_idx = dof_adr + d

            # Angular-only: just dot product of angular cdof with direction
            var ang_x = cdof[env, dof_idx * 6 + 0]
            var ang_y = cdof[env, dof_idx * 6 + 1]
            var ang_z = cdof[env, dof_idx * 6 + 2]

            var val = ang_x * dir_x + ang_y * dir_y + ang_z * dir_z

            if affects_a:
                J_row[dof_idx] += rebind[Scalar[DTYPE]](val)
            if affects_b:
                J_row[dof_idx] -= rebind[Scalar[DTYPE]](val)


# =============================================================================
# Equality constraints (port of build_and_solve_equality_gpu)
# =============================================================================


@always_inline
def _equality_env[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    V_SIZE: Int,
    BATCH: Int,
    NUM_ITERATIONS: Int,
](
    env: Int,
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    equality: LayoutTensor[
        DTYPE, Layout.row_major(NEQUALITY, MODEL_EQ_SIZE), MutAnyOrigin
    ],
    tendons: LayoutTensor[
        DTYPE, Layout.row_major(NTENDON, MODEL_TENDON_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    body_invweight0: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, 2), MutAnyOrigin
    ],
    dof_invweight0: LayoutTensor[DTYPE, Layout.row_major(NV), MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    m_inv: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
):
    """Build and solve equality constraints (connect + weld) for one env
    (verbatim from build_and_solve_equality_gpu).

    Reads equality constraint definitions from the equality record tensor,
    computes world anchors, Jacobians, impedance, and runs bilateral PGS
    iterations (no lambda >= 0 clamping) on `qacc_constrained`.
    """

    comptime if NEQUALITY == 0:
        return

    # Read number of equality constraints from model metadata
    var neq = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NEQUALITY])
    )
    if neq == 0:
        return
    if neq > NEQUALITY:
        neq = NEQUALITY

    # Max rows: 6 per constraint (3 connect + 3 weld orientation)
    comptime MAX_EQ_ROWS = _max_one[6 * NEQUALITY]()
    comptime MINVJ_EQ_SIZE = _max_one[6 * NEQUALITY * NV]()

    var eq_K = InlineArray[Scalar[DTYPE], MAX_EQ_ROWS](fill=Scalar[DTYPE](1))
    var eq_bias = InlineArray[Scalar[DTYPE], MAX_EQ_ROWS](fill=Scalar[DTYPE](0))
    var eq_inv_K_imp = InlineArray[Scalar[DTYPE], MAX_EQ_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var eq_lambda = InlineArray[Scalar[DTYPE], MAX_EQ_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var eq_J = InlineArray[Scalar[DTYPE], MINVJ_EQ_SIZE](fill=Scalar[DTYPE](0))
    var eq_MinvJ = InlineArray[Scalar[DTYPE], MINVJ_EQ_SIZE](
        fill=Scalar[DTYPE](0)
    )

    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))

    var num_eq_rows = 0

    # Build rows for each equality constraint
    for eq_i in range(neq):
        var eq_type = Int(rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_TYPE]))
        var body_a = Int(
            rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_BODY_A])
        )
        var body_b = Int(
            rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_BODY_B])
        )

        # Read anchors
        var anc_ax = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AX])
        var anc_ay = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AY])
        var anc_az = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AZ])
        var anc_bx = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_BX])
        var anc_by = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_BY])
        var anc_bz = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_BZ])

        # Read solref/solimp
        var sr_tc = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLREF_0])
        var sr_dr = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLREF_1])
        var si_dmin = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLIMP_0])
        var si_dmax = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLIMP_1])
        var si_width = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLIMP_2])
        var si_midpoint = rebind[Scalar[DTYPE]](
            equality[eq_i, EQ_IDX_SOLIMP_3]
        )
        var si_power = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLIMP_4])
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
        # interpolating (engine_core_constraint.c:1284-1287).
        comptime MJE_MINIMP = Scalar[DTYPE](0.0001)
        comptime MJE_MAXIMP = Scalar[DTYPE](0.9999)
        if si_dmin < MJE_MINIMP:
            si_dmin = MJE_MINIMP
        elif si_dmin > MJE_MAXIMP:
            si_dmin = MJE_MAXIMP
        if si_dmax < MJE_MINIMP:
            si_dmax = MJE_MINIMP
        elif si_dmax > MJE_MAXIMP:
            si_dmax = MJE_MAXIMP
        if si_power < Scalar[DTYPE](1):
            si_power = Scalar[DTYPE](1)
        # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
        # solref. See `constraints/constraint_data.solref_spring_damper` — the
        # formula lived in twelve copy-pasted sites until 2026-08-03.
        var (eq_K_spring, eq_B_damp) = solref_spring_damper[DTYPE](
            sr_tc, sr_dr, si_dmax,
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )

        # Compute world anchor A: xpos[body_a] + quat_rotate(xquat[body_a], anchor_a)
        var xpos_a_x = rebind[Scalar[DTYPE]](
            xpos[env, body_a * 3 + 0]
        )
        var xpos_a_y = rebind[Scalar[DTYPE]](
            xpos[env, body_a * 3 + 1]
        )
        var xpos_a_z = rebind[Scalar[DTYPE]](
            xpos[env, body_a * 3 + 2]
        )
        var xquat_a_x = rebind[Scalar[DTYPE]](
            xquat[env, body_a * 4 + 0]
        )
        var xquat_a_y = rebind[Scalar[DTYPE]](
            xquat[env, body_a * 4 + 1]
        )
        var xquat_a_z = rebind[Scalar[DTYPE]](
            xquat[env, body_a * 4 + 2]
        )
        var xquat_a_w = rebind[Scalar[DTYPE]](
            xquat[env, body_a * 4 + 3]
        )
        var rot_a = quat_rotate[DTYPE](
            xquat_a_x, xquat_a_y, xquat_a_z, xquat_a_w, anc_ax, anc_ay, anc_az
        )
        var world_ax = xpos_a_x + rot_a[0]
        var world_ay = xpos_a_y + rot_a[1]
        var world_az = xpos_a_z + rot_a[2]

        # Compute world anchor B
        var world_bx: Scalar[DTYPE]
        var world_by: Scalar[DTYPE]
        var world_bz: Scalar[DTYPE]
        if body_b > 0:
            var xpos_b_x = rebind[Scalar[DTYPE]](
                xpos[env, body_b * 3 + 0]
            )
            var xpos_b_y = rebind[Scalar[DTYPE]](
                xpos[env, body_b * 3 + 1]
            )
            var xpos_b_z = rebind[Scalar[DTYPE]](
                xpos[env, body_b * 3 + 2]
            )
            var xquat_b_x = rebind[Scalar[DTYPE]](
                xquat[env, body_b * 4 + 0]
            )
            var xquat_b_y = rebind[Scalar[DTYPE]](
                xquat[env, body_b * 4 + 1]
            )
            var xquat_b_z = rebind[Scalar[DTYPE]](
                xquat[env, body_b * 4 + 2]
            )
            var xquat_b_w = rebind[Scalar[DTYPE]](
                xquat[env, body_b * 4 + 3]
            )
            var rot_b = quat_rotate[DTYPE](
                xquat_b_x,
                xquat_b_y,
                xquat_b_z,
                xquat_b_w,
                anc_bx,
                anc_by,
                anc_bz,
            )
            world_bx = xpos_b_x + rot_b[0]
            world_by = xpos_b_y + rot_b[1]
            world_bz = xpos_b_z + rot_b[2]
        else:
            world_bx = anc_bx
            world_by = anc_by
            world_bz = anc_bz

        var pos_err_x = world_ax - world_bx
        var pos_err_y = world_ay - world_by
        var pos_err_z = world_az - world_bz

        # --- 3 position rows (connect + weld) ---
        var dirs = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        dirs[0] = Scalar[DTYPE](1)  # x-axis: (1,0,0)
        dirs[4] = Scalar[DTYPE](1)  # y-axis: (0,1,0)
        dirs[8] = Scalar[DTYPE](1)  # z-axis: (0,0,1)

        var pos_errs = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        pos_errs[0] = pos_err_x
        pos_errs[1] = pos_err_y
        pos_errs[2] = pos_err_z

        for d in range(3):
            if num_eq_rows >= MAX_EQ_ROWS:
                break
            var dx = dirs[d * 3 + 0]
            var dy = dirs[d * 3 + 1]
            var dz = dirs[d * 3 + 2]

            # Compute Jacobian: J = J_a(at world_a) - J_b(at world_b)
            # Each body's Jacobian uses its OWN anchor point (MuJoCo convention)
            for i in range(V_SIZE):
                J_row[i] = 0
            _weld_jacobian_row[
                DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH
            ](
                env,
                subtree_com,
                joints,
                bodies,
                mmeta,
                cdof,
                body_a,
                body_b,
                world_ax,
                world_ay,
                world_az,
                world_bx,
                world_by,
                world_bz,
                dx,
                dy,
                dz,
                J_row,
            )

            # Compute K = J @ M_inv @ J^T, store J and MinvJ
            var k: Scalar[DTYPE] = 0
            var v_n: Scalar[DTYPE] = 0
            for i in range(NV):
                eq_J[num_eq_rows * NV + i] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += (
                        rebind[Scalar[DTYPE]](
                            m_inv[env, i * NV + j_idx]
                        )
                        * J_row[j_idx]
                    )
                eq_MinvJ[num_eq_rows * NV + i] = mi_j_sum
                k += J_row[i] * mi_j_sum
                v_n += J_row[i] * rebind[Scalar[DTYPE]](
                    qvel[env, i]
                )

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            eq_K[num_eq_rows] = k

            # Impedance: MuJoCo piecewise power formula
            var err_d = pos_errs[d]
            var penetration = abs(err_d)
            var imp: Scalar[DTYPE]
            if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
                imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
            else:
                var x = penetration / si_width
                var y: Scalar[DTYPE]
                if x <= Scalar[DTYPE](0):
                    y = Scalar[DTYPE](0)
                elif x >= Scalar[DTYPE](1):
                    y = Scalar[DTYPE](1)
                elif si_power == Scalar[DTYPE](1):
                    y = x
                elif x <= si_midpoint:
                    var a = Scalar[DTYPE](1) / pow(
                        si_midpoint, si_power - Scalar[DTYPE](1)
                    )
                    y = a * pow(x, si_power)
                else:
                    var b = Scalar[DTYPE](1) / pow(
                        Scalar[DTYPE](1) - si_midpoint,
                        si_power - Scalar[DTYPE](1),
                    )
                    y = Scalar[DTYPE](1) - b * pow(
                        Scalar[DTYPE](1) - x, si_power
                    )
                imp = si_dmin + y * (si_dmax - si_dmin)
            if imp < Scalar[DTYPE](1e-6):
                imp = Scalar[DTYPE](1e-6)

            # MuJoCo equality bias: bias = -aref = B*vel + K*I*pos
            # where pos is the SIGNED error (not abs). Contact formula uses
            # -K*I*pen because contact pos = -penetration, but equality pos
            # is signed directly.
            var bias = eq_K_spring * imp * err_d + eq_B_damp * v_n
            eq_bias[num_eq_rows] = bias
            # MuJoCo: R = (1-imp)/imp * diagApprox (translation weights)
            # (legacy addresses body_invweight0 via the NTENDON/NSITE-less
            # offset — see _legacy_invw_read)
            var diag_eq: Scalar[DTYPE] = 0
            if body_a > 0 and body_a < NBODY:
                diag_eq += _legacy_invw_read[
                    DTYPE, NV, NBODY, NTENDON, NSITE
                ](body_a * 2, tendons, sites, body_invweight0, dof_invweight0)
            if body_b > 0 and body_b < NBODY:
                diag_eq += _legacy_invw_read[
                    DTYPE, NV, NBODY, NTENDON, NSITE
                ](body_b * 2, tendons, sites, body_invweight0, dof_invweight0)
            if diag_eq < Scalar[DTYPE](1e-10):
                diag_eq = rebind[Scalar[DTYPE]](k)
            var R_eq = (Scalar[DTYPE](1.0) - imp) / imp * diag_eq
            eq_inv_K_imp[num_eq_rows] = Scalar[DTYPE](1.0) / (
                rebind[Scalar[DTYPE]](k) + R_eq
            )

            num_eq_rows += 1

        # --- 3 orientation rows (weld only) ---
        if eq_type == EQ_WELD:
            # Read relpose
            var rp_x = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_RELPOSE_X]
            )
            var rp_y = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_RELPOSE_Y]
            )
            var rp_z = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_RELPOSE_Z]
            )
            var rp_w = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_RELPOSE_W]
            )

            # Compute orientation error: 0.5 * imag(conj(quat_b) * quat_a * relpose)
            var qa_x = xquat_a_x
            var qa_y = xquat_a_y
            var qa_z = xquat_a_z
            var qa_w = xquat_a_w

            var qb_x: Scalar[DTYPE]
            var qb_y: Scalar[DTYPE]
            var qb_z: Scalar[DTYPE]
            var qb_w: Scalar[DTYPE]
            if body_b > 0:
                qb_x = rebind[Scalar[DTYPE]](
                    xquat[env, body_b * 4 + 0]
                )
                qb_y = rebind[Scalar[DTYPE]](
                    xquat[env, body_b * 4 + 1]
                )
                qb_z = rebind[Scalar[DTYPE]](
                    xquat[env, body_b * 4 + 2]
                )
                qb_w = rebind[Scalar[DTYPE]](
                    xquat[env, body_b * 4 + 3]
                )
            else:
                qb_x = Scalar[DTYPE](0)
                qb_y = Scalar[DTYPE](0)
                qb_z = Scalar[DTYPE](0)
                qb_w = Scalar[DTYPE](1)

            # conj(qb) * qa
            var cqb = quat_conjugate[DTYPE](qb_x, qb_y, qb_z, qb_w)
            var temp = quat_mul[DTYPE](
                cqb[0], cqb[1], cqb[2], cqb[3], qa_x, qa_y, qa_z, qa_w
            )
            # * relpose
            var err_q = quat_mul[DTYPE](
                temp[0], temp[1], temp[2], temp[3], rp_x, rp_y, rp_z, rp_w
            )
            # 0.5 * imaginary part
            var rot_errs = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
            rot_errs[0] = Scalar[DTYPE](0.5) * err_q[0]
            rot_errs[1] = Scalar[DTYPE](0.5) * err_q[1]
            rot_errs[2] = Scalar[DTYPE](0.5) * err_q[2]

            for d in range(3):
                if num_eq_rows >= MAX_EQ_ROWS:
                    break
                var dx = dirs[d * 3 + 0]
                var dy = dirs[d * 3 + 1]
                var dz = dirs[d * 3 + 2]

                # Angular Jacobian
                for i in range(V_SIZE):
                    J_row[i] = 0
                _angular_jacobian_row_eq[
                    DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH
                ](
                    env,
                    joints,
                    bodies,
                    mmeta,
                    cdof,
                    body_a,
                    body_b,
                    dx,
                    dy,
                    dz,
                    J_row,
                )

                # K, store J and MinvJ
                var k: Scalar[DTYPE] = 0
                var v_n: Scalar[DTYPE] = 0
                for i in range(NV):
                    eq_J[num_eq_rows * NV + i] = J_row[i]
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            rebind[Scalar[DTYPE]](
                                m_inv[env, i * NV + j_idx]
                            )
                            * J_row[j_idx]
                        )
                    eq_MinvJ[num_eq_rows * NV + i] = mi_j_sum
                    k += J_row[i] * mi_j_sum
                    v_n += J_row[i] * rebind[Scalar[DTYPE]](
                        qvel[env, i]
                    )

                if k < Scalar[DTYPE](1e-10):
                    k = Scalar[DTYPE](1e-10)
                eq_K[num_eq_rows] = k

                # Impedance for orientation: MuJoCo piecewise power formula
                var err_d = rot_errs[d]
                var penetration = abs(err_d)
                var imp: Scalar[DTYPE]
                if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
                    imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
                else:
                    var x = penetration / si_width
                    var y: Scalar[DTYPE]
                    if x <= Scalar[DTYPE](0):
                        y = Scalar[DTYPE](0)
                    elif x >= Scalar[DTYPE](1):
                        y = Scalar[DTYPE](1)
                    elif si_power == Scalar[DTYPE](1):
                        y = x
                    elif x <= si_midpoint:
                        var a = Scalar[DTYPE](1) / pow(
                            si_midpoint, si_power - Scalar[DTYPE](1)
                        )
                        y = a * pow(x, si_power)
                    else:
                        var b = Scalar[DTYPE](1) / pow(
                            Scalar[DTYPE](1) - si_midpoint,
                            si_power - Scalar[DTYPE](1),
                        )
                        y = Scalar[DTYPE](1) - b * pow(
                            Scalar[DTYPE](1) - x, si_power
                        )
                    imp = si_dmin + y * (si_dmax - si_dmin)
                if imp < Scalar[DTYPE](1e-6):
                    imp = Scalar[DTYPE](1e-6)

                # MuJoCo equality bias: bias = K*I*pos + B*vel (signed pos)
                var bias = eq_K_spring * imp * err_d + eq_B_damp * v_n
                eq_bias[num_eq_rows] = bias
                # MuJoCo: R = (1-imp)/imp * diagApprox (rotation weights)
                # (legacy addresses body_invweight0 via the NTENDON/NSITE-less
                # offset — see _legacy_invw_read)
                var diag_rot: Scalar[DTYPE] = 0
                if body_a > 0 and body_a < NBODY:
                    diag_rot += _legacy_invw_read[
                        DTYPE, NV, NBODY, NTENDON, NSITE
                    ](
                        body_a * 2 + 1,
                        tendons,
                        sites,
                        body_invweight0,
                        dof_invweight0,
                    )
                if body_b > 0 and body_b < NBODY:
                    diag_rot += _legacy_invw_read[
                        DTYPE, NV, NBODY, NTENDON, NSITE
                    ](
                        body_b * 2 + 1,
                        tendons,
                        sites,
                        body_invweight0,
                        dof_invweight0,
                    )
                if diag_rot < Scalar[DTYPE](1e-10):
                    diag_rot = rebind[Scalar[DTYPE]](k)
                var R_rot = (Scalar[DTYPE](1.0) - imp) / imp * diag_rot
                eq_inv_K_imp[num_eq_rows] = Scalar[DTYPE](1.0) / (
                    rebind[Scalar[DTYPE]](k) + R_rot
                )

                num_eq_rows += 1

    if num_eq_rows == 0:
        return

    # Bilateral PGS iterations (no clamping)
    for _ in range(NUM_ITERATIONS):
        var max_delta: Scalar[DTYPE] = 0
        for r in range(num_eq_rows):
            # a_eq = J @ qacc
            var a_eq: Scalar[DTYPE] = 0
            for i in range(NV):
                a_eq += eq_J[r * NV + i] * rebind[Scalar[DTYPE]](
                    qacc_constrained[env, i]
                )

            var R_eq = Scalar[DTYPE](1.0) / eq_inv_K_imp[r] - eq_K[r]
            var residual = a_eq + eq_bias[r] + R_eq * eq_lambda[r]
            var delta = -residual * eq_inv_K_imp[r]
            var old_lambda = eq_lambda[r]
            eq_lambda[r] = eq_lambda[r] + delta
            # Bilateral: no clamping (force can push or pull)
            var actual = eq_lambda[r] - old_lambda
            var abs_d = abs(actual)
            if abs_d > max_delta:
                max_delta = abs_d
            # qacc += MinvJ * delta
            for i in range(NV):
                qacc_constrained[env, i] = (
                    rebind[Scalar[DTYPE]](qacc_constrained[env, i])
                    + eq_MinvJ[r * NV + i] * actual
                )

        if max_delta < Scalar[DTYPE](1e-4):
            break


# =============================================================================
# Fixed tendons (port of build_and_solve_tendon_gpu)
# =============================================================================


@always_inline
def _tendon_env[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NTENDON: Int,
    NSITE: Int,
    BATCH: Int,
    NUM_ITERATIONS: Int,
    SKIP_FIXED: Bool = False,
](
    env: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    tendons: LayoutTensor[
        DTYPE, Layout.row_major(NTENDON, MODEL_TENDON_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    body_invweight0: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, 2), MutAnyOrigin
    ],
    dof_invweight0: LayoutTensor[DTYPE, Layout.row_major(NV), MutAnyOrigin],
    m_inv: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
):
    """Build and solve fixed tendon equality constraints for one env
    (verbatim from build_and_solve_tendon_gpu).

    A fixed tendon is: ten_length = Σ(coef_i * qpos[joint_qposadr_i]).
    Equality constraint: ten_length - length_ref = 0.

    ⚠ `sites`, `body_invweight0` and `dof_invweight0` are NO LONGER READ. They
    fed the `_legacy_invw_read` diagApprox, which was the mjEQ_JOINT rule
    applied to a tendon row; the row now takes `TENDON_IDX_INVWEIGHT0` (see
    below). Kept in the signature only to avoid churning four solver call
    sites ahead of the rewrite that moves these rows INTO the Newton/CG
    systems — do not read them back in on the assumption they still matter.
    """

    comptime if NTENDON == 0:
        return

    # Read number of tendons from model metadata
    var nten = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NTENDON])
    )
    if nten == 0:
        return
    if nten > NTENDON:
        nten = NTENDON

    # One bilateral row per tendon
    comptime MAX_TEN_ROWS = _max_one[NTENDON]()
    comptime MINVJ_TEN_SIZE = _max_one[NTENDON * NV]()

    var ten_K = InlineArray[Scalar[DTYPE], MAX_TEN_ROWS](fill=Scalar[DTYPE](1))
    var ten_bias = InlineArray[Scalar[DTYPE], MAX_TEN_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var ten_inv_K_imp = InlineArray[Scalar[DTYPE], MAX_TEN_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var ten_lambda = InlineArray[Scalar[DTYPE], MAX_TEN_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var ten_J = InlineArray[Scalar[DTYPE], MINVJ_TEN_SIZE](
        fill=Scalar[DTYPE](0)
    )
    var ten_MinvJ = InlineArray[Scalar[DTYPE], MINVJ_TEN_SIZE](
        fill=Scalar[DTYPE](0)
    )

    var num_ten_rows = 0

    for t_i in range(nten):
        if num_ten_rows >= MAX_TEN_ROWS:
            break

        # A <tendon> DECLARATION is not a constraint. This pass imposes a
        # bilateral `ten_length == LENGTH_REF`, which only <equality><tendon>
        # asks for; a plain <fixed>/<spatial> is just a length definition used
        # by transmissions, springs and limits.
        #
        # This guard became load-bearing on 2026-07-31, when `fields_build`
        # stopped hardcoding `ntendon = 0`. humanoid and humanoid_standup each
        # declare two <fixed> hip-knee tendons that MuJoCo constrains in no
        # way; without it, populating the count would have welded them.
        if (
            Int(rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_IS_EQUALITY]))
            == 0
        ):
            continue

        # SKIP_FIXED: the caller already put this row INSIDE its solver
        # (`build_tendon_equality_rows`), so re-applying it here would double
        # the constraint force. Set by the PYRAMIDAL Newton paths; the
        # elliptic, CG and PGS paths still solve these here. Spatial equality
        # tendons are never row-built, so they stay this pass's job either way
        # — which is why the guard tests the KIND rather than skipping wholesale.
        comptime if SKIP_FIXED:
            if (
                Int(rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_KIND]))
                != TENDON_KIND_SPATIAL
            ):
                continue

        var num_joints = Int(
            rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_NUM_JOINTS])
        )
        var length_ref = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_LENGTH_REF]
        )

        # Compute tendon length and velocity, build trivial Jacobian.
        #
        # ⚠ THE SOLVER HELD ITS OWN COPY OF THE 4-WRAP CAP: four unrolled reads
        # into a pair of `InlineArray[..., 4]` locals. Widening the parser and
        # the record alone would have left a tendon with more than four wraps
        # silently short HERE instead — the same defect one layer down.
        #
        # ⚠ BUT WIDENING THOSE LOCALS TO 16 CORRUPTED THE SOLVE. This kernel is
        # Metal-compiled, and two 16-wide per-thread arrays moved Part A's
        # golden by 33% while the RECORD at 16 was provably fine — bisected as:
        #
        #     record 16, locals 16  ->  FAIL  -664336.715
        #     record 16, locals  4  ->  PASS
        #     record  4, locals  4  ->  PASS
        #
        # It is the local-memory hazard this codebase has hit before (the RK4
        # elliptic OOM, the two-non-owning-DeviceBuffer miscompile), and it
        # shows up as WRONG NUMBERS rather than a crash.
        #
        # So the wraps are read INLINE and never buffered, which is what
        # `tendon_limit.mojo` and `invweight.mojo` already do. The kernel now
        # holds less local state than before this change, and the cap it obeys
        # is the record's rather than one of its own.
        var ten_length: Scalar[DTYPE] = 0
        var ten_vel: Scalar[DTYPE] = 0
        var r = num_ten_rows

        for ji in range(TENDON_MAX_WRAPS):
            if ji >= num_joints:
                break
            var jnt_idx = Int(
                rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_JOINT_0 + ji])
            )
            if jnt_idx < 0 or jnt_idx >= NJOINT:
                continue
            # Read joint's qpos_adr and dof_adr from the joint records
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](joints[jnt_idx, JOINT_IDX_QPOS_ADR])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](joints[jnt_idx, JOINT_IDX_DOF_ADR])
            )
            var c = rebind[Scalar[DTYPE]](
                tendons[t_i, TENDON_IDX_COEF_0 + ji]
            )
            ten_length += c * rebind[Scalar[DTYPE]](
                qpos[env, qpos_adr]
            )
            ten_vel += c * rebind[Scalar[DTYPE]](qvel[env, dof_adr])
            # Trivial Jacobian: J[dof_adr] = coef
            ten_J[r * NV + dof_adr] = c

        # Tendon position error (bilateral)
        var pos_err = ten_length - length_ref

        # Compute K = J @ M_inv @ J^T and MinvJ
        var k: Scalar[DTYPE] = 0
        for i in range(NV):
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += (
                    rebind[Scalar[DTYPE]](
                        m_inv[env, i * NV + j_idx]
                    )
                    * ten_J[r * NV + j_idx]
                )
            ten_MinvJ[r * NV + i] = mi_j_sum
            k += ten_J[r * NV + i] * mi_j_sum

        if k < Scalar[DTYPE](1e-10):
            k = Scalar[DTYPE](1e-10)
        ten_K[r] = k

        # Read solref/solimp
        var sr_tc = rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_SOLREF_0])
        var sr_dr = rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_SOLREF_1])
        var si_dmin = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_0]
        )
        var si_dmax = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_1]
        )
        var si_width = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_2]
        )
        var si_midpoint = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_3]
        )
        var si_power = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_4]
        )
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
        # interpolating (engine_core_constraint.c:1284-1287).
        comptime MJE_MINIMP = Scalar[DTYPE](0.0001)
        comptime MJE_MAXIMP = Scalar[DTYPE](0.9999)
        if si_dmin < MJE_MINIMP:
            si_dmin = MJE_MINIMP
        elif si_dmin > MJE_MAXIMP:
            si_dmin = MJE_MAXIMP
        if si_dmax < MJE_MINIMP:
            si_dmax = MJE_MINIMP
        elif si_dmax > MJE_MAXIMP:
            si_dmax = MJE_MAXIMP
        if si_power < Scalar[DTYPE](1):
            si_power = Scalar[DTYPE](1)
        # MuJoCo: K = 1/(dmax² * timeconst² * dampratio²)
        # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
        # solref. See `constraints/constraint_data.solref_spring_damper` — the
        # formula lived in twelve copy-pasted sites until 2026-08-03.
        var (t_K_spring, t_B_damp) = solref_spring_damper[DTYPE](
            sr_tc, sr_dr, si_dmax,
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )

        # Impedance: MuJoCo piecewise power formula on |pos_err|
        var penetration = abs(pos_err)
        var imp: Scalar[DTYPE]
        if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
            imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
        else:
            var x = penetration / si_width
            var y: Scalar[DTYPE]
            if x <= Scalar[DTYPE](0):
                y = Scalar[DTYPE](0)
            elif x >= Scalar[DTYPE](1):
                y = Scalar[DTYPE](1)
            elif si_power == Scalar[DTYPE](1):
                y = x
            elif x <= si_midpoint:
                var a = Scalar[DTYPE](1) / pow(
                    si_midpoint, si_power - Scalar[DTYPE](1)
                )
                y = a * pow(x, si_power)
            else:
                var b = Scalar[DTYPE](1) / pow(
                    Scalar[DTYPE](1) - si_midpoint, si_power - Scalar[DTYPE](1)
                )
                y = Scalar[DTYPE](1) - b * pow(Scalar[DTYPE](1) - x, si_power)
            imp = si_dmin + y * (si_dmax - si_dmin)
        if imp < Scalar[DTYPE](1e-6):
            imp = Scalar[DTYPE](1e-6)

        # Bilateral equality constraint bias (MuJoCo formula):
        #   aref = -B*vel - K*imp*pos  (pos = pos_err, signed)
        #   bias = -aref = B*vel + K*imp*pos
        var bias = t_B_damp * ten_vel + t_K_spring * imp * pos_err
        ten_bias[r] = bias

        # R = (1-imp)/imp * diagApprox.
        #
        # ⚠ diagApprox for a TENDON equality is `tendon_invweight0[id]` — ONE
        # number, the tendon's own J M^-1 J^T at qpos0 (engine_core_constraint
        # .c:1091, `m->tendon_invweight0[m->eq_obj1id[id]]`). It is NOT the sum
        # of `dof_invweight0` over the tendon's joints, which is what this used
        # to compute: that is the mjEQ_JOINT branch's rule (:1090), applied to
        # the wrong constraint type. The two disagree by more than a constant —
        # a fixed tendon's coefficients and the joints' cross-coupling both
        # enter the real quantity and neither enters the sum.
        #
        # Silent, as usual: R only sets the row's COMPLIANCE, so the constraint
        # still held approximately and nothing errored. quadruped's four
        # coupling tendons are the first equality tendons any model here builds
        # through the parser, and they put ~7% on qacc.
        var diag_ten = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_INVWEIGHT0]
        )
        if diag_ten < Scalar[DTYPE](1e-10):
            diag_ten = k  # Fallback to exact K
        var R_ten = (Scalar[DTYPE](1.0) - imp) / imp * diag_ten
        ten_inv_K_imp[r] = Scalar[DTYPE](1.0) / (k + R_ten)

        num_ten_rows += 1

    if num_ten_rows == 0:
        return

    # Bilateral PGS iterations (no clamping — bilateral constraint)
    for _ in range(NUM_ITERATIONS):
        var max_delta: Scalar[DTYPE] = 0
        for r in range(num_ten_rows):
            # a_ten = J @ qacc
            var a_ten: Scalar[DTYPE] = 0
            for i in range(NV):
                a_ten += ten_J[r * NV + i] * rebind[Scalar[DTYPE]](
                    qacc_constrained[env, i]
                )

            var R_ten = Scalar[DTYPE](1.0) / ten_inv_K_imp[r] - ten_K[r]
            var residual = a_ten + ten_bias[r] + R_ten * ten_lambda[r]
            var delta = -residual * ten_inv_K_imp[r]
            var old_lambda = ten_lambda[r]
            ten_lambda[r] = ten_lambda[r] + delta
            # Bilateral: no clamping
            var actual = ten_lambda[r] - old_lambda
            var abs_d = abs(actual)
            if abs_d > max_delta:
                max_delta = abs_d
            # qacc += MinvJ * delta
            for i in range(NV):
                qacc_constrained[env, i] = (
                    rebind[Scalar[DTYPE]](qacc_constrained[env, i])
                    + ten_MinvJ[r * NV + i] * actual
                )

        if max_delta < Scalar[DTYPE](1e-4):
            break
