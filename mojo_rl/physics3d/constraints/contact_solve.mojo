"""PGS contact solve over per-field tensors (migration P4, single-source).

Per-field port of `PGSSolver.solve_gpu` (solver/pgs_solver.mojo:794) and the
shared constraint-builder helpers it uses
(constraints/constraint_builder_gpu.mojo: `init_common_normal_workspace_gpu`,
`precompute_contact_normal_gpu`, `precompute_contact_friction_gpu`,
`warmstart_normals_gpu`, `apply_solved_normals_gpu`) plus the contact
Jacobian rows (dynamics/jacobian.mojo: `compute_contact_jacobian_row_gpu`,
`compute_angular_jacobian_row_gpu`) — arithmetic verbatim.

Structural transformation (the only deviation): the legacy kernel is
2D-threaded (thread_y = contact slot) with barriers; this port SERIALIZES it
per env. Each `if valid_env:` per-contact parallel phase becomes a
`for contact_tid in range(MC)` loop (init + normal precompute, matching the
legacy internal `contact_tid < nc` guards) or `for contact_tid in range(nc)`
(friction precompute phase 3, whose legacy launch guard is
`contact_tid < nc`); barriers disappear and the `contact_tid == 0`
sequential sections run once. All phases write disjoint slots, so
serialization is value-identical.

The `detect_and_solve_limits_gpu` call of the legacy solve_gpu is wired via
its port limits.mojo; `build_and_solve_equality_gpu` /
`build_and_solve_tendon_gpu` via equality_tendon.mojo — all three at
the exact legacy position (after the normal PGS phase, before the friction
phase). Still excluded: the legacy `dt` metadata read, whose only consumer
was the limits call.

`precompute_contact_normal_gpu` is specialized to its only use here
(COMPUTE_RHS=False): the comptime rhs-write branch is dropped; the `a_n`
accumulation it fed is kept verbatim. `_precompute_contact_friction`
and `_apply_solved_normals` are ported (they are constraint-builder
helpers shared with the CG/Newton solvers) but not called by the PGS env
body — the legacy solve_gpu inlines its own friction phase 3 and force
write-back, kept inline here.

Operands (19): qpos, qvel, xpos, xquat, subtree_com, contacts, meta (data)
+ joints, bodies, meta, equality, tendons, sites, body_invweight0,
dof_invweight0 (model) + cdof, m_inv, qacc_constrained (scratch) + solver
(ContactScratch). The legacy contact-Jacobian row computes an xpos offset
it never reads (dropped); xpos/xquat here feed the equality world anchors,
qpos the tendon lengths, and sites only the legacy invweight0-offset
misread reproduction (see equality_tendon.mojo).
"""

from std.math import sqrt, pow, abs
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..types import _max_one, ConeType
from ..joint_types import JNT_FREE, JNT_BALL
from .qcqp import mj_qcqp2, mj_qcqp3, mj_qcqp5
from ..dynamics.jac_contact_row import _contact_jacobian_row
from .limits import _limits_env
from .friction_dof import _friction_env
from .equality_tendon import _equality_env, _tendon_env
from ..fields import Data, Model, DynamicsScratch, ContactScratch, Dims
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    METADATA_SIZE,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_INCLUDEMARGIN,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FORCE_TORSION,
    CONTACT_IDX_FORCE_ROLL1,
    CONTACT_IDX_FORCE_ROLL2,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_SOLIMP_4,
    CONTACT_IDX_SOLIMP_3,
    CONTACT_IDX_SOLIMP_2,
    CONTACT_IDX_SOLIMP_1,
    CONTACT_IDX_SOLIMP_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_IMPRATIO,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)
from ..collision.contact_frame import contact_tangent_frame
from .elliptic_layout import (
    ell_nt,
    ell_jt,
    ell_mu,
    ell_dn,
    ell_dt,
    ell_fr,
    ell_bt,
    ell_ntc,
)

from .constraint_data import solref_spring_damper

comptime CS_TPB: Int = 64

# PGS solver parameters (replicated from solver/pgs_solver.mojo:83)
comptime PGS_ITERATIONS: Int = 100
# Minimum K for friction tangent rows — below this, direction is degenerate
comptime FRICTION_K_MIN: Float64 = 1e-6


# =============================================================================
# Angular Jacobian rows (port of dynamics/jacobian.mojo GPU rows)
# ⚠ the TRANSLATIONAL row `_contact_jacobian_row` moved to
# `dynamics/jac_contact_row.mojo` in phase 2.0 — see that file for why.
# =============================================================================


@always_inline
def _angular_jacobian_row[
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
    """Angular-only Jacobian row for torsional/rolling friction (verbatim
    from compute_angular_jacobian_row_gpu).

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
# Constraint-builder helpers (port of constraints/constraint_builder_gpu.mojo)
# =============================================================================


@always_inline
def _init_common_normal_ws[
    DTYPE: DType,
    NV: Int,
    MAX_CONTACTS: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    env: Int,
    contact_tid: Int,
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
):
    """Zero-initialize common normal workspace fields for one contact slot
    (verbatim from init_common_normal_workspace_gpu; the `solver_idx` base
    is gone — offsets are row-relative)."""
    comptime MC = _max_one[MAX_CONTACTS]()

    solver[env, 0 * MC + contact_tid] = 0  # lambda_n
    solver[env, 1 * MC + contact_tid] = 1  # K_n
    solver[env, 2 * MC + contact_tid] = 0  # c_dist
    solver[env, 3 * MC + contact_tid] = 0  # c_body
    solver[env, 4 * MC + contact_tid] = -1  # c_body_b
    solver[env, 5 * MC + contact_tid] = 0  # c_px
    solver[env, 6 * MC + contact_tid] = 0  # c_py
    solver[env, 7 * MC + contact_tid] = 0  # c_pz
    solver[env, 8 * MC + contact_tid] = 0  # c_nx
    solver[env, 9 * MC + contact_tid] = 0  # c_ny
    solver[env, 10 * MC + contact_tid] = 1  # c_nz
    solver[env, 11 * MC + contact_tid] = 0  # pos_bias
    solver[env, 12 * MC + contact_tid] = 0  # inv_K_imp
    solver[env, 13 * MC + contact_tid] = 0  # imp_n
    solver[env, 14 * MC + contact_tid] = 0  # diag_n
    # Zero J_n and MinvJn for this slot
    for i in range(NV):
        solver[env, 15 * MC + contact_tid * NV + i] = 0
        solver[env, 15 * MC + MC * NV + contact_tid * NV + i] = 0


@always_inline
def _precompute_contact_normal[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    env: Int,
    contact_tid: Int,
    nc: Int,
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
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
    body_invweight0: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, 2), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    m_inv: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
    # ⚠ SUPERSEDED, AND DELIBERATELY RENAMED SO THEY CANNOT BE MISREAD AS
    # LIVE. These are the MODEL-LEVEL solref/solimp
    # (`MODEL_META_IDX_SOLREF_CONTACT_*`) every contact used to share. Since
    # 2026-08-03 the mixed PER-CONTACT values are read from the contact record
    # below, so changing the model-level ones does nothing here. They are still
    # passed because five call sites across four files hand them in positionally
    # and a second consumer (`_precompute_contact_friction`) takes the same
    # list; removing them is a follow-up, not a silent leftover.
    _unused_model_K: Scalar[DTYPE],
    _unused_model_B: Scalar[DTYPE],
    _unused_model_dmin: Scalar[DTYPE],
    _unused_model_dmax: Scalar[DTYPE],
    _unused_model_width: Scalar[DTYPE],
    _unused_model_midpoint: Scalar[DTYPE],
    _unused_model_power: Scalar[DTYPE],
):
    """Precompute one contact's normal constraint data (verbatim from
    precompute_contact_normal_gpu, specialized to COMPUTE_RHS=False — its
    only use in the PGS solve; the comptime rhs-write branch is dropped and
    the `a_n` accumulation it fed is kept verbatim)."""
    comptime MC = _max_one[MAX_CONTACTS]()

    # Common block offsets
    comptime ws_lambda_n = 0 * MC
    comptime ws_K_n = 1 * MC
    comptime ws_c_dist = 2 * MC
    comptime ws_c_body = 3 * MC
    comptime ws_c_body_b = 4 * MC
    comptime ws_c_px = 5 * MC
    comptime ws_c_py = 6 * MC
    comptime ws_c_pz = 7 * MC
    comptime ws_c_nx = 8 * MC
    comptime ws_c_ny = 9 * MC
    comptime ws_c_nz = 10 * MC
    comptime ws_pos_bias = 11 * MC
    comptime ws_inv_K_imp = 12 * MC
    comptime ws_imp_n = 13 * MC
    comptime ws_diag_n = 14 * MC
    comptime ws_J_n = 15 * MC
    comptime ws_MinvJn = 15 * MC + MC * NV

    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        J_row[i] = 0

    if contact_tid < nc:
        var c = contact_tid
        var c_off = c * CONTACT_SIZE

        # ── PER-CONTACT solver parameters ────────────────────────────────
        # Written by the narrow phase from MuJoCo's mixing rule
        # (`mix_contact_params`), which honours `<geom priority>` and the
        # two geoms' own solref/solimp. Before this the whole model shared
        # one solref, so `<geom solref="-10000 -30"/>` — dm_control's
        # quadruped ball — was parsed, stored and ignored.
        var si_dmin = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_SOLIMP_0]
        )
        var si_dmax = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_SOLIMP_1]
        )
        var si_width = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_SOLIMP_2]
        )
        var si_midpoint = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_SOLIMP_3]
        )
        var si_power = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_SOLIMP_4]
        )
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        # Same clamps MuJoCo applies before interpolating
        # (engine_core_constraint.c:1284-1287) — see the note at the old
        # hoisted block for why the dmin floor is the one that bites.
        comptime MJ_MINIMP_C = Scalar[DTYPE](0.0001)
        comptime MJ_MAXIMP_C = Scalar[DTYPE](0.9999)
        if si_dmin < MJ_MINIMP_C:
            si_dmin = MJ_MINIMP_C
        elif si_dmin > MJ_MAXIMP_C:
            si_dmin = MJ_MAXIMP_C
        if si_dmax < MJ_MINIMP_C:
            si_dmax = MJ_MINIMP_C
        elif si_dmax > MJ_MAXIMP_C:
            si_dmax = MJ_MAXIMP_C
        if si_power < Scalar[DTYPE](1):
            si_power = Scalar[DTYPE](1)
        var _kb = solref_spring_damper[DTYPE](
            rebind[Scalar[DTYPE]](
                contacts[env, c_off + CONTACT_IDX_SOLREF_0]
            ),
            rebind[Scalar[DTYPE]](
                contacts[env, c_off + CONTACT_IDX_SOLREF_1]
            ),
            si_dmax,
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )
        var K_spring = _kb[0]
        var B_damp = _kb[1]
        var body = Int(
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_BODY_A])
        )
        var body_b = Int(
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_BODY_B])
        )
        var dist = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_DIST]
        )
        var includemargin = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN]
        )

        # Store dist - includemargin so later >= 0 checks work correctly
        solver[env, ws_c_dist + c] = dist - includemargin
        solver[env, ws_c_body + c] = Scalar[DTYPE](body)
        solver[env, ws_c_body_b + c] = Scalar[DTYPE](body_b)

        if dist < includemargin:
            solver[env, ws_c_px + c] = contacts[
                env, c_off + CONTACT_IDX_POS_X
            ]
            solver[env, ws_c_py + c] = contacts[
                env, c_off + CONTACT_IDX_POS_Y
            ]
            solver[env, ws_c_pz + c] = contacts[
                env, c_off + CONTACT_IDX_POS_Z
            ]
            solver[env, ws_c_nx + c] = contacts[env, c_off + CONTACT_IDX_NX]
            solver[env, ws_c_ny + c] = contacts[env, c_off + CONTACT_IDX_NY]
            solver[env, ws_c_nz + c] = contacts[env, c_off + CONTACT_IDX_NZ]

            # Compute normal Jacobian
            _contact_jacobian_row[
                DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH
            ](
                env,
                subtree_com,
                joints,
                bodies,
                mmeta,
                cdof,
                body,
                body_b,
                rebind[Scalar[DTYPE]](solver[env, ws_c_px + c]),
                rebind[Scalar[DTYPE]](solver[env, ws_c_py + c]),
                rebind[Scalar[DTYPE]](solver[env, ws_c_pz + c]),
                rebind[Scalar[DTYPE]](solver[env, ws_c_nx + c]),
                rebind[Scalar[DTYPE]](solver[env, ws_c_ny + c]),
                rebind[Scalar[DTYPE]](solver[env, ws_c_nz + c]),
                J_row,
            )

            # Store J_n, compute MinvJn and K_n
            var k: solver.element_type = 0
            var v_n: solver.element_type = 0
            var a_n: solver.element_type = 0

            for i in range(NV):
                solver[env, ws_J_n + c * NV + i] = J_row[i]
                var mi_j_sum: solver.element_type = 0
                for j_idx in range(NV):
                    mi_j_sum += m_inv[env, i * NV + j_idx] * J_row[j_idx]
                solver[env, ws_MinvJn + c * NV + i] = mi_j_sum
                k += J_row[i] * mi_j_sum
                # Use current VELOCITY for damping in aref (MuJoCo: efc_vel = J*qvel)
                v_n += J_row[i] * rebind[Scalar[DTYPE]](qvel[env, i])
                # Constraint-space acceleration (for solver RHS)
                a_n += J_row[i] * qacc_constrained[env, i]

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            solver[env, ws_K_n + c] = k

            # Acceleration-level aref: MuJoCo piecewise power impedance
            # MuJoCo: aref uses (pos - includemargin), penetration = -(dist - margin)
            var penetration = -(dist - includemargin)
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
            # Impedance floor prevents zero-force contacts at surface
            if imp < Scalar[DTYPE](1e-6):
                imp = Scalar[DTYPE](1e-6)
            # MuJoCo: aref = -B*vel - K*imp*pos, bias = -aref = B*vel + K*imp*pen
            # bias = -aref = -(K*imp*pen - B*v_n) = -K*imp*pen + B*v_n
            var bias = -K_spring * imp * penetration + B_damp * v_n
            solver[env, ws_pos_bias + c] = bias
            # MuJoCo: R = (1-imp)/imp * diagApprox, inv_K_imp = 1/(K + R)
            # diagApprox = body_invweight0[2*body_a] + body_invweight0[2*body_b]
            var diag_n: Scalar[DTYPE] = 0
            if body > 0 and body < NBODY:
                diag_n += rebind[Scalar[DTYPE]](body_invweight0[body, 0])
            if body_b > 0 and body_b < NBODY:
                diag_n += rebind[Scalar[DTYPE]](body_invweight0[body_b, 0])
            if diag_n < Scalar[DTYPE](1e-10):
                diag_n = rebind[Scalar[DTYPE]](k)  # Fallback to exact K
            var R_n = (Scalar[DTYPE](1.0) - imp) / imp * diag_n
            solver[env, ws_inv_K_imp + c] = Scalar[DTYPE](1.0) / (
                rebind[Scalar[DTYPE]](k) + R_n
            )
            # Store imp and diag_n for direct R_n computation in friction builder
            # (avoids lossy float32 round-trip R = 1/inv_K_imp - K)
            solver[env, ws_imp_n + c] = imp
            solver[env, ws_diag_n + c] = diag_n

            # (COMPUTE_RHS=False specialization: legacy rhs write dropped)
            _ = a_n

            # Warm-start lambda
            solver[env, ws_lambda_n + c] = contacts[
                env, c_off + CONTACT_IDX_FORCE_N
            ]


@always_inline
def _precompute_contact_friction[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_CONDIM: Int = 3,
](
    env: Int,
    contact_tid: Int,
    nc: Int,
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
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
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
    B_damp: Scalar[DTYPE],
    impratio: Scalar[DTYPE],
    K_spring: Scalar[DTYPE],
):
    """Build friction tangent data for one contact (verbatim from
    precompute_contact_friction_gpu — the SHARED CG/Newton friction builder;
    NOT called by the PGS env body below, which inlines its own legacy
    friction phase 3). The legacy pyramidal branch declared an unused
    M_inv offset, dropped here.

    ⚠ THE WORKSPACE OFFSETS USED TO BE ARGUMENTS. Each of the three callers
    declared its own `comptime ws_*_idx = SC + k*MC` chain and passed seven of
    them in; that was safe only while the chain was a fixed seven entries. The
    ELLIPTIC region is now `MAX_CONDIM`-dependent (`solver/elliptic_layout`),
    and a caller left on a stale stride would read a DIFFERENT contact's
    friction rather than fail. They are derived here and read back through the
    same functions.
    """
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime NT = ell_nt[MAX_CONDIM]()
    comptime ws_Jt1_idx = ell_jt[MC, NV]()
    comptime ws_mu_idx = ell_mu[MC, NV, MAX_CONDIM]()
    comptime ws_D_n_idx = ell_dn[MC, NV, MAX_CONDIM]()
    comptime ws_Dt_idx = ell_dt[MC, NV, MAX_CONDIM]()
    comptime ws_fr_idx = ell_fr[MC, NV, MAX_CONDIM]()
    comptime ws_bt_idx = ell_bt[MC, NV, MAX_CONDIM]()
    comptime ws_ntc_idx = ell_ntc[MC, NV, MAX_CONDIM]()
    # Pyramid edges per contact: MuJoCo emits one OPPOSING PAIR per friction
    # dimension (engine_core_constraint.c `make pyramidal friction cone`), so
    # a condim-d contact owns 2*(d-1) rows — 4 at condim 3, 6 at 4, 10 at 6.
    # Slots are sized for the model's WORST condim and the tail is zeroed per
    # contact, because condim is per-geom-pair and only known at runtime.
    comptime NE_PYR = 2 * (MAX_CONDIM - 1)

    # Common normal block offsets
    comptime ws_c_dist = 2 * MC
    comptime ws_c_body = 3 * MC
    comptime ws_c_body_b = 4 * MC
    comptime ws_c_px = 5 * MC
    comptime ws_c_py = 6 * MC
    comptime ws_c_pz = 7 * MC
    comptime ws_c_nx = 8 * MC
    comptime ws_c_ny = 9 * MC
    comptime ws_c_nz = 10 * MC
    comptime ws_imp_n = 13 * MC
    comptime ws_diag_n = 14 * MC

    var c = contact_tid

    # Only process penetrating contacts (c_dist stores dist - includemargin)
    if rebind[Scalar[DTYPE]](solver[env, ws_c_dist + c]) >= Scalar[DTYPE](0):
        # Zero friction outputs for non-active contacts
        solver[env, ws_mu_idx + c] = 0
        solver[env, ws_D_n_idx + c] = 0
        comptime if CONE_TYPE == ConeType.ELLIPTIC:
            # ⚠ EVERY tangential row, and the row COUNT with them. A contact
            # that stopped touching keeps its slot; leaving `ntc` at the
            # previous step's value would make the solver read Jacobians it
            # just zeroed as if they were live rows.
            for t in range(NT):
                solver[env, ws_Dt_idx + t * MC + c] = 0
                solver[env, ws_fr_idx + t * MC + c] = 0
                solver[env, ws_bt_idx + t * MC + c] = 0
            solver[env, ws_ntc_idx + c] = 0
        comptime if CONE_TYPE == ConeType.PYRAMIDAL:
            # ⚠ ZERO EVERY EDGE, not just the two `ws_Jt1/Jt2` slots. Those
            # two names alias pyramid edges 0 and 1; edges 2.. live further
            # up the same region and would otherwise keep a previous step's
            # Jacobian for a contact that is no longer touching.
            var pyr_sc_z = ws_Jt1_idx + NE_PYR * MC * NV
            for e in range(NE_PYR):
                for i in range(NV):
                    solver[env, ws_Jt1_idx + e * MC * NV + c * NV + i] = 0
                solver[env, pyr_sc_z + e * MC + c] = 0
                solver[env, pyr_sc_z + NE_PYR * MC + e * MC + c] = 0
        else:
            for t in range(NT):
                for i in range(NV):
                    solver[env, ws_Jt1_idx + t * MC * NV + c * NV + i] = 0
        return

    var nx = rebind[Scalar[DTYPE]](solver[env, ws_c_nx + c])
    var ny = rebind[Scalar[DTYPE]](solver[env, ws_c_ny + c])
    var nz = rebind[Scalar[DTYPE]](solver[env, ws_c_nz + c])

    # --- Tangent frame from capsule axis hint (MuJoCo mju_makeFrame) ---
    var c_off = c * CONTACT_SIZE
    var hint_x = rebind[Scalar[DTYPE]](
        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X]
    )
    var hint_y = rebind[Scalar[DTYPE]](
        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y]
    )
    var hint_z = rebind[Scalar[DTYPE]](
        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z]
    )

    var frame = contact_tangent_frame[DTYPE](
        nx, ny, nz, hint_x, hint_y, hint_z
    )
    var t1x = frame[0]
    var t1y = frame[1]
    var t1z = frame[2]
    var t2x = frame[3]
    var t2y = frame[4]
    var t2z = frame[5]

    # --- Contact body and position ---
    var body_a = Int(rebind[Scalar[DTYPE]](solver[env, ws_c_body + c]))
    var body_b = Int(rebind[Scalar[DTYPE]](solver[env, ws_c_body_b + c]))
    var px = rebind[Scalar[DTYPE]](solver[env, ws_c_px + c])
    var py = rebind[Scalar[DTYPE]](solver[env, ws_c_py + c])
    var pz = rebind[Scalar[DTYPE]](solver[env, ws_c_pz + c])

    # --- Friction coefficient ---
    var mu_c = rebind[Scalar[DTYPE]](
        contacts[env, c_off + CONTACT_IDX_FRICTION]
    )
    if mu_c <= Scalar[DTYPE](0):
        mu_c = Scalar[DTYPE](0.5)

    # --- D values from normal precompute ---
    # Compute R_n directly from stored imp and diag_n (avoids lossy float32
    # round-trip R = 1/inv_K_imp - K which suffers catastrophic cancellation
    # when K >> R or K << R in deep penetration contacts)
    var imp_c = rebind[Scalar[DTYPE]](solver[env, ws_imp_n + c])
    var diag_n_c = rebind[Scalar[DTYPE]](solver[env, ws_diag_n + c])
    var R_n_c = (Scalar[DTYPE](1.0) - imp_c) / imp_c * diag_n_c
    if R_n_c < Scalar[DTYPE](1e-14):
        R_n_c = Scalar[DTYPE](1e-14)

    # --- Compute J_t1, J_t2 (needed by both ELLIPTIC and PYRAMIDAL) ---
    var J_t1 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var J_t2 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    _contact_jacobian_row[DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH](
        env, subtree_com, joints, bodies, mmeta, cdof,
        body_a, body_b, px, py, pz, t1x, t1y, t1z, J_t1,
    )
    _contact_jacobian_row[DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH](
        env, subtree_com, joints, bodies, mmeta, cdof,
        body_a, body_b, px, py, pz, t2x, t2y, t2z, J_t2,
    )

    # Read J_n from normal precompute
    comptime ws_J_n = 15 * MC

    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
        # === PYRAMIDAL: Build 2*(dim-1) edge Jacobians (J_n ± mu_k*J_k) ===
        # Workspace layout (PYRAMIDAL scalar base = ws_Jt1_idx + NE*MC*NV):
        #   Jacobians: NE * MC * NV at ws_Jt1_idx
        #   Scalars at PYR_SC:
        #     [0*MC..NE*MC)      D_edge[NE*MC]
        #     [NE*MC..2*NE*MC)   bias_edge[NE*MC]
        #     [2*NE*MC..+MC)     mu[MC]
        var pyr_sc = ws_Jt1_idx + NE_PYR * MC * NV

        # Use imp and diag_n from normal precompute (already read above)
        var diag_edge = diag_n_c + mu_c * mu_c * diag_n_c

        # R_edge = 2*mu²*(1-imp)/imp*diag_edge
        # Since R_n = (1-imp)/imp * diag_n → R_edge = 2*mu² * diag_edge/diag_n * R_n
        var R_edge = Scalar[DTYPE](2.0) * mu_c * mu_c * (
            diag_edge / diag_n_c
        ) * R_n_c
        if R_edge < Scalar[DTYPE](1e-14):
            R_edge = Scalar[DTYPE](1e-14)
        var D_edge_val = Scalar[DTYPE](1.0) / R_edge

        # Use imp directly from normal precompute
        var imp_n = imp_c
        var pen = -rebind[Scalar[DTYPE]](solver[env, ws_c_dist + c])

        # ⚠ `K_spring`/`B_damp` MUST COME FROM THIS CONTACT, NOT THE MODEL.
        # The caller computes them ONCE from the model-level
        # `MODEL_META_IDX_SOLREF_CONTACT_*` / `SOLIMP_CONTACT_*` and passes
        # them in, but MuJoCo derives `k = 1/(dmax^2 tc^2 dr^2)` and
        # `b = 2/(dmax tc)` from each contact's MIXED solref/solimp. Every
        # contact whose mixed parameters differ from the model default
        # therefore got the wrong `aref`, as a CONSTANT offset on all of its
        # rows — the normal builder above already recomputes these per contact
        # (see the `solref_spring_damper` call there); only this edge builder
        # did not, and the pyramidal edge rows are what the Newton solver
        # consumes.
        #
        # Measured on dm_control's dog at a settled pose: the five contacts
        # with mixed solimp (`foot_primitive` 0.9/0.95 against the floor's
        # 0.95/0.99, mixing to 0.925/0.97) carried `jar` offsets of +0.0464,
        # +0.0822, +0.1149, +0.0133, +0.0098 — constant within each contact,
        # which is the signature of a bias error rather than a Jacobian one.
        # Contact 1 checks out exactly: `k` is 10203.04 at the model's
        # dmax = 0.99 and 10628.10 at the mixed 0.97, and the difference in
        # `-k*imp*pen` is +0.04636.
        #
        # The consequence was a solver that converged PERFECTLY (gradient
        # 1.9e-14) to the minimum of a slightly different objective: its cost
        # understated the true one by 0.774 and its answer sat exactly one
        # Newton step from MuJoCo's.
        var _kb_c = solref_spring_damper[DTYPE](
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_SOLREF_0]),
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_SOLREF_1]),
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_SOLIMP_1]),
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )
        var K_spring_c = _kb_c[0]
        var B_damp_c = _kb_c[1]

        # Store mu
        solver[env, pyr_sc + 2 * NE_PYR * MC + c] = mu_c

        # This contact's own condim decides how many of the NE_PYR slots are
        # live; the rest are zeroed below so a condim-3 contact in a condim-6
        # model contributes exactly its 4 edges.
        var condim_c = Int(
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_CONDIM])
        )
        if condim_c < 1:
            condim_c = 3
        if condim_c > MAX_CONDIM:
            condim_c = MAX_CONDIM
        var n_edge_c = 2 * (condim_c - 1)

        # ⚠ A FRICTIONLESS CONTACT IS ONE ROW, NOT ZERO ROWS.
        # `2*(dim-1)` is zero at `dim == 1`, which used to zero every slot and
        # leave the contact contributing NOTHING — detected, recorded in
        # `d.contacts`, reported in `ncon`, and silently absent from the solve,
        # so the two geoms passed through each other. MuJoCo emits one
        # `mjCNSTR_CONTACT_FRICTIONLESS` row there (`efc_type == 5`): the pure
        # normal row, one-sided like every pyramid edge, with the NORMAL `R`
        # rather than the pyramid's `2*mu^2*R`.
        #
        # Measured on dm_control's dog at a settled pose: MuJoCo reports
        # `efc_type {6: 40, 5: 3, 3: 2}` — three of thirteen contacts are
        # frictionless, because `collision_primitive` sets `condim="1"` and dog
        # has 81 such geoms. Gated by
        # `tests/physics3d/test_frictionless_contact_pyramidal.mojo`.
        var frictionless = condim_c == 1
        if frictionless:
            n_edge_c = 1

        # Friction direction k (k = 1..dim-1) pairs MuJoCo's `jac` row k with
        # `con->friction[k-1]`:  k=1,2 -> the two SLIDE tangents (linear
        # Jacobian along t1/t2); k=3 -> TORSION about the contact normal;
        # k=4,5 -> ROLLING about t1/t2. Rows 3.. use the ANGULAR Jacobian,
        # which is why they cannot reuse J_t1/J_t2.
        var mu_spin_c = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN]
        )
        var mu_roll_c = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL]
        )
        var J_k = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Build 2*(dim-1) edges: one opposing pair per friction dimension.
        for edge in range(NE_PYR):
            var ws_Je = ws_Jt1_idx + edge * MC * NV
            if edge >= n_edge_c:
                for i in range(NV):
                    solver[env, ws_Je + c * NV + i] = 0
                solver[env, pyr_sc + edge * MC + c] = 0
                solver[env, pyr_sc + NE_PYR * MC + edge * MC + c] = 0
                continue

            var sign = Scalar[DTYPE](1.0) if (edge % 2 == 0) else Scalar[
                DTYPE
            ](-1.0)
            var k = edge // 2  # 0 -> t1, 1 -> t2, 2 -> torsion, 3/4 -> roll
            var mu_k = mu_c
            if frictionless:
                # `je = J_n + sign*mu_k*J_k` collapses to `J_n` at mu = 0, so
                # the single row IS the normal row. `k` is 0 here, which loads
                # `J_t1` into `J_k` — it is multiplied by zero, but it must
                # still be a real vector: `J_k` is `uninitialized=True` and
                # `0 * NaN` is NaN, not 0.
                mu_k = Scalar[DTYPE](0)
            elif k == 2:
                mu_k = mu_spin_c
            elif k >= 3:
                mu_k = mu_roll_c

            if k == 0:
                for i in range(NV):
                    J_k[i] = J_t1[i]
            elif k == 1:
                for i in range(NV):
                    J_k[i] = J_t2[i]
            else:
                # Angular row about n (k=2), t1 (k=3) or t2 (k=4).
                var ax = nx
                var ay = ny
                var az = nz
                if k == 3:
                    ax = t1x
                    ay = t1y
                    az = t1z
                elif k == 4:
                    ax = t2x
                    ay = t2y
                    az = t2z
                _angular_jacobian_row[
                    DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH
                ](
                    env, joints, bodies, mmeta, cdof,
                    body_a, body_b, ax, ay, az, J_k,
                )

            var v_edge: Scalar[DTYPE] = 0
            for i in range(NV):
                var jn_i = rebind[Scalar[DTYPE]](
                    solver[env, ws_J_n + c * NV + i]
                )
                var je = jn_i + sign * mu_k * J_k[i]
                solver[env, ws_Je + c * NV + i] = je
                v_edge += je * rebind[Scalar[DTYPE]](qvel[env, i])

            # ⚠ D_edge IS COMMON TO EVERY EDGE OF THE CONTACT, including the
            # torsional and rolling ones. MuJoCo's pyramidal branch assigns a
            # single `Rpy = 2*mu^2*R` to all 2*(dim-1) rows
            # (engine_core_constraint.c:1899). The per-direction
            # `R[j] = R[1]*friction[0]^2/friction[j]^2` rescaling right above
            # it belongs to the ELLIPTIC branch — applying it here makes the
            # spin row ~(mu_slide/mu_spin)^2 too soft, which for a ball at
            # 0.7/0.05 is a factor of 196 and reads as "torsion does nothing".
            # ⚠ THE FRICTIONLESS ROW TAKES THE NORMAL `R`, NOT THE PYRAMID'S.
            # `R_edge = 2*mu^2*R_n` is zero at mu = 0 and would be clamped to
            # 1e-14, i.e. `D = 1e14` — an infinitely rigid row that blows the
            # solve up rather than merely being wrong.
            solver[env, pyr_sc + edge * MC + c] = (
                Scalar[DTYPE](1.0) / R_n_c if frictionless else D_edge_val
            )
            # bias_edge = B*v_edge - K_spring*imp*pen, with THIS CONTACT's
            # mixed spring/damper — see the note above `_kb_c`.
            var bias_e = B_damp_c * v_edge - K_spring_c * imp_n * pen
            solver[env, pyr_sc + NE_PYR * MC + edge * MC + c] = bias_e

    else:
        # === ELLIPTIC: one normal row + `dim-1` tangential rows ===
        #
        # ⚠ THIS USED TO BUILD EXACTLY TWO TANGENTS AND ONE ISOTROPIC `mu`,
        # i.e. condim 3, whatever the geoms declared. `MAX_CONDIM` was already
        # threaded through to here and consumed only by the PYRAMIDAL branch,
        # so a `condim="4"` geom under `cone="elliptic"` silently lost its
        # torsional row — `manipulation/reach_site_features` has 3 such
        # contacts of 55 at qpos0 and every dm_control manipulation model
        # declares `cone="elliptic"`.
        #
        # Row `t` pairs with `con->friction[t]`: t=0,1 SLIDE (the linear
        # Jacobians along t1/t2 already built above), t=2 TORSION about the
        # normal, t=3,4 ROLLING about t1/t2 — the last three from the ANGULAR
        # Jacobian, which is why they cannot reuse `J_t1`/`J_t2`. Identical
        # direction convention to the pyramidal edge builder above; the cones
        # differ in how rows are COMBINED, not in what the rows are.
        var condim_e = Int(
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_CONDIM])
        )
        if condim_e < 1:
            condim_e = 3
        if condim_e > MAX_CONDIM:
            condim_e = MAX_CONDIM
        # A FRICTIONLESS contact (`condim="1"`) is one normal row and NOTHING
        # else — `nt_c = 0`. The cone then degenerates to `T == 0`, whose top /
        # bottom zones are exactly the one-sided normal constraint MuJoCo emits
        # as `mjCNSTR_CONTACT_FRICTIONLESS`. Before this, the elliptic path had
        # no condim branch at all and gave such a contact full sliding friction.
        var nt_c = condim_e - 1
        if nt_c < 0:
            nt_c = 0
        solver[env, ws_ntc_idx + c] = Scalar[DTYPE](nt_c)

        var mu_spin_e = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN]
        )
        var mu_roll_e = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL]
        )

        var D_n_c = Scalar[DTYPE](1.0) / R_n_c
        solver[env, ws_D_n_idx + c] = D_n_c

        # ⚠ `con->mu` IS NOT `friction[0]`; it is the REGULARIZED coefficient
        # `friction[0] * sqrt(R[1]/R[0])` (engine_core_constraint.c:1886), and
        # `R[1] = R[0]/impratio` — so `mu = friction[0]/sqrt(impratio)` and
        # `D[1] = D[0]*impratio`. This branch had the impratio exponent the
        # WRONG WAY (`D_f = D_n/impratio`, i.e. `R[1] = R[0]*impratio`) and
        # `mu` unregularized. Both are exact no-ops at the default
        # `impratio = 1`, which is what every model in the tree uses — no gate
        # here can move, and none did. Corrected rather than left in place
        # because the per-row `R` formula below is built ON `R[1]`, so
        # encoding it wrongly would have propagated to the torsional and
        # rolling rows where the error is no longer a scalar factor.
        var R_t0 = R_n_c / impratio
        if R_t0 < Scalar[DTYPE](1e-14):
            R_t0 = Scalar[DTYPE](1e-14)
        solver[env, ws_mu_idx + c] = mu_c * sqrt(R_t0 / R_n_c)

        # Friction velocity-damping bias: bt = B_damp * J_t * qvel, with THIS
        # CONTACT's mixed damper rather than the model-level one the caller
        # passes in — the same defect the PYRAMIDAL branch above carried, and
        # fixed with it so the two cones cannot drift. Untested on a model
        # that both uses `cone="elliptic"` AND mixes solref across a contacting
        # pair; no such model is in the tree, which is exactly why it is
        # corrected here rather than left as the odd one out.
        var _kb_e = solref_spring_damper[DTYPE](
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_SOLREF_0]),
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_SOLREF_1]),
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_SOLIMP_1]),
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )

        var J_te = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for t in range(NT):
            var ws_Jte = ws_Jt1_idx + t * MC * NV
            if t >= nt_c:
                # Dead row: zero Jacobian, zero stiffness, zero friction. The
                # consumers loop to `nt_c`, but a stale Jacobian left here
                # would still be read by anything that loops to NT.
                for i in range(NV):
                    solver[env, ws_Jte + c * NV + i] = 0
                solver[env, ws_Dt_idx + t * MC + c] = 0
                solver[env, ws_fr_idx + t * MC + c] = 0
                solver[env, ws_bt_idx + t * MC + c] = 0
                continue

            # ⚠ `friction[0]` AND `friction[1]` ARE THE SAME NUMBER HERE. The
            # contact record carries ONE slide coefficient
            # (`CONTACT_IDX_FRICTION`) where MuJoCo has two, so an anisotropic
            # `<geom friction="1 0.5 ...">` slides isotropically. That is a
            # property of the contact record, not of this cone, and the
            # pyramidal builder above shares it.
            var fr_t = mu_c
            if t == 2:
                fr_t = mu_spin_e
            elif t >= 3:
                fr_t = mu_roll_e
            solver[env, ws_fr_idx + t * MC + c] = fr_t

            if t == 0:
                for i in range(NV):
                    J_te[i] = J_t1[i]
            elif t == 1:
                for i in range(NV):
                    J_te[i] = J_t2[i]
            else:
                var ax = nx
                var ay = ny
                var az = nz
                if t == 3:
                    ax = t1x
                    ay = t1y
                    az = t1z
                elif t == 4:
                    ax = t2x
                    ay = t2y
                    az = t2z
                _angular_jacobian_row[
                    DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH
                ](
                    env, joints, bodies, mmeta, cdof,
                    body_a, body_b, ax, ay, az, J_te,
                )

            var bt_t: Scalar[DTYPE] = 0
            for i in range(NV):
                solver[env, ws_Jte + c * NV + i] = J_te[i]
                bt_t += J_te[i] * rebind[Scalar[DTYPE]](qvel[env, i])
            solver[env, ws_bt_idx + t * MC + c] = _kb_e[1] * bt_t

            # ⚠ EVERY TANGENTIAL ROW HAS ITS OWN `R`, SET SO THAT
            # `R[j]*friction[j]^2` IS CONSTANT — `R[j+1] = R[1]*f0^2/fj^2`
            # (engine_core_constraint.c:1893). That is what makes the cone
            # ELLIPTIC rather than circular: the torsional row of a ball at
            # `friction="0.7 0.7 0.05"` is 196x stiffer than its slide rows.
            # The single `D_f` shared by both tangents this replaces was the
            # right answer only because both were slide rows with the same
            # coefficient.
            #
            # A ZERO COEFFICIENT IS A REAL SETTING, not a missing one:
            # `<geom friction="1 0.005 0">` gives `R = inf`, `D = 0`, a row
            # that carries no force and contributes nothing to the cone. The
            # division would produce `inf` and then `1/inf`; taken explicitly
            # so float32 cannot turn it into a NaN.
            if t == 0:
                solver[env, ws_Dt_idx + t * MC + c] = Scalar[DTYPE](1.0) / R_t0
            elif fr_t <= Scalar[DTYPE](0):
                solver[env, ws_Dt_idx + t * MC + c] = 0
            else:
                var R_t = R_t0 * (mu_c * mu_c) / (fr_t * fr_t)
                if R_t < Scalar[DTYPE](1e-14):
                    R_t = Scalar[DTYPE](1e-14)
                solver[env, ws_Dt_idx + t * MC + c] = Scalar[DTYPE](1.0) / R_t


@always_inline
def _warmstart_normals[
    DTYPE: DType,
    NV: Int,
    MAX_CONTACTS: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    env: Int,
    nc: Int,
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
):
    """Apply warm-start normal impulses to predicted velocity (verbatim from
    warmstart_normals_gpu; sequential section)."""
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime ws_lambda_n = 0 * MC
    comptime ws_c_dist = 2 * MC
    comptime ws_MinvJn = 15 * MC + MC * NV

    for c in range(nc):
        if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
            continue
        if solver[env, ws_lambda_n + c] > Scalar[DTYPE](0):
            for i in range(NV):
                qacc_constrained[env, i] += (
                    solver[env, ws_MinvJn + c * NV + i]
                    * solver[env, ws_lambda_n + c]
                )


@always_inline
def _apply_solved_normals[
    DTYPE: DType,
    NV: Int,
    MAX_CONTACTS: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    env: Int,
    nc: Int,
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
):
    """Remove warm-start and apply final solved normal impulses (verbatim
    from apply_solved_normals_gpu — used by the CG/Newton solvers after
    their iterative solve phase; NOT called by the PGS env body)."""
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime ws_lambda_n = 0 * MC
    comptime ws_c_dist = 2 * MC
    comptime ws_MinvJn = 15 * MC + MC * NV

    # Remove warm-start impulses
    for c in range(nc):
        if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
            continue
        var c_off = c * CONTACT_SIZE
        var warm = rebind[Scalar[DTYPE]](
            contacts[env, c_off + CONTACT_IDX_FORCE_N]
        )
        if warm > Scalar[DTYPE](0):
            for i in range(NV):
                qacc_constrained[env, i] -= rebind[Scalar[DTYPE]](
                    solver[env, ws_MinvJn + c * NV + i] * warm
                )

    # Apply final solved impulses
    for c in range(nc):
        if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
            continue
        if solver[env, ws_lambda_n + c] > Scalar[DTYPE](0):
            for i in range(NV):
                qacc_constrained[env, i] += rebind[Scalar[DTYPE]](
                    solver[env, ws_MinvJn + c * NV + i]
                    * solver[env, ws_lambda_n + c]
                )


# =============================================================================
# PGS contact solve — single-source per-env body (port of PGSSolver.solve_gpu)
# =============================================================================


@always_inline
def _contact_solve_env[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    env: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
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
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
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
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
):
    """Full PGS contact solve for one env (verbatim from PGSSolver.solve_gpu,
    serialized per env — see module docstring; limits/equality/tendon run at
    the legacy position via their per-env fields ports)."""
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime V_SIZE = _max_one[NV]()

    # Common normal block offsets (for PGS normal iterations)
    comptime ws_lambda_n = 0 * MC
    comptime ws_K_n = 1 * MC
    comptime ws_c_dist = 2 * MC
    comptime ws_c_body = 3 * MC
    comptime ws_c_body_b = 4 * MC
    comptime ws_c_px = 5 * MC
    comptime ws_c_py = 6 * MC
    comptime ws_c_pz = 7 * MC
    comptime ws_c_nx = 8 * MC
    comptime ws_c_ny = 9 * MC
    comptime ws_c_nz = 10 * MC
    comptime ws_pos_bias = 11 * MC
    comptime ws_inv_K_imp = 12 * MC
    comptime ws_J_n = 15 * MC
    comptime ws_MinvJn = 15 * MC + MC * NV

    # Friction workspace offsets (66*MC + 10*MC*NV, same layout as friction_solver.mojo)
    comptime fws = 15 * MC + 2 * MC * NV
    comptime ws_lf = fws + 0 * MC  # lambda_f[5*MC]
    comptime ws_kf = fws + 5 * MC  # K_f[5*MC]
    comptime ws_df = fws + 10 * MC  # dir_f[15*MC]
    comptime ws_fc = fws + 25 * MC  # fric_coef[5*MC]
    comptime ws_cd = fws + 30 * MC  # condim[MC]
    comptime ws_rf = fws + 31 * MC  # R_f[5*MC] (friction regularizer)
    comptime ws_bf = fws + 36 * MC  # bias_f[5*MC] (velocity damping bias)
    comptime ws_jf = fws + 41 * MC  # J_f[5*MC*NV]
    comptime ws_mj = fws + 41 * MC + 5 * MC * NV  # MinvJ_f[5*MC*NV]
    # Pyramidal-only workspace offsets
    comptime ws_le_neg = fws + 41 * MC + 10 * MC * NV  # lambda_edge_neg[5*MC]
    comptime ws_cnt = ws_le_neg + 5 * MC  # C_nt[5*MC]
    comptime ws_kep = ws_cnt + 5 * MC  # K_edge_pos[5*MC]
    comptime ws_ken = ws_kep + 5 * MC  # K_edge_neg[5*MC]
    comptime ws_re = ws_ken + 5 * MC  # R_edge[5*MC]

    # === Initialize workspace (legacy: parallel, one thread per slot) ===
    for contact_tid in range(MC):
        _init_common_normal_ws[
            DTYPE, NV, MAX_CONTACTS, BATCH, SOLVER_WS
        ](env, contact_tid, solver)
        # Init friction workspace for this contact slot
        for d in range(5):
            solver[env, ws_lf + d * MC + contact_tid] = 0
            solver[env, ws_kf + d * MC + contact_tid] = 1
            solver[env, ws_fc + d * MC + contact_tid] = 0
            solver[env, ws_rf + d * MC + contact_tid] = 0
            solver[env, ws_bf + d * MC + contact_tid] = 0
            # Pyramidal workspace
            solver[env, ws_le_neg + d * MC + contact_tid] = 0
            solver[env, ws_cnt + d * MC + contact_tid] = 0
            solver[env, ws_kep + d * MC + contact_tid] = 1
            solver[env, ws_ken + d * MC + contact_tid] = 1
            solver[env, ws_re + d * MC + contact_tid] = 0
            for axis in range(3):
                solver[env, ws_df + (d * 3 + axis) * MC + contact_tid] = 0
        solver[env, ws_cd + contact_tid] = 3  # default condim=3

    # Read metadata (legacy `dt` read dropped — only the excluded limits
    # call consumed it)
    var nc = 0
    var K_spring: Scalar[DTYPE] = 0
    var B_damp: Scalar[DTYPE] = 0
    var si_dmin: Scalar[DTYPE] = 0
    var si_dmax: Scalar[DTYPE] = 0
    var si_width: Scalar[DTYPE] = 1
    var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
    var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)

    nc = Int(rebind[Scalar[DTYPE]](smeta[env, META_IDX_NUM_CONTACTS]))
    if nc > MAX_CONTACTS:
        nc = MAX_CONTACTS
    var sr_tc = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLREF_CONTACT_0]
    )
    var sr_dr = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLREF_CONTACT_1]
    )
    si_dmin = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_0])
    si_dmax = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_1])
    si_width = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_2])
    si_midpoint = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLIMP_CONTACT_3]
    )
    si_power = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_4])
    if si_width < Scalar[DTYPE](1e-6):
        si_width = Scalar[DTYPE](1e-6)
    # MuJoCo clamps BOTH ends of solimp to [mjMINIMP, mjMAXIMP] before ever
    # interpolating (engine_core_constraint.c:1284-1287), and the floor on
    # dmin is the one that bites: R = (1-imp)/imp * diagApprox blows up as
    # imp -> 0, so a model that asks for dmin=0 gets a contact that is soft
    # by orders of magnitude rather than merely soft. dm_control's finger is
    # the first model here to do exactly that (`solimp="0 0.9 0.01"`); every
    # earlier model used the 0.9 default, which is why clamping only dmax
    # survived this long. Our old floor sat on the INTERPOLATED imp at 1e-6,
    # 100x below MuJoCo's 1e-4, so first touch was ~100x too soft.
    comptime MJ_MINIMP = Scalar[DTYPE](0.0001)
    comptime MJ_MAXIMP = Scalar[DTYPE](0.9999)
    if si_dmin < MJ_MINIMP:
        si_dmin = MJ_MINIMP
    elif si_dmin > MJ_MAXIMP:
        si_dmin = MJ_MAXIMP
    if si_dmax < MJ_MINIMP:
        si_dmax = MJ_MINIMP
    elif si_dmax > MJ_MAXIMP:
        si_dmax = MJ_MAXIMP
    if si_power < Scalar[DTYPE](1):
        si_power = Scalar[DTYPE](1)
    # K = 1/(dmax^2 * timeconst^2 * dampratio^2), B = 2/(dmax * timeconst)
    # (engine_core_constraint.c:1432,1440). The dampratio belongs SQUARED in
    # K and not at all in B; this used to have it linearly in B and absent
    # from K. Every model in the repo runs dampratio=1, where the two forms
    # coincide exactly — which is why it never showed up — but `limits.mojo`
    # already had the MuJoCo form, so the two constraint paths disagreed.
    # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
    # solref. See `constraints/constraint_data.solref_spring_damper` — the
    # formula lived in twelve copy-pasted sites until 2026-08-03.
    (K_spring, B_damp) = solref_spring_damper[DTYPE](
        sr_tc, sr_dr, si_dmax,
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
    )

    # === PHASE 1: normal precompute (legacy: parallel, one thread per
    # contact slot; internal `contact_tid < nc` guard kept in the helper) ===
    for contact_tid in range(MC):
        _precompute_contact_normal[
            DTYPE, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, BATCH, SOLVER_WS
        ](
            env,
            contact_tid,
            nc,
            qvel,
            subtree_com,
            contacts,
            joints,
            bodies,
            mmeta,
            body_invweight0,
            cdof,
            m_inv,
            qacc_constrained,
            solver,
            K_spring,
            B_damp,
            si_dmin,
            si_dmax,
            si_width,
            si_midpoint,
            si_power,
        )

    # === SEQUENTIAL: Warm start + PGS normal (legacy: thread 0) ===
    _warmstart_normals[DTYPE, NV, MAX_CONTACTS, BATCH, SOLVER_WS](
        env, nc, qacc_constrained, solver
    )

    # PGS normal iterations (acceleration-level)
    for _ in range(PGS_ITERATIONS):
        var max_delta: solver.element_type = 0
        for c in range(nc):
            if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                continue
            var a_n: solver.element_type = 0
            for i in range(NV):
                a_n += (
                    solver[env, ws_J_n + c * NV + i]
                    * qacc_constrained[env, i]
                )
            var R_n = Scalar[DTYPE](1.0) / rebind[Scalar[DTYPE]](
                solver[env, ws_inv_K_imp + c]
            ) - rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
            var residual = (
                a_n
                + solver[env, ws_pos_bias + c]
                + R_n * solver[env, ws_lambda_n + c]
            )
            var delta = -residual * solver[env, ws_inv_K_imp + c]
            var old_lambda = solver[env, ws_lambda_n + c]
            solver[env, ws_lambda_n + c] = (
                solver[env, ws_lambda_n + c] + delta
            )
            if solver[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                solver[env, ws_lambda_n + c] = Scalar[DTYPE](0)
            var actual_delta = solver[env, ws_lambda_n + c] - old_lambda
            var abs_delta = abs(actual_delta)
            if abs_delta > max_delta:
                max_delta = abs_delta
            for i in range(NV):
                qacc_constrained[env, i] += (
                    solver[env, ws_MinvJn + c * NV + i] * actual_delta
                )
        if max_delta < Scalar[DTYPE](1e-4):
            break

    # Joint limits — legacy position (between the normal PGS and the
    # friction phase), legacy iteration count (PGS_ITERATIONS, not the
    # Newton path's 50).
    _limits_env[DTYPE, NQ, NV, NJOINT, BATCH, PGS_ITERATIONS](
        env, qpos, qvel, joints, mmeta, dof_invweight0, m_inv,
        qacc_constrained,
    )
    # Dry-friction dof rows (MuJoCo mjCNSTR_FRICTION_DOF), solved
    # beside the limit rows. No-op for a model with no frictionloss.
    _friction_env[DTYPE, NQ, NV, NJOINT, BATCH, PGS_ITERATIONS](
        env, qvel, joints,
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        dof_invweight0, m_inv, qacc_constrained
    )

    # Equality constraints — legacy position (right after limits; the legacy
    # call is unconditional with a comptime gate inside the builder, which
    # this call-site gate matches bit-identically for NEQUALITY == 0).
    comptime if NEQUALITY > 0:
        _equality_env[
            DTYPE, NQ, NV, NBODY, NJOINT, NEQUALITY, V_SIZE,
            BATCH, PGS_ITERATIONS,
        ](
            env, qpos, qvel, xpos, xquat, subtree_com, joints, bodies, mmeta,
            equality, body_invweight0, dof_invweight0, cdof,
            m_inv, qacc_constrained,
        )

    # Tendon equality constraints — legacy call-site gate
    # (`comptime if MAX_TENDON > 0` in PGSSolver.solve_gpu).
    comptime if NTENDON > 0:
        _tendon_env[
            DTYPE, NQ, NV, NBODY, NJOINT, NTENDON, NSITE, BATCH,
            PGS_ITERATIONS,
        ](
            env, qpos, qvel, joints, mmeta, tendons, sites, bodies,
            subtree_com, cdof, xpos, xquat, m_inv, qacc_constrained,
        )

    # === PHASE 3: friction precompute (legacy: parallel, guarded
    # `contact_tid < nc`) ===
    for contact_tid in range(nc):
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            J_row[i] = 0

        var c = contact_tid
        if solver[env, ws_lambda_n + c] > 0:
            var c_off = c * CONTACT_SIZE
            var nx = rebind[Scalar[DTYPE]](solver[env, ws_c_nx + c])
            var ny = rebind[Scalar[DTYPE]](solver[env, ws_c_ny + c])
            var nz = rebind[Scalar[DTYPE]](solver[env, ws_c_nz + c])

            # Read per-contact friction params
            var mu_slide = rebind[Scalar[DTYPE]](
                contacts[env, c_off + CONTACT_IDX_FRICTION]
            )
            if mu_slide <= Scalar[DTYPE](0):
                mu_slide = Scalar[DTYPE](0.5)  # fallback
            var mu_spin = rebind[Scalar[DTYPE]](
                contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN]
            )
            var mu_roll = rebind[Scalar[DTYPE]](
                contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL]
            )
            var condim = Int(
                rebind[Scalar[DTYPE]](
                    contacts[env, c_off + CONTACT_IDX_CONDIM]
                )
            )
            if condim < 1:
                condim = 3
            solver[env, ws_cd + c] = Scalar[DTYPE](condim)

            if condim > 1:
                # Tangent basis (MuJoCo mju_makeFrame with capsule axis hint)
                var hint_x = rebind[Scalar[DTYPE]](
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_X]
                )
                var hint_y = rebind[Scalar[DTYPE]](
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y]
                )
                var hint_z = rebind[Scalar[DTYPE]](
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z]
                )
                var frame = contact_tangent_frame[DTYPE](
                    nx, ny, nz, hint_x, hint_y, hint_z
                )
                var t1x = frame[0]
                var t1y = frame[1]
                var t1z = frame[2]
                var t2x = frame[3]
                var t2y = frame[4]
                var t2z = frame[5]

                # Store directions and friction coefficients
                solver[env, ws_df + (0 * 3 + 0) * MC + c] = t1x
                solver[env, ws_df + (0 * 3 + 1) * MC + c] = t1y
                solver[env, ws_df + (0 * 3 + 2) * MC + c] = t1z
                solver[env, ws_df + (1 * 3 + 0) * MC + c] = t2x
                solver[env, ws_df + (1 * 3 + 1) * MC + c] = t2y
                solver[env, ws_df + (1 * 3 + 2) * MC + c] = t2z
                solver[env, ws_fc + 0 * MC + c] = mu_slide
                solver[env, ws_fc + 1 * MC + c] = mu_slide

                var num_fric = 2
                if condim >= 4:
                    num_fric = 3
                    solver[env, ws_df + (2 * 3 + 0) * MC + c] = nx
                    solver[env, ws_df + (2 * 3 + 1) * MC + c] = ny
                    solver[env, ws_df + (2 * 3 + 2) * MC + c] = nz
                    solver[env, ws_fc + 2 * MC + c] = mu_spin
                if condim >= 6:
                    num_fric = 5
                    solver[env, ws_df + (3 * 3 + 0) * MC + c] = t1x
                    solver[env, ws_df + (3 * 3 + 1) * MC + c] = t1y
                    solver[env, ws_df + (3 * 3 + 2) * MC + c] = t1z
                    solver[env, ws_df + (4 * 3 + 0) * MC + c] = t2x
                    solver[env, ws_df + (4 * 3 + 1) * MC + c] = t2y
                    solver[env, ws_df + (4 * 3 + 2) * MC + c] = t2z
                    solver[env, ws_fc + 3 * MC + c] = mu_roll
                    solver[env, ws_fc + 4 * MC + c] = mu_roll

                var body_a = Int(solver[env, ws_c_body + c])
                var body_b = Int(solver[env, ws_c_body_b + c])
                var px = rebind[Scalar[DTYPE]](solver[env, ws_c_px + c])
                var py = rebind[Scalar[DTYPE]](solver[env, ws_c_py + c])
                var pz = rebind[Scalar[DTYPE]](solver[env, ws_c_pz + c])

                # Compute J, MinvJ, K for each friction direction
                for d in range(num_fric):
                    var dx = rebind[Scalar[DTYPE]](
                        solver[env, ws_df + (d * 3 + 0) * MC + c]
                    )
                    var dy = rebind[Scalar[DTYPE]](
                        solver[env, ws_df + (d * 3 + 1) * MC + c]
                    )
                    var dz = rebind[Scalar[DTYPE]](
                        solver[env, ws_df + (d * 3 + 2) * MC + c]
                    )

                    if d < 2:
                        _contact_jacobian_row[
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
                            px,
                            py,
                            pz,
                            dx,
                            dy,
                            dz,
                            J_row,
                        )
                    else:
                        _angular_jacobian_row[
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

                    var k_d: solver.element_type = 0
                    for i in range(NV):
                        solver[env, ws_jf + d * MC * NV + c * NV + i] = J_row[
                            i
                        ]
                        var mi_j_sum: solver.element_type = 0
                        for j_idx in range(NV):
                            mi_j_sum += (
                                m_inv[env, i * NV + j_idx] * J_row[j_idx]
                            )
                        solver[
                            env, ws_mj + d * MC * NV + c * NV + i
                        ] = mi_j_sum
                        k_d += J_row[i] * mi_j_sum
                    if k_d < Scalar[DTYPE](1e-10):
                        k_d = Scalar[DTYPE](1e-10)
                    solver[env, ws_kf + d * MC + c] = k_d

                # Compute friction regularizer R_f from parent normal's impedance
                var impratio_pgs = rebind[Scalar[DTYPE]](
                    mmeta[MODEL_META_IDX_IMPRATIO]
                )
                if impratio_pgs < Scalar[DTYPE](1e-6):
                    impratio_pgs = Scalar[DTYPE](1.0)
                var imp_n_pgs = rebind[Scalar[DTYPE]](
                    solver[env, ws_inv_K_imp + c]
                ) * rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
                var R_base_pgs = (
                    (Scalar[DTYPE](1.0) - imp_n_pgs)
                    / imp_n_pgs
                    * rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
                    / impratio_pgs
                )
                for d in range(num_fric):
                    var R_d_pgs = R_base_pgs
                    if d >= 2:
                        var mu_d_pgs = rebind[Scalar[DTYPE]](
                            solver[env, ws_fc + d * MC + c]
                        )
                        if mu_d_pgs > Scalar[DTYPE](1e-12):
                            R_d_pgs = (
                                R_base_pgs
                                * mu_slide
                                * mu_slide
                                / (mu_d_pgs * mu_d_pgs)
                            )
                    solver[env, ws_rf + d * MC + c] = R_d_pgs

                # Compute velocity damping bias for friction rows
                for d in range(num_fric):
                    var v_t: solver.element_type = 0
                    for i in range(NV):
                        v_t += rebind[Scalar[DTYPE]](
                            solver[env, ws_jf + d * MC * NV + c * NV + i]
                        ) * rebind[Scalar[DTYPE]](qvel[env, i])
                    solver[env, ws_bf + d * MC + c] = B_damp * rebind[
                        Scalar[DTYPE]
                    ](v_t)

                comptime if CONE_TYPE == ConeType.PYRAMIDAL:
                    # Pyramidal precomputation: C_nt, K_edge_pos/neg, R_edge
                    var R_n_val = (
                        (Scalar[DTYPE](1.0) - imp_n_pgs)
                        / imp_n_pgs
                        * rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
                    )
                    for d in range(num_fric):
                        var mu_d_p = rebind[Scalar[DTYPE]](
                            solver[env, ws_fc + d * MC + c]
                        )
                        # Cross-term: C_nt[d][c] = Σ_i J_n[c*NV+i] * MinvJ_f[d*MC*NV+c*NV+i]
                        var c_nt_val: solver.element_type = 0
                        for i in range(NV):
                            c_nt_val += rebind[Scalar[DTYPE]](
                                solver[env, ws_J_n + c * NV + i]
                            ) * rebind[Scalar[DTYPE]](
                                solver[env, ws_mj + d * MC * NV + c * NV + i]
                            )
                        solver[env, ws_cnt + d * MC + c] = c_nt_val
                        var K_n_c = rebind[Scalar[DTYPE]](
                            solver[env, ws_K_n + c]
                        )
                        var K_f_d = rebind[Scalar[DTYPE]](
                            solver[env, ws_kf + d * MC + c]
                        )
                        solver[env, ws_kep + d * MC + c] = (
                            K_n_c
                            + Scalar[DTYPE](2.0) * mu_d_p * c_nt_val
                            + mu_d_p * mu_d_p * K_f_d
                        )
                        solver[env, ws_ken + d * MC + c] = (
                            K_n_c
                            - Scalar[DTYPE](2.0) * mu_d_p * c_nt_val
                            + mu_d_p * mu_d_p * K_f_d
                        )
                        solver[env, ws_re + d * MC + c] = (
                            Scalar[DTYPE](2.0) * mu_d_p * mu_d_p * R_n_val
                        )
                    # No warm-start for pyramidal
                    for d in range(num_fric):
                        solver[env, ws_lf + d * MC + c] = Scalar[DTYPE](0)
                        solver[env, ws_le_neg + d * MC + c] = Scalar[DTYPE](0)
                else:
                    # Warm-start friction impulses (elliptic only)
                    var warm_idx = InlineArray[Int, 5](uninitialized=True)
                    warm_idx[0] = CONTACT_IDX_FORCE_T1
                    warm_idx[1] = CONTACT_IDX_FORCE_T2
                    warm_idx[2] = CONTACT_IDX_FORCE_TORSION
                    warm_idx[3] = CONTACT_IDX_FORCE_ROLL1
                    warm_idx[4] = CONTACT_IDX_FORCE_ROLL2
                    for d in range(num_fric):
                        solver[env, ws_lf + d * MC + c] = rebind[
                            Scalar[DTYPE]
                        ](contacts[env, c_off + warm_idx[d]])

    # === SEQUENTIAL: Coupled PGS (normals + friction) + impulse store
    # (legacy: thread 0) ===
    # Coupled PGS iterations (normals + friction together, MuJoCo-style)
    for _ in range(PGS_ITERATIONS):
        # --- Normal constraints PGS update ---
        for c in range(nc):
            if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                continue
            var a_n: solver.element_type = 0
            for i in range(NV):
                a_n += (
                    solver[env, ws_J_n + c * NV + i]
                    * qacc_constrained[env, i]
                )
            var R_n = Scalar[DTYPE](1.0) / rebind[Scalar[DTYPE]](
                solver[env, ws_inv_K_imp + c]
            ) - rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
            var residual = (
                a_n
                + solver[env, ws_pos_bias + c]
                + R_n * solver[env, ws_lambda_n + c]
            )
            var delta = -residual * solver[env, ws_inv_K_imp + c]
            var old_lambda = solver[env, ws_lambda_n + c]
            solver[env, ws_lambda_n + c] = (
                solver[env, ws_lambda_n + c] + delta
            )
            if solver[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                solver[env, ws_lambda_n + c] = Scalar[DTYPE](0)
            var actual_n = solver[env, ws_lambda_n + c] - old_lambda
            for i in range(NV):
                qacc_constrained[env, i] += (
                    solver[env, ws_MinvJn + c * NV + i] * actual_n
                )

        # --- Friction constraints PGS update ---
        for c in range(nc):
            if solver[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                # Zero friction when normal force is zero
                var condim_z = Int(solver[env, ws_cd + c])
                var num_fric_z = 2
                if condim_z >= 4:
                    num_fric_z = 3
                if condim_z >= 6:
                    num_fric_z = 5
                for d in range(num_fric_z):
                    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
                        var mu_d = rebind[Scalar[DTYPE]](
                            solver[env, ws_fc + d * MC + c]
                        )
                        var old_pos = rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                        var old_neg_v = rebind[Scalar[DTYPE]](
                            solver[env, ws_le_neg + d * MC + c]
                        )
                        if old_pos != Scalar[DTYPE](
                            0
                        ) or old_neg_v != Scalar[DTYPE](0):
                            solver[env, ws_lf + d * MC + c] = Scalar[DTYPE](0)
                            solver[env, ws_le_neg + d * MC + c] = Scalar[
                                DTYPE
                            ](0)
                            for i in range(NV):
                                var minvjn_i = rebind[Scalar[DTYPE]](
                                    solver[env, ws_MinvJn + c * NV + i]
                                )
                                var minvjf_i = rebind[Scalar[DTYPE]](
                                    solver[
                                        env, ws_mj + d * MC * NV + c * NV + i
                                    ]
                                )
                                qacc_constrained[env, i] -= (
                                    minvjn_i + mu_d * minvjf_i
                                ) * old_pos
                                qacc_constrained[env, i] -= (
                                    minvjn_i - mu_d * minvjf_i
                                ) * old_neg_v
                    else:
                        var old_f = rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                        if old_f != Scalar[DTYPE](0):
                            solver[env, ws_lf + d * MC + c] = Scalar[DTYPE](0)
                            for i in range(NV):
                                qacc_constrained[env, i] -= (
                                    solver[
                                        env, ws_mj + d * MC * NV + c * NV + i
                                    ]
                                    * old_f
                                )
                continue
            var condim = Int(solver[env, ws_cd + c])
            if condim == 1:
                continue

            var num_fric = 2
            if condim >= 4:
                num_fric = 3
            if condim >= 6:
                num_fric = 5

            var lambda_n = rebind[Scalar[DTYPE]](
                solver[env, ws_lambda_n + c]
            )

            comptime if CONE_TYPE == ConeType.PYRAMIDAL:
                # === PYRAMIDAL CONE: Edge constraints with λ ≥ 0 ===
                var bias_n = rebind[Scalar[DTYPE]](
                    solver[env, ws_pos_bias + c]
                )

                for d in range(num_fric):
                    var mu_d = rebind[Scalar[DTYPE]](
                        solver[env, ws_fc + d * MC + c]
                    )
                    if mu_d <= Scalar[DTYPE](1e-12):
                        continue

                    var a_n_val: solver.element_type = 0
                    var a_f_val: solver.element_type = 0
                    for i in range(NV):
                        var qi = rebind[Scalar[DTYPE]](
                            qacc_constrained[env, i]
                        )
                        a_n_val += (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_J_n + c * NV + i]
                            )
                            * qi
                        )
                        a_f_val += (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_jf + d * MC * NV + c * NV + i]
                            )
                            * qi
                        )

                    var R_e = rebind[Scalar[DTYPE]](
                        solver[env, ws_re + d * MC + c]
                    )

                    # Positive edge (+)
                    var a_edge_pos = a_n_val + mu_d * a_f_val
                    var K_ep = rebind[Scalar[DTYPE]](
                        solver[env, ws_kep + d * MC + c]
                    )
                    var residual_pos = (
                        a_edge_pos
                        + bias_n
                        + R_e
                        * rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                    )
                    var delta_pos = -residual_pos / (K_ep + R_e)
                    var new_lp = (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                        + delta_pos
                    )
                    if new_lp < Scalar[DTYPE](0):
                        new_lp = Scalar[DTYPE](0)
                    var actual_pos = new_lp - rebind[Scalar[DTYPE]](
                        solver[env, ws_lf + d * MC + c]
                    )
                    solver[env, ws_lf + d * MC + c] = new_lp
                    if actual_pos != Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc_constrained[env, i] += (
                                rebind[Scalar[DTYPE]](
                                    solver[env, ws_MinvJn + c * NV + i]
                                )
                                + mu_d
                                * rebind[Scalar[DTYPE]](
                                    solver[
                                        env,
                                        ws_mj + d * MC * NV + c * NV + i,
                                    ]
                                )
                            ) * actual_pos

                    # Recompute after positive edge
                    a_n_val = 0
                    a_f_val = 0
                    for i in range(NV):
                        var qi = rebind[Scalar[DTYPE]](
                            qacc_constrained[env, i]
                        )
                        a_n_val += (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_J_n + c * NV + i]
                            )
                            * qi
                        )
                        a_f_val += (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_jf + d * MC * NV + c * NV + i]
                            )
                            * qi
                        )

                    # Negative edge (-)
                    var a_edge_neg = a_n_val - mu_d * a_f_val
                    var K_en = rebind[Scalar[DTYPE]](
                        solver[env, ws_ken + d * MC + c]
                    )
                    var residual_neg = (
                        a_edge_neg
                        + bias_n
                        + R_e
                        * rebind[Scalar[DTYPE]](
                            solver[env, ws_le_neg + d * MC + c]
                        )
                    )
                    var delta_neg = -residual_neg / (K_en + R_e)
                    var new_ln = (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_le_neg + d * MC + c]
                        )
                        + delta_neg
                    )
                    if new_ln < Scalar[DTYPE](0):
                        new_ln = Scalar[DTYPE](0)
                    var actual_neg = new_ln - rebind[Scalar[DTYPE]](
                        solver[env, ws_le_neg + d * MC + c]
                    )
                    solver[env, ws_le_neg + d * MC + c] = new_ln
                    if actual_neg != Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc_constrained[env, i] += (
                                rebind[Scalar[DTYPE]](
                                    solver[env, ws_MinvJn + c * NV + i]
                                )
                                - mu_d
                                * rebind[Scalar[DTYPE]](
                                    solver[
                                        env,
                                        ws_mj + d * MC * NV + c * NV + i,
                                    ]
                                )
                            ) * actual_neg
                _ = lambda_n
            else:
                # === ELLIPTIC CONE: MuJoCo-style block update ===
                # Ray update + QCQP with AR submatrix + costChange
                var dim = 1 + num_fric

                # Build block AR matrix on-the-fly from J/MinvJ
                var AR = InlineArray[Scalar[DTYPE], 36](
                    fill=Scalar[DTYPE](0)
                )
                # Compute R_n directly from stored imp and diag_n
                comptime ws_imp_n_pgs = 13 * MC
                comptime ws_diag_n_pgs = 14 * MC
                var imp_pgs = rebind[Scalar[DTYPE]](
                    solver[env, ws_imp_n_pgs + c]
                )
                var diag_pgs = rebind[Scalar[DTYPE]](
                    solver[env, ws_diag_n_pgs + c]
                )
                var R_n_val = (
                    (Scalar[DTYPE](1.0) - imp_pgs) / imp_pgs * diag_pgs
                )
                AR[0] = (
                    rebind[Scalar[DTYPE]](solver[env, ws_K_n + c]) + R_n_val
                )

                for d1 in range(num_fric):
                    # Normal-friction cross: J_n @ MinvJ_f[d1]
                    var cross: Scalar[DTYPE] = 0
                    for i in range(NV):
                        cross += rebind[Scalar[DTYPE]](
                            solver[env, ws_J_n + c * NV + i]
                        ) * rebind[Scalar[DTYPE]](
                            solver[env, ws_mj + d1 * MC * NV + c * NV + i]
                        )
                    AR[(d1 + 1)] = cross
                    AR[(d1 + 1) * dim] = cross

                    for d2 in range(num_fric):
                        var ff: Scalar[DTYPE] = 0
                        for i in range(NV):
                            ff += rebind[Scalar[DTYPE]](
                                solver[
                                    env, ws_jf + d1 * MC * NV + c * NV + i
                                ]
                            ) * rebind[Scalar[DTYPE]](
                                solver[
                                    env, ws_mj + d2 * MC * NV + c * NV + i
                                ]
                            )
                        if d1 == d2:
                            ff += rebind[Scalar[DTYPE]](
                                solver[env, ws_rf + d1 * MC + c]
                            )
                        AR[(d1 + 1) * dim + (d2 + 1)] = ff

                # Compute block residual
                var block_res = InlineArray[Scalar[DTYPE], 6](
                    fill=Scalar[DTYPE](0)
                )
                var a_n_res: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_n_res += rebind[Scalar[DTYPE]](
                        solver[env, ws_J_n + c * NV + i]
                    ) * rebind[Scalar[DTYPE]](qacc_constrained[env, i])
                block_res[0] = (
                    a_n_res
                    + rebind[Scalar[DTYPE]](solver[env, ws_pos_bias + c])
                    + R_n_val
                    * rebind[Scalar[DTYPE]](solver[env, ws_lambda_n + c])
                )
                for d in range(num_fric):
                    var a_f_res: Scalar[DTYPE] = 0
                    for i in range(NV):
                        a_f_res += rebind[Scalar[DTYPE]](
                            solver[env, ws_jf + d * MC * NV + c * NV + i]
                        ) * rebind[Scalar[DTYPE]](qacc_constrained[env, i])
                    var R_f_d = rebind[Scalar[DTYPE]](
                        solver[env, ws_rf + d * MC + c]
                    )
                    block_res[1 + d] = (
                        a_f_res
                        + rebind[Scalar[DTYPE]](solver[env, ws_bf + d * MC + c])
                        + R_f_d
                        * rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                    )

                # Save old forces
                var oldforce = InlineArray[Scalar[DTYPE], 6](
                    fill=Scalar[DTYPE](0)
                )
                oldforce[0] = rebind[Scalar[DTYPE]](
                    solver[env, ws_lambda_n + c]
                )
                for d in range(num_fric):
                    oldforce[1 + d] = rebind[Scalar[DTYPE]](
                        solver[env, ws_lf + d * MC + c]
                    )

                var ARinv0: Scalar[DTYPE] = 0
                if AR[0] > Scalar[DTYPE](1e-10):
                    ARinv0 = Scalar[DTYPE](1.0) / AR[0]

                # --- Ray update ---
                if rebind[Scalar[DTYPE]](
                    solver[env, ws_lambda_n + c]
                ) < Scalar[DTYPE](1e-10):
                    solver[env, ws_lambda_n + c] = (
                        rebind[Scalar[DTYPE]](solver[env, ws_lambda_n + c])
                        - block_res[0] * ARinv0
                    )
                    if solver[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                        solver[env, ws_lambda_n + c] = Scalar[DTYPE](0)
                    for d in range(num_fric):
                        solver[env, ws_lf + d * MC + c] = Scalar[DTYPE](0)
                else:
                    var v = InlineArray[Scalar[DTYPE], 6](
                        fill=Scalar[DTYPE](0)
                    )
                    v[0] = rebind[Scalar[DTYPE]](
                        solver[env, ws_lambda_n + c]
                    )
                    for d in range(num_fric):
                        v[1 + d] = rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                    var denom: Scalar[DTYPE] = 0
                    for bi in range(dim):
                        for bj in range(dim):
                            denom += v[bi] * AR[bi * dim + bj] * v[bj]
                    if denom >= Scalar[DTYPE](1e-10):
                        var vdotr: Scalar[DTYPE] = 0
                        for bi in range(dim):
                            vdotr += v[bi] * block_res[bi]
                        var x = -vdotr / denom
                        if rebind[Scalar[DTYPE]](
                            solver[env, ws_lambda_n + c]
                        ) + x * v[0] < Scalar[DTYPE](0):
                            x = (
                                -rebind[Scalar[DTYPE]](
                                    solver[env, ws_lambda_n + c]
                                )
                                / v[0]
                            )
                        solver[env, ws_lambda_n + c] = (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_lambda_n + c]
                            )
                            + x * v[0]
                        )
                        for d in range(num_fric):
                            solver[env, ws_lf + d * MC + c] = (
                                rebind[Scalar[DTYPE]](
                                    solver[env, ws_lf + d * MC + c]
                                )
                                + x * v[1 + d]
                            )

                # --- QCQP friction update ---
                var fn_val = rebind[Scalar[DTYPE]](
                    solver[env, ws_lambda_n + c]
                )
                if fn_val >= Scalar[DTYPE](1e-10) and num_fric > 0:
                    var Ac = InlineArray[Scalar[DTYPE], 25](
                        fill=Scalar[DTYPE](0)
                    )
                    var bc_arr = InlineArray[Scalar[DTYPE], 5](
                        fill=Scalar[DTYPE](0)
                    )
                    for j in range(num_fric):
                        for j2 in range(num_fric):
                            Ac[j * num_fric + j2] = AR[
                                (1 + j) * dim + (1 + j2)
                            ]
                        bc_arr[j] = block_res[1 + j]
                        for j2 in range(num_fric):
                            bc_arr[j] -= (
                                Ac[j * num_fric + j2] * oldforce[1 + j2]
                            )
                        bc_arr[j] += AR[(1 + j) * dim + 0] * (
                            fn_val - oldforce[0]
                        )

                    var mu_arr = InlineArray[Scalar[DTYPE], 5](
                        fill=Scalar[DTYPE](0)
                    )
                    for d in range(num_fric):
                        mu_arr[d] = rebind[Scalar[DTYPE]](
                            solver[env, ws_fc + d * MC + c]
                        )

                    var flg_active = False
                    if num_fric == 2:
                        var A2 = InlineArray[Scalar[DTYPE], 4](
                            fill=Scalar[DTYPE](0)
                        )
                        var b2 = InlineArray[Scalar[DTYPE], 2](
                            fill=Scalar[DTYPE](0)
                        )
                        var d2 = InlineArray[Scalar[DTYPE], 2](
                            fill=Scalar[DTYPE](0)
                        )
                        for ii in range(2):
                            b2[ii] = bc_arr[ii]
                            d2[ii] = mu_arr[ii]
                            for jj in range(2):
                                A2[ii * 2 + jj] = Ac[ii * num_fric + jj]
                        var r0: Scalar[DTYPE] = 0
                        var r1: Scalar[DTYPE] = 0
                        flg_active = mj_qcqp2[DTYPE](
                            r0, r1, A2, b2, d2, fn_val
                        )
                        solver[env, ws_lf + 0 * MC + c] = r0
                        solver[env, ws_lf + 1 * MC + c] = r1
                    elif num_fric == 3:
                        var A3 = InlineArray[Scalar[DTYPE], 9](
                            fill=Scalar[DTYPE](0)
                        )
                        var b3 = InlineArray[Scalar[DTYPE], 3](
                            fill=Scalar[DTYPE](0)
                        )
                        var d3 = InlineArray[Scalar[DTYPE], 3](
                            fill=Scalar[DTYPE](0)
                        )
                        for ii in range(3):
                            b3[ii] = bc_arr[ii]
                            d3[ii] = mu_arr[ii]
                            for jj in range(3):
                                A3[ii * 3 + jj] = Ac[ii * num_fric + jj]
                        var r0: Scalar[DTYPE] = 0
                        var r1: Scalar[DTYPE] = 0
                        var r2: Scalar[DTYPE] = 0
                        flg_active = mj_qcqp3[DTYPE](
                            r0, r1, r2, A3, b3, d3, fn_val
                        )
                        solver[env, ws_lf + 0 * MC + c] = r0
                        solver[env, ws_lf + 1 * MC + c] = r1
                        solver[env, ws_lf + 2 * MC + c] = r2
                    elif num_fric == 5:
                        var A5 = InlineArray[Scalar[DTYPE], 25](
                            fill=Scalar[DTYPE](0)
                        )
                        var b5 = InlineArray[Scalar[DTYPE], 5](
                            fill=Scalar[DTYPE](0)
                        )
                        var d5 = InlineArray[Scalar[DTYPE], 5](
                            fill=Scalar[DTYPE](0)
                        )
                        for ii in range(5):
                            b5[ii] = bc_arr[ii]
                            d5[ii] = mu_arr[ii]
                            for jj in range(5):
                                A5[ii * 5 + jj] = Ac[ii * num_fric + jj]
                        var res5 = InlineArray[Scalar[DTYPE], 5](
                            fill=Scalar[DTYPE](0)
                        )
                        flg_active = mj_qcqp5[DTYPE](
                            res5, A5, b5, d5, fn_val
                        )
                        for d in range(5):
                            solver[env, ws_lf + d * MC + c] = res5[d]

                    # Rescale to exact ellipsoid if constrained
                    if flg_active:
                        var s: Scalar[DTYPE] = 0
                        for d in range(num_fric):
                            var fv = rebind[Scalar[DTYPE]](
                                solver[env, ws_lf + d * MC + c]
                            )
                            var mu_d = mu_arr[d]
                            if mu_d > Scalar[DTYPE](1e-10):
                                s += fv * fv / (mu_d * mu_d)
                        if s > Scalar[DTYPE](1e-10):
                            var scale = sqrt(fn_val * fn_val / s)
                            for d in range(num_fric):
                                solver[env, ws_lf + d * MC + c] = (
                                    rebind[Scalar[DTYPE]](
                                        solver[env, ws_lf + d * MC + c]
                                    )
                                    * scale
                                )

                # --- Cost descent check ---
                var cost_val: Scalar[DTYPE] = 0
                for bi in range(dim):
                    var new_i: Scalar[DTYPE]
                    var old_i: Scalar[DTYPE]
                    if bi == 0:
                        new_i = rebind[Scalar[DTYPE]](
                            solver[env, ws_lambda_n + c]
                        )
                        old_i = oldforce[0]
                    else:
                        new_i = rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + (bi - 1) * MC + c]
                        )
                        old_i = oldforce[bi]
                    var delta_i = new_i - old_i
                    cost_val += delta_i * block_res[bi]
                    for bj in range(dim):
                        var new_j: Scalar[DTYPE]
                        var old_j: Scalar[DTYPE]
                        if bj == 0:
                            new_j = rebind[Scalar[DTYPE]](
                                solver[env, ws_lambda_n + c]
                            )
                            old_j = oldforce[0]
                        else:
                            new_j = rebind[Scalar[DTYPE]](
                                solver[env, ws_lf + (bj - 1) * MC + c]
                            )
                            old_j = oldforce[bj]
                        var delta_j = new_j - old_j
                        cost_val += (
                            Scalar[DTYPE](0.5)
                            * delta_i
                            * AR[bi * dim + bj]
                            * delta_j
                        )

                if cost_val > Scalar[DTYPE](1e-10):
                    # Revert
                    solver[env, ws_lambda_n + c] = oldforce[0]
                    for d in range(num_fric):
                        solver[env, ws_lf + d * MC + c] = oldforce[1 + d]

                # Apply delta to qacc
                var actual_n = (
                    rebind[Scalar[DTYPE]](solver[env, ws_lambda_n + c])
                    - oldforce[0]
                )
                if actual_n != Scalar[DTYPE](0):
                    for i in range(NV):
                        qacc_constrained[env, i] += (
                            solver[env, ws_MinvJn + c * NV + i] * actual_n
                        )
                for d in range(num_fric):
                    var actual_f = (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                        - oldforce[1 + d]
                    )
                    if actual_f != Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc_constrained[env, i] += (
                                solver[env, ws_mj + d * MC * NV + c * NV + i]
                                * actual_f
                            )
                _ = lambda_n

    # Store impulses back to contact records for warm-starting
    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
        # Pyramidal: force_n includes edge contributions
        for c in range(nc):
            var c_off = c * CONTACT_SIZE
            var condim = Int(solver[env, ws_cd + c])
            var num_fric = 2
            if condim >= 4:
                num_fric = 3
            if condim >= 6:
                num_fric = 5
            var total_n = rebind[Scalar[DTYPE]](
                solver[env, ws_lambda_n + c]
            )
            for d in range(num_fric):
                total_n += rebind[Scalar[DTYPE]](
                    solver[env, ws_lf + d * MC + c]
                )
                total_n += rebind[Scalar[DTYPE]](
                    solver[env, ws_le_neg + d * MC + c]
                )
            contacts[env, c_off + CONTACT_IDX_FORCE_N] = total_n
            var mu_0 = rebind[Scalar[DTYPE]](
                solver[env, ws_fc + 0 * MC + c]
            )
            contacts[env, c_off + CONTACT_IDX_FORCE_T1] = mu_0 * (
                rebind[Scalar[DTYPE]](solver[env, ws_lf + 0 * MC + c])
                - rebind[Scalar[DTYPE]](solver[env, ws_le_neg + 0 * MC + c])
            )
            var mu_1 = rebind[Scalar[DTYPE]](
                solver[env, ws_fc + 1 * MC + c]
            )
            contacts[env, c_off + CONTACT_IDX_FORCE_T2] = mu_1 * (
                rebind[Scalar[DTYPE]](solver[env, ws_lf + 1 * MC + c])
                - rebind[Scalar[DTYPE]](solver[env, ws_le_neg + 1 * MC + c])
            )
            if condim >= 4:
                var mu_2 = rebind[Scalar[DTYPE]](
                    solver[env, ws_fc + 2 * MC + c]
                )
                contacts[env, c_off + CONTACT_IDX_FORCE_TORSION] = mu_2 * (
                    rebind[Scalar[DTYPE]](solver[env, ws_lf + 2 * MC + c])
                    - rebind[Scalar[DTYPE]](
                        solver[env, ws_le_neg + 2 * MC + c]
                    )
                )
            if condim >= 6:
                var mu_3 = rebind[Scalar[DTYPE]](
                    solver[env, ws_fc + 3 * MC + c]
                )
                contacts[env, c_off + CONTACT_IDX_FORCE_ROLL1] = mu_3 * (
                    rebind[Scalar[DTYPE]](solver[env, ws_lf + 3 * MC + c])
                    - rebind[Scalar[DTYPE]](
                        solver[env, ws_le_neg + 3 * MC + c]
                    )
                )
                var mu_4 = rebind[Scalar[DTYPE]](
                    solver[env, ws_fc + 4 * MC + c]
                )
                contacts[env, c_off + CONTACT_IDX_FORCE_ROLL2] = mu_4 * (
                    rebind[Scalar[DTYPE]](solver[env, ws_lf + 4 * MC + c])
                    - rebind[Scalar[DTYPE]](
                        solver[env, ws_le_neg + 4 * MC + c]
                    )
                )
    else:
        # Elliptic: direct force writeback
        for c in range(nc):
            var c_off = c * CONTACT_SIZE
            contacts[env, c_off + CONTACT_IDX_FORCE_N] = solver[
                env, ws_lambda_n + c
            ]
            contacts[env, c_off + CONTACT_IDX_FORCE_T1] = solver[
                env, ws_lf + 0 * MC + c
            ]
            contacts[env, c_off + CONTACT_IDX_FORCE_T2] = solver[
                env, ws_lf + 1 * MC + c
            ]
            var condim = Int(solver[env, ws_cd + c])
            if condim >= 4:
                contacts[env, c_off + CONTACT_IDX_FORCE_TORSION] = solver[
                    env, ws_lf + 2 * MC + c
                ]
            if condim >= 6:
                contacts[env, c_off + CONTACT_IDX_FORCE_ROLL1] = solver[
                    env, ws_lf + 3 * MC + c
                ]
                contacts[env, c_off + CONTACT_IDX_FORCE_ROLL2] = solver[
                    env, ws_lf + 4 * MC + c
                ]


def _contact_solve_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
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
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
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
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _contact_solve_env[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        CONE_TYPE,
        BATCH,
        SOLVER_WS,
    ](
        env, qpos, qvel, xpos, xquat, subtree_com, contacts, smeta, joints,
        bodies, mmeta, equality, tendons, sites, body_invweight0,
        dof_invweight0, cdof, m_inv, qacc_constrained, solver,
    )


def solve_contacts[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
    NPAIR: Int = 0,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: Model[
        DTYPE,
        NV,
        NBODY,
        NJOINT,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        NEXCLUDE,
        NMESH_VERTS,
        NPAIR,
    ],
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    mut cscratch: ContactScratch[DTYPE, Dims[nv=NV, max_contacts=MAX_CONTACTS], BATCH, _],
    ctx: Optional[DeviceContext] = None,
) raises:
    """PGS contact solve into `scratch.qacc_constrained` (+ solved forces
    back into `d.contacts` for warm-starting), both targets, one body.
    Joint limits, equality constraints, and fixed tendons run INSIDE at the
    legacy position (between the normal and friction phases)."""
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime SOLVER_WS = 81 * MC + 12 * MC * NV

    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, NBODY * 4)
    comptime L_CON = Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_EQ = Layout.row_major(NEQUALITY, MODEL_EQ_SIZE)
    comptime L_TEN = Layout.row_major(NTENDON, MODEL_TENDON_SIZE)
    comptime L_SITE = Layout.row_major(NSITE, MODEL_SITE_SIZE)
    comptime L_BW = Layout.row_major(NBODY, 2)
    comptime L_CDOF = Layout.row_major(BATCH, NV * 6)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_SOLVER = Layout.row_major(BATCH, SOLVER_WS)

    comptime L_QPOS = Layout.row_major(BATCH, NQ)
    comptime L_DW = Layout.row_major(NV)

    comptime if target == "cpu":
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var xpos_v = d.xpos.lt["cpu", L_B3]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var stcom_v = d.subtree_com.lt["cpu", L_B3]()
        var con_v = d.contacts.lt["cpu", L_CON]()
        var smeta_v = d.meta.lt["cpu", L_SMETA]()
        var joints_v = m.joints.lt["cpu", L_JOINT]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var mmeta_v = m.meta.lt["cpu", L_MMETA]()
        var eq_v = m.equality.lt["cpu", L_EQ]()
        var ten_v = m.tendons.lt["cpu", L_TEN]()
        var site_v = m.sites.lt["cpu", L_SITE]()
        var bw_v = m.body_invweight0.lt["cpu", L_BW]()
        var dw_v = m.dof_invweight0.lt["cpu", L_DW]()
        var cdof_v = scratch.cdof.lt["cpu", L_CDOF]()
        var mi_v = scratch.m_inv.lt["cpu", L_M]()
        var qc_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        var sol_v = cscratch.solver.lt["cpu", L_SOLVER]()
        for e in range(BATCH):
            _contact_solve_env[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                NGEOM,
                NEQUALITY,
                NTENDON,
                NSITE,
                CONE_TYPE,
                BATCH,
                SOLVER_WS,
            ](
                e, qpos_v, qvel_v, xpos_v, xquat_v, stcom_v, con_v, smeta_v,
                joints_v, bodies_v, mmeta_v, eq_v, ten_v, site_v, bw_v, dw_v,
                cdof_v, mi_v, qc_v, sol_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + CS_TPB - 1) // CS_TPB
        c.enqueue_function[
            _contact_solve_fields_kernel[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                NGEOM,
                NEQUALITY,
                NTENDON,
                NSITE,
                CONE_TYPE,
                BATCH,
                SOLVER_WS,
            ]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            d.subtree_com.lt["gpu", L_B3](),
            d.contacts.lt["gpu", L_CON](),
            d.meta.lt["gpu", L_SMETA](),
            m.joints.lt["gpu", L_JOINT](),
            m.bodies.lt["gpu", L_BODY](),
            m.meta.lt["gpu", L_MMETA](),
            m.equality.lt["gpu", L_EQ](),
            m.tendons.lt["gpu", L_TEN](),
            m.sites.lt["gpu", L_SITE](),
            m.body_invweight0.lt["gpu", L_BW](),
            m.dof_invweight0.lt["gpu", L_DW](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.m_inv.lt["gpu", L_M](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            cscratch.solver.lt["gpu", L_SOLVER](),
            grid_dim=(BLOCKS,),
            block_dim=(CS_TPB,),
        )
