"""Test Constraint Parameters: CPU vs GPU.

Compares constraint parameters (K_n, inv_K_imp, bias) computed on CPU (float64)
vs GPU (float32) for the HalfCheetah model at configurations with ground contacts.

CPU pipeline:  FK -> body_vel -> cdof -> contacts -> M -> LDL -> M_inv -> build_constraints
GPU pipeline:  FK -> body_vel -> contacts -> cdof -> crb -> M -> arm -> LDL -> M_inv
               -> init_common_normal_workspace -> precompute_contact_normal -> read results

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_constraint_params_cpu_vs_gpu.mojo
"""

from testing import assert_true, TestSuite
from math import abs, sqrt
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from gpu import block_idx

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_cdof_gpu,
    compute_composite_inertia,
    compute_composite_inertia_gpu,
)
from physics3d.dynamics.bias_forces import (
    compute_bias_forces_rne,
    compute_bias_forces_rne_gpu,
)
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    compute_mass_matrix_full_gpu,
    ldl_factor,
    ldl_factor_gpu,
    ldl_solve,
    ldl_solve_workspace_gpu,
    compute_M_inv_from_ldl,
    compute_M_inv_from_ldl_gpu,
)
from physics3d.collision.contact_detection import (
    detect_contacts,
    detect_contacts_gpu,
)
from physics3d.constraints.constraint_builder import build_constraints
from physics3d.constraints.constraint_data import ConstraintData
from physics3d.constraints.constraint_builder_gpu import (
    init_common_normal_workspace_gpu,
    precompute_contact_normal_gpu,
)
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    contacts_offset,
    metadata_offset,
    integrator_workspace_size,
    ws_M_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    model_metadata_offset,
    model_joint_offset,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
)
from physics3d.gpu.buffer_utils import (
    create_state_buffer,
)
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime GPU_DTYPE = DTYPE
comptime CPU_DTYPE = DTYPE
comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM
comptime BATCH = 1

comptime MC = _max_one[MAX_CONTACTS]()
comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()

# Workspace layout: [integrator_temps | M_inv(NV*NV) | solver_workspace]
# ws_solver_offset gives the start of solver workspace.
# We need at least common_normal_size = 13*MC + 2*MC*NV after solver offset.
comptime COMMON_NORMAL_SIZE = 13 * MC + 2 * MC * NV
comptime TOTAL_WS_SIZE = ws_solver_offset[NV, NBODY]() + COMMON_NORMAL_SIZE + MC

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()
comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT

# Result buffer layout per config: [K_n(MC) | pos_bias(MC) | inv_K_imp(MC) | num_contacts(1)]
comptime RESULT_PER_CONFIG = 3 * MC + 1

# Tolerances (float32 through full pipeline)
comptime ABS_TOL: Float64 = 1e-2
comptime REL_TOL: Float64 = 1e-2


# =============================================================================
# GPU kernel: full pipeline up to constraint params
# =============================================================================


fn constraint_params_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    WS_SIZE: Int,
    RESULT_SIZE: Int,
](
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    result: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, RESULT_SIZE), MutAnyOrigin
    ],
):
    var env = Int(block_idx.x)
    if env >= BATCH:
        return

    comptime M_idx = ws_M_offset[NV, NBODY]()
    comptime bias_idx = ws_bias_offset[NV, NBODY]()
    comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
    comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()
    comptime qacc_constrained_idx = ws_qacc_constrained_offset[NV, NBODY]()
    comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
    comptime si = ws_solver_offset[NV, NBODY]()
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime V_SIZE = _max_one[NV]()

    # 1. Forward kinematics
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # 2. Body velocities
    compute_body_velocities_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # 3. Detect contacts
    detect_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, NGEOM,
    ](env, state, model)

    # 4. Compute cdof
    compute_cdof_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 5. Composite inertia
    compute_composite_inertia_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 6. Mass matrix (CRBA)
    compute_mass_matrix_full_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 6b. Add armature only (Euler: M_solver = M + armature)
    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(model[0, joint_off + JOINT_IDX_TYPE])
        var dof_adr = Int(model[0, joint_off + JOINT_IDX_DOF_ADR])
        var arm = model[0, joint_off + JOINT_IDX_ARMATURE]
        if jnt_type == JNT_FREE:
            for d in range(6):
                var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                workspace[env, idx] += arm
        elif jnt_type == JNT_BALL:
            for d in range(3):
                var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                workspace[env, idx] += arm
        else:
            var idx = M_idx + dof_adr * NV + dof_adr
            workspace[env, idx] += arm

    # 7. LDL factorize
    ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)

    # 8. Compute M_inv from LDL
    compute_M_inv_from_ldl_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
        env, workspace
    )

    # 9. Bias forces
    compute_bias_forces_rne_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 10. f_net = qfrc - bias - damping*qvel - stiffness*(qpos-springref)
    var qvel_off = qvel_offset[NQ, NV]()
    var qfrc_off = qfrc_offset[NQ, NV]()
    var qpos_off = qpos_offset[NQ, NV]()
    for i in range(NV):
        var qfrc_val = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
        var bias_val = rebind[Scalar[DTYPE]](workspace[env, bias_idx + i])
        workspace[env, fnet_idx + i] = qfrc_val - bias_val

    # Passive forces: damping + stiffness
    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )
        var damp_val = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_DAMPING]
        )
        if damp_val > Scalar[DTYPE](0):
            if jnt_type == JNT_FREE:
                for d in range(6):
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr + d]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr + d]
                    )
                    workspace[env, fnet_idx + dof_adr + d] = cur - damp_val * v
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr + d]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr + d]
                    )
                    workspace[env, fnet_idx + dof_adr + d] = cur - damp_val * v
            else:
                var v = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + dof_adr]
                )
                var cur = rebind[Scalar[DTYPE]](
                    workspace[env, fnet_idx + dof_adr]
                )
                workspace[env, fnet_idx + dof_adr] = cur - damp_val * v

    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
        )
        var stiff = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_STIFFNESS]
        )
        var sref = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_SPRINGREF]
        )
        if stiff > Scalar[DTYPE](0):
            if jnt_type == JNT_FREE:
                for d in range(6):
                    var qp = rebind[Scalar[DTYPE]](
                        state[env, qpos_off + qpos_adr + d]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr + d]
                    )
                    workspace[env, fnet_idx + dof_adr + d] = cur - stiff * (
                        qp - sref
                    )
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    var qp = rebind[Scalar[DTYPE]](
                        state[env, qpos_off + qpos_adr + d]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr + d]
                    )
                    workspace[env, fnet_idx + dof_adr + d] = cur - stiff * (
                        qp - sref
                    )
            else:
                var qp = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr]
                )
                var cur = rebind[Scalar[DTYPE]](
                    workspace[env, fnet_idx + dof_adr]
                )
                workspace[env, fnet_idx + dof_adr] = cur - stiff * (qp - sref)

    # 11. LDL solve -> qacc0
    ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
        env, workspace
    )

    # 12. Copy qacc0 to qacc_constrained (needed by precompute_contact_normal_gpu)
    for i in range(NV):
        var qacc_val = rebind[Scalar[DTYPE]](
            workspace[env, qacc_ws_idx + i]
        )
        workspace[env, qacc_constrained_idx + i] = qacc_val

    # 13. Read contact count and solref/solimp
    comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()

    var nc = Int(
        rebind[Scalar[DTYPE]](state[env, meta_off + META_IDX_NUM_CONTACTS])
    )
    if nc > MAX_CONTACTS:
        nc = MAX_CONTACTS

    var sr_tc = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_0]
    )
    var sr_dr = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_1]
    )
    var si_dmin = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_0]
    )
    var si_dmax = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_1]
    )
    var si_width = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_2]
    )
    var si_midpoint = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_3]
    )
    var si_power = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_4]
    )
    if si_width < Scalar[DTYPE](1e-6):
        si_width = Scalar[DTYPE](1e-6)
    if si_dmax < Scalar[DTYPE](1e-4):
        si_dmax = Scalar[DTYPE](1e-4)
    var K_spring = Scalar[DTYPE](1.0) / (
        sr_tc * sr_tc * si_dmax * si_dmax
    )
    var B_damp = Scalar[DTYPE](2.0) * sr_dr / (sr_tc * si_dmax)

    # 14. Init common normal workspace for all contact slots
    for c in range(MC):
        init_common_normal_workspace_gpu[
            DTYPE, NV, NBODY, MAX_CONTACTS, WS_SIZE, BATCH,
        ](env, c, workspace)

    # 15. Precompute contact normals
    for c in range(nc):
        precompute_contact_normal_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, V_SIZE, BATCH, WS_SIZE, NGEOM,
        ](
            env, c, nc, state, model, workspace,
            K_spring, B_damp, si_dmin, si_dmax, si_width, si_midpoint, si_power,
        )

    # 16. Read results into result buffer
    # Layout: [K_n(MC) | pos_bias(MC) | inv_K_imp(MC) | num_contacts(1)]
    comptime ws_K_n = si + 1 * MC
    comptime ws_pos_bias = si + 11 * MC
    comptime ws_inv_K_imp = si + 12 * MC

    for c in range(MC):
        result[env, c] = workspace[env, ws_K_n + c]
        result[env, MC + c] = workspace[env, ws_pos_bias + c]
        result[env, 2 * MC + c] = workspace[env, ws_inv_K_imp + c]
    result[env, 3 * MC] = Scalar[DTYPE](nc)


# =============================================================================
# Main
# =============================================================================


fn run_constraint_params_test(
    test_name: String,
    test_qpos: InlineArray[Float64, NQ],
    test_qvel: InlineArray[Float64, NV],
) raises:
    print("--- Test:", test_name, "---")

    var ctx = DeviceContext()

    # === Create CPU model ===
    var model_cpu = Model[
        CPU_DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE
    ]()
    var data_ref_cpu = Data[CPU_DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data(model_cpu, data_ref_cpu)

    # GPU model buffer
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    # GPU buffers
    var state_host = create_state_buffer[
        GPU_DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[GPU_DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
        BATCH * TOTAL_WS_SIZE
    )
    var ws_host = ctx.enqueue_create_host_buffer[GPU_DTYPE](
        BATCH * TOTAL_WS_SIZE
    )
    var result_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
        BATCH * RESULT_PER_CONFIG
    )
    var result_host = ctx.enqueue_create_host_buffer[GPU_DTYPE](
        BATCH * RESULT_PER_CONFIG
    )

    comptime kernel_fn = constraint_params_kernel[
        GPU_DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, TOTAL_WS_SIZE, RESULT_PER_CONFIG,
    ]

    var state_tensor = LayoutTensor[
        GPU_DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model_tensor = LayoutTensor[
        GPU_DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())
    var ws_tensor = LayoutTensor[
        GPU_DTYPE, Layout.row_major(BATCH, TOTAL_WS_SIZE), MutAnyOrigin
    ](workspace_buf.unsafe_ptr())
    var result_tensor = LayoutTensor[
        GPU_DTYPE, Layout.row_major(BATCH, RESULT_PER_CONFIG), MutAnyOrigin
    ](result_buf.unsafe_ptr())

    # === CPU pipeline ===
    var data_cpu = Data[CPU_DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[CPU_DTYPE](test_qpos[i])
    for i in range(NV):
        data_cpu.qvel[i] = Scalar[CPU_DTYPE](test_qvel[i])

    forward_kinematics(model_cpu, data_cpu)
    compute_body_velocities(model_cpu, data_cpu)

    var cdof = List[Scalar[CPU_DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[CPU_DTYPE](0))
    compute_cdof(model_cpu, data_cpu, cdof)

    detect_contacts[CPU_DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model_cpu, data_cpu
    )

    var crb = List[Scalar[CPU_DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[CPU_DTYPE](0))
    compute_composite_inertia(model_cpu, data_cpu, crb)

    var M = List[Scalar[CPU_DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M.append(Scalar[CPU_DTYPE](0))
    compute_mass_matrix_full(model_cpu, data_cpu, cdof, crb, M)

    for j in range(model_cpu.num_joints):
        var joint = model_cpu.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                M[(dof_adr + d) * NV + (dof_adr + d)] += arm
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                M[(dof_adr + d) * NV + (dof_adr + d)] += arm
        else:
            M[dof_adr * NV + dof_adr] += arm

    var L = List[Scalar[CPU_DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[CPU_DTYPE](0))
    var D_ldl = List[Scalar[CPU_DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D_ldl.append(Scalar[CPU_DTYPE](0))
    ldl_factor[CPU_DTYPE, NV](M, L, D_ldl)

    var M_inv = List[Scalar[CPU_DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_inv.append(Scalar[CPU_DTYPE](0))
    compute_M_inv_from_ldl[CPU_DTYPE, NV](L, D_ldl, M_inv)

    var dt_cpu = Scalar[CPU_DTYPE](0.01)
    var constraints = ConstraintData[CPU_DTYPE, MAX_ROWS, NV]()
    build_constraints[CONE_TYPE = HalfCheetahModel.CONE_TYPE](
        model_cpu, data_cpu, cdof, M_inv, dt_cpu, constraints
    )

    var cpu_ncon = data_cpu.num_contacts
    var cpu_nnorm = constraints.num_normals
    print(
        "  CPU: contacts=", cpu_ncon,
        " normal_rows=", cpu_nnorm,
        " friction_rows=", constraints.num_friction,
        " limit_rows=", constraints.num_limits,
    )

    # === GPU pipeline ===
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[GPU_DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[GPU_DTYPE](test_qpos[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[GPU_DTYPE](test_qvel[i])

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())

    for i in range(BATCH * TOTAL_WS_SIZE):
        ws_host[i] = Scalar[GPU_DTYPE](0)
    ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())

    for i in range(BATCH * RESULT_PER_CONFIG):
        result_host[i] = Scalar[GPU_DTYPE](0)
    ctx.enqueue_copy(result_buf, result_host.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[kernel_fn, kernel_fn](
        state_tensor, model_tensor, ws_tensor, result_tensor,
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ctx.synchronize()

    ctx.enqueue_copy(result_host.unsafe_ptr(), result_buf)
    ctx.synchronize()

    # === Compare ===
    var gpu_nc = Int(Float64(result_host[3 * MC]))
    print("  GPU: contacts=", gpu_nc)

    if cpu_ncon == 0 and gpu_nc == 0:
        print("  ALL OK (no contacts)")
        print()
        return

    if cpu_ncon != gpu_nc:
        print("  WARN: contact count mismatch! CPU=", cpu_ncon, " GPU=", gpu_nc)

    var compare_count = cpu_ncon if cpu_ncon < gpu_nc else gpu_nc
    var all_pass = True
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for c in range(compare_count):
        var cpu_K = Float64(constraints.rows[c].K)
        var cpu_bias = Float64(constraints.rows[c].bias)
        var cpu_inv_K_imp = Float64(constraints.rows[c].inv_K_imp)

        var gpu_K = Float64(result_host[c])
        var gpu_bias = Float64(result_host[MC + c])
        var gpu_inv_K_imp = Float64(result_host[2 * MC + c])

        var abs_err_K = abs(cpu_K - gpu_K)
        var ref_K = abs(cpu_K)
        var rel_err_K: Float64 = 0.0
        if ref_K > 1e-10:
            rel_err_K = abs_err_K / ref_K
        if abs_err_K > max_abs_err:
            max_abs_err = abs_err_K
        if rel_err_K > max_rel_err:
            max_rel_err = rel_err_K
        var ok_K = abs_err_K < ABS_TOL or rel_err_K < REL_TOL
        if not ok_K:
            print("  FAIL K_n[", c, "]", " cpu=", cpu_K, " gpu=", gpu_K, " abs=", abs_err_K, " rel=", rel_err_K)
            fail_count += 1
            all_pass = False

        var abs_err_b = abs(cpu_bias - gpu_bias)
        var ref_b = abs(cpu_bias)
        var rel_err_b: Float64 = 0.0
        if ref_b > 1e-10:
            rel_err_b = abs_err_b / ref_b
        if abs_err_b > max_abs_err:
            max_abs_err = abs_err_b
        if rel_err_b > max_rel_err:
            max_rel_err = rel_err_b
        var ok_b = abs_err_b < ABS_TOL or rel_err_b < REL_TOL
        if not ok_b:
            print("  FAIL bias[", c, "]", " cpu=", cpu_bias, " gpu=", gpu_bias, " abs=", abs_err_b, " rel=", rel_err_b)
            fail_count += 1
            all_pass = False

        var abs_err_i = abs(cpu_inv_K_imp - gpu_inv_K_imp)
        var ref_i = abs(cpu_inv_K_imp)
        var rel_err_i: Float64 = 0.0
        if ref_i > 1e-10:
            rel_err_i = abs_err_i / ref_i
        if abs_err_i > max_abs_err:
            max_abs_err = abs_err_i
        if rel_err_i > max_rel_err:
            max_rel_err = rel_err_i
        var ok_i = abs_err_i < ABS_TOL or rel_err_i < REL_TOL
        if not ok_i:
            print("  FAIL inv_K_imp[", c, "]", " cpu=", cpu_inv_K_imp, " gpu=", gpu_inv_K_imp, " abs=", abs_err_i, " rel=", rel_err_i)
            fail_count += 1
            all_pass = False

        print(
            "  Contact", c, ":"
            " K cpu=", cpu_K, " gpu=", gpu_K,
            " | bias cpu=", cpu_bias, " gpu=", gpu_bias,
            " | inv_K_imp cpu=", cpu_inv_K_imp, " gpu=", gpu_inv_K_imp,
        )

    if all_pass:
        print("  ALL OK  max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err)
    else:
        print("  FAILED", fail_count, "checks  max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err)
    print()
    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


fn test_low_static() raises:
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.2  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    run_constraint_params_test("Low static (rootz=-0.2)", qpos, qvel)


fn test_low_moving() raises:
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.2  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0  # rootx velocity
    qvel[2] = -0.5  # rooty velocity
    run_constraint_params_test("Low moving (rootz=-0.2, vel)", qpos, qvel)


fn test_very_low() raises:
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.5  # rootz very low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    run_constraint_params_test("Very low (rootz=-0.5)", qpos, qvel)


fn test_bent_legs() raises:
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.15  # rootz slightly low
    qpos[3] = -0.5  # bthigh
    qpos[4] = 0.8  # bshin
    qpos[6] = 0.5  # fthigh
    qpos[7] = -0.8  # fshin
    var qvel = InlineArray[Float64, NV](fill=0.0)
    run_constraint_params_test("Bent legs (rootz=-0.15, joints)", qpos, qvel)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
