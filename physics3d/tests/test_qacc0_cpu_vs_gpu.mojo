"""Test Unconstrained Acceleration (qacc0): CPU vs GPU.

Compares qacc0 computed on CPU vs GPU for the HalfCheetah model.
Both use the same formula: qacc = (M + arm + dt*D)^{-1} * f_net
where f_net = qfrc - bias - damping*qvel - stiffness*(qpos-springref).

The GPU pipeline is: FK -> body_vel -> cdof -> crb -> M -> arm+dt*D ->
LDL factor -> bias -> f_net (with passive) -> LDL solve -> qacc.

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_qacc0_cpu_vs_gpu.mojo
"""

from testing import assert_true
from math import abs
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from gpu import block_idx

from physics3d.types import Model, Data, _max_one
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
)
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    integrator_workspace_size,
    ws_M_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    model_metadata_offset,
    model_joint_offset,
    MODEL_META_IDX_TIMESTEP,
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
    copy_data_to_buffer,
)
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime WS_SIZE = integrator_workspace_size[NV, NBODY]()

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()

# Tolerance (float32 precision, accumulated through full pipeline)
comptime ABS_TOL: Float64 = 1e-2
comptime REL_TOL: Float64 = 1e-2


# =============================================================================
# GPU kernel: full pipeline up to qacc (before contacts/solver)
# =============================================================================


fn qacc0_kernel[
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
](
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    var env = Int(block_idx.x)
    if env >= BATCH:
        return

    comptime M_idx = ws_M_offset[NV, NBODY]()
    comptime bias_idx = ws_bias_offset[NV, NBODY]()
    comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
    comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()

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

    # 3. Compute cdof
    compute_cdof_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 4. Composite inertia
    compute_composite_inertia_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 5. Mass matrix (CRBA)
    compute_mass_matrix_full_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 6. Add armature + dt*D to M diagonal
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var dt = model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(model[0, joint_off + JOINT_IDX_TYPE])
        var dof_adr = Int(model[0, joint_off + JOINT_IDX_DOF_ADR])
        var arm = model[0, joint_off + JOINT_IDX_ARMATURE]
        var damp = model[0, joint_off + JOINT_IDX_DAMPING]
        var diag_add = arm + dt * damp
        if jnt_type == JNT_FREE:
            for d in range(6):
                var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                workspace[env, idx] += diag_add
        elif jnt_type == JNT_BALL:
            for d in range(3):
                var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                workspace[env, idx] += diag_add
        else:
            var idx = M_idx + dof_adr * NV + dof_adr
            workspace[env, idx] += diag_add

    # 7. LDL factorize
    ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)

    # 8. Bias forces
    compute_bias_forces_rne_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 9. f_net = qfrc - bias
    var qvel_off = qvel_offset[NQ, NV]()
    var qfrc_off = qfrc_offset[NQ, NV]()
    for i in range(NV):
        var qfrc_val = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
        var bias_val = rebind[Scalar[DTYPE]](workspace[env, bias_idx + i])
        workspace[env, fnet_idx + i] = qfrc_val - bias_val

    # 9b. Passive forces: damping + stiffness + frictionloss
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

    var qpos_off = qpos_offset[NQ, NV]()
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
        var floss = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_FRICTIONLOSS]
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
        if floss > Scalar[DTYPE](0):
            comptime VEL_THRESH: Scalar[DTYPE] = 1e-4
            if jnt_type == JNT_FREE:
                for d in range(6):
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr + d]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr + d]
                    )
                    if v > VEL_THRESH:
                        workspace[env, fnet_idx + dof_adr + d] = cur - floss
                    elif v < -VEL_THRESH:
                        workspace[env, fnet_idx + dof_adr + d] = cur + floss
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr + d]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr + d]
                    )
                    if v > VEL_THRESH:
                        workspace[env, fnet_idx + dof_adr + d] = cur - floss
                    elif v < -VEL_THRESH:
                        workspace[env, fnet_idx + dof_adr + d] = cur + floss
            else:
                var v = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + dof_adr]
                )
                var cur = rebind[Scalar[DTYPE]](
                    workspace[env, fnet_idx + dof_adr]
                )
                if v > VEL_THRESH:
                    workspace[env, fnet_idx + dof_adr] = cur - floss
                elif v < -VEL_THRESH:
                    workspace[env, fnet_idx + dof_adr] = cur + floss

    # 10. LDL solve → qacc
    ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
        env, workspace
    )


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_qacc0(
    ctx: DeviceContext,
    test_name: String,
    test_qpos: InlineArray[Float64, NQ],
    test_qvel: InlineArray[Float64, NV],
    test_actions: InlineArray[Float64, ACTION_DIM],
    model_cpu: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE],
    model_buf: DeviceBuffer[DTYPE],
    mut state_host: HostBuffer[DTYPE],
    mut state_buf: DeviceBuffer[DTYPE],
    mut workspace_buf: DeviceBuffer[DTYPE],
    mut ws_host: HostBuffer[DTYPE],
) raises:
    print("--- Test:", test_name, "---")

    # === CPU pipeline (float32) ===
    var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](test_qpos[i])
    for i in range(NV):
        data_cpu.qvel[i] = Scalar[DTYPE](test_qvel[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(test_actions[i])
    HalfCheetahModel.apply_actions[DTYPE](data_cpu, action_list)

    forward_kinematics(model_cpu, data_cpu)
    compute_body_velocities(model_cpu, data_cpu)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model_cpu, data_cpu, cdof)

    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model_cpu, data_cpu, crb)

    var M = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(model_cpu, data_cpu, cdof, crb, M)

    var dt_cpu = model_cpu.timestep
    for j in range(model_cpu.num_joints):
        var joint = model_cpu.joints[j]
        var dof_adr = joint.dof_adr
        var diag_add = joint.armature + dt_cpu * joint.damping
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                M[(dof_adr + d) * NV + (dof_adr + d)] += diag_add
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                M[(dof_adr + d) * NV + (dof_adr + d)] += diag_add
        else:
            M[dof_adr * NV + dof_adr] += diag_add

    var L = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[DTYPE](0))
    var D_ldl = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D_ldl.append(Scalar[DTYPE](0))
    ldl_factor[DTYPE, NV](M, L, D_ldl)

    var bias_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        bias_cpu.append(Scalar[DTYPE](0))
    compute_bias_forces_rne(model_cpu, data_cpu, cdof, bias_cpu)

    var f_net = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(NV):
        f_net.append(data_cpu.qfrc[i] - bias_cpu[i])

    for j in range(model_cpu.num_joints):
        var joint = model_cpu.joints[j]
        var dof_adr = joint.dof_adr
        if joint.damping > Scalar[DTYPE](0):
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr + d] -= joint.damping * data_cpu.qvel[dof_adr + d]
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr + d] -= joint.damping * data_cpu.qvel[dof_adr + d]
            else:
                f_net[dof_adr] -= joint.damping * data_cpu.qvel[dof_adr]

    for j in range(model_cpu.num_joints):
        var joint = model_cpu.joints[j]
        var dof_adr = joint.dof_adr
        var qpos_adr = joint.qpos_adr
        if joint.stiffness > Scalar[DTYPE](0):
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr + d] -= joint.stiffness * (
                        data_cpu.qpos[qpos_adr + d] - joint.springref
                    )
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr + d] -= joint.stiffness * (
                        data_cpu.qpos[qpos_adr + d] - joint.springref
                    )
            else:
                f_net[dof_adr] -= joint.stiffness * (
                    data_cpu.qpos[qpos_adr] - joint.springref
                )
        if joint.frictionloss > Scalar[DTYPE](0):
            comptime VEL_THRESH: Scalar[DTYPE] = 1e-4
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    var v = data_cpu.qvel[dof_adr + d]
                    if v > VEL_THRESH:
                        f_net[dof_adr + d] -= joint.frictionloss
                    elif v < -VEL_THRESH:
                        f_net[dof_adr + d] += joint.frictionloss
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    var v = data_cpu.qvel[dof_adr + d]
                    if v > VEL_THRESH:
                        f_net[dof_adr + d] -= joint.frictionloss
                    elif v < -VEL_THRESH:
                        f_net[dof_adr + d] += joint.frictionloss
            else:
                var v = data_cpu.qvel[dof_adr]
                if v > VEL_THRESH:
                    f_net[dof_adr] -= joint.frictionloss
                elif v < -VEL_THRESH:
                    f_net[dof_adr] += joint.frictionloss

    var qacc_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        qacc_cpu.append(Scalar[DTYPE](0))
    ldl_solve[DTYPE, NV](L, D_ldl, f_net, qacc_cpu)

    # === GPU pipeline ===
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](test_qpos[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](test_qvel[i])
    for i in range(NV):
        state_host[qfrc_offset[NQ, NV]() + i] = data_cpu.qfrc[i]

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())

    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    comptime kernel_fn = qacc0_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ]

    var state_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model_tensor = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())
    var ws_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ](workspace_buf.unsafe_ptr())

    ctx.enqueue_function[kernel_fn, kernel_fn](
        state_tensor, model_tensor, ws_tensor,
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ctx.synchronize()

    ctx.enqueue_copy(ws_host.unsafe_ptr(), workspace_buf)
    ctx.synchronize()

    # === Compare qacc ===
    comptime qacc_off = ws_qacc_ws_offset[NV, NBODY]()
    var all_pass = True
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for i in range(NV):
        var cpu_val = Float64(qacc_cpu[i])
        var gpu_val = Float64(ws_host[qacc_off + i])
        var abs_err = abs(cpu_val - gpu_val)
        var ref_mag = abs(cpu_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        var ok = abs_err < ABS_TOL or rel_err < REL_TOL
        if not ok:
            print(
                "  FAIL qacc[", i, "]",
                " cpu=", cpu_val,
                " gpu=", gpu_val,
                " abs_err=", abs_err,
                " rel_err=", rel_err,
            )
            fail_count += 1
            all_pass = False

    if all_pass:
        print(
            "  ALL OK  max_abs_err=", max_abs_err,
            " max_rel_err=", max_rel_err,
        )
    else:
        print(
            "  FAILED", fail_count, "elements  max_abs_err=", max_abs_err,
            " max_rel_err=", max_rel_err,
        )

    print("  CPU qacc:", end="")
    for i in range(NV):
        print(" ", Float64(qacc_cpu[i]), end="")
    print()
    print("  GPU qacc:", end="")
    for i in range(NV):
        print(" ", Float64(ws_host[qacc_off + i]), end="")
    print()

    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


fn test_gravity_only() raises:
    print("=" * 60)
    print("Unconstrained Acceleration (qacc0): CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NV=9)")
    print("Precision: float32")
    print("Tolerances: abs=", ABS_TOL, " rel=", REL_TOL)
    print()

    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_qacc0(ctx, "Gravity only (default pose)", qpos, qvel, actions, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_with_actions() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0
    actions[1] = -0.5
    actions[2] = 0.3
    actions[3] = 1.0
    actions[4] = -0.5
    actions[5] = 0.3
    compare_qacc0(ctx, "With actions", qpos, qvel, actions, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_nonzero_vel_coriolis_damping() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[2] = 0.1
    qpos[3] = -0.3
    qpos[6] = 0.4
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0
    qvel[2] = 0.5
    qvel[3] = -1.0
    qvel[4] = 0.8
    qvel[6] = 1.2
    qvel[7] = -0.6
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_qacc0(ctx, "Nonzero vel (Coriolis + damping)", qpos, qvel, actions, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_full_combo() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[2] = 0.1
    qpos[3] = -0.3
    qpos[4] = 0.5
    qpos[6] = 0.4
    qpos[7] = -0.8
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0
    qvel[1] = -0.5
    qvel[2] = 0.5
    qvel[3] = -1.0
    qvel[4] = 0.8
    qvel[5] = -0.3
    qvel[6] = 1.2
    qvel[7] = -0.6
    qvel[8] = 0.4
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8
    actions[1] = -0.5
    actions[2] = 0.3
    actions[3] = 0.8
    actions[4] = -0.5
    actions[5] = 0.3
    compare_qacc0(ctx, "Full combo (vel + actions)", qpos, qvel, actions, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_extreme_velocities() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[3] = -0.52
    qpos[6] = -1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 5.0
    qvel[1] = -2.0
    qvel[2] = 3.0
    qvel[3] = -5.0
    qvel[4] = 5.0
    qvel[5] = -3.0
    qvel[6] = 5.0
    qvel[7] = -5.0
    qvel[8] = 3.0
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_qacc0(ctx, "Extreme velocities", qpos, qvel, actions, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn main() raises:
    test_gravity_only()
    test_with_actions()
    test_nonzero_vel_coriolis_damping()
    test_full_combo()
    test_extreme_velocities()
    print("All qacc0 CPU vs GPU tests passed.")
