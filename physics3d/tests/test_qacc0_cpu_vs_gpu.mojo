"""Test Unconstrained Acceleration (qacc0): CPU vs GPU.

Compares qacc0 computed on CPU vs GPU for the HalfCheetah model.
Both use the same formula: qacc = (M + arm + dt*D)^{-1} * f_net
where f_net = qfrc - bias - damping*qvel - stiffness*(qpos-springref).

The GPU pipeline is: FK -> body_vel -> cdof -> crb -> M -> arm+dt*D ->
LDL factor -> bias -> f_net (with passive) -> LDL solve -> qacc.

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_qacc0_cpu_vs_gpu.mojo
"""

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
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("Unconstrained Acceleration (qacc0): CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NV=9)")
    print("Precision: float32")
    print("Tolerances: abs=", ABS_TOL, " rel=", REL_TOL)
    print()

    # Initialize GPU
    var ctx = DeviceContext()
    print("GPU device initialized")

    # Create model (CPU + GPU) once
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)

    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    print("Model copied to GPU")

    # Pre-allocate GPU buffers
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    print("GPU buffers allocated")
    print()

    # Compile kernel once
    comptime kernel_fn = qacc0_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ]

    # LayoutTensors for kernel launch
    var state_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model_tensor = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())
    var ws_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ](workspace_buf.unsafe_ptr())

    # =================================================================
    # Test configurations
    # =================================================================

    comptime NUM_TESTS = 5
    var test_names = InlineArray[String, NUM_TESTS](uninitialized=True)
    test_names[0] = "Gravity only (default pose)"
    test_names[1] = "With actions"
    test_names[2] = "Nonzero vel (Coriolis + damping)"
    test_names[3] = "Full combo (vel + actions)"
    test_names[4] = "Extreme velocities"

    var test_qpos = InlineArray[InlineArray[Float64, NQ], NUM_TESTS](
        uninitialized=True
    )
    var test_qvel = InlineArray[InlineArray[Float64, NV], NUM_TESTS](
        uninitialized=True
    )
    var test_actions = InlineArray[InlineArray[Float64, ACTION_DIM], NUM_TESTS](
        uninitialized=True
    )

    # Config 0: Gravity only
    test_qpos[0] = InlineArray[Float64, NQ](fill=0.0)
    test_qpos[0][1] = 0.7
    test_qvel[0] = InlineArray[Float64, NV](fill=0.0)
    test_actions[0] = InlineArray[Float64, ACTION_DIM](fill=0.0)

    # Config 1: With actions
    test_qpos[1] = InlineArray[Float64, NQ](fill=0.0)
    test_qpos[1][1] = 0.7
    test_qvel[1] = InlineArray[Float64, NV](fill=0.0)
    test_actions[1] = InlineArray[Float64, ACTION_DIM](fill=0.0)
    test_actions[1][0] = 1.0
    test_actions[1][1] = -0.5
    test_actions[1][2] = 0.3
    test_actions[1][3] = 1.0
    test_actions[1][4] = -0.5
    test_actions[1][5] = 0.3

    # Config 2: Nonzero vel
    test_qpos[2] = InlineArray[Float64, NQ](fill=0.0)
    test_qpos[2][1] = 0.7
    test_qpos[2][2] = 0.1
    test_qpos[2][3] = -0.3
    test_qpos[2][6] = 0.4
    test_qvel[2] = InlineArray[Float64, NV](fill=0.0)
    test_qvel[2][0] = 2.0
    test_qvel[2][2] = 0.5
    test_qvel[2][3] = -1.0
    test_qvel[2][4] = 0.8
    test_qvel[2][6] = 1.2
    test_qvel[2][7] = -0.6
    test_actions[2] = InlineArray[Float64, ACTION_DIM](fill=0.0)

    # Config 3: Full combo
    test_qpos[3] = InlineArray[Float64, NQ](fill=0.0)
    test_qpos[3][1] = 0.7
    test_qpos[3][2] = 0.1
    test_qpos[3][3] = -0.3
    test_qpos[3][4] = 0.5
    test_qpos[3][6] = 0.4
    test_qpos[3][7] = -0.8
    test_qvel[3] = InlineArray[Float64, NV](fill=0.0)
    test_qvel[3][0] = 2.0
    test_qvel[3][1] = -0.5
    test_qvel[3][2] = 0.5
    test_qvel[3][3] = -1.0
    test_qvel[3][4] = 0.8
    test_qvel[3][5] = -0.3
    test_qvel[3][6] = 1.2
    test_qvel[3][7] = -0.6
    test_qvel[3][8] = 0.4
    test_actions[3] = InlineArray[Float64, ACTION_DIM](fill=0.0)
    test_actions[3][0] = 0.8
    test_actions[3][1] = -0.5
    test_actions[3][2] = 0.3
    test_actions[3][3] = 0.8
    test_actions[3][4] = -0.5
    test_actions[3][5] = 0.3

    # Config 4: Extreme velocities
    test_qpos[4] = InlineArray[Float64, NQ](fill=0.0)
    test_qpos[4][1] = 0.7
    test_qpos[4][3] = -0.52
    test_qpos[4][6] = -1.0
    test_qvel[4] = InlineArray[Float64, NV](fill=0.0)
    test_qvel[4][0] = 5.0
    test_qvel[4][1] = -2.0
    test_qvel[4][2] = 3.0
    test_qvel[4][3] = -5.0
    test_qvel[4][4] = 5.0
    test_qvel[4][5] = -3.0
    test_qvel[4][6] = 5.0
    test_qvel[4][7] = -5.0
    test_qvel[4][8] = 3.0
    test_actions[4] = InlineArray[Float64, ACTION_DIM](fill=0.0)

    # =================================================================
    # Run all tests
    # =================================================================

    var num_pass = 0
    var num_fail = 0

    for t in range(NUM_TESTS):
        print("--- Test:", test_names[t], "---")

        # === CPU pipeline (float32) ===
        var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
        for i in range(NQ):
            data_cpu.qpos[i] = Scalar[DTYPE](test_qpos[t][i])
        for i in range(NV):
            data_cpu.qvel[i] = Scalar[DTYPE](test_qvel[t][i])

        # Apply actuator forces
        var action_list = List[Float64]()
        for i in range(ACTION_DIM):
            action_list.append(test_actions[t][i])
        HalfCheetahModel.apply_actions[DTYPE](data_cpu, action_list)

        # FK + body velocities
        forward_kinematics(model_cpu, data_cpu)
        compute_body_velocities(model_cpu, data_cpu)

        # cdof
        var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
        compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
            model_cpu, data_cpu, cdof
        )

        # Composite inertia + mass matrix
        var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
        for i in range(CRB_SIZE):
            crb[i] = Scalar[DTYPE](0)
        compute_composite_inertia(model_cpu, data_cpu, crb)

        var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M[i] = Scalar[DTYPE](0)
        compute_mass_matrix_full(model_cpu, data_cpu, cdof, crb, M)

        # Add armature + dt*D
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

        # LDL factorize
        var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var D_ldl = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D_ldl)

        # Bias forces
        var bias_cpu = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            bias_cpu[i] = Scalar[DTYPE](0)
        compute_bias_forces_rne[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
        ](model_cpu, data_cpu, cdof, bias_cpu)

        # f_net = qfrc - bias
        var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            f_net[i] = data_cpu.qfrc[i] - bias_cpu[i]

        # Passive forces
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

        # LDL solve → qacc
        var qacc_cpu = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qacc_cpu[i] = Scalar[DTYPE](0)
        ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D_ldl, f_net, qacc_cpu)

        # === GPU pipeline ===
        # Set state: zero, then qpos/qvel/qfrc
        for i in range(BATCH * STATE_SIZE):
            state_host[i] = Scalar[DTYPE](0)
        for i in range(NQ):
            state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](
                test_qpos[t][i]
            )
        for i in range(NV):
            state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](
                test_qvel[t][i]
            )
        # Copy actuator forces to state qfrc
        for i in range(NV):
            state_host[qfrc_offset[NQ, NV]() + i] = data_cpu.qfrc[i]

        ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())

        # Zero workspace
        for i in range(BATCH * WS_SIZE):
            ws_host[i] = Scalar[DTYPE](0)
        ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
        ctx.synchronize()

        # Launch kernel
        ctx.enqueue_function[kernel_fn, kernel_fn](
            state_tensor, model_tensor, ws_tensor,
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.synchronize()

        # Copy workspace back
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
            num_pass += 1
        else:
            print(
                "  FAILED", fail_count, "elements  max_abs_err=", max_abs_err,
                " max_rel_err=", max_rel_err,
            )
            num_fail += 1

        # Print values
        print("  CPU qacc:", end="")
        for i in range(NV):
            print(" ", Float64(qacc_cpu[i]), end="")
        print()
        print("  GPU qacc:", end="")
        for i in range(NV):
            print(" ", Float64(ws_host[qacc_off + i]), end="")
        print()
        print()

    print("=" * 60)
    print(
        "Results:",
        num_pass,
        "passed,",
        num_fail,
        "failed out of",
        num_pass + num_fail,
    )
    if num_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
