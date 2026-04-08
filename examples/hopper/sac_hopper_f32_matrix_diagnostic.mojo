"""Diagnostic: Compare M, bias, fnet, qacc between CPU f32 and GPU f32
at the exact state where the Euler step shows 0.16 qacc divergence.

Manually runs the forward dynamics pipeline on CPU and dumps
intermediate values. Reads GPU workspace after a step to compare.

Run with:
    pixi run -e apple mojo run -I . examples/hopper/sac_hopper_f32_matrix_diagnostic.mojo
"""

from std.random import seed
from std.math import abs, tanh, sqrt
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.physics3d.types import Model, Data, _max_one
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_composite_inertia,
)
from mojo_rl.physics3d.dynamics.bias_forces import compute_bias_forces_rne
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    compute_M_inv_from_ldl,
)
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.joint_types import JNT_FREE, JNT_BALL
from mojo_rl.physics3d.gpu.constants import (
    state_size, model_size_with_invweight,
    integrator_workspace_size, rk4_extra_workspace_size,
    qpos_offset, qvel_offset, qfrc_offset, qacc_offset,
    ws_M_offset, ws_bias_offset, ws_fnet_offset,
    ws_qacc_constrained_offset, ws_m_inv_offset,
)
from mojo_rl.physics3d.gpu.buffer_utils import create_state_buffer

from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.deep_agents.core.configs.offpolicy_config import SACConfig
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.nn.training import Network


comptime OBS_DIM = HopperConfig.OBS_DIM
comptime ACTION_DIM = HopperConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4
comptime NQ = HopperModel.NQ
comptime NV = HopperModel.NV
comptime NBODY = HopperModel.NBODY
comptime NJOINT = HopperModel.NJOINT
comptime NGEOM = HopperModel.NGEOM
comptime NSITE = HopperModel.NSITE
comptime FRAME_SKIP = HopperConfig.FRAME_SKIP
comptime MAX_CONTACTS = HopperConfig.MAX_CONTACTS

comptime DTYPE = DType.float32
comptime GPU_BATCH = 1
comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime EULER_WS = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + SOLVER_WS

comptime TARGET_SUBSTEP = 172  # State just before divergence

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM].ActorOpt
comptime AgentType = DeepSACAgent[
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE,
    0.0003, 0.0003, 0, NoOpLogger, MAX_N_ENVS,
]


def _get_greedy_action(
    agent: AgentType, obs: List[Float64],
) -> List[Float64]:
    var obs_arr = obs_to_inline[OBS_DIM, DType.float64](obs)
    var obs_f32 = InlineArray[Scalar[nn_dtype], OBS_DIM](uninitialized=True)
    for i in range(OBS_DIM):
        obs_f32[i] = Scalar[nn_dtype](obs_arr[i])
    var obs_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, OBS_DIM), MutAnyOrigin
    ](obs_f32.unsafe_ptr())
    comptime ACTOR_OUT = ActorModel.OUT_DIM
    var out_arr = InlineArray[Scalar[nn_dtype], ACTOR_OUT](uninitialized=True)
    var out_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, ACTOR_OUT), MutAnyOrigin
    ](out_arr.unsafe_ptr())
    comptime PS = ActorModel.PARAM_SIZE
    var p = LayoutTensor[nn_dtype, Layout.row_major(PS), MutAnyOrigin](
        agent.state.actor.online.params
    )
    Network[ActorModel, ActorOpt].forward[1](obs_t, out_t, p)
    var result = List[Float64](capacity=ACTION_DIM)
    for i in range(ACTION_DIM):
        result.append(tanh(Float64(out_arr[i])) * agent.action_scale)
    return result^


def main() raises:
    seed(42)
    print("=" * 70)
    print("f32 Matrix Diagnostic at divergence point")
    print("=" * 70)

    var agent = AgentType(
        gamma=0.99, tau=0.005, action_scale=1.0,
        alpha=0.2, auto_alpha=False, target_entropy=-3.0,
    )
    agent.load_checkpoint("sac_hopper_1000.ckpt")

    # Run CPU to TARGET_SUBSTEP
    var cpu_model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        HopperModel.MAX_EQUALITY, HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON, HopperModel.NSITE,
    ]()
    var cpu_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    HopperModel.setup_model_and_data(cpu_model, cpu_data)

    var cpu_env = Hopper[DTYPE, TERMINATE_ON_UNHEALTHY=True]()
    _ = cpu_env.reset()
    for i in range(NQ):
        cpu_data.qpos[i] = cpu_env.get_qpos(i)
    for i in range(NV):
        cpu_data.qvel[i] = cpu_env.get_qvel(i)
    forward_kinematics(cpu_model, cpu_data)

    var substep = 0
    for _ in range(TARGET_SUBSTEP // FRAME_SKIP + 1):
        var obs = List[Float64](capacity=OBS_DIM)
        for k in range(1, 6):
            obs.append(Float64(cpu_data.qpos[k]))
        for k in range(6):
            var v = Float64(cpu_data.qvel[k])
            if v > 10.0:
                v = 10.0
            elif v < -10.0:
                v = -10.0
            obs.append(v)
        var action = _get_greedy_action(agent, obs)

        for i in range(NV):
            cpu_data.qfrc[i] = Scalar[DTYPE](0)
        for i in range(ACTION_DIM):
            var ctrl = action[i]
            if ctrl > HopperModel._acd.motor_ctrl_max[i]:
                ctrl = HopperModel._acd.motor_ctrl_max[i]
            elif ctrl < HopperModel._acd.motor_ctrl_min[i]:
                ctrl = HopperModel._acd.motor_ctrl_min[i]
            cpu_data.qfrc[HopperModel._acd.motor_dof_adr[i]] = Scalar[DTYPE](
                HopperModel._acd.motor_gears[i] * ctrl
            )

        for _ in range(FRAME_SKIP):
            if substep >= TARGET_SUBSTEP:
                break
            RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
                cpu_model, cpu_data
            )
            substep += 1
        if substep >= TARGET_SUBSTEP:
            break

    print("State at substep " + String(TARGET_SUBSTEP))
    print("ncon:", Int(cpu_data.num_contacts))

    # === CPU: Manual forward dynamics (identical to Euler step internals) ===
    # Create fresh model+data for clean computation
    var m2 = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        HopperModel.MAX_EQUALITY, HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON, HopperModel.NSITE,
    ]()
    var d2 = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    HopperModel.setup_model_and_data(m2, d2)
    for i in range(NQ):
        d2.qpos[i] = cpu_data.qpos[i]
    for i in range(NV):
        d2.qvel[i] = cpu_data.qvel[i]
        d2.qfrc[i] = cpu_data.qfrc[i]

    # FK + body velocities + cdof
    forward_kinematics(m2, d2)
    compute_body_velocities(m2, d2)
    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(m2, d2, cdof)

    # Mass matrix
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(m2, d2, crb)

    var M_cpu = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_cpu.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(m2, d2, cdof, crb, M_cpu)

    # Add armature
    for j in range(m2.num_joints):
        var joint = m2.joints[j]
        var dof = joint.dof_adr
        var arm = joint.armature
        if joint.jnt_type == JNT_FREE:
            for dd in range(6):
                M_cpu[(dof + dd) * NV + (dof + dd)] += arm
        elif joint.jnt_type == JNT_BALL:
            for dd in range(3):
                M_cpu[(dof + dd) * NV + (dof + dd)] += arm
        else:
            M_cpu[dof * NV + dof] += arm

    # Bias forces
    var bias_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        bias_cpu.append(Scalar[DTYPE](0))
    compute_bias_forces_rne(m2, d2, cdof, bias_cpu)

    # fnet = qfrc - bias + passive
    var fnet_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(NV):
        fnet_cpu.append(d2.qfrc[i] - bias_cpu[i])
    for j in range(m2.num_joints):
        var joint = m2.joints[j]
        var dof = joint.dof_adr
        var damp = joint.damping
        if damp > Scalar[DTYPE](0):
            fnet_cpu[dof] -= damp * d2.qvel[dof]

    # LDL + qacc0
    var L = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[DTYPE](0))
    var D_ldl = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D_ldl.append(Scalar[DTYPE](0))
    ldl_factor[DTYPE, NV](M_cpu, L, D_ldl)

    var qacc0_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        qacc0_cpu.append(Scalar[DTYPE](0))
    ldl_solve[DTYPE, NV](L, D_ldl, fnet_cpu, qacc0_cpu)

    # === GPU: Run Euler step, read back workspace ===
    var ctx = DeviceContext()
    var gpu_state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)
    var gpu_state_buf = ctx.enqueue_create_buffer[DTYPE](
        GPU_BATCH * STATE_SIZE
    )
    var gpu_model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HopperModel.init_model_gpu(ctx, gpu_model_buf)
    var gpu_ws_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * EULER_WS)
    ctx.synchronize()

    # Copy state to GPU
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        gpu_state_host[qpos_offset[NQ, NV]() + i] = cpu_data.qpos[i]
    for i in range(NV):
        gpu_state_host[qvel_offset[NQ, NV]() + i] = cpu_data.qvel[i]
        gpu_state_host[qfrc_offset[NQ, NV]() + i] = cpu_data.qfrc[i]
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
    ctx.synchronize()

    # Run one Euler step
    EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, GPU_BATCH,
        NGEOM=NGEOM, CONE_TYPE=HopperModel.CONE_TYPE,
    ](ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    ctx.synchronize()

    # Read back workspace
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH * EULER_WS)
    ctx.enqueue_copy(ws_host.unsafe_ptr(), gpu_ws_buf)
    ctx.synchronize()

    # === Compare intermediates ===
    print()
    print("=== Mass Matrix Diagonal ===")
    comptime M_off = ws_M_offset[NV, NBODY]()
    for i in range(NV):
        var cpu_m = Float64(M_cpu[i * NV + i])
        var gpu_m = Float64(ws_host[M_off + i * NV + i])
        var err = abs(cpu_m - gpu_m)
        print(
            "  M[" + String(i) + "," + String(i) + "]"
            + " cpu=" + String(cpu_m)
            + " gpu=" + String(gpu_m)
            + " err=" + String(err)
        )

    print()
    print("=== Mass Matrix Max Error (all elements) ===")
    var m_max_err: Float64 = 0.0
    var m_max_i = 0
    var m_max_j = 0
    for i in range(NV):
        for j in range(NV):
            var cpu_m = Float64(M_cpu[i * NV + j])
            var gpu_m = Float64(ws_host[M_off + i * NV + j])
            var err = abs(cpu_m - gpu_m)
            if err > m_max_err:
                m_max_err = err
                m_max_i = i
                m_max_j = j
    print(
        "  max err=" + String(m_max_err)
        + " at M[" + String(m_max_i) + "," + String(m_max_j) + "]"
        + " cpu=" + String(Float64(M_cpu[m_max_i * NV + m_max_j]))
        + " gpu=" + String(Float64(ws_host[M_off + m_max_i * NV + m_max_j]))
    )

    print()
    print("=== Bias Forces ===")
    comptime bias_off = ws_bias_offset[NV, NBODY]()
    for i in range(NV):
        var cpu_b = Float64(bias_cpu[i])
        var gpu_b = Float64(ws_host[bias_off + i])
        var err = abs(cpu_b - gpu_b)
        if err > 1e-6:
            print(
                "  bias[" + String(i) + "]"
                + " cpu=" + String(cpu_b)
                + " gpu=" + String(gpu_b)
                + " err=" + String(err)
            )

    print()
    print("=== fnet (qfrc - bias + passive) ===")
    comptime fnet_off = ws_fnet_offset[NV, NBODY]()
    for i in range(NV):
        var cpu_f = Float64(fnet_cpu[i])
        var gpu_f = Float64(ws_host[fnet_off + i])
        var err = abs(cpu_f - gpu_f)
        if err > 1e-6:
            print(
                "  fnet[" + String(i) + "]"
                + " cpu=" + String(cpu_f)
                + " gpu=" + String(gpu_f)
                + " err=" + String(err)
            )

    print()
    print("=== Unconstrained qacc (M^{-1} * fnet) ===")
    comptime qacc_ws_off = ws_qacc_constrained_offset[NV, NBODY]()
    for i in range(NV):
        var cpu_a = Float64(qacc0_cpu[i])
        # GPU qacc_constrained after solver includes constraint forces,
        # so read qacc from state buffer (final integrated value).
        # For unconstrained comparison, we'd need a pre-solver workspace read.
        # Instead, compare the final qacc (post-solver).
        print(
            "  qacc0[" + String(i) + "] cpu=" + String(cpu_a)
        )

    # Read GPU final qacc from state buffer
    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
    ctx.synchronize()
    print()
    print("=== Final qacc (after solver + integration) ===")
    for i in range(NV):
        var cpu_a = Float64(d2.qacc[i])  # d2 wasn't stepped — use Euler on d2
        var gpu_a = Float64(gpu_state_host[qacc_offset[NQ, NV]() + i])
        var err = abs(cpu_a - gpu_a)
        print(
            "  qacc[" + String(i) + "]"
            + " gpu=" + String(gpu_a)
        )

    # Also run CPU Euler for comparison
    EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](m2, d2)
    print()
    print("=== CPU Euler final qacc ===")
    for i in range(NV):
        var cpu_a = Float64(d2.qacc[i])
        var gpu_a = Float64(gpu_state_host[qacc_offset[NQ, NV]() + i])
        var err = abs(cpu_a - gpu_a)
        print(
            "  qacc[" + String(i) + "]"
            + " cpu=" + String(cpu_a)
            + " gpu=" + String(gpu_a)
            + " err=" + String(err)
        )

    print()
    print("=== M_inv comparison ===")
    var M_inv_cpu = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_inv_cpu.append(Scalar[DTYPE](0))
    compute_M_inv_from_ldl[DTYPE, NV](L, D_ldl, M_inv_cpu)

    comptime minv_off = ws_m_inv_offset[NV, NBODY]()
    var minv_max_err: Float64 = 0.0
    var minv_max_i = 0
    var minv_max_j = 0
    for i in range(NV):
        for j in range(NV):
            var cpu_v = Float64(M_inv_cpu[i * NV + j])
            var gpu_v = Float64(ws_host[minv_off + i * NV + j])
            var err = abs(cpu_v - gpu_v)
            if err > minv_max_err:
                minv_max_err = err
                minv_max_i = i
                minv_max_j = j
    print(
        "  max err=" + String(minv_max_err)
        + " at M_inv[" + String(minv_max_i) + "," + String(minv_max_j) + "]"
        + " cpu=" + String(Float64(M_inv_cpu[minv_max_i * NV + minv_max_j]))
        + " gpu=" + String(Float64(ws_host[minv_off + minv_max_i * NV + minv_max_j]))
    )

    cpu_env.close()
