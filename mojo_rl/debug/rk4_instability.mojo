"""Deep diagnostic: per-RK4-stage CPU vs GPU comparison at substep 173.

Runs simulation to the state at substep 173, then launches each RK4 stage
individually (stage kernel + solver) with sync between them. Compares the
constrained qacc (A[stage]) after each stage to find which stage amplifies
the error.

Phases are split into separate functions to reduce compilation pressure
and make it easy to comment out individual sections.

Run with:
    pixi run -e apple mojo run -I . mojo_rl/debug/rk4_instability.mojo
"""

from std.random import seed
from std.math import abs, tanh, sqrt
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.physics3d.types import Model, Data, _max_one, ConeType
from mojo_rl.physics3d.kinematics import forward_kinematics
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.integrator.rk4_integrator import (
    RK4Integrator,
    _forward_dynamics,
    _solve_constraints,
    _integrate_pos,
)
from mojo_rl.physics3d.gpu.constants import (
    TPB,
    state_size,
    model_size_with_invweight,
    integrator_workspace_size,
    rk4_extra_workspace_size,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    qacc_offset,
    contacts_offset,
    metadata_offset,
    ws_qacc_constrained_offset,
    ws_fnet_offset,
    ws_M_offset,
    ws_rk4_A_offset,
    ws_solver_offset,
    CONTACT_SIZE,
    CONTACT_IDX_DIST,
    META_IDX_NUM_CONTACTS,
)
from mojo_rl.physics3d.gpu.buffer_utils import create_state_buffer
from mojo_rl.physics3d.constraints.constraint_data import ConstraintData
from mojo_rl.physics3d.constraints.constraint_builder import build_constraints
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_M_inv_from_ldl,
    ldl_factor as ldl_fac,
)
from mojo_rl.physics3d.solver.primal_common import primal_D
from mojo_rl.physics3d.solver.cholesky import (
    chol_factor as chol_fac,
    chol_solve as chol_slv,
)

from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.deep_agents.core.configs.offpolicy_config import SACConfig
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.nn.training import Network


# =============================================================================
# Constants
# =============================================================================

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
comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime DBG_SIZE = 38  # grad(6)+search(6)+H_diag(6)+qacc(6)+L_diag(6)+nc+alpha
comptime WS_SIZE = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + SOLVER_WS + rk4_extra_workspace_size[NQ, NV]() + DBG_SIZE

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()

comptime TARGET_SUBSTEP = 173

# Grid configuration
comptime ENV_BLOCKS = (GPU_BATCH + TPB - 1) // TPB
comptime THREADS = NewtonSolver.solver_threads[
    NQ, NV, NBODY, NJOINT, MAX_CONTACTS
]()
comptime SOLVER_ENV_TPB = TPB // THREADS
comptime SOLVER_ENV_BLOCKS = (GPU_BATCH + SOLVER_ENV_TPB - 1) // SOLVER_ENV_TPB
comptime SOLVER_THREADS_BLOCKS = (THREADS + THREADS - 1) // THREADS

# Workspace offsets
comptime QACC_CON_OFF = ws_qacc_constrained_offset[NV, NBODY]()
comptime FNET_OFF = ws_fnet_offset[NV, NBODY]()
comptime M_OFF = ws_M_offset[NV, NBODY]()
comptime META_OFF = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
comptime QPOS_OFF = qpos_offset[NQ, NV]()
comptime QVEL_OFF = qvel_offset[NQ, NV]()
comptime QACC_OFF = qacc_offset[NQ, NV]()
comptime QFRC_OFF = qfrc_offset[NQ, NV]()
comptime SOL_OFF = ws_solver_offset[NV, NBODY]()
comptime MC = _max_one[MAX_CONTACTS]()
# Debug area is at the very end of the workspace (after RK4 extra)
comptime DBG_WS_OFF = WS_SIZE - DBG_SIZE
comptime PYR_J_BASE = SOL_OFF + 15 * MC + 2 * MC * NV
comptime PYR_SC = PYR_J_BASE + 4 * MC * NV
comptime CONTACTS_OFF = contacts_offset[NQ, NV, NBODY]()
comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT


# =============================================================================
# Type Aliases
# =============================================================================

comptime ModelType = Model[
    DTYPE,
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    NGEOM,
    HopperModel.MAX_EQUALITY,
    HopperModel.CONE_TYPE,
    HopperModel.MAX_TENDON,
    HopperModel.NSITE,
]
comptime DataType = Data[
    DTYPE,
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    HopperModel.NSITE,
]

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM].ActorOpt
comptime AgentType = DeepSACAgent[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    0.0003,
    0.0003,
    0,
    NoOpLogger,
    MAX_N_ENVS,
]


# =============================================================================
# CPU Physics Wrappers (single monomorphization each)
# =============================================================================


def _cpu_fwd(
    model: ModelType,
    mut data: DataType,
    mut a: List[Scalar[DTYPE]],
    mut cdof: List[Scalar[DTYPE]],
    mut M_inv: List[Scalar[DTYPE]],
    mut M: List[Scalar[DTYPE]],
):
    """Compute unconstrained acceleration."""
    _forward_dynamics[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HopperModel.MAX_EQUALITY,
        V_SIZE,
        M_SIZE,
        CDOF_SIZE,
        CRB_SIZE,
    ](model, data, a, cdof, M_inv, M)


def _cpu_solve(
    model: ModelType,
    mut data: DataType,
    cdof: List[Scalar[DTYPE]],
    M_inv: List[Scalar[DTYPE]],
    M: List[Scalar[DTYPE]],
    mut a: List[Scalar[DTYPE]],
    dt: Scalar[DTYPE],
    is_last: Bool,
):
    """Build and solve constraints, modifying qacc in place."""
    _solve_constraints[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HopperModel.MAX_EQUALITY,
        V_SIZE,
        M_SIZE,
        CDOF_SIZE,
        HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON,
        NewtonSolver,
    ](model, data, cdof, M_inv, M, a, dt, is_last)


def _cpu_integ_pos(
    model: ModelType,
    q0: List[Scalar[DTYPE]],
    v: List[Scalar[DTYPE]],
    dt: Scalar[DTYPE],
    mut q_out: List[Scalar[DTYPE]],
):
    """Integrate position: q_out = q0 + v * dt."""
    _integrate_pos[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HopperModel.MAX_EQUALITY,
    ](model, q0, v, dt, q_out)


# =============================================================================
# GPU Kernel Launchers
# =============================================================================


def _gpu_stage[
    STAGE: Int
](
    ctx: DeviceContext,
    mut state_buf: DeviceBuffer[DTYPE],
    mut model_buf: DeviceBuffer[DTYPE],
    mut ws_buf: DeviceBuffer[DTYPE],
) raises:
    """Launch RK4 stage kernel for given STAGE."""
    var state_lt = LayoutTensor[
        DTYPE, Layout.row_major(GPU_BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf)
    var model_lt = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf)
    var ws_lt = LayoutTensor[
        DTYPE, Layout.row_major(GPU_BATCH, WS_SIZE), MutAnyOrigin
    ](ws_buf)
    comptime kernel = RK4Integrator[SOLVER=NewtonSolver].rk4_stage_kernel[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        GPU_BATCH,
        WS_SIZE,
        NGEOM,
        SOLVER_WS,
        STAGE,
    ]
    ctx.enqueue_function[kernel, kernel](
        state_lt,
        model_lt,
        ws_lt,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


def _gpu_solver(
    ctx: DeviceContext,
    mut state_buf: DeviceBuffer[DTYPE],
    mut model_buf: DeviceBuffer[DTYPE],
    mut ws_buf: DeviceBuffer[DTYPE],
) raises:
    """Launch Newton constraint solver kernel."""
    var state_lt = LayoutTensor[
        DTYPE, Layout.row_major(GPU_BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf)
    var model_lt = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf)
    var ws_lt = LayoutTensor[
        DTYPE, Layout.row_major(GPU_BATCH, WS_SIZE), MutAnyOrigin
    ](ws_buf)
    comptime kernel = NewtonSolver.solve_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        V_SIZE,
        GPU_BATCH,
        WS_SIZE,
        NGEOM,
        0,
        HopperModel.CONE_TYPE,
        0,
        NSITE,
    ]
    ctx.enqueue_function[kernel, kernel](
        state_lt,
        model_lt,
        ws_lt,
        grid_dim=(SOLVER_ENV_BLOCKS, SOLVER_THREADS_BLOCKS),
        block_dim=(SOLVER_ENV_TPB, THREADS),
    )


def _gpu_combine(
    ctx: DeviceContext,
    mut state_buf: DeviceBuffer[DTYPE],
    mut model_buf: DeviceBuffer[DTYPE],
    mut ws_buf: DeviceBuffer[DTYPE],
) raises:
    """Launch RK4 combine kernel."""
    var state_lt = LayoutTensor[
        DTYPE, Layout.row_major(GPU_BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf)
    var model_lt = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf)
    var ws_lt = LayoutTensor[
        DTYPE, Layout.row_major(GPU_BATCH, WS_SIZE), MutAnyOrigin
    ](ws_buf)
    comptime kernel = RK4Integrator[SOLVER=NewtonSolver].rk4_combine_kernel[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        GPU_BATCH,
        WS_SIZE,
        SOLVER_WS,
    ]
    ctx.enqueue_function[kernel, kernel](
        state_lt,
        model_lt,
        ws_lt,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


# =============================================================================
# Comparison Helpers
# =============================================================================


def _print_compare_qacc(
    label: String,
    cpu_a: List[Scalar[DTYPE]],
    ws_host: HostBuffer[DTYPE],
) -> Float64:
    """Print CPU vs GPU qacc and return max error."""
    var max_err: Float64 = 0
    print("  CPU " + label + ":", end="")
    for i in range(NV):
        print(" " + String(Float64(cpu_a[i]))[byte=:14], end="")
    print()
    print("  GPU qacc:", end="")
    for i in range(NV):
        var gpu_val = Float64(ws_host[QACC_CON_OFF + i])
        print(" " + String(gpu_val)[byte=:14], end="")
        var err = abs(Float64(cpu_a[i]) - gpu_val)
        if err > max_err:
            max_err = err
    print()
    print("  " + label + " max_err = " + String(max_err))
    return max_err


def _print_compare_state(
    cpu_data: DataType,
    gpu_state_host: HostBuffer[DTYPE],
):
    """Print qpos/qvel error and ncon comparison."""
    var qpos_err: Float64 = 0
    var qvel_err: Float64 = 0
    for i in range(NQ):
        var err = abs(
            Float64(cpu_data.qpos[i]) - Float64(gpu_state_host[QPOS_OFF + i])
        )
        if err > qpos_err:
            qpos_err = err
    for i in range(NV):
        var err = abs(
            Float64(cpu_data.qvel[i]) - Float64(gpu_state_host[QVEL_OFF + i])
        )
        if err > qvel_err:
            qvel_err = err
    var ncon_cpu = Int(cpu_data.num_contacts)
    var ncon_gpu = Int(gpu_state_host[META_OFF + META_IDX_NUM_CONTACTS])
    print(
        "  intermediate qpos_err="
        + String(qpos_err)
        + " qvel_err="
        + String(qvel_err)
    )
    print("  ncon: cpu=" + String(ncon_cpu) + " gpu=" + String(ncon_gpu))


def _print_ncon(
    cpu_data: DataType,
    gpu_state_host: HostBuffer[DTYPE],
):
    """Print contact count comparison only."""
    var ncon_cpu = Int(cpu_data.num_contacts)
    var ncon_gpu = Int(gpu_state_host[META_OFF + META_IDX_NUM_CONTACTS])
    print("  ncon: cpu=" + String(ncon_cpu) + " gpu=" + String(ncon_gpu))


# =============================================================================
# Agent Action
# =============================================================================


def _get_greedy_action(agent: AgentType, obs: List[Float64]) -> List[Float64]:
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


# =============================================================================
# Phase: Advance to Target Substep
# =============================================================================


def _advance_to_substep(
    agent: AgentType,
    model: ModelType,
    mut data: DataType,
) raises -> Int:
    """Advance simulation to TARGET_SUBSTEP, return actual substep count."""
    var substep_count = 0
    for _ in range(50):
        var obs = List[Float64](capacity=OBS_DIM)
        for k in range(1, 6):
            obs.append(Float64(data.qpos[k]))
        for k in range(6):
            var v = Float64(data.qvel[k])
            if v > 10.0:
                v = 10.0
            elif v < -10.0:
                v = -10.0
            obs.append(v)
        var action = _get_greedy_action(agent, obs)
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)
        for i in range(ACTION_DIM):
            var ctrl = action[i]
            if ctrl > HopperModel._acd.motor_ctrl_max[i]:
                ctrl = HopperModel._acd.motor_ctrl_max[i]
            elif ctrl < HopperModel._acd.motor_ctrl_min[i]:
                ctrl = HopperModel._acd.motor_ctrl_min[i]
            data.qfrc[HopperModel._acd.motor_dof_adr[i]] = Scalar[DTYPE](
                HopperModel._acd.motor_gears[i] * ctrl
            )
        for _ in range(FRAME_SKIP):
            if substep_count == TARGET_SUBSTEP:
                break
            RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)
            substep_count += 1
        if substep_count == TARGET_SUBSTEP:
            break
    return substep_count


# =============================================================================
# Phase: Stage 1 Detailed Analysis
# =============================================================================


def _analyze_stage1(
    model: ModelType,
    mut data: DataType,
    cpu_a1_smooth: List[Scalar[DTYPE]],
    M: List[Scalar[DTYPE]],
    cdof: List[Scalar[DTYPE]],
    a1: List[Scalar[DTYPE]],
    ctx: DeviceContext,
    mut gpu_state_buf: DeviceBuffer[DTYPE],
    mut gpu_model_buf: DeviceBuffer[DTYPE],
    mut gpu_ws_buf: DeviceBuffer[DTYPE],
    mut ws_host: HostBuffer[DTYPE],
    mut gpu_state_host: HostBuffer[DTYPE],
) raises -> Float64:
    """Stage 1 analysis: constraints, Hessian, search direction, linesearch.

    Returns A[1] max error after solver.
    """
    # --- Pre-solver qacc_smooth comparison ---
    print("  --- Stage 1 qacc_smooth (BEFORE solver) ---")
    var a1_smooth_err: Float64 = 0
    print("  CPU qacc_smooth:", end="")
    for i in range(NV):
        print(" " + String(Float64(cpu_a1_smooth[i]))[byte=:14], end="")
    print()
    print("  GPU qacc_smooth:", end="")
    for i in range(NV):
        var gpu_val = Float64(ws_host[QACC_CON_OFF + i])
        print(" " + String(gpu_val)[byte=:14], end="")
        var err = abs(Float64(cpu_a1_smooth[i]) - gpu_val)
        if err > a1_smooth_err:
            a1_smooth_err = err
    print()
    print("  qacc_smooth max_err = " + String(a1_smooth_err))

    # M_diag comparison
    var m1_diag_err: Float64 = 0
    for i in range(NV):
        var err = abs(
            Float64(M[i * NV + i]) - Float64(ws_host[M_OFF + i * NV + i])
        )
        if err > m1_diag_err:
            m1_diag_err = err
    print("  M_diag max_err = " + String(m1_diag_err))

    # f_smooth comparison: CPU M*qacc_smooth vs GPU f_net
    print("  --- CPU f_smooth (M * qacc_smooth) vs GPU f_smooth (f_net) ---")
    print("  CPU M*qacc_s:", end="")
    for i in range(NV):
        var cpu_f = Scalar[DTYPE](0)
        for j in range(NV):
            cpu_f += M[i * NV + j] * cpu_a1_smooth[j]
        print(" " + String(Float64(cpu_f))[byte=:14], end="")
    print()
    print("  GPU f_net:   ", end="")
    for i in range(NV):
        print(" " + String(Float64(ws_host[FNET_OFF + i]))[byte=:14], end="")
    print()
    var f_smooth_err: Float64 = 0
    for i in range(NV):
        var cpu_f = Scalar[DTYPE](0)
        for j in range(NV):
            cpu_f += M[i * NV + j] * cpu_a1_smooth[j]
        var gpu_f = Float64(ws_host[FNET_OFF + i])
        var err = abs(Float64(cpu_f) - gpu_f)
        if err > f_smooth_err:
            f_smooth_err = err
    print("  f_smooth diff = " + String(f_smooth_err) + " (LDL residual)")

    # Contact comparison
    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
    ctx.synchronize()
    var ncon_cpu = Int(data.num_contacts)
    var ncon_gpu = Int(gpu_state_host[META_OFF + META_IDX_NUM_CONTACTS])
    print("  ncon: cpu=" + String(ncon_cpu) + " gpu=" + String(ncon_gpu))
    if ncon_cpu > 0:
        print("  CPU contact dist=" + String(Float64(data.contacts[0].dist)))
        print(
            "  GPU contact dist="
            + String(
                Float64(
                    gpu_state_host[
                        CONTACTS_OFF + 0 * CONTACT_SIZE + CONTACT_IDX_DIST
                    ]
                )
            )
        )

    # --- Build CPU constraints ---
    var cpu_M_inv_s1 = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        cpu_M_inv_s1.append(Scalar[DTYPE](0))
    var cpu_L_s1 = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        cpu_L_s1.append(Scalar[DTYPE](0))
    var cpu_D_s1 = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        cpu_D_s1.append(Scalar[DTYPE](0))
    ldl_fac[DTYPE, NV](M, cpu_L_s1, cpu_D_s1)
    compute_M_inv_from_ldl[DTYPE, NV](cpu_L_s1, cpu_D_s1, cpu_M_inv_s1)

    var constraints_s1 = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints[CONE_TYPE=HopperModel.CONE_TYPE](
        model,
        data,
        cdof,
        cpu_M_inv_s1,
        model.timestep,
        constraints_s1,
    )

    print("\n  --- CPU Stage 1 constraint edge data ---")
    print("  num_rows=" + String(constraints_s1.num_rows))
    for r in range(constraints_s1.num_rows):
        var row = constraints_s1.rows[r]
        var D_r = primal_D(row.inv_K_imp, row.K)
        print(
            "  edge["
            + String(r)
            + "]: K="
            + String(Float64(row.K))[byte=:16]
            + " D="
            + String(Float64(D_r))[byte=:16]
            + " bias="
            + String(Float64(row.bias))[byte=:16]
        )
        print("    J:", end="")
        for i in range(NV):
            print(
                " " + String(Float64(constraints_s1.J[r * NV + i]))[byte=:14],
                end="",
            )
        print()

    # --- Run GPU solver (debug writes to end of workspace) ---
    _gpu_solver(ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    ctx.synchronize()
    ctx.enqueue_copy(ws_host.unsafe_ptr(), gpu_ws_buf)
    ctx.synchronize()

    # --- GPU solver iteration 0 internals (from workspace debug area) ---
    print("\n  --- GPU solver iter 0 (from workspace end) ---")
    print("  GPU grad:  ", end="")
    for i in range(NV):
        print(" " + String(Float64(ws_host[DBG_WS_OFF + i]))[byte=:14], end="")
    print()
    print("  GPU search:", end="")
    for i in range(NV):
        print(" " + String(Float64(ws_host[DBG_WS_OFF + NV + i]))[byte=:14], end="")
    print()
    print("  GPU H_diag:", end="")
    for i in range(NV):
        print(" " + String(Float64(ws_host[DBG_WS_OFF + 2 * NV + i]))[byte=:14], end="")
    print()
    print("  GPU qacc:  ", end="")
    for i in range(NV):
        print(" " + String(Float64(ws_host[DBG_WS_OFF + 3 * NV + i]))[byte=:14], end="")
    print()
    print("  GPU L_diag:", end="")
    for i in range(NV):
        print(" " + String(Float64(ws_host[DBG_WS_OFF + 4 * NV + i]))[byte=:14], end="")
    print()
    print("  GPU nc=" + String(Float64(ws_host[DBG_WS_OFF + 5 * NV]))
        + " alpha=" + String(Float64(ws_host[DBG_WS_OFF + 5 * NV + 1])))
    print("  GPU iter1_scaled_grad=" + String(Float64(ws_host[DBG_WS_OFF + 5 * NV + 2]))
        + " total_iters=" + String(Float64(ws_host[DBG_WS_OFF + 5 * NV + 3])))
    print("  GPU final_qacc:", end="")
    for i in range(NV):
        print(" " + String(Float64(ws_host[DBG_WS_OFF + 5 * NV + 4 + i]))[byte=:14], end="")
    print()

    # --- GPU edge data ---
    print("\n  --- GPU Stage 1 PYRAMIDAL edge data ---")
    for e in range(4):
        var D_e = Float64(ws_host[PYR_SC + e * MC + 0])
        var bias_e = Float64(ws_host[PYR_SC + 4 * MC + e * MC + 0])
        print(
            "  edge["
            + String(e)
            + "]: D="
            + String(D_e)[byte=:16]
            + " bias="
            + String(bias_e)[byte=:16]
        )
        print("    J:", end="")
        for i in range(NV):
            var je = Float64(ws_host[PYR_J_BASE + e * MC * NV + 0 * NV + i])
            print(" " + String(je)[byte=:14], end="")
        print()

    # --- Hessian & Cholesky analysis ---
    print("\n  --- Hessian conditioning at Stage 1 ---")

    # CPU Hessian: H = M + sum(D_e * J_e * J_e^T) for active edges
    var H_cpu = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        H_cpu.append(Scalar[DTYPE](0))
    for i in range(NV * NV):
        H_cpu[i] = M[i]
    for r in range(constraints_s1.num_rows):
        var jar_r = constraints_s1.rows[r].bias
        for i in range(NV):
            jar_r += constraints_s1.J[r * NV + i] * cpu_a1_smooth[i]
        if jar_r < Scalar[DTYPE](0):
            var D_r = primal_D(
                constraints_s1.rows[r].inv_K_imp, constraints_s1.rows[r].K
            )
            for i in range(NV):
                for j in range(NV):
                    H_cpu[i * NV + j] += (
                        D_r
                        * constraints_s1.J[r * NV + i]
                        * constraints_s1.J[r * NV + j]
                    )

    var L_cpu_h = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L_cpu_h.append(Scalar[DTYPE](0))
    _ = chol_fac[DTYPE, NV, M_SIZE](H_cpu, L_cpu_h)

    print("  CPU H diag:", end="")
    for i in range(NV):
        print(" " + String(Float64(H_cpu[i * NV + i]))[byte=:14], end="")
    print()
    print("  CPU L diag:", end="")
    for i in range(NV):
        print(" " + String(Float64(L_cpu_h[i * NV + i]))[byte=:14], end="")
    print()
    var cpu_L_min = Float64(L_cpu_h[0])
    var cpu_L_max = Float64(L_cpu_h[0])
    for i in range(1, NV):
        var v = Float64(L_cpu_h[i * NV + i])
        if v < cpu_L_min:
            cpu_L_min = v
        if v > cpu_L_max:
            cpu_L_max = v
    print(
        "  CPU cond(H) ≈ "
        + String((cpu_L_max / cpu_L_min) * (cpu_L_max / cpu_L_min))
    )

    # GPU Hessian
    var H_gpu = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        H_gpu.append(Scalar[DTYPE](0))
    for i in range(NV * NV):
        H_gpu[i] = M[i]
    for e in range(4):
        var D_e_val = Float64(ws_host[PYR_SC + e * MC + 0])
        var jar_e = Float64(ws_host[PYR_SC + 4 * MC + e * MC + 0])
        for i in range(NV):
            jar_e += Float64(ws_host[PYR_J_BASE + e * MC * NV + i]) * Float64(
                cpu_a1_smooth[i]
            )
        if jar_e < 0:
            for i in range(NV):
                for j in range(NV):
                    var ji = Float64(ws_host[PYR_J_BASE + e * MC * NV + i])
                    var jj = Float64(ws_host[PYR_J_BASE + e * MC * NV + j])
                    H_gpu[i * NV + j] += Scalar[DTYPE](D_e_val * ji * jj)

    var L_gpu_h = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L_gpu_h.append(Scalar[DTYPE](0))
    _ = chol_fac[DTYPE, NV, M_SIZE](H_gpu, L_gpu_h)

    print("  GPU H diag:", end="")
    for i in range(NV):
        print(" " + String(Float64(H_gpu[i * NV + i]))[byte=:14], end="")
    print()
    print("  GPU L diag:", end="")
    for i in range(NV):
        print(" " + String(Float64(L_gpu_h[i * NV + i]))[byte=:14], end="")
    print()
    var gpu_L_min = Float64(L_gpu_h[0])
    var gpu_L_max = Float64(L_gpu_h[0])
    for i in range(1, NV):
        var v = Float64(L_gpu_h[i * NV + i])
        if v < gpu_L_min:
            gpu_L_min = v
        if v > gpu_L_max:
            gpu_L_max = v
    print(
        "  GPU cond(H) ≈ "
        + String((gpu_L_max / gpu_L_min) * (gpu_L_max / gpu_L_min))
    )

    # H element-by-element comparison
    var H_max_err: Float64 = 0
    var H_max_idx = 0
    for i in range(NV * NV):
        var err = abs(Float64(H_cpu[i]) - Float64(H_gpu[i]))
        if err > H_max_err:
            H_max_err = err
            H_max_idx = i
    print(
        "  H cpu-gpu max_err="
        + String(H_max_err)
        + " at ["
        + String(H_max_idx)
        + "]"
    )

    # --- Search direction: search = -H^{-1} * grad ---
    var grad_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    var qfrc_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        grad_cpu.append(Scalar[DTYPE](0))
        qfrc_cpu.append(Scalar[DTYPE](0))
    for r in range(constraints_s1.num_rows):
        var jar_r = constraints_s1.rows[r].bias
        for i in range(NV):
            jar_r += constraints_s1.J[r * NV + i] * cpu_a1_smooth[i]
        if jar_r < Scalar[DTYPE](0):
            var D_r = primal_D(
                constraints_s1.rows[r].inv_K_imp, constraints_s1.rows[r].K
            )
            var force_r = -D_r * jar_r
            for i in range(NV):
                qfrc_cpu[i] += constraints_s1.J[r * NV + i] * force_r
    for i in range(NV):
        var ma_i = Scalar[DTYPE](0)
        for j in range(NV):
            ma_i += M[i * NV + j] * cpu_a1_smooth[j]
        grad_cpu[i] = ma_i - ma_i - qfrc_cpu[i]
    var search_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        search_cpu.append(Scalar[DTYPE](0))
    chol_slv[DTYPE, NV, M_SIZE, V_SIZE](L_cpu_h, grad_cpu, search_cpu)
    for i in range(NV):
        search_cpu[i] = -search_cpu[i]
    print("  CPU search dir:", end="")
    for i in range(NV):
        print(" " + String(Float64(search_cpu[i]))[byte=:14], end="")
    print()

    # GPU search direction
    var grad_gpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    var qfrc_gpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        grad_gpu.append(Scalar[DTYPE](0))
        qfrc_gpu.append(Scalar[DTYPE](0))
    for e in range(4):
        var D_e_val = Scalar[DTYPE](ws_host[PYR_SC + e * MC + 0])
        var jar_e = Scalar[DTYPE](ws_host[PYR_SC + 4 * MC + e * MC + 0])
        for i in range(NV):
            jar_e += (
                Scalar[DTYPE](ws_host[PYR_J_BASE + e * MC * NV + i])
                * cpu_a1_smooth[i]
            )
        if jar_e < Scalar[DTYPE](0):
            var force_e = -D_e_val * jar_e
            for i in range(NV):
                qfrc_gpu[i] += (
                    Scalar[DTYPE](ws_host[PYR_J_BASE + e * MC * NV + i])
                    * force_e
                )
    for i in range(NV):
        var ma_i = Scalar[DTYPE](0)
        for j in range(NV):
            ma_i += M[i * NV + j] * cpu_a1_smooth[j]
        grad_gpu[i] = ma_i - ma_i - qfrc_gpu[i]
    var search_gpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        search_gpu.append(Scalar[DTYPE](0))
    chol_slv[DTYPE, NV, M_SIZE, V_SIZE](L_gpu_h, grad_gpu, search_gpu)
    for i in range(NV):
        search_gpu[i] = -search_gpu[i]
    print("  GPU search dir:", end="")
    for i in range(NV):
        print(" " + String(Float64(search_gpu[i]))[byte=:14], end="")
    print()
    var search_err: Float64 = 0
    for i in range(NV):
        var err = abs(Float64(search_cpu[i]) - Float64(search_gpu[i]))
        if err > search_err:
            search_err = err
    print("  search dir max_err = " + String(search_err))

    # --- Linesearch alpha analysis ---
    print("\n  --- Linesearch alpha_1 analysis ---")

    # CPU linesearch
    var Mv_cpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        Mv_cpu.append(Scalar[DTYPE](0))
    for i in range(NV):
        for j in range(NV):
            Mv_cpu[i] += M[i * NV + j] * search_cpu[j]

    var gauss_a_cpu: Float64 = 0
    for i in range(NV):
        gauss_a_cpu += Float64(Mv_cpu[i]) * Float64(search_cpu[i])
    print("  CPU gauss_a (search.M.search) = " + String(gauss_a_cpu))

    var d1_cpu: Float64 = 0
    var d2_cpu: Float64 = gauss_a_cpu
    print("  CPU per-edge at alpha=0:")
    for r in range(constraints_s1.num_rows):
        var jar_r: Float64 = Float64(constraints_s1.rows[r].bias)
        var Jv_r: Float64 = 0
        for i in range(NV):
            jar_r += Float64(constraints_s1.J[r * NV + i]) * Float64(
                cpu_a1_smooth[i]
            )
            Jv_r += Float64(constraints_s1.J[r * NV + i]) * Float64(
                search_cpu[i]
            )
        var D_r = Float64(
            primal_D(constraints_s1.rows[r].inv_K_imp, constraints_s1.rows[r].K)
        )
        if jar_r < 0:
            d1_cpu += D_r * jar_r * Jv_r
            d2_cpu += D_r * Jv_r * Jv_r
        var alpha_cross: Float64 = -1.0
        if abs(Jv_r) > 1e-10:
            alpha_cross = -jar_r / Jv_r
        print(
            "    edge["
            + String(r)
            + "]: jar="
            + String(jar_r)[byte=:16]
            + " Jv="
            + String(Jv_r)[byte=:14]
            + " alpha_cross="
            + String(alpha_cross)[byte=:14]
            + " active="
            + String(jar_r < 0)
        )
    print("  CPU d1=" + String(d1_cpu) + " d2=" + String(d2_cpu))
    var alpha1_cpu: Float64 = 0
    if d2_cpu > 1e-12:
        alpha1_cpu = -d1_cpu / d2_cpu
    print("  CPU alpha_1 = " + String(alpha1_cpu))

    # GPU linesearch
    var Mv_gpu = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        Mv_gpu.append(Scalar[DTYPE](0))
    for i in range(NV):
        for j in range(NV):
            Mv_gpu[i] += M[i * NV + j] * search_gpu[j]

    var gauss_a_gpu: Float64 = 0
    for i in range(NV):
        gauss_a_gpu += Float64(Mv_gpu[i]) * Float64(search_gpu[i])

    var d1_gpu: Float64 = 0
    var d2_gpu: Float64 = gauss_a_gpu
    print("  GPU per-edge at alpha=0:")
    for e in range(4):
        var jar_e: Float64 = Float64(ws_host[PYR_SC + 4 * MC + e * MC + 0])
        var Jv_e: Float64 = 0
        for i in range(NV):
            jar_e += Float64(ws_host[PYR_J_BASE + e * MC * NV + i]) * Float64(
                cpu_a1_smooth[i]
            )
            Jv_e += Float64(ws_host[PYR_J_BASE + e * MC * NV + i]) * Float64(
                search_gpu[i]
            )
        var D_e_val = Float64(ws_host[PYR_SC + e * MC + 0])
        if jar_e < 0:
            d1_gpu += D_e_val * jar_e * Jv_e
            d2_gpu += D_e_val * Jv_e * Jv_e
        var alpha_cross: Float64 = -1.0
        if abs(Jv_e) > 1e-10:
            alpha_cross = -jar_e / Jv_e
        print(
            "    edge["
            + String(e)
            + "]: jar="
            + String(jar_e)[byte=:16]
            + " Jv="
            + String(Jv_e)[byte=:14]
            + " alpha_cross="
            + String(alpha_cross)[byte=:14]
            + " active="
            + String(jar_e < 0)
        )
    print("  GPU d1=" + String(d1_gpu) + " d2=" + String(d2_gpu))
    var alpha1_gpu: Float64 = 0
    if d2_gpu > 1e-12:
        alpha1_gpu = -d1_gpu / d2_gpu
    print("  GPU alpha_1 = " + String(alpha1_gpu))

    # --- A[1] comparison after solver ---
    var a1_max_err = _print_compare_qacc("A[1]", a1, ws_host)

    # --- Intermediate state comparison ---
    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
    ctx.synchronize()
    _print_compare_state(data, gpu_state_host)

    return a1_max_err


# =============================================================================
# Main Diagnostic Function
# =============================================================================


def debug_rk4_instability() raises:
    seed(42)
    print("=" * 70)
    print(
        "Per-Stage RK4 Diagnostic at substep "
        + String(TARGET_SUBSTEP)
        + " linesearch convergence tolerance"
    )
    print("=" * 70)

    var agent = AgentType(
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=False,
        target_entropy=-3.0,
    )
    agent.load_checkpoint("sac_hopper_1000.ckpt")
    print("Loaded checkpoint")

    # === CPU setup ===
    var cpu_env = Hopper[DTYPE, TERMINATE_ON_UNHEALTHY=True]()
    _ = cpu_env.reset()
    var cpu_model = ModelType()
    var cpu_data = DataType()
    HopperModel.setup_model_and_data(cpu_model, cpu_data)
    for i in range(NQ):
        cpu_data.qpos[i] = cpu_env.get_qpos(i)
    for i in range(NV):
        cpu_data.qvel[i] = cpu_env.get_qvel(i)
    forward_kinematics(cpu_model, cpu_data)

    # === GPU setup ===
    var ctx = DeviceContext()
    var gpu_state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)
    var gpu_state_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * STATE_SIZE)
    var gpu_model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HopperModel.init_model_gpu(ctx, gpu_model_buf)
    var gpu_ws_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * WS_SIZE)
    ctx.synchronize()
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        gpu_state_host[QPOS_OFF + i] = cpu_data.qpos[i]
    for i in range(NV):
        gpu_state_host[QVEL_OFF + i] = cpu_data.qvel[i]
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
    ctx.synchronize()

    # =========================================================================
    # Advance to target substep
    # =========================================================================
    print("\nAdvancing to substep " + String(TARGET_SUBSTEP) + "...")
    var substep_count = _advance_to_substep(agent, cpu_model, cpu_data)
    print("Reached substep " + String(substep_count))

    # Save initial state for CPU manual RK4
    var q0 = List[Scalar[DTYPE]](capacity=NQ)
    var v0 = List[Scalar[DTYPE]](capacity=NV)
    for i in range(NQ):
        q0.append(cpu_data.qpos[i])
    for i in range(NV):
        v0.append(cpu_data.qvel[i])
    var dt = cpu_model.timestep
    var half_dt = dt * Scalar[DTYPE](0.5)

    # Sync to GPU
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        gpu_state_host[QPOS_OFF + i] = cpu_data.qpos[i]
    for i in range(NV):
        gpu_state_host[QVEL_OFF + i] = cpu_data.qvel[i]
        gpu_state_host[QACC_OFF + i] = cpu_data.qacc[i]
        gpu_state_host[QFRC_OFF + i] = cpu_data.qfrc[i]
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH * WS_SIZE)
    for i in range(GPU_BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(gpu_ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # CPU workspace for manual RK4
    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    var M_inv = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_inv.append(Scalar[DTYPE](0))
    var M = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M.append(Scalar[DTYPE](0))

    var a0 = List[Scalar[DTYPE]](capacity=V_SIZE)
    var a1 = List[Scalar[DTYPE]](capacity=V_SIZE)
    var a2 = List[Scalar[DTYPE]](capacity=V_SIZE)
    var a3 = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        a0.append(Scalar[DTYPE](0))
        a1.append(Scalar[DTYPE](0))
        a2.append(Scalar[DTYPE](0))
        a3.append(Scalar[DTYPE](0))

    var q_stage = List[Scalar[DTYPE]](capacity=NQ)
    for _ in range(NQ):
        q_stage.append(Scalar[DTYPE](0))

    # Reset warmstart so CPU starts from qacc_smooth like GPU
    for i in range(NV):
        cpu_data.qacc_warmstart[i] = Scalar[DTYPE](0)

    # =========================================================================
    # Stage 0: evaluate at (q0, v0)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Stage 0: evaluate at (q0, v0)")
    print("=" * 70)

    _cpu_fwd(cpu_model, cpu_data, a0, cdof, M_inv, M)
    _cpu_solve(cpu_model, cpu_data, cdof, M_inv, M, a0, dt, False)

    _gpu_stage[0](ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    _gpu_solver(ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    ctx.synchronize()
    ctx.enqueue_copy(ws_host.unsafe_ptr(), gpu_ws_buf)
    ctx.synchronize()

    var a0_max_err = _print_compare_qacc("A[0]", a0, ws_host)

    # M_diag comparison
    print("  f_net max_err:", end="")
    var fnet_err: Float64 = 0
    for i in range(NV):
        var cpu_fnet = Float64(M[i * NV + i])
        var gpu_fnet = Float64(ws_host[M_OFF + i * NV + i])
        var err = abs(cpu_fnet - gpu_fnet)
        if err > fnet_err:
            fnet_err = err
    print(" M_diag_err=" + String(fnet_err))

    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
    ctx.synchronize()
    _print_ncon(cpu_data, gpu_state_host)

    # =========================================================================
    # Stage 1: evaluate at (q0 + dt/2*v0, v0 + dt/2*A[0])
    # =========================================================================
    print("\n" + "=" * 70)
    print("Stage 1: evaluate at (q0 + dt/2*v0, v0 + dt/2*A[0])")
    print("=" * 70)

    # CPU: set intermediate state for stage 1
    for i in range(NV):
        cpu_data.qvel[i] = v0[i] + half_dt * a0[i]
    _cpu_integ_pos(cpu_model, q0, v0, half_dt, q_stage)
    for i in range(NQ):
        cpu_data.qpos[i] = q_stage[i]
    for i in range(NV):
        cpu_data.qacc_warmstart[i] = Scalar[DTYPE](0)
    _cpu_fwd(cpu_model, cpu_data, a1, cdof, M_inv, M)

    # Save CPU qacc_smooth before solver modifies a1
    var cpu_a1_smooth = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(NV):
        cpu_a1_smooth.append(a1[i])

    _cpu_solve(cpu_model, cpu_data, cdof, M_inv, M, a1, dt, False)

    # GPU: stage 1 kernel only (no solver yet — analysis runs solver)
    _gpu_stage[1](ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    ctx.synchronize()
    ctx.enqueue_copy(ws_host.unsafe_ptr(), gpu_ws_buf)
    ctx.synchronize()

    # Detailed analysis (constraints, Hessian, search, linesearch, solver)
    var a1_max_err = _analyze_stage1(
        cpu_model,
        cpu_data,
        cpu_a1_smooth,
        M,
        cdof,
        a1,
        ctx,
        gpu_state_buf,
        gpu_model_buf,
        gpu_ws_buf,
        ws_host,
        gpu_state_host,
    )

    # =========================================================================
    # Stage 2: evaluate at (q0 + dt/2*C[1], v0 + dt/2*A[1])
    # =========================================================================
    print("\n" + "=" * 70)
    print("Stage 2: evaluate at (q0 + dt/2*C[1], v0 + dt/2*A[1])")
    print("=" * 70)

    var c1 = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        c1.append(Scalar[DTYPE](0))
    for i in range(NV):
        c1[i] = v0[i] + half_dt * a0[i]
    for i in range(NV):
        cpu_data.qvel[i] = v0[i] + half_dt * a1[i]
    _cpu_integ_pos(cpu_model, q0, c1, half_dt, q_stage)
    for i in range(NQ):
        cpu_data.qpos[i] = q_stage[i]
    for i in range(NV):
        cpu_data.qacc_warmstart[i] = Scalar[DTYPE](0)
    _cpu_fwd(cpu_model, cpu_data, a2, cdof, M_inv, M)
    _cpu_solve(cpu_model, cpu_data, cdof, M_inv, M, a2, dt, False)

    _gpu_stage[2](ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    _gpu_solver(ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    ctx.synchronize()
    ctx.enqueue_copy(ws_host.unsafe_ptr(), gpu_ws_buf)
    ctx.synchronize()

    var a2_max_err = _print_compare_qacc("A[2]", a2, ws_host)
    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
    ctx.synchronize()
    _print_compare_state(cpu_data, gpu_state_host)

    # =========================================================================
    # Stage 3: evaluate at (q0 + dt*C[2], v0 + dt*A[2])
    # =========================================================================
    print("\n" + "=" * 70)
    print("Stage 3: evaluate at (q0 + dt*C[2], v0 + dt*A[2])")
    print("=" * 70)

    var c2 = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        c2.append(Scalar[DTYPE](0))
    for i in range(NV):
        c2[i] = v0[i] + half_dt * a1[i]
    for i in range(NV):
        cpu_data.qvel[i] = v0[i] + dt * a2[i]
    _cpu_integ_pos(cpu_model, q0, c2, dt, q_stage)
    for i in range(NQ):
        cpu_data.qpos[i] = q_stage[i]
    for i in range(NV):
        cpu_data.qacc_warmstart[i] = Scalar[DTYPE](0)
    _cpu_fwd(cpu_model, cpu_data, a3, cdof, M_inv, M)
    _cpu_solve(cpu_model, cpu_data, cdof, M_inv, M, a3, dt, True)

    _gpu_stage[3](ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    _gpu_solver(ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    ctx.synchronize()
    ctx.enqueue_copy(ws_host.unsafe_ptr(), gpu_ws_buf)
    ctx.synchronize()

    var a3_max_err = _print_compare_qacc("A[3]", a3, ws_host)
    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
    ctx.synchronize()
    _print_compare_state(cpu_data, gpu_state_host)

    # =========================================================================
    # Combine
    # =========================================================================
    print("\n" + "=" * 70)
    print("RK4 Combine")
    print("=" * 70)

    _gpu_combine(ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
    ctx.synchronize()

    # CPU combine
    comptime ONE_SIXTH: Scalar[DTYPE] = 1.0 / 6.0
    comptime ONE_THIRD: Scalar[DTYPE] = 1.0 / 3.0
    var cpu_qacc_combined = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        cpu_qacc_combined.append(Scalar[DTYPE](0))
    for i in range(NV):
        cpu_qacc_combined[i] = (
            ONE_SIXTH * a0[i]
            + ONE_THIRD * a1[i]
            + ONE_THIRD * a2[i]
            + ONE_SIXTH * a3[i]
        )

    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
    ctx.synchronize()

    var combined_err: Float64 = 0
    print("  CPU qacc_combined:", end="")
    for i in range(NV):
        print(" " + String(Float64(cpu_qacc_combined[i]))[byte=:14], end="")
    print()
    print("  GPU qacc_combined:", end="")
    for i in range(NV):
        var gpu_val = Float64(gpu_state_host[QACC_OFF + i])
        print(" " + String(gpu_val)[byte=:14], end="")
        var err = abs(Float64(cpu_qacc_combined[i]) - gpu_val)
        if err > combined_err:
            combined_err = err
    print()
    print("  combined max_err = " + String(combined_err))

    cpu_env.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY — per-stage error amplification")
    print("=" * 70)
    print("  Stage 0: A[0] max_err = " + String(a0_max_err))
    print("  Stage 1: A[1] max_err = " + String(a1_max_err))
    print("  Stage 2: A[2] max_err = " + String(a2_max_err))
    print("  Stage 3: A[3] max_err = " + String(a3_max_err))
    print("  Combined qacc max_err = " + String(combined_err))
