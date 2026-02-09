"""Benchmark individual phases of the EulerIntegrator GPU kernel.

Strategy: Since separate per-phase kernels are dominated by Metal kernel launch
overhead (~300us), we instead measure:
  1. Full step with each solver (PGS, CG, Newton)
  2. A "no-solver" kernel that does ALL phases EXCEPT the constraint solve
  3. Subtract to get solver-only cost

Also tests multiple batch sizes to see scaling behavior.

Uses a Hopper model (4 bodies, 6 joints, NQ=6, NV=6, MAX_CONTACTS=10).

Run with:
    cd mojo-rl
    pixi run -e apple mojo run physics3d/tests/benchmark_integrator_phases.mojo
"""

from time import perf_counter_ns
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data, _max_one, compute_capsule_inertia
from physics3d.constants import GEOM_CAPSULE
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE

from physics3d.kinematics.forward_kinematics import (
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full_gpu,
    ldl_factor_gpu,
    ldl_solve_gpu,
    ldl_solve_workspace_gpu,
    compute_M_inv_from_ldl_gpu,
)
from physics3d.dynamics.bias_forces import compute_bias_forces_rne_gpu
from physics3d.dynamics.jacobian import (
    compute_cdof_gpu,
    compute_composite_inertia_gpu,
)
from physics3d.collision.contact_detection import (
    detect_ground_contacts_gpu,
    detect_body_body_contacts_gpu,
    normalize_qpos_quaternions_gpu,
)
from physics3d.solver.pgs_solver import PGSSolver
from physics3d.solver.cg_solver import CGSolver
from physics3d.solver.newton_solver import NewtonSolver
from physics3d.integrator.euler_integrator import EulerIntegrator

from physics3d.gpu.constants import (
    TPB,
    state_size,
    model_size,
    integrator_workspace_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    model_joint_offset,
    model_metadata_offset,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    MODEL_META_IDX_TIMESTEP,
    ws_M_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qvel_pred_offset,
    ws_m_inv_offset,
)
from physics3d.gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_data_to_buffer,
)


# =============================================================================
# Model dimensions (Hopper)
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ: Int = 6
comptime NV: Int = 6
comptime NBODY: Int = 4
comptime NJOINT: Int = 6
comptime MAX_CONTACTS: Int = 10

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size[NBODY, NJOINT]()
# Use Newton (largest) workspace size since all 3 solvers share the buffer
comptime WS_SIZE = integrator_workspace_size[NV, NBODY]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime V_SIZE: Int = 6
comptime M_SIZE: Int = 36
comptime CDOF_SIZE: Int = 36
comptime CRB_SIZE: Int = 40

# Benchmark config
comptime WARMUP: Int = 50
comptime ITERS: Int = 200


# =============================================================================
# Kernel: Full step without solver (phases 1-10 + 13, skipping constraint solve)
# =============================================================================

@always_inline
fn no_solver_kernel[BATCH: Int](
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin],
):
    """Full integrator step minus the constraint solver.

    Reproduces the same phases as step_kernel + step_finalize_kernel:
    - Skips the solver phase (phase 11)
    - All computation is identical, but fused into a single kernel
    """
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return

    # Phase 1: Forward kinematics
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # Phase 2: Body velocities
    compute_body_velocities_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # Phase 3: Contact detection
    detect_ground_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)
    detect_body_body_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # Phase 4: CDOF (writes to workspace)
    compute_cdof_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # Phase 5: Composite inertia (writes to workspace)
    compute_composite_inertia_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # Phase 6: Mass matrix (reads cdof/crb, writes M in workspace)
    compute_mass_matrix_full_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # Phase 6b: Armature + implicit damping
    comptime M_idx = ws_M_offset[NV, NBODY]()
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var dt = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_TIMESTEP])
    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR]))
        var arm = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_ARMATURE])
        var damp = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DAMPING])
        var diag_add = arm + dt * damp
        if jnt_type == JNT_SLIDE or jnt_type == JNT_HINGE:
            var idx = M_idx + dof_adr * NV + dof_adr
            workspace[env, idx] = workspace[env, idx] + diag_add

    # Phase 7: LDL factorization + M_inv (all in workspace)
    ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)
    compute_M_inv_from_ldl_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)

    # Phase 8: Bias forces (reads cdof, writes bias in workspace)
    compute_bias_forces_rne_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # Phase 9: Net forces + stiffness (writes f_net in workspace)
    comptime bias_idx = ws_bias_offset[NV, NBODY]()
    comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
    var qfrc_off = qfrc_offset[NQ, NV]()
    var qpos_off_stiff = qpos_offset[NQ, NV]()
    for i in range(NV):
        var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
        var bias_val = rebind[Scalar[DTYPE]](workspace[env, bias_idx + i])
        workspace[env, fnet_idx + i] = qfrc - bias_val

    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR]))
        var qpos_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR]))
        var stiff = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_STIFFNESS])
        if stiff > Scalar[DTYPE](0):
            if jnt_type == JNT_SLIDE or jnt_type == JNT_HINGE:
                var qpos_d = rebind[Scalar[DTYPE]](state[env, qpos_off_stiff + qpos_adr])
                var cur = rebind[Scalar[DTYPE]](workspace[env, fnet_idx + dof_adr])
                workspace[env, fnet_idx + dof_adr] = cur - stiff * qpos_d

    # Phase 10: Unconstrained accel + predicted velocity (all in workspace)
    comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()
    comptime qvel_pred_idx = ws_qvel_pred_offset[NV, NBODY]()
    ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)

    var qacc_off = qacc_offset[NQ, NV]()
    for i in range(NV):
        var qacc_val = rebind[Scalar[DTYPE]](workspace[env, qacc_ws_idx + i])
        state[env, qacc_off + i] = qacc_val

    var qvel_off = qvel_offset[NQ, NV]()
    for i in range(NV):
        var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        var qacc_val = rebind[Scalar[DTYPE]](workspace[env, qacc_ws_idx + i])
        workspace[env, qvel_pred_idx + i] = qvel + qacc_val * dt

    # *** SOLVER SKIPPED (phase 11) ***

    # Phase 12-13: Write back + integrate
    var qpos_off = qpos_offset[NQ, NV]()
    for i in range(NV):
        var old_qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        var constrained_vel = rebind[Scalar[DTYPE]](workspace[env, qvel_pred_idx + i])
        state[env, qacc_off + i] = (constrained_vel - old_qvel) / dt
        state[env, qvel_off + i] = constrained_vel

    comptime MAX_QVEL: Scalar[DTYPE] = 20.0
    for i in range(NV):
        var v = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        if v > MAX_QVEL:
            state[env, qvel_off + i] = MAX_QVEL
        elif v < -MAX_QVEL:
            state[env, qvel_off + i] = -MAX_QVEL

    for i in range(NQ):
        if i < NV:
            var qpos = rebind[Scalar[DTYPE]](state[env, qpos_off + i])
            var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            state[env, qpos_off + i] = qpos + qvel * dt

    normalize_qpos_quaternions_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)


# =============================================================================
# 3-phase timing helper (step_kernel + solve_gpu + step_finalize_kernel)
# =============================================================================

fn time_full_step[
    BATCH: Int,
    solve_fn: fn (
        LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
        LayoutTensor[DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin],
    ) -> None,
    SOLVER_ENV_BLOCKS: Int,
    SOLVER_TH_BLOCKS: Int,
    SOLVER_ENV_TPB: Int,
    SOLVER_THREADS: Int,
](
    ctx: DeviceContext,
    state_buf: DeviceBuffer[DTYPE],
    state_host: HostBuffer[DTYPE],
    st: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    md: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    ws: LayoutTensor[DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin],
    name: String,
) raises -> Float64:
    comptime BLOCKS = (BATCH + TPB - 1) // TPB
    comptime step_fn = EulerIntegrator[PGSSolver].step_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ]
    comptime finalize_fn = EulerIntegrator[PGSSolver].step_finalize_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ]

    # Reset state
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.synchronize()

    # Warmup
    for _ in range(WARMUP):
        ctx.enqueue_function[step_fn, step_fn](
            st, md, ws, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
        ctx.enqueue_function[solve_fn, solve_fn](
            st, md, ws,
            grid_dim=(SOLVER_ENV_BLOCKS, SOLVER_TH_BLOCKS),
            block_dim=(SOLVER_ENV_TPB, SOLVER_THREADS),
        )
        ctx.enqueue_function[finalize_fn, finalize_fn](
            st, md, ws, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
    ctx.synchronize()

    # Timed
    var start = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[step_fn, step_fn](
            st, md, ws, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
        ctx.enqueue_function[solve_fn, solve_fn](
            st, md, ws,
            grid_dim=(SOLVER_ENV_BLOCKS, SOLVER_TH_BLOCKS),
            block_dim=(SOLVER_ENV_TPB, SOLVER_THREADS),
        )
        ctx.enqueue_function[finalize_fn, finalize_fn](
            st, md, ws, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    var us = Float64(end - start) / Float64(ITERS) / 1000.0
    print("  ", name, ": ", us, " us/iter")
    return us


# =============================================================================
# Timing helper (top-level, parametric)
# =============================================================================

fn time_kernel[
    BATCH: Int,
    KFn: fn (
        LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
        LayoutTensor[DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin],
    ) -> None,
](
    ctx: DeviceContext,
    state_buf: DeviceBuffer[DTYPE],
    state_host: HostBuffer[DTYPE],
    st: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    md: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    ws: LayoutTensor[DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin],
    name: String,
) raises -> Float64:
    comptime BLOCKS = (BATCH + TPB - 1) // TPB

    # Reset state
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.synchronize()

    # Warmup
    for _ in range(WARMUP):
        ctx.enqueue_function[KFn, KFn](
            st, md, ws, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
    ctx.synchronize()

    # Timed
    var start = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[KFn, KFn](
            st, md, ws, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    var us = Float64(end - start) / Float64(ITERS) / 1000.0
    print("  ", name, ": ", us, " us/iter")
    return us


# =============================================================================
# Benchmark runner for a given batch size
# =============================================================================

fn run_benchmark[BATCH: Int](
    ctx: DeviceContext,
    model_host: HostBuffer[DTYPE],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
) raises:
    """Run full benchmark suite for a given BATCH size."""
    print()
    print("=" * 70)
    print("BATCH =", BATCH)
    print("=" * 70)

    # Allocate buffers
    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for b in range(BATCH):
        copy_data_to_buffer[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](data, state_host, b)

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    var st = LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin](
        state_buf.unsafe_ptr()
    )
    var md = LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin](
        model_buf.unsafe_ptr()
    )
    var ws_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws = LayoutTensor[DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin](
        ws_buf.unsafe_ptr()
    )

    # --- Run benchmarks ---
    var t_no_solver = time_kernel[BATCH, no_solver_kernel[BATCH]](
        ctx, state_buf, state_host, st, md, ws, "No-solver (phases 1-10,13)",
    )

    # Solver grid/block dimensions for 3-phase pipeline
    comptime SV = _max_one[NV]()

    comptime PGS_TH = PGSSolver.solver_threads[NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    comptime PGS_ETPB = TPB // PGS_TH
    comptime PGS_EB = (BATCH + PGS_ETPB - 1) // PGS_ETPB
    comptime PGS_TB = (PGS_TH + PGS_TH - 1) // PGS_TH

    comptime CG_TH = CGSolver.solver_threads[NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    comptime CG_ETPB = TPB // CG_TH
    comptime CG_EB = (BATCH + CG_ETPB - 1) // CG_ETPB
    comptime CG_TB = (CG_TH + CG_TH - 1) // CG_TH

    comptime NWT_TH = NewtonSolver.solver_threads[NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    comptime NWT_ETPB = TPB // NWT_TH
    comptime NWT_EB = (BATCH + NWT_ETPB - 1) // NWT_ETPB
    comptime NWT_TB = (NWT_TH + NWT_TH - 1) // NWT_TH

    var t_pgs = time_full_step[
        BATCH,
        PGSSolver.solve_gpu[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, SV, BATCH, WS_SIZE],
        PGS_EB, PGS_TB, PGS_ETPB, PGS_TH,
    ](ctx, state_buf, state_host, st, md, ws, "Full step [PGS]           ")
    var t_cg = time_full_step[
        BATCH,
        CGSolver.solve_gpu[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, SV, BATCH, WS_SIZE],
        CG_EB, CG_TB, CG_ETPB, CG_TH,
    ](ctx, state_buf, state_host, st, md, ws, "Full step [CG]            ")
    var t_newton = time_full_step[
        BATCH,
        NewtonSolver.solve_gpu[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, SV, BATCH, WS_SIZE],
        NWT_EB, NWT_TB, NWT_ETPB, NWT_TH,
    ](ctx, state_buf, state_host, st, md, ws, "Full step [Newton]        ")

    # --- Derived metrics ---
    var s_pgs = t_pgs - t_no_solver
    var s_cg = t_cg - t_no_solver
    var s_newton = t_newton - t_no_solver

    print()
    print("  Solver-only (full - no_solver):")
    print("    PGS    : ", s_pgs, " us")
    print("    CG     : ", s_cg, " us")
    print("    Newton : ", s_newton, " us")
    if s_pgs > 0:
        print("    CG/PGS : ", s_cg / s_pgs, "x")
        print("    Nwt/PGS: ", s_newton / s_pgs, "x")
    print()
    print("  Non-solver overhead: ", t_no_solver, " us (", t_no_solver / t_pgs * 100.0, "% of PGS step)")
    print()
    print("  Throughput (envs/sec):")
    if t_pgs > 0:
        print("    PGS    : ", Float64(BATCH) / t_pgs * 1e6, " envs/s")
    if t_cg > 0:
        print("    CG     : ", Float64(BATCH) / t_cg * 1e6, " envs/s")
    if t_newton > 0:
        print("    Newton : ", Float64(BATCH) / t_newton * 1e6, " envs/s")
    print()
    print("  Per-env time:")
    if t_pgs > 0:
        print("    PGS    : ", t_pgs / Float64(BATCH) * 1000.0, " ns/env")
    if t_cg > 0:
        print("    CG     : ", t_cg / Float64(BATCH) * 1000.0, " ns/env")
    if t_newton > 0:
        print("    Newton : ", t_newton / Float64(BATCH) * 1000.0, " ns/env")


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    print("=" * 70)
    print("Physics3D EulerIntegrator GPU Phase Benchmark")
    print("=" * 70)
    print("Model: Hopper (NBODY=4, NJOINT=6, NQ=6, NV=6, MAX_CONTACTS=10)")
    print("STATE_SIZE:", STATE_SIZE, " MODEL_SIZE:", MODEL_SIZE)
    print("Warmup:", WARMUP, " Iters:", ITERS)

    # =========================================================================
    # Create Hopper model
    # =========================================================================
    var torso_mass = Scalar[DTYPE](3.53429174)
    var torso_radius = Scalar[DTYPE](0.05)
    var torso_half_length = Scalar[DTYPE](0.2)
    var thigh_mass = Scalar[DTYPE](3.92699082)
    var thigh_radius = Scalar[DTYPE](0.05)
    var thigh_half_length = Scalar[DTYPE](0.225)
    var leg_mass = Scalar[DTYPE](2.71433605)
    var leg_radius = Scalar[DTYPE](0.04)
    var leg_half_length = Scalar[DTYPE](0.25)
    var foot_mass = Scalar[DTYPE](5.0893801)
    var foot_radius = Scalar[DTYPE](0.06)
    var foot_half_length = Scalar[DTYPE](0.195)

    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.002),
        ground_z=Scalar[DTYPE](0.0),
        friction=Scalar[DTYPE](0.5),
    )

    var torso_inertia = compute_capsule_inertia(torso_mass, torso_radius, torso_half_length)
    model.set_body(0, mass=torso_mass, inertia=torso_inertia, radius=torso_radius)
    model.set_body_parent(0, -1)
    model.body_geom_type[0] = GEOM_CAPSULE
    model.body_half_length[0] = torso_half_length

    var thigh_inertia = compute_capsule_inertia(thigh_mass, thigh_radius, thigh_half_length)
    model.set_body(1, mass=thigh_mass, inertia=thigh_inertia, radius=thigh_radius)
    model.set_body_parent(1, 0)
    model.body_geom_type[1] = GEOM_CAPSULE
    model.body_half_length[1] = thigh_half_length
    model.set_body_local_frame(
        1, pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -(torso_half_length + thigh_half_length)),
    )

    var leg_inertia = compute_capsule_inertia(leg_mass, leg_radius, leg_half_length)
    model.set_body(2, mass=leg_mass, inertia=leg_inertia, radius=leg_radius)
    model.set_body_parent(2, 1)
    model.body_geom_type[2] = GEOM_CAPSULE
    model.body_half_length[2] = leg_half_length
    model.set_body_local_frame(
        2, pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -(thigh_half_length + leg_half_length)),
    )

    var foot_inertia = compute_capsule_inertia(foot_mass, foot_radius, foot_half_length)
    model.set_body(3, mass=foot_mass, inertia=foot_inertia, radius=foot_radius)
    model.set_body_parent(3, 2)
    model.body_geom_type[3] = GEOM_CAPSULE
    model.body_half_length[3] = foot_half_length
    model.set_body_local_frame(
        3, pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -leg_half_length),
        quat=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.70710678), Scalar[DTYPE](0.0), Scalar[DTYPE](0.70710678)),
    )

    _ = model.add_slide_joint(
        body_id=0,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](1.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        force_limit=Scalar[DTYPE](0.0),
    )
    _ = model.add_slide_joint(
        body_id=0,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](1.0)),
        force_limit=Scalar[DTYPE](0.0),
    )
    _ = model.add_hinge_joint(
        body_id=0,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
        tau_limit=Scalar[DTYPE](0.0),
    )
    _ = model.add_hinge_joint(
        body_id=1,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -torso_half_length),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
        tau_limit=Scalar[DTYPE](200.0),
        range_min=Scalar[DTYPE](-2.618), range_max=Scalar[DTYPE](0.0),
        armature=Scalar[DTYPE](1.0), damping=Scalar[DTYPE](1.0),
    )
    _ = model.add_hinge_joint(
        body_id=2,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -thigh_half_length),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
        tau_limit=Scalar[DTYPE](200.0),
        range_min=Scalar[DTYPE](-2.618), range_max=Scalar[DTYPE](0.0),
        armature=Scalar[DTYPE](1.0), damping=Scalar[DTYPE](1.0),
    )
    _ = model.add_hinge_joint(
        body_id=3,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -leg_half_length),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
        tau_limit=Scalar[DTYPE](200.0),
        range_min=Scalar[DTYPE](-0.785), range_max=Scalar[DTYPE](0.785),
        armature=Scalar[DTYPE](1.0), damping=Scalar[DTYPE](1.0),
    )

    # =========================================================================
    # Set up GPU context and model buffer (shared across batch sizes)
    # =========================================================================
    var ctx = DeviceContext()

    var model_host = ctx.enqueue_create_host_buffer[DTYPE](MODEL_SIZE)
    for i in range(MODEL_SIZE):
        model_host[i] = Scalar[DTYPE](0)
    copy_model_to_buffer[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](model, model_host)

    # Initial data with perturbation (contacts will be generated)
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    data.qpos[0] = Scalar[DTYPE](0.0)
    data.qpos[1] = Scalar[DTYPE](1.05)
    data.qpos[2] = Scalar[DTYPE](0.1)
    data.qpos[3] = Scalar[DTYPE](-0.2)
    data.qpos[4] = Scalar[DTYPE](-0.1)
    data.qpos[5] = Scalar[DTYPE](0.05)
    data.qvel[0] = Scalar[DTYPE](0.5)
    data.qvel[1] = Scalar[DTYPE](-0.3)
    data.qvel[2] = Scalar[DTYPE](0.1)
    data.qvel[3] = Scalar[DTYPE](0.2)
    data.qvel[4] = Scalar[DTYPE](-0.1)
    data.qvel[5] = Scalar[DTYPE](0.05)
    data.qfrc[3] = Scalar[DTYPE](50.0)
    data.qfrc[4] = Scalar[DTYPE](-30.0)
    data.qfrc[5] = Scalar[DTYPE](20.0)

    # =========================================================================
    # Run benchmarks at multiple batch sizes
    # =========================================================================
    run_benchmark[256](ctx, model_host, data)
    run_benchmark[1024](ctx, model_host, data)
    run_benchmark[4096](ctx, model_host, data)

    print()
    print("Done.")
