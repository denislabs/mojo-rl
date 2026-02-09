"""Benchmark: HalfCheetah vs Hopper GPU physics to measure scaling impact.

Compares the physics step kernel at two model complexities to understand
register pressure and O(N²) scaling effects with InlineArrays.

Each iteration runs STEPS (100) consecutive physics steps to simulate a
realistic rollout, with state reset between iterations.

HalfCheetah: NBODY=8, NJOINT=10, NQ=10, NV=10, MAX_CONTACTS=20
  → InlineArray register pressure: ~1654 floats/thread (with Newton)
Hopper:      NBODY=4, NJOINT=6,  NQ=6,  NV=6,  MAX_CONTACTS=10
  → InlineArray register pressure: ~664 floats/thread (with Newton)

Run with:
    cd mojo-rl
    pixi run -e apple mojo run physics3d/tests/benchmark_cheetah_vs_hopper.mojo
"""

from time import perf_counter_ns
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data, _max_one, compute_capsule_inertia
from physics3d.constants import GEOM_CAPSULE
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from physics3d.solver.pgs_solver import PGSSolver
from physics3d.solver.cg_solver import CGSolver
from physics3d.solver.newton_solver import NewtonSolver
from physics3d.integrator.euler_integrator import EulerIntegrator

from physics3d.gpu.constants import (
    TPB,
    state_size,
    model_size,
    integrator_workspace_size,
)
from physics3d.gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_data_to_buffer,
)


# Benchmark config
comptime WARMUP: Int = 10
comptime ITERS: Int = 50
comptime STEPS: Int = 100  # Physics steps per iteration (simulates a rollout)
comptime BATCH: Int = 256


# =============================================================================
# Hopper dimensions
# =============================================================================
comptime H_NQ: Int = 6
comptime H_NV: Int = 6
comptime H_NBODY: Int = 4
comptime H_NJOINT: Int = 6
comptime H_MAX_CONTACTS: Int = 10
comptime H_STATE_SIZE = state_size[H_NQ, H_NV, H_NBODY, H_MAX_CONTACTS]()
comptime H_MODEL_SIZE = model_size[H_NBODY, H_NJOINT]()
# Use Newton (largest) workspace size since all 3 solvers share the buffer
comptime H_WS_SIZE = integrator_workspace_size[H_NV, H_NBODY]() + H_NV * H_NV + NewtonSolver.solver_workspace_size[H_NV, H_MAX_CONTACTS]()


# =============================================================================
# HalfCheetah dimensions
# =============================================================================
comptime C_NQ: Int = 10
comptime C_NV: Int = 10
comptime C_NBODY: Int = 8
comptime C_NJOINT: Int = 10
comptime C_MAX_CONTACTS: Int = 20
comptime C_STATE_SIZE = state_size[C_NQ, C_NV, C_NBODY, C_MAX_CONTACTS]()
comptime C_MODEL_SIZE = model_size[C_NBODY, C_NJOINT]()
comptime C_WS_SIZE = integrator_workspace_size[C_NV, C_NBODY]() + C_NV * C_NV + NewtonSolver.solver_workspace_size[C_NV, C_MAX_CONTACTS]()


comptime DTYPE = DType.float32


# =============================================================================
# Solver step kernels (parametric)
# =============================================================================

@always_inline
fn hopper_pgs_kernel(
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, H_STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, H_MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(BATCH, H_WS_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    EulerIntegrator[PGSSolver].step_constraint_kernel[
        DTYPE, H_NQ, H_NV, H_NBODY, H_NJOINT, H_MAX_CONTACTS,
        H_STATE_SIZE, H_MODEL_SIZE, BATCH, H_WS_SIZE,
    ](env, state, model, workspace)

@always_inline
fn hopper_cg_kernel(
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, H_STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, H_MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(BATCH, H_WS_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    EulerIntegrator[CGSolver].step_constraint_kernel[
        DTYPE, H_NQ, H_NV, H_NBODY, H_NJOINT, H_MAX_CONTACTS,
        H_STATE_SIZE, H_MODEL_SIZE, BATCH, H_WS_SIZE,
    ](env, state, model, workspace)

@always_inline
fn hopper_newton_kernel(
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, H_STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, H_MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(BATCH, H_WS_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    EulerIntegrator[NewtonSolver].step_constraint_kernel[
        DTYPE, H_NQ, H_NV, H_NBODY, H_NJOINT, H_MAX_CONTACTS,
        H_STATE_SIZE, H_MODEL_SIZE, BATCH, H_WS_SIZE,
    ](env, state, model, workspace)


@always_inline
fn cheetah_pgs_kernel(
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, C_STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, C_MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(BATCH, C_WS_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    EulerIntegrator[PGSSolver].step_constraint_kernel[
        DTYPE, C_NQ, C_NV, C_NBODY, C_NJOINT, C_MAX_CONTACTS,
        C_STATE_SIZE, C_MODEL_SIZE, BATCH, C_WS_SIZE,
    ](env, state, model, workspace)

@always_inline
fn cheetah_cg_kernel(
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, C_STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, C_MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(BATCH, C_WS_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    EulerIntegrator[CGSolver].step_constraint_kernel[
        DTYPE, C_NQ, C_NV, C_NBODY, C_NJOINT, C_MAX_CONTACTS,
        C_STATE_SIZE, C_MODEL_SIZE, BATCH, C_WS_SIZE,
    ](env, state, model, workspace)

@always_inline
fn cheetah_newton_kernel(
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, C_STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, C_MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(BATCH, C_WS_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    EulerIntegrator[NewtonSolver].step_constraint_kernel[
        DTYPE, C_NQ, C_NV, C_NBODY, C_NJOINT, C_MAX_CONTACTS,
        C_STATE_SIZE, C_MODEL_SIZE, BATCH, C_WS_SIZE,
    ](env, state, model, workspace)


# =============================================================================
# Hopper model setup (reused from existing benchmark)
# =============================================================================

fn setup_hopper_model(mut model: Model[DTYPE, H_NQ, H_NV, H_NBODY, H_NJOINT, H_MAX_CONTACTS]):
    var torso_mass = Scalar[DTYPE](3.53)
    var torso_r = Scalar[DTYPE](0.05)
    var torso_h = Scalar[DTYPE](0.2)
    var thigh_mass = Scalar[DTYPE](3.93)
    var thigh_r = Scalar[DTYPE](0.05)
    var thigh_h = Scalar[DTYPE](0.225)
    var leg_mass = Scalar[DTYPE](2.71)
    var leg_r = Scalar[DTYPE](0.04)
    var leg_h = Scalar[DTYPE](0.25)
    var foot_mass = Scalar[DTYPE](5.09)
    var foot_r = Scalar[DTYPE](0.06)
    var foot_h = Scalar[DTYPE](0.195)

    var ti = compute_capsule_inertia(torso_mass, torso_r, torso_h)
    model.set_body(0, mass=torso_mass, inertia=ti, radius=torso_r)
    model.set_body_parent(0, -1)
    model.body_geom_type[0] = GEOM_CAPSULE
    model.body_half_length[0] = torso_h

    var thi = compute_capsule_inertia(thigh_mass, thigh_r, thigh_h)
    model.set_body(1, mass=thigh_mass, inertia=thi, radius=thigh_r)
    model.set_body_parent(1, 0)
    model.body_geom_type[1] = GEOM_CAPSULE
    model.body_half_length[1] = thigh_h
    model.set_body_local_frame(1, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -(torso_h + thigh_h)))

    var li = compute_capsule_inertia(leg_mass, leg_r, leg_h)
    model.set_body(2, mass=leg_mass, inertia=li, radius=leg_r)
    model.set_body_parent(2, 1)
    model.body_geom_type[2] = GEOM_CAPSULE
    model.body_half_length[2] = leg_h
    model.set_body_local_frame(2, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -(thigh_h + leg_h)))

    var fi = compute_capsule_inertia(foot_mass, foot_r, foot_h)
    model.set_body(3, mass=foot_mass, inertia=fi, radius=foot_r)
    model.set_body_parent(3, 2)
    model.body_geom_type[3] = GEOM_CAPSULE
    model.body_half_length[3] = foot_h
    model.set_body_local_frame(
        3, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -leg_h),
        quat=(Scalar[DTYPE](0), Scalar[DTYPE](0.707), Scalar[DTYPE](0), Scalar[DTYPE](0.707)),
    )

    _ = model.add_slide_joint(body_id=0, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0)),
        axis=(Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0)), force_limit=Scalar[DTYPE](0))
    _ = model.add_slide_joint(body_id=0, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0)),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1)), force_limit=Scalar[DTYPE](0))
    _ = model.add_hinge_joint(body_id=0, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0)),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](0))
    _ = model.add_hinge_joint(body_id=1, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -torso_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](200),
        range_min=Scalar[DTYPE](-2.618), range_max=Scalar[DTYPE](0),
        armature=Scalar[DTYPE](1), damping=Scalar[DTYPE](1))
    _ = model.add_hinge_joint(body_id=2, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -thigh_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](200),
        range_min=Scalar[DTYPE](-2.618), range_max=Scalar[DTYPE](0),
        armature=Scalar[DTYPE](1), damping=Scalar[DTYPE](1))
    _ = model.add_hinge_joint(body_id=3, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -leg_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](200),
        range_min=Scalar[DTYPE](-0.785), range_max=Scalar[DTYPE](0.785),
        armature=Scalar[DTYPE](1), damping=Scalar[DTYPE](1))


# =============================================================================
# HalfCheetah model setup (8 bodies, 10 joints)
# =============================================================================

fn setup_cheetah_model(mut model: Model[DTYPE, C_NQ, C_NV, C_NBODY, C_NJOINT, C_MAX_CONTACTS]):
    var r = Scalar[DTYPE](0.046)  # capsule radius

    # Body 0: Torso
    var torso_h = Scalar[DTYPE](0.5)
    var torso_m = Scalar[DTYPE](6.36)
    var ti = compute_capsule_inertia(torso_m, r, torso_h)
    model.set_body(0, mass=torso_m, inertia=ti, radius=r)
    model.set_body_parent(0, -1)
    model.body_geom_type[0] = GEOM_CAPSULE
    model.body_half_length[0] = torso_h

    # Body 1: Back Thigh
    var bth_h = Scalar[DTYPE](0.145)
    var bth_m = Scalar[DTYPE](4.11)
    var bthi = compute_capsule_inertia(bth_m, r, bth_h)
    model.set_body(1, mass=bth_m, inertia=bthi, radius=r)
    model.set_body_parent(1, 0)
    model.body_geom_type[1] = GEOM_CAPSULE
    model.body_half_length[1] = bth_h
    model.set_body_local_frame(1, pos=(bth_h, Scalar[DTYPE](0), -torso_h))

    # Body 2: Back Shin
    var bsh_h = Scalar[DTYPE](0.15)
    var bsh_m = Scalar[DTYPE](2.78)
    var bshi = compute_capsule_inertia(bsh_m, r, bsh_h)
    model.set_body(2, mass=bsh_m, inertia=bshi, radius=r)
    model.set_body_parent(2, 1)
    model.body_geom_type[2] = GEOM_CAPSULE
    model.body_half_length[2] = bsh_h
    model.set_body_local_frame(2, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -(bth_h + bsh_h)))

    # Body 3: Back Foot
    var bf_h = Scalar[DTYPE](0.094)
    var bf_m = Scalar[DTYPE](1.80)
    var bfi = compute_capsule_inertia(bf_m, r, bf_h)
    model.set_body(3, mass=bf_m, inertia=bfi, radius=r)
    model.set_body_parent(3, 2)
    model.body_geom_type[3] = GEOM_CAPSULE
    model.body_half_length[3] = bf_h
    model.set_body_local_frame(3, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -bsh_h),
        quat=(Scalar[DTYPE](0), Scalar[DTYPE](0.707), Scalar[DTYPE](0), Scalar[DTYPE](0.707)))

    # Body 4: Front Thigh
    var fth_h = Scalar[DTYPE](0.133)
    var fth_m = Scalar[DTYPE](4.11)
    var fthi = compute_capsule_inertia(fth_m, r, fth_h)
    model.set_body(4, mass=fth_m, inertia=fthi, radius=r)
    model.set_body_parent(4, 0)
    model.body_geom_type[4] = GEOM_CAPSULE
    model.body_half_length[4] = fth_h
    model.set_body_local_frame(4, pos=(fth_h, Scalar[DTYPE](0), torso_h))

    # Body 5: Front Shin
    var fsh_h = Scalar[DTYPE](0.106)
    var fsh_m = Scalar[DTYPE](2.78)
    var fshi = compute_capsule_inertia(fsh_m, r, fsh_h)
    model.set_body(5, mass=fsh_m, inertia=fshi, radius=r)
    model.set_body_parent(5, 4)
    model.body_geom_type[5] = GEOM_CAPSULE
    model.body_half_length[5] = fsh_h
    model.set_body_local_frame(5, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -(fth_h + fsh_h)))

    # Body 6: Front Foot
    var ff_h = Scalar[DTYPE](0.07)
    var ff_m = Scalar[DTYPE](1.80)
    var ffi = compute_capsule_inertia(ff_m, r, ff_h)
    model.set_body(6, mass=ff_m, inertia=ffi, radius=r)
    model.set_body_parent(6, 5)
    model.body_geom_type[6] = GEOM_CAPSULE
    model.body_half_length[6] = ff_h
    model.set_body_local_frame(6, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -fsh_h),
        quat=(Scalar[DTYPE](0), Scalar[DTYPE](0.707), Scalar[DTYPE](0), Scalar[DTYPE](0.707)))

    # Body 7: Head
    var hd_h = Scalar[DTYPE](0.117)
    var hd_m = Scalar[DTYPE](3.68)
    var hdi = compute_capsule_inertia(hd_m, r, hd_h)
    model.set_body(7, mass=hd_m, inertia=hdi, radius=r)
    model.set_body_parent(7, 0)
    model.body_geom_type[7] = GEOM_CAPSULE
    model.body_half_length[7] = hd_h
    model.set_body_local_frame(7, pos=(Scalar[DTYPE](-0.1), Scalar[DTYPE](0), Scalar[DTYPE](0.6)))

    # Joint 0: RootX (slide)
    _ = model.add_slide_joint(body_id=0, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0)),
        axis=(Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0)), force_limit=Scalar[DTYPE](0))
    # Joint 1: RootZ (slide)
    _ = model.add_slide_joint(body_id=0, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0)),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1)), force_limit=Scalar[DTYPE](0))
    # Joint 2: RootY (hinge)
    _ = model.add_hinge_joint(body_id=0, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0)),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](0))
    # Joint 3: BThigh
    _ = model.add_hinge_joint(body_id=1, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -torso_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](120),
        range_min=Scalar[DTYPE](-0.52), range_max=Scalar[DTYPE](1.05),
        armature=Scalar[DTYPE](0.1), damping=Scalar[DTYPE](6), stiffness=Scalar[DTYPE](240))
    # Joint 4: BShin
    _ = model.add_hinge_joint(body_id=2, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -bth_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](90),
        range_min=Scalar[DTYPE](-0.785), range_max=Scalar[DTYPE](0.785),
        armature=Scalar[DTYPE](0.1), damping=Scalar[DTYPE](4.5), stiffness=Scalar[DTYPE](180))
    # Joint 5: BFoot
    _ = model.add_hinge_joint(body_id=3, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -bsh_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](60),
        range_min=Scalar[DTYPE](-0.4), range_max=Scalar[DTYPE](0.785),
        armature=Scalar[DTYPE](0.1), damping=Scalar[DTYPE](3), stiffness=Scalar[DTYPE](120))
    # Joint 6: FThigh
    _ = model.add_hinge_joint(body_id=4, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), torso_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](120),
        range_min=Scalar[DTYPE](-1.0), range_max=Scalar[DTYPE](0.7),
        armature=Scalar[DTYPE](0.1), damping=Scalar[DTYPE](4.5), stiffness=Scalar[DTYPE](180))
    # Joint 7: FShin
    _ = model.add_hinge_joint(body_id=5, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -fth_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](60),
        range_min=Scalar[DTYPE](-1.2), range_max=Scalar[DTYPE](0.87),
        armature=Scalar[DTYPE](0.1), damping=Scalar[DTYPE](3), stiffness=Scalar[DTYPE](120))
    # Joint 8: FFoot
    _ = model.add_hinge_joint(body_id=6, pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), -fsh_h),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](30),
        range_min=Scalar[DTYPE](-0.5), range_max=Scalar[DTYPE](0.5),
        armature=Scalar[DTYPE](0.1), damping=Scalar[DTYPE](1.5), stiffness=Scalar[DTYPE](60))
    # Joint 9: Head (fixed via tight limits)
    _ = model.add_hinge_joint(body_id=7, pos=(Scalar[DTYPE](-0.1), Scalar[DTYPE](0), Scalar[DTYPE](0.6)),
        axis=(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0)), tau_limit=Scalar[DTYPE](0),
        range_min=Scalar[DTYPE](-0.001), range_max=Scalar[DTYPE](0.001),
        armature=Scalar[DTYPE](0.1), damping=Scalar[DTYPE](0.01), stiffness=Scalar[DTYPE](8))


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    print("=" * 70)
    print("Physics3D: HalfCheetah vs Hopper GPU Benchmark")
    print("=" * 70)
    print()
    print("Purpose: Measure register pressure / O(N²) scaling impact")
    print()
    print("Hopper:      NBODY=4, NJOINT=6,  NQ=6,  NV=6,  MAX_CONTACTS=10")
    print("  STATE_SIZE:", H_STATE_SIZE, " MODEL_SIZE:", H_MODEL_SIZE)
    print("  M_SIZE=36,  InlineArray floats/thread: ~664 (with Newton)")
    print()
    print("HalfCheetah: NBODY=8, NJOINT=10, NQ=10, NV=10, MAX_CONTACTS=20")
    print("  STATE_SIZE:", C_STATE_SIZE, " MODEL_SIZE:", C_MODEL_SIZE)
    print("  M_SIZE=100, InlineArray floats/thread: ~1654 (with Newton)")
    print()
    print("BATCH:", BATCH, " Warmup:", WARMUP, " Iters:", ITERS, " Steps/iter:", STEPS)
    print()

    var ctx = DeviceContext()

    # =========================================================================
    # Setup Hopper
    # =========================================================================
    var hopper = Model[DTYPE, H_NQ, H_NV, H_NBODY, H_NJOINT, H_MAX_CONTACTS](
        gravity_z=Scalar[DTYPE](-9.81), timestep=Scalar[DTYPE](0.002),
        ground_z=Scalar[DTYPE](0.0), friction=Scalar[DTYPE](0.5))
    setup_hopper_model(hopper)
    var hopper_data = Data[DTYPE, H_NQ, H_NV, H_NBODY, H_NJOINT, H_MAX_CONTACTS]()
    hopper_data.qpos[0] = Scalar[DTYPE](0.0)
    hopper_data.qpos[1] = Scalar[DTYPE](1.05)
    hopper_data.qpos[2] = Scalar[DTYPE](0.1)
    hopper_data.qpos[3] = Scalar[DTYPE](-0.2)
    hopper_data.qpos[4] = Scalar[DTYPE](-0.1)
    hopper_data.qpos[5] = Scalar[DTYPE](0.05)
    hopper_data.qfrc[3] = Scalar[DTYPE](50.0)
    hopper_data.qfrc[4] = Scalar[DTYPE](-30.0)
    hopper_data.qfrc[5] = Scalar[DTYPE](20.0)

    var h_model_host = ctx.enqueue_create_host_buffer[DTYPE](H_MODEL_SIZE)
    for i in range(H_MODEL_SIZE):
        h_model_host[i] = Scalar[DTYPE](0)
    copy_model_to_buffer[DTYPE, H_NQ, H_NV, H_NBODY, H_NJOINT, H_MAX_CONTACTS](hopper, h_model_host)

    var h_state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * H_STATE_SIZE)
    for i in range(BATCH * H_STATE_SIZE):
        h_state_host[i] = Scalar[DTYPE](0)
    for b in range(BATCH):
        copy_data_to_buffer[DTYPE, H_NQ, H_NV, H_NBODY, H_NJOINT, H_MAX_CONTACTS](hopper_data, h_state_host, b)

    var h_state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * H_STATE_SIZE)
    var h_model_buf = ctx.enqueue_create_buffer[DTYPE](H_MODEL_SIZE)
    ctx.enqueue_copy(h_model_buf, h_model_host.unsafe_ptr())
    ctx.synchronize()

    var h_st = LayoutTensor[DTYPE, Layout.row_major(BATCH, H_STATE_SIZE), MutAnyOrigin](h_state_buf.unsafe_ptr())
    var h_md = LayoutTensor[DTYPE, Layout.row_major(1, H_MODEL_SIZE), MutAnyOrigin](h_model_buf.unsafe_ptr())
    var h_ws_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * H_WS_SIZE)
    var h_ws = LayoutTensor[DTYPE, Layout.row_major(BATCH, H_WS_SIZE), MutAnyOrigin](h_ws_buf.unsafe_ptr())

    # =========================================================================
    # Setup HalfCheetah
    # =========================================================================
    var cheetah = Model[DTYPE, C_NQ, C_NV, C_NBODY, C_NJOINT, C_MAX_CONTACTS](
        gravity_z=Scalar[DTYPE](-9.81), timestep=Scalar[DTYPE](0.002),
        ground_z=Scalar[DTYPE](0.0), friction=Scalar[DTYPE](1.0))
    setup_cheetah_model(cheetah)
    var cheetah_data = Data[DTYPE, C_NQ, C_NV, C_NBODY, C_NJOINT, C_MAX_CONTACTS]()
    cheetah_data.qpos[0] = Scalar[DTYPE](0.0)   # rootx
    cheetah_data.qpos[1] = Scalar[DTYPE](0.7)    # rootz
    cheetah_data.qpos[2] = Scalar[DTYPE](0.05)   # rooty angle
    cheetah_data.qpos[3] = Scalar[DTYPE](0.1)
    cheetah_data.qpos[4] = Scalar[DTYPE](-0.1)
    cheetah_data.qpos[5] = Scalar[DTYPE](0.05)
    cheetah_data.qpos[6] = Scalar[DTYPE](-0.1)
    cheetah_data.qpos[7] = Scalar[DTYPE](0.1)
    cheetah_data.qpos[8] = Scalar[DTYPE](-0.05)
    cheetah_data.qpos[9] = Scalar[DTYPE](0.0)
    cheetah_data.qfrc[3] = Scalar[DTYPE](50.0)
    cheetah_data.qfrc[4] = Scalar[DTYPE](-30.0)
    cheetah_data.qfrc[5] = Scalar[DTYPE](20.0)
    cheetah_data.qfrc[6] = Scalar[DTYPE](40.0)
    cheetah_data.qfrc[7] = Scalar[DTYPE](-20.0)
    cheetah_data.qfrc[8] = Scalar[DTYPE](10.0)

    var c_model_host = ctx.enqueue_create_host_buffer[DTYPE](C_MODEL_SIZE)
    for i in range(C_MODEL_SIZE):
        c_model_host[i] = Scalar[DTYPE](0)
    copy_model_to_buffer[DTYPE, C_NQ, C_NV, C_NBODY, C_NJOINT, C_MAX_CONTACTS](cheetah, c_model_host)

    var c_state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * C_STATE_SIZE)
    for i in range(BATCH * C_STATE_SIZE):
        c_state_host[i] = Scalar[DTYPE](0)
    for b in range(BATCH):
        copy_data_to_buffer[DTYPE, C_NQ, C_NV, C_NBODY, C_NJOINT, C_MAX_CONTACTS](cheetah_data, c_state_host, b)

    var c_state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * C_STATE_SIZE)
    var c_model_buf = ctx.enqueue_create_buffer[DTYPE](C_MODEL_SIZE)
    ctx.enqueue_copy(c_model_buf, c_model_host.unsafe_ptr())
    ctx.synchronize()

    var c_st = LayoutTensor[DTYPE, Layout.row_major(BATCH, C_STATE_SIZE), MutAnyOrigin](c_state_buf.unsafe_ptr())
    var c_md = LayoutTensor[DTYPE, Layout.row_major(1, C_MODEL_SIZE), MutAnyOrigin](c_model_buf.unsafe_ptr())
    var c_ws_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * C_WS_SIZE)
    var c_ws = LayoutTensor[DTYPE, Layout.row_major(BATCH, C_WS_SIZE), MutAnyOrigin](c_ws_buf.unsafe_ptr())

    comptime BLOCKS = (BATCH + TPB - 1) // TPB

    # =========================================================================
    # Benchmark Hopper
    # =========================================================================
    print("=" * 70)
    print("HOPPER (NV=6, M_SIZE=36, MAX_CONTACTS=10)")
    print("=" * 70)

    # --- PGS ---
    ctx.enqueue_copy(h_state_buf, h_state_host.unsafe_ptr())
    ctx.synchronize()
    for _ in range(WARMUP):
        for _ in range(STEPS):
            ctx.enqueue_function[hopper_pgs_kernel, hopper_pgs_kernel](
                h_st, h_md, h_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_copy(h_state_buf, h_state_host.unsafe_ptr())
        for _ in range(STEPS):
            ctx.enqueue_function[hopper_pgs_kernel, hopper_pgs_kernel](
                h_st, h_md, h_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    var h_pgs_us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    print("  PGS:     ", h_pgs_us, " us/rollout (", h_pgs_us / STEPS, " us/step)")

    # --- CG ---
    ctx.enqueue_copy(h_state_buf, h_state_host.unsafe_ptr())
    ctx.synchronize()
    for _ in range(WARMUP):
        for _ in range(STEPS):
            ctx.enqueue_function[hopper_cg_kernel, hopper_cg_kernel](
                h_st, h_md, h_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_copy(h_state_buf, h_state_host.unsafe_ptr())
        for _ in range(STEPS):
            ctx.enqueue_function[hopper_cg_kernel, hopper_cg_kernel](
                h_st, h_md, h_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    var h_cg_us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    print("  CG:      ", h_cg_us, " us/rollout (", h_cg_us / STEPS, " us/step)")

    # --- Newton ---
    ctx.enqueue_copy(h_state_buf, h_state_host.unsafe_ptr())
    ctx.synchronize()
    for _ in range(WARMUP):
        for _ in range(STEPS):
            ctx.enqueue_function[hopper_newton_kernel, hopper_newton_kernel](
                h_st, h_md, h_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_copy(h_state_buf, h_state_host.unsafe_ptr())
        for _ in range(STEPS):
            ctx.enqueue_function[hopper_newton_kernel, hopper_newton_kernel](
                h_st, h_md, h_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    var h_newton_us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    print("  Newton:  ", h_newton_us, " us/rollout (", h_newton_us / STEPS, " us/step)")

    # =========================================================================
    # Benchmark HalfCheetah
    # =========================================================================
    print()
    print("=" * 70)
    print("HALF CHEETAH (NV=10, M_SIZE=100, MAX_CONTACTS=20)")
    print("=" * 70)

    # --- PGS ---
    ctx.enqueue_copy(c_state_buf, c_state_host.unsafe_ptr())
    ctx.synchronize()
    for _ in range(WARMUP):
        for _ in range(STEPS):
            ctx.enqueue_function[cheetah_pgs_kernel, cheetah_pgs_kernel](
                c_st, c_md, c_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_copy(c_state_buf, c_state_host.unsafe_ptr())
        for _ in range(STEPS):
            ctx.enqueue_function[cheetah_pgs_kernel, cheetah_pgs_kernel](
                c_st, c_md, c_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    var c_pgs_us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    print("  PGS:     ", c_pgs_us, " us/rollout (", c_pgs_us / STEPS, " us/step)")

    # --- CG ---
    ctx.enqueue_copy(c_state_buf, c_state_host.unsafe_ptr())
    ctx.synchronize()
    for _ in range(WARMUP):
        for _ in range(STEPS):
            ctx.enqueue_function[cheetah_cg_kernel, cheetah_cg_kernel](
                c_st, c_md, c_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_copy(c_state_buf, c_state_host.unsafe_ptr())
        for _ in range(STEPS):
            ctx.enqueue_function[cheetah_cg_kernel, cheetah_cg_kernel](
                c_st, c_md, c_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    var c_cg_us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    print("  CG:      ", c_cg_us, " us/rollout (", c_cg_us / STEPS, " us/step)")

    # --- Newton ---
    ctx.enqueue_copy(c_state_buf, c_state_host.unsafe_ptr())
    ctx.synchronize()
    for _ in range(WARMUP):
        for _ in range(STEPS):
            ctx.enqueue_function[cheetah_newton_kernel, cheetah_newton_kernel](
                c_st, c_md, c_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_copy(c_state_buf, c_state_host.unsafe_ptr())
        for _ in range(STEPS):
            ctx.enqueue_function[cheetah_newton_kernel, cheetah_newton_kernel](
                c_st, c_md, c_ws, grid_dim=(BLOCKS,), block_dim=(TPB,))
    ctx.synchronize()
    var c_newton_us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    print("  Newton:  ", c_newton_us, " us/rollout (", c_newton_us / STEPS, " us/step)")

    # =========================================================================
    # Comparison
    # =========================================================================
    # Per-step averages (rollout time / STEPS)
    var h_pgs_step = h_pgs_us / STEPS
    var h_cg_step = h_cg_us / STEPS
    var h_newton_step = h_newton_us / STEPS
    var c_pgs_step = c_pgs_us / STEPS
    var c_cg_step = c_cg_us / STEPS
    var c_newton_step = c_newton_us / STEPS

    print()
    print("=" * 70)
    print("SCALING: HalfCheetah / Hopper (per-step)")
    print("=" * 70)
    if h_pgs_step > 0:
        print("  PGS:    ", c_pgs_step / h_pgs_step, "x slower")
    if h_cg_step > 0:
        print("  CG:     ", c_cg_step / h_cg_step, "x slower")
    if h_newton_step > 0:
        print("  Newton: ", c_newton_step / h_newton_step, "x slower")

    print()
    print("Per-env per-step time (us/env/step):")
    print("  Hopper  PGS:", h_pgs_step / BATCH, "  CG:", h_cg_step / BATCH, "  Newton:", h_newton_step / BATCH)
    print("  Cheetah PGS:", c_pgs_step / BATCH, "  CG:", c_cg_step / BATCH, "  Newton:", c_newton_step / BATCH)

    print()
    print("Rollout time (", STEPS, " steps × ", BATCH, " envs):")
    print("  Hopper  PGS:", h_pgs_us / 1000.0, " ms  CG:", h_cg_us / 1000.0, " ms  Newton:", h_newton_us / 1000.0, " ms")
    print("  Cheetah PGS:", c_pgs_us / 1000.0, " ms  CG:", c_cg_us / 1000.0, " ms  Newton:", c_newton_us / 1000.0, " ms")

    print()
    print("Expected scaling (O(N²) of NV):", Float64(C_NV * C_NV) / Float64(H_NV * H_NV), "x")
    print("Expected scaling (O(N³) of NV):", Float64(C_NV * C_NV * C_NV) / Float64(H_NV * H_NV * H_NV), "x")

    # Training estimate: 512 steps/env × 256 envs = 131072 total steps per rollout
    # Each kernel dispatch handles all BATCH envs in parallel → 512 dispatches
    comptime ROLLOUT_STEPS: Int = 512
    print()
    print("Training estimate (", ROLLOUT_STEPS, " steps/env × ", BATCH, " envs = ", ROLLOUT_STEPS * BATCH, " total steps/rollout):")
    print("  Hopper  PGS:   ", h_pgs_step * ROLLOUT_STEPS / 1e6, "s/rollout")
    print("  Hopper  Newton:", h_newton_step * ROLLOUT_STEPS / 1e6, "s/rollout")
    print("  Cheetah PGS:   ", c_pgs_step * ROLLOUT_STEPS / 1e6, "s/rollout")
    print("  Cheetah Newton:", c_newton_step * ROLLOUT_STEPS / 1e6, "s/rollout")

    print()
    print("Done.")
