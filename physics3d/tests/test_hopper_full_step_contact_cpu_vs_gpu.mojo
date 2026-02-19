"""Test Full Step with Contacts: CPU vs GPU for Hopper.

Compares qpos/qvel after running physics steps with ground contacts on CPU vs GPU
for the Hopper model (ELLIPTIC cone, condim=1 frictionless).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_hopper_full_step_contact_cpu_vs_gpu.mojo
"""

from math import abs
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data, ConeType
from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver import NewtonSolver
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    integrator_workspace_size,
)
from physics3d.gpu.buffer_utils import (
    create_state_buffer,
)
from envs.hopper.hopper_def import (
    HopperModel,
    HopperParams,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HopperModel.NQ  # 6
comptime NV = HopperModel.NV  # 6
comptime NBODY = HopperModel.NBODY  # 5
comptime NJOINT = HopperModel.NJOINT  # 6
comptime NGEOM = HopperModel.NGEOM  # 5
comptime MAX_CONTACTS = HopperParams[DTYPE].MAX_CONTACTS  # 20
comptime ACTION_DIM = HopperParams[DTYPE].ACTION_DIM  # 3
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime WS_SIZE = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

# Tolerances (float32)
comptime QPOS_ABS_TOL: Float64 = 3e-2
comptime QPOS_REL_TOL: Float64 = 2e-1
comptime QVEL_ABS_TOL: Float64 = 5e-1
comptime QVEL_REL_TOL: Float64 = 3e-1


# =============================================================================
# Compare helper
# =============================================================================


fn compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int,
    ctx: DeviceContext,
    mut model_buf: DeviceBuffer[DTYPE],
    mut state_host: HostBuffer[DTYPE],
    mut state_buf: DeviceBuffer[DTYPE],
    mut workspace_buf: DeviceBuffer[DTYPE],
    mut ws_host: HostBuffer[DTYPE],
) raises -> Bool:
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === CPU pipeline ===
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        0,
        HopperModel.CONE_TYPE,
    ]()
    var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    HopperModel.setup_model_and_data(model_cpu, data_cpu)

    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data_cpu.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    for _ in range(num_steps):
        for i in range(NV):
            data_cpu.qfrc[i] = Scalar[DTYPE](0)
        HopperModel.apply_actions(data_cpu, action_list)
        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
            model_cpu, data_cpu
        )

    # === GPU pipeline ===
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](qvel_init[i])

    var data_temp = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    HopperModel.apply_actions(data_temp, action_list)
    for i in range(NV):
        state_host[qfrc_offset[NQ, NV]() + i] = data_temp.qfrc[i]

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())

    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    for step in range(num_steps):
        if step > 0:
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()
            for i in range(NV):
                state_host[qfrc_offset[NQ, NV]() + i] = data_temp.qfrc[i]
            ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
            for i in range(BATCH * WS_SIZE):
                ws_host[i] = Scalar[DTYPE](0)
            ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
            ctx.synchronize()

        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM=NGEOM,
            CONE_TYPE = HopperModel.CONE_TYPE,
        ](
            ctx,
            state_buf,
            model_buf,
            workspace_buf,
        )
        ctx.synchronize()

    # Read back GPU results
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # === Compare qpos ===
    var qpos_pass = True
    var qpos_max_abs: Float64 = 0.0
    var qpos_max_rel: Float64 = 0.0
    var qpos_fails = 0

    for i in range(NQ):
        var cpu_val = Float64(data_cpu.qpos[i])
        var gpu_val = Float64(state_host[qpos_offset[NQ, NV]() + i])
        var abs_err = abs(cpu_val - gpu_val)
        var ref_mag = abs(cpu_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > qpos_max_abs:
            qpos_max_abs = abs_err
        if rel_err > qpos_max_rel:
            qpos_max_rel = rel_err

        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            if qpos_fails < 5:
                print(
                    "  FAIL qpos[",
                    i,
                    "]",
                    " cpu=",
                    cpu_val,
                    " gpu=",
                    gpu_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qpos_fails += 1
            qpos_pass = False

    # === Compare qvel ===
    var qvel_pass = True
    var qvel_max_abs: Float64 = 0.0
    var qvel_max_rel: Float64 = 0.0
    var qvel_fails = 0

    for i in range(NV):
        var cpu_val = Float64(data_cpu.qvel[i])
        var gpu_val = Float64(state_host[qvel_offset[NQ, NV]() + i])
        var abs_err = abs(cpu_val - gpu_val)
        var ref_mag = abs(cpu_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > qvel_max_abs:
            qvel_max_abs = abs_err
        if rel_err > qvel_max_rel:
            qvel_max_rel = rel_err

        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            if qvel_fails < 5:
                print(
                    "  FAIL qvel[",
                    i,
                    "]",
                    " cpu=",
                    cpu_val,
                    " gpu=",
                    gpu_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass
    if all_pass:
        print(
            "  ALL OK  qpos(abs=",
            qpos_max_abs,
            " rel=",
            qpos_max_rel,
            ") qvel(abs=",
            qvel_max_abs,
            " rel=",
            qvel_max_rel,
            ")",
        )
    else:
        print(
            "  FAILED  qpos:",
            qpos_fails,
            "fails (abs=",
            qpos_max_abs,
            " rel=",
            qpos_max_rel,
            ")",
            " qvel:",
            qvel_fails,
            "fails (abs=",
            qvel_max_abs,
            " rel=",
            qvel_max_rel,
            ")",
        )

    # Print values
    print("  CPU qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(data_cpu.qpos[i]), end="")
    print()
    print("  GPU qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(state_host[qpos_offset[NQ, NV]() + i]), end="")
    print()
    print("  CPU qvel:", end="")
    for i in range(NV):
        print(" ", Float64(data_cpu.qvel[i]), end="")
    print()
    print("  GPU qvel:", end="")
    for i in range(NV):
        print(" ", Float64(state_host[qvel_offset[NQ, NV]() + i]), end="")
    print()
    print("  CPU contacts:", Int(data_cpu.num_contacts))

    return all_pass


fn main() raises:
    print("=" * 60)
    print("Full Step with Contacts: CPU vs GPU (Hopper)")
    print("=" * 60)
    print("Model: Hopper (NQ=6, NV=6, NGEOM=", NGEOM, ")")
    print("Integrator: Euler + NewtonSolver (elliptic)")
    print("Precision: float32")
    print("Tolerances: qpos abs=", QPOS_ABS_TOL, " rel=", QPOS_REL_TOL)
    print("            qvel abs=", QVEL_ABS_TOL, " rel=", QVEL_REL_TOL)
    print()

    var ctx = DeviceContext()
    print("GPU device initialized")

    # Create GPU model buffer
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HopperModel.init_model_gpu(ctx, model_buf)
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

    var num_pass = 0
    var num_fail = 0

    # --- Config 1: Ground contact, no actions (1 step) ---
    # Hopper: torso at 1.25, foot ~0.6m below. rootz=-0.8 pushes foot to ground.
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    qpos1[1] = -0.8  # rootz low => contacts
    var qvel1 = InlineArray[Float64, NV](fill=0.0)
    var act1 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    if compare_step(
        "Ground contact (1 step)",
        qpos1,
        qvel1,
        act1,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 2: Ground contact with actions (1 step) ---
    var act2 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act2[0] = 0.5  # thigh
    act2[1] = -0.3  # leg
    act2[2] = 0.2  # foot
    if compare_step(
        "Ground contact + actions (1 step)",
        qpos1,
        qvel1,
        act2,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 3: Deep penetration (1 step) ---
    var qpos3 = InlineArray[Float64, NQ](fill=0.0)
    qpos3[1] = -1.1  # very low
    if compare_step(
        "Deep penetration (1 step)",
        qpos3,
        qvel1,
        act1,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 4: Moving + contacts (1 step) ---
    var qvel4 = InlineArray[Float64, NV](fill=0.0)
    qvel4[0] = 1.0  # rootx vel
    qvel4[1] = -1.0  # rootz vel (falling)
    qvel4[2] = -0.5  # rooty vel
    if compare_step(
        "Moving + contacts (1 step)",
        qpos1,
        qvel4,
        act2,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 5: Ground contact, 5 steps ---
    if compare_step(
        "Ground contact (5 steps)",
        qpos1,
        qvel1,
        act1,
        5,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 6: Ground contact + actions, 5 steps ---
    if compare_step(
        "Ground contact + actions (5 steps)",
        qpos1,
        qvel1,
        act2,
        5,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
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
