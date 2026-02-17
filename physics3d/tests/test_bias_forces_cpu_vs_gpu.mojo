"""Test Bias Forces (RNE): CPU vs GPU.

Compares our CPU bias forces (RNE) with GPU bias forces for the HalfCheetah
model at multiple configurations. Both should produce identical results
(up to float32 precision).

The GPU pipeline is: FK -> cdof -> bias_forces_rne.

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_bias_forces_cpu_vs_gpu.mojo
"""

from math import abs
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from gpu import block_idx

from physics3d.types import Model, Data, _max_one
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
    forward_kinematics_gpu,
)
from physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_cdof_gpu,
)
from physics3d.dynamics.bias_forces import (
    compute_bias_forces_rne,
    compute_bias_forces_rne_gpu,
)
from physics3d.gpu.constants import (
    state_size,
    model_size,
    qpos_offset,
    qvel_offset,
    integrator_workspace_size,
    ws_bias_offset,
)
from physics3d.gpu.buffer_utils import (
    create_state_buffer,
    create_model_buffer,
    copy_model_to_buffer,
    copy_data_to_buffer,
)
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahGeoms,
    HalfCheetahParams,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY  # 7
comptime NJOINT = HalfCheetahModel.NJOINT  # 9
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS  # 20
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size[NBODY, NJOINT]()
comptime WS_SIZE = integrator_workspace_size[NV, NBODY]()

comptime V_SIZE = _max_one[NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()

# Tolerance (float32)
comptime ABS_TOL: Float64 = 1e-2
comptime REL_TOL: Float64 = 1e-2


# =============================================================================
# GPU kernel: FK + cdof + bias_forces_rne
# =============================================================================


fn bias_forces_kernel[
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

    # 1. Forward kinematics
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # 2. Compute cdof
    compute_cdof_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 3. Compute bias forces (reads cdof from workspace, writes bias to workspace)
    compute_bias_forces_rne_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)


# =============================================================================
# Main — allocates GPU buffers once, reuses across all tests
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("Bias Forces (RNE) Validation: CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NV=9)")
    print("Precision: float32")
    print("Tolerances: abs=", ABS_TOL, " rel=", REL_TOL)
    print()

    # Initialize GPU
    var ctx = DeviceContext()
    print("GPU device initialized")

    # Create model (CPU + GPU) once
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
    )
    HalfCheetahBodies.setup_model(model_cpu)
    HalfCheetahJoints.setup_model(model_cpu)
    HalfCheetahGeoms.setup_model(model_cpu)

    var model_host = create_model_buffer[DTYPE, NBODY, NJOINT](ctx)
    copy_model_to_buffer[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        model_cpu, model_host
    )
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()
    print("Model copied to GPU")

    # Pre-allocate GPU buffers ONCE (reused across all tests)
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    print("GPU buffers allocated")
    print()

    # Compile kernel once
    comptime kernel_fn = bias_forces_kernel[
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
    # Test configurations: (name, qpos, qvel)
    # =================================================================

    comptime NUM_TESTS = 5
    var test_names = InlineArray[String, NUM_TESTS](
        uninitialized=True
    )
    test_names[0] = "Default qpos, zero vel (gravity only)"
    test_names[1] = "Zero qpos, zero vel"
    test_names[2] = "Non-zero joints, zero vel"
    test_names[3] = "Non-zero vel (gravity + Coriolis)"
    test_names[4] = "Extreme velocities"

    var test_qpos = InlineArray[InlineArray[Float64, NQ], NUM_TESTS](
        uninitialized=True
    )
    var test_qvel = InlineArray[InlineArray[Float64, NV], NUM_TESTS](
        uninitialized=True
    )

    # Config 0: Default qpos, zero vel
    test_qpos[0] = InlineArray[Float64, NQ](fill=0.0)
    test_qpos[0][1] = 0.7
    test_qvel[0] = InlineArray[Float64, NV](fill=0.0)

    # Config 1: Zero qpos, zero vel
    test_qpos[1] = InlineArray[Float64, NQ](fill=0.0)
    test_qvel[1] = InlineArray[Float64, NV](fill=0.0)

    # Config 2: Non-zero joints, zero vel
    test_qpos[2] = InlineArray[Float64, NQ](fill=0.0)
    test_qpos[2][0] = 1.0
    test_qpos[2][1] = 0.7
    test_qpos[2][2] = 0.3
    test_qpos[2][3] = -0.4
    test_qpos[2][4] = 0.5
    test_qpos[2][5] = -0.2
    test_qpos[2][6] = 0.6
    test_qpos[2][7] = -0.8
    test_qpos[2][8] = 0.3
    test_qvel[2] = InlineArray[Float64, NV](fill=0.0)

    # Config 3: Non-zero vel (gravity + Coriolis)
    test_qpos[3] = InlineArray[Float64, NQ](fill=0.0)
    test_qpos[3][1] = 0.7
    test_qpos[3][2] = 0.1
    test_qpos[3][3] = -0.3
    test_qpos[3][6] = 0.4
    test_qvel[3] = InlineArray[Float64, NV](fill=0.0)
    test_qvel[3][0] = 2.0
    test_qvel[3][2] = 0.5
    test_qvel[3][3] = -1.0
    test_qvel[3][4] = 0.8
    test_qvel[3][6] = 1.2
    test_qvel[3][7] = -0.6

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

    # =================================================================
    # Run all tests, reusing GPU buffers
    # =================================================================

    var num_pass = 0
    var num_fail = 0

    for t in range(NUM_TESTS):
        print("--- Test:", test_names[t], "---")

        # === CPU pipeline ===
        var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
        for i in range(NQ):
            data_cpu.qpos[i] = Scalar[DTYPE](test_qpos[t][i])
        for i in range(NV):
            data_cpu.qvel[i] = Scalar[DTYPE](test_qvel[t][i])

        forward_kinematics(model_cpu, data_cpu)
        compute_body_velocities(model_cpu, data_cpu)

        var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
        compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
            model_cpu, data_cpu, cdof
        )

        var bias_cpu = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            bias_cpu[i] = Scalar[DTYPE](0)
        compute_bias_forces_rne[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
        ](model_cpu, data_cpu, cdof, bias_cpu)

        # === GPU pipeline (reuse buffers) ===
        # Zero state host buffer and set qpos/qvel
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

        # Copy workspace back to read bias
        ctx.enqueue_copy(ws_host.unsafe_ptr(), workspace_buf)
        ctx.synchronize()

        # === Compare ===
        comptime bias_off = ws_bias_offset[NV, NBODY]()
        var all_pass = True
        var max_abs_err: Float64 = 0.0
        var max_rel_err: Float64 = 0.0
        var fail_count = 0

        for i in range(NV):
            var cpu_val = Float64(bias_cpu[i])
            var gpu_val = Float64(ws_host[bias_off + i])
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
                    "  FAIL bias[", i, "]",
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
        print("  CPU bias:", end="")
        for i in range(NV):
            print(" ", Float64(bias_cpu[i]), end="")
        print()
        print("  GPU bias:", end="")
        for i in range(NV):
            print(" ", Float64(ws_host[bias_off + i]), end="")
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
