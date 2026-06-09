"""Test tree-walk CRBA mass matrix on a FREE-joint model (Ant).

Validates `compute_mass_matrix_treewalk_gpu_mt` against the proven serial GPU
mass matrix (`compute_mass_matrix_full_gpu`, already validated vs CPU + MuJoCo)
on the Ant model, which has a 3D FREE joint (6 DOFs). HalfCheetah (slide/hinge
only) does NOT exercise the free-joint `dof_parent` chain, so this is the gate
before enabling RK4_PARALLEL_CRBA for Humanoid/Ant.

GPU-vs-GPU: a single block runs the shared upstream pipeline
(FK -> subtree_com -> cdof -> composite_inertia) once on tid0, computes the
dense M and stashes it in the unused LDL-L scratch slot, then ALL threads run
the cooperative tree-walk M into the M slot. We compare the two M's directly,
so any difference is purely the tree-walk algorithm (identical inputs).

Run with:
    cd mojo-rl && pixi run -e apple mojo run -I . tests/physics3d/test_mass_matrix_treewalk_ant.mojo
"""

from std.testing import assert_true
from std.math import abs
from std.collections import InlineArray
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from std.gpu import block_idx, thread_idx, barrier

from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics_gpu
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_cdof_gpu,
    compute_composite_inertia_gpu,
    compute_subtree_com_gpu,
)
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full_gpu,
    compute_mass_matrix_treewalk_gpu_mt,
)
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    integrator_workspace_size,
    ws_M_offset,
    ws_L_offset,
)
from mojo_rl.physics3d.gpu.buffer_utils import create_state_buffer
from mojo_rl.envs.ant.ant_xml import AntModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = AntModel.NQ  # 15 (7 free-joint + 8 hinge)
comptime NV = AntModel.NV  # 14 (6 free-joint + 8 hinge)
comptime NBODY = AntModel.NBODY  # 14
comptime NJOINT = AntModel.NJOINT  # 9 (1 free + 8 hinge)
comptime NGEOM = AntModel.NGEOM  # 15
comptime MAX_CONTACTS = AntModel.MAX_CONTACTS  # 40
comptime NSITE = AntModel.NSITE  # 0
comptime BATCH = 1
comptime THREADS = NV  # cooperative block: one thread per DOF

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime WS_SIZE = integrator_workspace_size[NV, NBODY]()

# Tolerance (float32). The tree-walk reorders the composite sum + uses a
# parallel-axis shift, so it agrees to a looser tolerance than bit-identity.
comptime M_TOL: Float64 = 1e-3
comptime M_REL_TOL: Float64 = 1e-2


# =============================================================================
# GPU kernel: dense MM (tid0) stashed, then cooperative tree-walk MM
# =============================================================================


def mm_treewalk_kernel[
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
    var tid = Int(thread_idx.x)
    var n_threads = THREADS
    var valid = env < BATCH

    comptime M_off = ws_M_offset[NV, NBODY]()
    comptime L_off = ws_L_offset[NV, NBODY]()

    # --- tid0: shared upstream pipeline + dense (oracle) mass matrix ---
    if tid == 0 and valid:
        forward_kinematics_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, BATCH,
        ](env, state, model)
        compute_subtree_com_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, BATCH, NSITE,
        ](env, state, model)
        compute_cdof_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
        ](env, state, model, workspace)
        compute_composite_inertia_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
        ](env, state, model, workspace)
        compute_mass_matrix_full_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
        ](env, state, model, workspace)
        # Stash dense M into the (unused-here) LDL-L scratch slot.
        for k in range(NV * NV):
            workspace[env, L_off + k] = workspace[env, M_off + k]

    barrier()

    # --- all threads: cooperative tree-walk M into the M slot ---
    compute_mass_matrix_treewalk_gpu_mt[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, tid, n_threads, valid, state, model, workspace)


# =============================================================================
# Comparison helper
# =============================================================================


def compare(
    ctx: DeviceContext,
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    model_buf: DeviceBuffer[DTYPE],
) raises:
    print("--- Test:", test_name, "---")

    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_values[i])

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var ws_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())

    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(ws_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    var state_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model_tensor = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())
    var ws_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ](ws_buf.unsafe_ptr())

    comptime kernel_def = mm_treewalk_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ]
    ctx.enqueue_function[kernel_def](
        state_tensor,
        model_tensor,
        ws_tensor,
        grid_dim=(BATCH,),
        block_dim=(THREADS,),
    )
    ctx.synchronize()

    ctx.enqueue_copy(ws_host.unsafe_ptr(), ws_buf)
    ctx.synchronize()

    comptime M_off = ws_M_offset[NV, NBODY]()
    comptime L_off = ws_L_offset[NV, NBODY]()
    var all_pass = True
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for i in range(NV):
        for j in range(NV):
            var dense_val = Float64(ws_host[L_off + i * NV + j])
            var tree_val = Float64(ws_host[M_off + i * NV + j])
            var abs_err = abs(dense_val - tree_val)
            var ref_mag = abs(dense_val)
            var rel_err: Float64 = 0.0
            if ref_mag > 1e-10:
                rel_err = abs_err / ref_mag
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            var ok = abs_err < M_TOL or rel_err < M_REL_TOL
            if not ok:
                if fail_count < 10:
                    print(
                        "  FAIL M[", i, ",", j, "]",
                        " dense=", dense_val,
                        " tree=", tree_val,
                        " abs_err=", abs_err,
                        " rel_err=", rel_err,
                    )
                fail_count += 1
                all_pass = False

    if all_pass:
        print(
            "  ALL OK  max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err
        )
    else:
        print(
            "  FAILED", fail_count, "elements  max_abs_err=", max_abs_err,
            " max_rel_err=", max_rel_err,
        )

    print("  dense M diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(ws_host[L_off + i * NV + i]), end="")
    print()
    print("  tree  M diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(ws_host[M_off + i * NV + i]), end="")
    print()

    assert_true(all_pass, "dense vs tree-walk mismatch for: " + test_name)


# =============================================================================
# Test cases (free-joint Ant configs, mirror test_ant_fk_cpu_vs_gpu.mojo)
# =============================================================================


def main() raises:
    print("=" * 60)
    print("Tree-walk CRBA Validation: dense vs tree-walk — Ant (FREE joint)")
    print("=" * 60)
    print("Model: Ant (NBODY=14, NV=14, free joint + 8 hinge), THREADS=", THREADS)
    print("Tolerances: abs=", M_TOL, " rel=", M_REL_TOL)
    print()

    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    AntModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    # Config 1: default init_qpos (z=0.55, identity quat)
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    qpos1[2] = 0.55
    qpos1[3] = 1.0
    qpos1[8] = 1.0
    qpos1[10] = -1.0
    qpos1[12] = -1.0
    qpos1[14] = 1.0
    compare(ctx, "Default init_qpos", qpos1, model_buf)
    print()

    # Config 2: raised torso, identity quat
    var qpos2 = InlineArray[Float64, NQ](fill=0.0)
    qpos2[2] = 2.0
    qpos2[3] = 1.0
    compare(ctx, "Raised torso (z=2.0)", qpos2, model_buf)
    print()

    # Config 3: nonzero translation + joint angles
    var qpos3 = InlineArray[Float64, NQ](fill=0.0)
    qpos3[0] = 1.0
    qpos3[1] = 0.5
    qpos3[2] = 0.55
    qpos3[3] = 1.0
    qpos3[7] = 0.3
    qpos3[8] = 0.5
    qpos3[9] = -0.3
    qpos3[10] = 0.5
    qpos3[11] = 0.2
    qpos3[12] = -0.4
    qpos3[13] = -0.2
    qpos3[14] = 0.4
    compare(ctx, "Nonzero joint angles", qpos3, model_buf)
    print()

    # Config 4: rotated torso (30 deg about z) — exercises free-joint rotation
    var qpos4 = InlineArray[Float64, NQ](fill=0.0)
    qpos4[2] = 0.55
    qpos4[3] = 0.866
    qpos4[6] = 0.5
    compare(ctx, "Rotated torso (30 deg about z)", qpos4, model_buf)
    print()

    print("All Ant tree-walk CRBA tests passed.")
