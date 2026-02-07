"""Minimal GPU test to isolate Metal compilation failure.

Tests parts of the constraint GC kernel separately to find what breaks Metal.

Run with:
    cd mojo-rl
    pixi run -e apple mojo run physics3d/tests/test_constraint_kernel_gpu.mojo
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from physics3d.gpu.kernels import (
    detect_ground_contacts_gpu,
    normalize_qpos_quaternions_gpu,
)
from physics3d.gpu.constants import (
    TPB,
    state_size,
    model_size,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    model_metadata_offset,
    model_body_offset,
    model_joint_offset,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_GROUND_Z,
    MODEL_META_IDX_FRICTION,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_RADIUS,
    BODY_IDX_HALF_LENGTH,
    BODY_IDX_PARENT,
    BODY_IDX_QUAT_W,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JNT_SLIDE,
    JNT_HINGE,
)

from physics3d.kinematics.forward_kinematics import (
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_diagonal_gpu,
    compute_mass_matrix_full_gpu,
    ldl_factor_gpu,
    ldl_solve_gpu,
    compute_M_inv_from_ldl_gpu,
)
from physics3d.dynamics.bias_forces import compute_bias_forces_gpu
from physics3d.dynamics.jacobian import (
    compute_cdof_gpu,
    compute_composite_inertia_gpu,
)
from physics3d.solver.pgs_solver import PGSSolver


# HalfCheetahGC dimensions
comptime DTYPE = DType.float32
comptime NQ: Int = 10
comptime NV: Int = 10
comptime NBODY: Int = 8
comptime NJOINT: Int = 10
comptime MAX_CONTACTS: Int = 20
comptime BATCH: Int = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size[NBODY, NJOINT]()
comptime V_SIZE: Int = 10
comptime M_SIZE: Int = 100  # NV*NV
comptime CDOF_SIZE: Int = 60  # NV*6
comptime CRB_SIZE: Int = 80  # NBODY*10


fn setup_minimal_model(model_host: HostBuffer[DTYPE]):
    """Set up a minimal HalfCheetah-like model in the buffer."""
    var meta_off = model_metadata_offset[NBODY, NJOINT]()
    model_host[meta_off + MODEL_META_IDX_NBODY] = Float32(NBODY)
    model_host[meta_off + MODEL_META_IDX_NJOINT] = Float32(NJOINT)
    model_host[meta_off + MODEL_META_IDX_GRAVITY_Z] = Float32(-9.81)
    model_host[meta_off + MODEL_META_IDX_TIMESTEP] = Float32(0.002)
    model_host[meta_off + MODEL_META_IDX_GROUND_Z] = Float32(0.0)
    model_host[meta_off + MODEL_META_IDX_FRICTION] = Float32(0.9)

    for b in range(NBODY):
        var off = model_body_offset(b)
        model_host[off + BODY_IDX_MASS] = Float32(1.0)
        model_host[off + BODY_IDX_INV_MASS] = Float32(1.0)
        model_host[off + BODY_IDX_IXX] = Float32(0.01)
        model_host[off + BODY_IDX_IYY] = Float32(0.01)
        model_host[off + BODY_IDX_IZZ] = Float32(0.01)
        model_host[off + BODY_IDX_RADIUS] = Float32(0.046)
        model_host[off + BODY_IDX_HALF_LENGTH] = Float32(0.1)
        model_host[off + BODY_IDX_PARENT] = Float32(-1) if b == 0 else Float32(0)
        model_host[off + BODY_IDX_QUAT_W] = Float32(1.0)

    for j in range(NJOINT):
        var off = model_joint_offset[NBODY](j)
        model_host[off + JOINT_IDX_BODY_ID] = Float32(0) if j < 3 else Float32(j - 2)
        model_host[off + JOINT_IDX_QPOS_ADR] = Float32(j)
        model_host[off + JOINT_IDX_DOF_ADR] = Float32(j)
        if j < 2:  # slides
            model_host[off + JOINT_IDX_TYPE] = Float32(JNT_SLIDE)
            if j == 0:
                model_host[off + JOINT_IDX_AXIS_X] = Float32(1.0)
            else:
                model_host[off + JOINT_IDX_AXIS_Z] = Float32(1.0)
            model_host[off + JOINT_IDX_RANGE_MIN] = Float32(-100.0)
            model_host[off + JOINT_IDX_RANGE_MAX] = Float32(100.0)
        else:  # hinges
            model_host[off + JOINT_IDX_TYPE] = Float32(JNT_HINGE)
            model_host[off + JOINT_IDX_AXIS_Y] = Float32(1.0)
            model_host[off + JOINT_IDX_RANGE_MIN] = Float32(-1.0)
            model_host[off + JOINT_IDX_RANGE_MAX] = Float32(1.0)
            model_host[off + JOINT_IDX_ARMATURE] = Float32(0.1)
            model_host[off + JOINT_IDX_DAMPING] = Float32(3.0)
            model_host[off + JOINT_IDX_STIFFNESS] = Float32(100.0)


fn main() raises:
    print("=" * 60)
    print("    Metal GPU Compilation Isolation Test")
    print("    HalfCheetahGC dims: NQ=10, NV=10, NBODY=8")
    print("=" * 60)
    print()

    var ctx = DeviceContext()
    print("GPU initialized")

    # Allocate buffers
    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_host = ctx.enqueue_create_host_buffer[DTYPE](MODEL_SIZE)
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(MODEL_SIZE):
        model_host[i] = Scalar[DTYPE](0)

    setup_minimal_model(model_host)

    # Set initial rootz
    var qpos_off = qpos_offset[NQ, NV]()
    state_host[qpos_off + 1] = Float32(0.7)

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # =====================================================
    # Test 1: FK + body velocities (simple GPU functions)
    # =====================================================
    print("\nTest 1: Forward kinematics GPU kernel...")
    @always_inline
    fn fk_kernel(
        state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return
        forward_kinematics_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, state, model)
        compute_body_velocities_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, state, model)

    var st = LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin](state_buf.unsafe_ptr())
    var md = LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin](model_buf.unsafe_ptr())
    ctx.enqueue_function[fk_kernel, fk_kernel](st, md, grid_dim=(1,), block_dim=(1,))
    ctx.synchronize()
    print("  PASSED")

    # =====================================================
    # Test 2: Contact detection
    # =====================================================
    print("Test 2: Contact detection GPU kernel...")
    @always_inline
    fn contact_kernel(
        state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return
        detect_ground_contacts_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, state, model)

    st = LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin](state_buf.unsafe_ptr())
    md = LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin](model_buf.unsafe_ptr())
    ctx.enqueue_function[contact_kernel, contact_kernel](st, md, grid_dim=(1,), block_dim=(1,))
    ctx.synchronize()
    print("  PASSED")

    # =====================================================
    # Test 3: Mass matrix (CRBA + LDL)
    # =====================================================
    print("Test 3: Mass matrix (CRBA + LDL) GPU kernel...")
    @always_inline
    fn mass_kernel(
        state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
        for i in range(CDOF_SIZE):
            cdof[i] = Scalar[DTYPE](0)

        compute_cdof_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, CDOF_SIZE, BATCH
        ](env, state, model, cdof)

        var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
        for i in range(CRB_SIZE):
            crb[i] = Scalar[DTYPE](0)
        compute_composite_inertia_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, CRB_SIZE, BATCH,
        ](env, state, model, crb)

        var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M[i] = Scalar[DTYPE](0)
        compute_mass_matrix_full_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, M_SIZE, CDOF_SIZE, CRB_SIZE, BATCH,
        ](env, state, model, cdof, crb, M)

        # LDL factorize
        var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var D = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        ldl_factor_gpu[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

        # M_inv
        var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M_inv[i] = Scalar[DTYPE](0)
        compute_M_inv_from_ldl_gpu[DTYPE, NV, M_SIZE, V_SIZE](L, D, M_inv)

    st = LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin](state_buf.unsafe_ptr())
    md = LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin](model_buf.unsafe_ptr())
    ctx.enqueue_function[mass_kernel, mass_kernel](st, md, grid_dim=(1,), block_dim=(1,))
    ctx.synchronize()
    print("  PASSED")

    # =====================================================
    # Test 4: Mass matrix + armature/damping + stiffness + LDL + PGS solve
    # (The full pipeline in step_constraint_kernel_with_solver)
    # =====================================================
    print("Test 4: Combined pipeline (M + armature + bias + LDL + PGS)...")
    @always_inline
    fn combined_kernel(
        state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        # Replicate the full pipeline from step_constraint_kernel_with_solver
        var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            bias[i] = Scalar[DTYPE](0)
        for i in range(CDOF_SIZE):
            cdof[i] = Scalar[DTYPE](0)

        # FK
        forward_kinematics_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, state, model)
        compute_body_velocities_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, state, model)

        # Contacts
        detect_ground_contacts_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, state, model)

        # Cdof
        compute_cdof_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, CDOF_SIZE, BATCH
        ](env, state, model, cdof)

        # CRB
        var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
        for i in range(CRB_SIZE):
            crb[i] = Scalar[DTYPE](0)
        compute_composite_inertia_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, CRB_SIZE, BATCH,
        ](env, state, model, crb)

        # Full M
        var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M[i] = Scalar[DTYPE](0)
        compute_mass_matrix_full_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, M_SIZE, CDOF_SIZE, CRB_SIZE, BATCH,
        ](env, state, model, cdof, crb, M)

        # Armature + implicit damping
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_TIMESTEP])
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR]))
            var arm = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_ARMATURE])
            var damp = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DAMPING])
            M[dof_adr * NV + dof_adr] = M[dof_adr * NV + dof_adr] + arm + dt * damp

        # LDL
        var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var D = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        ldl_factor_gpu[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

        # M_inv
        var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M_inv[i] = Scalar[DTYPE](0)
        compute_M_inv_from_ldl_gpu[DTYPE, NV, M_SIZE, V_SIZE](L, D, M_inv)

        # Bias forces
        compute_bias_forces_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, V_SIZE, BATCH
        ](env, state, model, bias)

        # f_net + stiffness
        var qpos_off2 = qpos_offset[NQ, NV]()
        var qfrc_off = qfrc_offset[NQ, NV]()
        var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            f_net[i] = rebind[Scalar[DTYPE]](state[env, qfrc_off + i]) - bias[i]

        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR]))
            var qpos_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR]))
            var stiff = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_STIFFNESS])
            if stiff > Scalar[DTYPE](0):
                var qpos_d = rebind[Scalar[DTYPE]](state[env, qpos_off2 + qpos_adr])
                f_net[dof_adr] = f_net[dof_adr] - stiff * qpos_d

        # LDL solve for qacc
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qacc[i] = Scalar[DTYPE](0)
        ldl_solve_gpu[DTYPE, NV, M_SIZE, V_SIZE](L, D, f_net, qacc)

        # Predicted velocity
        var qvel_off = qvel_offset[NQ, NV]()
        var qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            qvel_pred[i] = qvel + qacc[i] * dt

        # PGS solve (DISABLED to test if rest compiles)
        # PGSSolver.solve_gpu[
        #     DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        #     STATE_SIZE, MODEL_SIZE, V_SIZE, M_SIZE, CDOF_SIZE, BATCH,
        # ](env, state, model, M_inv, cdof, qvel_pred, dt)
        _ = M_inv
        _ = cdof
        _ = qvel_pred

    st = LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin](state_buf.unsafe_ptr())
    md = LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin](model_buf.unsafe_ptr())
    ctx.enqueue_function[combined_kernel, combined_kernel](st, md, grid_dim=(1,), block_dim=(1,))
    ctx.synchronize()
    print("  PASSED")

    # Test 5: PGS solver alone
    print("Test 5: PGS solver alone (separate kernel)...")
    @always_inline
    fn pgs_only_kernel(
        state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        # Dummy M_inv (identity) and cdof
        var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M_inv[i] = Scalar[DTYPE](0)
        for i in range(NV):
            M_inv[i * NV + i] = Scalar[DTYPE](1)

        var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
        for i in range(CDOF_SIZE):
            cdof[i] = Scalar[DTYPE](0)

        var qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qvel_pred[i] = Scalar[DTYPE](0)

        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_TIMESTEP])

        PGSSolver.solve_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, V_SIZE, M_SIZE, CDOF_SIZE, BATCH,
        ](env, state, model, M_inv, cdof, qvel_pred, dt)

    st = LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin](state_buf.unsafe_ptr())
    md = LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin](model_buf.unsafe_ptr())
    ctx.enqueue_function[pgs_only_kernel, pgs_only_kernel](st, md, grid_dim=(1,), block_dim=(1,))
    ctx.synchronize()
    print("  PASSED")

    print()
    print("ALL TESTS PASSED!")
