"""Test Contact Detection: CPU vs GPU.

Compares contact detection output (positions, normals, distances, body pairs)
computed on CPU vs GPU for the HalfCheetah model at multiple configurations.

CPU pipeline:  FK -> detect_contacts -> data.contacts[]
GPU pipeline:  FK_gpu -> detect_contacts_gpu -> state buffer contacts

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_contacts_cpu_vs_gpu.mojo
"""

from std.testing import assert_true
from std.math import abs, sqrt
from std.collections import InlineArray
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from std.gpu import block_idx

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from physics3d.collision.contact_detection import (
    detect_contacts,
    detect_contacts_gpu,
)
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    contacts_offset,
    metadata_offset,
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_CONDIM,
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
comptime BATCH = 1

comptime MC = _max_one[MAX_CONTACTS]()
comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()

# Tolerances (float32 through FK pipeline)
comptime POS_TOL: Float64 = 1e-3
comptime DIST_TOL: Float64 = 1e-3
comptime NORMAL_DOT_MIN: Float64 = 0.999


# =============================================================================
# GPU kernel: FK + contact detection
# =============================================================================


fn contact_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    NGEOM: Int,
](
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= BATCH:
        return

    # 1. Forward kinematics
    forward_kinematics_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
    ](env, state, model)

    # 2. Detect contacts
    detect_contacts_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
        NGEOM,
    ](env, state, model)


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_contacts(
    ctx: DeviceContext,
    test_name: String,
    test_qpos: InlineArray[Float64, NQ],
    model_cpu: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ],
    model_buf: DeviceBuffer[DTYPE],
    mut state_host: HostBuffer[DTYPE],
    mut state_buf: DeviceBuffer[DTYPE],
) raises:
    print("--- Test:", test_name, "---")

    # === CPU ===
    var data_cpu = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](test_qpos[i])

    forward_kinematics(model_cpu, data_cpu)
    detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model_cpu, data_cpu
    )

    var cpu_ncon = data_cpu.num_contacts
    print("  CPU: contacts=", cpu_ncon)

    # === GPU ===
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    comptime qpos_off = qpos_offset[NQ, NV]()
    for i in range(NQ):
        state_host[qpos_off + i] = Scalar[DTYPE](test_qpos[i])

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.synchronize()

    comptime kernel_fn = contact_kernel[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
        NGEOM,
    ]

    var state_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model_tensor = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())

    ctx.enqueue_function[kernel_fn, kernel_fn](
        state_tensor,
        model_tensor,
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ctx.synchronize()

    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    comptime cont_off = contacts_offset[NQ, NV, NBODY]()
    var gpu_ncon = Int(Float64(state_host[meta_off + META_IDX_NUM_CONTACTS]))
    print("  GPU: contacts=", gpu_ncon)

    # === Compare ===
    var all_pass = True

    if cpu_ncon != gpu_ncon:
        print(
            "  FAIL: contact count mismatch! CPU=", cpu_ncon, " GPU=", gpu_ncon
        )
        all_pass = False

    var matched = InlineArray[Int, MC](fill=-1)

    for gc in range(gpu_ncon):
        var g_off = cont_off + gc * CONTACT_SIZE
        var g_body_a = Int(Float64(state_host[g_off + CONTACT_IDX_BODY_A]))
        var g_body_b = Int(Float64(state_host[g_off + CONTACT_IDX_BODY_B]))
        var g_px = Float64(state_host[g_off + CONTACT_IDX_POS_X])
        var g_py = Float64(state_host[g_off + CONTACT_IDX_POS_Y])
        var g_pz = Float64(state_host[g_off + CONTACT_IDX_POS_Z])
        var g_nx = Float64(state_host[g_off + CONTACT_IDX_NX])
        var g_ny = Float64(state_host[g_off + CONTACT_IDX_NY])
        var g_nz = Float64(state_host[g_off + CONTACT_IDX_NZ])
        var g_dist = Float64(state_host[g_off + CONTACT_IDX_DIST])

        var best_idx = -1
        var best_pos_err: Float64 = 1e10

        for cc in range(cpu_ncon):
            var already = False
            for k in range(gc):
                if matched[k] == cc:
                    already = True
                    break
            if already:
                continue

            var ci = data_cpu.contacts[cc]
            var body_match = (
                ci.body_a == g_body_a and ci.body_b == g_body_b
            ) or (ci.body_a == g_body_b and ci.body_b == g_body_a)
            if not body_match:
                continue

            var dx = Float64(ci.pos_x) - g_px
            var dy = Float64(ci.pos_y) - g_py
            var dz = Float64(ci.pos_z) - g_pz
            var pos_err = sqrt(dx * dx + dy * dy + dz * dz)
            if pos_err < best_pos_err:
                best_pos_err = pos_err
                best_idx = cc

        if best_idx < 0:
            if g_dist < 0:
                print(
                    "  FAIL: no CPU match for GPU[",
                    gc,
                    "] body(",
                    g_body_a,
                    ",",
                    g_body_b,
                    ") dist=",
                    g_dist,
                )
                all_pass = False
            continue

        matched[gc] = best_idx
        var ci = data_cpu.contacts[best_idx]

        if best_pos_err > POS_TOL:
            print(
                "  FAIL pos[",
                gc,
                "] err=",
                best_pos_err,
                " cpu=(",
                Float64(ci.pos_x),
                ",",
                Float64(ci.pos_y),
                ",",
                Float64(ci.pos_z),
                ") gpu=(",
                g_px,
                ",",
                g_py,
                ",",
                g_pz,
                ")",
            )
            all_pass = False

        var dot = (
            Float64(ci.normal_x) * g_nx
            + Float64(ci.normal_y) * g_ny
            + Float64(ci.normal_z) * g_nz
        )
        if dot < NORMAL_DOT_MIN:
            print(
                "  FAIL normal[",
                gc,
                "] dot=",
                dot,
                " cpu=(",
                Float64(ci.normal_x),
                ",",
                Float64(ci.normal_y),
                ",",
                Float64(ci.normal_z),
                ") gpu=(",
                g_nx,
                ",",
                g_ny,
                ",",
                g_nz,
                ")",
            )
            all_pass = False

        var dist_err = abs(Float64(ci.dist) - g_dist)
        if dist_err > DIST_TOL:
            print(
                "  FAIL dist[",
                gc,
                "] err=",
                dist_err,
                " cpu=",
                Float64(ci.dist),
                " gpu=",
                g_dist,
            )
            all_pass = False

        print(
            "  Contact",
            gc,
            ": body(",
            g_body_a,
            ",",
            g_body_b,
            ") dist cpu=",
            Float64(ci.dist),
            " gpu=",
            g_dist,
            " pos_err=",
            best_pos_err,
            " normal_dot=",
            dot,
        )

    if all_pass:
        print("  ALL OK")
    else:
        print("  FAILED")

    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


fn test_high_above_ground() raises:
    print("=" * 60)
    print("Contact Detection: CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NGEOM=", NGEOM, ")")
    print("Precision: float32")
    print(
        "Tolerances: pos=",
        POS_TOL,
        " dist=",
        DIST_TOL,
        " normal_dot>",
        NORMAL_DOT_MIN,
    )
    print()

    var ctx = DeviceContext()
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var _setup_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.5
    compare_contacts(
        ctx,
        "High above ground (rootz=0.5)",
        qpos,
        model_cpu,
        model_buf,
        state_host,
        state_buf,
    )
    print()


fn test_default_pose() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var _setup_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    compare_contacts(
        ctx,
        "Default pose (rootz=0)",
        qpos,
        model_cpu,
        model_buf,
        state_host,
        state_buf,
    )
    print()


fn test_low_static() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var _setup_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.2
    compare_contacts(
        ctx,
        "Low static (rootz=-0.2)",
        qpos,
        model_cpu,
        model_buf,
        state_host,
        state_buf,
    )
    print()


fn test_very_low() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var _setup_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.5
    compare_contacts(
        ctx,
        "Very low (rootz=-0.5)",
        qpos,
        model_cpu,
        model_buf,
        state_host,
        state_buf,
    )
    print()


fn test_bent_legs() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var _setup_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.15
    qpos[3] = -0.5  # bthigh
    qpos[4] = 0.8  # bshin
    qpos[6] = 0.5  # fthigh
    qpos[7] = -0.8  # fshin
    compare_contacts(
        ctx,
        "Bent legs (rootz=-0.15)",
        qpos,
        model_cpu,
        model_buf,
        state_host,
        state_buf,
    )
    print()


fn test_tilted() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var _setup_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    qpos[2] = 0.3  # rooty
    compare_contacts(
        ctx,
        "Tilted (rootz=-0.3, rooty=0.3)",
        qpos,
        model_cpu,
        model_buf,
        state_host,
        state_buf,
    )
    print()


fn main() raises:
    test_high_above_ground()
    test_default_pose()
    test_low_static()
    test_very_low()
    test_bent_legs()
    test_tilted()
    print("All contacts CPU vs GPU tests passed.")
