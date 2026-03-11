"""GPU kernel for computing cfrc_ext: external contact forces per body.

Mirrors the CPU compute_cfrc_ext() in physics3d/dynamics/cfrc_ext.mojo.

Algorithm (one GPU thread per environment):
1. Zero cfrc_ext region in state buffer.
2. Compute subtree CoM for each body (backward pass using body_mass/body_parent from model).
3. Compute body_rootid for each body.
4. For each active contact: reconstruct world-frame force/torque, apply moment-arm
   correction, accumulate into cfrc_ext[body_a] and subtract from cfrc_ext[body_b].

cfrc_ext[b*6 + 0..5] = [torque_x, torque_y, torque_z, force_x, force_y, force_z]
expressed in world frame at subtree CoM of the body's kinematic root.
"""

from std.collections import InlineArray

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from .constants import (
    TPB,
    state_size,
    xipos_offset,
    contacts_offset,
    metadata_offset,
    cfrc_ext_offset,
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
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FORCE_TORSION,
    CONTACT_IDX_FORCE_ROLL1,
    CONTACT_IDX_FORCE_ROLL2,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    BODY_IDX_MASS,
    BODY_IDX_PARENT,
    model_body_offset,
)


fn compute_cfrc_ext_gpu[
    DTYPE: DType,
    BATCH_SIZE: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int = 0,
](
    ctx: DeviceContext,
    mut states_buf: DeviceBuffer[DTYPE],
    mut model_buf: DeviceBuffer[DTYPE],
) raises:
    """Compute cfrc_ext for all environments on GPU.

    Accumulates contact forces into cfrc_ext[body*6] in the state buffer.
    One thread per environment.

    Args:
        ctx: GPU device context.
        states_buf: State buffer [BATCH_SIZE, STATE_SIZE].
        model_buf: Model buffer [1, MODEL_SIZE] (read-only; body_mass and body_parent).
    """
    var states = LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
    ](states_buf)
    var model = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf)

    comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

    comptime XIPOS_OFF = xipos_offset[NQ, NV, NBODY]()
    comptime CONTACTS_OFF = contacts_offset[NQ, NV, NBODY]()
    comptime META_OFF = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    comptime CFRC_OFF = cfrc_ext_offset[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
    comptime EPS = Scalar[DTYPE](1e-10)

    @always_inline
    fn cfrc_ext_kernel(
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH_SIZE:
            return

        # --- 1. Zero cfrc_ext ---
        for i in range(NBODY * 6):
            states[env, CFRC_OFF + i] = Scalar[DTYPE](0)

        # --- 2. Compute subtree_com for each body ---
        var stmass = InlineArray[Scalar[DTYPE], NBODY](uninitialized=True)
        var stcom = InlineArray[Scalar[DTYPE], NBODY * 3](uninitialized=True)

        for i in range(NBODY):
            var body_off = model_body_offset(i)
            var m = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_MASS])
            stmass[i] = m
            stcom[i * 3 + 0] = m * rebind[Scalar[DTYPE]](
                states[env, XIPOS_OFF + i * 3 + 0]
            )
            stcom[i * 3 + 1] = m * rebind[Scalar[DTYPE]](
                states[env, XIPOS_OFF + i * 3 + 1]
            )
            stcom[i * 3 + 2] = m * rebind[Scalar[DTYPE]](
                states[env, XIPOS_OFF + i * 3 + 2]
            )

        # Backward sweep: add child contribution to parent
        for i in range(NBODY - 1, 0, -1):
            var body_off = model_body_offset(i)
            var p = Int(
                rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
            )
            stmass[p] += stmass[i]
            stcom[p * 3 + 0] += stcom[i * 3 + 0]
            stcom[p * 3 + 1] += stcom[i * 3 + 1]
            stcom[p * 3 + 2] += stcom[i * 3 + 2]

        # Normalize to get CoM position
        for i in range(NBODY):
            var sm = stmass[i]
            if sm > EPS:
                stcom[i * 3 + 0] = stcom[i * 3 + 0] / sm
                stcom[i * 3 + 1] = stcom[i * 3 + 1] / sm
                stcom[i * 3 + 2] = stcom[i * 3 + 2] / sm
            else:
                stcom[i * 3 + 0] = rebind[Scalar[DTYPE]](
                    states[env, XIPOS_OFF + i * 3 + 0]
                )
                stcom[i * 3 + 1] = rebind[Scalar[DTYPE]](
                    states[env, XIPOS_OFF + i * 3 + 1]
                )
                stcom[i * 3 + 2] = rebind[Scalar[DTYPE]](
                    states[env, XIPOS_OFF + i * 3 + 2]
                )

        # --- 3. Compute body_rootid ---
        var rootid = InlineArray[Int, NBODY](uninitialized=True)
        rootid[0] = 0
        for i in range(1, NBODY):
            var body_off = model_body_offset(i)
            var p = Int(
                rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
            )
            if p == 0:
                rootid[i] = i
            else:
                rootid[i] = rootid[p]

        # --- 4. Accumulate contact forces ---
        var num_contacts = Int(
            rebind[Scalar[DTYPE]](states[env, META_OFF + META_IDX_NUM_CONTACTS])
        )

        for ci in range(MAX_CONTACTS):
            if ci >= num_contacts:
                break

            var con_base = CONTACTS_OFF + ci * CONTACT_SIZE

            # Contact frame axes
            var nx = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_NX]
            )
            var ny = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_NY]
            )
            var nz = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_NZ]
            )
            var t1x = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FRAME_T1_X]
            )
            var t1y = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FRAME_T1_Y]
            )
            var t1z = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FRAME_T1_Z]
            )
            # T2 = N × T1
            var t2x = ny * t1z - nz * t1y
            var t2y = nz * t1x - nx * t1z
            var t2z = nx * t1y - ny * t1x

            # Contact forces in contact-local frame
            var f_n = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FORCE_N]
            )
            var f_t1 = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FORCE_T1]
            )
            var f_t2 = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FORCE_T2]
            )
            var f_tors = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FORCE_TORSION]
            )
            var f_roll1 = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FORCE_ROLL1]
            )
            var f_roll2 = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_FORCE_ROLL2]
            )

            # World-frame force: f_n*N + f_t1*T1 + f_t2*T2
            var fw_x = f_n * nx + f_t1 * t1x + f_t2 * t2x
            var fw_y = f_n * ny + f_t1 * t1y + f_t2 * t2y
            var fw_z = f_n * nz + f_t1 * t1z + f_t2 * t2z
            # World-frame torque: f_tors*N + f_roll1*T1 + f_roll2*T2
            var tw_x = f_tors * nx + f_roll1 * t1x + f_roll2 * t2x
            var tw_y = f_tors * ny + f_roll1 * t1y + f_roll2 * t2y
            var tw_z = f_tors * nz + f_roll1 * t1z + f_roll2 * t2z

            # Contact point
            var cx = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_POS_X]
            )
            var cy = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_POS_Y]
            )
            var cz = rebind[Scalar[DTYPE]](
                states[env, con_base + CONTACT_IDX_POS_Z]
            )

            var ka = Int(
                rebind[Scalar[DTYPE]](
                    states[env, con_base + CONTACT_IDX_BODY_A]
                )
            )
            var kb = Int(
                rebind[Scalar[DTYPE]](
                    states[env, con_base + CONTACT_IDX_BODY_B]
                )
            )

            # body_a: add direct force
            if ka > 0:
                var rid = rootid[ka]
                var dx = stcom[rid * 3 + 0] - cx
                var dy = stcom[rid * 3 + 1] - cy
                var dz = stcom[rid * 3 + 2] - cz
                # moment arm correction: tw_corrected = tw - (d × fw)
                var cx_ = dy * fw_z - dz * fw_y
                var cy_ = dz * fw_x - dx * fw_z
                var cz_ = dx * fw_y - dy * fw_x
                var base_off = CFRC_OFF + ka * 6
                states[env, base_off + 0] = rebind[Scalar[DTYPE]](
                    states[env, base_off + 0]
                ) + (tw_x - cx_)
                states[env, base_off + 1] = rebind[Scalar[DTYPE]](
                    states[env, base_off + 1]
                ) + (tw_y - cy_)
                states[env, base_off + 2] = rebind[Scalar[DTYPE]](
                    states[env, base_off + 2]
                ) + (tw_z - cz_)
                states[env, base_off + 3] = (
                    rebind[Scalar[DTYPE]](states[env, base_off + 3]) + fw_x
                )
                states[env, base_off + 4] = (
                    rebind[Scalar[DTYPE]](states[env, base_off + 4]) + fw_y
                )
                states[env, base_off + 5] = (
                    rebind[Scalar[DTYPE]](states[env, base_off + 5]) + fw_z
                )

            # body_b: subtract reaction force (Newton's 3rd law)
            if kb > 0:
                var rid = rootid[kb]
                var dx = stcom[rid * 3 + 0] - cx
                var dy = stcom[rid * 3 + 1] - cy
                var dz = stcom[rid * 3 + 2] - cz
                var cx_ = dy * fw_z - dz * fw_y
                var cy_ = dz * fw_x - dx * fw_z
                var cz_ = dx * fw_y - dy * fw_x
                var base_off = CFRC_OFF + kb * 6
                states[env, base_off + 0] = rebind[Scalar[DTYPE]](
                    states[env, base_off + 0]
                ) - (tw_x - cx_)
                states[env, base_off + 1] = rebind[Scalar[DTYPE]](
                    states[env, base_off + 1]
                ) - (tw_y - cy_)
                states[env, base_off + 2] = rebind[Scalar[DTYPE]](
                    states[env, base_off + 2]
                ) - (tw_z - cz_)
                states[env, base_off + 3] = (
                    rebind[Scalar[DTYPE]](states[env, base_off + 3]) - fw_x
                )
                states[env, base_off + 4] = (
                    rebind[Scalar[DTYPE]](states[env, base_off + 4]) - fw_y
                )
                states[env, base_off + 5] = (
                    rebind[Scalar[DTYPE]](states[env, base_off + 5]) - fw_z
                )

    ctx.enqueue_function[cfrc_ext_kernel, cfrc_ext_kernel](
        states,
        model,
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )
