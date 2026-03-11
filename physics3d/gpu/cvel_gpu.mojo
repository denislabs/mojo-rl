"""GPU kernel for computing cvel: body CoM spatial velocities.

For each body b:
  omega = xangvel[b]                     (3 floats: angular velocity at body origin)
  v     = xvel[b]                        (3 floats: linear velocity at body origin)
  com   = xipos[b]                       (3 floats: CoM world position)
  ori   = xpos[b]                        (3 floats: body origin world position)
  d     = com - ori                      (CoM offset from body origin in world frame)
  v_com = v + omega × d                  (linear velocity at CoM)

  cvel[b*6 + 0..2] = omega
  cvel[b*6 + 3..5] = v_com

No model buffer access required — pure state-buffer to state-buffer computation.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from .constants import (
    TPB,
    state_size,
    xpos_offset,
    xvel_offset,
    xangvel_offset,
    xipos_offset,
    cvel_offset,
)


fn compute_cvel_gpu[
    DTYPE: DType,
    BATCH_SIZE: Int,
    STATE_SIZE: Int,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int = 0,
](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE],) raises:
    """Compute cvel (body CoM spatial velocities) for all environments on GPU.

    Writes to cvel region of the state buffer. One thread per environment.

    Args:
        ctx: GPU device context.
        states_buf: State buffer [BATCH_SIZE, STATE_SIZE] (read xvel/xangvel/xpos/xipos,
                    write cvel).
    """
    var states = LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
    ](states_buf)

    comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

    comptime XPOS_OFF = xpos_offset[NQ, NV, NBODY]()
    comptime XVEL_OFF = xvel_offset[NQ, NV, NBODY]()
    comptime XANGVEL_OFF = xangvel_offset[NQ, NV, NBODY]()
    comptime XIPOS_OFF = xipos_offset[NQ, NV, NBODY]()
    comptime CVEL_OFF = cvel_offset[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()

    @always_inline
    fn cvel_kernel(
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH_SIZE:
            return

        for b in range(NBODY):
            # Angular velocity at body origin (world frame)
            var ox = rebind[Scalar[DTYPE]](states[env, XANGVEL_OFF + b * 3 + 0])
            var oy = rebind[Scalar[DTYPE]](states[env, XANGVEL_OFF + b * 3 + 1])
            var oz = rebind[Scalar[DTYPE]](states[env, XANGVEL_OFF + b * 3 + 2])

            # Linear velocity at body origin (world frame)
            var vx = rebind[Scalar[DTYPE]](states[env, XVEL_OFF + b * 3 + 0])
            var vy = rebind[Scalar[DTYPE]](states[env, XVEL_OFF + b * 3 + 1])
            var vz = rebind[Scalar[DTYPE]](states[env, XVEL_OFF + b * 3 + 2])

            # Body origin world position
            var px = rebind[Scalar[DTYPE]](states[env, XPOS_OFF + b * 3 + 0])
            var py = rebind[Scalar[DTYPE]](states[env, XPOS_OFF + b * 3 + 1])
            var pz = rebind[Scalar[DTYPE]](states[env, XPOS_OFF + b * 3 + 2])

            # Body CoM world position
            var cx = rebind[Scalar[DTYPE]](states[env, XIPOS_OFF + b * 3 + 0])
            var cy = rebind[Scalar[DTYPE]](states[env, XIPOS_OFF + b * 3 + 1])
            var cz = rebind[Scalar[DTYPE]](states[env, XIPOS_OFF + b * 3 + 2])

            # CoM offset from body origin: d = com - ori
            var dx = cx - px
            var dy = cy - py
            var dz = cz - pz

            # Linear velocity at CoM: v_com = v + omega × d
            # omega × d = (oy*dz - oz*dy, oz*dx - ox*dz, ox*dy - oy*dx)
            var vcx = vx + (oy * dz - oz * dy)
            var vcy = vy + (oz * dx - ox * dz)
            var vcz = vz + (ox * dy - oy * dx)

            # Write cvel[b*6]: [omega, v_com]
            var cvel_base = CVEL_OFF + b * 6
            states[env, cvel_base + 0] = ox
            states[env, cvel_base + 1] = oy
            states[env, cvel_base + 2] = oz
            states[env, cvel_base + 3] = vcx
            states[env, cvel_base + 4] = vcy
            states[env, cvel_base + 5] = vcz

    ctx.enqueue_function[cvel_kernel, cvel_kernel](
        states,
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )
