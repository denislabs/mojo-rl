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

G5: operates on the per-field tensors (the `[BATCH, STATE_SIZE]` hook slab
died with the fields sunset). Arithmetic verbatim from the slab kernel.
"""

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from .constants import TPB


def compute_cvel[
    DTYPE: DType,
    BATCH_SIZE: Int,
    NBODY: Int,
](
    ctx: DeviceContext,
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
    ],
    xvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
    ],
    xangvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
    ],
    cvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
    ],
) raises:
    """Compute cvel (body CoM spatial velocities) for all environments on GPU.

    One thread per environment; reads xvel/xangvel/xpos/xipos, writes cvel.
    """
    comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

    @parameter
    @always_inline
    def cvel_kernel(
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH_SIZE:
            return

        for b in range(NBODY):
            # Angular velocity at body origin (world frame)
            var ox = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 0])
            var oy = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 1])
            var oz = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 2])

            # Linear velocity at body origin (world frame)
            var vx = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 0])
            var vy = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 1])
            var vz = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 2])

            # Body origin world position
            var px = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 0])
            var py = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 1])
            var pz = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2])

            # Body CoM world position
            var cx = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 0])
            var cy = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 1])
            var cz = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 2])

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
            var cvel_base = b * 6
            cvel[env, cvel_base + 0] = ox
            cvel[env, cvel_base + 1] = oy
            cvel[env, cvel_base + 2] = oz
            cvel[env, cvel_base + 3] = vcx
            cvel[env, cvel_base + 4] = vcy
            cvel[env, cvel_base + 5] = vcz

    ctx.enqueue_function[cvel_kernel](
        xpos,
        xvel,
        xangvel,
        xipos,
        cvel,
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )
