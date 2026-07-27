"""Subtree center-of-mass over per-field tensors (migration P2,
single-source). Per-field port of `compute_subtree_com_gpu`
(dynamics/jacobian.mojo) — arithmetic verbatim, addressing per-field.
Operands: xipos + body records -> subtree_com (3 operands). One formula body
for both targets; per-body mass accumulator stays a per-thread InlineArray
(local scratch, not a field)."""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..fields import Data, Model
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_PARENT,
)

comptime STCOM_TPB: Int = 64


@always_inline
def _max_one[N: Int]() -> Int:
    return N if N > 0 else 1


@always_inline
def _subtree_com_env[
    DTYPE: DType,
    NBODY: Int,
    BATCH: Int,
](
    env: Int,
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    """Bottom-up mass*xipos accumulation, then normalize (verbatim from
    compute_subtree_com_gpu)."""
    comptime MASS_SIZE = _max_one[NBODY]()
    var stmass = InlineArray[Scalar[DTYPE], MASS_SIZE](uninitialized=True)
    for b in range(NBODY):
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        stmass[b] = mass
        subtree_com[env, b * 3 + 0] = mass * rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 0]
        )
        subtree_com[env, b * 3 + 1] = mass * rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 1]
        )
        subtree_com[env, b * 3 + 2] = mass * rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 2]
        )

    for b in range(NBODY - 1, 0, -1):
        var p = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        stmass[p] = stmass[p] + stmass[b]
        subtree_com[env, p * 3 + 0] = rebind[Scalar[DTYPE]](
            subtree_com[env, p * 3 + 0]
        ) + rebind[Scalar[DTYPE]](subtree_com[env, b * 3 + 0])
        subtree_com[env, p * 3 + 1] = rebind[Scalar[DTYPE]](
            subtree_com[env, p * 3 + 1]
        ) + rebind[Scalar[DTYPE]](subtree_com[env, b * 3 + 1])
        subtree_com[env, p * 3 + 2] = rebind[Scalar[DTYPE]](
            subtree_com[env, p * 3 + 2]
        ) + rebind[Scalar[DTYPE]](subtree_com[env, b * 3 + 2])

    for b in range(NBODY):
        if stmass[b] > Scalar[DTYPE](1e-10):
            subtree_com[env, b * 3 + 0] = (
                rebind[Scalar[DTYPE]](subtree_com[env, b * 3 + 0]) / stmass[b]
            )
            subtree_com[env, b * 3 + 1] = (
                rebind[Scalar[DTYPE]](subtree_com[env, b * 3 + 1]) / stmass[b]
            )
            subtree_com[env, b * 3 + 2] = (
                rebind[Scalar[DTYPE]](subtree_com[env, b * 3 + 2]) / stmass[b]
            )
        else:
            subtree_com[env, b * 3 + 0] = rebind[Scalar[DTYPE]](
                xipos[env, b * 3 + 0]
            )
            subtree_com[env, b * 3 + 1] = rebind[Scalar[DTYPE]](
                xipos[env, b * 3 + 1]
            )
            subtree_com[env, b * 3 + 2] = rebind[Scalar[DTYPE]](
                xipos[env, b * 3 + 2]
            )


def _subtree_com_fields_kernel[
    DTYPE: DType,
    NBODY: Int,
    BATCH: Int,
](
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _subtree_com_env[DTYPE, NBODY, BATCH](
        env, bodies, xipos, subtree_com
    )


def compute_subtree_com[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
    BATCH: Int = 1,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: Model[
        DTYPE,
        NV,
        NBODY,
        NJOINT,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        NEXCLUDE,
        NMESH_VERTS,
    ],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Subtree CoM from xipos + body masses, both targets, one body."""
    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)

    comptime if target == "cpu":
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var xipos_v = d.xipos.lt["cpu", L_B3]()
        var stcom_v = d.subtree_com.lt["cpu", L_B3]()
        for e in range(BATCH):
            _subtree_com_env[DTYPE, NBODY, BATCH](
                e, bodies_v, xipos_v, stcom_v
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + STCOM_TPB - 1) // STCOM_TPB
        c.enqueue_function[_subtree_com_fields_kernel[DTYPE, NBODY, BATCH]](
            m.bodies.lt["gpu", L_BODY](),
            d.xipos.lt["gpu", L_B3](),
            d.subtree_com.lt["gpu", L_B3](),
            grid_dim=(BLOCKS,),
            block_dim=(STCOM_TPB,),
        )
