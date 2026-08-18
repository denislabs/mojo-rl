"""Subtree center-of-mass over per-field tensors (migration P2,
single-source). Per-field port of `compute_subtree_com_gpu`
(dynamics/jacobian.mojo) — arithmetic verbatim, addressing per-field.
Operands: xipos + body records -> subtree_com (3 operands). One formula body
for both targets; per-body mass accumulator stays a per-thread InlineArray
(local scratch, not a field)."""

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..fields import (
    Data,
    Model,
    Dims,
    DimsLike,
    AsStatic,
    Scratch,
    cap,
    DYN2,
    rl2,
)
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
    D: DimsLike,
    L_BODIES: Layout,
    L_XIPOS: Layout,
](
    env: Int,
    dims: D,
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, L_XIPOS, MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, L_XIPOS, MutAnyOrigin
    ],
):
    """Bottom-up mass*xipos accumulation, then normalize (verbatim from
    compute_subtree_com_gpu)."""
    var nbody = dims.get_nbody()
    comptime MASS_SIZE = cap[D.NBODY]()
    var stmass = Scratch[Scalar[DTYPE], MASS_SIZE](nbody, uninitialized=0)
    for b in range(nbody):
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

    for b in range(nbody - 1, 0, -1):
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

    for b in range(nbody):
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
    _subtree_com_env[DTYPE](
        env, Dims[nbody=NBODY](), bodies, xipos, subtree_com
    )


def compute_subtree_com[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Subtree CoM from xipos + body masses, both targets, one body."""
    comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
    comptime L_BODY = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)

    comptime if target == "cpu":
        var dm = d.dims
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BODY)
        var xipos_v = d.xipos.lt_dyn["cpu", DYN2](rl_B3)
        var stcom_v = d.subtree_com.lt_dyn["cpu", DYN2](rl_B3)
        for e in range(BATCH):
            _subtree_com_env[DTYPE](
                e, dm, bodies_v, xipos_v, stcom_v
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + STCOM_TPB - 1) // STCOM_TPB
        c.enqueue_function[_subtree_com_fields_kernel[DTYPE, D.NBODY, BATCH]](
            m.bodies.lt["gpu", L_BODY](),
            d.xipos.lt["gpu", L_B3](),
            d.subtree_com.lt["gpu", L_B3](),
            grid_dim=(BLOCKS,),
            block_dim=(STCOM_TPB,),
        )
