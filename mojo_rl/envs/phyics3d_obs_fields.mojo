"""Default observation extraction over per-field tensors (migration P3/P5).

Single-source port of the default physics3d observation — `qpos` (with an
optional leading skip, e.g. hiding rootx) concatenated with `qvel` — which
is what `ModelDefFromXML.extract_obs` produces on CPU and the generic env
obs kernel produces on GPU. One formula body, both targets, reading
`DataFields.qpos/qvel` and writing a caller-owned obs tensor
`[BATCH, OBS_DIM]` (OBS_DIM = NQ - OBS_QPOS_SKIP + NV).

Env configs with custom obs (xpos/cvel features etc.) get their own fields
extractors as they are ported; this covers the default-obs family
(InvertedPendulum, hopper/walker-style qpos-skip envs)."""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields

comptime OBS_TPB: Int = 64


@always_inline
def _obs_qpos_qvel_env[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    OBS_QPOS_SKIP: Int,
    BATCH: Int,
](
    env: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    obs: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, NQ - OBS_QPOS_SKIP + NV),
        MutAnyOrigin,
    ],
):
    comptime NQO = NQ - OBS_QPOS_SKIP
    for i in range(NQO):
        obs[env, i] = qpos[env, OBS_QPOS_SKIP + i]
    for i in range(NV):
        obs[env, NQO + i] = qvel[env, i]


def _obs_qpos_qvel_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    OBS_QPOS_SKIP: Int,
    BATCH: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    obs: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, NQ - OBS_QPOS_SKIP + NV),
        MutAnyOrigin,
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _obs_qpos_qvel_env[DTYPE, NQ, NV, OBS_QPOS_SKIP, BATCH](
        env, qpos, qvel, obs
    )


def extract_obs_qpos_qvel_fields[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int = 0,
    OBS_QPOS_SKIP: Int = 0,
    BATCH: Int = 1,
](
    mut d: DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut obs: TensorImpl[DTYPE],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Default obs (qpos[SKIP:] ‖ qvel) into a caller-owned obs tensor,
    both targets, one body."""
    comptime L_QPOS = Layout.row_major(BATCH, NQ)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_OBS = Layout.row_major(BATCH, NQ - OBS_QPOS_SKIP + NV)

    comptime if target == "cpu":
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var obs_v = obs.lt["cpu", L_OBS]()
        for e in range(BATCH):
            _obs_qpos_qvel_env[DTYPE, NQ, NV, OBS_QPOS_SKIP, BATCH](
                e, qpos_v, qvel_v, obs_v
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + OBS_TPB - 1) // OBS_TPB
        c.enqueue_function[
            _obs_qpos_qvel_kernel[DTYPE, NQ, NV, OBS_QPOS_SKIP, BATCH]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            obs.lt["gpu", L_OBS](),
            grid_dim=(BLOCKS,),
            block_dim=(OBS_TPB,),
        )
