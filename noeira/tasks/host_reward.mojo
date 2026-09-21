"""A family's GPU reward hook, evaluated on the HOST over a CPU env's `Data`.

    from noeira.tasks.host_reward import family_reward_host

    var rd = family_reward_host[So101TowerConfig, DType.float64, E.MD, ACT_DIM](
        env.d, env.mf, action, env.current_step, env.frame_skip, MODEL.TIMESTEP
    )
    var reward = rd[0]     # what the batched env would have paid for this state
    var holds = rd[1]      # the goal predicate

## WHY THIS EXISTS

`So101FamilyConfig.compute_reward_and_done_cpu` returns a constant zero on
purpose — the family's reward is a GPU kernel over the tape in `meta`, and
a static CPU hook has no tape. A recorder that runs the family in the CPU
viewer (`examples/so101/tower_teleop_record.mojo`) still needs every
transition's reward to be THE TRAINER'S reward, or the demonstrations go
into the replay on a different scale from the online rows beside them and
the critic fits two reward functions at once.

## HOW IT IS THE SAME REWARD

It IS the kernel's function. `compute_reward_and_done_gpu` is generic over
dtype and over its tensors' origin, and `tests/tasks/test_goal_distance.mojo`
already instantiates its callees over host `LayoutTensor`s. This wraps the
CPU env's `Data` and `Model` columns in `[1, N]` host views and calls the
config's hook for lane 0 — no second implementation of the shaping, the
rung, the closing bonus or the tape walk exists to drift from the device one.
What can differ is dtype: the batched env runs `DT` (float32) and the CPU env
float64, so a demo's reward is the float64 evaluation of the float32 formula,
a rounding-level gap.

⚠ THE `meta` AND `curriculum` WORDS MUST BE THE TASK'S. The tape, mask,
init and shaping words come from `posed_reset.task_meta_words` (written by
`run_view` after every reset) and the region table from
`gpu_eval.region_table_words` (`ViewerState.reset_curriculum`). With an
empty tape the hook evaluates op 0 against body 0 — a real, wrong
predicate, not an error — exactly as the SAC driver's header warns.

⚠ `META_IDX_NUM_CONTACTS` IS THE CPU COLLISION PASS'S TOO (`contact_detection`
writes `smeta[env, META_IDX_NUM_CONTACTS]` on both targets), so the grasp rung
reads the same contact records the device would.
"""

from layout import Layout, LayoutTensor

from noeira.nn.core.tensor import TensorImpl
from noeira.envs.phyics3d_env_config import Phyics3dEnvConfig
from noeira.physics3d.fields import Data, Model, DimsLike
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE, METADATA_SIZE, MODEL_BODY_SIZE, MODEL_CURRICULUM_SIZE,
    MODEL_GEOM_SIZE, MODEL_SITE_SIZE,
)


def family_reward_host[
    C: Phyics3dEnvConfig, DTYPE: DType, D: DimsLike, ACT: Int
](
    mut d: Data[DTYPE, D, 1],
    mut mf: Model[DTYPE, D],
    ref action: List[Float64],
    step_count: Int,
    frame_skip: Int,
    timestep: Float64,
) raises -> Tuple[Scalar[DTYPE], Bool]:
    """`C.compute_reward_and_done_gpu` for lane 0 of a BATCH-1 host `Data`.

    Writes `meta[META_IDX_GOAL_HELD]` as the kernel does. `action` is the
    action just applied (the hook takes it; the SO-101 family ignores it)."""
    comptime NQ = D.NQ
    comptime NV = D.NV
    comptime NB = D.NBODY
    comptime NS = D.NSITE
    comptime NG = D.NGEOM
    comptime MC = D.MAX_CONTACTS
    comptime L_Q = Layout.row_major(1, NQ)
    comptime L_V = Layout.row_major(1, NV)
    comptime L_B3 = Layout.row_major(1, NB * 3)
    comptime L_B4 = Layout.row_major(1, NB * 4)
    comptime L_B6 = Layout.row_major(1, NB * 6)
    comptime L_S3 = Layout.row_major(1, NS * 3)
    comptime L_CON = Layout.row_major(1, MC * CONTACT_SIZE)
    comptime L_META = Layout.row_major(1, METADATA_SIZE)
    comptime L_CUR = Layout.row_major(1, MODEL_CURRICULUM_SIZE)
    comptime L_ACT = Layout.row_major(1, ACT)
    comptime L_NA = Layout.row_major(1, 1)
    comptime L_BODIES = Layout.row_major(NB, MODEL_BODY_SIZE)
    comptime L_SITES = Layout.row_major(NS, MODEL_SITE_SIZE)
    comptime L_GEOMS = Layout.row_major(NG, MODEL_GEOM_SIZE)

    if len(action) != ACT:
        raise Error(
            "family_reward_host: action has " + String(len(action))
            + " words, ACT is " + String(ACT)
        )
    var actions_t = TensorImpl[DTYPE].alloc(ACT)
    for j in range(ACT):
        actions_t.data[j] = Scalar[DTYPE](action[j])
    # `act` (actuator activations): the family has none; one word keeps the
    # layout legal.
    var act_t = TensorImpl[DTYPE].alloc(1)

    return C.compute_reward_and_done_gpu[
        DTYPE, 1, NQ, NV, NB, ACT, NS * 3, MC, NS, NG, 1
    ](
        d.qpos.lt["cpu", L_Q](),
        d.qvel.lt["cpu", L_V](),
        d.xpos.lt["cpu", L_B3](),
        d.xipos.lt["cpu", L_B3](),
        d.xquat.lt["cpu", L_B4](),
        d.xvel.lt["cpu", L_B3](),
        mf.bodies.lt["cpu", L_BODIES](),
        d.site_xpos.lt["cpu", L_S3](),
        d.contacts.lt["cpu", L_CON](),
        mf.sites.lt["cpu", L_SITES](),
        mf.geoms.lt["cpu", L_GEOMS](),
        d.cfrc_ext.lt["cpu", L_B6](),
        d.cvel.lt["cpu", L_B6](),
        d.meta.lt["cpu", L_META](),
        mf.curriculum.lt["cpu", L_CUR](),
        actions_t.lt["cpu", L_ACT](),
        d.xangvel.lt["cpu", L_B3](),
        d.cacc.lt["cpu", L_B6](),
        d.cfrc_int.lt["cpu", L_B6](),
        d.subtree_com.lt["cpu", L_B3](),
        d.site_xpos_acc.lt["cpu", L_S3](),
        d.xquat_acc.lt["cpu", L_B4](),
        act_t.lt["cpu", L_NA](),
        0,
        step_count,
        frame_skip,
        Scalar[DTYPE](timestep),
    )
