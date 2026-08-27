"""Dataset collection for FB — generalized coordinates into a TrajectoryStore.

One generic routine over `(MODEL_DEF, CONFIG)`, so walker, cheetah and
quadruped share a collector rather than three copies that drift.

## What is stored, and why it is not the observation

`qpos`, `qvel`, `action` — nothing else. `docs/BFM_ZERO_SHOT_RL.md` §6 calls
this a functional requirement rather than a space optimisation: dm_control
rewards read `Data` (FK products, `xmat`, `subtree_linvel`), NOT the observation
vector, so a dataset of observations can only ever be scored under rewards that
were labelled at collection time. Storing generalized coordinates is what lets
`Phyics3dEnv.reward_at` replay a state under a reward invented afterwards —
which is the whole of zero-shot inference.

The observation is recoverable through `obs_at`; the reward is not recoverable
from the observation. That asymmetry is the entire argument.

⚠ The row convention is the state AFTER the step together with the action that
produced it. `tests/dm_control/test_reward_relabel.mojo` gates exactly that
pairing, so a collector that stored the PRE-step state would silently produce a
dataset whose relabelled rewards are all off by one control step.

## Episode boundaries

`end_episode()` per episode, so `EpisodeIndex` can later refuse a window that
straddles two of them. A trainer that samples `s'` as "the next row" must
respect those boundaries or it will occasionally pair a terminal state with the
next episode's reset — see the boundary handling in the milestone-1 example.
"""

from std.random import random_float64

from mojo_rl.data.column import ColumnSpec
from mojo_rl.data.store import TrajectoryStoreWriter

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig


struct CollectStats(Movable & Deinitable):
    var rows: Int
    var episodes: Int

    def __init__(out self, rows: Int, episodes: Int):
        self.rows = rows
        self.episodes = episodes

    def __init__(out self, *, deinit move: Self):
        self.rows = move.rows
        self.episodes = move.episodes


def collect_random[
    MODEL_DEF: ModelDefLike, CONFIG: Phyics3dEnvConfig
](
    path: String,
    env_id: String,
    n_episodes: Int,
    ep_len: Int,
    seed_value: Int,
    chunk_rows: Int = 4096,
) raises -> CollectStats:
    """Uniform-random actions into a `TrajectoryStore`.

    ⚠ A random policy is a WEAK collector. `docs/BFM_ZERO_SHOT_RL.md` component
    1 ranks four levers and puts varied start states above curiosity; that lever
    lives in `CONFIG` (pass a `WideResetConfig`), not here. This routine is
    deliberately dumb so the interesting choice stays visible at the call site.

    The stronger levers the plan lists — dumping complete SAC replay buffers,
    and the FB feedback loop — are not implemented. Do not read a run of this
    as "the dataset is as good as it gets"; read it as the baseline the
    coverage metrics are measured against.
    """
    comptime Env = Phyics3dEnv[MODEL_DEF, CONFIG, DType.float64, False]
    comptime NQ = MODEL_DEF.NQ
    comptime NV = MODEL_DEF.NV
    comptime NACT = MODEL_DEF.ACTION_DIM

    if n_episodes <= 0 or ep_len <= 0:
        raise Error("collect_random: n_episodes and ep_len must be > 0")

    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, NQ))
    cols.append(ColumnSpec(String("qvel"), DType.float32, NV))
    cols.append(ColumnSpec(String("action"), DType.float32, NACT))

    var w = TrajectoryStoreWriter(
        path, cols^, env_id=env_id, seed=seed_value, chunk_rows=chunk_rows
    )
    var env = Env()

    var qbuf = List[Float32](length=NQ, fill=Float32(0))
    var vbuf = List[Float32](length=NV, fill=Float32(0))
    var abuf = List[Float32](length=NACT, fill=Float32(0))

    var rows = 0
    for _ep in range(n_episodes):
        _ = env.reset()
        for _t in range(ep_len):
            var act = Env.ActionType()
            for k in range(NACT):
                var v = random_float64() * 2.0 - 1.0
                act.data[k] = v
                abuf[k] = Float32(v)
            _ = env.step(act)
            # State AFTER the step, paired with the action that produced it.
            for i in range(NQ):
                qbuf[i] = Float32(Float64(env.d.qpos.data[i]))
            for i in range(NV):
                vbuf[i] = Float32(Float64(env.d.qvel.data[i]))
            w.append[DType.float32](
                String("qpos"), qbuf.unsafe_ptr().as_unsafe_any_origin(), 1
            )
            w.append[DType.float32](
                String("qvel"), vbuf.unsafe_ptr().as_unsafe_any_origin(), 1
            )
            w.append[DType.float32](
                String("action"), abuf.unsafe_ptr().as_unsafe_any_origin(), 1
            )
            rows += 1
        w.end_episode()
    w.close()
    return CollectStats(rows, n_episodes)
