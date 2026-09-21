"""Demonstrations into an off-policy replay, as its pinned prefix.

The HIL-SERL / RLPD recipe: the demo transitions are added BEFORE the loop,
pinned (`SampleBlock.pin_demo_prefix`), and half of every minibatch is drawn
from them for the whole run while the online ring never overwrites them.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.training.blocks.sample_block import SampleBlock
from .file import read_demo_file
from .filter import DemoFilter


struct DemoLoadReport(Copyable, Movable, ImplicitlyCopyable):
    var n_rows: Int
    """Rows added and pinned — after the filter."""
    var n_file_rows: Int
    """Rows in the files, before the filter."""
    var mean_reward: Float64
    """Mean reward of the kept rows."""

    def __init__(out self, n_rows: Int, n_file_rows: Int, mean_reward: Float64):
        self.n_rows = n_rows
        self.n_file_rows = n_file_rows
        self.mean_reward = mean_reward


def load_demos_into_replay[S: SampleBlock](
    mut blk: S,
    paths: List[String],
    filter: DemoFilter,
    ctx: Optional[DeviceContext],
    who: String = "demos",
) raises -> DemoLoadReport:
    """Every kept row of every file in `paths` into `blk`, then pinned.

    ⚠ THROUGH THE SAMPLE BLOCK, NOT `trainer.record`: `record` also feeds the
    episode tracker, and a demo is not an episode this run played. Call after
    the trainer's `setup` and before the first online transition — and before
    the first `train_step`, which a CUDA graph may capture.

    Refuses a file whose obs / act widths are not the block's (recorded on
    another family?) and a filter that keeps nothing."""
    var d_obs = List[Scalar[DT]](length=S.OBS, fill=Scalar[DT](0))
    var d_nxt = List[Scalar[DT]](length=S.OBS, fill=Scalar[DT](0))
    var d_act = List[Scalar[DT]](length=S.ACT, fill=Scalar[DT](0))
    var n_rows = 0
    var n_file_rows = 0
    var sum_r = 0.0
    for pi in range(len(paths)):
        var ds = read_demo_file(paths[pi])
        print("  demo file:", paths[pi], "—", ds.summary())
        if ds.obs_dim != S.OBS or ds.act_dim != S.ACT:
            raise Error(
                who + ": " + paths[pi] + " is obs " + String(ds.obs_dim)
                + " / act " + String(ds.act_dim) + " but this env is "
                + String(S.OBS) + " / " + String(S.ACT)
                + " — recorded on another family?"
            )
        n_file_rows += ds.count()
        for r in range(ds.count()):
            if not filter.keeps(ds, r):
                continue
            ds.row_obs[DT](r, d_obs)
            ds.row_act[DT](r, d_act)
            ds.row_next_obs[DT](r, d_nxt)
            blk.add(
                d_obs, d_act, Scalar[DT](ds.rew[r]), d_nxt,
                Scalar[DT](ds.done[r]), ctx=ctx,
            )
            sum_r += Float64(ds.rew[r])
            n_rows += 1
    if n_rows == 0:
        raise Error(
            who + ": --demos loaded " + String(n_file_rows)
            + " rows and the filter '" + filter.name() + "' kept none"
        )
    blk.pin_demo_prefix(n_rows, ctx=ctx)
    if ctx:
        ctx.value().synchronize()
    return DemoLoadReport(n_rows, n_file_rows, sum_r / Float64(n_rows))
