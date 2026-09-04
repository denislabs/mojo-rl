# +--------------------------------------------------------------------------+ #
# | SmolVLA — batches of (state, action chunk, validity) from a store
# +--------------------------------------------------------------------------+ #
"""The sampling half of a fine-tune: which rows, which chunk, what padding.

Per batch element this draws a row `g`, then produces

    state    [SDIM_PAD]         (qpos[g] - mean) / (std + eps), zero-padded
    actions  [CHUNK, ADIM_PAD]  rows g .. g+CHUNK-1, normalized, zero-padded
    valid    [CHUNK]            1.0 while inside the episode, else 0.0
    task     the row's task_index, for the instruction table

## ⚠ The pad value is the LAST REAL ACTION, not zero — and here that matters

`ACTDataset` pads its chunk with a normalized zero, because the original ACT
`utils.py` pads before normalizing. `lerobot`'s reader does something else:
`_get_query_indices` CLAMPS the query index,

    max(ep_start, min(ep_end - 1, abs_idx + delta))

so a chunk running off the end of its episode repeats the last real action.

For ACT the difference is invisible: those slots feed the L1 loss and nothing
else, and the loss masks them. **For SmolVLA it is not**, because the action
chunk is also a network INPUT:

    x_t = t*noise + (1 - t)*actions     ->  action_in  ->  the expert

Every one of the 50 action tokens attends to every other. A padded slot whose
value is 0 rather than the last real action produces a different `x_t`, a
different suffix embedding, and therefore a different gradient on the VALID
timesteps too. Masking the loss does not undo it.

So this clamps, and `test_dataset_chunks.mojo` asserts the padded rows equal
the last real one AND that zero-padding would be observably different — the
second half because "it equals the last action" is also true of a fixture
whose last action happens to be zero.

⚠ `valid` is 1.0 for real, 0.0 for padding — the INVERSE of `action_is_pad`,
and the same polarity `ACTDataset` already uses. One convention in this repo.

## What this does NOT do

Images. They need the SigLIP tower and `resize_with_pad`, both of which exist
(`vision.mojo`, `vision/resize_pad.mojo`), and wiring them is the prefix
build rather than the sampler. The store column is `images`, uint8 CHW, and
`ACTDataset.image_row_u8` already streams one row of it.
"""

from std.math import sqrt

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.data.store import TrajectoryStore

from .normalize import SmolVLAStats


@always_inline
def _splitmix64(mut state: UInt64) -> UInt64:
    """The same local, seeded generator `ACTDataset` uses, and for the same
    reason: a gate that compares against a recorded index sequence has to
    reproduce it, which the global RNG cannot promise."""
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


struct SmolVLABatchSampler[
    SDIM: Int, ADIM_REAL: Int, PAD: Int, CHUNK: Int, B: Int
](Movable):
    """`SDIM`/`ADIM_REAL` are the robot's; `PAD` is SmolVLA's 32."""

    comptime SN: Int = Self.B * Self.PAD
    comptime AN: Int = Self.B * Self.CHUNK * Self.PAD
    comptime VN: Int = Self.B * Self.CHUNK

    var store: TrajectoryStore
    var qpos: List[Scalar[DT]]
    var action: List[Scalar[DT]]
    var task_index: List[Scalar[DType.int32]]
    var stats: SmolVLAStats
    var rng: UInt64

    def __init__(
        out self, var path: String, var stats: SmolVLAStats, seed: UInt64 = 0
    ) raises:
        comptime assert Self.ADIM_REAL <= Self.PAD, (
            "SmolVLABatchSampler: the robot's action width cannot exceed"
            " SmolVLA's padded width"
        )
        comptime assert Self.SDIM <= Self.PAD, (
            "SmolVLABatchSampler: the robot's state width cannot exceed"
            " SmolVLA's padded width"
        )
        self.store = TrajectoryStore(path^)
        self.qpos = self.store.load_column[DT](String("qpos"))
        self.action = self.store.load_column[DT](String("action"))
        self.task_index = self.store.load_column[DType.int32](
            String("task_index")
        )
        self.stats = stats^
        self.rng = seed if seed != 0 else UInt64(0x243F6A8885A308D3)

        var rows = self.store.n_rows()
        if len(self.qpos) != rows * Self.SDIM:
            raise Error(
                "SmolVLABatchSampler: qpos is "
                + String(len(self.qpos) // rows) + " wide, expected "
                + String(Self.SDIM)
            )
        if len(self.action) != rows * Self.ADIM_REAL:
            raise Error(
                "SmolVLABatchSampler: action is "
                + String(len(self.action) // rows) + " wide, expected "
                + String(Self.ADIM_REAL)
            )
        if self.stats.state_dim() != Self.SDIM:
            raise Error(
                "SmolVLABatchSampler: stats carry "
                + String(self.stats.state_dim())
                + " state dims, the store has " + String(Self.SDIM)
            )
        if self.stats.action_dim() != Self.ADIM_REAL:
            raise Error(
                "SmolVLABatchSampler: stats carry "
                + String(self.stats.action_dim())
                + " action dims, the store has " + String(Self.ADIM_REAL)
            )

    def __init__(out self, *, deinit move: Self):
        self.store = move.store^
        self.qpos = move.qpos^
        self.action = move.action^
        self.task_index = move.task_index^
        self.stats = move.stats^
        self.rng = move.rng

    def n_rows(self) -> Int:
        return self.store.n_rows()

    def sample(
        mut self,
        mut state: Tensor,
        mut actions: Tensor,
        mut valid: Tensor,
        mut tasks: List[Int],
    ) raises -> Int:
        """Draw `B` rows. Returns `n_valid`, the count of in-episode steps.

        ⚠ Every row is a legal anchor, including the last of an episode — the
        reference samples over all frames and pads. Restricting to rows with
        `CHUNK` steps left would silently drop the ends of every episode,
        which is where the interesting part of a manipulation task is.
        """
        state.ensure(Self.SN)
        actions.ensure(Self.AN)
        valid.ensure(Self.VN)
        tasks.clear()

        var n_valid = 0
        for b in range(Self.B):
            var g = Int(_splitmix64(self.rng) % UInt64(self.store.n_rows()))
            var ep = self.store.episodes.episode_of(g)
            var ep_end = self.store.episodes.end_of(ep)
            tasks.append(Int(self.task_index[g]))

            # state: normalize the real dims, ZERO the rest
            for j in range(Self.PAD):
                var v = Scalar[DT](0)
                if j < Self.SDIM:
                    v = (
                        self.qpos[g * Self.SDIM + j] - self.stats.state_mean[j]
                    ) / (self.stats.state_std[j] + Scalar[DT](1e-8))
                state.data[b * Self.PAD + j] = v

            for t in range(Self.CHUNK):
                var row = g + t
                var inside = row < ep_end
                # ⚠ CLAMP, do not skip: past the end the reference repeats the
                # last real row, and that value reaches the network through
                # x_t even though the loss masks it.
                if not inside:
                    row = ep_end - 1
                var ab = b * Self.CHUNK * Self.PAD + t * Self.PAD
                for j in range(Self.PAD):
                    var v = Scalar[DT](0)
                    if j < Self.ADIM_REAL:
                        v = (
                            self.action[row * Self.ADIM_REAL + j]
                            - self.stats.action_mean[j]
                        ) / (self.stats.action_std[j] + Scalar[DT](1e-8))
                    actions.data[ab + j] = v
                valid.data[b * Self.CHUNK + t] = Scalar[DT](
                    1.0
                ) if inside else Scalar[DT](0.0)
                if inside:
                    n_valid += 1

        if n_valid == 0:
            raise Error(
                "SmolVLABatchSampler.sample: no in-episode timestep in the"
                " whole batch — the episode index and the row count disagree"
            )
        return n_valid
