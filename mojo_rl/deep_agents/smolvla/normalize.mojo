"""The boundary around the network: dataset stats in, robot units out.

`state_proj` is `Linear[32, 960]` and `action_out_proj` emits 32 dims, but the
SO-101 has **6** joints. The reference closes that gap in two steps that are
easy to get subtly wrong and impossible to see afterwards -- a mis-normalised
state is a plausible pose and a mis-unnormalised action is a plausible motion.

    raw qpos [6]  -> (x - mean) / (std + 1e-8)  -> zero-pad to [32]
    chunk [50,32] -> take the first 6           -> x * std + mean

⚠ **THE TWO DIRECTIONS ARE NOT EXACT INVERSES, AND THAT IS THE REFERENCE.**
`normalize_processor.py:359` divides by `std + eps` but the inverse branch is
`tensor * std + mean` -- **no eps**.

⚠⚠ **AND NO GATE BELOW CAN SEE THIS.** For this dataset the analytic gap is
`(x-mean)*eps/std` = **1.35e-08**, while one float32 ulp at that magnitude is
**3.8e-06** -- 280x larger. Round-tripping a value through both directions
returns it bit-identically whether the eps is in one, both, or neither. So this
is transcribed on discipline alone: a deviation that is invisible today is
still a deviation, and the moment anything here moves to float64 it stops being
invisible. Do not "fix" the asymmetry, and do not write a gate claiming to
check it -- that gate would pass on every variant and prove nothing.

The reference normalises upstream (the processor pipeline) and pads inside the
policy (`modeling_smolvla.py:412`). The order is not a hazard -- the stats are
6-dim, so "pad first, then normalise" cannot be written -- but the padded dims
landing at exactly **0** is: after mean/std they sit at the dataset mean, which
is what the fine-tune saw.

⚠ **The inverse must DROP the padded dims, not unnormalise them.** The model
emits values above `action_dim`, and `x*std+mean` turns them into perfectly
plausible joint angles for joints that do not exist.

⚠ `prepare_state` takes `batch[OBS_STATE][:, -1, :]` when the observation is
stacked -- the **LAST** frame, not the first. Single-frame today; the moment a
history stack appears, `[0]` is a silent one-tick lag.

⚠ **IMAGES ARE NOT NORMALISED HERE.** SmolVLA's `normalization_mapping` maps
VISUAL to `IDENTITY`; the only thing done to pixels is the `*2-1` in
`vision/resize_pad.mojo`. ACT's ImageNet mean/std does NOT apply, and applying
it would be a silent, plausible-looking corruption of every frame.

⚠ **THE STATS COME FROM THE DATASET, NOT FROM US.** `meta/stats.json` is what a
LeRobot fine-tune normalises with, so it is what deployment must normalise with.
Our importer computes its own from the rows and they are close but not equal:
measured on `record-test_20260828_092736`, the means agree to 3e-5 but the
**stds differ by exactly sqrt(N/(N-1)) = 1.0000324** -- LeRobot takes the
population std, `data/lerobot.mojo` deliberately takes the sample std to match
`torch.std`'s unbiased default for the ACT path. Both are defensible; only one
is what this checkpoint was fine-tuned against. Hence `from_stats_json`.
"""

from mojo_rl.io.json import JsonDoc, load_json

comptime NORM_EPS: Float32 = 1.0e-8
"""`NormalizeProcessorStep.eps`. Forward only -- see the header."""

comptime SMOLVLA_MAX_STATE: Int = 32
"""`config.max_state_dim`."""

comptime SMOLVLA_MAX_ACTION: Int = 32
"""`config.max_action_dim`."""


struct SmolVLAStats(Movable):
    """`meta/stats.json`'s mean/std for `observation.state` and `action`."""

    var state_mean: List[Float32]
    var state_std: List[Float32]
    var action_mean: List[Float32]
    var action_std: List[Float32]

    def __init__(out self):
        self.state_mean = List[Float32]()
        self.state_std = List[Float32]()
        self.action_mean = List[Float32]()
        self.action_std = List[Float32]()

    def __init__(out self, *, deinit move: Self):
        self.state_mean = move.state_mean^
        self.state_std = move.state_std^
        self.action_mean = move.action_mean^
        self.action_std = move.action_std^

    def state_dim(self) -> Int:
        return len(self.state_mean)

    def action_dim(self) -> Int:
        return len(self.action_mean)

    @staticmethod
    def from_stats_json(path: String) raises -> Self:
        """Read `<dataset>/meta/stats.json`.

        A missing key raises rather than defaulting to mean 0 / std 1: an
        unnormalised state is finite, in range, and wrong, and the policy would
        simply behave badly with nothing to point at."""
        var doc = load_json(path)
        var s = Self()
        _read_vec(doc, "observation.state", "mean", s.state_mean)
        _read_vec(doc, "observation.state", "std", s.state_std)
        _read_vec(doc, "action", "mean", s.action_mean)
        _read_vec(doc, "action", "std", s.action_std)
        if len(s.state_mean) != len(s.state_std):
            raise Error("smolvla stats: state mean/std lengths disagree")
        if len(s.action_mean) != len(s.action_std):
            raise Error("smolvla stats: action mean/std lengths disagree")
        # A zero std means a joint never moved in the recording. Dividing by
        # eps alone would send that column to ~1e8 and the tower with it, so
        # name it here instead of finding it in the activations.
        for i in range(len(s.state_std)):
            if s.state_std[i] <= 0.0:
                raise Error(
                    "smolvla stats: observation.state["
                    + String(i)
                    + "] has std 0 — that joint never moved in the recording"
                )
        for i in range(len(s.action_std)):
            if s.action_std[i] <= 0.0:
                raise Error(
                    "smolvla stats: action["
                    + String(i)
                    + "] has std 0 — that joint never moved in the recording"
                )
        return s^


def _read_vec(
    ref doc: JsonDoc, key: String, field: String, mut out: List[Float32]
) raises:
    var root = doc.root()
    var node = doc.field(root, key)
    if node < 0:
        raise Error("smolvla stats: no '" + key + "' in stats.json")
    var arr = doc.field(node, field)
    if arr < 0:
        raise Error(
            "smolvla stats: '" + key + "' has no '" + field + "'"
        )
    out.clear()
    for i in range(doc.size(arr)):
        out.append(Float32(doc.number(doc.at(arr, i))))


def normalize_state(
    ref stats: SmolVLAStats,
    ref raw: List[Float32],
    mut out: List[Float32],
    max_dim: Int = SMOLVLA_MAX_STATE,
) raises:
    """`(x - mean) / (std + eps)`, then zero-padded to `max_dim`."""
    var d = stats.state_dim()
    if len(raw) != d:
        raise Error(
            "normalize_state: got "
            + String(len(raw))
            + " values, stats describe "
            + String(d)
        )
    if d > max_dim:
        raise Error(
            "normalize_state: state dim "
            + String(d)
            + " exceeds max_state_dim "
            + String(max_dim)
        )
    out.resize(max_dim, 0.0)
    for i in range(max_dim):
        out[i] = 0.0
    for i in range(d):
        out[i] = (raw[i] - stats.state_mean[i]) / (
            stats.state_std[i] + NORM_EPS
        )


def unnormalize_action(
    ref stats: SmolVLAStats,
    ref chunk: List[Float32],
    steps: Int,
    mut out: List[Float32],
    max_dim: Int = SMOLVLA_MAX_ACTION,
) raises:
    """`[steps, max_dim]` -> `[steps, action_dim]` in robot units.

    ⚠ `x * std + mean`, with NO eps — the reference's inverse branch omits it.
    The padded dims above `action_dim` are dropped, not unnormalised: the model
    emits values there and they mean nothing."""
    var d = stats.action_dim()
    if len(chunk) < steps * max_dim:
        raise Error(
            "unnormalize_action: chunk holds "
            + String(len(chunk))
            + " values, needs "
            + String(steps * max_dim)
        )
    out.resize(steps * d, 0.0)
    for t in range(steps):
        for i in range(d):
            out[t * d + i] = (
                chunk[t * max_dim + i] * stats.action_std[i]
                + stats.action_mean[i]
            )
