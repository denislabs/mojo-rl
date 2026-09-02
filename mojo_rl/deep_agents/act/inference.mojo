# +--------------------------------------------------------------------------+ #
# | Temporal ensembling — combining overlapping action chunks
# +--------------------------------------------------------------------------+ #
"""ACT queries the policy at EVERY timestep, so at step `t` there are up to `K`
chunks that each predicted an action for `t`. Temporal ensembling combines them
with an exponential weighting (paper Algorithm 2, `imitate_episodes.py:248`):

    a_t = sum_i w_i * A_t[i] / sum_i w_i,     w_i = exp(-m * i)

⚠⚠ **`i = 0` is the OLDEST chunk, so the OLDEST prediction gets the LARGEST
weight.** The paper states it outright ("where `w_0` is the weight for the
oldest action") and the reference implements it by gathering
`all_time_actions[:, t]` in ascending query order, whose first populated row is
the earliest query. Reading `exp(-m*i)` as "decay toward older" — the intuitive
reading, and the wrong one — silently inverts the smoothing: it would weight the
freshest, least-settled prediction most and produce exactly the jerkiness
ensembling exists to remove. `m` smaller means slower incorporation of new
observations.

## Why not the reference's storage

The reference allocates `[max_timesteps, max_timesteps + K, state_dim]` — quadratic
in episode length — and detects which entries are real with
`torch.all(actions != 0, axis=1)`.

That test has a latent bug: a genuine prediction whose every component happened
to be exactly 0.0 would be treated as absent. It never fires on float
predictions, so the reference is fine in practice, but reproducing a bug is not
the same as reproducing behaviour. Here occupancy is tracked explicitly, and the
buffer is a ring of the last `K` chunks — only queries in `[t-K+1, t]` can
contribute to step `t`, so nothing older needs keeping. Same arithmetic, `K x K`
instead of `T x (T+K)`.
"""

from std.math import exp

from mojo_rl.nn.constants import DT

from .config import (
    ACT_TEMPORAL_ENSEMBLE_M,
    IMAGENET_MEAN_B,
    IMAGENET_MEAN_G,
    IMAGENET_MEAN_R,
    IMAGENET_STD_B,
    IMAGENET_STD_G,
    IMAGENET_STD_R,
)


struct TemporalEnsemble[ADIM: Int, K: Int](Movable & Deinitable):
    """Ring of the last `K` predicted chunks, combined per timestep."""

    var chunks: List[Scalar[DT]]
    """`K` slots of `K*ADIM`. Slot `i % K` holds the chunk queried at step `i`."""
    var query_step: List[Int]
    """Which step produced each slot; `-1` = never filled."""
    var m: Float64

    def __init__(out self, m: Float64 = ACT_TEMPORAL_ENSEMBLE_M):
        comptime assert Self.K > 0 and Self.ADIM > 0, (
            "TemporalEnsemble: K and ADIM must be > 0"
        )
        self.chunks = List[Scalar[DT]](
            length=Self.K * Self.K * Self.ADIM, fill=Scalar[DT](0.0)
        )
        self.query_step = List[Int](length=Self.K, fill=-1)
        self.m = m

    def __init__(out self, *, deinit move: Self):
        self.chunks = move.chunks^
        self.query_step = move.query_step^
        self.m = move.m

    def reset(mut self):
        for i in range(Self.K):
            self.query_step[i] = -1

    def push(
        mut self, t: Int, ref chunk: List[Scalar[DT]], offset: Int = 0
    ) raises:
        """Record the chunk predicted AT step `t`.

        `chunk[offset : offset + K*ADIM]` is the `K x ADIM` prediction, in
        normalized units — the whole ensemble runs in normalized space and the
        caller denormalizes the single combined action, which is what the
        reference does (`post_process` is applied to `raw_action`).
        """
        var slot = t % Self.K
        self.query_step[slot] = t
        var base = slot * Self.K * Self.ADIM
        for i in range(Self.K * Self.ADIM):
            self.chunks[base + i] = chunk[offset + i]

    def action_at(
        mut self, t: Int, mut out: List[Scalar[DT]], out_offset: Int = 0
    ) raises:
        """The ensembled action for step `t`, written to `out[out_offset:]`.

        Contributions come from every query `i` in `[t-K+1, t]` that has
        actually run; query `i` predicted step `t` at chunk index `t - i`.
        """
        if len(out) < out_offset + Self.ADIM:
            raise Error("TemporalEnsemble.action_at: output buffer too small")

        var i_min = t - Self.K + 1
        if i_min < 0:
            i_min = 0

        # Pass 1: total weight. The exponent is the offset from the WINDOW
        # START, `i - i_min`, so the oldest step the window can reach takes
        # weight exp(0) = 1 and newer queries decay from there.
        #
        # ⚠ THAT IS THE RANK ONLY WHEN QUERIES ARE DENSE. Training, the
        # open-loop evaluation and the reference all query at EVERY step, so
        # every `i` in the window is populated and `i - i_min` IS the position
        # in ascending query order — which is what the paper's `w_i = exp(-k*i)`
        # indexes. `act_so101_deploy_real.mojo` cannot query every step (one
        # forward is ~95 ms against a 30 Hz grid), so its window is sparse and
        # the exponent becomes an AGE IN GRID STEPS rather than a rank.
        #
        # That is the right generalisation and not an accident: two chunks
        # queried 1 step apart should not be weighted as differently as two
        # queried 20 steps apart merely because nothing was queried in
        # between. It is a DIVERGENCE FROM A RANK-BASED READING though, and
        # with `m = 0.01` over `K = 60` the whole weight range is 1.0 down to
        # 0.55, so the two readings differ by a few percent at most.
        var wsum = Float64(0.0)
        var n = 0
        for i in range(i_min, t + 1):
            var slot = i % Self.K
            if self.query_step[slot] != i:
                continue  # that query never ran (or has been overwritten)
            wsum += exp(-self.m * Float64(i - i_min))
            n += 1
        if n == 0:
            raise Error(
                "TemporalEnsemble.action_at: no chunk covers step "
                + String(t)
                + " — push() the query for this step first"
            )

        for j in range(Self.ADIM):
            out[out_offset + j] = Scalar[DT](0.0)
        for i in range(i_min, t + 1):
            var slot = i % Self.K
            if self.query_step[slot] != i:
                continue
            var w = exp(-self.m * Float64(i - i_min)) / wsum
            var base = slot * Self.K * Self.ADIM + (t - i) * Self.ADIM
            for j in range(Self.ADIM):
                out[out_offset + j] += Scalar[DT](w) * self.chunks[base + j]

    def n_contributors(self, t: Int) -> Int:
        """How many chunks cover step `t` — the diagnostic that distinguishes
        "the ensemble is warming up" from "the ring is not being filled"."""
        var i_min = t - Self.K + 1
        if i_min < 0:
            i_min = 0
        var n = 0
        for i in range(i_min, t + 1):
            if self.query_step[i % Self.K] == i:
                n += 1
        return n


def denormalize(
    ref src: List[Scalar[DT]],
    offset: Int,
    ref mean: List[Scalar[DT]],
    ref std: List[Scalar[DT]],
    mut out: List[Scalar[DT]],
    out_offset: Int,
    dim: Int,
) raises:
    """`a * std + mean` — back to the units the robot speaks.

    `imitate_episodes.py:172 post_process`. Applied to the ENSEMBLED action, not
    to each chunk: the weighting is affine and the normalization is affine, so
    the order does not change the result, but denormalizing once is cheaper and
    keeps the ensemble in one unit system.
    """
    for j in range(dim):
        out[out_offset + j] = src[offset + j] * std[j] + mean[j]


def normalize_camera_chw[
    IMG_H: Int, IMG_W: Int
](
    ref src: List[Scalar[DType.uint8]],
    src_off: Int,
    mut dst: List[Scalar[DT]],
    dst_off: Int,
) raises:
    """One camera's `[3, H, W]` uint8 CHW slot -> `/255` -> ImageNet normalize.

    ⚠⚠ **THIS IS SHARED BECAUSE THE DEPLOYMENT PATH MUST NOT REIMPLEMENT IT.**
    `ACTDataset._fill_one` prepares the images the model TRAINS on and
    `act_so101_deploy_real.mojo` prepares the ones it is DEPLOYED on, and a
    divergence between them does not raise, does not fail a gate and does not
    look like a bug: it looks like a policy that behaves worse on the robot
    than it did on the dataset, which is the most expensive thing to debug in
    this project. A rule written inline twice drifts — so it is written once.

    ⚠ `* (1/std)`, NOT `/ std`. The two differ in the last bit and the training
    tensors were built with the reciprocal; matching it is free.
    """
    comptime HW = IMG_H * IMG_W
    if len(src) < src_off + 3 * HW:
        raise Error("normalize_camera_chw: source slot is short")
    if len(dst) < dst_off + 3 * HW:
        raise Error("normalize_camera_chw: destination slot is short")
    for ch in range(3):
        var mean = Scalar[DT](IMAGENET_MEAN_R) if ch == 0 else (
            Scalar[DT](IMAGENET_MEAN_G) if ch == 1 else Scalar[DT](
                IMAGENET_MEAN_B
            )
        )
        var std = Scalar[DT](IMAGENET_STD_R) if ch == 0 else (
            Scalar[DT](IMAGENET_STD_G) if ch == 1 else Scalar[DT](
                IMAGENET_STD_B
            )
        )
        var inv = Scalar[DT](1.0) / std
        var base = ch * HW
        for p in range(HW):
            var v = Scalar[DT](Int(src[src_off + base + p])) / Scalar[DT](255.0)
            dst[dst_off + base + p] = (v - mean) * inv
