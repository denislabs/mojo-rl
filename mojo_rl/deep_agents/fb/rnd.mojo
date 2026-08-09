"""Random Network Distillation — novelty as a measurable quantity.

Two MLPs of identical shape. `target` is initialised once and **never
trained**; `predictor` is trained to imitate it on whatever states it is shown.
The prediction error

    novelty(s) = || predictor(s) - target(s) ||^2

is then large on states the predictor has not seen and small on states it has.
That is the entire mechanism (Burda et al., 2018).

## What this is for in M2

`docs/BFM_ZERO_SHOT_RL.md` component 1 lists RND as the weakest of four
diversity levers and — more importantly — demands something else outright:

> Coverage metrics to compute BEFORE training FB: k-NN entropy over states,
> histograms of torso height / velocities / joint angles. Otherwise you
> discover the dataset is poor after 2 M gradient steps.

RND is the cheapest such metric that works in high dimension, and that is its
first job here: `mean_novelty` over a held-out slice of a dataset, AFTER the
predictor has been fitted on a training slice, says how much of the state space
the dataset actually covers. `intrinsic` is also the reward an exploration
agent would maximise, but note what that would take: a full online RL loop on
CPU envs. This module supplies the signal; it is not itself an ExORL agent, and
calling it one would overstate what has been built.

## The normalisation, which is the part that actually fails

⚠⚠ The trap is not the algorithm, it is the normalisation — on the OBSERVATIONS
and on the intrinsic reward. Without both, RND does not converge: the predictor
chases whichever input dimension happens to have the largest scale, and the
reward's magnitude drifts by orders of magnitude over training.

`RunningNorm` below implements the Chan et al. parallel update, **the same math
as `core/obs_norm.mojo`'s `update_obs_norm_kernel`**, deliberately — the two
must agree if a dataset is ever normalised on one path and consumed on the
other.

⚠ Why it is not simply `ObsNormStats`, which the plan says to reuse: that struct
is GPU-RESIDENT. Its statistics live in `DeviceBuffer`s, `_update` is a kernel,
and the host mirror is refreshed by `sync_host` for `apply_cpu` only — there is
no host-side update path at all. dm_control collection is CPU-only (gap G10),
so an RND driven during collection cannot use it without a device round trip
per batch. `ObsNormStats` remains the right tool on the GPU side; this is its
host twin, with the same `count_prior = 1e3` so the first batches do not move
the statistics by orders of magnitude.

⚠ `tdmpc2/running_scale.mojo` is NOT an alternative: it is an EMA of a 95-5
inter-percentile range, built to commensurate Q values, not a standard
deviation.

## Scope

CPU. Collection is CPU-only, which is where the novelty signal is consumed, and
scoring a coverage slice (~1e5 rows through a small MLP) is not the expensive
part of anything. The GPU extension is mechanical — the same `*_t` helpers the
trainer uses, with `ObsNormStats` in place of `RunningNorm` — and should be
written when something actually needs it, not before.
"""

from std.math import sqrt

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.call import call_forward, call_vjp
from mojo_rl.nn.core.initializer import Initializer, Xavier
from mojo_rl.nn.optimizer.adam import Adam


struct RunningNorm[DIM: Int](Movable & Deinitable):
    """Per-dimension running mean/variance, Chan et al. parallel update.

    Host twin of `core/obs_norm.mojo`. `count_prior` defaults to 1e3 for the
    reason that file gives: without it the first few batches move the running
    statistics by orders of magnitude, and everything downstream sees a moving
    target.
    """

    var mean: List[Float64]
    var var_: List[Float64]
    var count: Float64
    var eps: Float64
    var frozen: Bool

    def __init__(out self, count_prior: Float64 = 1e3, eps: Float64 = 1e-8):
        self.mean = List[Float64](length=Self.DIM, fill=0.0)
        self.var_ = List[Float64](length=Self.DIM, fill=1.0)
        self.count = count_prior
        self.eps = eps
        self.frozen = False

    def __init__(out self, *, deinit move: Self):
        self.mean = move.mean^
        self.var_ = move.var_^
        self.count = move.count
        self.eps = move.eps
        self.frozen = move.frozen

    def update(mut self, ref x: Tensor, n_rows: Int) raises:
        """Merge `[n_rows, DIM]` into the running statistics."""
        if self.frozen or n_rows <= 0:
            return
        var n = Float64(n_rows)
        for d in range(Self.DIM):
            var batch_mean = Float64(0)
            for r in range(n_rows):
                batch_mean += Float64(x.data[r * Self.DIM + d])
            batch_mean /= n

            var batch_m2 = Float64(0)
            for r in range(n_rows):
                var diff = Float64(x.data[r * Self.DIM + d]) - batch_mean
                batch_m2 += diff * diff

            var old_mean = self.mean[d]
            var old_var = self.var_[d]
            var old_count = self.count
            var new_count = old_count + n
            var delta = batch_mean - old_mean
            var new_mean = old_mean + delta * n / new_count
            var m2_old = old_var * old_count
            var m2_new = (
                m2_old + batch_m2 + delta * delta * old_count * n / new_count
            )
            self.mean[d] = new_mean
            self.var_[d] = m2_new / new_count
        self.count += n

    def apply(self, ref x: Tensor, n_rows: Int, mut dst: Tensor) raises:
        """`dst = (x - mean) / sqrt(var + eps)`, `[n_rows, DIM]`."""
        dst.ensure(n_rows * Self.DIM)
        for r in range(n_rows):
            for d in range(Self.DIM):
                var inv = 1.0 / sqrt(self.var_[d] + self.eps)
                dst.data[r * Self.DIM + d] = Scalar[DT](
                    (Float64(x.data[r * Self.DIM + d]) - self.mean[d]) * inv
                )

    def freeze(mut self):
        self.frozen = True


struct RNDStats(Movable & Deinitable):
    """What one `fit` call did. `loss` is the predictor's MSE against the
    frozen target; `mean_novelty` is that same quantity before normalisation,
    which is the coverage number worth reporting."""

    var loss: Float64
    var mean_novelty: Float64

    def __init__(out self, loss: Float64, mean_novelty: Float64):
        self.loss = loss
        self.mean_novelty = mean_novelty

    def __init__(out self, *, deinit move: Self):
        self.loss = move.loss
        self.mean_novelty = move.mean_novelty


struct RND[NET: Module, OBS: Int, FEAT: Int, BATCH: Int](
    Movable & Deinitable
):
    """Frozen target + trained predictor, both `NET` ([OBS] -> [FEAT]).

    ⚠ `target` must be initialised DIFFERENTLY from `predictor`, or novelty is
    identically zero everywhere and the whole signal vanishes silently. `make`
    enforces this by construction and `test_fb_rnd.mojo` gates it: the two nets
    are built from the same factory but the initializer is stateful, so
    consecutive `make` calls draw different weights. A `Deterministic`
    initializer would break that — which is exactly why the gate checks the
    initial novelty is non-zero rather than assuming it.
    """

    var target: Self.NET
    var predictor: Self.NET
    var opt: Adam
    var obs_norm: RunningNorm[Self.OBS]
    var rew_norm: RunningNorm[1]
    var _sized: Bool

    var _xn: Tensor
    var _ft: Tensor
    var _fp: Tensor
    var _g: Tensor
    var _nov: Tensor

    def __init__(out self):
        self.target = Self.NET()
        self.predictor = Self.NET()
        self.opt = Adam(lr=Scalar[DT](1e-4))
        self.obs_norm = RunningNorm[Self.OBS]()
        self.rew_norm = RunningNorm[1]()
        self._sized = False
        self._xn = Tensor()
        self._ft = Tensor()
        self._fp = Tensor()
        self._g = Tensor()
        self._nov = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.target = move.target^
        self.predictor = move.predictor^
        self.opt = move.opt^
        self.obs_norm = move.obs_norm^
        self.rew_norm = move.rew_norm^
        self._sized = move._sized
        self._xn = move._xn^
        self._ft = move._ft^
        self._fp = move._fp^
        self._g = move._g^
        self._nov = move._nov^

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](lr: Float64 = 1e-4) raises -> Self:
        var r = Self()
        r.target = Self.NET.make["cpu", INIT](None)
        r.predictor = Self.NET.make["cpu", INIT](None)
        r.opt = Adam(lr=Scalar[DT](lr))
        return r^

    def _size(mut self, n_rows: Int) raises:
        self._xn.ensure(n_rows * Self.OBS)
        self._ft.ensure(n_rows * Self.FEAT)
        self._fp.ensure(n_rows * Self.FEAT)
        self._g.ensure(n_rows * Self.FEAT)
        self._nov.ensure(n_rows)

    def _forward_both[N: Int](mut self, ref x: Tensor) raises:
        """Normalised forward through both nets; fills `_ft`, `_fp`, `_nov`."""
        self._size(N)
        self.obs_norm.apply(x, N, self._xn)

        var pack = TensorPack[1]()
        pack[0].ensure(N * Self.OBS)
        for i in range(N * Self.OBS):
            pack[0].data[i] = self._xn.data[i]

        call_forward["cpu", N](
            self.target, TensorRefs[1, MutAnyOrigin](pack[0]), self._ft, None
        )
        call_forward["cpu", N](
            self.predictor, TensorRefs[1, MutAnyOrigin](pack[0]), self._fp, None
        )
        for r in range(N):
            var acc = Float64(0)
            for k in range(Self.FEAT):
                var d = (
                    Float64(self._fp.data[r * Self.FEAT + k])
                    - Float64(self._ft.data[r * Self.FEAT + k])
                )
                acc += d * d
            self._nov.data[r] = Scalar[DT](acc)

    def novelty[
        N: Int
    ](mut self, ref x: Tensor, mut dst: Tensor) raises -> Float64:
        """Raw `||predictor(s) - target(s)||^2` per row. Returns the mean.

        Does NOT update the observation statistics — a coverage measurement
        must not move the normaliser it is measured under, or two calls on the
        same data disagree.
        """
        self._forward_both[N](x)
        dst.ensure(N)
        var s = Float64(0)
        for r in range(N):
            dst.data[r] = self._nov.data[r]
            s += Float64(self._nov.data[r])
        return s / Float64(N)

    def intrinsic[
        N: Int
    ](mut self, ref x: Tensor, mut dst: Tensor) raises -> Float64:
        """Novelty divided by its own running standard deviation.

        ⚠ This is the number an exploration agent should consume, NOT `novelty`.
        The raw prediction error shrinks by orders of magnitude as the predictor
        fits, so an agent rewarded with it would see its incentive silently
        evaporate.

        ⚠⚠ **It mitigates that, it does not eliminate it.** `rew_norm` is a
        CUMULATIVE estimator (the count grows without bound, as in
        `ObsNormStats`), so its standard deviation reflects the whole history
        rather than the recent scale, and a shrinking novelty still produces a
        shrinking reward. Measured over 300 fit steps
        (`test_fb_rnd.mojo` [4]): raw novelty fell 24x while the normalised
        reward fell 9x. Better, not flat.

        Full scale-stationarity needs an EMA or windowed standard deviation,
        which cumulative statistics cannot express. That is a deliberate
        deferral, not an oversight — it only starts to matter once something
        actually trains a policy on this reward, and at that point the fix is
        a decay factor on `rew_norm.count`.

        The reward statistics ARE updated here (unlike `novelty`), because the
        scale has to track the training run.
        """
        self._forward_both[N](x)
        self.rew_norm.update(self._nov, N)
        var sd = sqrt(self.rew_norm.var_[0] + self.rew_norm.eps)
        dst.ensure(N)
        var s = Float64(0)
        for r in range(N):
            var v = Float64(self._nov.data[r]) / sd
            dst.data[r] = Scalar[DT](v)
            s += v
        return s / Float64(N)

    def fit[N: Int](mut self, ref x: Tensor) raises -> RNDStats:
        """One predictor step towards the frozen target on `[N, OBS]`.

        Updates the OBSERVATION statistics first — this is the call that sees
        the data stream, so it is where the normaliser should learn.
        """
        self.obs_norm.update(x, N)
        self._forward_both[N](x)

        var inv = 1.0 / (Float64(N) * Float64(Self.FEAT))
        var loss = Float64(0)
        var nov = Float64(0)
        for r in range(N):
            nov += Float64(self._nov.data[r])
        for i in range(N * Self.FEAT):
            var d = Float64(self._fp.data[i]) - Float64(self._ft.data[i])
            loss += d * d * inv
            self._g.data[i] = Scalar[DT](2.0 * d * inv)

        var pack = TensorPack[1]()
        pack[0].ensure(N * Self.OBS)
        for i in range(N * Self.OBS):
            pack[0].data[i] = self._xn.data[i]

        # ⚠ ONLY the predictor is stepped. The target must stay frozen — a
        # trained target makes novelty collapse to zero everywhere and the
        # signal disappears without any error. `test_fb_rnd.mojo` asserts the
        # target's output is bit-identical before and after fitting.
        self.predictor.zero_grad["cpu"](None)
        var sink = TensorPack[1]()
        call_vjp["cpu", N](
            self.predictor, TensorRefs[1, MutAnyOrigin](pack[0]), self._g,
            TensorRefs[1, MutAnyOrigin](sink[0]), None,
        )
        self.opt.step["cpu"](self.predictor, None)
        return RNDStats(loss, nov / Float64(N))
