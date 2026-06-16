"""DynamicsEnsembleBlock[DynNet, N, NUM_ELITES, IN_DIM, OUT_DIM, BATCH].

Phase I.1.b. The probabilistic-ensemble world model used by MBPO. Owns
N independent dynamics networks + N independent optimisers + one shared
GaussianNLLLoss instance + per-member elite ranking.

Why a single block (not 7 free-standing nets in the trainer)?
  - Lifecycle uniformity: one `make[target]()` builds and inits all
    members + opts.  Mirrors how `TwinCriticUpdateBlock` bundles
    `(c1, c2, _mb_sa)` rather than asking the trainer to own each
    field individually.
  - Elite-ranking state (`elite_indices`) is per-ensemble, not
    per-member — it belongs to the ensemble's lifetime, not the
    trainer's.
  - Member-indexed scratch (`_mb_pred`, `_mb_grad`) is shared across
    all member calls — one slab per direction, reused per member step.
    Each `train_member_step` reads/writes through `_mb_pred` then
    `_mb_grad` before returning, so members never race on the slab.

Scope (I.1.a/b; GPU added Phase 4.3a 2026-05-30):
  - CPU + GPU.  `make[gpu](ctx)` builds members/opts/loss/scratch on
    device; `predict_member`/`train_member_step`/`eval_member_loss` have
    GPU branches (member.forward/vjp + GaussianNLL kernels + a split/clamp-
    logvar kernel). `train_member_step[gpu]` D2Hs the scalar loss once per
    call — fine on the periodic `model_train_freq` cadence.
  - Fixed logvar bounds `[LOGVAR_MIN, LOGVAR_MAX]`. Reference MBPO
    learns the bounds via L2 regularisation; deferred (it's GPU-tied
    in the production agent and not on the I.1 critical path).
  - No input scaler.  Pendulum's `(cosθ, sinθ, ω̇)` obs is bounded so
    raw inputs work; HalfCheetah-style unbounded obs will need one,
    handled in I.1.* follow-up.
  - Single-pass training per `train_step` call: the trainer chooses
    epoch count by calling `train_member_step` in a loop.  Early-
    stopping logic lives in the trainer if needed.

Trait conventions:
  - `predict_member`: pure forward through `members[member_idx]` and
    in-place logvar clamp; result split into `out_mu` (DIM cols) +
    `out_lv` (DIM cols).
  - `train_member_step`: one Gaussian-NLL gradient step on the named
    member's parameters; returns the scalar loss.  Caller owns the
    mini-batch tensors.
  - `eval_member_loss`: forward + loss only, no gradient — for
    holdout-set scoring.
  - `update_elites`: re-rank members by passed-in holdout losses,
    refresh `elite_indices` (lowest NUM_ELITES losses are elite).
"""

from std.math import exp as fexp, log as flog, sqrt as fsqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module, mptr
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.loss.gaussian_nll_loss import GaussianNLLLoss
from mojo_rl.nn.optimizer.adamw import AdamW


def _split_clamp_logvar_kernel[
    BATCH: Int, OUT_DIM: Int, PRED_DIM: Int
](
    pred: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    out_mu: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
    out_lv: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
    lv_min: Scalar[DT],
    lv_max: Scalar[DT],
):
    """GPU mirror of the CPU split+clamp loop in `predict_member`:
    `out_mu[b,j] = pred[b,j]`, `out_lv[b,j] = clamp(pred[b,PRED+j], lv_min,
    lv_max)`. One thread per (b, j) over BATCH × PRED_DIM."""
    var idx = Int(global_idx.x)
    var total = BATCH * PRED_DIM
    if idx >= total:
        return
    var b = idx // PRED_DIM
    var j = idx % PRED_DIM
    out_mu[b, j] = rebind[Scalar[DT]](pred[b, j])
    var v = rebind[Scalar[DT]](pred[b, PRED_DIM + j])
    if v > lv_max:
        v = lv_max
    elif v < lv_min:
        v = lv_min
    out_lv[b, j] = v


# ──────────────────────────────────────────────────────────────────────
# Learnable per-member/per-dim logvar bounds (MBPO/PETS reference,
# deep_agents/core/agents/mbpo_agent.mojo + kernels.mojo). Faithful port:
# soft DOUBLE-softplus clamp lets gradients flow to BOTH the network logvar
# output AND the bounds, which are Adam-updated with a 0.01 L2 penalty
# (`+0.01·Σmax − 0.01·Σmin`) that learns the upper bound DOWN from +0.5 to
# ≈[−1,−2]. This is the regulariser the fixed nn clamp lacked.
# ──────────────────────────────────────────────────────────────────────


def _softplus_dt(x: Scalar[DT]) -> Scalar[DT]:
    """Numerically-stable softplus log(1+exp(x)) with ±20 saturation
    (matches legacy kernels.mojo)."""
    if x > Scalar[DT](20.0):
        return x
    elif x > Scalar[DT](-20.0):
        return flog(Scalar[DT](1.0) + fexp(x))
    return Scalar[DT](0.0)


def _sigmoid_dt(x: Scalar[DT]) -> Scalar[DT]:
    """Sigmoid 1/(1+exp(-x)) with ±20 saturation = d softplus(x)/dx."""
    if x > Scalar[DT](20.0):
        return Scalar[DT](1.0)
    elif x > Scalar[DT](-20.0):
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + fexp(-x))
    return Scalar[DT](0.0)


def _soft_clamp_lv(
    raw: Scalar[DT], max_d: Scalar[DT], min_d: Scalar[DT]
) -> Scalar[DT]:
    """Double-softplus soft clamp of `raw` into (min_d, max_d):
    `lv = min_d + softplus((max_d − softplus(max_d − raw)) − min_d)`."""
    var lv_inter = max_d - _softplus_dt(max_d - raw)
    return min_d + _softplus_dt(lv_inter - min_d)


def _dyn_learnable_nll_grad_kernel[
    BATCH: Int, PRED_DIM: Int
](
    pred: LayoutTensor[DT, Layout.row_major(BATCH, 2 * PRED_DIM), MutAnyOrigin],
    target: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
    max_lv: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    min_lv: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    grad_out: LayoutTensor[
        DT, Layout.row_major(BATCH, 2 * PRED_DIM), MutAnyOrigin
    ],
    gmax: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
    gmin: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
    ploss: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Combined Gaussian-NLL forward + grad with soft logvar bounds. One
    thread per batch row. Grads use legacy normalisation `2/(BATCH·PRED)`
    so the downstream 0.01 L2 keeps its reference weight; `ploss` keeps
    nn's `Σ_d[½d²σ⁻²+½lv]` convention for metric continuity."""
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var inv_norm = Scalar[DT](2.0) / Scalar[DT](BATCH * PRED_DIM)
    var row_loss = Scalar[DT](0.0)
    for d in range(PRED_DIM):
        var mu = rebind[Scalar[DT]](pred[b, d])
        var raw = rebind[Scalar[DT]](pred[b, PRED_DIM + d])
        var y = rebind[Scalar[DT]](target[b, d])
        var max_d = rebind[Scalar[DT]](max_lv[d])
        var min_d = rebind[Scalar[DT]](min_lv[d])
        var a1 = max_d - raw
        var g1 = _sigmoid_dt(a1)
        var lv_inter = max_d - _softplus_dt(a1)
        var a2 = lv_inter - min_d
        var g2 = _sigmoid_dt(a2)
        var lv = min_d + _softplus_dt(a2)
        var inv_var = fexp(-lv)
        var diff = mu - y
        var dsq = diff * diff
        row_loss += Scalar[DT](0.5) * dsq * inv_var + Scalar[DT](0.5) * lv
        var grad_lv = Scalar[DT](0.5) * (Scalar[DT](1.0) - dsq * inv_var) * inv_norm
        grad_out[b, d] = diff * inv_var * inv_norm
        grad_out[b, PRED_DIM + d] = grad_lv * g1 * g2
        gmax[b, d] = grad_lv * g2 * (Scalar[DT](1.0) - g1)
        gmin[b, d] = grad_lv * (Scalar[DT](1.0) - g2)
    ploss[b] = row_loss


def _dyn_bounds_adam_kernel[
    PRED_DIM: Int, BATCH: Int
](
    max_lv: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    min_lv: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    max_m: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    max_v: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    min_m: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    min_v: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    gmax: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
    gmin: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
    l2_coef: Scalar[DT],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
):
    """Reduce per-batch bound grads, add the L2 term (`+l2` to max pulls it
    down, `−l2` to min pushes it up → bounds tighten toward each other),
    then Adam-step each per-dim bound. One thread per output dim."""
    var d = Int(global_idx.x)
    if d >= PRED_DIM:
        return
    var one = Scalar[DT](1.0)
    var g_max = Scalar[DT](0.0)
    var g_min = Scalar[DT](0.0)
    for b in range(BATCH):
        g_max += rebind[Scalar[DT]](gmax[b, d])
        g_min += rebind[Scalar[DT]](gmin[b, d])
    g_max += l2_coef
    g_min -= l2_coef
    # Adam on max_lv[d].
    var m1 = beta1 * rebind[Scalar[DT]](max_m[d]) + (one - beta1) * g_max
    var v1 = beta2 * rebind[Scalar[DT]](max_v[d]) + (one - beta2) * g_max * g_max
    max_m[d] = m1
    max_v[d] = v1
    max_lv[d] = rebind[Scalar[DT]](max_lv[d]) - lr * (m1 / bc1) / (
        fsqrt(v1 / bc2) + eps
    )
    # Adam on min_lv[d].
    var m2 = beta1 * rebind[Scalar[DT]](min_m[d]) + (one - beta1) * g_min
    var v2 = beta2 * rebind[Scalar[DT]](min_v[d]) + (one - beta2) * g_min * g_min
    min_m[d] = m2
    min_v[d] = v2
    min_lv[d] = rebind[Scalar[DT]](min_lv[d]) - lr * (m2 / bc1) / (
        fsqrt(v2 / bc2) + eps
    )


def _soft_clamp_split_kernel[
    BATCH: Int, OUT_DIM: Int, PRED_DIM: Int
](
    pred: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    max_lv: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    min_lv: LayoutTensor[DT, Layout.row_major(PRED_DIM), MutAnyOrigin],
    out_mu: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
    out_lv: LayoutTensor[DT, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin],
):
    """Soft-clamp variant of `_split_clamp_logvar_kernel` using the member's
    learnable per-dim bounds. Used by `predict_member` (→ rollout/eval) so
    sampled σ² matches the training-time soft clamp."""
    var idx = Int(global_idx.x)
    var total = BATCH * PRED_DIM
    if idx >= total:
        return
    var b = idx // PRED_DIM
    var j = idx % PRED_DIM
    out_mu[b, j] = rebind[Scalar[DT]](pred[b, j])
    var raw = rebind[Scalar[DT]](pred[b, PRED_DIM + j])
    out_lv[b, j] = _soft_clamp_lv(
        raw, rebind[Scalar[DT]](max_lv[j]), rebind[Scalar[DT]](min_lv[j])
    )


struct DynamicsEnsembleBlock[
    DynNet: Module,
    N: Int,
    NUM_ELITES: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    BATCH: Int,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
](Movable & ImplicitlyDeletable):
    """N-member probabilistic dynamics ensemble.

    `DynNet.OUT_DIM` MUST equal `OUT_DIM == 2 * PRED_DIM` where
    PRED_DIM = 1 + obs_dim (reward + Δobs)."""

    comptime PRED_DIM: Int = Self.OUT_DIM // 2

    var members: List[Self.DynNet]
    var opts: List[AdamW]
    var loss: GaussianNLLLoss[Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX]
    var elite_indices: List[Int]

    var _mb_pred: Scratch["mb_pred", Self.BATCH * Self.OUT_DIM]
    var _mb_grad: Scratch["mb_grad", Self.BATCH * Self.OUT_DIM]

    # ─── Learnable logvar bounds (opt-in via `learnable_bounds`) ──────────
    # Per-member/per-dim [N * PRED_DIM]: bounds + Adam moments. Always
    # allocated (cheap: ~N·PRED floats); only USED when `learnable_bounds`
    # is set. Default off ⇒ fixed-clamp path is bit-identical.
    var learnable_bounds: Bool
    var _bnd_lr: Scalar[DT]
    var _bnd_step: Int
    var _max_lv: Scratch["dyn_max_lv", Self.N * Self.PRED_DIM]
    var _min_lv: Scratch["dyn_min_lv", Self.N * Self.PRED_DIM]
    var _max_lv_m: Scratch["dyn_max_lv_m", Self.N * Self.PRED_DIM]
    var _max_lv_v: Scratch["dyn_max_lv_v", Self.N * Self.PRED_DIM]
    var _min_lv_m: Scratch["dyn_min_lv_m", Self.N * Self.PRED_DIM]
    var _min_lv_v: Scratch["dyn_min_lv_v", Self.N * Self.PRED_DIM]
    # Per-member reusable grad scratch [BATCH * PRED_DIM] + loss partial.
    var _bnd_gmax: Scratch["dyn_bnd_gmax", Self.BATCH * Self.PRED_DIM]
    var _bnd_gmin: Scratch["dyn_bnd_gmin", Self.BATCH * Self.PRED_DIM]
    # STAGING ⇒ GPU make also allocates the CPU mirror so the per-row loss
    # can be D2H'd through `cpu_ptr()` for the host-side reduction.
    var _bnd_ploss: Scratch["dyn_bnd_ploss", Self.BATCH, True]

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.OUT_DIM == 2 * Self.PRED_DIM, (
            "DynamicsEnsembleBlock: OUT_DIM must be 2 * PRED_DIM"
        )
        comptime assert Self.NUM_ELITES <= Self.N, (
            "NUM_ELITES must not exceed ensemble size N"
        )
        comptime assert Self.OUT_DIM >= Self.IN_DIM, (
            "DynamicsEnsembleBlock: _mb_pred is reused as grad-input"
            " sink during member.vjp, so OUT_DIM must be >= IN_DIM"
            " (holds for typical MBPO surfaces where OBS > ACT - 2)"
        )
        self.members = List[Self.DynNet]()
        self.opts = List[AdamW]()
        self.loss = GaussianNLLLoss[
            Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX
        ]()
        self.elite_indices = List[Int]()
        self._mb_pred = Scratch["mb_pred", Self.BATCH * Self.OUT_DIM]()
        self._mb_grad = Scratch["mb_grad", Self.BATCH * Self.OUT_DIM]()
        self.learnable_bounds = False
        self._bnd_lr = Scalar[DT](1e-3)
        self._bnd_step = 0
        self._max_lv = Scratch["dyn_max_lv", Self.N * Self.PRED_DIM]()
        self._min_lv = Scratch["dyn_min_lv", Self.N * Self.PRED_DIM]()
        self._max_lv_m = Scratch["dyn_max_lv_m", Self.N * Self.PRED_DIM]()
        self._max_lv_v = Scratch["dyn_max_lv_v", Self.N * Self.PRED_DIM]()
        self._min_lv_m = Scratch["dyn_min_lv_m", Self.N * Self.PRED_DIM]()
        self._min_lv_v = Scratch["dyn_min_lv_v", Self.N * Self.PRED_DIM]()
        self._bnd_gmax = Scratch["dyn_bnd_gmax", Self.BATCH * Self.PRED_DIM]()
        self._bnd_gmin = Scratch["dyn_bnd_gmin", Self.BATCH * Self.PRED_DIM]()
        self._bnd_ploss = Scratch["dyn_bnd_ploss", Self.BATCH, True]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer,
    ]() raises -> Self:
        """CPU factory.  Each member is initialised independently from
        the host RNG so members differ; the ensemble's variance comes
        from initialisation + bootstrap-sample stochasticity in the
        outer training loop."""
        comptime assert target == "cpu", (
            "DynamicsEnsembleBlock.make[target='gpu', INIT] requires DeviceContext"
        )
        comptime assert Self.DynNet.IN_DIMS[0] == Self.IN_DIM, (
            "DynNet.IN_DIM must equal IN_DIM"
        )
        comptime assert Self.DynNet.OUT_DIM == Self.OUT_DIM, (
            "DynNet.OUT_DIM must equal OUT_DIM"
        )
        var blk = Self()
        for _ in range(Self.N):
            var net = Self.DynNet.make[target, INIT]()
            var opt = AdamW.make[target, M=Self.DynNet](net)
            # PETS/MBPO-reference dynamics weight decay (legacy default
            # `dyn_weight_decay=5e-5`). Without this the ensemble overfits:
            # train NLL collapses while holdout NLL diverges → optimistic
            # OOD synthetic data. Tunable post-make via `set_weight_decay`.
            opt.weight_decay = Scalar[DT](5e-5)
            blk.members.append(net^)
            blk.opts.append(opt^)
        for i in range(Self.NUM_ELITES):
            blk.elite_indices.append(i)
        blk.loss = GaussianNLLLoss[
            Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX
        ].make[target]()
        blk.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](blk)
        blk._init_bounds["cpu"]()
        return blk^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer,
    ](ctx: DeviceContext) raises -> Self:
        """GPU factory. Each member + Adam + the shared GaussianNLLLoss and
        member-indexed scratch live in device memory; the train/predict
        paths run member.forward/vjp + the NLL loss kernels on-device."""
        comptime assert target == "gpu", (
            "DynamicsEnsembleBlock.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        comptime assert Self.DynNet.IN_DIMS[0] == Self.IN_DIM, (
            "DynNet.IN_DIM must equal IN_DIM"
        )
        comptime assert Self.DynNet.OUT_DIM == Self.OUT_DIM, (
            "DynNet.OUT_DIM must equal OUT_DIM"
        )
        var blk = Self()
        for _ in range(Self.N):
            var net = Self.DynNet.make[target, INIT](ctx=ctx)
            var opt = AdamW.make[target, M=Self.DynNet](net, ctx=ctx)
            # PETS/MBPO-reference dynamics weight decay (legacy default
            # `dyn_weight_decay=5e-5`). Without this the ensemble overfits:
            # train NLL collapses while holdout NLL diverges → optimistic
            # OOD synthetic data. Tunable post-make via `set_weight_decay`.
            opt.weight_decay = Scalar[DT](5e-5)
            blk.members.append(net^)
            blk.opts.append(opt^)
        for i in range(Self.NUM_ELITES):
            blk.elite_indices.append(i)
        blk.loss = GaussianNLLLoss[
            Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX
        ].make[target](ctx=ctx)
        blk.ts = TargetStorage.make_gpu(ctx)
        init_scratch_auto[Self, target="gpu"](blk, ctx)
        blk._init_bounds["gpu"]()
        return blk^

    def _init_bounds[target: StaticString](mut self) raises:
        """Init the learnable-bound state: max_lv=+0.5, min_lv=−10 (PETS/MBPO
        `bnn.py` inits), Adam moments=0. Device buffers from `enqueue_create`
        are uninitialised, so we fill all of them explicitly on GPU; on CPU
        the lists are already zero so only the bounds need filling."""
        comptime if target == "cpu":
            var maxp = self._max_lv.cpu_ptr()
            var minp = self._min_lv.cpu_ptr()
            for i in range(Self.N * Self.PRED_DIM):
                maxp[i] = Scalar[DT](0.5)
                minp[i] = Scalar[DT](-10.0)
        else:
            self._max_lv.dev.value().enqueue_fill(Scalar[DT](0.5))
            self._min_lv.dev.value().enqueue_fill(Scalar[DT](-10.0))
            self._max_lv_m.dev.value().enqueue_fill(Scalar[DT](0.0))
            self._max_lv_v.dev.value().enqueue_fill(Scalar[DT](0.0))
            self._min_lv_m.dev.value().enqueue_fill(Scalar[DT](0.0))
            self._min_lv_v.dev.value().enqueue_fill(Scalar[DT](0.0))

    # ------------------------------------------------------------------
    # Public knobs.
    # ------------------------------------------------------------------

    def enable_learnable_bounds(mut self):
        """Opt into learnable per-member/per-dim logvar bounds (soft
        double-softplus clamp + 0.01-L2 Adam-updated bounds, init +0.5/−10).
        Off by default ⇒ the fixed `[LOGVAR_MIN, LOGVAR_MAX]` hard clamp."""
        self.learnable_bounds = True

    def set_lr(mut self, lr: Scalar[DT]):
        """Set every member's AdamW LR. Matches the deep_agents config
        convention (single `model_lr` applies to all ensemble members)."""
        self._bnd_lr = lr
        for i in range(Self.N):
            self.opts[i].lr = lr

    def set_weight_decay(mut self, wd: Scalar[DT]):
        """Set every member's AdamW decoupled weight decay. Defaults to the
        PETS/MBPO-reference `5e-5` at make-time; this overrides it. The
        dynamics ensemble REQUIRES decay to generalise — without it the
        members overfit (train NLL ↓, holdout NLL ↑) and synthetic data
        becomes optimistic OOD garbage. Mirrors legacy `dyn_weight_decay`."""
        for i in range(Self.N):
            self.opts[i].weight_decay = wd

    def set_max_grad_norm(mut self, threshold: Scalar[DT]):
        """No-op: the dynamics ensemble uses AdamW, which does not implement
        a grad-norm clip (decoupled weight decay is the regulariser here).
        Kept for API compatibility; never invoked on the MBPO path."""
        pass

    # ------------------------------------------------------------------
    # Predict — forward through one member, split + clamp logvar.
    # ------------------------------------------------------------------

    def predict_member[target: StaticString, POLICY: AMPPolicy = NoAMP](
        mut self,
        member_idx: Int,
        in_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut out_mu_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut out_lv_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """Forward `members[member_idx]` on `in_t` (BATCH × IN_DIM).
        Split the BATCH × OUT_DIM output into `out_mu_t` (BATCH × PRED_DIM,
        means) and `out_lv_t` (BATCH × PRED_DIM, clamped logvars).

        The clamped logvar is what callers want — Gaussian sampling and
        diagnostic logging both need σ² = exp(clamped_lv)."""
        assert_tag_for["DynamicsEnsembleBlock", target](self.ts.target_tag)

        comptime if target == "cpu":
            var pred_p = self._mb_pred.cpu_ptr()
            var pred_t = TileTensor(
                pred_p, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            self.members[member_idx].forward[target, Self.BATCH, POLICY](
                in_t, output=pred_t,
            )

            var mu_p = mptr(out_mu_t.ptr)
            var lv_p = mptr(out_lv_t.ptr)
            var lv_min = Scalar[DT](Self.LOGVAR_MIN)
            var lv_max = Scalar[DT](Self.LOGVAR_MAX)
            var bo = member_idx * Self.PRED_DIM
            var maxp = self._max_lv.cpu_ptr()
            var minp = self._min_lv.cpu_ptr()
            for b in range(Self.BATCH):
                var src = b * Self.OUT_DIM
                var dst = b * Self.PRED_DIM
                for j in range(Self.PRED_DIM):
                    mu_p[dst + j] = pred_p[src + j]
                    var raw = pred_p[src + Self.PRED_DIM + j]
                    if self.learnable_bounds:
                        lv_p[dst + j] = _soft_clamp_lv(
                            raw, maxp[bo + j], minp[bo + j]
                        )
                    else:
                        var v = raw
                        if v > lv_max:
                            v = lv_max
                        elif v < lv_min:
                            v = lv_min
                        lv_p[dst + j] = v
        else:
            var ctx = self.ts.ctx.value()
            var pred_p = self._mb_pred.dev_ptr()
            var pred_t = TileTensor(
                pred_p, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            self.members[member_idx].forward[target, Self.BATCH, POLICY](
                in_t, output=pred_t,
            )
            var pred_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](pred_p)
            var mu_p = mptr(out_mu_t.ptr)
            var lv_p = mptr(out_lv_t.ptr)
            var mu_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PRED_DIM), MutAnyOrigin,
            ](mu_p)
            var lv_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PRED_DIM), MutAnyOrigin,
            ](lv_p)
            comptime total = Self.BATCH * Self.PRED_DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            if self.learnable_bounds:
                var bo = member_idx * Self.PRED_DIM
                var max_lt = LayoutTensor[
                    DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
                ](self._max_lv.dev_ptr() + bo)
                var min_lt = LayoutTensor[
                    DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
                ](self._min_lv.dev_ptr() + bo)
                comptime soft_kernel = _soft_clamp_split_kernel[
                    Self.BATCH, Self.OUT_DIM, Self.PRED_DIM
                ]
                ctx.enqueue_function[soft_kernel](
                    pred_lt, max_lt, min_lt, mu_lt, lv_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )
            else:
                comptime split_kernel = _split_clamp_logvar_kernel[
                    Self.BATCH, Self.OUT_DIM, Self.PRED_DIM
                ]
                ctx.enqueue_function[split_kernel](
                    pred_lt, mu_lt, lv_lt,
                    Scalar[DT](Self.LOGVAR_MIN), Scalar[DT](Self.LOGVAR_MAX),
                    grid_dim=n_blocks, block_dim=TPB,
                )

    # ------------------------------------------------------------------
    # Learnable-bounds helpers (soft-clamp NLL grad + bound Adam step).
    # ------------------------------------------------------------------

    def _nll_grad_learnable[target: StaticString](
        mut self, member_idx: Int,
        pred_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mb_target_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises -> Scalar[DT]:
        """Soft-clamp Gaussian-NLL forward+grad for one member. Reads the
        already-computed `pred_t` (BATCH × OUT_DIM), writes the network
        output grad into `_mb_grad` and the per-(b,d) bound grads into
        `_bnd_gmax`/`_bnd_gmin`. Returns the scalar NLL (nn convention)."""
        var bo = member_idx * Self.PRED_DIM
        comptime if target == "cpu":
            var pp = mptr(pred_t.ptr)
            var tp = mptr(mb_target_t.ptr)
            var gp = self._mb_grad.cpu_ptr()
            var gmaxp = self._bnd_gmax.cpu_ptr()
            var gminp = self._bnd_gmin.cpu_ptr()
            var maxp = self._max_lv.cpu_ptr()
            var minp = self._min_lv.cpu_ptr()
            var inv_norm = Scalar[DT](2.0) / Scalar[DT](Self.BATCH * Self.PRED_DIM)
            var total = Scalar[DT](0.0)
            for b in range(Self.BATCH):
                var po = b * Self.OUT_DIM
                var to = b * Self.PRED_DIM
                for d in range(Self.PRED_DIM):
                    var mu = pp[po + d]
                    var raw = pp[po + Self.PRED_DIM + d]
                    var y = tp[to + d]
                    var max_d = maxp[bo + d]
                    var min_d = minp[bo + d]
                    var a1 = max_d - raw
                    var g1 = _sigmoid_dt(a1)
                    var lv_inter = max_d - _softplus_dt(a1)
                    var a2 = lv_inter - min_d
                    var g2 = _sigmoid_dt(a2)
                    var lv = min_d + _softplus_dt(a2)
                    var inv_var = fexp(-lv)
                    var diff = mu - y
                    var dsq = diff * diff
                    total += Scalar[DT](0.5) * dsq * inv_var + Scalar[DT](0.5) * lv
                    var grad_lv = (
                        Scalar[DT](0.5) * (Scalar[DT](1.0) - dsq * inv_var)
                        * inv_norm
                    )
                    gp[po + d] = diff * inv_var * inv_norm
                    gp[po + Self.PRED_DIM + d] = grad_lv * g1 * g2
                    gmaxp[to + d] = grad_lv * g2 * (Scalar[DT](1.0) - g1)
                    gminp[to + d] = grad_lv * (Scalar[DT](1.0) - g2)
            return total / Scalar[DT](Self.BATCH)
        else:
            var ctx = self.ts.ctx.value()
            var pred_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](mptr(pred_t.ptr))
            var tgt_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PRED_DIM), MutAnyOrigin,
            ](mptr(mb_target_t.ptr))
            var max_lt = LayoutTensor[
                DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
            ](self._max_lv.dev_ptr() + bo)
            var min_lt = LayoutTensor[
                DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
            ](self._min_lv.dev_ptr() + bo)
            var grad_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](self._mb_grad.dev_ptr())
            var gmax_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PRED_DIM), MutAnyOrigin,
            ](self._bnd_gmax.dev_ptr())
            var gmin_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PRED_DIM), MutAnyOrigin,
            ](self._bnd_gmin.dev_ptr())
            var ploss_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](self._bnd_ploss.dev_ptr())
            comptime n_rows = (Self.BATCH + TPB - 1) // TPB
            comptime grad_kernel = _dyn_learnable_nll_grad_kernel[
                Self.BATCH, Self.PRED_DIM
            ]
            ctx.enqueue_function[grad_kernel](
                pred_lt, tgt_lt, max_lt, min_lt,
                grad_lt, gmax_lt, gmin_lt, ploss_lt,
                grid_dim=n_rows, block_dim=TPB,
            )
            # D2H the per-row losses through the staging CPU mirror.
            ctx.enqueue_copy(
                self._bnd_ploss.cpu_ptr(), self._bnd_ploss.dev.value()
            )
            ctx.synchronize()
            var hp = self._bnd_ploss.cpu_ptr()
            var total = Scalar[DT](0.0)
            for b in range(Self.BATCH):
                total += hp[b]
            return total / Scalar[DT](Self.BATCH)

    def _bounds_step[target: StaticString](mut self, member_idx: Int) raises:
        """Adam-update member `member_idx`'s per-dim bounds from the grads in
        `_bnd_gmax`/`_bnd_gmin` (reduced over batch) + the 0.01 L2 penalty."""
        self._bnd_step += 1
        var beta1 = Scalar[DT](0.9)
        var beta2 = Scalar[DT](0.999)
        var eps = Scalar[DT](1e-8)
        var l2 = Scalar[DT](0.01)
        var sf = Scalar[DT](self._bnd_step)
        var bc1 = Scalar[DT](1.0) - fexp(sf * flog(beta1))
        var bc2 = Scalar[DT](1.0) - fexp(sf * flog(beta2))
        var bo = member_idx * Self.PRED_DIM
        comptime if target == "cpu":
            var maxp = self._max_lv.cpu_ptr()
            var minp = self._min_lv.cpu_ptr()
            var mmp = self._max_lv_m.cpu_ptr()
            var mvp = self._max_lv_v.cpu_ptr()
            var nmp = self._min_lv_m.cpu_ptr()
            var nvp = self._min_lv_v.cpu_ptr()
            var gmaxp = self._bnd_gmax.cpu_ptr()
            var gminp = self._bnd_gmin.cpu_ptr()
            for d in range(Self.PRED_DIM):
                var g_max = l2
                var g_min = -l2
                for b in range(Self.BATCH):
                    g_max += gmaxp[b * Self.PRED_DIM + d]
                    g_min += gminp[b * Self.PRED_DIM + d]
                var i = bo + d
                var m1 = beta1 * mmp[i] + (Scalar[DT](1.0) - beta1) * g_max
                var v1 = beta2 * mvp[i] + (Scalar[DT](1.0) - beta2) * g_max * g_max
                mmp[i] = m1
                mvp[i] = v1
                maxp[i] = maxp[i] - self._bnd_lr * (m1 / bc1) / (
                    fsqrt(v1 / bc2) + eps
                )
                var m2 = beta1 * nmp[i] + (Scalar[DT](1.0) - beta1) * g_min
                var v2 = beta2 * nvp[i] + (Scalar[DT](1.0) - beta2) * g_min * g_min
                nmp[i] = m2
                nvp[i] = v2
                minp[i] = minp[i] - self._bnd_lr * (m2 / bc1) / (
                    fsqrt(v2 / bc2) + eps
                )
        else:
            var ctx = self.ts.ctx.value()
            var max_lt = LayoutTensor[
                DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
            ](self._max_lv.dev_ptr() + bo)
            var min_lt = LayoutTensor[
                DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
            ](self._min_lv.dev_ptr() + bo)
            var mm_lt = LayoutTensor[
                DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
            ](self._max_lv_m.dev_ptr() + bo)
            var mv_lt = LayoutTensor[
                DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
            ](self._max_lv_v.dev_ptr() + bo)
            var nm_lt = LayoutTensor[
                DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
            ](self._min_lv_m.dev_ptr() + bo)
            var nv_lt = LayoutTensor[
                DT, Layout.row_major(Self.PRED_DIM), MutAnyOrigin,
            ](self._min_lv_v.dev_ptr() + bo)
            var gmax_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PRED_DIM), MutAnyOrigin,
            ](self._bnd_gmax.dev_ptr())
            var gmin_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PRED_DIM), MutAnyOrigin,
            ](self._bnd_gmin.dev_ptr())
            comptime n_dim_blocks = (Self.PRED_DIM + TPB - 1) // TPB
            comptime bnd_kernel = _dyn_bounds_adam_kernel[
                Self.PRED_DIM, Self.BATCH
            ]
            ctx.enqueue_function[bnd_kernel](
                max_lt, min_lt, mm_lt, mv_lt, nm_lt, nv_lt,
                gmax_lt, gmin_lt,
                l2, self._bnd_lr, beta1, beta2, eps, bc1, bc2,
                grid_dim=n_dim_blocks, block_dim=TPB,
            )

    # ------------------------------------------------------------------
    # Train member — one Gaussian-NLL gradient step.
    # ------------------------------------------------------------------

    def train_member_step[target: StaticString, POLICY: AMPPolicy = NoAMP](
        mut self,
        member_idx: Int,
        mb_in_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mb_target_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises -> Scalar[DT]:
        """One Gaussian-NLL gradient step on member `member_idx`.

        Caller owns `mb_in_t` (BATCH × IN_DIM) and `mb_target_t`
        (BATCH × PRED_DIM = 1 + obs_dim).  Returns the scalar NLL loss
        (averaged over BATCH)."""
        assert_tag_for["DynamicsEnsembleBlock", target](self.ts.target_tag)

        comptime if target == "cpu":
            var pred_p = self._mb_pred.cpu_ptr()
            var grad_p = self._mb_grad.cpu_ptr()
            var pred_t = TileTensor(
                pred_p, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            var grad_t = TileTensor(
                grad_p, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            self.opts[member_idx].zero_grad[target, M=Self.DynNet](
                self.members[member_idx]
            )
            self.members[member_idx].forward[target, Self.BATCH, POLICY](
                mb_in_t, output=pred_t,
            )
            var loss: Scalar[DT]
            if self.learnable_bounds:
                # Soft-clamp NLL with learnable bounds: fills `grad_t`
                # (= _mb_grad) with the network output grad + the per-(b,d)
                # bound grads into _bnd_gmax/_bnd_gmin.
                loss = self._nll_grad_learnable[target](
                    member_idx, pred_t, mb_target_t,
                )
            else:
                loss = self.loss.forward[target, Self.BATCH, POLICY](
                    pred_t, mb_target_t,
                )
                self.loss.vjp[target, Self.BATCH, POLICY](mb_target_t, grad_t)
            # Reuse pred buffer for grad-input scratch (member backward
            # writes into a slab the same size as IN_DIM; we don't need
            # to inspect those grad-inputs, just have a sink for them).
            var gi_p = self._mb_pred.cpu_ptr()  # reused as discard sink.
            var gi_t = TileTensor(
                gi_p, row_major[Self.BATCH, Self.IN_DIM]()
            )
            self.members[member_idx].vjp[target, Self.BATCH, POLICY](
                grad_t, gi_t,
            )
            self.opts[member_idx].step[target, M=Self.DynNet](
                self.members[member_idx]
            )
            if self.learnable_bounds:
                self._bounds_step[target](member_idx)
            return loss
        else:
            var pred_p = self._mb_pred.dev_ptr()
            var grad_p = self._mb_grad.dev_ptr()
            var pred_t = TileTensor(
                pred_p, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            var grad_t = TileTensor(
                grad_p, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            self.opts[member_idx].zero_grad[target, M=Self.DynNet](
                self.members[member_idx]
            )
            self.members[member_idx].forward[target, Self.BATCH, POLICY](
                mb_in_t, output=pred_t,
            )
            var loss: Scalar[DT]
            if self.learnable_bounds:
                # Soft-clamp NLL with learnable bounds: fills `grad_t`
                # (= _mb_grad) with the network output grad + the per-(b,d)
                # bound grads into _bnd_gmax/_bnd_gmin.
                loss = self._nll_grad_learnable[target](
                    member_idx, pred_t, mb_target_t,
                )
            else:
                loss = self.loss.forward[target, Self.BATCH, POLICY](
                    pred_t, mb_target_t,
                )
                self.loss.vjp[target, Self.BATCH, POLICY](mb_target_t, grad_t)
            # Reuse pred buffer as a discard sink for member grad-inputs
            # (OUT_DIM >= IN_DIM asserted in __init__).
            var gi_p = self._mb_pred.dev_ptr()
            var gi_t = TileTensor(
                gi_p, row_major[Self.BATCH, Self.IN_DIM]()
            )
            self.members[member_idx].vjp[target, Self.BATCH, POLICY](
                grad_t, gi_t,
            )
            self.opts[member_idx].step[target, M=Self.DynNet](
                self.members[member_idx]
            )
            if self.learnable_bounds:
                self._bounds_step[target](member_idx)
            return loss

    # ------------------------------------------------------------------
    # Eval member loss — for holdout-set scoring.
    # ------------------------------------------------------------------

    def eval_member_loss[
        target: StaticString, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        member_idx: Int,
        mb_in_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mb_target_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises -> Scalar[DT]:
        """Holdout-set forward only.  No gradient, no opt step.

        Returns the same NLL as `train_member_step` would compute but
        without mutating member weights — used to refresh
        `elite_indices` after a training pass."""
        assert_tag_for["DynamicsEnsembleBlock", target](self.ts.target_tag)

        comptime if target == "cpu":
            var pred_p = self._mb_pred.cpu_ptr()
            var pred_t = TileTensor(
                pred_p, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            self.members[member_idx].forward[target, Self.BATCH, POLICY](
                mb_in_t, output=pred_t,
            )
            if self.learnable_bounds:
                # Soft-clamp NLL with the member's learnable bounds (grads
                # land in throwaway scratch; no opt step here).
                return self._nll_grad_learnable[target](
                    member_idx, pred_t, mb_target_t,
                )
            return self.loss.forward[target, Self.BATCH, POLICY](
                pred_t, mb_target_t,
            )
        else:
            var pred_p = self._mb_pred.dev_ptr()
            var pred_t = TileTensor(
                pred_p, row_major[Self.BATCH, Self.OUT_DIM]()
            )
            self.members[member_idx].forward[target, Self.BATCH, POLICY](
                mb_in_t, output=pred_t,
            )
            if self.learnable_bounds:
                # Soft-clamp NLL with the member's learnable bounds (grads
                # land in throwaway scratch; no opt step here).
                return self._nll_grad_learnable[target](
                    member_idx, pred_t, mb_target_t,
                )
            return self.loss.forward[target, Self.BATCH, POLICY](
                pred_t, mb_target_t,
            )

    # ------------------------------------------------------------------
    # Elite ranking — refresh elite_indices from per-member holdout losses.
    # ------------------------------------------------------------------

    def update_elites(mut self, mut holdout_losses: List[Scalar[DT]]):
        """Sort members by ascending holdout loss; keep top-NUM_ELITES.

        Caller passes a fresh list of N losses (one per member).  Uses
        a selection sort for clarity over speed — N ≤ ~10 in practice,
        so O(N²) is fine."""
        # Build a parallel index list and partial-selection-sort it
        # against the holdout_losses values.
        var sorted_idx = List[Int]()
        for i in range(Self.N):
            sorted_idx.append(i)
        for i in range(Self.NUM_ELITES):
            var min_pos = i
            for j in range(i + 1, Self.N):
                if (
                    holdout_losses[sorted_idx[j]]
                    < holdout_losses[sorted_idx[min_pos]]
                ):
                    min_pos = j
            var tmp = sorted_idx[i]
            sorted_idx[i] = sorted_idx[min_pos]
            sorted_idx[min_pos] = tmp
        self.elite_indices.clear()
        for i in range(Self.NUM_ELITES):
            self.elite_indices.append(sorted_idx[i])
