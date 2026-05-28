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

Scope (I.1.a/b):
  - CPU only.  GPU `make`/`step` paths raise.  Phase G's audit puts GPU
    on the rollout-after-CPU-validates track.
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

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.loss.gaussian_nll_loss import GaussianNLLLoss
from mojo_rl.nn2.optimizer.adam import Adam


struct DynamicsEnsembleBlock[
    DynNet: Module,
    N: Int,
    NUM_ELITES: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    BATCH: Int,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
](Movable & ImplicitlyDestructible):
    """N-member probabilistic dynamics ensemble.

    `DynNet.OUT_DIM` MUST equal `OUT_DIM == 2 * PRED_DIM` where
    PRED_DIM = 1 + obs_dim (reward + Δobs)."""

    comptime PRED_DIM: Int = Self.OUT_DIM // 2

    var members: List[Self.DynNet]
    var opts: List[Adam]
    var loss: GaussianNLLLoss[Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX]
    var elite_indices: List[Int]

    var _mb_pred: Scratch["mb_pred", Self.BATCH * Self.OUT_DIM]
    var _mb_grad: Scratch["mb_grad", Self.BATCH * Self.OUT_DIM]

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
        self.opts = List[Adam]()
        self.loss = GaussianNLLLoss[
            Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX
        ]()
        self.elite_indices = List[Int]()
        self._mb_pred = Scratch["mb_pred", Self.BATCH * Self.OUT_DIM]()
        self._mb_grad = Scratch["mb_grad", Self.BATCH * Self.OUT_DIM]()
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
            var opt = Adam.make[target, M=Self.DynNet](net)
            blk.members.append(net^)
            blk.opts.append(opt^)
        for i in range(Self.NUM_ELITES):
            blk.elite_indices.append(i)
        blk.loss = GaussianNLLLoss[
            Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX
        ].make[target]()
        blk.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](blk)
        return blk^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer,
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "DynamicsEnsembleBlock.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        # GPU path deferred — see module docstring.
        raise Error(
            "DynamicsEnsembleBlock GPU not yet implemented (I.1 is CPU-first)"
        )

    # ------------------------------------------------------------------
    # Public knobs.
    # ------------------------------------------------------------------

    def set_lr(mut self, lr: Scalar[DT]):
        """Set every member's Adam LR. Matches the deep_agents config
        convention (single `model_lr` applies to all ensemble members)."""
        for i in range(Self.N):
            self.opts[i].lr = lr

    def set_max_grad_norm(mut self, threshold: Scalar[DT]):
        """Apply a global grad-norm clip to every member's Adam.
        `0.0` disables.  Mirrors `Adam.max_grad_norm`."""
        for i in range(Self.N):
            self.opts[i].max_grad_norm = threshold

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

            var mu_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out_mu_t.ptr)
            var lv_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out_lv_t.ptr)
            var lv_min = Scalar[DT](Self.LOGVAR_MIN)
            var lv_max = Scalar[DT](Self.LOGVAR_MAX)
            for b in range(Self.BATCH):
                var src = b * Self.OUT_DIM
                var dst = b * Self.PRED_DIM
                for j in range(Self.PRED_DIM):
                    mu_p[dst + j] = pred_p[src + j]
                    var raw = pred_p[src + Self.PRED_DIM + j]
                    var v = raw
                    if v > lv_max:
                        v = lv_max
                    elif v < lv_min:
                        v = lv_min
                    lv_p[dst + j] = v
        else:
            raise Error(
                "DynamicsEnsembleBlock.predict_member['gpu'] not implemented"
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
            var loss = self.loss.forward[target, Self.BATCH, POLICY](
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
            return loss
        else:
            raise Error(
                "DynamicsEnsembleBlock.train_member_step['gpu'] not implemented"
            )

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
            return self.loss.forward[target, Self.BATCH, POLICY](
                pred_t, mb_target_t,
            )
        else:
            raise Error(
                "DynamicsEnsembleBlock.eval_member_loss['gpu'] not implemented"
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
