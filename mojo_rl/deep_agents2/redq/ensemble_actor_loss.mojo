"""EnsembleActorLoss — REDQ SAC-style actor loss over N online critics.

Phase R.2 (CPU). Same shape as SAC's actor loss but uses the MEAN
of all N online critics instead of `min(Q1, Q2)`:

    loss_per_b[b] = α · log_prob[b] − combined_Q[b]
    combined_Q[b] = (1/N) · Σᵢ Qᵢ(s[b], rsample(π(s[b])))
    loss          = mean_b(loss_per_b)

Backward derivation:
    ∂loss/∂loss_per_b[b] = 1/B
    ∂loss/∂log_prob[b]   = α / B          (α treated as constant in actor opt)
    ∂loss/∂combined_Q[b] = −1 / B
    ∂loss/∂Qᵢ[b]         = (1/N) · ∂loss/∂combined_Q[b] = −1/(N·B)

Critic param-grad gating — `mode="input_only"` on every critic.vjp.
Per `Module.vjp` docs (`mojo_rl/nn2/core/module.mojo`), this skips the
param-grad accumulation step entirely. We never call `opt.step` on
any critic in this block, so even if `mode` were ignored the critic
params would stay unchanged; the explicit `input_only` is the
semantic stop-grad (matches SAC's `ExternalNode[..., MODE="input_only"]`
pattern in `sac/actor_loss.mojo`).

Returns `EnsembleActorLossResult { loss, log_prob_mean }`. The
trainer reads `log_prob_mean` for `AlphaUpdateStep` (the entropy
temperature gradient is `log_prob_mean + target_entropy` — identical
to SAC, hence no `AlphaUpdateStep` changes needed in R.3).

R.2 is CPU-only. GPU comes alongside the full GPU REDQ trainer; the
forward + backward layout here is shaped so that each kernel call
swaps to its `_gpu` variant without restructuring the loop.
"""

from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.optimizer.adam import Adam

from std.gpu.host import DeviceContext

from ..primitives.rsample import RSample
from .ensemble import CriticEnsemble


@fieldwise_init
struct EnsembleActorLossResult(Movable & ImplicitlyDestructible):
    """Forward/backward result: scalar loss + log_prob_mean (the
    Σ_b log π(a|s) / B used by the AlphaUpdateStep)."""

    var loss: Scalar[DT]
    var log_prob_mean: Scalar[DT]


struct EnsembleActorLoss[
    ACTOR: Module,
    CRITIC: Module,
    N_: Int,
    BATCH_: Int,
    OBS_: Int,
    ACT_: Int,
](Movable & ImplicitlyDestructible):
    comptime N = Self.N_
    comptime BATCH = Self.BATCH_
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime SA_DIM = Self.OBS + Self.ACT
    comptime ALP_DIM = Self.ACT + 1

    var rsample: RSample[Self.ACT]

    # Forward scratches.
    var _mb_ao: Scratch["eal_mb_ao", Self.BATCH * (2 * Self.ACT)]
    var _mb_alp: Scratch["eal_mb_alp", Self.BATCH * (Self.ACT + 1)]
    var _mb_sa: Scratch["eal_mb_sa", Self.BATCH * Self.SA_DIM]
    var _mb_q_i: Scratch["eal_mb_q_i", Self.BATCH]
    var _mb_q_sum: Scratch["eal_mb_q_sum", Self.BATCH]

    # Backward scratches.
    var _mb_grad_q_i: Scratch["eal_mb_grad_q_i", Self.BATCH]
    var _mb_grad_sa_i: Scratch["eal_mb_grad_sa_i", Self.BATCH * Self.SA_DIM]
    var _mb_grad_sa_sum: Scratch[
        "eal_mb_grad_sa_sum", Self.BATCH * Self.SA_DIM,
    ]
    var _mb_grad_alp: Scratch[
        "eal_mb_grad_alp", Self.BATCH * (Self.ACT + 1),
    ]
    var _mb_grad_ao: Scratch[
        "eal_mb_grad_ao", Self.BATCH * (2 * Self.ACT),
    ]
    var _mb_grad_obs: Scratch["eal_mb_grad_obs", Self.BATCH * Self.OBS]

    var ts: TargetStorage

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Scratch["eal_mb_ao", Self.BATCH * (2 * Self.ACT)]()
        self._mb_alp = Scratch["eal_mb_alp", Self.BATCH * (Self.ACT + 1)]()
        self._mb_sa = Scratch["eal_mb_sa", Self.BATCH * Self.SA_DIM]()
        self._mb_q_i = Scratch["eal_mb_q_i", Self.BATCH]()
        self._mb_q_sum = Scratch["eal_mb_q_sum", Self.BATCH]()
        self._mb_grad_q_i = Scratch["eal_mb_grad_q_i", Self.BATCH]()
        self._mb_grad_sa_i = Scratch[
            "eal_mb_grad_sa_i", Self.BATCH * Self.SA_DIM,
        ]()
        self._mb_grad_sa_sum = Scratch[
            "eal_mb_grad_sa_sum", Self.BATCH * Self.SA_DIM,
        ]()
        self._mb_grad_alp = Scratch[
            "eal_mb_grad_alp", Self.BATCH * (Self.ACT + 1),
        ]()
        self._mb_grad_ao = Scratch[
            "eal_mb_grad_ao", Self.BATCH * (2 * Self.ACT),
        ]()
        self._mb_grad_obs = Scratch[
            "eal_mb_grad_obs", Self.BATCH * Self.OBS,
        ]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu", (
            "EnsembleActorLoss: R.2 supports CPU only — GPU follow-up."
        )
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.OBS, (
            "EnsembleActorLoss: ACTOR.IN_DIM must equal OBS"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "EnsembleActorLoss: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.SA_DIM, (
            "EnsembleActorLoss: CRITIC.IN_DIM must equal OBS+ACT"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "EnsembleActorLoss: CRITIC.OUT_DIM must equal 1"
        )
        var b = Self()
        b.rsample = RSample[Self.ACT].make[target, Zero](ctx=ctx)
        b.rsample.action_scale = action_scale
        b.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](b, ctx)
        return b^

    def forward_backward[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> EnsembleActorLossResult:
        """One actor gradient step. Reads `mb_s_ptr` (BATCH × OBS),
        consumes `alpha`, writes through `actor` + `actor_opt`.
        Returns (loss, log_prob_mean).

        The N online critics get forward + vjp[input_only] — their
        param grads are NOT touched. Caller MUST NOT call
        `ensemble.opts[i].step` between this method and the next
        ensemble-critic update.
        """
        assert_tag_for["EnsembleActorLoss", target](self.ts.target_tag)
        comptime assert target == "cpu", (
            "EnsembleActorLoss.forward_backward: R.2 CPU only"
        )

        # ── Step 0 — zero actor grad slab.
        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        # ── Step 1 — actor.forward(s) → _mb_ao [B, 2·ACT].
        var ao_p = self._mb_ao.target_ptr[target]()
        var s_t = TileTensor(mb_s_ptr, row_major[Self.BATCH, Self.OBS]())
        var ao_t = TileTensor(
            ao_p, row_major[Self.BATCH, 2 * Self.ACT](),
        )
        actor.forward[target, Self.BATCH, POLICY](s_t, output=ao_t)

        # ── Step 2 — rsample.forward(ao) → _mb_alp [B, ACT+1].
        var alp_p = self._mb_alp.target_ptr[target]()
        var alp_t = TileTensor(
            alp_p, row_major[Self.BATCH, Self.ALP_DIM](),
        )
        self.rsample.forward[target, Self.BATCH, POLICY](
            ao_t, output=alp_t,
        )

        # ── Step 3 — sa = concat(s, action). action lives at alp[:, :ACT].
        var sa_p = self._mb_sa.target_ptr[target]()
        for b in range(Self.BATCH):
            for d in range(Self.OBS):
                sa_p[b * Self.SA_DIM + d] = mb_s_ptr[b * Self.OBS + d]
            for j in range(Self.ACT):
                sa_p[b * Self.SA_DIM + Self.OBS + j] = alp_p[
                    b * Self.ALP_DIM + j
                ]
        var sa_t = TileTensor(
            sa_p, row_major[Self.BATCH, Self.SA_DIM](),
        )

        # ── Step 4 — loop N online critic forwards; accumulate Σᵢ Qᵢ(s,a).
        var q_sum_p = self._mb_q_sum.target_ptr[target]()
        for b in range(Self.BATCH):
            q_sum_p[b] = Scalar[DT](0.0)
        var q_i_p = self._mb_q_i.target_ptr[target]()
        for i in range(Self.N):
            var q_i_t = TileTensor(q_i_p, row_major[Self.BATCH, 1]())
            ensemble.pairs[i].online.forward[
                target, Self.BATCH, POLICY,
            ](sa_t, output=q_i_t)
            for b in range(Self.BATCH):
                q_sum_p[b] += q_i_p[b]

        # ── Step 5 — host-side scalar reduction: loss + log_prob_mean.
        var inv_n: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.N)
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        var loss: Scalar[DT] = Scalar[DT](0.0)
        var lp_sum: Scalar[DT] = Scalar[DT](0.0)
        for b in range(Self.BATCH):
            var combined = q_sum_p[b] * inv_n
            var lp = alp_p[b * Self.ALP_DIM + Self.ACT]
            loss += alpha * lp - combined
            lp_sum += lp
        loss *= inv_b
        var log_prob_mean = lp_sum * inv_b

        # ── Step 6 — backward seed: grad_qᵢ[b] = −1/(N·B) for every (i, b).
        var grad_q_i_p = self._mb_grad_q_i.target_ptr[target]()
        var grad_q_val: Scalar[DT] = -inv_n * inv_b
        for b in range(Self.BATCH):
            grad_q_i_p[b] = grad_q_val
        var grad_q_i_t = TileTensor(
            grad_q_i_p, row_major[Self.BATCH, 1](),
        )

        # ── Step 7 — for each critic: vjp[input_only] → accumulate grad_sa.
        var grad_sa_sum_p = self._mb_grad_sa_sum.target_ptr[target]()
        for k in range(Self.BATCH * Self.SA_DIM):
            grad_sa_sum_p[k] = Scalar[DT](0.0)
        var grad_sa_i_p = self._mb_grad_sa_i.target_ptr[target]()
        for i in range(Self.N):
            var grad_sa_i_t = TileTensor(
                grad_sa_i_p, row_major[Self.BATCH, Self.SA_DIM](),
            )
            # Re-run critic.forward with the exact same sa — critic caches the
            # forward state for the immediately-following vjp call. We did
            # forward(sa) earlier per critic in step 4 but the cache may have
            # been clobbered by later critics' forwards (each critic owns its
            # OWN cache, so actually it survives — but we'll re-forward to be
            # robust to any future caching changes).
            var q_i_t = TileTensor(q_i_p, row_major[Self.BATCH, 1]())
            ensemble.pairs[i].online.forward[
                target, Self.BATCH, POLICY,
            ](sa_t, output=q_i_t)
            ensemble.pairs[i].online.vjp[
                target, Self.BATCH, POLICY, mode="input_only",
            ](grad_q_i_t, grad_sa_i_t)
            for k in range(Self.BATCH * Self.SA_DIM):
                grad_sa_sum_p[k] += grad_sa_i_p[k]

        # ── Step 8 — assemble grad_alp [B, ACT+1]:
        # grad_action[b, j] = grad_sa_sum[b, OBS + j]
        # grad_log_prob[b]  = α / B
        var grad_alp_p = self._mb_grad_alp.target_ptr[target]()
        var grad_lp_val: Scalar[DT] = alpha * inv_b
        for b in range(Self.BATCH):
            for j in range(Self.ACT):
                grad_alp_p[b * Self.ALP_DIM + j] = grad_sa_sum_p[
                    b * Self.SA_DIM + Self.OBS + j
                ]
            grad_alp_p[b * Self.ALP_DIM + Self.ACT] = grad_lp_val
        var grad_alp_t = TileTensor(
            grad_alp_p, row_major[Self.BATCH, Self.ALP_DIM](),
        )

        # ── Step 9 — rsample.vjp(grad_alp) → grad_ao [B, 2·ACT].
        var grad_ao_p = self._mb_grad_ao.target_ptr[target]()
        var grad_ao_t = TileTensor(
            grad_ao_p, row_major[Self.BATCH, 2 * Self.ACT](),
        )
        self.rsample.vjp[target, Self.BATCH, POLICY](
            grad_alp_t, grad_ao_t,
        )

        # ── Step 10 — actor.vjp(grad_ao) → grad_obs (discarded);
        # accumulates actor param grads.
        var grad_obs_p = self._mb_grad_obs.target_ptr[target]()
        var grad_obs_t = TileTensor(
            grad_obs_p, row_major[Self.BATCH, Self.OBS](),
        )
        actor.vjp[target, Self.BATCH, POLICY, mode="all"](
            grad_ao_t, grad_obs_t,
        )

        # ── Step 11 — actor_opt.step(actor).
        actor_opt.step[target, M=Self.ACTOR](actor)

        return EnsembleActorLossResult(
            loss=loss, log_prob_mean=log_prob_mean,
        )
