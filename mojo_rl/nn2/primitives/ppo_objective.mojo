"""PPOObjective[ACT_] — binary Module for the PPO clipped-surrogate loss.

Phase I.2.b. Binary leaf consumed by `PPOActorLossCG` (the FullGraph
form of the PPO actor loss). Inputs:

  - actor_output  [BATCH, 2*ACT]   from GaussianHead: [mu | log_std].
  - aux           [BATCH, ACT+2]   packed [action | old_log_prob | advantage].

Output:

  - loss_per_b    [BATCH, 1]       per-sample (un-averaged) PPO loss.

The 1/BATCH mean factor lives in the seed gradient
(`seed_grad_inv_batch`), not inside this kernel — matches the
SACActorLossCG convention so the graph hosts the reduction.

Math (per sample b):
    new_log_prob = Σ_j  -0.5 * (LOG_2PI + 2·ls_j + ((a_j-mu_j)/std_j)²)
    diff         = clamp(new_log_prob - old_log_prob, ±20)
    ratio        = exp(diff)
    unclipped    = ratio * adv
    clipped      = clip(ratio, 1-ε, 1+ε) * adv
    entropy      = Σ_j  0.5 * (LOG_2PI + 1 + 2·ls_j)
    loss_per_b   = -min(unclipped, clipped) - entropy_coef * entropy

Backward (per sample b, with go = grad_loss_per_b[b]):
    is_clipped = clipped < unclipped
    If clipped (entropy still flows on log_std):
        grad_mu_j  = 0
        grad_ls_j  = -entropy_coef * go
    Else:
        d_lp_d_mu_j = z_j / std_j
        d_lp_d_ls_j = z_j² - 1
        grad_mu_j   = -adv * ratio * d_lp_d_mu_j * go            (clip ±10)
        grad_ls_j   = (-adv * ratio * d_lp_d_ls_j - entropy_coef) * go  (clip ±10)

Per-element grad clip ±10 matches the bespoke `ppo_actor_loss.mojo`
kernel exactly. grad_aux is written as zeros (action / old_log_prob /
advantage are non-differentiable inputs).

Forward caches the input pointers (no copy — the graph keeps the
buffers live across forward + vjp), `vjp` reads them. Mirrors
`MSELoss.cache_logits` — required because the Module trait `vjp`
signature receives only grad_output + grad_inputs, not the originals.

CPU only for I.2. GPU follow-up tracked under I.2.f / I.3.
"""

from std.math import exp
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0
comptime LOG_PROB_DIFF_MAX: Scalar[DT] = 20.0
comptime GRAD_CLIP: Scalar[DT] = 10.0
comptime EPS_STD: Scalar[DT] = 1e-6
comptime LOG_2PI: Scalar[DT] = 1.8378770664093453


struct PPOObjective[ACT_: Int](Module):
    """PPO clipped-surrogate + entropy bonus as a binary Module."""

    comptime ARITY: Int = 2
    comptime IN_DIM: Int = 2 * Self.ACT_      # actor_output: [mu | log_std]
    comptime IN1_DIM: Int = Self.ACT_ + 2     # aux: [action | old_log_prob | advantage]
    comptime OUT_DIM: Int = 1                 # per-sample loss

    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]

    # Input-pointer cache populated by forward, consumed by vjp.
    var _cache_ao_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _cache_ax_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var ts: TargetStorage

    def __init__(out self):
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._cache_ao_ptr = null_p
        self._cache_ax_ptr = null_p
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "PPOObjective.make[target='gpu', INIT] requires a DeviceContext"
        )
        var op = Self()
        op.ts = TargetStorage.make_cpu()
        return op^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "PPOObjective.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        # GPU path deferred — I.2 scope is CPU FullGraph. PPO GPU lands
        # alongside the GPU on-policy driver (I.2 follow-up or I.3).
        comptime assert False, (
            "PPOObjective GPU path not implemented yet (Phase I.2 CPU only)."
        )
        var op = Self()
        op.ts = TargetStorage.make_gpu(ctx)
        return op^

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "clip_eps":
            self.clip_eps = value
        elif ATTR == "entropy_coef":
            self.entropy_coef = value

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["PPOObjective", target](self.ts.target_tag)
        comptime ACT = Self.ACT_

        var ao = typed_view[BATCH, Self.IN_DIM](inputs[0])
        var ax = typed_view[BATCH, Self.IN1_DIM](inputs[1])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

        # Cache input pointers for vjp.
        self._cache_ao_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](ao.ptr)
        self._cache_ax_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](ax.ptr)

        comptime if target == "cpu":
            for b in range(BATCH):
                var new_log_prob: Scalar[DT] = 0.0
                var entropy: Scalar[DT] = 0.0
                for j in range(ACT):
                    var mu = ao[b, j]
                    var ls = ao[b, ACT + j]
                    if ls < LOG_STD_MIN:
                        ls = LOG_STD_MIN
                    elif ls > LOG_STD_MAX:
                        ls = LOG_STD_MAX
                    var std = exp(ls)
                    var a = ax[b, j]
                    var z = (a - mu) / (std + EPS_STD)
                    new_log_prob += Scalar[DT](-0.5) * (
                        LOG_2PI + Scalar[DT](2.0) * ls + z * z
                    )
                    entropy += Scalar[DT](0.5) * (
                        LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls
                    )
                var olp = ax[b, ACT]
                var adv = ax[b, ACT + 1]
                var diff = new_log_prob - olp
                if diff > LOG_PROB_DIFF_MAX:
                    diff = LOG_PROB_DIFF_MAX
                elif diff < -LOG_PROB_DIFF_MAX:
                    diff = -LOG_PROB_DIFF_MAX
                var ratio = exp(diff)
                var clipped_ratio = ratio
                if clipped_ratio < Scalar[DT](1.0) - self.clip_eps:
                    clipped_ratio = Scalar[DT](1.0) - self.clip_eps
                elif clipped_ratio > Scalar[DT](1.0) + self.clip_eps:
                    clipped_ratio = Scalar[DT](1.0) + self.clip_eps
                var unclipped_obj = ratio * adv
                var clipped_obj = clipped_ratio * adv
                var min_obj: Scalar[DT] = unclipped_obj
                if clipped_obj < unclipped_obj:
                    min_obj = clipped_obj
                out[b, 0] = -min_obj - self.entropy_coef * entropy
        else:
            comptime assert False, (
                "PPOObjective.forward GPU not implemented (Phase I.2 CPU only)."
            )

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["PPOObjective", target](self.ts.target_tag)
        comptime ACT = Self.ACT_

        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi0 = typed_view_mut[BATCH, Self.IN_DIM](grad_inputs[0])
        var gi1 = typed_view_mut[BATCH, Self.IN1_DIM](grad_inputs[1])

        # grad_aux is identically zero — action/old_log_prob/advantage are
        # not differentiable. The graph still scatter-adds this buffer into
        # the "aux" InputSlot's grad accumulator, so we write zeros.
        for b in range(BATCH):
            for d in range(Self.IN1_DIM):
                gi1[b, d] = Scalar[DT](0.0)

        comptime if target == "cpu":
            var ao_p = self._cache_ao_ptr
            var ax_p = self._cache_ax_ptr
            for b in range(BATCH):
                var new_log_prob: Scalar[DT] = 0.0
                for j in range(ACT):
                    var mu = ao_p[b * Self.IN_DIM + j]
                    var ls = ao_p[b * Self.IN_DIM + ACT + j]
                    if ls < LOG_STD_MIN:
                        ls = LOG_STD_MIN
                    elif ls > LOG_STD_MAX:
                        ls = LOG_STD_MAX
                    var std = exp(ls)
                    var a = ax_p[b * Self.IN1_DIM + j]
                    var z = (a - mu) / (std + EPS_STD)
                    new_log_prob += Scalar[DT](-0.5) * (
                        LOG_2PI + Scalar[DT](2.0) * ls + z * z
                    )
                var olp = ax_p[b * Self.IN1_DIM + ACT]
                var adv = ax_p[b * Self.IN1_DIM + ACT + 1]
                var diff = new_log_prob - olp
                if diff > LOG_PROB_DIFF_MAX:
                    diff = LOG_PROB_DIFF_MAX
                elif diff < -LOG_PROB_DIFF_MAX:
                    diff = -LOG_PROB_DIFF_MAX
                var ratio = exp(diff)
                var clipped_ratio = ratio
                if clipped_ratio < Scalar[DT](1.0) - self.clip_eps:
                    clipped_ratio = Scalar[DT](1.0) - self.clip_eps
                elif clipped_ratio > Scalar[DT](1.0) + self.clip_eps:
                    clipped_ratio = Scalar[DT](1.0) + self.clip_eps
                var unclipped_obj = ratio * adv
                var clipped_obj = clipped_ratio * adv
                var is_clipped = clipped_obj < unclipped_obj

                var go_b = go[b, 0]

                for j in range(ACT):
                    if is_clipped:
                        gi0[b, j] = Scalar[DT](0.0)
                        # Entropy still flows even when clipped — matches
                        # bespoke ppo_actor_loss.mojo line 150.
                        gi0[b, ACT + j] = (
                            -self.entropy_coef * Scalar[DT](1.0) * go_b
                        )
                    else:
                        var mu = ao_p[b * Self.IN_DIM + j]
                        var ls = ao_p[b * Self.IN_DIM + ACT + j]
                        if ls < LOG_STD_MIN:
                            ls = LOG_STD_MIN
                        elif ls > LOG_STD_MAX:
                            ls = LOG_STD_MAX
                        var std = exp(ls)
                        var a = ax_p[b * Self.IN1_DIM + j]
                        var z = (a - mu) / (std + EPS_STD)
                        var d_lp_d_mu = z / (std + EPS_STD)
                        var d_lp_d_ls = z * z - Scalar[DT](1.0)
                        var gmu = -adv * ratio * d_lp_d_mu * go_b
                        var gls = (
                            -adv * ratio * d_lp_d_ls
                            - self.entropy_coef
                        ) * go_b
                        if gmu > GRAD_CLIP:
                            gmu = GRAD_CLIP
                        elif gmu < -GRAD_CLIP:
                            gmu = -GRAD_CLIP
                        if gls > GRAD_CLIP:
                            gls = GRAD_CLIP
                        elif gls < -GRAD_CLIP:
                            gls = -GRAD_CLIP
                        gi0[b, j] = gmu
                        gi0[b, ACT + j] = gls
        else:
            comptime assert False, (
                "PPOObjective.vjp GPU not implemented (Phase I.2 CPU only)."
            )
