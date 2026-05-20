"""PPOActorLoss[ACT] — CleanRL-style clipped-surrogate loss + entropy bonus.

Does NOT conform to the `Loss` trait — PPO needs four input tensors
(actor_output, action, old_log_prob, advantage) not two, so it would
distort the trait. Instead this is a bespoke composite that the PPO
example wires up directly.

Forward inputs (BATCH-major):
    actor_output    [BATCH, 2*ACT]   from GaussianHead: [mu | log_std]
    action          [BATCH, ACT]     unbounded actions taken during rollout
    old_log_prob    [BATCH]          log p_θ_old(action | s) recorded at rollout
    advantage       [BATCH]          GAE advantage, already normalized

Output: scalar L = mean over batch of
    -min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)   [clipped surrogate]
    - entropy_coef * H(π(·|s))                       [entropy bonus]

  where ratio = exp(clamp(new_log_prob - old_log_prob, ±20))

Backward emits grad_actor_output[BATCH, 2*ACT] = ∂L/∂[mu | log_std].
Returns 0 for the clipped samples (PyTorch detach semantics on the clip
branch). Matches v1 `ppo_continuous_actor_grad_kernel` exactly:
    LOG_STD ∈ [-5, 2], log_prob_diff ∈ [-20, 20], grad clip ±10.

Phase 6.4. CPU + GPU.
"""

from std.math import exp, log
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import (
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0
comptime LOG_PROB_DIFF_MAX: Scalar[DT] = 20.0
comptime GRAD_CLIP: Scalar[DT] = 10.0
comptime EPS_STD: Scalar[DT] = 1e-6
comptime LOG_2PI: Scalar[DT] = 1.8378770664093453


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────


def _ppo_actor_forward_kernel[BATCH: Int, ACT: Int](
    actor_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    action: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
    old_log_prob: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    advantage: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    partial_loss: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var new_log_prob: Scalar[DT] = 0.0
        var entropy: Scalar[DT] = 0.0
        for j in range(ACT):
            var mu = rebind[Scalar[DT]](actor_output[b, j])
            var ls = rebind[Scalar[DT]](actor_output[b, ACT + j])
            if ls < LOG_STD_MIN:
                ls = LOG_STD_MIN
            elif ls > LOG_STD_MAX:
                ls = LOG_STD_MAX
            var std = exp(ls)
            var a = rebind[Scalar[DT]](action[b, j])
            var z = (a - mu) / (std + EPS_STD)
            new_log_prob += Scalar[DT](-0.5) * (
                LOG_2PI + Scalar[DT](2.0) * ls + z * z
            )
            entropy += Scalar[DT](0.5) * (
                LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls
            )
        var diff = new_log_prob - rebind[Scalar[DT]](old_log_prob[b])
        if diff > LOG_PROB_DIFF_MAX:
            diff = LOG_PROB_DIFF_MAX
        elif diff < -LOG_PROB_DIFF_MAX:
            diff = -LOG_PROB_DIFF_MAX
        var ratio = exp(diff)
        var clipped_ratio = ratio
        if clipped_ratio < Scalar[DT](1.0) - clip_eps:
            clipped_ratio = Scalar[DT](1.0) - clip_eps
        elif clipped_ratio > Scalar[DT](1.0) + clip_eps:
            clipped_ratio = Scalar[DT](1.0) + clip_eps
        var adv = rebind[Scalar[DT]](advantage[b])
        var unclipped_obj = ratio * adv
        var clipped_obj = clipped_ratio * adv
        # min(s1, s2)
        var min_obj: Scalar[DT] = unclipped_obj
        if clipped_obj < unclipped_obj:
            min_obj = clipped_obj
        partial_loss[b] = -min_obj - entropy_coef * entropy


def _ppo_actor_backward_kernel[BATCH: Int, ACT: Int](
    actor_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    action: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
    old_log_prob: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    advantage: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    grad_actor_output: LayoutTensor[
        DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin
    ],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        # Recompute new_log_prob, ratio, is_clipped.
        var new_log_prob: Scalar[DT] = 0.0
        for j in range(ACT):
            var mu = rebind[Scalar[DT]](actor_output[b, j])
            var ls = rebind[Scalar[DT]](actor_output[b, ACT + j])
            if ls < LOG_STD_MIN:
                ls = LOG_STD_MIN
            elif ls > LOG_STD_MAX:
                ls = LOG_STD_MAX
            var std = exp(ls)
            var a = rebind[Scalar[DT]](action[b, j])
            var z = (a - mu) / (std + EPS_STD)
            new_log_prob += Scalar[DT](-0.5) * (
                LOG_2PI + Scalar[DT](2.0) * ls + z * z
            )
        var diff = new_log_prob - rebind[Scalar[DT]](old_log_prob[b])
        if diff > LOG_PROB_DIFF_MAX:
            diff = LOG_PROB_DIFF_MAX
        elif diff < -LOG_PROB_DIFF_MAX:
            diff = -LOG_PROB_DIFF_MAX
        var ratio = exp(diff)
        var clipped_ratio = ratio
        if clipped_ratio < Scalar[DT](1.0) - clip_eps:
            clipped_ratio = Scalar[DT](1.0) - clip_eps
        elif clipped_ratio > Scalar[DT](1.0) + clip_eps:
            clipped_ratio = Scalar[DT](1.0) + clip_eps
        var adv = rebind[Scalar[DT]](advantage[b])
        var unclipped_obj = ratio * adv
        var clipped_obj = clipped_ratio * adv
        var is_clipped = clipped_obj < unclipped_obj

        var inv_batch: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
        for j in range(ACT):
            if is_clipped:
                grad_actor_output[b, j] = 0.0
                # Entropy still flows even when clipped.
                grad_actor_output[b, ACT + j] = (
                    -entropy_coef * Scalar[DT](1.0) * inv_batch
                )
            else:
                var mu = rebind[Scalar[DT]](actor_output[b, j])
                var ls = rebind[Scalar[DT]](actor_output[b, ACT + j])
                if ls < LOG_STD_MIN:
                    ls = LOG_STD_MIN
                elif ls > LOG_STD_MAX:
                    ls = LOG_STD_MAX
                var std = exp(ls)
                var a = rebind[Scalar[DT]](action[b, j])
                var z = (a - mu) / (std + EPS_STD)
                # d_log_prob / d_mu = z / std
                # d_log_prob / d_log_std = z^2 - 1
                var d_lp_d_mu = z / (std + EPS_STD)
                var d_lp_d_ls = z * z - Scalar[DT](1.0)
                # Sign: L = -ratio * adv, so dL/d_lp = -ratio * adv (chain through ratio).
                # d_lp/d_mu chains: dL/d_mu = -ratio * adv * d_lp/d_mu
                # Entropy term: dE/d_ls = 1; L includes -entropy_coef * E, so
                # dL/d_ls from entropy = -entropy_coef.
                var gmu = -adv * ratio * d_lp_d_mu * inv_batch
                var gls = (
                    -adv * ratio * d_lp_d_ls
                    - entropy_coef * Scalar[DT](1.0)
                ) * inv_batch
                # Per-element grad clip ±10.
                if gmu > GRAD_CLIP:
                    gmu = GRAD_CLIP
                elif gmu < -GRAD_CLIP:
                    gmu = -GRAD_CLIP
                if gls > GRAD_CLIP:
                    gls = GRAD_CLIP
                elif gls < -GRAD_CLIP:
                    gls = -GRAD_CLIP
                grad_actor_output[b, j] = gmu
                grad_actor_output[b, ACT + j] = gls


# ──────────────────────────────────────────────────────────────────────────
# PPOActorLoss struct.
# ──────────────────────────────────────────────────────────────────────────


struct PPOActorLoss[ACT: Int](Defaultable, Movable, ImplicitlyDestructible):
    """PPO actor loss (clipped surrogate + entropy bonus), state-indep log_std."""

    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]

    # GPU scratch.
    var partial_loss_dev: Optional[DeviceBuffer[DT]]
    var partial_loss_host: Optional[HostBuffer[DT]]
    var partial_loss_n: Int
    var ctx: Optional[DeviceContext]

    var _target_tag: Int8

    def __init__(out self):
        self.clip_eps = 0.2
        self.entropy_coef = 0.0
        self.partial_loss_dev = None
        self.partial_loss_host = None
        self.partial_loss_n = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT

    @staticmethod
    def make[target: StaticString](
        clip_eps: Scalar[DT] = 0.2,
        entropy_coef: Scalar[DT] = 0.0,
    ) raises -> Self:
        comptime assert target == "cpu", (
            "PPOActorLoss.make[target='gpu'] requires a DeviceContext"
        )
        var lo = Self()
        lo.clip_eps = clip_eps
        lo.entropy_coef = entropy_coef
        lo._target_tag = TARGET_CPU
        return lo^

    @staticmethod
    def make[target: StaticString](
        ctx: DeviceContext,
        clip_eps: Scalar[DT] = 0.2,
        entropy_coef: Scalar[DT] = 0.0,
    ) raises -> Self:
        comptime assert target == "gpu", (
            "PPOActorLoss.make[target='cpu'](ctx, ...) — drop ctx for CPU"
        )
        var lo = Self()
        lo.clip_eps = clip_eps
        lo.entropy_coef = entropy_coef
        lo.partial_loss_dev = ctx.enqueue_create_buffer[DT](1)
        lo.partial_loss_host = ctx.enqueue_create_host_buffer[DT](1)
        lo.ctx = ctx
        lo._target_tag = TARGET_GPU
        return lo^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "PPOActorLoss: method called with [target='" + String(target)
                + "'] but loss was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_partial_gpu(mut self, batch: Int) raises:
        if self.partial_loss_n < batch:
            var c = self.ctx.value()
            self.partial_loss_dev = c.enqueue_create_buffer[DT](batch)
            self.partial_loss_host = c.enqueue_create_host_buffer[DT](batch)
            self.partial_loss_n = batch

    def forward[
        target: StaticString,
        BATCH: Int,
    ](
        mut self,
        actor_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        action: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        old_log_prob: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        advantage: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises -> Scalar[DT]:
        comptime assert actor_output.flat_rank == 2, "actor_output rank-2"
        comptime assert action.flat_rank == 2, "action rank-2"
        comptime assert old_log_prob.flat_rank == 1, "old_log_prob rank-1"
        comptime assert advantage.flat_rank == 1, "advantage rank-1"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var total: Scalar[DT] = 0.0
            for b in range(BATCH):
                var new_log_prob: Scalar[DT] = 0.0
                var entropy: Scalar[DT] = 0.0
                for j in range(Self.ACT):
                    var mu = actor_output[b, j]
                    var ls = actor_output[b, Self.ACT + j]
                    if ls < LOG_STD_MIN:
                        ls = LOG_STD_MIN
                    elif ls > LOG_STD_MAX:
                        ls = LOG_STD_MAX
                    var std = exp(ls)
                    var a = action[b, j]
                    var z = (a - mu) / (std + EPS_STD)
                    new_log_prob += Scalar[DT](-0.5) * (
                        LOG_2PI + Scalar[DT](2.0) * ls + z * z
                    )
                    entropy += Scalar[DT](0.5) * (
                        LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls
                    )
                var diff = new_log_prob - old_log_prob[b]
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
                var adv = advantage[b]
                var unclipped_obj = ratio * adv
                var clipped_obj = clipped_ratio * adv
                var min_obj: Scalar[DT] = unclipped_obj
                if clipped_obj < unclipped_obj:
                    min_obj = clipped_obj
                total += -min_obj - self.entropy_coef * entropy
            return total / Scalar[DT](BATCH)
        else:
            self._ensure_partial_gpu(BATCH)
            var ctx = self.ctx.value()
            comptime ao_layout = Layout.row_major(BATCH, 2 * Self.ACT)
            comptime ac_layout = Layout.row_major(BATCH, Self.ACT)
            comptime row_layout = Layout.row_major(BATCH)
            var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](actor_output.ptr)
            var ac_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](action.ptr)
            var ol_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](old_log_prob.ptr)
            var ad_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](advantage.ptr)
            var ao_lt = LayoutTensor[DT, ao_layout, MutAnyOrigin](ao_p)
            var ac_lt = LayoutTensor[DT, ac_layout, MutAnyOrigin](ac_p)
            var ol_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](ol_p)
            var ad_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](ad_p)
            var pl_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](
                self.partial_loss_dev.value()
            )
            comptime TPB = 64
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _ppo_actor_forward_kernel[BATCH, Self.ACT]
            ctx.enqueue_function[kernel](
                ao_lt, ac_lt, ol_lt, ad_lt, pl_lt,
                self.clip_eps, self.entropy_coef,
                grid_dim=n_blocks, block_dim=TPB,
            )
            ctx.enqueue_copy(self.partial_loss_host.value(), self.partial_loss_dev.value())
            ctx.synchronize()
            var total: Scalar[DT] = 0.0
            var hp = self.partial_loss_host.value().unsafe_ptr()
            for b in range(BATCH):
                total += hp[b]
            return total / Scalar[DT](BATCH)

    def backward[
        target: StaticString,
        BATCH: Int,
    ](
        mut self,
        actor_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        action: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        old_log_prob: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        advantage: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_actor_output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert actor_output.flat_rank == 2, "actor_output rank-2"
        comptime assert action.flat_rank == 2, "action rank-2"
        comptime assert old_log_prob.flat_rank == 1, "old_log_prob rank-1"
        comptime assert advantage.flat_rank == 1, "advantage rank-1"
        comptime assert grad_actor_output.flat_rank == 2, "grad_actor_output rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var inv_batch: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
            for b in range(BATCH):
                # Recompute new_log_prob, ratio, is_clipped.
                var new_log_prob: Scalar[DT] = 0.0
                for j in range(Self.ACT):
                    var mu = actor_output[b, j]
                    var ls = actor_output[b, Self.ACT + j]
                    if ls < LOG_STD_MIN:
                        ls = LOG_STD_MIN
                    elif ls > LOG_STD_MAX:
                        ls = LOG_STD_MAX
                    var std = exp(ls)
                    var a = action[b, j]
                    var z = (a - mu) / (std + EPS_STD)
                    new_log_prob += Scalar[DT](-0.5) * (
                        LOG_2PI + Scalar[DT](2.0) * ls + z * z
                    )
                var diff = new_log_prob - old_log_prob[b]
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
                var adv = advantage[b]
                var unclipped_obj = ratio * adv
                var clipped_obj = clipped_ratio * adv
                var is_clipped = clipped_obj < unclipped_obj

                for j in range(Self.ACT):
                    if is_clipped:
                        grad_actor_output[b, j] = 0.0
                        grad_actor_output[b, Self.ACT + j] = (
                            -self.entropy_coef * Scalar[DT](1.0) * inv_batch
                        )
                    else:
                        var mu = actor_output[b, j]
                        var ls = actor_output[b, Self.ACT + j]
                        if ls < LOG_STD_MIN:
                            ls = LOG_STD_MIN
                        elif ls > LOG_STD_MAX:
                            ls = LOG_STD_MAX
                        var std = exp(ls)
                        var a = action[b, j]
                        var z = (a - mu) / (std + EPS_STD)
                        var d_lp_d_mu = z / (std + EPS_STD)
                        var d_lp_d_ls = z * z - Scalar[DT](1.0)
                        var gmu = -adv * ratio * d_lp_d_mu * inv_batch
                        var gls = (
                            -adv * ratio * d_lp_d_ls
                            - self.entropy_coef * Scalar[DT](1.0)
                        ) * inv_batch
                        if gmu > GRAD_CLIP:
                            gmu = GRAD_CLIP
                        elif gmu < -GRAD_CLIP:
                            gmu = -GRAD_CLIP
                        if gls > GRAD_CLIP:
                            gls = GRAD_CLIP
                        elif gls < -GRAD_CLIP:
                            gls = -GRAD_CLIP
                        grad_actor_output[b, j] = gmu
                        grad_actor_output[b, Self.ACT + j] = gls
        else:
            var ctx = self.ctx.value()
            comptime ao_layout = Layout.row_major(BATCH, 2 * Self.ACT)
            comptime ac_layout = Layout.row_major(BATCH, Self.ACT)
            comptime row_layout = Layout.row_major(BATCH)
            var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](actor_output.ptr)
            var ac_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](action.ptr)
            var ol_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](old_log_prob.ptr)
            var ad_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](advantage.ptr)
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_actor_output.ptr)
            var ao_lt = LayoutTensor[DT, ao_layout, MutAnyOrigin](ao_p)
            var ac_lt = LayoutTensor[DT, ac_layout, MutAnyOrigin](ac_p)
            var ol_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](ol_p)
            var ad_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](ad_p)
            var go_lt = LayoutTensor[DT, ao_layout, MutAnyOrigin](go_p)
            comptime TPB = 64
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _ppo_actor_backward_kernel[BATCH, Self.ACT]
            ctx.enqueue_function[kernel](
                ao_lt, ac_lt, ol_lt, ad_lt, go_lt,
                self.clip_eps, self.entropy_coef,
                grid_dim=n_blocks, block_dim=TPB,
            )
