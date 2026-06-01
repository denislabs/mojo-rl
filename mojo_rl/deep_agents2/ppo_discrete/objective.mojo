"""PPODiscreteObjective[N_] — quaternary Module for the categorical PPO loss.

The discrete (categorical) sibling of `ppo/objective.mojo`'s
`PPOObjective`. Same quaternary shape — (actor_out, action, old_log_prob,
advantage) — but the actor output is a vector of `N_` logits (not a
[mu | log_std] pair) and the action is a single discrete index.

Inputs (ARITY=4):

  - actor_output     [BATCH, N_]      logits over N_ discrete actions.
  - action           [BATCH, 1]       sampled action INDEX (stored as a float).
  - old_log_prob     [BATCH, 1]       log p_θ_old(a | s) at rollout time.
  - advantage        [BATCH, 1]       GAE advantage, normalised per minibatch.

Output:

  - loss_per_b       [BATCH, 1]       per-sample (un-averaged) PPO loss.

The 1/BATCH mean factor lives in the seed gradient
(`seed_grad_inv_batch`), not inside this kernel — identical to the
continuous objective.

Math (per sample b, with a = action index):
    max_l        = max_j logits_j
    sum_exp      = Σ_j exp(logits_j - max_l)
    log_sum      = log(sum_exp)
    p_j          = exp(logits_j - max_l) / sum_exp
    log_p_j      = (logits_j - max_l) - log_sum
    new_log_prob = log_p_a
    entropy      = -Σ_j p_j · log_p_j
    diff         = clamp(new_log_prob - old_log_prob, ±20)
    ratio        = exp(diff)
    unclipped    = ratio · adv
    clipped      = clip(ratio, 1-ε, 1+ε) · adv
    loss_per_b   = -min(unclipped, clipped) - entropy_coef · entropy

Backward (per sample b, with go = grad_loss_per_b[b]):
    is_clipped = clipped < unclipped
    For each action j:
        # ratio (surrogate) term — flows only when NOT clipped:
        d_nlp_j   = (1 if j == a else 0) - p_j
        g_ratio_j = 0                          if is_clipped
                    -adv · ratio · d_nlp_j     otherwise
        # entropy bonus term — always flows (clip flattens only the ratio):
        #   dH/d logit_j = -p_j · (log_p_j + H)
        #   d(-entropy_coef·H)/d logit_j = entropy_coef · p_j · (log_p_j + H)
        g_ent_j   = entropy_coef · p_j · (log_p_j + entropy)
        grad_j    = (g_ratio_j + g_ent_j) · go        (clip ±10)

grad_action / grad_old_log_prob / grad_advantage are identically zero
(non-differentiable rollout-time inputs).

Forward caches the four input pointers (no copy — the graph keeps the
buffers live across forward + vjp), `vjp` reads them. Mirrors
`PPOObjective`'s `cache_logits` pattern exactly.

GPU forward / vjp are one-thread-per-batch-row kernels (per-row math
is purely local; no cross-batch reductions).
"""

from std.math import exp, log as flog
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for


comptime LOG_PROB_DIFF_MAX: Scalar[DT] = 20.0
comptime GRAD_CLIP: Scalar[DT] = 10.0


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one thread per batch row. Per-row math is purely local
# (softmax over N_ + a single selected index), so this maps 1:1 to a
# `BATCH`-wide grid.
# ──────────────────────────────────────────────────────────────────────


def _ppo_disc_forward_kernel[N: Int, BATCH: Int](
    ao: LayoutTensor[DT, Layout.row_major(BATCH, N), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    olp: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    adv: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    loss_out: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var a_idx = Int(rebind[Scalar[DT]](act[b, 0]))
    var max_l = rebind[Scalar[DT]](ao[b, 0])
    for j in range(1, N):
        var lj = rebind[Scalar[DT]](ao[b, j])
        if lj > max_l:
            max_l = lj
    var sum_exp: Scalar[DT] = 0.0
    for j in range(N):
        sum_exp += exp(rebind[Scalar[DT]](ao[b, j]) - max_l)
    var log_sum = flog(sum_exp)
    var entropy: Scalar[DT] = 0.0
    for j in range(N):
        var lp_j = (rebind[Scalar[DT]](ao[b, j]) - max_l) - log_sum
        var p_j = exp(lp_j)
        entropy += -p_j * lp_j
    var new_log_prob = (rebind[Scalar[DT]](ao[b, a_idx]) - max_l) - log_sum
    var olp_b = rebind[Scalar[DT]](olp[b, 0])
    var adv_b = rebind[Scalar[DT]](adv[b, 0])
    var diff = new_log_prob - olp_b
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
    var unclipped_obj = ratio * adv_b
    var clipped_obj = clipped_ratio * adv_b
    var min_obj: Scalar[DT] = unclipped_obj
    if clipped_obj < unclipped_obj:
        min_obj = clipped_obj
    loss_out[b, 0] = -min_obj - entropy_coef * entropy


def _ppo_disc_vjp_kernel[N: Int, BATCH: Int](
    ao: LayoutTensor[DT, Layout.row_major(BATCH, N), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    olp: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    adv: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    gi0: LayoutTensor[DT, Layout.row_major(BATCH, N), MutAnyOrigin],
    gi1: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    gi2: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    gi3: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    # Non-differentiable rollout inputs — zero their grad slots.
    gi1[b, 0] = Scalar[DT](0.0)
    gi2[b, 0] = Scalar[DT](0.0)
    gi3[b, 0] = Scalar[DT](0.0)
    var a_idx = Int(rebind[Scalar[DT]](act[b, 0]))
    var max_l = rebind[Scalar[DT]](ao[b, 0])
    for j in range(1, N):
        var lj = rebind[Scalar[DT]](ao[b, j])
        if lj > max_l:
            max_l = lj
    var sum_exp: Scalar[DT] = 0.0
    for j in range(N):
        sum_exp += exp(rebind[Scalar[DT]](ao[b, j]) - max_l)
    var log_sum = flog(sum_exp)
    var entropy: Scalar[DT] = 0.0
    for j in range(N):
        var lp_j = (rebind[Scalar[DT]](ao[b, j]) - max_l) - log_sum
        var p_j = exp(lp_j)
        entropy += -p_j * lp_j
    var new_log_prob = (rebind[Scalar[DT]](ao[b, a_idx]) - max_l) - log_sum
    var olp_b = rebind[Scalar[DT]](olp[b, 0])
    var adv_b = rebind[Scalar[DT]](adv[b, 0])
    var diff = new_log_prob - olp_b
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
    var unclipped_obj = ratio * adv_b
    var clipped_obj = clipped_ratio * adv_b
    var is_clipped = clipped_obj < unclipped_obj
    var go_b = rebind[Scalar[DT]](go[b, 0])
    for j in range(N):
        var lp_j = (rebind[Scalar[DT]](ao[b, j]) - max_l) - log_sum
        var p_j = exp(lp_j)
        var d_nlp_j = (Scalar[DT](1.0) if j == a_idx else Scalar[DT](0.0)) - p_j
        var g_ratio_j: Scalar[DT] = 0.0
        if not is_clipped:
            g_ratio_j = -adv_b * ratio * d_nlp_j
        var g_ent_j = entropy_coef * p_j * (lp_j + entropy)
        var g = (g_ratio_j + g_ent_j) * go_b
        if g > GRAD_CLIP:
            g = GRAD_CLIP
        elif g < -GRAD_CLIP:
            g = -GRAD_CLIP
        gi0[b, j] = g


struct PPODiscreteObjective[N_: Int](Module):
    """Categorical PPO clipped-surrogate + entropy bonus as a quaternary
    Module. `N_` is the number of discrete actions (actor logit width)."""

    comptime ARITY: Int = 4
    # [logits (N_) | action_idx (1) | old_log_prob (1) | advantage (1)].
    comptime IN_DIMS = Self._build_in_dims()
    comptime OUT_DIM: Int = 1                  # per-sample loss

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, 4]:
        var d = InlineArray[Int, 4](fill=1)
        d[0] = Self.N_
        # d[1] = d[2] = d[3] = 1 already from fill=1
        return d

    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]

    # Input-pointer cache populated by forward, consumed by vjp.
    var _cache_ao_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _cache_act_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _cache_olp_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _cache_adv_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    var ts: TargetStorage

    def __init__(out self):
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)
        self._cache_ao_ptr = None
        self._cache_act_ptr = None
        self._cache_olp_ptr = None
        self._cache_adv_ptr = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "PPODiscreteObjective: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "PPODiscreteObjective.make[target='gpu']: ctx required"
                )
        var op = Self()
        op.ts = TargetStorage.make[target](ctx=ctx)
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
        assert_tag_for["PPODiscreteObjective", target](self.ts.target_tag)
        comptime N = Self.N_

        var ao = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var act = typed_view[BATCH, Self.IN_DIMS[1]](inputs[1])
        var olp = typed_view[BATCH, Self.IN_DIMS[2]](inputs[2])
        var adv = typed_view[BATCH, Self.IN_DIMS[3]](inputs[3])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

        # Cache input pointers for vjp.
        self._cache_ao_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](ao.ptr)
        self._cache_act_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](act.ptr)
        self._cache_olp_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](olp.ptr)
        self._cache_adv_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](adv.ptr)

        comptime if target == "cpu":
            for b in range(BATCH):
                var a_idx = Int(act[b, 0])
                var max_l = ao[b, 0]
                for j in range(1, N):
                    var lj = ao[b, j]
                    if lj > max_l:
                        max_l = lj
                var sum_exp: Scalar[DT] = 0.0
                for j in range(N):
                    sum_exp += exp(ao[b, j] - max_l)
                var log_sum = flog(sum_exp)
                var entropy: Scalar[DT] = 0.0
                for j in range(N):
                    var lp_j = (ao[b, j] - max_l) - log_sum
                    var p_j = exp(lp_j)
                    entropy += -p_j * lp_j
                var new_log_prob = (ao[b, a_idx] - max_l) - log_sum
                var olp_b = olp[b, 0]
                var adv_b = adv[b, 0]
                var diff = new_log_prob - olp_b
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
                var unclipped_obj = ratio * adv_b
                var clipped_obj = clipped_ratio * adv_b
                var min_obj: Scalar[DT] = unclipped_obj
                if clipped_obj < unclipped_obj:
                    min_obj = clipped_obj
                out[b, 0] = -min_obj - self.entropy_coef * entropy
        else:
            var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ao.ptr)
            var act_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](act.ptr)
            var olp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](olp.ptr)
            var adv_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](adv.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.ptr)
            var ao_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, N), MutAnyOrigin,
            ](ao_p)
            var act_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](act_p)
            var olp_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](olp_p)
            var adv_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](adv_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](out_p)
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime fwd_kernel = _ppo_disc_forward_kernel[N, BATCH]
            var ctx = self.ts.ctx.value()
            ctx.enqueue_function[fwd_kernel](
                ao_lt, act_lt, olp_lt, adv_lt, out_lt,
                self.clip_eps, self.entropy_coef,
                grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["PPODiscreteObjective", target](self.ts.target_tag)
        comptime N = Self.N_

        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi0 = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])  # grad logits
        var gi1 = typed_view_mut[BATCH, Self.IN_DIMS[1]](grad_inputs[1])  # grad action
        var gi2 = typed_view_mut[BATCH, Self.IN_DIMS[2]](grad_inputs[2])  # grad olp
        var gi3 = typed_view_mut[BATCH, Self.IN_DIMS[3]](grad_inputs[3])  # grad adv

        comptime if target == "cpu":
            var ao_p = self._cache_ao_ptr.value()
            var act_p = self._cache_act_ptr.value()
            var olp_p = self._cache_olp_ptr.value()
            var adv_p = self._cache_adv_ptr.value()
            for b in range(BATCH):
                # Zero non-differentiable rollout grad slots.
                gi1[b, 0] = Scalar[DT](0.0)
                gi2[b, 0] = Scalar[DT](0.0)
                gi3[b, 0] = Scalar[DT](0.0)
                var a_idx = Int(act_p[b])
                var max_l = ao_p[b * N + 0]
                for j in range(1, N):
                    var lj = ao_p[b * N + j]
                    if lj > max_l:
                        max_l = lj
                var sum_exp: Scalar[DT] = 0.0
                for j in range(N):
                    sum_exp += exp(ao_p[b * N + j] - max_l)
                var log_sum = flog(sum_exp)
                var entropy: Scalar[DT] = 0.0
                for j in range(N):
                    var lp_j = (ao_p[b * N + j] - max_l) - log_sum
                    var p_j = exp(lp_j)
                    entropy += -p_j * lp_j
                var new_log_prob = (ao_p[b * N + a_idx] - max_l) - log_sum
                var olp_b = olp_p[b]
                var adv_b = adv_p[b]
                var diff = new_log_prob - olp_b
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
                var unclipped_obj = ratio * adv_b
                var clipped_obj = clipped_ratio * adv_b
                var is_clipped = clipped_obj < unclipped_obj
                var go_b = go[b, 0]
                for j in range(N):
                    var lp_j = (ao_p[b * N + j] - max_l) - log_sum
                    var p_j = exp(lp_j)
                    var d_nlp_j = (
                        Scalar[DT](1.0) if j == a_idx else Scalar[DT](0.0)
                    ) - p_j
                    var g_ratio_j: Scalar[DT] = 0.0
                    if not is_clipped:
                        g_ratio_j = -adv_b * ratio * d_nlp_j
                    var g_ent_j = self.entropy_coef * p_j * (lp_j + entropy)
                    var g = (g_ratio_j + g_ent_j) * go_b
                    if g > GRAD_CLIP:
                        g = GRAD_CLIP
                    elif g < -GRAD_CLIP:
                        g = -GRAD_CLIP
                    gi0[b, j] = g
        else:
            var ao_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, N), MutAnyOrigin,
            ](self._cache_ao_ptr.value())
            var act_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](self._cache_act_ptr.value())
            var olp_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](self._cache_olp_ptr.value())
            var adv_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](self._cache_adv_ptr.value())
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr)
            var gi0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi0.ptr)
            var gi1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi1.ptr)
            var gi2_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi2.ptr)
            var gi3_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi3.ptr)
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](go_p)
            var gi0_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, N), MutAnyOrigin,
            ](gi0_p)
            var gi1_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](gi1_p)
            var gi2_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](gi2_p)
            var gi3_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](gi3_p)
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime vjp_kernel = _ppo_disc_vjp_kernel[N, BATCH]
            var ctx = self.ts.ctx.value()
            ctx.enqueue_function[vjp_kernel](
                ao_lt, act_lt, olp_lt, adv_lt, go_lt,
                gi0_lt, gi1_lt, gi2_lt, gi3_lt,
                self.clip_eps, self.entropy_coef,
                grid_dim=n_blocks, block_dim=TPB,
            )
