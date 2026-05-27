"""PPOObjective[ACT_] — quaternary Module for the PPO clipped-surrogate loss.

Phase I.2.5. Upgraded from binary (actor_out, packed_aux) to quaternary
(actor_out, action, old_log_prob, advantage) — the natural shape of
the PPO loss. The earlier aux-packing workaround was a leak in the
FullGraph thesis at I.2 landing time; I.2.5's GraphNode N-ary refactor
makes it possible to declare each of the four inputs as a distinct
InputSlot in `PPOActorLoss` without packing.

Inputs (ARITY=4):

  - actor_output     [BATCH, 2*ACT]   from GaussianHead: [mu | log_std].
  - action           [BATCH, ACT]     unbounded sample stored at rollout time.
  - old_log_prob     [BATCH, 1]       log p_θ_old(action | s) at rollout.
  - advantage        [BATCH, 1]       GAE advantage, normalised per minibatch.

Output:

  - loss_per_b       [BATCH, 1]       per-sample (un-averaged) PPO loss.

The 1/BATCH mean factor lives in the seed gradient
(`seed_grad_inv_batch`), not inside this kernel.

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

Per-element grad clip ±10. grad_action / grad_old_log_prob / grad_advantage
are all identically zero (non-differentiable rollout-time inputs).

Forward caches the four input pointers (no copy — graph keeps the buffers
live across forward + vjp), `vjp` reads them. Mirrors MSELoss's
`cache_logits` pattern.

GPU forward / vjp are one-thread-per-batch-row kernels (per-row math
is purely local; no cross-batch reductions). Cached input pointers
stamped by forward are device-side tile pointers; the graph keeps
those buffers live across forward + vjp, so vjp's kernel reads from
the same device addresses without an extra cache copy.
"""

from std.math import exp
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

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


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one thread per batch row. Per-row math is purely local
# (no cross-batch reductions), so this maps 1:1 to a `BATCH`-wide grid.
# ──────────────────────────────────────────────────────────────────────


def _ppo_forward_kernel[ACT: Int, BATCH: Int](
    ao: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
    olp: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    adv: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    loss_out: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var new_log_prob: Scalar[DT] = 0.0
    var entropy: Scalar[DT] = 0.0
    for j in range(ACT):
        var mu = rebind[Scalar[DT]](ao[b, j])
        var ls = rebind[Scalar[DT]](ao[b, ACT + j])
        if ls < LOG_STD_MIN:
            ls = LOG_STD_MIN
        elif ls > LOG_STD_MAX:
            ls = LOG_STD_MAX
        var std = exp(ls)
        var a = rebind[Scalar[DT]](act[b, j])
        var z = (a - mu) / (std + EPS_STD)
        new_log_prob += Scalar[DT](-0.5) * (
            LOG_2PI + Scalar[DT](2.0) * ls + z * z
        )
        entropy += Scalar[DT](0.5) * (
            LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls
        )
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


def _ppo_vjp_kernel[ACT: Int, BATCH: Int](
    ao: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
    olp: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    adv: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    gi0: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    gi1: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
    gi2: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    gi3: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    # Non-differentiable rollout inputs — zero their grad slots.
    for j in range(ACT):
        gi1[b, j] = Scalar[DT](0.0)
    gi2[b, 0] = Scalar[DT](0.0)
    gi3[b, 0] = Scalar[DT](0.0)
    # Re-derive ratio + is_clipped (same math as forward).
    var new_log_prob: Scalar[DT] = 0.0
    for j in range(ACT):
        var mu = rebind[Scalar[DT]](ao[b, j])
        var ls = rebind[Scalar[DT]](ao[b, ACT + j])
        if ls < LOG_STD_MIN:
            ls = LOG_STD_MIN
        elif ls > LOG_STD_MAX:
            ls = LOG_STD_MAX
        var std = exp(ls)
        var a = rebind[Scalar[DT]](act[b, j])
        var z = (a - mu) / (std + EPS_STD)
        new_log_prob += Scalar[DT](-0.5) * (
            LOG_2PI + Scalar[DT](2.0) * ls + z * z
        )
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
    for j in range(ACT):
        if is_clipped:
            gi0[b, j] = Scalar[DT](0.0)
            gi0[b, ACT + j] = -entropy_coef * Scalar[DT](1.0) * go_b
        else:
            var mu = rebind[Scalar[DT]](ao[b, j])
            var ls = rebind[Scalar[DT]](ao[b, ACT + j])
            if ls < LOG_STD_MIN:
                ls = LOG_STD_MIN
            elif ls > LOG_STD_MAX:
                ls = LOG_STD_MAX
            var std = exp(ls)
            var a = rebind[Scalar[DT]](act[b, j])
            var z = (a - mu) / (std + EPS_STD)
            var d_lp_d_mu = z / (std + EPS_STD)
            var d_lp_d_ls = z * z - Scalar[DT](1.0)
            var gmu = -adv_b * ratio * d_lp_d_mu * go_b
            var gls = (
                -adv_b * ratio * d_lp_d_ls - entropy_coef
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


struct PPOObjective[ACT_: Int](Module):
    """PPO clipped-surrogate + entropy bonus as a quaternary Module."""

    comptime ARITY: Int = 4
    # [actor_output (2*ACT) | action (ACT) | old_log_prob (1) | advantage (1)].
    comptime IN_DIMS = Self._build_in_dims()
    comptime OUT_DIM: Int = 1                  # per-sample loss

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, 4]:
        var d = InlineArray[Int, 4](fill=1)
        d[0] = 2 * Self.ACT_
        d[1] = Self.ACT_
        # d[2] = 1, d[3] = 1 already from fill=1
        return d

    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]

    # Input-pointer cache populated by forward, consumed by vjp.
    var _cache_ao_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _cache_act_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _cache_olp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _cache_adv_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var ts: TargetStorage

    def __init__(out self):
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._cache_ao_ptr = null_p
        self._cache_act_ptr = null_p
        self._cache_olp_ptr = null_p
        self._cache_adv_ptr = null_p
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "PPOObjective: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "PPOObjective.make[target='gpu']: ctx required"
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
        assert_tag_for["PPOObjective", target](self.ts.target_tag)
        comptime ACT = Self.ACT_

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
                    var a = act[b, j]
                    var z = (a - mu) / (std + EPS_STD)
                    new_log_prob += Scalar[DT](-0.5) * (
                        LOG_2PI + Scalar[DT](2.0) * ls + z * z
                    )
                    entropy += Scalar[DT](0.5) * (
                        LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls
                    )
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
            # GPU: reconstruct LayoutTensors over the typed-view raw
            # pointers and dispatch one thread per batch row.
            var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ao.ptr)
            var act_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](act.ptr)
            var olp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](olp.ptr)
            var adv_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](adv.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.ptr)
            var ao_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin,
            ](ao_p)
            var act_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, ACT), MutAnyOrigin,
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
            comptime TPB = 128
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime fwd_kernel = _ppo_forward_kernel[ACT, BATCH]
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
        assert_tag_for["PPOObjective", target](self.ts.target_tag)
        comptime ACT = Self.ACT_

        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi0 = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])    # grad_actor_output
        var gi1 = typed_view_mut[BATCH, Self.IN_DIMS[1]](grad_inputs[1])   # grad_action
        var gi2 = typed_view_mut[BATCH, Self.IN_DIMS[2]](grad_inputs[2])   # grad_old_log_prob
        var gi3 = typed_view_mut[BATCH, Self.IN_DIMS[3]](grad_inputs[3])   # grad_advantage

        comptime if target == "cpu":
            # Non-differentiable rollout inputs — zero their grad slots
            # so ComputeGraph's scatter-add is a no-op.
            for b in range(BATCH):
                for j in range(ACT):
                    gi1[b, j] = Scalar[DT](0.0)
                gi2[b, 0] = Scalar[DT](0.0)
                gi3[b, 0] = Scalar[DT](0.0)
            var ao_p = self._cache_ao_ptr
            var act_p = self._cache_act_ptr
            var olp_p = self._cache_olp_ptr
            var adv_p = self._cache_adv_ptr
            for b in range(BATCH):
                var new_log_prob: Scalar[DT] = 0.0
                for j in range(ACT):
                    var mu = ao_p[b * Self.IN_DIMS[0] + j]
                    var ls = ao_p[b * Self.IN_DIMS[0] + ACT + j]
                    if ls < LOG_STD_MIN:
                        ls = LOG_STD_MIN
                    elif ls > LOG_STD_MAX:
                        ls = LOG_STD_MAX
                    var std = exp(ls)
                    var a = act_p[b * Self.IN_DIMS[1] + j]
                    var z = (a - mu) / (std + EPS_STD)
                    new_log_prob += Scalar[DT](-0.5) * (
                        LOG_2PI + Scalar[DT](2.0) * ls + z * z
                    )
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

                for j in range(ACT):
                    if is_clipped:
                        gi0[b, j] = Scalar[DT](0.0)
                        gi0[b, ACT + j] = (
                            -self.entropy_coef * Scalar[DT](1.0) * go_b
                        )
                    else:
                        var mu = ao_p[b * Self.IN_DIMS[0] + j]
                        var ls = ao_p[b * Self.IN_DIMS[0] + ACT + j]
                        if ls < LOG_STD_MIN:
                            ls = LOG_STD_MIN
                        elif ls > LOG_STD_MAX:
                            ls = LOG_STD_MAX
                        var std = exp(ls)
                        var a = act_p[b * Self.IN_DIMS[1] + j]
                        var z = (a - mu) / (std + EPS_STD)
                        var d_lp_d_mu = z / (std + EPS_STD)
                        var d_lp_d_ls = z * z - Scalar[DT](1.0)
                        var gmu = -adv_b * ratio * d_lp_d_mu * go_b
                        var gls = (
                            -adv_b * ratio * d_lp_d_ls
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
            # GPU: kernel zeros gi1/gi2/gi3 and computes gi0 in one pass.
            # Cached input pointers are the device tile pointers stamped
            # by forward (graph buffers stay live across forward+vjp).
            var ao_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin,
            ](self._cache_ao_ptr)
            var act_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, ACT), MutAnyOrigin,
            ](self._cache_act_ptr)
            var olp_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](self._cache_olp_ptr)
            var adv_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](self._cache_adv_ptr)
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr)
            var gi0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi0.ptr)
            var gi1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi1.ptr)
            var gi2_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi2.ptr)
            var gi3_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi3.ptr)
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](go_p)
            var gi0_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin,
            ](gi0_p)
            var gi1_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, ACT), MutAnyOrigin,
            ](gi1_p)
            var gi2_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](gi2_p)
            var gi3_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](gi3_p)
            comptime TPB = 128
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime vjp_kernel = _ppo_vjp_kernel[ACT, BATCH]
            var ctx = self.ts.ctx.value()
            ctx.enqueue_function[vjp_kernel](
                ao_lt, act_lt, olp_lt, adv_lt, go_lt,
                gi0_lt, gi1_lt, gi2_lt, gi3_lt,
                self.clip_eps, self.entropy_coef,
                grid_dim=n_blocks, block_dim=TPB,
            )
