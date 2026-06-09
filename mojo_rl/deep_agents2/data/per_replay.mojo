"""Prioritized Experience Replay (PER) — host sum-tree + GPU data.

Phase C.3 — Schaul et al. 2016. Sampling probability of slot `i` is
proportional to `p_i^α` where `p_i = |TD_error_i| + ε`. New slots get
the current `max_priority` so they're sampled often initially. The
training loss is corrected with Importance-Sampling weights
`w_i = (N · P(i))^{−β} / max_w` to compensate for the bias.

Hybrid layout (matches `deep_agents`' `gpu_per_replay_buffer.mojo`
pattern):

  * Replay data (s, a, r, s', done) lives on the GPU in a wrapped
    `GPUReplay[OBS, ACT, CAP]` — `add`, `add_batch`, and the shared
    `_gather_batch_kernel` are reused as-is.
  * The sum-tree (2·CAP − 1 entries) lives on the host. Tree updates
    are O(log CAP) and don't benefit from GPU; sampling produces a
    list of indices that we upload alongside the IS-weight vector.
  * Per `sample[BATCH]`: stratified sampling on the host (one draw
    per `[0, total)`-segment of width `total/BATCH`), enqueue_copy
    indices + weights to device, then call `_gather_batch_kernel`
    to populate the caller's device minibatch buffers. IS weights
    + indices are accessible via the `weights` / `indices` device-
    buffer fields after the call so the caller can:
      1. Compute per-sample TD errors during the gradient step.
      2. Call `update_priorities[BATCH](ctx, td_errors_dev)` to
         refresh tree leaves and `max_priority`.

API surface:

  * `new(ctx, alpha=0.6, beta=0.4, epsilon=1e-6, batch_capacity=BATCH)`
  * `add(ctx, obs_p, act_p, r, nxt_p, d)` — single-transition
    (CPU env step → host pointer inputs).
  * `add_batch[N_ENVS](ctx, src_obs, src_act, src_rew, src_nxt,
    src_dne)` — N_ENVS-batched (GPU env step → device-buffer inputs).
  * `sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)` — uniform-
    stratified PER sampling. After the call, `self.indices`
    (DeviceBuffer[int32]) and `self.weights` (DeviceBuffer[DT]) hold
    the sampled indices and IS weights.
  * `update_priorities[BATCH](ctx, td_errors_dev)` — D2H `td_errors`,
    refresh tree leaves at `self.indices`, update `max_priority`.
  * `set_beta(beta)` — annealed-IS schedule hook.
  * `is_ready[BATCH]() -> Bool` — passthrough to base.

Trainer integration is left to the caller. A `use_per` Saveable
config flag on SACConfig + automatic routing in `SACTrainer.train_step`
is a future C.3b chunk; this commit ships the primitive.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import pow as fpow
from std.memory import alloc
from std.random import random_float64
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.module import mptr
from ..training.replay_buffer import ReplayBuffer
from ..training.trainer_block import TrainerState
from .gpu_replay import GPUReplay, _gather_batch_kernel


# ──────────────────────────────────────────────────────────────────────
# GPUPrioritizedReplay struct.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct GPUPrioritizedReplay[OBS_: Int, ACT_: Int, CAP_: Int](ReplayBuffer):
    """Hybrid PER buffer: GPU data + host sum-tree.

    Conforms to `ReplayBuffer`: `make` / `add(Lists, ctx)` /
    `sample_into` / `count` / `configure_per` / `set_beta` /
    `update_priorities` form the trait surface; the legacy `new` /
    pointer-based `add` / device-buffer `sample` / `update_priorities`
    methods are retained for callers that pre-date the trait.
    `sample_into` additionally H2D-copies IS weights into `state.mb_w`
    and flips `state.has_per`.

    Storage:
      * `base` — wrapped `GPUReplay[OBS, ACT, CAP]` holding the actual
        transitions and the device-side `indices` scratch.
      * `tree` — host `List[Scalar[DT]]` of length `2·CAP − 1`. Leaves
        live at `[CAP-1 .. 2·CAP-2]` (0-indexed). Internal nodes at
        `[0 .. CAP-2]`. Root is `tree[0]`.
      * `weights` — device `[batch_capacity]` IS-weight vector,
        repopulated each `sample` call.

    PER hyperparameters:
      * `alpha` — priority exponent (0=uniform, 1=full prioritization).
      * `beta`  — IS correction exponent (callers anneal 0.4 → 1.0).
      * `epsilon` — small constant added to `|TD|` so zero-error slots
        keep nonzero priority.
      * `max_priority` — current maximum priority; new slots get this
        so they're sampled at the top of the distribution initially.
    """

    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var base: GPUReplay[Self.OBS, Self.ACT, Self.CAP]

    # Host sum-tree.
    var tree: List[Scalar[DT]]

    # Device IS weights (populated each sample).
    var weights: DeviceBuffer[DT]

    # Host scratch for sample-time bookkeeping. Sized to
    # `batch_capacity` so multiple `sample[BATCH]` calls with
    # different BATCH ≤ batch_capacity reuse the same buffers.
    var _host_indices: List[Int32]
    var _host_weights: List[Scalar[DT]]
    var _host_td: List[Scalar[DT]]

    # Last sample's BATCH (set by `sample`, consumed by
    # `update_priorities`).
    var _last_batch: Int

    # PER hyperparams.
    var alpha: Scalar[DT]
    var beta: Scalar[DT]
    var epsilon: Scalar[DT]
    var max_priority: Scalar[DT]

    var batch_capacity: Int

    @staticmethod
    def new(
        ctx: DeviceContext,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
        batch_capacity: Int = 4096,
    ) raises -> Self:
        var base = GPUReplay[Self.OBS, Self.ACT, Self.CAP].new(
            ctx, batch_capacity=batch_capacity,
        )
        var tree = List[Scalar[DT]](
            length=2 * Self.CAP - 1, fill=Scalar[DT](0.0),
        )
        var weights = ctx.enqueue_create_buffer[DT](batch_capacity)
        weights.enqueue_fill(Scalar[DT](1.0))
        var host_indices = List[Int32](
            length=batch_capacity, fill=Int32(0),
        )
        var host_weights = List[Scalar[DT]](
            length=batch_capacity, fill=Scalar[DT](1.0),
        )
        var host_td = List[Scalar[DT]](
            length=batch_capacity, fill=Scalar[DT](0.0),
        )
        return Self(
            base=base^,
            tree=tree^,
            weights=weights^,
            _host_indices=host_indices^,
            _host_weights=host_weights^,
            _host_td=host_td^,
            _last_batch=0,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            max_priority=Scalar[DT](1.0),
            batch_capacity=batch_capacity,
        )

    def is_ready[BATCH: Int](self) -> Bool:
        return self.base.size >= BATCH

    def set_beta(mut self, beta: Scalar[DT]):
        """Hook for annealed-IS schedule; callers typically ramp
        β from 0.4 → 1.0 over training."""
        self.beta = beta

    # ──────────────────────────────────────────────────────────────
    # Sum-tree primitives (host-side; O(log CAP) per call).
    # ──────────────────────────────────────────────────────────────

    def _tree_total(self) -> Scalar[DT]:
        return self.tree[0]

    def _tree_update_leaf(mut self, leaf_idx: Int, priority: Scalar[DT]):
        """Set leaf at `leaf_idx` (0..CAP-1) to `priority`; propagate
        the delta up to the root."""
        var tree_idx = leaf_idx + Self.CAP - 1
        var diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] = self.tree[tree_idx] + diff

    def _tree_sample_one(self, u: Scalar[DT]) -> Int:
        """Descend from the root: at each internal node, go left if
        `u <= left_sum`, else subtract `left_sum` from `u` and go
        right. Returns the leaf index in `[0, CAP)`."""
        var idx: Int = 0
        var v = u
        while idx < Self.CAP - 1:
            var left = 2 * idx + 1
            var right = left + 1
            var left_sum = self.tree[left]
            if v <= left_sum:
                idx = left
            else:
                v = v - left_sum
                idx = right
        return idx - (Self.CAP - 1)

    # ──────────────────────────────────────────────────────────────
    # Add — mirrors GPUReplay surface, plus tree leaf update.
    # ──────────────────────────────────────────────────────────────

    def add(
        mut self,
        ctx: DeviceContext,
        s: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        sp: UnsafePointer[Scalar[DT], MutAnyOrigin],
        d: Scalar[DT],
    ) raises:
        """Single-transition add. The base GPUReplay writes the slot
        device-side; we then update the sum-tree leaf at the (previous)
        write position to `max_priority`."""
        var leaf_idx = self.base.pos  # slot that's about to be written
        self.base.add(ctx, s, a, r, sp, d)
        # Use max_priority^alpha so the priority lives in the same
        # space as updates done by `update_priorities`.
        var p = Scalar[DT](
            fpow(Float64(self.max_priority), Float64(self.alpha))
        )
        self._tree_update_leaf(leaf_idx, p)

    def add_batch[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_nxt: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """N_ENVS-batched add. Writes all N_ENVS device-side via the
        base buffer's store_batch kernel; then updates N_ENVS tree
        leaves to `max_priority^alpha`."""
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        var start_pos = self.base.pos
        self.base.add_batch[N_ENVS](
            ctx, src_obs, src_act, src_rew, src_nxt, src_dne,
        )
        var p = Scalar[DT](
            fpow(Float64(self.max_priority), Float64(self.alpha))
        )
        for e in range(N_ENVS):
            var leaf_idx = (start_pos + e) % Self.CAP
            self._tree_update_leaf(leaf_idx, p)

    # ──────────────────────────────────────────────────────────────
    # Sample (stratified PER sampling + gather kernel).
    # ──────────────────────────────────────────────────────────────

    def sample[BATCH: Int](
        mut self,
        ctx: DeviceContext,
        mb_s: DeviceBuffer[DT],
        mb_a: DeviceBuffer[DT],
        mb_r: DeviceBuffer[DT],
        mb_sp: DeviceBuffer[DT],
        mb_d: DeviceBuffer[DT],
    ) raises:
        """Stratified PER sampling: partition `[0, total)` into BATCH
        equal segments and draw one sample per segment from the sum-
        tree. The shared `_gather_batch_kernel` then writes the
        sampled transitions into caller-provided device minibatch
        buffers.

        After the call, `self.indices` (device `[BATCH]` int32) holds
        the sampled slot indices and `self.weights` (device `[BATCH]`
        DT) holds the IS weights — both consumed by
        `update_priorities[BATCH](ctx, td_errors_dev)`.
        """
        comptime assert BATCH > 0, "BATCH must be > 0"
        if BATCH > self.batch_capacity:
            raise Error(
                "GPUPrioritizedReplay.sample[BATCH=" + String(BATCH)
                + "] exceeds batch_capacity="
                + String(self.batch_capacity)
            )
        if self.base.size < BATCH:
            raise Error(
                "GPUPrioritizedReplay.sample[BATCH=" + String(BATCH)
                + "] called before buffer holds BATCH transitions ("
                + "size=" + String(self.base.size) + ")"
            )

        var total = self._tree_total()
        var segment = total / Scalar[DT](BATCH)

        # Host-side stratified sampling + (un-normalised) IS weights.
        # `p_min` is the minimum of the leaf priorities we touched;
        # using it as `max_w` normaliser is the standard PER trick
        # (it's the maximum w_i over the *sampled* slice — see
        # Schaul et al. §3.4).
        var max_w_inv = Scalar[DT](0.0)
        var n_inv = Scalar[DT](1.0) / Scalar[DT](self.base.size)
        for i in range(BATCH):
            var lo = segment * Scalar[DT](i)
            var hi = segment * Scalar[DT](i + 1)
            var u = lo + (hi - lo) * Scalar[DT](random_float64())
            # Clamp to avoid edge effects from total accumulation.
            if u >= total:
                u = total - Scalar[DT](1e-7)
            if u < Scalar[DT](0.0):
                u = Scalar[DT](0.0)
            var leaf = self._tree_sample_one(u)
            if leaf >= self.base.size:
                leaf = self.base.size - 1
            self._host_indices[i] = Int32(leaf)
            # P(i) = p_i / total
            var p_leaf = self.tree[leaf + Self.CAP - 1]
            var P = p_leaf / total
            # w_i = (N * P)^{-β}. Track raw value here; normalise after.
            var w = Scalar[DT](
                fpow(Float64(self.base.size) * Float64(P), Float64(-self.beta))
            )
            self._host_weights[i] = w
            if w > max_w_inv:
                max_w_inv = w

        # Normalise so max sampled weight == 1.0.
        if max_w_inv <= Scalar[DT](0.0):
            max_w_inv = Scalar[DT](1.0)
        for i in range(BATCH):
            self._host_weights[i] = self._host_weights[i] / max_w_inv

        # Upload indices + weights to device.
        ctx.enqueue_copy(self.base.indices, self._host_indices.unsafe_ptr())
        ctx.enqueue_copy(self.weights, self._host_weights.unsafe_ptr())
        self._last_batch = BATCH

        # Gather kernel — reuses `_gather_batch_kernel` from
        # `gpu_replay.mojo` over `(self.base.{obs,act,rew,nxt,dne},
        # self.base.indices)`.
        var idx_lt = LayoutTensor[
            DType.int32, Layout.row_major(BATCH), MutAnyOrigin,
        ](self.base.indices.unsafe_ptr())
        var mb_s_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin,
        ](mb_s.unsafe_ptr())
        var mb_a_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.ACT), MutAnyOrigin,
        ](mb_a.unsafe_ptr())
        var mb_r_lt = LayoutTensor[
            DT, Layout.row_major(BATCH), MutAnyOrigin,
        ](mb_r.unsafe_ptr())
        var mb_sp_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin,
        ](mb_sp.unsafe_ptr())
        var mb_d_lt = LayoutTensor[
            DT, Layout.row_major(BATCH), MutAnyOrigin,
        ](mb_d.unsafe_ptr())
        var buf_s_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin,
        ](self.base.obs.unsafe_ptr())
        var buf_a_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.ACT), MutAnyOrigin,
        ](self.base.act.unsafe_ptr())
        var buf_r_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin,
        ](self.base.rew.unsafe_ptr())
        var buf_sp_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin,
        ](self.base.nxt.unsafe_ptr())
        var buf_d_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin,
        ](self.base.dne.unsafe_ptr())

        comptime n_blocks = (BATCH * Self.OBS + TPB - 1) // TPB
        comptime gather_kernel = _gather_batch_kernel[
            BATCH, Self.OBS, Self.ACT, Self.CAP,
        ]
        ctx.enqueue_function[gather_kernel](
            mb_s_lt, mb_a_lt, mb_r_lt, mb_sp_lt, mb_d_lt,
            buf_s_lt, buf_a_lt, buf_r_lt, buf_sp_lt, buf_d_lt,
            idx_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )

    # ──────────────────────────────────────────────────────────────
    # Priority update — D2H td_errors, refresh tree leaves.
    # ──────────────────────────────────────────────────────────────

    def update_priorities[BATCH: Int](
        mut self,
        ctx: DeviceContext,
        td_errors_dev: DeviceBuffer[DT],
    ) raises:
        """Refresh priorities for the indices returned by the most
        recent `sample[BATCH]` call. Reads `td_errors_dev` device-
        side, computes `p = (|TD| + ε)^α`, updates the sum-tree, and
        bumps `max_priority` (in raw |TD| space) so future inserts
        get the new ceiling.
        """
        comptime assert BATCH > 0, "BATCH must be > 0"
        if BATCH != self._last_batch:
            raise Error(
                "GPUPrioritizedReplay.update_priorities[BATCH="
                + String(BATCH)
                + "] called with a different BATCH than the last "
                + "sample (last=" + String(self._last_batch) + ")"
            )

        ctx.enqueue_copy(self._host_td.unsafe_ptr(), td_errors_dev)
        ctx.synchronize()

        var new_max = self.max_priority
        for i in range(BATCH):
            var td = self._host_td[i]
            var td_abs = td if td >= Scalar[DT](0.0) else -td
            var raw = td_abs + self.epsilon
            if raw > new_max:
                new_max = raw
            var p = Scalar[DT](
                fpow(Float64(raw), Float64(self.alpha))
            )
            var leaf = Int(self._host_indices[i])
            self._tree_update_leaf(leaf, p)
        self.max_priority = new_max

    # ─── ReplayBuffer trait surface ──────────────────────────────────

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        """Trait factory. `ctx` required (device storage). PER exponents
        take `new`'s defaults; the block sets the real ones via
        `configure_per` before any `add`."""
        if not ctx:
            raise Error(
                "GPUPrioritizedReplay.make: ctx required for device storage"
            )
        return Self.new(ctx.value(), batch_capacity=batch_capacity)

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Trait-surface add: stage the host Lists and reuse the
        pointer-based `add`. `ctx` required (raises if None)."""
        if not ctx:
            raise Error("GPUPrioritizedReplay.add: ctx required")
        var s_p = mptr(s.unsafe_ptr())
        var a_p = mptr(a.unsafe_ptr())
        var sp_p = mptr(sp.unsafe_ptr())
        self.add(ctx.value(), s_p, a_p, r, sp_p, d)

    def sample_into[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        """Device PER sample into `state.mb_*`, H2D IS weights into
        `state.mb_w`, flip `state.has_per`."""
        var ctx = state.ctx.value()
        self.sample[BATCH](
            ctx,
            state.mb_s.dev.value(),
            state.mb_a.dev.value(),
            state.mb_r.dev.value(),
            state.mb_sp.dev.value(),
            state.mb_d.dev.value(),
        )
        ctx.enqueue_copy(
            state.mb_w.dev.value(), self._host_weights.unsafe_ptr()
        )
        state.has_per = True

    def update_priorities[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        """Trait-surface priority refresh: reads `state.td_residuals`
        (device) and updates the sum-tree."""
        self.update_priorities[BATCH](
            state.ctx.value(), state.td_residuals.dev.value()
        )

    def count(self) -> Int:
        return self.base.size
