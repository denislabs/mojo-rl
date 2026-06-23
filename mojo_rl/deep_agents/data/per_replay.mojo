"""Prioritized Experience Replay (PER) — GPU data + sum-tree (device-
resident by default, host fallback behind `DEVICE_TREE_=False`).

Phase C.3 — Schaul et al. 2016. Sampling probability of slot `i` is
proportional to `p_i^α` where `p_i = |TD_error_i| + ε`. New slots get
the current `max_priority` so they're sampled often initially. The
training loss is corrected with Importance-Sampling weights
`w_i = (N · P(i))^{−β} / max_w` to compensate for the bias.

Two tree backends, selected by the comptime `DEVICE_TREE_` flag:

  * **Device tree (default)** — `tree_dev` (2·CAP − 1 `DT` entries) +
    `max_priority_dev` live on the GPU. Sampling is a per-lane Philox
    stratified draw + iterative tree descent in one kernel; priority
    updates read `td_errors_dev` directly on device. Zero D2H / H2D /
    `synchronize` per train step — which is what makes the discrete
    train step CUDA-graph-capturable (the host tree forced a
    `enqueue_copy + synchronize` mid-step). Internal nodes are rebuilt
    bottom-up, level-by-level, in a single-block barrier kernel — no
    atomics, deterministic. RNG: device Philox (the host tree used
    `random_float64`), so the device tree is statistically equivalent
    but NOT bit-identical to the host tree; the convergence gate is the
    Rainbow Pong run (see docs/DEVICE_PER_TREE_PLAN.md §4-5).
  * **Host tree (`DEVICE_TREE_=False`)** — the original hybrid: host
    sum-tree, host stratified sampling, indices + IS weights H2D'd each
    sample, `td_errors` D2H'd (+ sync) each priority update. Kept as
    the debugging / A-B oracle.

Shared data path (both backends):

  * Replay data (s, a, r, s', done) lives on the GPU in a wrapped
    `GPUReplay[OBS, ACT, CAP, OBS_STORE_DT_]` — `add`, `add_batch`, and
    the shared `_gather_batch_kernel` are reused as-is (including the
    uint8 pixel-obs storage option, see gpu_replay.mojo).
  * After `sample[BATCH]`, `self.base.indices` (device int32) and
    `self.weights` (device DT, normalized) hold the sampled slot
    indices and IS weights so the caller can:
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

from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.math import pow as fpow
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.ptr import mptr
from ..training.replay_buffer import ReplayBuffer
from ..training.trainer_block import TrainerState
from .gpu_replay import (
    GPUReplay,
    _gather_batch_kernel,
    _increment_rng_offset_kernel,
)


# ──────────────────────────────────────────────────────────────────────
# Device sum-tree kernels (Part A — docs/DEVICE_PER_TREE_PLAN.md §3).
#
# Tree layout matches the host `tree` List: array heap of 2·CAP − 1 DT
# entries, leaves at [CAP−1, 2·CAP−1), root at 0. All kernels run on
# Apple + NVIDIA (no atomics, no host-captured device addresses).
# ──────────────────────────────────────────────────────────────────────


def _per_leafset_td_kernel[
    BATCH: Int, CAP: Int
](
    tree: LayoutTensor[DT, Layout.row_major(2 * CAP - 1), MutAnyOrigin],
    indices: LayoutTensor[DType.int32, Layout.row_major(BATCH), MutAnyOrigin],
    td: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    alpha: Scalar[DT],
    epsilon: Scalar[DT],
):
    """Set leaf priorities from per-sample TD errors: one thread per
    batch lane, `tree[CAP-1+leaf] = (|td| + ε)^α`. Reads `td` on device
    — no D2H, no synchronize (the host path's capture blocker).

    Duplicate-leaf determinism: the host loop applied lanes
    sequentially, so the LAST lane touching a leaf won. Mirror that by
    letting only the last duplicate write (O(BATCH) scan per thread —
    negligible vs the gather). Internal nodes are NOT touched here;
    `_per_tree_propagate_kernel` rebuilds them after."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var leaf = Int(indices[i])
    for j in range(i + 1, BATCH):
        if Int(indices[j]) == leaf:
            return
    var td_i = rebind[Scalar[DT]](td[i])
    var td_abs = td_i if td_i >= Scalar[DT](0.0) else -td_i
    var raw = td_abs + epsilon
    tree[CAP - 1 + leaf] = fpow(raw, alpha)


def _per_max_priority_kernel[
    BATCH: Int
](
    max_p: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    td: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    epsilon: Scalar[DT],
):
    """Bump the device max-priority ceiling (raw `|TD| + ε` space) over
    the batch. Single-thread tail (mirrors `_compute_scale_kernel` in
    grad_clip) — BATCH is small and this avoids float atomics."""
    if Int(thread_idx.x) != 0:
        return
    var m = rebind[Scalar[DT]](max_p[0])
    for i in range(BATCH):
        var td_i = rebind[Scalar[DT]](td[i])
        var td_abs = td_i if td_i >= Scalar[DT](0.0) else -td_i
        var raw = td_abs + epsilon
        if raw > m:
            m = raw
    max_p[0] = m


def _per_leafset_new_kernel[
    N: Int, CAP: Int
](
    tree: LayoutTensor[DT, Layout.row_major(2 * CAP - 1), MutAnyOrigin],
    max_p: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    start_pos: Int32,
    alpha: Scalar[DT],
):
    """New-insert leaf init: slots `(start_pos + e) % CAP` get
    `max_priority^α` (the standard "new samples get max priority"
    rule). `max_p` is read on device so the ceiling tracks
    `_per_max_priority_kernel` updates without host involvement."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N:
        return
    var leaf = (Int(start_pos) + e) % CAP
    var p = fpow(rebind[Scalar[DT]](max_p[0]), alpha)
    tree[CAP - 1 + leaf] = p


def _per_tree_propagate_kernel[
    CAP: Int
](tree: LayoutTensor[DT, Layout.row_major(2 * CAP - 1), MutAnyOrigin],):
    """Rebuild ALL internal nodes bottom-up, level by level, in ONE
    single-block launch: threads stride over each level's nodes
    (`tree[n] = tree[2n+1] + tree[2n+2]`), `barrier()` between levels.
    Deterministic, no atomics (the codebase avoids float atomics by
    convention), and full-level recompute sidesteps the concurrent
    ancestor-path race entirely. Work is trivial: ~CAP total adds.

    Launch grid=(1,), block=(TPB,). Level bounds are uniform across
    threads, so the barrier is safe."""
    var tid = Int(thread_idx.x)
    var tpb = Int(block_dim.x)
    # Deepest internal level: largest l with first node 2^l − 1 ≤ CAP − 2.
    var l_start = 0
    while (1 << (l_start + 1)) - 1 <= CAP - 2:
        l_start += 1
    var l = l_start
    while l >= 0:
        var lo = (1 << l) - 1
        var hi = (1 << (l + 1)) - 2
        if hi > CAP - 2:
            hi = CAP - 2
        var node = lo + tid
        while node <= hi:
            tree[node] = rebind[Scalar[DT]](tree[2 * node + 1]) + rebind[
                Scalar[DT]
            ](tree[2 * node + 2])
            node += tpb
        barrier()
        l -= 1


def _per_sample_kernel[
    BATCH: Int, CAP: Int
](
    tree: LayoutTensor[DT, Layout.row_major(2 * CAP - 1), MutAnyOrigin],
    size_buf: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    out_idx: LayoutTensor[DType.int32, Layout.row_major(BATCH), MutAnyOrigin],
    out_w: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    beta: Scalar[DT],
    seed: UInt64,
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Stratified PER sample, one thread per batch lane: draw
    `u ~ Philox uniform` within lane i's `[i·seg, (i+1)·seg)` segment
    (`seg = tree_total / BATCH`), descend the tree iteratively (the
    device form of `_tree_sample_one`), write the leaf index and the
    raw IS weight `(size · P_leaf)^{−β}`. Normalization (divide by the
    max sampled weight) is a follow-up single-thread kernel, matching
    the host two-pass normalize.

    `tree_total = tree[0]` and `size = size_buf[0]` are read ON DEVICE
    (not host scalars) so the kernel stays correct under CUDA-graph
    capture — same lesson as `_sample_indices_kernel`'s `size_buf`.
    NOTE: `beta` IS a host scalar; capture would bake the IS-anneal at
    capture time (acceptable: β is a slow schedule, and the capture
    path can re-capture on anneal milestones if it matters)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var total = rebind[Scalar[DT]](tree[0])
    var size = Int(size_buf[0])
    if total <= Scalar[DT](0.0) or size < 1:
        out_idx[i] = Int32(0)
        out_w[i] = Scalar[DT](1.0)
        return
    var segment = total / Scalar[DT](BATCH)
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var r = Scalar[DT](Float32(philox.step_uniform()[0]))
    var u = segment * (Scalar[DT](i) + r)
    if u >= total:
        u = total - Scalar[DT](1e-7)
    if u < Scalar[DT](0.0):
        u = Scalar[DT](0.0)
    # Iterative root→leaf descent: go left if u ≤ left_sum, else
    # subtract and go right.
    var idx = 0
    while idx < CAP - 1:
        var left = 2 * idx + 1
        var left_sum = rebind[Scalar[DT]](tree[left])
        if u <= left_sum:
            idx = left
        else:
            u = u - left_sum
            idx = left + 1
    var leaf = idx - (CAP - 1)
    if leaf >= size:
        leaf = size - 1
    out_idx[i] = Int32(leaf)
    var p_leaf = rebind[Scalar[DT]](tree[CAP - 1 + leaf])
    var prob = p_leaf / total
    out_w[i] = fpow(Scalar[DT](size) * prob, -beta)


def _per_normalize_weights_kernel[
    BATCH: Int
](w: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],):
    """Normalize IS weights so the max sampled weight is 1.0 (host
    two-pass normalize, `per_replay` host path). Single-thread —
    deterministic and cheap at BATCH scale."""
    if Int(thread_idx.x) != 0:
        return
    var max_w = Scalar[DT](0.0)
    for i in range(BATCH):
        var wi = rebind[Scalar[DT]](w[i])
        if wi > max_w:
            max_w = wi
    if max_w <= Scalar[DT](0.0):
        max_w = Scalar[DT](1.0)
    for i in range(BATCH):
        w[i] = rebind[Scalar[DT]](w[i]) / max_w


def _per_copy_weights_kernel[
    BATCH: Int
](
    dst: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """D2D copy of the normalized IS weights into `state.mb_w` (the
    device-tree `sample_into` path; the host path H2D-copies instead)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    dst[i] = src[i]


# ──────────────────────────────────────────────────────────────────────
# GPUPrioritizedReplay struct.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct GPUPrioritizedReplay[
    OBS_: Int,
    ACT_: Int,
    CAP_: Int,
    OBS_STORE_DT_: DType = DT,
    DEVICE_TREE_: Bool = True,
](ReplayBuffer):
    """PER buffer: GPU data + sum-tree (device-resident by default;
    `DEVICE_TREE_=False` restores the host-tree hybrid).

    Conforms to `ReplayBuffer`: `make` / `add(Lists, ctx)` /
    `sample_into` / `count` / `configure_per` / `set_beta` /
    `update_priorities` form the trait surface; the legacy `new` /
    pointer-based `add` / device-buffer `sample` / `update_priorities`
    methods are retained for callers that pre-date the trait.
    `sample_into` additionally H2D-copies IS weights into `state.mb_w`
    and flips `state.has_per`.

    Storage:
      * `base` — wrapped `GPUReplay[OBS, ACT, CAP, OBS_STORE_DT_]`
        holding the actual transitions and the device-side `indices`
        scratch.
      * `tree_dev` — device sum-tree, `2·CAP − 1` DT entries (the hot
        tree when `DEVICE_TREE_`, default). Leaves at
        `[CAP-1 .. 2·CAP-2]`, internal nodes `[0 .. CAP-2]`, root at 0.
      * `max_priority_dev` — 1-elem device raw-|TD| ceiling.
      * `tree` — host `List[Scalar[DT]]` mirror layout (the hot tree
        when `DEVICE_TREE_=False`; otherwise unused).
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
    comptime SDT = Self.OBS_STORE_DT_

    var base: GPUReplay[Self.OBS, Self.ACT, Self.CAP, Self.OBS_STORE_DT_]

    # Host sum-tree (hot only when DEVICE_TREE_=False).
    var tree: List[Scalar[DT]]

    # Device sum-tree + raw-|TD| ceiling (hot when DEVICE_TREE_, the
    # default). Always allocated (2·CAP−1 floats ≈ 100 KB at CAP=12k)
    # so the flag flips behaviour without changing the field layout.
    var tree_dev: DeviceBuffer[DT]
    var max_priority_dev: DeviceBuffer[DT]

    # Device IS weights (populated each sample).
    var weights: DeviceBuffer[DT]

    # Host scratch for sample-time bookkeeping. Sized to
    # `batch_capacity` so multiple `sample[BATCH]` calls with
    # different BATCH ≤ batch_capacity reuse the same buffers. Pinned
    # `HostBuffer`s — direct `[i]` access AND passed straight to
    # `enqueue_copy` (no `.unsafe_ptr()`), pinned for faster H2D/D2H.
    var _host_indices: HostBuffer[DType.int32]
    var _host_weights: HostBuffer[DT]
    var _host_td: HostBuffer[DT]

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
        var base = GPUReplay[
            Self.OBS, Self.ACT, Self.CAP, Self.OBS_STORE_DT_
        ].new(ctx, batch_capacity=batch_capacity)
        var tree = List[Scalar[DT]](
            length=2 * Self.CAP - 1,
            fill=Scalar[DT](0.0),
        )
        var tree_dev = ctx.enqueue_create_buffer[DT](2 * Self.CAP - 1)
        tree_dev.enqueue_fill(Scalar[DT](0.0))
        var max_priority_dev = ctx.enqueue_create_buffer[DT](1)
        max_priority_dev.enqueue_fill(Scalar[DT](1.0))
        var weights = ctx.enqueue_create_buffer[DT](batch_capacity)
        weights.enqueue_fill(Scalar[DT](1.0))
        # Pinned host scratch (overwritten before read each sample; the fills
        # below just match the prior List-init for determinism).
        var host_indices = ctx.enqueue_create_host_buffer[DType.int32](
            batch_capacity
        )
        var host_weights = ctx.enqueue_create_host_buffer[DT](batch_capacity)
        var host_td = ctx.enqueue_create_host_buffer[DT](batch_capacity)
        ctx.synchronize()
        for i in range(batch_capacity):
            host_indices[i] = Int32(0)
            host_weights[i] = Scalar[DT](1.0)
            host_td[i] = Scalar[DT](0.0)
        return Self(
            base=base^,
            tree=tree^,
            tree_dev=tree_dev^,
            max_priority_dev=max_priority_dev^,
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

    def tree_total_sync(self, ctx: DeviceContext) raises -> Scalar[DT]:
        """Live sum-tree root, backend-agnostic — debug/test ONLY (the
        device path D2H-copies the tree and synchronizes, exactly the
        round-trip the hot path exists to avoid)."""
        comptime if Self.DEVICE_TREE_:
            var h = ctx.enqueue_create_host_buffer[DT](2 * Self.CAP - 1)
            ctx.enqueue_copy(h, self.tree_dev)
            ctx.synchronize()
            return h[0]
        else:
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
    # Device sum-tree primitives (DEVICE_TREE_ hot path).
    # ──────────────────────────────────────────────────────────────

    def _tree_dev_lt(
        self,
    ) -> LayoutTensor[DT, Layout.row_major(2 * Self.CAP - 1), MutAnyOrigin]:
        return rebind[
            LayoutTensor[DT, Layout.row_major(2 * Self.CAP - 1), MutAnyOrigin]
        ](LayoutTensor[DT, Layout.row_major(2 * Self.CAP - 1)](self.tree_dev))

    def _max_p_lt(
        self,
    ) -> LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin]:
        return rebind[LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin]](
            LayoutTensor[DT, Layout.row_major(1)](self.max_priority_dev)
        )

    def _device_propagate(self, ctx: DeviceContext) raises:
        """Rebuild all internal tree nodes from the (already-written)
        leaves — one single-block barrier kernel."""
        ctx.enqueue_function[_per_tree_propagate_kernel[Self.CAP]](
            self._tree_dev_lt(),
            grid_dim=1,
            block_dim=TPB,
        )

    def _device_leafset_new[
        N: Int
    ](self, ctx: DeviceContext, start_pos: Int,) raises:
        """Init `N` new leaves at `(start_pos + e) % CAP` to
        `max_priority^α`, then propagate."""
        comptime n_blocks = (N + TPB - 1) // TPB
        ctx.enqueue_function[_per_leafset_new_kernel[N, Self.CAP]](
            self._tree_dev_lt(),
            self._max_p_lt(),
            Int32(start_pos),
            self.alpha,
            grid_dim=n_blocks,
            block_dim=TPB,
        )
        self._device_propagate(ctx)

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
        comptime if Self.DEVICE_TREE_:
            self._device_leafset_new[1](ctx, leaf_idx)
        else:
            var p = Scalar[DT](
                fpow(Float64(self.max_priority), Float64(self.alpha))
            )
            self._tree_update_leaf(leaf_idx, p)

    def add_batch[
        N_ENVS: Int
    ](
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
            ctx,
            src_obs,
            src_act,
            src_rew,
            src_nxt,
            src_dne,
        )
        comptime if Self.DEVICE_TREE_:
            self._device_leafset_new[N_ENVS](ctx, start_pos)
        else:
            var p = Scalar[DT](
                fpow(Float64(self.max_priority), Float64(self.alpha))
            )
            for e in range(N_ENVS):
                var leaf_idx = (start_pos + e) % Self.CAP
                self._tree_update_leaf(leaf_idx, p)

    # ──────────────────────────────────────────────────────────────
    # Sample (stratified PER sampling + gather kernel).
    # ──────────────────────────────────────────────────────────────

    def sample[
        BATCH: Int
    ](
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
                "GPUPrioritizedReplay.sample[BATCH="
                + String(BATCH)
                + "] exceeds batch_capacity="
                + String(self.batch_capacity)
            )
        if self.base.size < BATCH:
            raise Error(
                "GPUPrioritizedReplay.sample[BATCH="
                + String(BATCH)
                + "] called before buffer holds BATCH transitions ("
                + "size="
                + String(self.base.size)
                + ")"
            )

        comptime if Self.DEVICE_TREE_:
            # Device path: stratified Philox sample + tree descent in one
            # kernel writing `base.indices` + raw weights, then the
            # single-thread normalize. Zero H2D, zero host tree walk.
            var idx_lt_s = LayoutTensor[
                DType.int32,
                Layout.row_major(BATCH),
            ](self.base.indices)
            var w_lt = LayoutTensor[
                DT,
                Layout.row_major(BATCH),
            ](self.weights)
            var size_lt = LayoutTensor[
                DType.int32,
                Layout.row_major(1),
            ](self.base._size_dev)
            var off_lt = LayoutTensor[
                DType.uint64,
                Layout.row_major(1),
            ](self.base._rng_offset_dev)
            comptime n_blocks_s = (BATCH + TPB - 1) // TPB
            ctx.enqueue_function[_per_sample_kernel[BATCH, Self.CAP]](
                self._tree_dev_lt(),
                size_lt,
                idx_lt_s,
                w_lt,
                self.beta,
                self.base.rng_seed,
                off_lt,
                grid_dim=n_blocks_s,
                block_dim=TPB,
            )
            ctx.enqueue_function[_increment_rng_offset_kernel[BATCH]](
                off_lt,
                grid_dim=1,
                block_dim=1,
            )
            ctx.enqueue_function[_per_normalize_weights_kernel[BATCH]](
                w_lt,
                grid_dim=1,
                block_dim=1,
            )
        else:
            var total = self._tree_total()
            var segment = total / Scalar[DT](BATCH)

            # Host-side stratified sampling + (un-normalised) IS weights.
            # `p_min` is the minimum of the leaf priorities we touched;
            # using it as `max_w` normaliser is the standard PER trick
            # (it's the maximum w_i over the *sampled* slice — see
            # Schaul et al. §3.4).
            var max_w_inv = Scalar[DT](0.0)
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
                # w_i = (N * P)^{-β}. Track raw value; normalise after.
                var w = Scalar[DT](
                    fpow(
                        Float64(self.base.size) * Float64(P),
                        Float64(-self.beta),
                    )
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
            ctx.enqueue_copy(self.base.indices, self._host_indices)
            ctx.enqueue_copy(self.weights, self._host_weights)
        self._last_batch = BATCH

        # Gather kernel — reuses `_gather_batch_kernel` from
        # `gpu_replay.mojo` over `(self.base.{obs,act,rew,nxt,dne},
        # self.base.indices)`.
        var idx_lt = LayoutTensor[
            DType.int32,
            Layout.row_major(BATCH),
        ](self.base.indices)
        var mb_s_lt = rebind[
            LayoutTensor[DT, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin]
        ](LayoutTensor[DT, Layout.row_major(BATCH, Self.OBS)](mb_s))
        var mb_a_lt = rebind[
            LayoutTensor[DT, Layout.row_major(BATCH, Self.ACT), MutAnyOrigin]
        ](LayoutTensor[DT, Layout.row_major(BATCH, Self.ACT)](mb_a))
        var mb_r_lt = rebind[
            LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin]
        ](LayoutTensor[DT, Layout.row_major(BATCH)](mb_r))
        var mb_sp_lt = rebind[
            LayoutTensor[DT, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin]
        ](LayoutTensor[DT, Layout.row_major(BATCH, Self.OBS)](mb_sp))
        var mb_d_lt = rebind[
            LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin]
        ](LayoutTensor[DT, Layout.row_major(BATCH)](mb_d))
        var buf_s_lt = LayoutTensor[
            Self.SDT,
            Layout.row_major(Self.CAP, Self.OBS),
        ](self.base.obs)
        var buf_a_lt = LayoutTensor[
            DT,
            Layout.row_major(Self.CAP, Self.ACT),
        ](self.base.act)
        var buf_r_lt = LayoutTensor[
            DT,
            Layout.row_major(Self.CAP),
        ](self.base.rew)
        var buf_sp_lt = LayoutTensor[
            Self.SDT,
            Layout.row_major(Self.CAP, Self.OBS),
        ](self.base.nxt)
        var buf_d_lt = LayoutTensor[
            DT,
            Layout.row_major(Self.CAP),
        ](self.base.dne)

        comptime n_blocks = (BATCH * Self.OBS + TPB - 1) // TPB
        comptime gather_kernel = _gather_batch_kernel[
            BATCH,
            Self.OBS,
            Self.ACT,
            Self.CAP,
            Self.SDT,
        ]
        ctx.enqueue_function[gather_kernel](
            mb_s_lt,
            mb_a_lt,
            mb_r_lt,
            mb_sp_lt,
            mb_d_lt,
            buf_s_lt,
            buf_a_lt,
            buf_r_lt,
            buf_sp_lt,
            buf_d_lt,
            idx_lt,
            grid_dim=n_blocks,
            block_dim=TPB,
        )

    # ──────────────────────────────────────────────────────────────
    # Priority update — D2H td_errors, refresh tree leaves.
    # ──────────────────────────────────────────────────────────────

    def update_priorities[
        BATCH: Int
    ](mut self, ctx: DeviceContext, td_errors_dev: DeviceBuffer[DT],) raises:
        """Refresh priorities for the indices returned by the most
        recent `sample[BATCH]` call. Reads `td_errors_dev` device-
        side, computes `p = (|TD| + ε)^α`, updates the sum-tree, and
        bumps `max_priority` (in raw |TD| space) so future inserts
        get the new ceiling.

        Device-tree path: three kernel enqueues (leaf-set, max bump,
        propagate) — no D2H, no `synchronize` (the host path's
        per-train-step capture blocker)."""
        comptime assert BATCH > 0, "BATCH must be > 0"
        if BATCH != self._last_batch:
            raise Error(
                "GPUPrioritizedReplay.update_priorities[BATCH="
                + String(BATCH)
                + "] called with a different BATCH than the last "
                + "sample (last="
                + String(self._last_batch)
                + ")"
            )

        comptime if Self.DEVICE_TREE_:
            var td_lt = rebind[
                LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin]
            ](LayoutTensor[DT, Layout.row_major(BATCH)](td_errors_dev))
            var idx_lt = LayoutTensor[
                DType.int32,
                Layout.row_major(BATCH),
            ](self.base.indices)
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            ctx.enqueue_function[_per_leafset_td_kernel[BATCH, Self.CAP]](
                self._tree_dev_lt(),
                idx_lt,
                td_lt,
                self.alpha,
                self.epsilon,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            ctx.enqueue_function[_per_max_priority_kernel[BATCH]](
                self._max_p_lt(),
                td_lt,
                self.epsilon,
                grid_dim=1,
                block_dim=1,
            )
            self._device_propagate(ctx)
        else:
            ctx.enqueue_copy(self._host_td, td_errors_dev)
            ctx.synchronize()

            var new_max = self.max_priority
            for i in range(BATCH):
                var td = self._host_td[i]
                var td_abs = td if td >= Scalar[DT](0.0) else -td
                var raw = td_abs + self.epsilon
                if raw > new_max:
                    new_max = raw
                var p = Scalar[DT](fpow(Float64(raw), Float64(self.alpha)))
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

    def sample_into[
        BATCH: Int
    ](mut self, mut state: TrainerState[Self.OBS, Self.ACT, BATCH],) raises:
        """Device PER sample into `state.mb_*`, IS weights into
        `state.mb_w` (D2D copy kernel on the device-tree path, H2D on
        the host-tree path), flip `state.has_per`."""
        var ctx = state.ctx.value()
        self.sample[BATCH](
            ctx,
            state.mb_s.dev.value(),
            state.mb_a.dev.value(),
            state.mb_r.dev.value(),
            state.mb_sp.dev.value(),
            state.mb_d.dev.value(),
        )
        comptime if Self.DEVICE_TREE_:
            var src_lt = LayoutTensor[
                DT,
                Layout.row_major(BATCH),
            ](self.weights)
            var dst_lt = LayoutTensor[
                DT,
                Layout.row_major(BATCH),
            ](state.mb_w.dev.value())
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            ctx.enqueue_function[_per_copy_weights_kernel[BATCH]](
                dst_lt,
                src_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
        else:
            ctx.enqueue_copy(
                state.mb_w.dev.value(), self._host_weights
            )
        state.has_per = True

    def update_priorities[
        BATCH: Int
    ](mut self, mut state: TrainerState[Self.OBS, Self.ACT, BATCH],) raises:
        """Trait-surface priority refresh: reads `state.td_residuals`
        (device) and updates the sum-tree."""
        self.update_priorities[BATCH](
            state.ctx.value(), state.td_residuals.dev.value()
        )

    def count(self) -> Int:
        return self.base.size
