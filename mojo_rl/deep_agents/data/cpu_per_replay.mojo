"""CPUPrioritizedReplay[OBS, ACT, CAP] — CPU port of GPUPrioritizedReplay.

Mirrors `GPUPrioritizedReplay`'s surface (Schaul et al. 2016) but the
transitions live in plain CPU allocations (via `CPUReplay`) and the
sampling produces caller-side host pointers — no DeviceContext, no
gather kernel.

Why this exists: the C51 trainer is currently CPU-only, so PER on C51
needs a CPU-side block. The `GPUPrioritizedReplay` algorithmic body
ports cleanly because its sum-tree was already host-resident; only the
data-side (storage + gather) changes.

API:
  - `new(alpha, beta, epsilon, batch_capacity)` — CPU constructor.
  - `add(s_p, a_p, r, sp_p, d)` — single-transition add. Caller passes
    host pointers (or List unsafe_ptr) into the same five buffers
    CPUReplay owns.
  - `sample[BATCH](mb_s, mb_a, mb_r, mb_sp, mb_d)` — stratified PER
    sample into caller-provided host pointers. After the call,
    `self._host_indices[BATCH]` and `self._host_weights[BATCH]` hold
    the sampled leaf indices and the IS weights (normalised so
    max(w)==1 over the sampled slice).
  - `update_priorities[BATCH](td_residuals_p)` — read host
    `td_residuals_p [BATCH]`, refresh tree leaves at
    `self._host_indices[:BATCH]`, bump `max_priority`.
  - `set_beta(beta)` — IS-β anneal hook.
  - `is_ready[BATCH]() -> Bool` — passthrough to base.

The corresponding `PerSampleCpuStep` SampleBlock (in
`mojo_rl/deep_agents/training/blocks/per_sample_cpu_step.mojo`)
wraps this struct, copies the sampled minibatch into `state.mb_*`,
copies IS weights into `state.mb_w`, and flips `state.has_per = True`.
"""

from std.math import pow as fpow
from std.memory import alloc
from std.random import random_float64
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..training.replay_buffer import ReplayBuffer
from ..training.trainer_block import TrainerState
from .cpu_replay import CPUReplay


@fieldwise_init
struct CPUPrioritizedReplay[OBS_: Int, ACT_: Int, CAP_: Int](ReplayBuffer):
    """Host PER buffer: CPU data + host sum-tree.

    Conforms to `ReplayBuffer`: `make` / `add` / `sample_into` / `count`
    / `configure_per` / `set_beta` / `update_priorities` form the trait
    surface; the legacy `new` / pointer-based `sample` /
    `update_priorities` methods are retained for callers that pre-date
    the trait. `sample_into` additionally fills `state.mb_w` and flips
    `state.has_per`.

    Storage:
      * `base` — wrapped `CPUReplay[OBS, ACT, CAP]` holding the actual
        transitions.
      * `tree` — host `List[Scalar[DT]]` of length `2·CAP − 1`. Leaves
        live at `[CAP−1 .. 2·CAP−2]`. Internal nodes at `[0 .. CAP−2]`.
        Root is `tree[0]`.

    Bookkeeping:
      * `_host_indices [batch_capacity]` — leaf indices from the most
        recent `sample[BATCH]`. Consumed by `update_priorities[BATCH]`.
      * `_host_weights [batch_capacity]` — IS weights from the most
        recent `sample[BATCH]`. The caller (PerSampleCpuStep) copies
        these into `state.mb_w` post-sample.
      * `_last_batch` — verifies sample/update_priorities BATCH match.

    PER hyperparams: `alpha`, `beta`, `epsilon`, `max_priority`
    (raw |TD| ceiling — new slots get `max_priority^α`).
    """

    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var base: CPUReplay[Self.OBS, Self.ACT, Self.CAP]

    var tree: List[Scalar[DT]]
    var _host_indices: List[Int]
    var _host_weights: List[Scalar[DT]]
    var _last_batch: Int

    var alpha: Scalar[DT]
    var beta: Scalar[DT]
    var epsilon: Scalar[DT]
    var max_priority: Scalar[DT]

    var batch_capacity: Int

    @staticmethod
    def new(
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
        batch_capacity: Int = 4096,
    ) raises -> Self:
        var base = CPUReplay[Self.OBS, Self.ACT, Self.CAP].new()
        var tree = List[Scalar[DT]](
            length=2 * Self.CAP - 1, fill=Scalar[DT](0.0),
        )
        var host_indices = List[Int](
            length=batch_capacity, fill=0,
        )
        var host_weights = List[Scalar[DT]](
            length=batch_capacity, fill=Scalar[DT](1.0),
        )
        return Self(
            base=base^,
            tree=tree^,
            _host_indices=host_indices^,
            _host_weights=host_weights^,
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
        self.beta = beta

    # ──────────────────────────────────────────────────────────────
    # Sum-tree primitives (O(log CAP) per call).
    # ──────────────────────────────────────────────────────────────

    def _tree_total(self) -> Scalar[DT]:
        return self.tree[0]

    def _tree_update_leaf(mut self, leaf_idx: Int, priority: Scalar[DT]):
        var tree_idx = leaf_idx + Self.CAP - 1
        var diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] = self.tree[tree_idx] + diff

    def _tree_sample_one(self, u: Scalar[DT]) -> Int:
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
    # Add — wraps CPUReplay's slot update + tree leaf update.
    # ──────────────────────────────────────────────────────────────

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ):
        """Single-transition add. New slot priority = `max_priority^α`
        so newly-inserted experiences are sampled at the top of the
        distribution initially."""
        var leaf_idx = self.base.pos  # slot about to be written
        self.base.add(s, a, r, sp, d)
        var p = Scalar[DT](
            fpow(Float64(self.max_priority), Float64(self.alpha))
        )
        self._tree_update_leaf(leaf_idx, p)

    # ──────────────────────────────────────────────────────────────
    # Sample (stratified PER sampling, host-side).
    # ──────────────────────────────────────────────────────────────

    def sample[BATCH: Int](
        mut self,
        mut mb_s: List[Scalar[DT]],
        mut mb_a: List[Scalar[DT]],
        mut mb_r: List[Scalar[DT]],
        mut mb_sp: List[Scalar[DT]],
        mut mb_d: List[Scalar[DT]],
    ) raises:
        """Stratified PER sample: partition `[0, total)` into BATCH
        equal segments, draw one uniform per segment, descend the sum-
        tree. Writes the sampled transitions into caller-provided host
        buffers. IS weights stored in `self._host_weights[:BATCH]`,
        normalised so `max(w) == 1` over the sampled slice."""
        comptime assert BATCH > 0, "BATCH must be > 0"
        if BATCH > self.batch_capacity:
            raise Error(
                "CPUPrioritizedReplay.sample[BATCH=" + String(BATCH)
                + "] exceeds batch_capacity="
                + String(self.batch_capacity)
            )
        if self.base.size < BATCH:
            raise Error(
                "CPUPrioritizedReplay.sample[BATCH=" + String(BATCH)
                + "] called before buffer holds BATCH transitions ("
                + "size=" + String(self.base.size) + ")"
            )

        var total = self._tree_total()
        var segment = total / Scalar[DT](BATCH)
        var max_w = Scalar[DT](0.0)
        for i in range(BATCH):
            var lo = segment * Scalar[DT](i)
            var hi = segment * Scalar[DT](i + 1)
            var u = lo + (hi - lo) * Scalar[DT](random_float64())
            if u >= total:
                u = total - Scalar[DT](1e-7)
            if u < Scalar[DT](0.0):
                u = Scalar[DT](0.0)
            var leaf = self._tree_sample_one(u)
            if leaf >= self.base.size:
                leaf = self.base.size - 1
            self._host_indices[i] = leaf
            var p_leaf = self.tree[leaf + Self.CAP - 1]
            var P = p_leaf / total
            # w_i = (N · P_i)^{−β}.
            var w = Scalar[DT](
                fpow(
                    Float64(self.base.size) * Float64(P),
                    Float64(-self.beta),
                )
            )
            self._host_weights[i] = w
            if w > max_w:
                max_w = w

            # Gather transition data into caller buffers.
            for o in range(Self.OBS):
                mb_s[i * Self.OBS + o] = self.base.obs[leaf * Self.OBS + o]
                mb_sp[i * Self.OBS + o] = self.base.nxt[leaf * Self.OBS + o]
            for j in range(Self.ACT):
                mb_a[i * Self.ACT + j] = self.base.act[leaf * Self.ACT + j]
            mb_r[i] = self.base.rew[leaf]
            mb_d[i] = self.base.dne[leaf]

        # Normalise so max sampled weight == 1.
        if max_w <= Scalar[DT](0.0):
            max_w = Scalar[DT](1.0)
        for i in range(BATCH):
            self._host_weights[i] = self._host_weights[i] / max_w

        self._last_batch = BATCH

    # ──────────────────────────────────────────────────────────────
    # Priority update — refresh tree leaves from caller-supplied
    # td_residuals (host pointer).
    # ──────────────────────────────────────────────────────────────

    def update_priorities[BATCH: Int](
        mut self,
        ref td_residuals_p: List[Scalar[DT]],
    ) raises:
        """Refresh priorities for the indices returned by the most
        recent `sample[BATCH]` call. Computes `p = (|TD| + ε)^α`,
        updates the sum-tree, and bumps `max_priority` (raw |TD|
        space) so future inserts get the new ceiling."""
        comptime assert BATCH > 0, "BATCH must be > 0"
        if BATCH != self._last_batch:
            raise Error(
                "CPUPrioritizedReplay.update_priorities[BATCH="
                + String(BATCH)
                + "] called with a different BATCH than the last "
                + "sample (last=" + String(self._last_batch) + ")"
            )

        var new_max = self.max_priority
        for i in range(BATCH):
            var td = td_residuals_p[i]
            var td_abs = td if td >= Scalar[DT](0.0) else -td
            var raw = td_abs + self.epsilon
            if raw > new_max:
                new_max = raw
            var p = Scalar[DT](
                fpow(Float64(raw), Float64(self.alpha))
            )
            var leaf = self._host_indices[i]
            self._tree_update_leaf(leaf, p)
        self.max_priority = new_max

    # ─── ReplayBuffer trait surface ──────────────────────────────────

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        """Trait factory. PER exponents take `new`'s defaults; the block
        sets the real ones via `configure_per` before any `add`. CPU
        backend — `ctx` ignored."""
        return Self.new(batch_capacity=batch_capacity)

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def sample_into[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        """Stratified PER sample into `state.mb_*`, copy normalised IS
        weights into `state.mb_w` (host), flip `state.has_per`."""
        self.sample[BATCH](
            state.mb_s.data,
            state.mb_a.data,
            state.mb_r.data,
            state.mb_sp.data,
            state.mb_d.data,
        )
        for i in range(BATCH):
            state.mb_w.data[i] = self._host_weights[i]
        state.has_per = True

    def update_priorities[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        """Trait-surface priority refresh: reads `state.td_residuals`
        (host) and updates the sum-tree."""
        self.update_priorities[BATCH](state.td_residuals.data)

    def count(self) -> Int:
        return self.base.size
