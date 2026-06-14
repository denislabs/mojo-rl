"""Prioritized MuZero/EZv2 trajectory replay — PER over sequence windows.

The prioritized sibling of `MCTSSequenceReplay` (same flat episode rings + the
`sample_training_batch_seq` slab contract the EZv2 unroll consumes), adding
proportional **prioritized experience replay** keyed per stored step:

  * a `SumTree` over the ``CAP`` ring slots holds each step's priority;
  * `sample_training_batch_seq_per` draws each of ``B`` window-starts ∝ priorityᵅ
    (stratified over ``B`` equal mass bins, the standard PER batch draw) and
    emits per-sample importance-sampling weights ``w_i = (N·P_i)^(−β)`` (β=1,
    α=1 ⇒ atari.yaml `priority_prob_alpha/beta`), normalized by the batch max;
  * `update_priorities` writes back fresh priorities (|TD error|, the value-
    prediction error) after the train step; new steps enter at the running max
    priority so they are sampled at least once.

Ring/prune correctness: by the time an episode is pruned (its start has fallen
out of the last ``CAP`` steps) every one of its slots has ALREADY been
overwritten by newer steps — whose `store_episode` reset those slots' priorities
— so a slot's priority always corresponds to its current resident absolute step.
No prune-time priority bookkeeping is needed; the sum-tree stays consistent.

Targets (n-step value, two-player sign flip, absorbing/obs-repeat padding,
truncation bootstrap, consistency boundary mask) are identical to the uniform
replay — only the *which-window* draw and the IS weights differ.
"""

from std.memory import alloc

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import mptr
from mojo_rl.core.sum_tree import SumTree
from .nstep_targets import compute_nstep_value_targets
from ..data.gpu_replay import _obs_quant, _obs_dequant


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


def _asdt[SDT: DType](n: Int) -> UnsafePointer[Scalar[SDT], MutAnyOrigin]:
    return mptr(alloc[Scalar[SDT]](n))


struct PrioritizedMCTSSequenceReplay[
    OBS: Int, ACT: Int, CAP: Int, OBS_STORE_DT: DType = DT
](Movable, ImplicitlyDestructible):
    """Prioritized ring of MCTS-labelled steps + episode index + a per-slot
    `SumTree`. ``CAP`` = max resident steps. ``OBS_STORE_DT`` mirrors
    `MCTSSequenceReplay` (DT or uint8 pixel storage). ``alpha``/``beta`` are
    the PER exponents (EZ Atari: 1.0 / 1.0)."""

    comptime SDT = Self.OBS_STORE_DT

    var obs: UnsafePointer[Scalar[Self.SDT], MutAnyOrigin]   # [CAP, OBS]
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var rew: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var done: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var pol: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP, ACT]
    var val: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var tp: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var legal: UnsafePointer[Scalar[DT], MutAnyOrigin]  # [CAP, ACT]

    var ep_start: List[Int]
    var ep_len: List[Int]
    var ep_trunc: List[Bool]
    var total: Int
    var rng: UInt64

    var tree: SumTree[DT]          # priority per ring slot
    var max_prio: Scalar[DT]       # running max priority (new-sample init)
    var alpha: Scalar[DT]
    var beta: Scalar[DT]
    var eps: Scalar[DT]            # priority floor so nothing is unreachable

    def __init__(
        out self,
        seed: UInt64 = 0,
        alpha: Scalar[DT] = Scalar[DT](1.0),
        beta: Scalar[DT] = Scalar[DT](1.0),
    ):
        self.obs = _asdt[Self.SDT](Self.CAP * Self.OBS)
        self.act = _a(Self.CAP)
        self.rew = _a(Self.CAP)
        self.done = _a(Self.CAP)
        self.pol = _a(Self.CAP * Self.ACT)
        self.val = _a(Self.CAP)
        self.tp = _a(Self.CAP)
        self.legal = _a(Self.CAP * Self.ACT)
        self.ep_start = List[Int]()
        self.ep_len = List[Int]()
        self.ep_trunc = List[Bool]()
        self.total = 0
        self.rng = seed ^ UInt64(0x9E3779B97F4A7C15)
        self.tree = SumTree[DT](Self.CAP)
        self.max_prio = Scalar[DT](1.0)
        self.alpha = alpha
        self.beta = beta
        self.eps = Scalar[DT](1e-6)

    def __del__(deinit self):
        self.obs.free(); self.act.free(); self.rew.free(); self.done.free()
        self.pol.free(); self.val.free(); self.tp.free(); self.legal.free()

    def _xorshift(mut self) -> UInt64:
        var x = self.rng
        x = x ^ (x << 13); x = x ^ (x >> 7); x = x ^ (x << 17)
        self.rng = x
        return x

    def num_episodes(self) -> Int:
        return len(self.ep_start)

    def num_steps(self) -> Int:
        return self.total if self.total < Self.CAP else Self.CAP

    def store_episode(
        mut self,
        ep_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ep_act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ep_rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ep_pol: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ep_val: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ep_tp: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ep_legal: UnsafePointer[Scalar[DT], MutAnyOrigin],
        length: Int,
        truncated: Bool = False,
    ):
        """Append a finished episode; each new step enters the sum-tree at the
        running max priority (standard PER — guarantees it is seen). See
        `MCTSSequenceReplay.store_episode` for the absorbing/truncation rules."""
        var start = self.total
        for i in range(length):
            var slot = (self.total) % Self.CAP
            for j in range(Self.OBS):
                self.obs[slot * Self.OBS + j] = _obs_quant[Self.SDT](
                    ep_obs[i * Self.OBS + j]
                )
            for a in range(Self.ACT):
                self.pol[slot * Self.ACT + a] = ep_pol[i * Self.ACT + a]
            for a in range(Self.ACT):
                self.legal[slot * Self.ACT + a] = ep_legal[i * Self.ACT + a]
            self.act[slot] = ep_act[i]
            self.rew[slot] = ep_rew[i]
            self.val[slot] = ep_val[i]
            self.tp[slot] = ep_tp[i]
            self.done[slot] = (
                Scalar[DT](1.0) if (
                    i == length - 1 and not truncated
                ) else Scalar[DT](0.0)
            )
            # new step → max priority (overwrites any stale priority on this slot)
            self.tree.update(slot, self.max_prio)
            self.total += 1
        self.ep_start.append(start)
        self.ep_len.append(length)
        self.ep_trunc.append(truncated)
        var floor = self.total - Self.CAP
        while len(self.ep_start) > 0 and self.ep_start[0] < floor:
            _ = self.ep_start.pop(0)
            _ = self.ep_len.pop(0)
            _ = self.ep_trunc.pop(0)

    def _slot_to_pos(self, slot: Int) -> Tuple[Int, Int]:
        """Map a ring slot to its current resident ``(episode_idx, offset)``,
        or ``(-1, -1)`` if the slot holds no resident-episode step. ``slot``'s
        resident absolute step is the unique ``a ≡ slot (mod CAP)`` in
        ``[total-num_steps, total)``."""
        var ns = self.num_steps()
        if ns == 0:
            return (-1, -1)
        var lo = self.total - ns
        # largest a <= total-1 with a % CAP == slot
        var a = slot + Self.CAP * ((self.total - 1 - slot) // Self.CAP)
        if a < lo or a >= self.total:
            return (-1, -1)
        # binary search: rightmost episode with ep_start <= a
        var ne = len(self.ep_start)
        if ne == 0:
            return (-1, -1)
        var lo_i = 0
        var hi_i = ne - 1
        var e = -1
        while lo_i <= hi_i:
            var mid = (lo_i + hi_i) // 2
            if self.ep_start[mid] <= a:
                e = mid
                lo_i = mid + 1
            else:
                hi_i = mid - 1
        if e < 0:
            return (-1, -1)
        var off = a - self.ep_start[e]
        if off < 0 or off >= self.ep_len[e]:
            return (-1, -1)
        return (e, off)

    def sample_training_batch_seq_per[
        B: Int, K: Int, N: Int,
    ](
        mut self,
        gamma: Scalar[DT],
        mut obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [K+1, B, OBS]
        mut actions: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [K, B]
        mut policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin], # [K+1, B, ACT]
        mut value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [K+1, B]
        mut reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin], # [K, B]
        mut is_weights: UnsafePointer[Scalar[DT], MutAnyOrigin], # [B] IS weights
        mut sample_slots: UnsafePointer[Int, MutAnyOrigin],      # [B] ring slots
        cons_mask: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None,                                                # [K, B]
    ):
        """Prioritized variant of `sample_training_batch_seq`: each window start
        is drawn ∝ priorityᵅ (stratified over B equal-mass bins), with
        per-sample IS weights ``(N·P_i)^(−β)`` normalized by the batch max.
        ``sample_slots[b]`` records the chosen ring slot so the caller can
        `update_priorities` after the step. Caller guarantees num_episodes()>0
        and num_steps()>0."""
        comptime HV = K + N + 1
        comptime HR = K + N
        var w_rew = _a(HR)
        var w_done = _a(HR)
        var w_val = _a(HV)
        var w_tp = _a(HV)
        var w_vt = _a(K + 1)

        var ns = self.num_steps()
        var total_p = self.tree.total_sum()
        if total_p <= Scalar[DT](0.0):
            total_p = Scalar[DT](1.0)
        var seg = total_p / Scalar[DT](B)

        # First pass: pick slots + raw IS weights; track max weight for norm.
        var max_w = Scalar[DT](0.0)
        for b in range(B):
            # stratified target in bin b
            var u = (Float64(self._xorshift() % UInt64(1_000_000)) / 1_000_000.0)
            var target = (Scalar[DT](b) + Scalar[DT](u)) * seg
            var slot = self.tree.sample(target)
            sample_slots[b] = slot
            var p = self.tree.get(slot)
            if p <= Scalar[DT](0.0):
                p = self.eps
            var prob = p / total_p
            # w = (N * P)^(-beta)
            var w = (Scalar[DT](ns) * prob) ** (-self.beta)
            is_weights[b] = w
            if w > max_w:
                max_w = w

        if max_w <= Scalar[DT](0.0):
            max_w = Scalar[DT](1.0)

        for b in range(B):
            is_weights[b] = is_weights[b] / max_w   # normalize by batch max
            var pos = self._slot_to_pos(sample_slots[b])
            var e = pos[0]
            var s = pos[1]
            if e < 0:
                # Degenerate (slot not resident) — fall back to a uniform step.
                e = Int(self._xorshift() % UInt64(len(self.ep_start)))
                s = Int(self._xorshift() % UInt64(self.ep_len[e]))
                sample_slots[b] = (self.ep_start[e] + s) % Self.CAP
            var L = self.ep_len[e]
            var lv = K + N + 1
            if self.ep_trunc[e]:
                lv = L - 1 - s

            for k in range(K + 1):
                var off = s + k
                if off >= L:
                    off = L - 1
                var oslot = (self.ep_start[e] + off) % Self.CAP
                var ob = k * B * Self.OBS + b * Self.OBS
                for j in range(Self.OBS):
                    obs_seq[ob + j] = _obs_dequant[Self.SDT](
                        self.obs[oslot * Self.OBS + j]
                    )
            if cons_mask:
                var cm = cons_mask.value()
                for k in range(K):
                    cm[k * B + b] = (
                        Scalar[DT](1.0) if s + k + 1 < L else Scalar[DT](0.0)
                    )

            for h in range(HR):
                if s + h >= L:
                    w_rew[h] = Scalar[DT](0.0)
                    w_done[h] = Scalar[DT](1.0)
                else:
                    var rslot = (self.ep_start[e] + s + h) % Self.CAP
                    w_rew[h] = self.rew[rslot]
                    w_done[h] = self.done[rslot]
            for h in range(HV):
                if s + h >= L:
                    w_val[h] = Scalar[DT](0.0)
                    w_tp[h] = Scalar[DT](0.0)
                else:
                    var vslot = (self.ep_start[e] + s + h) % Self.CAP
                    w_val[h] = self.val[vslot]
                    w_tp[h] = self.tp[vslot]

            compute_nstep_value_targets[K, N](
                w_rew, w_done, w_val, w_tp, gamma, w_vt, last_valid=lv
            )

            for k in range(K + 1):
                value_tgt[k * B + b] = w_vt[k]
                var pbase = k * B * Self.ACT + b * Self.ACT
                if s + k >= L:
                    var uni = Scalar[DT](1.0) / Scalar[DT](Self.ACT)
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = uni
                else:
                    var pslot = (self.ep_start[e] + s + k) % Self.CAP
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = self.pol[pslot * Self.ACT + a]
            for k in range(K):
                if s + k >= L:
                    actions[k * B + b] = Scalar[DT](0.0)
                else:
                    var aslot = (self.ep_start[e] + s + k) % Self.CAP
                    actions[k * B + b] = self.act[aslot]
                reward_tgt[k * B + b] = w_rew[k]

        w_rew.free(); w_done.free(); w_val.free(); w_tp.free(); w_vt.free()

    def update_priorities(
        mut self,
        slots: UnsafePointer[Int, MutAnyOrigin],      # [n] ring slots
        priorities: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [n] |TD error|
        n: Int,
    ):
        """Write back fresh priorities (``|TD error|`` + eps)ᵅ for the sampled
        slots and update the running max priority (so new steps enter at it)."""
        for i in range(n):
            var p = priorities[i]
            if p < Scalar[DT](0.0):
                p = -p
            p = (p + self.eps) ** self.alpha
            self.tree.update(slots[i], p)
            if p > self.max_prio:
                self.max_prio = p

    # ── reanalyze hooks (mirror the uniform replay) ──

    def update_targets(
        mut self,
        ep_idx: Int,
        offset: Int,
        new_policy: UnsafePointer[Scalar[DT], MutAnyOrigin],
        new_value: Scalar[DT],
    ):
        if ep_idx < 0 or ep_idx >= len(self.ep_start):
            return
        if offset < 0 or offset >= self.ep_len[ep_idx]:
            return
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for a in range(Self.ACT):
            self.pol[slot * Self.ACT + a] = new_policy[a]
        self.val[slot] = new_value

    def sample_position(mut self) -> Tuple[Int, Int]:
        var e = Int(self._xorshift() % UInt64(len(self.ep_start)))
        var o = Int(self._xorshift() % UInt64(self.ep_len[e]))
        return (e, o)

    def read_obs(
        self,
        ep_idx: Int,
        offset: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for j in range(Self.OBS):
            out[j] = _obs_dequant[Self.SDT](self.obs[slot * Self.OBS + j])
