"""Continuous-action MuZero/EZv2 trajectory replay — action **vectors**.

The continuous twin of `sequence_replay_mcts.mojo::MCTSSequenceReplay`. Same
ring-of-episodes design and the same n-step value bootstrapping, but each step
stores a continuous **action vector** ``a ∈ ℝ^ACT_DIM`` (the planner's chosen
action) instead of a discrete index + visit distribution. For continuous EZv2
the policy is behavior-cloned onto that chosen action, so the same stored vector
serves two roles in the unroll batch:

  * ``actions[K, B, ACT_DIM]`` — the transition actions fed to the dynamics net
    (positions ``s+k``, ``k = 0..K-1``).
  * ``policy_act_tgt[K+1, B, ACT_DIM]`` — the per-position target actions the
    squashed-Gaussian policy clones (positions ``s+k``, ``k = 0..K``).

No visit-policy / legal-mask / to-play fields are stored (single-player
continuous, no reanalyze in v1). Past an episode's terminal: obs is obs-repeat
absorbing (offset clamped to ``L−1``, matching the discrete seq sampler), and
the action is absorbing-zero.
"""

from std.memory import alloc

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import mptr
from .nstep_targets import compute_nstep_value_targets


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


struct MCTSContSequenceReplay[OBS: Int, ACT_DIM: Int, CAP: Int](
    Movable, ImplicitlyDestructible
):
    """Ring of continuous MCTS-labelled steps + episode index. ``CAP`` = max
    resident steps. Host-side (the GPU search feeds it via host copies)."""

    var obs: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP, OBS]
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP, ACT_DIM] action vector
    var rew: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP]
    var done: UnsafePointer[Scalar[DT], MutAnyOrigin]  # [CAP] terminal flag
    var val: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP] root value

    var ep_start: List[Int]   # absolute start step of each resident episode
    var ep_len: List[Int]
    var ep_trunc: List[Bool]  # time-limit truncated (last step NOT terminal)
    var total: Int            # monotonic steps written
    var rng: UInt64

    def __init__(out self, seed: UInt64 = 0):
        self.obs = _a(Self.CAP * Self.OBS)
        self.act = _a(Self.CAP * Self.ACT_DIM)
        self.rew = _a(Self.CAP)
        self.done = _a(Self.CAP)
        self.val = _a(Self.CAP)
        self.ep_start = List[Int]()
        self.ep_len = List[Int]()
        self.ep_trunc = List[Bool]()
        self.total = 0
        self.rng = seed ^ UInt64(0x9E3779B97F4A7C15)

    def __del__(deinit self):
        self.obs.free(); self.act.free(); self.rew.free()
        self.done.free(); self.val.free()

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
        ep_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L, OBS]
        ep_act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L, ACT_DIM]
        ep_rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L]
        ep_val: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L]
        length: Int,
        truncated: Bool = False,
    ):
        """Append a finished episode of ``length`` steps. The terminal flag is
        set on the last step — unless ``truncated`` (time-limit cut, not a real
        terminal): then no step is flagged and the n-step targets bootstrap from
        the last stored root value instead of going to zero. This matters
        enormously for never-terminating envs (Pendulum): EVERY episode is a
        time-limit cut, and terminal-0 there is an *optimistic* corruption
        (0 > any real all-negative-reward value). Evicts episodes that fall out
        of the last ``CAP`` steps so every resident episode stays readable."""
        var start = self.total
        for i in range(length):
            var slot = (self.total) % Self.CAP
            for j in range(Self.OBS):
                self.obs[slot * Self.OBS + j] = ep_obs[i * Self.OBS + j]
            for d in range(Self.ACT_DIM):
                self.act[slot * Self.ACT_DIM + d] = ep_act[i * Self.ACT_DIM + d]
            self.rew[slot] = ep_rew[i]
            self.val[slot] = ep_val[i]
            self.done[slot] = (
                Scalar[DT](1.0) if (
                    i == length - 1 and not truncated
                ) else Scalar[DT](0.0)
            )
            self.total += 1
        self.ep_start.append(start)
        self.ep_len.append(length)
        self.ep_trunc.append(truncated)
        var floor = self.total - Self.CAP
        while len(self.ep_start) > 0 and self.ep_start[0] < floor:
            _ = self.ep_start.pop(0)
            _ = self.ep_len.pop(0)
            _ = self.ep_trunc.pop(0)

    def _read1(
        self,
        field: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ep_idx: Int,
        offset: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        out_base: Int,
    ):
        """Read one scalar ring cell at episode-relative ``offset`` (absorbing
        zeros past the episode end)."""
        if offset >= self.ep_len[ep_idx]:
            out[out_base] = Scalar[DT](0.0)
            return
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        out[out_base] = field[slot]

    def _sample_step_uniform(mut self) -> Tuple[Int, Int]:
        """Pick a resident (episode, offset) with every resident **step**
        equally likely. Episode-uniform sampling over-weights steps from short
        episodes — early random-policy failures keep a majority of the sample
        mass long after the policy improves."""
        var tot = 0
        for e in range(len(self.ep_len)):
            tot += self.ep_len[e]
        var u = Int(self._xorshift() % UInt64(tot))
        var e = 0
        while u >= self.ep_len[e]:
            u -= self.ep_len[e]
            e += 1
        return (e, u)

    def sample_training_batch_seq[
        B: Int, K: Int, N: Int,
    ](
        mut self,
        gamma: Scalar[DT],
        mut obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [K+1, B, OBS]
        mut actions: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [K, B, ACT_DIM]
        mut policy_act_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin], # [K+1, B, ACT_DIM]
        mut value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [K+1, B]
        mut reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin], # [K, B]
        cons_mask: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None,                                                # [K, B]
    ):
        """Fill the continuous EZv2 time-major unroll batch for ``B`` windows.
        Each row picks a random (episode, start); the K+N horizon is read with
        absorbing padding and the value targets are n-step bootstrapped. The obs
        output is the **full sequence** ``obs_seq[K+1, B, OBS]`` (obs-repeat
        absorbing) for the SimSiam targets; ``actions`` / ``policy_act_tgt`` both
        read the stored chosen action vector (absorbing-zero past terminal).
        ``cons_mask`` (if given) receives the ``[K, B]`` episode-boundary mask
        for the consistency loss — row ``(k-1, b)`` is 1 when ``obs_seq[k]`` is
        a real stored observation, 0 when it is absorbing obs-repeat padding.
        Caller guarantees ``num_episodes() > 0``."""
        comptime HV = K + N + 1
        comptime HR = K + N
        var w_rew = _a(HR)
        var w_done = _a(HR)
        var w_val = _a(HV)
        var w_tp = _a(HV)
        var w_vt = _a(K + 1)

        for b in range(B):
            var pos = self._sample_step_uniform()
            var e = pos[0]
            var s = pos[1]
            var L = self.ep_len[e]
            # truncation boundary: last window index with a stored root value
            # (uncapped for naturally-terminated episodes — dones handle those).
            var lv = K + N + 1
            if self.ep_trunc[e]:
                lv = L - 1 - s

            # obs sequence: obs at s+k for k=0..K (obs-repeat absorbing).
            for k in range(K + 1):
                var off = s + k
                if off >= L:
                    off = L - 1
                var slot = (self.ep_start[e] + off) % Self.CAP
                var ob = k * B * Self.OBS + b * Self.OBS
                for j in range(Self.OBS):
                    obs_seq[ob + j] = self.obs[slot * Self.OBS + j]
            # consistency boundary mask: step k = 1..K is real iff s+k < L.
            if cons_mask:
                var cm = cons_mask.value()
                for k in range(K):
                    cm[k * B + b] = (
                        Scalar[DT](1.0) if s + k + 1 < L else Scalar[DT](0.0)
                    )

            for h in range(HR):
                self._read1(self.rew, e, s + h, w_rew, h)
                if s + h >= L:
                    w_done[h] = Scalar[DT](1.0)
                else:
                    self._read1(self.done, e, s + h, w_done, h)
            for h in range(HV):
                if s + h >= L:
                    w_val[h] = Scalar[DT](0.0)
                    w_tp[h] = Scalar[DT](0.0)
                else:
                    self._read1(self.val, e, s + h, w_val, h)
                    w_tp[h] = Scalar[DT](0.0)   # single-player

            compute_nstep_value_targets[K, N](
                w_rew, w_done, w_val, w_tp, gamma, w_vt, last_valid=lv
            )

            # policy-clone targets (chosen action at s+k, absorbing-zero).
            for k in range(K + 1):
                value_tgt[k * B + b] = w_vt[k]
                var pbase = k * B * Self.ACT_DIM + b * Self.ACT_DIM
                if s + k >= L:
                    for d in range(Self.ACT_DIM):
                        policy_act_tgt[pbase + d] = Scalar[DT](0.0)
                else:
                    var slot = (self.ep_start[e] + s + k) % Self.CAP
                    for d in range(Self.ACT_DIM):
                        policy_act_tgt[pbase + d] = self.act[
                            slot * Self.ACT_DIM + d
                        ]
            # transition actions (s+k, absorbing-zero) + horizon reward targets.
            for k in range(K):
                var abase = k * B * Self.ACT_DIM + b * Self.ACT_DIM
                if s + k >= L:
                    for d in range(Self.ACT_DIM):
                        actions[abase + d] = Scalar[DT](0.0)
                else:
                    var slot = (self.ep_start[e] + s + k) % Self.CAP
                    for d in range(Self.ACT_DIM):
                        actions[abase + d] = self.act[slot * Self.ACT_DIM + d]
                reward_tgt[k * B + b] = w_rew[k]

        w_rew.free(); w_done.free(); w_val.free(); w_tp.free(); w_vt.free()

    # ──────────────────────────────────────────────────────────────────
    # Reanalyze hooks — refresh stale targets with a fresher (target) model
    # ──────────────────────────────────────────────────────────────────

    def sample_position(mut self) -> Tuple[Int, Int]:
        """Pick a uniform random resident (episode, in-episode offset) for
        reanalyze. Caller guarantees ``num_episodes() > 0``."""
        var e = Int(self._xorshift() % UInt64(len(self.ep_start)))
        var o = Int(self._xorshift() % UInt64(self.ep_len[e]))
        return (e, o)

    def read_obs(
        self,
        ep_idx: Int,
        offset: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [OBS]
    ):
        """Copy the stored observation at ``(ep_idx, offset)`` into ``out`` — the
        root obs for a reanalyze search."""
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for j in range(Self.OBS):
            out[j] = self.obs[slot * Self.OBS + j]

    def update_targets(
        mut self,
        ep_idx: Int,
        offset: Int,
        new_action: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [ACT_DIM]
        new_value: Scalar[DT],
    ):
        """Reanalyze hook: overwrite a stored step's chosen action **vector**
        (the behavior-clone policy target *and* the dynamics input) and its root
        value with fresh search outputs from a lagging/target network. Pure
        in-place data refresh — timing is the driver's."""
        if ep_idx < 0 or ep_idx >= len(self.ep_start):
            return
        if offset < 0 or offset >= self.ep_len[ep_idx]:
            return
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for d in range(Self.ACT_DIM):
            self.act[slot * Self.ACT_DIM + d] = new_action[d]
        self.val[slot] = new_value
