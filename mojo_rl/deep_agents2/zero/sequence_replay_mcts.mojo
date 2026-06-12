"""MuZero trajectory replay — episodes of MCTS-labelled steps → unroll batches.

Stores complete self-play episodes (each step carries obs, action, reward, the
MCTS visit-policy, the MCTS root value, the player-to-move, and a terminal flag)
in flat ring buffers indexed by a monotonic step counter, with a side list of
``(start, length)`` episode records. Only fully-resident episodes are kept (the
prune rule drops any whose start fell out of the last ``CAP`` steps), so window
reads never cross into a different episode's data.

``sample_training_batch`` produces **exactly** the time-major slabs the unroll
(`muzero/blocks.mojo::mz_unroll_train_step_cpu`) consumes — picking a random
(episode, start) per batch row, reading the K+N horizon with **absorbing
padding** past the episode's terminal (obs repeats, action 0, reward 0, done 1,
uniform policy, value 0), and computing the n-step value targets in place via
`nstep_targets.compute_nstep_value_targets` (the two-player sign flip lives
there). So this module is the seam between data collection and the BPTT step:
the driver only does ``store_episode`` → ``sample_training_batch`` → unroll.

``update_targets`` is the reanalyze hook: overwrite a stored step's MCTS
policy/value with a fresh search from a lagging network (the driver/agent drives
timing). Pure data refresh — no recompute here.
"""

from std.memory import alloc

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import mptr
from .nstep_targets import compute_nstep_value_targets
# Same quantize/dequantize helpers the GPU replays use: `SDT == DT` is a pure
# rebind; `uint8` stores `round(x·255)` and reads back `k/255` — bit-lossless
# for the `k/255` pixel pipeline (see data/gpu_replay.mojo).
from ..data.gpu_replay import _obs_quant, _obs_dequant


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


def _asdt[SDT: DType](n: Int) -> UnsafePointer[Scalar[SDT], MutAnyOrigin]:
    return mptr(alloc[Scalar[SDT]](n))


struct MCTSSequenceReplay[OBS: Int, ACT: Int, CAP: Int, OBS_STORE_DT: DType = DT](
    Movable, ImplicitlyDestructible
):
    """Ring of MCTS-labelled steps + episode index. ``CAP`` = max resident
    steps. Host-side (the CartPole lighthouse trains on CPU; the GPU search
    feeds it via host copies).

    ``OBS_STORE_DT`` (default ``DT`` — no behaviour change for vector-obs
    callers) selects the obs ring's storage dtype. ``DType.uint8`` is the
    pixel-obs capacity option: the ring stores ``round(x·255)`` and dequantizes
    ``k/255`` on read — bit-lossless for the arcade pixel pipeline (exact
    ``k/255`` grayscale) and 4× the resident steps of a float ring. Only the
    obs ring changes dtype; all other fields stay ``DT``."""

    comptime SDT = Self.OBS_STORE_DT

    var obs: UnsafePointer[Scalar[Self.SDT], MutAnyOrigin]   # [CAP, OBS] in SDT
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP] action index
    var rew: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP]
    var done: UnsafePointer[Scalar[DT], MutAnyOrigin]  # [CAP] terminal flag
    var pol: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP, ACT] visit policy
    var val: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [CAP] root value
    var tp: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [CAP] to_play
    var legal: UnsafePointer[Scalar[DT], MutAnyOrigin] # [CAP, ACT] legal mask (reanalyze)

    var ep_start: List[Int]   # absolute start step of each resident episode
    var ep_len: List[Int]
    var ep_trunc: List[Bool]  # time-limit truncated (last step NOT terminal)
    var total: Int            # monotonic steps written
    var rng: UInt64

    def __init__(out self, seed: UInt64 = 0):
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
        ep_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L, OBS]
        ep_act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L]
        ep_rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L]
        ep_pol: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L, ACT]
        ep_val: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [L]
        ep_tp: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [L]
        ep_legal: UnsafePointer[Scalar[DT], MutAnyOrigin], # [L, ACT] legal mask
        length: Int,
        truncated: Bool = False,
    ):
        """Append a finished episode of ``length`` steps. The terminal flag is
        set on the last step — unless ``truncated`` (time-limit cut, not a real
        terminal): then no step is flagged and the n-step targets bootstrap
        from the last stored root value instead of going to zero. Evicts
        episodes that fall out of the last ``CAP`` steps so every resident
        episode stays fully readable."""
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
            self.total += 1
        self.ep_start.append(start)
        self.ep_len.append(length)
        self.ep_trunc.append(truncated)
        # prune episodes no longer fully resident (start older than CAP window).
        var floor = self.total - Self.CAP
        while len(self.ep_start) > 0 and self.ep_start[0] < floor:
            _ = self.ep_start.pop(0)
            _ = self.ep_len.pop(0)
            _ = self.ep_trunc.pop(0)

    def _read_step[
        FIELD_DIM: Int,
    ](
        self,
        field: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ep_idx: Int,
        offset: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        out_base: Int,
    ):
        """Read ``FIELD_DIM`` cells of one ring field at episode-relative
        ``offset`` into ``out[out_base..]``. Past the episode end the read is
        absorbing: zeros (caller overlays obs-repeat / uniform-policy where
        those differ from zero)."""
        if offset >= self.ep_len[ep_idx]:
            for j in range(FIELD_DIM):
                out[out_base + j] = Scalar[DT](0.0)
            return
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for j in range(FIELD_DIM):
            out[out_base + j] = field[slot * FIELD_DIM + j]

    def _read_obs_step(
        self,
        ep_idx: Int,
        offset: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        out_base: Int,
    ):
        """Dequantized obs read (``Self.OBS`` cells) at episode-relative
        ``offset`` into ``out[out_base..]``. The obs ring is ``Self.SDT``-typed,
        so it cannot go through the generic ``_read_step[DT]``. Absorbing past
        terminal is zeros — matching ``_read_step`` (obs0 is always in-episode,
        so the absorbing branch is a no-op there; kept for symmetry)."""
        if offset >= self.ep_len[ep_idx]:
            for j in range(Self.OBS):
                out[out_base + j] = Scalar[DT](0.0)
            return
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for j in range(Self.OBS):
            out[out_base + j] = _obs_dequant[Self.SDT](
                self.obs[slot * Self.OBS + j]
            )

    def _sample_step_uniform(mut self) -> Tuple[Int, Int]:
        """Pick a resident (episode, offset) with every resident **step**
        equally likely. Episode-uniform sampling (episode first, then offset)
        over-weights steps from short episodes — early random-policy failures
        keep a majority of the sample mass long after the policy improves."""
        var tot = 0
        for e in range(len(self.ep_len)):
            tot += self.ep_len[e]
        var u = Int(self._xorshift() % UInt64(tot))
        var e = 0
        while u >= self.ep_len[e]:
            u -= self.ep_len[e]
            e += 1
        return (e, u)

    def sample_training_batch[
        B: Int, K: Int, N: Int,
    ](
        mut self,
        gamma: Scalar[DT],
        mut obs0: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B, OBS]
        mut actions: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [K, B]
        mut policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin], # [K+1, B, ACT]
        mut value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [K+1, B]
        mut reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin], # [K, B]
    ):
        """Fill the time-major unroll batch for ``B`` windows. Each row picks a
        step-uniform (episode, start); the K+N horizon is read with absorbing
        padding and the value targets are n-step bootstrapped (sign-flipped for
        two-player; truncated episodes bootstrap from their last root value).
        Caller guarantees ``num_episodes() > 0``."""
        comptime HV = K + N + 1   # value/to_play horizon (positions)
        comptime HR = K + N       # reward/done horizon (transitions)
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

            # obs0 = obs at start (always in-episode); dequant from the ring.
            self._read_obs_step(e, s, obs0, b * Self.OBS)

            # reward / done horizon (HR), with absorbing past terminal.
            for h in range(HR):
                self._read_step[1](self.rew, e, s + h, w_rew, h)
                if s + h >= L:
                    w_done[h] = Scalar[DT](1.0)   # absorbing = terminal
                else:
                    self._read_step[1](self.done, e, s + h, w_done, h)
            # value / to_play horizon (HV).
            for h in range(HV):
                if s + h >= L:
                    w_val[h] = Scalar[DT](0.0)     # terminal value 0
                    w_tp[h] = Scalar[DT](0.0)
                else:
                    self._read_step[1](self.val, e, s + h, w_val, h)
                    self._read_step[1](self.tp, e, s + h, w_tp, h)

            compute_nstep_value_targets[K, N](
                w_rew, w_done, w_val, w_tp, gamma, w_vt, last_valid=lv
            )

            # write time-major slabs.
            for k in range(K + 1):
                value_tgt[k * B + b] = w_vt[k]
                # policy target at position s+k (absorbing → uniform).
                var pbase = k * B * Self.ACT + b * Self.ACT
                if s + k >= L:
                    var u = Scalar[DT](1.0) / Scalar[DT](Self.ACT)
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = u
                else:
                    var slot = (self.ep_start[e] + s + k) % Self.CAP
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = self.pol[slot * Self.ACT + a]
            for k in range(K):
                # action at s+k (absorbing → 0); reward target = horizon reward.
                if s + k >= L:
                    actions[k * B + b] = Scalar[DT](0.0)
                else:
                    var slot = (self.ep_start[e] + s + k) % Self.CAP
                    actions[k * B + b] = self.act[slot]
                reward_tgt[k * B + b] = w_rew[k]

        w_rew.free(); w_done.free(); w_val.free(); w_tp.free(); w_vt.free()

    def sample_training_batch_seq[
        B: Int, K: Int, N: Int,
    ](
        mut self,
        gamma: Scalar[DT],
        mut obs_seq: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [K+1, B, OBS]
        mut actions: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [K, B]
        mut policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin], # [K+1, B, ACT]
        mut value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [K+1, B]
        mut reward_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin], # [K, B]
        cons_mask: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None,                                                # [K, B]
    ):
        """EZv2 variant of ``sample_training_batch``: identical targets, but the
        observation output is the **full time-major sequence** ``obs_seq[K+1, B,
        OBS]`` (``obs_seq[0]`` is the root obs) so the SimSiam consistency loss
        can encode the real future observations. Past an episode's terminal the
        obs read is **obs-repeat absorbing** (offset clamped to ``L−1``), matching
        the legacy consistency handling. ``cons_mask`` (if given) receives the
        ``[K, B]`` episode-boundary mask for the consistency loss — row ``(k-1,
        b)`` is 1 when ``obs_seq[k]`` is a real stored observation and 0 when it
        is the absorbing obs-repeat (the EZv2 reference zeroes those terms via
        ``mask_batch``). Caller guarantees ``num_episodes() > 0``.
        """
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
                    obs_seq[ob + j] = _obs_dequant[Self.SDT](
                        self.obs[slot * Self.OBS + j]
                    )
            # consistency boundary mask: step k = 1..K is real iff s+k < L.
            if cons_mask:
                var cm = cons_mask.value()
                for k in range(K):
                    cm[k * B + b] = (
                        Scalar[DT](1.0) if s + k + 1 < L else Scalar[DT](0.0)
                    )

            for h in range(HR):
                self._read_step[1](self.rew, e, s + h, w_rew, h)
                if s + h >= L:
                    w_done[h] = Scalar[DT](1.0)
                else:
                    self._read_step[1](self.done, e, s + h, w_done, h)
            for h in range(HV):
                if s + h >= L:
                    w_val[h] = Scalar[DT](0.0)
                    w_tp[h] = Scalar[DT](0.0)
                else:
                    self._read_step[1](self.val, e, s + h, w_val, h)
                    self._read_step[1](self.tp, e, s + h, w_tp, h)

            compute_nstep_value_targets[K, N](
                w_rew, w_done, w_val, w_tp, gamma, w_vt, last_valid=lv
            )

            for k in range(K + 1):
                value_tgt[k * B + b] = w_vt[k]
                var pbase = k * B * Self.ACT + b * Self.ACT
                if s + k >= L:
                    var u = Scalar[DT](1.0) / Scalar[DT](Self.ACT)
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = u
                else:
                    var slot = (self.ep_start[e] + s + k) % Self.CAP
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = self.pol[slot * Self.ACT + a]
            for k in range(K):
                if s + k >= L:
                    actions[k * B + b] = Scalar[DT](0.0)
                else:
                    var slot = (self.ep_start[e] + s + k) % Self.CAP
                    actions[k * B + b] = self.act[slot]
                reward_tgt[k * B + b] = w_rew[k]

        w_rew.free(); w_done.free(); w_val.free(); w_tp.free(); w_vt.free()

    def update_targets(
        mut self,
        ep_idx: Int,
        offset: Int,
        new_policy: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [ACT]
        new_value: Scalar[DT],
    ):
        """Reanalyze hook: overwrite a stored step's MCTS policy + root value
        with fresh search outputs from a lagging network. Timing is the
        driver's; this is pure in-place data refresh."""
        if ep_idx < 0 or ep_idx >= len(self.ep_start):
            return
        if offset < 0 or offset >= self.ep_len[ep_idx]:
            return
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for a in range(Self.ACT):
            self.pol[slot * Self.ACT + a] = new_policy[a]
        self.val[slot] = new_value

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
        """Copy the stored observation at ``(ep_idx, offset)`` into ``out``.
        MuZero reanalyze replans from the observation alone (no env state)."""
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for j in range(Self.OBS):
            out[j] = _obs_dequant[Self.SDT](self.obs[slot * Self.OBS + j])

    def read_legal(self, ep_idx: Int, offset: Int) -> List[Bool]:
        """The stored root legal-action mask at ``(ep_idx, offset)`` — needed so
        reanalyze masks the same illegal actions the original search did."""
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        var m = List[Bool](capacity=Self.ACT)
        for a in range(Self.ACT):
            m.append(self.legal[slot * Self.ACT + a] > Scalar[DT](0.5))
        return m^
