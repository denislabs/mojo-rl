# +--------------------------------------------------------------------------+ #
# | StoreReplay — the data layer behind the legacy ReplayBuffer seam
# +--------------------------------------------------------------------------+ #
"""A `ReplayBuffer` conformer built on the Stage-3 index policies.

**No trainer changes are required.** `ReplaySampleStep[R, BATCH]` is already
generic over any `ReplayBuffer`, so migrating an algorithm is repointing one
comptime alias — not editing SAC/TD3/DQN/C51/DDPG.

**This is where the layer's central claim gets paid off.** `PRIORITIZED` is a
comptime *flag on one storage struct*, not a second struct: `StoreReplay`
replaces BOTH `CPUReplay` and `CPUPrioritizedReplay`, because the sum-tree
lives in a sampler and storage does not know it exists. Under the legacy
design PER was a storage subclass (`CPUPrioritizedReplay` wraps a `CPUReplay`
as `base`), which is exactly why {policy x backend x column set} multiplied
out into twelve buffers.

⚠ **The legacy `ReplayBuffer` trait hardcodes the column set** — `add` and
`sample_into` name obs/act/rew/next_obs/done in their signatures. So a
drop-in conformer cannot be column-general, and this file is fixed to the
six-pack on purpose. The column generality of `TrajectoryStore` /
`ResidentColumn` pays off for consumers that do NOT go through this trait
(BFM's qpos/qvel, PushT's pixels/proprio/state, Atari's ram) — it is not lost
here, it is simply not expressible at this seam.

⚠ Ring semantics, not append: an online replay buffer overwrites its oldest
row once full. `TrajectoryStore` is append-only offline storage; the two are
different objects and this is the online one.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.training.replay_buffer import ReplayBuffer
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from .sampler import PrioritizedSampler, UniformSampler


struct StoreReplay[
    OBS_: Int, ACT_: Int, CAP_: Int, PRIORITIZED: Bool = False
](ReplayBuffer):
    """Ring storage + a pluggable index policy.

    `PRIORITIZED=False` reproduces `CPUReplay`; `PRIORITIZED=True` reproduces
    `CPUPrioritizedReplay`. Both were gated bit-identical against those buffers
    before they were deleted; the surviving gate is
    `tests/data/test_replay_seam.mojo` (minibatches through the real
    `ReplaySampleStep`, ring wraparound, post-priority-update draw), over the
    index sequences pinned in `tests/data/test_sampler_golden.mojo`.
    """

    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var obs: List[Scalar[DT]]
    var act: List[Scalar[DT]]
    var rew: List[Scalar[DT]]
    var nxt: List[Scalar[DT]]
    var dne: List[Scalar[DT]]
    var size: Int
    var pos: Int

    var per: Optional[PrioritizedSampler]
    """Present only when `PRIORITIZED`. The sum-tree is a SAMPLER, not a
    storage subclass — that is the whole point."""

    var _pending_alpha: Scalar[DT]
    var _pending_beta: Scalar[DT]
    var _pending_epsilon: Scalar[DT]

    def __init__(out self):
        self.obs = List[Scalar[DT]]()
        self.act = List[Scalar[DT]]()
        self.rew = List[Scalar[DT]]()
        self.nxt = List[Scalar[DT]]()
        self.dne = List[Scalar[DT]]()
        self.size = 0
        self.pos = 0
        self.per = None
        self._pending_alpha = Scalar[DT](0.6)
        self._pending_beta = Scalar[DT](0.4)
        self._pending_epsilon = Scalar[DT](1e-6)

    def __init__(out self, *, deinit move: Self):
        self.obs = move.obs^
        self.act = move.act^
        self.rew = move.rew^
        self.nxt = move.nxt^
        self.dne = move.dne^
        self.size = move.size
        self.pos = move.pos
        self.per = move.per^
        self._pending_alpha = move._pending_alpha
        self._pending_beta = move._pending_beta
        self._pending_epsilon = move._pending_epsilon

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        var s = Self()
        s.obs = List[Scalar[DT]](
            length=Self.CAP * Self.OBS, fill=Scalar[DT](0)
        )
        s.act = List[Scalar[DT]](
            length=Self.CAP * Self.ACT, fill=Scalar[DT](0)
        )
        s.rew = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        s.nxt = List[Scalar[DT]](
            length=Self.CAP * Self.OBS, fill=Scalar[DT](0)
        )
        s.dne = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        comptime if Self.PRIORITIZED:
            s.per = PrioritizedSampler(
                Self.CAP,
                alpha=s._pending_alpha,
                beta=s._pending_beta,
                epsilon=s._pending_epsilon,
            )
        return s^

    def count(self) -> Int:
        return self.size

    # ── PER hooks ─────────────────────────────────────────────────────

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        self._pending_alpha = alpha
        self._pending_beta = beta
        self._pending_epsilon = epsilon
        comptime if Self.PRIORITIZED:
            if self.per:
                self.per = PrioritizedSampler(
                    Self.CAP, alpha=alpha, beta=beta, epsilon=epsilon
                )

    def set_beta(mut self, beta: Scalar[DT]):
        self._pending_beta = beta
        comptime if Self.PRIORITIZED:
            if self.per:
                self.per.value().set_beta(beta)

    # ── add ───────────────────────────────────────────────────────────

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var p = self.pos
        for i in range(Self.OBS):
            self.obs[p * Self.OBS + i] = s[i]
            self.nxt[p * Self.OBS + i] = sp[i]
        for j in range(Self.ACT):
            self.act[p * Self.ACT + j] = a[j]
        self.rew[p] = r
        self.dne[p] = d
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1
        comptime if Self.PRIORITIZED:
            # New rows enter at `max_priority^alpha`, matching the legacy
            # insert. The sampler owns that rule; storage just reports the row.
            self.per.value().note_added(p)

    # ── sample ────────────────────────────────────────────────────────

    def _gather_into[
        BATCH: Int
    ](
        self,
        ref idx: List[Scalar[DType.int32]],
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ):
        for k in range(BATCH):
            var row = Int(idx[k])
            for i in range(Self.OBS):
                state.mb_s.data[k * Self.OBS + i] = self.obs[
                    row * Self.OBS + i
                ]
                state.mb_sp.data[k * Self.OBS + i] = self.nxt[
                    row * Self.OBS + i
                ]
            for j in range(Self.ACT):
                state.mb_a.data[k * Self.ACT + j] = self.act[
                    row * Self.ACT + j
                ]
            state.mb_r.data[k] = self.rew[row]
            state.mb_d.data[k] = self.dne[row]

    def sample(
        mut self,
        n: Int,
        mut s_out: List[Scalar[DT]],
        mut a_out: List[Scalar[DT]],
        mut r_out: List[Scalar[DT]],
        mut sp_out: List[Scalar[DT]],
        mut d_out: List[Scalar[DT]],
        row_offset: Int = 0,
    ) raises:
        """Uniform draw into caller lists at `row_offset`.

        MBPO's `DualSampleStep` stacks a real and a synthetic partition into
        one minibatch by calling this twice with different offsets, so the raw
        form is needed alongside `sample_into`.
        """
        var sampler = UniformSampler(self.size)
        var batch = sampler.draw(n)
        for k in range(n):
            var row = Int(batch.host[k])
            var dst = row_offset + k
            for i in range(Self.OBS):
                s_out[dst * Self.OBS + i] = self.obs[row * Self.OBS + i]
                sp_out[dst * Self.OBS + i] = self.nxt[row * Self.OBS + i]
            for j in range(Self.ACT):
                a_out[dst * Self.ACT + j] = self.act[row * Self.ACT + j]
            r_out[dst] = self.rew[row]
            d_out[dst] = self.dne[row]

    def sample_into[
        BATCH: Int
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        comptime if Self.PRIORITIZED:
            var batch = self.per.value().draw(BATCH)
            self._gather_into[BATCH](batch.host, state)
            for i in range(BATCH):
                state.mb_w.data[i] = self.per.value().last_weights[i]
            state.has_per = True
        else:
            var sampler = UniformSampler(self.size)
            var batch = sampler.draw(BATCH)
            self._gather_into[BATCH](batch.host, state)

    def update_priorities[
        BATCH: Int
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        comptime if Self.PRIORITIZED:
            self.per.value().update_priorities(state.td_residuals.data)
