"""DDPGTargetYBlock — deterministic target-value compute.

Phase 2 Track B migration: 3 raw UnsafePointers + manual alloc/free
replaced with 3 `Scratch[NAME, SIZE]` fields + `init_scratch_auto`. The
`__del__` deallocator disappears (Scratch owns its CPU list).

Formula:
    a'    = actor_target(s')          (deterministic)
    sa'   = concat(s', a')
    q'    = critic_target(sa')
    y[b]  = r[b] + γ·nonterm·q'[b]

CPU only (mirrors SAC's `TargetYBlock` scope). `nonterm = 1.0` for
Pendulum-style truncation envs; see `feedback_ppo_pendulum_timelimit_gae`.

Sibling of `TargetYBlock` (SAC) — DDPG/TD3-specific shape (no
log_prob/min reduction). TD3 uses a different `TD3TargetYBlock` because
of clipped-noise action smoothing + twin-critic min.
"""

from layout import TileTensor, row_major

from ..constants import DT
from ..core.module import Module
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..core.target_storage import TargetStorage, assert_tag_for
from ..loss.loss_block import LossBlock
from .off_policy_critic import concat_sa


struct DDPGTargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT

    var _mb_a_sp: Scratch["mb_a_sp", Self.BATCH * Self.ACT]
    var _mb_sa: Scratch["mb_sa", Self.BATCH * Self.SA_DIM]
    var _mb_q: Scratch["mb_q", Self.BATCH]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.ACTOR.OUT_DIM == Self.ACT, (
            "DDPGTargetYBlock: ACTOR.OUT_DIM must equal ACT"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "DDPGTargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "DDPGTargetYBlock: CRITIC.OUT_DIM must equal 1"
        )
        self._mb_a_sp = Scratch["mb_a_sp", Self.BATCH * Self.ACT]()
        self._mb_sa = Scratch["mb_sa", Self.BATCH * Self.SA_DIM]()
        self._mb_q = Scratch["mb_q", Self.BATCH]()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
    ) raises -> Self:
        comptime assert target == "cpu", "DDPGTargetYBlock: CPU only"
        var b = Self()
        b.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](b)
        b.action_scale = action_scale
        b.gamma = gamma
        return b^

    def step[
        target: StaticString,
    ](
        mut self,
        mut actor_target: Self.ACTOR,
        mut critic_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Writes mb_y_ptr[BATCH] in-place. CPU only."""
        comptime assert target == "cpu", "DDPGTargetYBlock: CPU only"
        assert_tag_for["DDPGTargetYBlock", target](self.ts.target_tag)

        var a_sp_p = self._mb_a_sp.cpu_ptr()
        var sa_p = self._mb_sa.cpu_ptr()
        var q_p = self._mb_q.cpu_ptr()

        # a' = actor_target(s'), clamped element-wise to [-action_scale, action_scale]
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_a_sp_t = TileTensor(a_sp_p, row_major[Self.BATCH, Self.ACT]())
        actor_target.forward[target, Self.BATCH](mb_sp_t, mb_a_sp_t)
        for k in range(Self.BATCH * Self.ACT):
            var v = a_sp_p[k]
            if v > self.action_scale:
                v = self.action_scale
            elif v < -self.action_scale:
                v = -self.action_scale
            a_sp_p[k] = v

        # sa' = concat(s', a')
        concat_sa[Self.OBS, Self.ACT, Self.BATCH](
            mb_sp_ptr, a_sp_p, sa_p,
        )
        var mb_sa_t = TileTensor(sa_p, row_major[Self.BATCH, Self.SA_DIM]())
        var mb_q_t = TileTensor(q_p, row_major[Self.BATCH, 1]())
        critic_target.forward[target, Self.BATCH](mb_sa_t, mb_q_t)

        # y = r + γ·nonterm·q. nonterm = 1 for Pendulum-style truncation.
        for b in range(Self.BATCH):
            mb_y_ptr[b] = mb_r_ptr[b] + self.gamma * q_p[b]
