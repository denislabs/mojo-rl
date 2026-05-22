"""DDPGTargetYBlock — deterministic target-value compute (Block E-4).

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

from std.memory import alloc
from layout import TileTensor, row_major

from ..constants import DT
from ..core.module import Module
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

    var _mb_a_sp: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [BATCH, ACT]
    var _mb_sa: UnsafePointer[Scalar[DT], MutAnyOrigin]     # [BATCH, SA_DIM]
    var _mb_q: UnsafePointer[Scalar[DT], MutAnyOrigin]      # [BATCH, 1]

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
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)
        self._mb_a_sp = null_p
        self._mb_sa = null_p
        self._mb_q = null_p
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
        b._mb_a_sp = alloc[Scalar[DT]](Self.BATCH * Self.ACT)
        b._mb_sa = alloc[Scalar[DT]](Self.BATCH * Self.SA_DIM)
        b._mb_q = alloc[Scalar[DT]](Self.BATCH)
        b.action_scale = action_scale
        b.gamma = gamma
        b.ts = TargetStorage.make_cpu()
        return b^

    def __del__(deinit self):
        if Int(self._mb_a_sp) != 0:
            self._mb_a_sp.free()
        if Int(self._mb_sa) != 0:
            self._mb_sa.free()
        if Int(self._mb_q) != 0:
            self._mb_q.free()

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

        # a' = actor_target(s'), clamped element-wise to [-action_scale, action_scale]
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_a_sp_t = TileTensor(self._mb_a_sp, row_major[Self.BATCH, Self.ACT]())
        actor_target.forward[target, Self.BATCH](mb_sp_t, mb_a_sp_t)
        for k in range(Self.BATCH * Self.ACT):
            var v = self._mb_a_sp[k]
            if v > self.action_scale:
                v = self.action_scale
            elif v < -self.action_scale:
                v = -self.action_scale
            self._mb_a_sp[k] = v

        # sa' = concat(s', a')
        concat_sa[Self.OBS, Self.ACT, Self.BATCH](
            mb_sp_ptr, self._mb_a_sp, self._mb_sa,
        )
        var mb_sa_t = TileTensor(self._mb_sa, row_major[Self.BATCH, Self.SA_DIM]())
        var mb_q_t = TileTensor(self._mb_q, row_major[Self.BATCH, 1]())
        critic_target.forward[target, Self.BATCH](mb_sa_t, mb_q_t)

        # y = r + γ·nonterm·q. nonterm = 1 for Pendulum-style truncation.
        for b in range(Self.BATCH):
            mb_y_ptr[b] = mb_r_ptr[b] + self.gamma * self._mb_q[b]
