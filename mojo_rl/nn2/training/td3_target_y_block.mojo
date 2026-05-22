"""TD3TargetYBlock — twin-critic target-y with clipped noise (Block E-4).

TD3 target-policy smoothing (Fujimoto et al., 2018):
    a'    = clamp(actor_target(s') + clamp(ε, -c, c), -action_scale, action_scale)
            with ε ~ N(0, σ_target^2)
    sa'   = concat(s', a')
    qmin  = min(critic1_target(sa'), critic2_target(sa'))
    y[b]  = r[b] + γ·nonterm·qmin

Differences vs DDPG target-y:
  - Clipped Gaussian noise added to target action (smoothing → reduces
    overestimation from sharp critic peaks).
  - Min over twin target critics (the SAC trick, also used here to fight
    overestimation).

Differences vs SAC target-y:
  - No α·log_prob term (deterministic policy).
  - Noise is clipped (not unclipped squashed-Gaussian).

CPU only.
"""

from std.memory import alloc
from layout import TileTensor, row_major

from ..constants import DT
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for
from ..loss.loss_block import LossBlock
from ..random.box_muller import box_muller_normal
from .off_policy_critic import concat_sa


struct TD3TargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT

    var _mb_a_sp: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [BATCH, ACT]
    var _mb_noise: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [BATCH * ACT]
    var _mb_sa: UnsafePointer[Scalar[DT], MutAnyOrigin]      # [BATCH, SA_DIM]
    var _mb_q1: UnsafePointer[Scalar[DT], MutAnyOrigin]      # [BATCH, 1]
    var _mb_q2: UnsafePointer[Scalar[DT], MutAnyOrigin]      # [BATCH, 1]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var noise_std: Scalar[DT]    # σ for target-policy smoothing
    var noise_clip: Scalar[DT]   # c — noise clamped to ±c·action_scale
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.ACTOR.OUT_DIM == Self.ACT, (
            "TD3TargetYBlock: ACTOR.OUT_DIM must equal ACT"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "TD3TargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "TD3TargetYBlock: CRITIC.OUT_DIM must equal 1"
        )
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)
        self._mb_a_sp = null_p
        self._mb_noise = null_p
        self._mb_sa = null_p
        self._mb_q1 = null_p
        self._mb_q2 = null_p
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.noise_std = Scalar[DT](0.2)
        self.noise_clip = Scalar[DT](0.5)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        noise_std: Scalar[DT] = Scalar[DT](0.2),
        noise_clip: Scalar[DT] = Scalar[DT](0.5),
    ) raises -> Self:
        comptime assert target == "cpu", "TD3TargetYBlock: CPU only"
        var b = Self()
        b._mb_a_sp = alloc[Scalar[DT]](Self.BATCH * Self.ACT)
        b._mb_noise = alloc[Scalar[DT]](Self.BATCH * Self.ACT)
        b._mb_sa = alloc[Scalar[DT]](Self.BATCH * Self.SA_DIM)
        b._mb_q1 = alloc[Scalar[DT]](Self.BATCH)
        b._mb_q2 = alloc[Scalar[DT]](Self.BATCH)
        b.action_scale = action_scale
        b.gamma = gamma
        b.noise_std = noise_std
        b.noise_clip = noise_clip
        b.ts = TargetStorage.make_cpu()
        return b^

    def __del__(deinit self):
        if Int(self._mb_a_sp) != 0:
            self._mb_a_sp.free()
        if Int(self._mb_noise) != 0:
            self._mb_noise.free()
        if Int(self._mb_sa) != 0:
            self._mb_sa.free()
        if Int(self._mb_q1) != 0:
            self._mb_q1.free()
        if Int(self._mb_q2) != 0:
            self._mb_q2.free()

    def step[
        target: StaticString,
    ](
        mut self,
        mut actor_target: Self.ACTOR,
        mut critic1_target: Self.CRITIC,
        mut critic2_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime assert target == "cpu", "TD3TargetYBlock: CPU only"
        assert_tag_for["TD3TargetYBlock", target](self.ts.target_tag)

        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_a_sp_t = TileTensor(self._mb_a_sp, row_major[Self.BATCH, Self.ACT]())
        actor_target.forward[target, Self.BATCH](mb_sp_t, mb_a_sp_t)

        # Sample noise + clamp + add + clamp action to ±action_scale.
        box_muller_normal(self._mb_noise, Self.BATCH * Self.ACT)
        var sigma = self.noise_std * self.action_scale
        var clip_lim = self.noise_clip * self.action_scale
        for k in range(Self.BATCH * Self.ACT):
            var n = self._mb_noise[k] * sigma
            if n > clip_lim:
                n = clip_lim
            elif n < -clip_lim:
                n = -clip_lim
            var v = self._mb_a_sp[k] + n
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
        var mb_q1_t = TileTensor(self._mb_q1, row_major[Self.BATCH, 1]())
        var mb_q2_t = TileTensor(self._mb_q2, row_major[Self.BATCH, 1]())
        critic1_target.forward[target, Self.BATCH](mb_sa_t, mb_q1_t)
        critic2_target.forward[target, Self.BATCH](mb_sa_t, mb_q2_t)

        # y = r + γ·min(q1, q2). nonterm = 1 (truncation env).
        for b in range(Self.BATCH):
            var qmin = self._mb_q1[b] if self._mb_q1[b] < self._mb_q2[b] else self._mb_q2[b]
            mb_y_ptr[b] = mb_r_ptr[b] + self.gamma * qmin
