"""TD3TargetYBlock — twin-critic target-y with clipped noise.

Phase 2 Track B migration: 5 raw UnsafePointers + manual alloc/free
replaced with 5 `Scratch[NAME, SIZE]` fields + `init_scratch_auto`. The
`__del__` deallocator disappears (Scratch owns its CPU lists).

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

from layout import TileTensor, row_major

from ..constants import DT
from ..core.module import Module
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
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

    var _mb_a_sp: Scratch["mb_a_sp", Self.BATCH * Self.ACT]
    var _mb_noise: Scratch["mb_noise", Self.BATCH * Self.ACT]
    var _mb_sa: Scratch["mb_sa", Self.BATCH * Self.SA_DIM]
    var _mb_q1: Scratch["mb_q1", Self.BATCH]
    var _mb_q2: Scratch["mb_q2", Self.BATCH]

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
        self._mb_a_sp = Scratch["mb_a_sp", Self.BATCH * Self.ACT]()
        self._mb_noise = Scratch["mb_noise", Self.BATCH * Self.ACT]()
        self._mb_sa = Scratch["mb_sa", Self.BATCH * Self.SA_DIM]()
        self._mb_q1 = Scratch["mb_q1", Self.BATCH]()
        self._mb_q2 = Scratch["mb_q2", Self.BATCH]()
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
        b.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](b)
        b.action_scale = action_scale
        b.gamma = gamma
        b.noise_std = noise_std
        b.noise_clip = noise_clip
        return b^

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

        var a_sp_p = self._mb_a_sp.cpu_ptr()
        var noise_p = self._mb_noise.cpu_ptr()
        var sa_p = self._mb_sa.cpu_ptr()
        var q1_p = self._mb_q1.cpu_ptr()
        var q2_p = self._mb_q2.cpu_ptr()

        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_a_sp_t = TileTensor(a_sp_p, row_major[Self.BATCH, Self.ACT]())
        actor_target.forward[target, Self.BATCH](mb_sp_t, mb_a_sp_t)

        # Sample noise + clamp + add + clamp action to ±action_scale.
        box_muller_normal(noise_p, Self.BATCH * Self.ACT)
        var sigma = self.noise_std * self.action_scale
        var clip_lim = self.noise_clip * self.action_scale
        for k in range(Self.BATCH * Self.ACT):
            var n = noise_p[k] * sigma
            if n > clip_lim:
                n = clip_lim
            elif n < -clip_lim:
                n = -clip_lim
            var v = a_sp_p[k] + n
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
        var mb_q1_t = TileTensor(q1_p, row_major[Self.BATCH, 1]())
        var mb_q2_t = TileTensor(q2_p, row_major[Self.BATCH, 1]())
        critic1_target.forward[target, Self.BATCH](mb_sa_t, mb_q1_t)
        critic2_target.forward[target, Self.BATCH](mb_sa_t, mb_q2_t)

        # y = r + γ·min(q1, q2). nonterm = 1 (truncation env).
        for b in range(Self.BATCH):
            var qmin = q1_p[b] if q1_p[b] < q2_p[b] else q2_p[b]
            mb_y_ptr[b] = mb_r_ptr[b] + self.gamma * qmin
