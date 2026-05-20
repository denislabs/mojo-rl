"""TargetYBlock — SAC target-y computation as a self-contained block. Phase 10F.

Encapsulates the off-policy target-value compute used by both critic
losses (and shared by SAC, TD3, and DDPG variants with minor tweaks).

Forward formula (CleanRL-style continuous SAC):
    a'           ~ squashed-Gaussian(actor(s'))      (no grad — fresh z)
    log_prob(a') = Σ log_N(z) - log_std - 0.5·log(2π)
                   - log(action_scale·(1 - tanh²) + ε)
    sa'          = concat(s', a')
    q1_tgt       = critic1_target.forward(sa')
    q2_tgt       = critic2_target.forward(sa')
    qmin         = min(q1_tgt, q2_tgt)
    y[b]         = r[b] + γ·nonterm·(qmin - α·log_prob)
        nonterm = 1.0 for time-limit-only envs (Pendulum); see
        `feedback_ppo_pendulum_timelimit_gae`.

Uses the **free-function** `squashed_gaussian_sample` rather than the
`RSample` Module — no gradient flow back through this path, so the
caching the Module would do is wasted. Same approach as the original
Phase 9B trainer.

Surface:
    TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        - `make[target](action_scale, gamma) raises -> Self`
        - `step[target](mut actor, mut critic1_target, mut critic2_target,
                        mb_sp_ptr, mb_r_ptr, alpha, mb_y_ptr) raises`
            Writes `mb_y_ptr` ([BATCH]) in-place.

CPU only (Phase 10F).
"""

from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module,
    TARGET_UNINIT,
    TARGET_CPU,
    target_tag_for,
)
from ..loss.sac_actor_loss import squashed_gaussian_sample
from ..random.box_muller import box_muller_normal
from .off_policy_critic import concat_sa


struct TargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](Movable & ImplicitlyDestructible):
    comptime SA_DIM = Self.OBS + Self.ACT

    var _mb_ao_sp: List[Scalar[DT]]    # [BATCH, 2*ACT]   actor(s') output
    var _mb_z_sp: List[Scalar[DT]]     # [BATCH, ACT]     z noise
    var _mb_act_sp: List[Scalar[DT]]   # [BATCH, ACT]     sampled a'
    var _mb_lp_sp: List[Scalar[DT]]    # [BATCH]          log_prob(a')
    var _mb_sa: List[Scalar[DT]]       # [BATCH, SA_DIM]  concat(s', a')
    var _mb_q1_tgt: List[Scalar[DT]]   # [BATCH, 1]
    var _mb_q2_tgt: List[Scalar[DT]]   # [BATCH, 1]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var _target_tag: Int8

    def __init__(out self):
        self._mb_ao_sp = List[Scalar[DT]]()
        self._mb_z_sp = List[Scalar[DT]]()
        self._mb_act_sp = List[Scalar[DT]]()
        self._mb_lp_sp = List[Scalar[DT]]()
        self._mb_sa = List[Scalar[DT]]()
        self._mb_q1_tgt = List[Scalar[DT]]()
        self._mb_q2_tgt = List[Scalar[DT]]()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self._target_tag = TARGET_UNINIT

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
    ) raises -> Self:
        comptime assert target == "cpu", (
            "TargetYBlock.make[target='gpu'] not yet implemented (Phase 10F CPU only)"
        )
        comptime assert Self.ACTOR.IN_DIM == Self.OBS, (
            "TargetYBlock: ACTOR.IN_DIM must equal OBS"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "TargetYBlock: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "TargetYBlock: CRITIC.IN_DIM must equal OBS + ACT"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "TargetYBlock: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        var zero: Scalar[DT] = 0.0
        blk._mb_ao_sp.resize(Self.BATCH * 2 * Self.ACT, zero)
        blk._mb_z_sp.resize(Self.BATCH * Self.ACT, zero)
        blk._mb_act_sp.resize(Self.BATCH * Self.ACT, zero)
        blk._mb_lp_sp.resize(Self.BATCH, zero)
        blk._mb_sa.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._mb_q1_tgt.resize(Self.BATCH, zero)
        blk._mb_q2_tgt.resize(Self.BATCH, zero)
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk._target_tag = TARGET_CPU
        return blk^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "TargetYBlock: method called with [target='"
                + String(target)
                + "'] but block was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

    def step[target: StaticString](
        mut self,
        mut actor: Self.ACTOR,
        mut critic1_target: Self.CRITIC,
        mut critic2_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Compute `mb_y[b] = r[b] + γ·(min(Q1_t, Q2_t)(s', a') − α·log_prob(a'|s'))`
        in-place into `mb_y_ptr`. `nonterm=1.0` for time-limit-only envs."""
        comptime assert target == "cpu", (
            "TargetYBlock.step: GPU path not yet implemented"
        )
        self._assert_tag[target]()

        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_ao_sp_p = self._mb_ao_sp.unsafe_ptr()
        var mb_ao_sp_t = TileTensor(
            mb_ao_sp_p, row_major[Self.BATCH, 2 * Self.ACT]()
        )
        actor.forward["cpu", Self.BATCH](mb_sp_t, mb_ao_sp_t)

        var mb_z_sp_p = self._mb_z_sp.unsafe_ptr()
        box_muller_normal(mb_z_sp_p, Self.BATCH * Self.ACT)
        var mb_z_sp_t = TileTensor(mb_z_sp_p, row_major[Self.BATCH, Self.ACT]())

        var mb_act_sp_p = self._mb_act_sp.unsafe_ptr()
        var mb_act_sp_t = TileTensor(
            mb_act_sp_p, row_major[Self.BATCH, Self.ACT]()
        )
        var mb_lp_sp_p = self._mb_lp_sp.unsafe_ptr()
        var mb_lp_sp_t = TileTensor(mb_lp_sp_p, row_major[Self.BATCH]())
        squashed_gaussian_sample[Self.ACT, Self.BATCH](
            mb_ao_sp_t, mb_z_sp_t, self.action_scale, mb_act_sp_t, mb_lp_sp_t
        )

        var mb_sa_p = self._mb_sa.unsafe_ptr()
        concat_sa[Self.OBS, Self.ACT, Self.BATCH](
            mb_sp_ptr, mb_act_sp_p, mb_sa_p
        )
        var mb_sa_t = TileTensor(mb_sa_p, row_major[Self.BATCH, Self.SA_DIM]())
        var mb_q1_tgt_p = self._mb_q1_tgt.unsafe_ptr()
        var mb_q2_tgt_p = self._mb_q2_tgt.unsafe_ptr()
        var mb_q1_tgt_t = TileTensor(mb_q1_tgt_p, row_major[Self.BATCH, 1]())
        var mb_q2_tgt_t = TileTensor(mb_q2_tgt_p, row_major[Self.BATCH, 1]())
        critic1_target.forward["cpu", Self.BATCH](mb_sa_t, mb_q1_tgt_t)
        critic2_target.forward["cpu", Self.BATCH](mb_sa_t, mb_q2_tgt_t)

        for b in range(Self.BATCH):
            var q1 = mb_q1_tgt_p[b]
            var q2 = mb_q2_tgt_p[b]
            var qmin = q1 if q1 < q2 else q2
            var nonterm: Scalar[DT] = 1.0
            mb_y_ptr[b] = mb_r_ptr[b] + self.gamma * nonterm * (
                qmin - alpha * mb_lp_sp_p[b]
            )
