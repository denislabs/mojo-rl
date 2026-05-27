"""MBPOTrainer — J.1.g-redesign-v2 Step 4 — MBPO via ref-based blocks.

Pipeline (6 blocks):
  DualSample → TargetY → TwinCritic → SACActor → AlphaUpdate → Polyak
(5 of 6 blocks reused unchanged from SAC.)

Dynamics ensemble training + synthetic rollouts are NOT in the pipeline —
they're trainer methods invoked from train_step on a `model_train_freq`
cadence (block decomposition isn't the right fit for multi-epoch /
multi-step orchestration).

CPU only. Mirrors MBPOTrainerV2 surface (OffPolicyTrainable).
"""

from std.math import exp as fexp, sqrt as fsqrt, log as flog
from std.random import random_float64, randn_float64
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.online_target_pair import OnlineTargetPair
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..initializer import Xavier, Kaiming
from ..optimizer.adam import Adam
from ..optimizer.scalar_adam import ScalarAdam
from .dynamics_ensemble_block import DynamicsEnsembleBlock
from .episode_tracker import EpisodeTracker
from .trainer_block import TrainerState
from .driver_cpu import OffPolicyTrainable
from .blocks import (
    DualSampleCpuStep,
    TargetYStep,
    TwinCriticStep,
    SACActorStep,
    AlphaUpdateStep,
    PolyakStep,
)


struct MBPOTrainer[
    ACTOR: Module,
    CRITIC: Module,
    DynNet: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
    SYNTH_CAPACITY: Int,
    N_ENSEMBLE: Int,
    NUM_ELITES: Int,
    REAL_RATIO_PCT: Int = 5,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
](OffPolicyTrainable):
    comptime DYN_IN: Int = Self.OBS_DIM + Self.ACT_DIM
    comptime DYN_PRED: Int = 1 + Self.OBS_DIM
    comptime DYN_OUT: Int = 2 * Self.DYN_PRED
    comptime REAL_BS: Int = (Self.BATCH * Self.REAL_RATIO_PCT) // 100
    comptime SYNTH_BS: Int = Self.BATCH - Self.REAL_BS

    comptime ENSEMBLE = DynamicsEnsembleBlock[
        Self.DynNet,
        Self.N_ENSEMBLE,
        Self.NUM_ELITES,
        Self.DYN_IN,
        Self.DYN_OUT,
        Self.BATCH,
        Self.LOGVAR_MIN,
        Self.LOGVAR_MAX,
    ]

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM

    var actor: Self.ACTOR
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: DualSampleCpuStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.REPLAY_CAPACITY,
        Self.SYNTH_CAPACITY,
        Self.REAL_BS,
        Self.SYNTH_BS,
    ]
    var target_y_blk: TargetYStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.ACTOR,
        Self.CRITIC,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.CRITIC,
    ]
    var actor_blk: SACActorStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.ACTOR,
        Self.CRITIC,
    ]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var polyak_blk: PolyakStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.CRITIC,
    ]

    var ensemble: Self.ENSEMBLE
    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker

    # select_action scratches (mirror SACTrainer).
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _ao1: Scratch["ao1", 2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    # Dynamics training / rollout scratches.
    var _dyn_in: Scratch["dyn_in", Self.BATCH * Self.DYN_IN]
    var _dyn_tgt: Scratch["dyn_tgt", Self.BATCH * Self.DYN_PRED]
    var _ro_obs: Scratch["ro_obs", Self.BATCH * Self.OBS_DIM]
    var _ro_act: Scratch["ro_act", Self.BATCH * Self.ACT_DIM]
    var _ro_nxt: Scratch["ro_nxt", Self.BATCH * Self.OBS_DIM]
    var _ro_mu: Scratch["ro_mu", Self.BATCH * Self.DYN_PRED]
    var _ro_lv: Scratch["ro_lv", Self.BATCH * Self.DYN_PRED]

    var action_scale: Scalar[DT]
    var learning_starts: Int

    var model_train_freq: Int
    var dyn_epochs_per_round: Int
    var rollout_length: Int
    var num_rollouts_per_step: Int
    var sac_updates_per_step: Int
    var dyn_batch_size: Int
    var last_dyn_step: Int

    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _update_count: Int

    def __init__(out self):
        comptime assert (
            Self.DynNet.IN_DIMS[0] == Self.DYN_IN
        ), "MBPOTrainer: DynNet.IN_DIM must equal OBS_DIM + ACT_DIM"
        comptime assert (
            Self.DynNet.OUT_DIM == Self.DYN_OUT
        ), "MBPOTrainer: DynNet.OUT_DIM must equal 2 * (1 + OBS_DIM)"
        comptime assert (
            Self.REAL_RATIO_PCT >= 0 and Self.REAL_RATIO_PCT <= 100
        ), "REAL_RATIO_PCT must be in [0, 100]"
        comptime assert Self.REAL_BS >= 1, "REAL_BS must be >= 1"
        comptime assert Self.SYNTH_BS >= 1, "SYNTH_BS must be >= 1"

        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.alpha_opt = ScalarAdam(
            value=0.0,
            m=0.0,
            v=0.0,
            t=0,
            lr=0.0003,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
        )
        self.sample_blk = DualSampleCpuStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.REPLAY_CAPACITY,
            Self.SYNTH_CAPACITY,
            Self.REAL_BS,
            Self.SYNTH_BS,
        ]()
        self.target_y_blk = TargetYStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ]()
        self.actor_blk = SACActorStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.polyak_blk = PolyakStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ]()
        self.ensemble = Self.ENSEMBLE()
        self.state = TrainerState[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._ao1 = Scratch["ao1", 2 * Self.ACT_DIM, True]()
        self._alp1 = Scratch["alp1", Self.ACT_DIM + 1, True]()
        self._dyn_in = Scratch["dyn_in", Self.BATCH * Self.DYN_IN]()
        self._dyn_tgt = Scratch["dyn_tgt", Self.BATCH * Self.DYN_PRED]()
        self._ro_obs = Scratch["ro_obs", Self.BATCH * Self.OBS_DIM]()
        self._ro_act = Scratch["ro_act", Self.BATCH * Self.ACT_DIM]()
        self._ro_nxt = Scratch["ro_nxt", Self.BATCH * Self.OBS_DIM]()
        self._ro_mu = Scratch["ro_mu", Self.BATCH * Self.DYN_PRED]()
        self._ro_lv = Scratch["ro_lv", Self.BATCH * Self.DYN_PRED]()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self.model_train_freq = 250
        self.dyn_epochs_per_round = 4
        self.rollout_length = 1
        self.num_rollouts_per_step = 400
        self.sac_updates_per_step = 20
        self.dyn_batch_size = 256
        self.last_dyn_step = -1
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0

    @staticmethod
    def make[
        target: StaticString
    ](
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        model_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        model_train_freq: Int = 250,
        dyn_epochs_per_round: Int = 4,
        rollout_length: Int = 1,
        num_rollouts_per_step: Int = 400,
        sac_updates_per_step: Int = 20,
        dyn_batch_size: Int = 256,
    ) raises -> Self:
        comptime assert target == "cpu", "MBPOTrainer: CPU only"
        var t = Self()
        t.actor = Self.ACTOR.make[target="cpu", INIT=Xavier]()
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](t.actor)
        t.actor_opt.lr = actor_lr
        t.critic1_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair1.online)
        t.critic1_opt.lr = critic_lr
        t.critic2_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair2.online)
        t.critic2_opt.lr = critic_lr
        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        t.target_y_blk = TargetYStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make["cpu"](action_scale=action_scale, gamma=gamma)
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ].make["cpu"]()
        t.actor_blk = SACActorStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make["cpu"](action_scale=action_scale)
        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.polyak_blk = PolyakStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ].make(tau=tau)

        t.ensemble = Self.ENSEMBLE.make[target, INIT=Kaiming]()
        t.ensemble.set_lr(model_lr)
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )
        t.state = TrainerState[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make["cpu"]()

        init_scratch_auto[Self, target="cpu"](t)

        t.action_scale = action_scale
        t.learning_starts = learning_starts
        t.model_train_freq = model_train_freq
        t.dyn_epochs_per_round = dyn_epochs_per_round
        t.rollout_length = rollout_length
        t.num_rollouts_per_step = num_rollouts_per_step
        t.sac_updates_per_step = sac_updates_per_step
        t.dyn_batch_size = dyn_batch_size

        t.sample_blk.setup(learning_starts)
        return t^

    # ─── OffPolicyTrainable surface ───────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        if step_idx < self.learning_starts:
            for j in range(Self.ACT_DIM):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return
        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        var alp1_cpu_p = self._alp1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        var alp1_t = TileTensor(alp1_cpu_p, row_major[1, Self.ACT_DIM + 1]())
        self.actor_blk.inner.rsample.forward["cpu", 1](ao1_t, output=alp1_t)
        for j in range(Self.ACT_DIM):
            var a = alp1_cpu_p[j]
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        from std.math import tanh as ftanh

        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        for j in range(Self.ACT_DIM):
            var mean = ao1_cpu_p[j]
            var a = ftanh(mean) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self.sample_blk.real_add(obs, action, reward, next_obs, done)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        if step_idx < self.learning_starts:
            return False

        # Periodic dynamics + rollout phase (orchestration outside pipeline).
        var should_train_dyn = (
            self.last_dyn_step < 0
            or step_idx - self.last_dyn_step >= self.model_train_freq
        )
        if should_train_dyn:
            self._train_dynamics_ensemble()
            self._generate_synthetic_rollouts()
            self.last_dyn_step = step_idx

        # Need both buffers populated enough for the dual sample.
        if self.sample_blk.real_buf.size < Self.REAL_BS:
            return False
        if self.sample_blk.synth_buf.size < Self.SYNTH_BS:
            return False

        var any = False
        for _ in range(self.sac_updates_per_step):
            self.state.step_idx = step_idx
            self.state.did_step = True
            self.state.alpha = fexp(self.alpha_opt.value)

            self.sample_blk.step(self.state)
            if not self.state.did_step:
                continue

            self.target_y_blk.step["cpu"](
                self.state,
                self.actor,
                self.pair1.target_net,
                self.pair2.target_net,
            )
            self.twin_critic_blk.step["cpu"](
                self.state,
                self.pair1.online,
                self.critic1_opt,
                self.pair2.online,
                self.critic2_opt,
            )
            self.actor_blk.step["cpu"](
                self.state,
                self.actor,
                self.actor_opt,
                self.pair1.online,
                self.pair2.online,
            )
            self.alpha_blk.step(self.state, self.alpha_opt)
            self.polyak_blk.step["cpu"](
                self.state,
                self.pair1,
                self.pair2,
            )

            self._actor_L_accum += self.state.actor_loss
            self._critic_L_accum += self.state.critic_loss
            self._update_count += 1
            any = True
        return any

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Dynamics training + synthetic rollouts (orchestration) ──────

    def _train_dynamics_ensemble(mut self) raises:
        var n_data = self.sample_blk.real_buf.size
        if n_data < self.dyn_batch_size:
            return
        var bs = self.dyn_batch_size
        var steps_per_epoch = n_data // bs
        if steps_per_epoch < 1:
            steps_per_epoch = 1
        var total_steps = steps_per_epoch * self.dyn_epochs_per_round

        var dyn_in_p = self._dyn_in.cpu_ptr()
        var dyn_tgt_p = self._dyn_tgt.cpu_ptr()

        # Snapshot raw buffer pointers (stable; CPUReplay's circular
        # storage is fixed-size).
        var rb_obs = self.sample_blk.real_buf.obs
        var rb_act = self.sample_blk.real_buf.act
        var rb_rew = self.sample_blk.real_buf.rew
        var rb_nxt = self.sample_blk.real_buf.nxt

        for m in range(Self.N_ENSEMBLE):
            for _ in range(total_steps):
                for k in range(Self.BATCH):
                    var idx = Int(random_float64() * Float64(n_data))
                    if idx >= n_data:
                        idx = n_data - 1
                    for d in range(Self.OBS_DIM):
                        dyn_in_p[k * Self.DYN_IN + d] = rb_obs[
                            idx * Self.OBS_DIM + d
                        ]
                    for j in range(Self.ACT_DIM):
                        dyn_in_p[k * Self.DYN_IN + Self.OBS_DIM + j] = rb_act[
                            idx * Self.ACT_DIM + j
                        ]
                    dyn_tgt_p[k * Self.DYN_PRED + 0] = rb_rew[idx]
                    for d in range(Self.OBS_DIM):
                        dyn_tgt_p[k * Self.DYN_PRED + 1 + d] = (
                            rb_nxt[idx * Self.OBS_DIM + d]
                            - rb_obs[idx * Self.OBS_DIM + d]
                        )
                var dyn_in_t = TileTensor(
                    dyn_in_p, row_major[Self.BATCH, Self.DYN_IN]()
                )
                var dyn_tgt_t = TileTensor(
                    dyn_tgt_p, row_major[Self.BATCH, Self.DYN_PRED]()
                )
                _ = self.ensemble.train_member_step["cpu"](
                    m,
                    dyn_in_t,
                    dyn_tgt_t,
                )

    def _generate_synthetic_rollouts(mut self) raises:
        var real_buf_size = self.sample_blk.real_buf.size
        if real_buf_size < 1:
            return

        var rollouts_done = 0
        while rollouts_done < self.num_rollouts_per_step:
            var this_batch = Self.BATCH
            var remaining = self.num_rollouts_per_step - rollouts_done
            if remaining < this_batch:
                this_batch = remaining

            var roll_obs_p = self._ro_obs.cpu_ptr()
            var roll_act_p = self._ro_act.cpu_ptr()
            var roll_nxt_p = self._ro_nxt.cpu_ptr()
            var ro_mu_p = self._ro_mu.cpu_ptr()
            var ro_lv_p = self._ro_lv.cpu_ptr()
            var dyn_in_p = self._dyn_in.cpu_ptr()

            var rb_obs = self.sample_blk.real_buf.obs
            for k in range(this_batch):
                var idx = Int(random_float64() * Float64(real_buf_size))
                if idx >= real_buf_size:
                    idx = real_buf_size - 1
                for d in range(Self.OBS_DIM):
                    roll_obs_p[k * Self.OBS_DIM + d] = rb_obs[
                        idx * Self.OBS_DIM + d
                    ]

            for _ in range(self.rollout_length):
                for k in range(this_batch):
                    var obs_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                    for d in range(Self.OBS_DIM):
                        obs_list.append(roll_obs_p[k * Self.OBS_DIM + d])
                    var act_list = List[Scalar[DT]](capacity=Self.ACT_DIM)
                    for _ in range(Self.ACT_DIM):
                        act_list.append(Scalar[DT](0.0))
                    self.select_action(
                        obs_list,
                        act_list,
                        self.learning_starts + 1,
                    )
                    for j in range(Self.ACT_DIM):
                        roll_act_p[k * Self.ACT_DIM + j] = act_list[j]

                for k in range(this_batch):
                    for d in range(Self.OBS_DIM):
                        dyn_in_p[k * Self.DYN_IN + d] = roll_obs_p[
                            k * Self.OBS_DIM + d
                        ]
                    for j in range(Self.ACT_DIM):
                        dyn_in_p[
                            k * Self.DYN_IN + Self.OBS_DIM + j
                        ] = roll_act_p[k * Self.ACT_DIM + j]
                var dyn_in_t = TileTensor(
                    dyn_in_p, row_major[Self.BATCH, Self.DYN_IN]()
                )
                var ro_mu_t = TileTensor(
                    ro_mu_p, row_major[Self.BATCH, Self.DYN_PRED]()
                )
                var ro_lv_t = TileTensor(
                    ro_lv_p, row_major[Self.BATCH, Self.DYN_PRED]()
                )
                var n_elites = len(self.ensemble.elite_indices)
                var elite_pick = Int(random_float64() * Float64(n_elites))
                if elite_pick >= n_elites:
                    elite_pick = n_elites - 1
                var member_idx = self.ensemble.elite_indices[elite_pick]
                self.ensemble.predict_member["cpu"](
                    member_idx,
                    dyn_in_t,
                    ro_mu_t,
                    ro_lv_t,
                )

                var s_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                var a_list = List[Scalar[DT]](capacity=Self.ACT_DIM)
                var sp_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                for _ in range(Self.OBS_DIM):
                    s_list.append(Scalar[DT](0.0))
                    sp_list.append(Scalar[DT](0.0))
                for _ in range(Self.ACT_DIM):
                    a_list.append(Scalar[DT](0.0))
                for k in range(this_batch):
                    var mu_r = ro_mu_p[k * Self.DYN_PRED + 0]
                    var lv_r = ro_lv_p[k * Self.DYN_PRED + 0]
                    var std_r = fsqrt(fexp(lv_r))
                    var noise_r = Scalar[DT](randn_float64())
                    var rew = mu_r + std_r * noise_r
                    for d in range(Self.OBS_DIM):
                        s_list[d] = roll_obs_p[k * Self.OBS_DIM + d]
                        var mu_d = ro_mu_p[k * Self.DYN_PRED + 1 + d]
                        var lv_d = ro_lv_p[k * Self.DYN_PRED + 1 + d]
                        var std_d = fsqrt(fexp(lv_d))
                        var noise = Scalar[DT](randn_float64())
                        var delta = mu_d + std_d * noise
                        var nxt = roll_obs_p[k * Self.OBS_DIM + d] + delta
                        sp_list[d] = nxt
                        roll_nxt_p[k * Self.OBS_DIM + d] = nxt
                    for j in range(Self.ACT_DIM):
                        a_list[j] = roll_act_p[k * Self.ACT_DIM + j]
                    self.sample_blk.synth_add(
                        s_list,
                        a_list,
                        rew,
                        sp_list,
                        Scalar[DT](0.0),
                    )

                for k in range(this_batch * Self.OBS_DIM):
                    roll_obs_p[k] = roll_nxt_p[k]

            rollouts_done += this_batch
