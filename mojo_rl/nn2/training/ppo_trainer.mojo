"""PPOTrainer — on-policy actor-critic via PPOActorLossCG FullGraph.

Phase I.2.e. CleanRL-style continuous-action PPO:

  - Two nets: actor (Linear/Tanh/GaussianHead → 2*ACT) + critic
    (→ scalar V).
  - Two Adam optimisers (distinct lr defaults: actor 3e-4, critic 1e-3).
  - PPOActorLossCG (the FullGraph form) for the actor loss.
  - MSELoss for the critic.
  - Internal rollout buffers (obs/action/old_log_prob/value/reward/
    done/terminated), filled by `record_transition`, drained by
    `train_step` once `ROLLOUT_LEN` env-steps have accumulated.

Conforms to `OnPolicyTrainable`. The driver loop in `driver_cpu.mojo`
treats PPOTrainer identically to off-policy trainers: one `train_step`
per env-step; the trainer returns False for the (ROLLOUT_LEN − 1) idle
steps and True on the boundary step where the K-epoch update fires.

Action plumbing: `select_action` writes the *env-ready* (action_scale
× clamped) action and internally caches the *unbounded* sample +
log_prob + value for the upcoming `record_transition` to push into the
rollout buffer. Action clamping is shared with the existing bespoke
example (`pendulum_ppo_nn2.mojo`) — clamp at ±action_scale, sample is
the un-clamped Gaussian draw used for the log_prob computation in the
update.

CPU only for Phase I.2.
"""

from std.math import exp as fexp, log as flog
from std.memory import alloc

from layout import TileTensor, row_major

from ..constants import DT
from ..combinators.sequential import Sequential
from ..core import Module, Optimizer
from ..core.target_storage import TargetStorage
from ..loss.mse import MSELoss
from ..loss.ppo_actor_loss_cg import PPOActorLossCG
from ..optimizer.adam import Adam
from ..initializer import Xavier
from ..primitives.gaussian_head import GaussianHead
from ..random.box_muller import box_muller_normal
from .driver_cpu import OnPolicyTrainable
from .episode_tracker import EpisodeTracker
from .gae import compute_gae, normalize_in_place


comptime LOG_2PI: Scalar[DT] = 1.8378770664093453
comptime EPS_STD: Scalar[DT] = 1e-6
comptime LOG_STD_MIN_F: Scalar[DT] = -5.0
comptime LOG_STD_MAX_F: Scalar[DT] = 2.0


def _clamp_log_std(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < LOG_STD_MIN_F:
        return LOG_STD_MIN_F
    elif ls > LOG_STD_MAX_F:
        return LOG_STD_MAX_F
    return ls


def _gaussian_log_prob_sum(
    action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mu_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ls_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act_dim: Int,
) -> Scalar[DT]:
    var s: Scalar[DT] = 0.0
    for j in range(act_dim):
        var ls = _clamp_log_std(ls_ptr[j])
        var std = fexp(ls)
        var z = (action_ptr[j] - mu_ptr[j]) / (std + EPS_STD)
        s += Scalar[DT](-0.5) * (
            LOG_2PI + Scalar[DT](2.0) * ls + z * z
        )
    return s


def _shuffle_indices(
    indices: UnsafePointer[Int32, MutAnyOrigin], n: Int,
):
    """Fisher-Yates on Int32 indices. Reuses the example's RNG path."""
    from std.random import random_float64
    for t in range(n - 1, 0, -1):
        var j = Int(random_float64() * Float64(t + 1))
        if j > t:
            j = t
        var tmp = indices[t]
        indices[t] = indices[j]
        indices[j] = tmp


struct PPOTrainer[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
](OnPolicyTrainable):
    """CleanRL-style PPO continuous trainer composed from nn2 blocks."""

    comptime N_MINIBATCHES = Self.ROLLOUT_LEN // Self.MINIBATCH
    comptime AUX_DIM = Self.ACT_DIM + 2

    # Networks + optimisers + loss blocks.
    var actor: Self.ACTOR
    var critic: Self.CRITIC
    var actor_opt: Adam
    var critic_opt: Adam
    var ppo_loss: PPOActorLossCG[Self.ACTOR, Self.MINIBATCH]
    var mse_loss: MSELoss[1]

    # Hyperparameters.
    var gamma: Scalar[DT]
    var gae_lambda: Scalar[DT]
    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]
    var action_scale: Scalar[DT]

    # Rollout buffers (full rollout — caller never touches).
    var obs_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var act_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var olp_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var rew_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var val_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var done_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var term_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var adv_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ret_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var indices: UnsafePointer[Int32, MutAnyOrigin]

    # Bootstrap obs (the obs after the last rollout step → critic forward
    # for V(s_T)).
    var bootstrap_obs: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # Per-step scratch (BATCH=1 actor/critic forward).
    var ob1: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ao1: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var v1: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var z_scratch: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # Per-step rollout cache (filled by select_action, consumed by
    # record_transition).
    var cached_action: UnsafePointer[Scalar[DT], MutAnyOrigin]  # unbounded
    var cached_log_prob: Scalar[DT]
    var cached_value: Scalar[DT]

    # Minibatch scratch (BATCH=MINIBATCH update).
    var mb_obs: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mb_aux: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [action | olp | adv]
    var mb_adv: UnsafePointer[Scalar[DT], MutAnyOrigin]   # for per-mb normalize
    var mb_ret: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mb_v: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mb_gv: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mb_gi: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # Rollout cursor — `train_step` triggers the update when this hits
    # ROLLOUT_LEN.
    var rollout_idx: Int

    # Episode return tracker.
    var tracker: EpisodeTracker

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.ACTOR.IN_DIM == Self.OBS_DIM
        ), "PPOTrainer: ACTOR.IN_DIM must equal OBS_DIM"
        comptime assert (
            Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM
        ), "PPOTrainer: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        comptime assert (
            Self.CRITIC.IN_DIM == Self.OBS_DIM
        ), "PPOTrainer: CRITIC.IN_DIM must equal OBS_DIM"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "PPOTrainer: CRITIC.OUT_DIM must equal 1"
        comptime assert (
            Self.ROLLOUT_LEN % Self.MINIBATCH == 0
        ), "PPOTrainer: ROLLOUT_LEN must be divisible by MINIBATCH"

        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        var null_pi = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.actor = Self.ACTOR()
        self.critic = Self.CRITIC()
        self.actor_opt = Adam()
        self.critic_opt = Adam()
        self.ppo_loss = PPOActorLossCG[Self.ACTOR, Self.MINIBATCH]()
        self.mse_loss = MSELoss[1]()

        self.gamma = Scalar[DT](0.99)
        self.gae_lambda = Scalar[DT](0.95)
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)
        self.action_scale = Scalar[DT](1.0)

        self.obs_buf = null_p
        self.act_buf = null_p
        self.olp_buf = null_p
        self.rew_buf = null_p
        self.val_buf = null_p
        self.done_buf = null_p
        self.term_buf = null_p
        self.adv_buf = null_p
        self.ret_buf = null_p
        self.indices = null_pi
        self.bootstrap_obs = null_p
        self.ob1 = null_p
        self.ao1 = null_p
        self.v1 = null_p
        self.z_scratch = null_p
        self.cached_action = null_p
        self.cached_log_prob = Scalar[DT](0.0)
        self.cached_value = Scalar[DT](0.0)
        self.mb_obs = null_p
        self.mb_aux = null_p
        self.mb_adv = null_p
        self.mb_ret = null_p
        self.mb_v = null_p
        self.mb_gv = null_p
        self.mb_gi = null_p
        self.rollout_idx = 0
        self.tracker = EpisodeTracker.new(
            window_size=10, initial_fill=Scalar[DT](-1600.0)
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        gae_lambda: Scalar[DT] = Scalar[DT](0.95),
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        log_std_init: Scalar[DT] = Scalar[DT](-0.5),
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1600.0),
    ) raises -> Self:
        comptime assert target == "cpu", (
            "PPOTrainer.make[target='gpu'] not implemented (Phase I.2 CPU only)"
        )
        var t = Self()
        t.actor = Self.ACTOR.make[target="cpu", INIT=Xavier]()
        t.critic = Self.CRITIC.make[target="cpu", INIT=Xavier]()
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](t.actor)
        t.actor_opt.lr = actor_lr
        t.critic_opt = Adam.make[target="cpu", M=Self.CRITIC](t.critic)
        t.critic_opt.lr = critic_lr
        t.ppo_loss = PPOActorLossCG[Self.ACTOR, Self.MINIBATCH].make["cpu"](
            clip_eps=clip_eps, entropy_coef=entropy_coef,
        )
        t.mse_loss = MSELoss[1].make["cpu"]()

        t.gamma = gamma
        t.gae_lambda = gae_lambda
        t.clip_eps = clip_eps
        t.entropy_coef = entropy_coef
        t.action_scale = action_scale

        # log_std initialisation: the CleanRL recipe sets all log_std
        # entries to `log_std_init`. Reflecting through arbitrary actor
        # topologies isn't supported by Mojo nightly's trait-typed
        # comptime params, so the caller is responsible (the actor field
        # is publicly accessible — see example for the idiom). The
        # `log_std_init` arg here is kept for forward-compat / documentation.
        _ = log_std_init

        # Heap allocs (MutAnyOrigin required for the Module variadic API).
        t.obs_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN * Self.OBS_DIM)
        t.act_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN * Self.ACT_DIM)
        t.olp_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN)
        t.rew_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN)
        t.val_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN)
        t.done_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN)
        t.term_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN)
        t.adv_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN)
        t.ret_buf = alloc[Scalar[DT]](Self.ROLLOUT_LEN)
        t.indices = alloc[Int32](Self.ROLLOUT_LEN)
        t.bootstrap_obs = alloc[Scalar[DT]](Self.OBS_DIM)
        # Pendulum has no real terminal — caller fills term_buf at
        # record_transition time (0.0 for truncation, 1.0 for real
        # terminal). We zero-init for safety.
        for k in range(Self.ROLLOUT_LEN):
            t.term_buf[k] = Scalar[DT](0.0)

        t.ob1 = alloc[Scalar[DT]](Self.OBS_DIM)
        t.ao1 = alloc[Scalar[DT]](2 * Self.ACT_DIM)
        t.v1 = alloc[Scalar[DT]](1)
        t.z_scratch = alloc[Scalar[DT]](Self.ACT_DIM)
        t.cached_action = alloc[Scalar[DT]](Self.ACT_DIM)

        t.mb_obs = alloc[Scalar[DT]](Self.MINIBATCH * Self.OBS_DIM)
        t.mb_aux = alloc[Scalar[DT]](Self.MINIBATCH * Self.AUX_DIM)
        t.mb_adv = alloc[Scalar[DT]](Self.MINIBATCH)
        t.mb_ret = alloc[Scalar[DT]](Self.MINIBATCH * 1)
        t.mb_v = alloc[Scalar[DT]](Self.MINIBATCH * 1)
        t.mb_gv = alloc[Scalar[DT]](Self.MINIBATCH * 1)
        t.mb_gi = alloc[Scalar[DT]](Self.MINIBATCH * Self.OBS_DIM)

        t.rollout_idx = 0
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )
        t.ts = TargetStorage.make_cpu()
        return t^

    # ──────────────────────────────────────────────────────────────────
    # Public action surface. Non-parametric wrappers come BEFORE the
    # parametric forms — mirrors SAC's method ordering, which is the
    # one Mojo's trait-conformance check accepts.
    # ──────────────────────────────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Non-parametric wrapper — Mojo trait conformance shim. See
        SAC's docstring for the rationale."""
        self.select_action["cpu"](obs, action_out, step_idx)

    def select_action[target: StaticString = "cpu"](
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Sample one PPO action. Writes env-ready (clamped, scaled)
        action into `action_out`; caches the unbounded sample, log_prob,
        and V(s) internally for the next `record_transition` call."""
        comptime assert target == "cpu", "PPOTrainer CPU only"
        for d in range(Self.OBS_DIM):
            self.ob1[d] = obs[d]
        var ob1_t = TileTensor(self.ob1, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(self.ao1, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)

        box_muller_normal(self.z_scratch, Self.ACT_DIM)
        var lp_total: Scalar[DT] = 0.0
        for j in range(Self.ACT_DIM):
            var mu = self.ao1[j]
            var ls = _clamp_log_std(self.ao1[Self.ACT_DIM + j])
            var sample = mu + fexp(ls) * self.z_scratch[j]
            self.cached_action[j] = sample
            # Env-ready: clamp to ±action_scale.
            var env_a = sample
            if env_a > self.action_scale:
                env_a = self.action_scale
            elif env_a < -self.action_scale:
                env_a = -self.action_scale
            action_out[j] = env_a
            # Log-prob is on the unbounded sample (no tanh squash).
            var z = (sample - mu) / (fexp(ls) + EPS_STD)
            lp_total += Scalar[DT](-0.5) * (
                LOG_2PI + Scalar[DT](2.0) * ls + z * z
            )
        self.cached_log_prob = lp_total

        var v1_t = TileTensor(self.v1, row_major[1, 1]())
        self.critic.forward["cpu", 1](ob1_t, output=v1_t)
        self.cached_value = self.v1[0]

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.select_greedy_action["cpu"](obs, action_out)

    def select_greedy_action[target: StaticString = "cpu"](
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic action for eval — uses µ directly (no sampling)."""
        comptime assert target == "cpu", "PPOTrainer CPU only"
        for d in range(Self.OBS_DIM):
            self.ob1[d] = obs[d]
        var ob1_t = TileTensor(self.ob1, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(self.ao1, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        for j in range(Self.ACT_DIM):
            var env_a = self.ao1[j]
            if env_a > self.action_scale:
                env_a = self.action_scale
            elif env_a < -self.action_scale:
                env_a = -self.action_scale
            action_out[j] = env_a

    # ──────────────────────────────────────────────────────────────────
    # Rollout / update orchestration.
    # ──────────────────────────────────────────────────────────────────

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        """Push the most-recent (obs, action, reward, done) into the
        rollout buffer using the cached unbounded action + log_prob +
        value from `select_action`.

        `action` here is the *env-ready* action (the clamped/scaled
        thing the driver fed to env.step) — IGNORED; we use the cached
        unbounded sample for PPO math. `done` is treated as truncation
        by default (Pendulum-style). Real terminals require callers to
        set `term_buf[t]` explicitly via `mark_terminal()`.
        """
        var t = self.rollout_idx
        if t >= Self.ROLLOUT_LEN:
            # Update hasn't fired yet — the train_step() at the boundary
            # will reset rollout_idx to 0. This branch indicates the
            # driver called record_transition more than ROLLOUT_LEN
            # times between train_step boundaries; the trainer relies
            # on the standard "one record per train_step" cadence.
            return
        for d in range(Self.OBS_DIM):
            self.obs_buf[t * Self.OBS_DIM + d] = obs[d]
        for j in range(Self.ACT_DIM):
            self.act_buf[t * Self.ACT_DIM + j] = self.cached_action[j]
        self.olp_buf[t] = self.cached_log_prob
        self.val_buf[t] = self.cached_value
        self.rew_buf[t] = reward
        self.done_buf[t] = done
        # term_buf stays at 0.0 unless caller explicitly marks terminal.
        self.tracker.add_reward(reward)
        # Cache next_obs in bootstrap_obs every step — at rollout end
        # it'll already hold the right value.
        for d in range(Self.OBS_DIM):
            self.bootstrap_obs[d] = next_obs[d]
        self.rollout_idx += 1

    def mark_terminal(mut self):
        """Mark the last-recorded transition as a real terminal (V=0
        bootstrap). Default behaviour is truncation (Pendulum-style)."""
        if self.rollout_idx > 0:
            self.term_buf[self.rollout_idx - 1] = Scalar[DT](1.0)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """Non-parametric wrapper — Mojo trait conformance shim."""
        return self.train_step["cpu"](step_idx)

    def train_step[target: StaticString = "cpu"](
        mut self, step_idx: Int,
    ) raises -> Bool:
        """Returns True iff a full K-epoch update fired this step
        (i.e. rollout_idx hit ROLLOUT_LEN)."""
        comptime assert target == "cpu", "PPOTrainer CPU only"
        if self.rollout_idx < Self.ROLLOUT_LEN:
            return False

        # ── Bootstrap V(s_T) from the obs after the rollout end.
        var ob1_t = TileTensor(self.bootstrap_obs, row_major[1, Self.OBS_DIM]())
        var v1_t = TileTensor(self.v1, row_major[1, 1]())
        self.critic.forward["cpu", 1](ob1_t, output=v1_t)
        var next_value = self.v1[0]

        # ── GAE.
        compute_gae(
            Self.ROLLOUT_LEN, self.rew_buf, self.val_buf, self.term_buf,
            next_value, self.gamma, self.gae_lambda,
            self.adv_buf, self.ret_buf,
        )

        # ── K-epoch minibatch SGD.
        for k in range(Self.ROLLOUT_LEN):
            self.indices[k] = Int32(k)

        for _epoch in range(Self.N_EPOCHS):
            _shuffle_indices(self.indices, Self.ROLLOUT_LEN)
            for mb in range(Self.N_MINIBATCHES):
                # Gather minibatch into mb_obs / mb_aux / mb_ret / mb_adv.
                for k in range(Self.MINIBATCH):
                    var src = Int(self.indices[mb * Self.MINIBATCH + k])
                    for d in range(Self.OBS_DIM):
                        self.mb_obs[k * Self.OBS_DIM + d] = (
                            self.obs_buf[src * Self.OBS_DIM + d]
                        )
                    for j in range(Self.ACT_DIM):
                        self.mb_aux[k * Self.AUX_DIM + j] = (
                            self.act_buf[src * Self.ACT_DIM + j]
                        )
                    self.mb_aux[k * Self.AUX_DIM + Self.ACT_DIM] = (
                        self.olp_buf[src]
                    )
                    self.mb_adv[k] = self.adv_buf[src]
                    self.mb_ret[k] = self.ret_buf[src]
                # Per-minibatch advantage normalisation (CleanRL style).
                normalize_in_place(Self.MINIBATCH, self.mb_adv)
                # Splat normalized advantage into the aux slot.
                for k in range(Self.MINIBATCH):
                    self.mb_aux[k * Self.AUX_DIM + Self.ACT_DIM + 1] = (
                        self.mb_adv[k]
                    )

                # ── Actor update (FullGraph).
                _ = self.ppo_loss.forward_backward[
                    target="cpu", OPT=Adam,
                ](self.actor, self.actor_opt, self.mb_obs, self.mb_aux)

                # ── Critic update (vanilla MSE).
                var mb_obs_t = TileTensor(
                    self.mb_obs, row_major[Self.MINIBATCH, Self.OBS_DIM](),
                )
                var mb_v_t = TileTensor(
                    self.mb_v, row_major[Self.MINIBATCH, 1](),
                )
                var mb_gv_t = TileTensor(
                    self.mb_gv, row_major[Self.MINIBATCH, 1](),
                )
                var mb_gi_t = TileTensor(
                    self.mb_gi, row_major[Self.MINIBATCH, Self.OBS_DIM](),
                )
                var mb_ret_t = TileTensor(
                    self.mb_ret, row_major[Self.MINIBATCH, 1](),
                )

                self.critic.forward["cpu", Self.MINIBATCH](
                    mb_obs_t, output=mb_v_t,
                )
                _ = self.mse_loss.forward["cpu", Self.MINIBATCH](
                    mb_v_t, mb_ret_t
                )
                self.mse_loss.vjp["cpu", Self.MINIBATCH](mb_ret_t, mb_gv_t)
                self.critic_opt.zero_grad["cpu", M=Self.CRITIC](self.critic)
                self.critic.vjp["cpu", Self.MINIBATCH](mb_gv_t, mb_gi_t)
                self.critic_opt.step["cpu", M=Self.CRITIC](self.critic)

        # Reset the rollout cursor + the term buffer.
        self.rollout_idx = 0
        for k in range(Self.ROLLOUT_LEN):
            self.term_buf[k] = Scalar[DT](0.0)
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count


