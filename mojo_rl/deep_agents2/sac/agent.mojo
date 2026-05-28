"""SACAgent — user-facing facade over SACTrainer + off-policy drivers.

The trainer owns the algorithm (parameters, optimizers, replay buffer,
training step pipeline); the off-policy drivers own the env loop. This
agent blends the two so users write:

    var agent = SACAgent["cpu", SAMPLE, ACTOR, CRITIC].make(actor_lr=...)
    var ep_returns = agent.train(env, total_timesteps=30_000)
    var mean_eval = agent.eval(env, num_episodes=10)

instead of wiring `SACTrainer.make(...)` + free-function
`run_offpolicy_train_batched[type, type, N_ENVS](trainer, env, ...)`
by hand.

The wrapped `SACTrainer` remains accessible as `agent.trainer` for
power users who want to compose custom training loops (e.g. multi-seed
sweeps, save/resume integration, manual gradient inspection).

Two training entry points:

  * `train(env, ...)`     — `BatchedEnv` (CPU+CPU or GPU+GPU at any
                            N_ENVS ≥ 1). Routes to
                            `run_offpolicy_train_batched`.
  * `train_single(env, ...)` — `BoxContinuousActionEnv` (single-env,
                               typically used for cross-target CPU env
                               + GPU trainer). Routes to
                               `run_offpolicy_train`.

Eval is a single method (`eval`) bound on `BoxContinuousActionEnv`,
matching the existing `run_offpolicy_eval`.

Bit-identity: this agent is a pure facade — it forwards every call
unchanged. `pendulum_sac_nn2_driver.mojo` continues to produce
`mean_ret(10) = -169.04118` at 30k steps seed=42.
"""

from std.gpu.host import DeviceContext

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.checkpoint import save_state_v2, load_state_v2
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import BoxContinuousActionEnv

from ..training.blocks import SampleBlock
from ..training.batched_env import BatchedEnv
from ..training.driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_train_batched,
    run_offpolicy_eval,
)
from ..core.checkpoint_helpers import (
    save_optimizer_v2,
    load_optimizer_v2,
    save_scalar_adam_v2,
    load_scalar_adam_v2,
)

from .metrics import SACMetrics
from .trainer import SACTrainer


struct SACAgent[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](Movable & ImplicitlyDestructible):
    """Thin facade over `SACTrainer` + off-policy drivers.

    Comptime parameters mirror `SACTrainer` one-for-one. Dimensions
    (OBS / ACT / BATCH) are derived from `SAMPLE` so they appear in
    one place at instantiation.
    """

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    var trainer: SACTrainer[
        Self.train_target,
        Self.SAMPLE,
        Self.ACTOR,
        Self.CRITIC,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 1e-3,
        alpha_lr: Scalar[DT] = 3e-4,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        action_scale: Scalar[DT] = 1.0,
        init_alpha: Scalar[DT] = 0.2,
        target_entropy: Scalar[DT] = -1.0,
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1250.0,
        max_grad_norm: Scalar[DT] = 0.0,
        per_alpha: Scalar[DT] = 0.6,
        per_beta: Scalar[DT] = 0.4,
        per_epsilon: Scalar[DT] = 1e-6,
        use_bf16: Bool = False,
        use_ere: Bool = False,
        ere_eta: Scalar[DT] = 0.996,
        ere_c_min: Int = 1,
        ere_k_max: Int = 1000,
    ) raises:
        """Construct a SACAgent. Forwards every kwarg to `SACTrainer.make`."""

        self.trainer = SACTrainer[
            Self.train_target,
            Self.SAMPLE,
            Self.ACTOR,
            Self.CRITIC,
        ].make(
            ctx=ctx,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            gamma=gamma,
            tau=tau,
            action_scale=action_scale,
            init_alpha=init_alpha,
            target_entropy=target_entropy,
            learning_starts=learning_starts,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_epsilon=per_epsilon,
            use_bf16=use_bf16,
            use_ere=use_ere,
            ere_eta=ere_eta,
            ere_c_min=ere_c_min,
            ere_k_max=ere_k_max,
        )

    # ─── Training entry points ─────────────────────────────────────────

    def train[
        E: BatchedEnv,
        N_ENVS: Int = 1,
        NS: Int = 1,
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        rng_seed: UInt64 = 42,
        updates_per_step: Int = 1,
        print_every: Int = 5_000,
        verbose: Bool = True,
        nstep_gamma: Scalar[DT] = 0.99,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    ) raises -> List[Scalar[DT]]:
        """Off-policy training via `run_offpolicy_train_batched`.

        Covers same-target (cpu+cpu, gpu+gpu) at any `N_ENVS >= 1`. Pass
        `N_ENVS` as a comptime method param when wrapping multiple envs,
        e.g. `agent.train[N_ENVS=8](env, total_timesteps=200_000)`.

        Pass `logger=Optional[UnsafePointer[L, MutAnyOrigin]](
        UnsafePointer(to=my_logger))` to stream `env/mean_ret` and
        `env/ep_count` at `print_every` cadence. Default `L=NoOpLogger`
        comptime-elides the emit path entirely (bit-identical no-op)."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train_batched[
            SACTrainer[
                Self.train_target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
            ],
            E,
            N_ENVS,
            NS,
            L,
        ](
            ctx,
            self.trainer,
            env,
            total_timesteps,
            rng_seed=rng_seed,
            updates_per_step=updates_per_step,
            print_every=print_every,
            verbose=verbose,
            nstep_gamma=nstep_gamma,
            logger=logger,
        )

    def train_single[
        E: BoxContinuousActionEnv,
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        print_every: Int = 1_000,
        verbose: Bool = True,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    ) raises -> List[Scalar[DT]]:
        """Single-env off-policy training via `run_offpolicy_train`.
        Covers `(env=cpu, train=cpu)` and `(env=cpu, train=gpu)` cross-
        target. For batched same-target, use `train()` instead."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train[
            SACTrainer[
                Self.train_target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
            ],
            E,
            L,
        ](
            self.trainer,
            env,
            total_timesteps,
            ctx=ctx,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
        )

    # ─── Evaluation ─────────────────────────────────────────────────────

    def eval[
        E: BoxContinuousActionEnv,
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        *,
        max_steps_per_episode: Int = 1_000,
        verbose: Bool = False,
    ) raises -> Scalar[DT]:
        """Greedy eval via `run_offpolicy_eval`. Returns mean episode
        return across `num_episodes`. Does not touch the replay buffer,
        optimizers, or episode tracker."""
        return run_offpolicy_eval[
            SACTrainer[
                Self.train_target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
            ],
            E,
        ](
            self.trainer,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            verbose=verbose,
        )

    # ─── Single-step inference (host-list interface) ───────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Stochastic action sample. `step_idx` decides warmup vs policy
        path inside the trainer."""
        self.trainer.select_action(obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic action — uses the actor's mean (no Gaussian
        sample). Use this at inference time after training."""
        self.trainer.select_greedy_action(obs, action_out)

    # ─── Episode-tracker accessors ─────────────────────────────────────

    def mean_return(self) -> Scalar[DT]:
        """Mean episodic return over the last `window_size` episodes."""
        return self.trainer.mean_return()

    def ep_count(self) -> Int:
        """Total completed episodes since training began."""
        return self.trainer.ep_count()

    # ─── Metrics / logging passthrough ─────────────────────────────────

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> SACMetrics:
        """Drain trainer accumulators into a SACMetrics bundle. Pass an
        UnsafePointer-to-logger to also emit one `log_scalar` call per
        metric field. Resets accumulators on every call."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        """Per-section wall-time report (resets accumulators)."""
        return self.trainer.flush_timer_log()

    # ─── Checkpointing (CPU only) ──────────────────────────────────────

    def save(mut self, path: String) raises:
        """Persist networks + optimizer state to `path/` (a directory
        that must already exist). Replay buffer and episode tracker are
        NOT included — resume starts with a fresh replay.

        CPU-only: GPU agents store params in DeviceBuffers; this writes
        the stale CPU mirror. A GPU sync helper will land in a follow-up."""
        comptime if Self.train_target != "cpu":
            raise Error(
                "SACAgent.save: GPU save/load not yet supported. Train on"
                " CPU or wait for the device-sync helper."
            )
        save_state_v2(self.trainer.actor, path + "/actor.ckpt")
        save_state_v2(self.trainer.pair1.online, path + "/critic1.ckpt")
        save_state_v2(self.trainer.pair2.online, path + "/critic2.ckpt")
        save_optimizer_v2(self.trainer.actor_opt, path + "/actor_opt.ckpt")
        save_optimizer_v2(self.trainer.critic1_opt, path + "/critic1_opt.ckpt")
        save_optimizer_v2(self.trainer.critic2_opt, path + "/critic2_opt.ckpt")
        save_scalar_adam_v2(self.trainer.alpha_opt, path + "/alpha_opt.ckpt")

    def load(mut self, path: String) raises:
        """Restore networks + optimizer state from a directory previously
        written by `save()`. Target critics are hard-copied from their
        online twins."""
        comptime if Self.train_target != "cpu":
            raise Error(
                "SACAgent.load: GPU save/load not yet supported. Train on"
                " CPU or wait for the device-sync helper."
            )
        load_state_v2(self.trainer.actor, path + "/actor.ckpt")
        load_state_v2(self.trainer.pair1.online, path + "/critic1.ckpt")
        load_state_v2(self.trainer.pair2.online, path + "/critic2.ckpt")
        hard_copy_params[Self.train_target, M=Self.CRITIC](
            self.trainer.pair1.online, self.trainer.pair1.target_net,
            self.trainer.ctx,
        )
        hard_copy_params[Self.train_target, M=Self.CRITIC](
            self.trainer.pair2.online, self.trainer.pair2.target_net,
            self.trainer.ctx,
        )
        load_optimizer_v2(self.trainer.actor_opt, path + "/actor_opt.ckpt")
        load_optimizer_v2(self.trainer.critic1_opt, path + "/critic1_opt.ckpt")
        load_optimizer_v2(self.trainer.critic2_opt, path + "/critic2_opt.ckpt")
        load_scalar_adam_v2(self.trainer.alpha_opt, path + "/alpha_opt.ckpt")
