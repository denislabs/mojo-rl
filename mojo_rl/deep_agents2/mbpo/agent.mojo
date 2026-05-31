"""MBPOAgent — user-facing facade over MBPOTrainer + off-policy drivers.

Model-Based Policy Optimization: SAC actor/critic trained against a
mixture of real and synthetic transitions, where the synthetic ones come
from a learned dynamics ensemble (`DynNet`).

CPU + GPU — `train_target` is the first comptime param (Phase 4.3).

Usage:

    var agent = MBPOAgent[
        "cpu", ACTOR, CRITIC, DynNet,
        OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENSEMBLE, NUM_ELITES,
    ](actor_lr=3e-4, model_lr=1e-3)

    var ep_returns = agent.train(env, total_timesteps=30_000)

For GPU: `MBPOAgent["gpu", ...](ctx=ctx, ...)`.
"""

from std.gpu.host import DeviceContext

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import BoxContinuousActionEnv

from ..training.batched_env import BatchedEnv
from ..training.driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_train_batched,
    run_offpolicy_eval,
)

from .metrics import MBPOMetrics
from .trainer import MBPOTrainer


struct MBPOAgent[
    train_target: StaticString,
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
](Movable & ImplicitlyDestructible):
    """Thin facade over `MBPOTrainer` + off-policy drivers."""

    comptime TrainerT = MBPOTrainer[
        Self.train_target, Self.ACTOR, Self.CRITIC, Self.DynNet,
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        Self.REPLAY_CAPACITY, Self.SYNTH_CAPACITY,
        Self.N_ENSEMBLE, Self.NUM_ELITES,
        Self.REAL_RATIO_PCT, Self.LOGVAR_MIN, Self.LOGVAR_MAX,
    ]

    var trainer: Self.TrainerT

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 3e-4,
        alpha_lr: Scalar[DT] = 3e-4,
        model_lr: Scalar[DT] = 1e-3,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        action_scale: Scalar[DT] = 1.0,
        init_alpha: Scalar[DT] = 0.2,
        target_entropy: Scalar[DT] = -1.0,
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1250.0,
        model_train_freq: Int = 250,
        dyn_epochs_per_round: Int = 4,
        rollout_length: Int = 1,
        num_rollouts_per_step: Int = 400,
        sac_updates_per_step: Int = 20,
        dyn_batch_size: Int = 256,
    ) raises:
        """Construct an MBPOAgent. Forwards every kwarg to `MBPOTrainer.make`.
        `ctx` is required for `train_target='gpu'`."""
        self.trainer = Self.TrainerT.make(
            ctx=ctx,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            model_lr=model_lr,
            gamma=gamma,
            tau=tau,
            action_scale=action_scale,
            init_alpha=init_alpha,
            target_entropy=target_entropy,
            learning_starts=learning_starts,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            model_train_freq=model_train_freq,
            dyn_epochs_per_round=dyn_epochs_per_round,
            rollout_length=rollout_length,
            num_rollouts_per_step=num_rollouts_per_step,
            sac_updates_per_step=sac_updates_per_step,
            dyn_batch_size=dyn_batch_size,
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

        Note: `updates_per_step` is the per-env-step driver knob; MBPO's
        internal `sac_updates_per_step` controls how many SAC mini-updates
        run inside one train_step against the synthetic buffer."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train_batched[Self.TrainerT, E, N_ENVS, NS, L](
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
        diag_every: Int = 0,
        checkpoint_path: String = "",
        checkpoint_every: Int = 0,
    ) raises -> List[Scalar[DT]]:
        """Single-env off-policy training via `run_offpolicy_train`."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train[Self.TrainerT, E, L](
            self.trainer,
            env,
            total_timesteps,
            ctx=ctx,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
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
        """Greedy eval — actor mean action, no sampling."""
        return run_offpolicy_eval[Self.TrainerT, E](
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
        self.trainer.select_action(obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.trainer.select_greedy_action(obs, action_out)

    # ─── Episode-tracker accessors ─────────────────────────────────────

    def mean_return(self) -> Scalar[DT]:
        return self.trainer.mean_return()

    def ep_count(self) -> Int:
        return self.trainer.ep_count()

    # ─── Metrics / logging passthrough ─────────────────────────────────

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> MBPOMetrics:
        """Drain trainer accumulators into an MBPOMetrics bundle."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        return self.trainer.flush_timer_log()

    # ─── Checkpointing ──────────────────────────────────────────────────
    # Dynamics ensemble nets / dyn optimizers are NOT included; resume
    # restarts dynamics from scratch (re-trains in the first
    # `model_train_freq` env steps).

    def save(mut self, path: String) raises:
        """Passthrough to `trainer.save_state(path)` — actor, critic1/2,
        their opts, alpha_opt. Dynamics ensemble + replay NOT included."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. Target critics hard-copied from online."""
        self.trainer.load_state(path)
