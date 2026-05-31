"""PPODiscreteAgent — facade over PPODiscreteTrainer + discrete driver.

On-policy categorical PPO. Single-env training entry point
(`train`) routes to `run_onpolicy_discrete_train`; `eval` routes to
`run_onpolicy_discrete_eval`. Mirrors the continuous `PPOAgent`
(see ppo/agent.mojo) but for `BoxDiscreteActionEnv`.

Usage:

    comptime ActorNet = Sequential[
        Linear[OBS, H], Tanh[H], Linear[H, H], Tanh[H], Linear[H, N_ACTIONS],
    ]
    comptime CriticNet = Sequential[
        Linear[OBS, H], Tanh[H], Linear[H, H], Tanh[H], Linear[H, 1],
    ]
    var agent = PPODiscreteAgent[
        "cpu", ActorNet, CriticNet, OBS, N_ACTIONS, ROLLOUT, MINIBATCH, EPOCHS,
    ](actor_lr=3e-4, critic_lr=1e-3, entropy_coef=0.01)
    var ep_returns = agent.train(env, total_timesteps=50_000)

For GPU: `PPODiscreteAgent["gpu", ...](ctx=ctx, ...)`.
"""

from std.gpu.host import DeviceContext

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import BoxDiscreteActionEnv

from ..training.driver_onpolicy_discrete import (
    run_onpolicy_discrete_train,
    run_onpolicy_discrete_eval,
)

from ..ppo.metrics import PPOMetrics
from .trainer import PPODiscreteTrainer


struct PPODiscreteAgent[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    N_ACTIONS: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int = 1,
](Movable & ImplicitlyDestructible):
    """Thin facade over `PPODiscreteTrainer` + the discrete on-policy
    driver."""

    comptime TrainerT = PPODiscreteTrainer[
        Self.train_target, Self.ACTOR, Self.CRITIC,
        Self.OBS_DIM, Self.N_ACTIONS, Self.ROLLOUT_LEN, Self.MINIBATCH,
        Self.N_EPOCHS, Self.N_ENVS,
    ]

    var trainer: Self.TrainerT

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 1e-3,
        gamma: Scalar[DT] = 0.99,
        gae_lambda: Scalar[DT] = 0.95,
        clip_eps: Scalar[DT] = 0.2,
        entropy_coef: Scalar[DT] = 0.01,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = 0.0,
        max_grad_norm: Scalar[DT] = 0.0,
    ) raises:
        """Construct a PPODiscreteAgent. Forwards every kwarg to
        `PPODiscreteTrainer.make`. `ctx` is required for
        `train_target='gpu'`."""
        self.trainer = Self.TrainerT.make(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
            ctx=ctx,
        )

    # ─── Training entry points ─────────────────────────────────────────

    def train[
        E: BoxDiscreteActionEnv,
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
        """Single-env discrete on-policy training via
        `run_onpolicy_discrete_train`. Covers `(env=cpu, train=cpu)`
        and `(env=cpu, train=gpu)`."""
        return run_onpolicy_discrete_train[Self.TrainerT, E, L](
            self.trainer,
            env,
            total_timesteps,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
        )

    # ─── Evaluation ─────────────────────────────────────────────────────

    def eval[
        E: BoxDiscreteActionEnv,
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        *,
        max_steps_per_episode: Int = 1_000,
        verbose: Bool = False,
    ) raises -> Scalar[DT]:
        """Greedy eval — argmax over logits, no sampling."""
        return run_onpolicy_discrete_eval[Self.TrainerT, E](
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
        step_idx: Int,
    ) raises -> Int:
        """Categorical sample from the softmax policy."""
        return self.trainer.select_action(obs, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Deterministic argmax over logits."""
        return self.trainer.select_greedy_action(obs)

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
    ) raises -> PPOMetrics:
        """Drain trainer accumulators into a PPOMetrics bundle."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        return self.trainer.flush_timer_log()

    # ─── Checkpointing ──────────────────────────────────────────────────

    def save(mut self, path: String) raises:
        """Passthrough to `trainer.save_state(path)` — actor, critic,
        their opts. Rollout buffer NOT included (on-policy resume
        re-rolls)."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. No target nets in PPO."""
        self.trainer.load_state(path)
