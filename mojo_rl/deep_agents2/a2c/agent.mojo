"""A2C — Advantage Actor-Critic as the degenerate single-epoch PPO.

A2C is exactly PPO with **one epoch** over **one full-batch minibatch**
and **no clipping**: with `N_EPOCHS=1` and `MINIBATCH = ROLLOUT_LEN`
there is a single gradient step per rollout, and at that step the data
is still on-policy (the policy hasn't changed since collection), so the
importance ratio is identically 1. At ratio≡1 the PPO clipped surrogate
`-min(ratio·adv, clip(ratio)·adv)` collapses to `-ratio·adv`, whose
gradient is `-adv · ∇logπ(a|s)` — the vanilla advantage policy-gradient
A2C optimises. The clip bound is therefore irrelevant and never engages,
so no separate "disable clip" knob is needed.

These two facades are thin wrappers that pin `N_EPOCHS=1` and
`MINIBATCH=ROLLOUT_LEN` over the discrete / continuous PPO agents:

  * `A2CDiscreteAgent` — categorical softmax policy (CartPole etc.),
    over `PPODiscreteAgent`. Matches the legacy `A2CConfig`
    (GAE + softmax policy).
  * `A2CAgent` — diagonal-Gaussian continuous policy, over `PPOAgent`.

GAE (advantage estimation) is inherited unchanged — A2C with
`gae_lambda=1.0` recovers Monte-Carlo advantages, `<1.0` the usual
bias/variance trade-off. Default `entropy_coef=0.01` (A2C convention).
"""

from std.gpu.host import DeviceContext

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import BoxDiscreteActionEnv, BoxContinuousActionEnv

from ..ppo.agent import PPOAgent
from ..ppo.metrics import PPOMetrics
from ..ppo_discrete.agent import PPODiscreteAgent
from ..training.batched_env import BatchedEnv


# ──────────────────────────────────────────────────────────────────────
# A2CDiscreteAgent — categorical A2C over PPODiscreteAgent.
# ──────────────────────────────────────────────────────────────────────


struct A2CDiscreteAgent[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    N_ACTIONS: Int,
    ROLLOUT_LEN: Int,
](Movable & ImplicitlyDestructible):
    """Discrete A2C = PPODiscreteAgent[N_EPOCHS=1, MINIBATCH=ROLLOUT_LEN]."""

    comptime Inner = PPODiscreteAgent[
        Self.train_target,
        Self.ACTOR,
        Self.CRITIC,
        Self.OBS_DIM,
        Self.N_ACTIONS,
        Self.ROLLOUT_LEN,
        Self.ROLLOUT_LEN,  # MINIBATCH = full rollout
        1,  # N_EPOCHS = 1
    ]

    var inner: Self.Inner

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 7e-4,
        critic_lr: Scalar[DT] = 7e-4,
        gamma: Scalar[DT] = 0.99,
        gae_lambda: Scalar[DT] = 0.95,
        entropy_coef: Scalar[DT] = 0.01,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = 0.0,
        max_grad_norm: Scalar[DT] = 0.5,
    ) raises:
        """Construct a discrete A2C agent. `ctx` required for
        `train_target='gpu'`. `clip_eps` is intentionally absent — the
        single-epoch full-batch update never clips (see module docstring)."""
        self.inner = Self.Inner(
            ctx=ctx,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=Scalar[DT](0.2),  # inert at N_EPOCHS=1 (ratio≡1)
            entropy_coef=entropy_coef,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
        )

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
        return self.inner.train[E, L](
            env,
            total_timesteps,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
        )

    def eval[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        *,
        max_steps_per_episode: Int = 1_000,
        verbose: Bool = False,
    ) raises -> Scalar[DT]:
        return self.inner.eval[E](
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            verbose=verbose,
        )

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        step_idx: Int,
    ) raises -> Int:
        return self.inner.select_action(obs, step_idx)

    def select_greedy_action(mut self, ref obs: List[Scalar[DT]]) raises -> Int:
        return self.inner.select_greedy_action(obs)

    def mean_return(self) -> Scalar[DT]:
        return self.inner.mean_return()

    def ep_count(self) -> Int:
        return self.inner.ep_count()

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> PPOMetrics:
        return self.inner.flush_metrics[L](logger, step)

    def save(mut self, path: String) raises:
        self.inner.save(path)

    def load(mut self, path: String) raises:
        self.inner.load(path)


# ──────────────────────────────────────────────────────────────────────
# A2CAgent — continuous (diagonal-Gaussian) A2C over PPOAgent.
# ──────────────────────────────────────────────────────────────────────


struct A2CAgent[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    ROLLOUT_LEN: Int,
    N_ENVS: Int = 1,
](Movable & ImplicitlyDestructible):
    """Continuous A2C = PPOAgent[N_EPOCHS=1, MINIBATCH=ROLLOUT_LEN·N_ENVS]."""

    comptime Inner = PPOAgent[
        Self.train_target,
        Self.ACTOR,
        Self.CRITIC,
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.ROLLOUT_LEN,
        Self.ROLLOUT_LEN * Self.N_ENVS,  # MINIBATCH = full rollout
        1,  # N_EPOCHS = 1
        Self.N_ENVS,
    ]

    var inner: Self.Inner

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 7e-4,
        critic_lr: Scalar[DT] = 7e-4,
        gamma: Scalar[DT] = 0.99,
        gae_lambda: Scalar[DT] = 0.95,
        entropy_coef: Scalar[DT] = 0.0,
        action_scale: Scalar[DT] = 1.0,
        log_std_init: Scalar[DT] = -0.5,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1600.0,
        max_grad_norm: Scalar[DT] = 0.5,
    ) raises:
        self.inner = Self.Inner(
            ctx=ctx,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=Scalar[DT](0.2),  # inert at N_EPOCHS=1 (ratio≡1)
            entropy_coef=entropy_coef,
            action_scale=action_scale,
            log_std_init=log_std_init,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
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
        return self.inner.train_single[E, L](
            env,
            total_timesteps,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
        )

    def train[
        E: BatchedEnv,
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        rng_seed: UInt64 = 42,
        print_every: Int = 5_000,
        verbose: Bool = True,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        checkpoint_path: String = "",
        checkpoint_every: Int = 0,
    ) raises -> List[Scalar[DT]]:
        return self.inner.train[E, L](
            env,
            total_timesteps,
            rng_seed=rng_seed,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
        )

    # def eval[
    #     E: BoxContinuousActionEnv
    # ](
    #     mut self,
    #     mut env: E,
    #     num_episodes: Int = 10,
    #     *,
    #     max_steps_per_episode: Int = 1_000,
    #     verbose: Bool = False,
    # ) raises -> Scalar[DT]:
    #     return self.inner.eval[E](
    #         env,
    #         num_episodes,
    #         max_steps_per_episode=max_steps_per_episode,
    #         verbose=verbose,
    #     )

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        self.inner.select_action(obs, action_out, step_idx)

    def mean_return(self) -> Scalar[DT]:
        return self.inner.mean_return()

    def ep_count(self) -> Int:
        return self.inner.ep_count()

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> PPOMetrics:
        return self.inner.flush_metrics[L](logger, step)

    # The underlying PPO trainer is reachable via `agent.inner.trainer`
    # for log_std_init poking (mirrors the PPOAgent examples).

    def save(mut self, path: String) raises:
        self.inner.save(path)

    def load(mut self, path: String) raises:
        self.inner.load(path)
