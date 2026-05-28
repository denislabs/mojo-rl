"""TD3Agent — user-facing facade over TD3Trainer + off-policy drivers.

Mirrors `SACAgent` (see sac/agent.mojo) but specialised for TD3:

  * Twin critics, deterministic actor, target policy smoothing,
    delayed actor update.
  * CPU-only — TD3Trainer.make is comptime-asserted on `target="cpu"`.

Usage:

    var agent = TD3Agent[ACTOR, CRITIC, OBS, ACT, BATCH, CAPACITY](
        actor_lr=3e-4, critic_lr=3e-4,
        exploration_noise=0.1, target_policy_noise=0.2,
        target_noise_clip=0.5, policy_delay=2,
    )
    var ep_returns = agent.train(env, total_timesteps=30_000)
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import BoxContinuousActionEnv

from ..training.batched_env import BatchedEnv
from ..training.driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_train_batched,
    run_offpolicy_eval,
)

from .trainer import TD3Trainer


struct TD3Agent[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](Movable & ImplicitlyDestructible):
    """Thin facade over `TD3Trainer` (CPU-only) + off-policy drivers."""

    var trainer: TD3Trainer[
        Self.ACTOR, Self.CRITIC,
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
    ]

    def __init__(
        out self,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 3e-4,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        action_scale: Scalar[DT] = 1.0,
        exploration_noise: Scalar[DT] = 0.1,
        target_policy_noise: Scalar[DT] = 0.2,
        target_noise_clip: Scalar[DT] = 0.5,
        policy_delay: Int = 2,
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1250.0,
        max_grad_norm: Scalar[DT] = 0.0,
    ) raises:
        """Construct a TD3Agent. Forwards every kwarg to `TD3Trainer.make`."""
        self.trainer = TD3Trainer[
            Self.ACTOR, Self.CRITIC,
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
        ].make["cpu"](
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            action_scale=action_scale,
            exploration_noise=exploration_noise,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            policy_delay=policy_delay,
            learning_starts=learning_starts,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
        )

    # ─── Training entry points ─────────────────────────────────────────

    def train[
        E: BatchedEnv,
        N_ENVS: Int = 1,
        NS: Int = 1,
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
    ) raises -> List[Scalar[DT]]:
        """Off-policy training via `run_offpolicy_train_batched` (CPU)."""
        return run_offpolicy_train_batched[
            TD3Trainer[
                Self.ACTOR, Self.CRITIC,
                Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
            ],
            E,
            N_ENVS,
            NS,
        ](
            None,
            self.trainer,
            env,
            total_timesteps,
            rng_seed=rng_seed,
            updates_per_step=updates_per_step,
            print_every=print_every,
            verbose=verbose,
            nstep_gamma=nstep_gamma,
        )

    def train_single[
        E: BoxContinuousActionEnv,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        print_every: Int = 1_000,
        verbose: Bool = True,
    ) raises -> List[Scalar[DT]]:
        """Single-env off-policy training via `run_offpolicy_train`."""
        return run_offpolicy_train[
            TD3Trainer[
                Self.ACTOR, Self.CRITIC,
                Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
            ],
            E,
        ](
            self.trainer,
            env,
            total_timesteps,
            print_every=print_every,
            verbose=verbose,
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
        """Greedy eval — deterministic actor, no exploration noise."""
        return run_offpolicy_eval[
            TD3Trainer[
                Self.ACTOR, Self.CRITIC,
                Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
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
