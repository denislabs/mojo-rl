"""DDPGAgent — user-facing facade over DDPGTrainer + off-policy drivers.

Mirrors `SACAgent` (see sac/agent.mojo) but specialised for DDPG:

  * Single critic (no twin-critic min); deterministic actor with Gaussian
    exploration noise; no entropy temperature.
  * CPU-only — DDPGTrainer.make is comptime-asserted on `target="cpu"`.

Usage:

    var agent = DDPGAgent[ACTOR, CRITIC, OBS, ACT, BATCH, CAPACITY](
        actor_lr=1e-4, critic_lr=1e-3, noise_scale=0.1,
    )
    var ep_returns = agent.train(env, total_timesteps=30_000)
    var mean_eval = agent.eval(env, num_episodes=10)

`agent.trainer` remains exposed for power users.
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

from .trainer import DDPGTrainer


struct DDPGAgent[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](Movable & ImplicitlyDestructible):
    """Thin facade over `DDPGTrainer` (CPU-only) + off-policy drivers."""

    var trainer: DDPGTrainer[
        Self.ACTOR, Self.CRITIC,
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
    ]

    def __init__(
        out self,
        actor_lr: Scalar[DT] = 1e-4,
        critic_lr: Scalar[DT] = 1e-3,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        action_scale: Scalar[DT] = 1.0,
        noise_scale: Scalar[DT] = 0.1,
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1250.0,
        max_grad_norm: Scalar[DT] = 0.0,
    ) raises:
        """Construct a DDPGAgent. Forwards every kwarg to `DDPGTrainer.make`."""
        self.trainer = DDPGTrainer[
            Self.ACTOR, Self.CRITIC,
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
        ].make["cpu"](
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            action_scale=action_scale,
            noise_scale=noise_scale,
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
            DDPGTrainer[
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
            DDPGTrainer[
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
        """Greedy eval — actor mean action, no exploration noise."""
        return run_offpolicy_eval[
            DDPGTrainer[
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
