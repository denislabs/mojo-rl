"""DQNAgent — user-facing facade over DQNTrainer + discrete off-policy drivers.

Discrete-action DQN. Returns integer actions (argmax over Q) rather than
the continuous-action float vector that SAC/DDPG/TD3/PPO produce.

Usage:

    var agent = DQNAgent[
        "cpu", UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
    ](lr=1e-3, gamma=0.99, epsilon=1.0)
    var ep_returns = agent.train(env, total_timesteps=10_000)
    var mean_eval = agent.eval(env, num_episodes=10)

`DOUBLE=True` enables Double-DQN target-Y (van Hasselt et al. 2016).
"""

from std.gpu.host import DeviceContext

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.checkpoint import save_state_v2, load_state_v2
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import BoxDiscreteActionEnv

from ..training.blocks import SampleBlock
from ..training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from ..core.checkpoint_helpers import (
    save_optimizer_v2,
    load_optimizer_v2,
)

from .metrics import DQNMetrics
from .trainer import DQNTrainer


struct DQNAgent[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    DOUBLE: Bool = False,
](Movable & ImplicitlyDestructible):
    """Thin facade over `DQNTrainer` + discrete off-policy drivers."""

    var trainer: DQNTrainer[
        Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = 1e-3,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        epsilon: Scalar[DT] = 1.0,
        epsilon_decay: Scalar[DT] = 0.995,
        epsilon_min: Scalar[DT] = 0.01,
        learning_starts: Int = 1_000,
        target_update_freq: Int = 0,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = 0.0,
        max_grad_norm: Scalar[DT] = 0.0,
    ) raises:
        """Construct a DQNAgent. Forwards every kwarg to `DQNTrainer.make`."""
        self.trainer = DQNTrainer[
            Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
        ].make(
            ctx=ctx,
            lr=lr,
            gamma=gamma,
            tau=tau,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            learning_starts=learning_starts,
            target_update_freq=target_update_freq,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
        )

    # ─── Training entry point ──────────────────────────────────────────

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
    ) raises -> List[Scalar[DT]]:
        """Single-env discrete off-policy training via
        `run_offpolicy_discrete_train`. Covers `(env=cpu, train=cpu)` and
        `(env=cpu, train=gpu)` — env is always CPU-side for discrete."""
        var ctx = self.trainer.ctx
        return run_offpolicy_discrete_train[
            DQNTrainer[
                Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
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
        E: BoxDiscreteActionEnv,
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        *,
        max_steps_per_episode: Int = 1_000,
        verbose: Bool = False,
    ) raises -> Scalar[DT]:
        """Greedy eval — argmax Q, no epsilon. Returns mean episode return."""
        return run_offpolicy_discrete_eval[
            DQNTrainer[
                Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
            ],
            E,
        ](
            self.trainer,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            verbose=verbose,
        )

    # ─── Single-step inference ─────────────────────────────────────────

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Argmax-Q action (no epsilon). Returns the integer action index."""
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
    ) raises -> DQNMetrics:
        """Drain trainer accumulators into a DQNMetrics bundle."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        return self.trainer.flush_timer_log()

    # ─── Checkpointing (CPU only) ──────────────────────────────────────

    def save(mut self, path: String) raises:
        """Persist Q-network + optimizer to `path/` (must exist).
        CPU-only — see SACAgent.save for the GPU caveat."""
        comptime if Self.train_target != "cpu":
            raise Error(
                "DQNAgent.save: GPU save/load not yet supported. Train on"
                " CPU or wait for the device-sync helper."
            )
        save_state_v2(self.trainer.pair.online, path + "/q_net.ckpt")
        save_optimizer_v2(self.trainer.q_opt, path + "/q_opt.ckpt")

    def load(mut self, path: String) raises:
        """Restore Q-network + optimizer. Target net hard-copied from
        the online net."""
        comptime if Self.train_target != "cpu":
            raise Error(
                "DQNAgent.load: GPU save/load not yet supported. Train on"
                " CPU or wait for the device-sync helper."
            )
        load_state_v2(self.trainer.pair.online, path + "/q_net.ckpt")
        hard_copy_params[Self.train_target, M=Self.Q_NET](
            self.trainer.pair.online, self.trainer.pair.target_net,
            self.trainer.ctx,
        )
        load_optimizer_v2(self.trainer.q_opt, path + "/q_opt.ckpt")
