"""MBPO training loop.

Thin wrappers around `MBPOAgent._run_train_impl` / `_run_train_gpu_impl`.
The loop bodies live on the struct so convenience methods on `MBPOAgent`
(`train` / `train_gpu`) can call them via method dispatch on `self`,
sidestepping a Mojo nightly L-value unification bug that trips when
`self` is passed to a free function typed as
`mut agent: MBPOAgent[Config, L, ...]`.
"""

from std.gpu.host import DeviceContext
from mojo_rl.core import TrainingMetrics, BoxContinuousActionEnv, GPUContinuousEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.agents.mbpo_agent import MBPOAgent
from mojo_rl.deep_agents.core.configs.mbpo_config import MBPOConfig


def run_mbpo_train[
    E: BoxContinuousActionEnv,
    Config: MBPOConfig,
    L: Logger = NoOpLogger,
    TRAIN_N_ENVS: Int = 1,
    REAL_RATIO_PCT: Int = 5,
](
    mut agent: MBPOAgent[Config, L, TRAIN_N_ENVS, REAL_RATIO_PCT],
    mut cpu_state: MBPOAgent[
        Config, L, TRAIN_N_ENVS, REAL_RATIO_PCT
    ].CPUStateType,
    mut env: E,
    num_epochs: Int,
    steps_per_epoch: Int = 1000,
    max_steps_per_episode: Int = 1000,
    warmup_steps: Int = 5000,
    eval_episodes: Int = 5,
    eval_every: Int = 1,
    verbose: Bool = False,
    print_every: Int = 1,
    environment_name: String = "Environment",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
) raises -> TrainingMetrics:
    """MBPO CPU training loop. Delegates to `MBPOAgent._run_train_impl`."""
    return agent._run_train_impl[E](
        cpu_state,
        env,
        num_epochs,
        steps_per_epoch=steps_per_epoch,
        max_steps_per_episode=max_steps_per_episode,
        warmup_steps=warmup_steps,
        eval_episodes=eval_episodes,
        eval_every=eval_every,
        verbose=verbose,
        print_every=print_every,
        environment_name=environment_name,
        logger=logger,
    )


def run_mbpo_train_gpu[
    E: GPUContinuousEnv,
    Config: MBPOConfig,
    L: Logger = NoOpLogger,
    USE_CUDA_GRAPH: Bool = False,
    TRAIN_N_ENVS: Int = 1,
    REAL_RATIO_PCT: Int = 5,
](
    mut agent: MBPOAgent[Config, L, TRAIN_N_ENVS, REAL_RATIO_PCT],
    mut cpu_state: MBPOAgent[
        Config, L, TRAIN_N_ENVS, REAL_RATIO_PCT
    ].CPUStateType,
    ctx: DeviceContext,
    num_steps: Int,
    warmup_steps: Int = 5000,
    verbose: Bool = False,
    print_every: Int = 50_000,
    environment_name: String = "Environment",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
) raises -> TrainingMetrics:
    """MBPO GPU training loop. Delegates to `MBPOAgent._run_train_gpu_impl`."""
    return agent._run_train_gpu_impl[E, USE_CUDA_GRAPH](
        cpu_state,
        ctx,
        num_steps,
        warmup_steps=warmup_steps,
        verbose=verbose,
        print_every=print_every,
        environment_name=environment_name,
        logger=logger,
    )
