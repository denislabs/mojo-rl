"""REDQ GPU training loop.

Thin wrapper around `REDQAgent._run_train_gpu_impl`. The loop body lives
on the struct so `REDQAgent.train_gpu()` can call it via method dispatch
on `self`, sidestepping a Mojo nightly L-value unification bug that
trips when `self` is passed to a free function typed as
`mut agent: REDQAgent[Config, max_n_envs=n_envs]`.
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.core import TrainingMetrics, GPUContinuousEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.agents.redq_agent import REDQAgent
from mojo_rl.deep_agents.core.configs.redq_config import REDQConfig


def run_redq_train_gpu[
    E: GPUContinuousEnv,
    Config: REDQConfig,
    L: Logger = NoOpLogger,
    n_envs: Int = 1,
](
    mut agent: REDQAgent[Config, max_n_envs=n_envs],
    ctx: DeviceContext,
    num_steps: Int,
    warmup_steps: Int = 5_000,
    verbose: Bool = False,
    print_every: Int = 10_000,
    environment_name: String = "Environment",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
    rng_seed: UInt64 = 42,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
) raises -> TrainingMetrics:
    """REDQ GPU training loop. Delegates to `REDQAgent._run_train_gpu_impl`."""
    return agent._run_train_gpu_impl[E, L](
        ctx,
        num_steps,
        warmup_steps=warmup_steps,
        verbose=verbose,
        print_every=print_every,
        environment_name=environment_name,
        logger=logger,
        rng_seed=rng_seed,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
    )
