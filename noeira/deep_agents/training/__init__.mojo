"""RL training loop coordinators.

Agent-specific trainers (SAC / DDPG / TD3 / PPO / DQN / MBPO) and their
target-Y blocks now live under each agent's own `<agent>/` package
after the per-agent reorganization. This module retains only the
shared infrastructure (drivers, replay glue, episode tracking, GAE,
batched env adapter).
"""

from .episode_tracker import EpisodeTracker
from .off_policy_critic import (
    concat_sa,
)
from .action_sampling_block import ActionSamplingBlock
from .driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_train_batched,
    run_offpolicy_train_cpu_env_gpu_agent,
    run_offpolicy_eval,
)
from .driver_offpolicy_discrete import (
    OffPolicyDiscreteAgent,
    OffPolicyDiscreteAgentGpu,
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
    run_offpolicy_discrete_train_gpu_batched,
    run_offpolicy_discrete_eval_batched,
    run_offpolicy_discrete_train_cpu_env_gpu_agent,
    run_offpolicy_discrete_eval_cpu_env_gpu_agent,
)
from .driver_onpolicy import (
    OnPolicyAgent,
    OnPolicyAgentBatched,
    run_onpolicy_train,
    run_onpolicy_train_batched,
)
from .batched_env import (
    BatchedEnv,
    BatchedCpuEnv,
    BatchedCpuDiscreteEnv,
    BatchedGpuEnv,
    BatchedGpuDiscreteEnv,
)
