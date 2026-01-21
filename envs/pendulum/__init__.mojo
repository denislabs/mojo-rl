"""Pendulum environment module.

This module provides both CPU and GPU implementations of Pendulum:
- PendulumEnv: Original CPU implementation (from ../pendulum.mojo)
- PendulumV2: GPU-accelerated implementation with GPUContinuousEnv trait

Usage:
    from envs.pendulum import PendulumV2, PendulumV2State, PendulumV2Action

    # Create environment
    var env = PendulumV2[DType.float32]()

    # CPU mode
    var obs = env.reset_obs_list()
    var action = List[Scalar[DType.float32]]()
    action.append(0.5)  # torque
    var result = env.step_continuous_vec(action)

    # GPU mode (batch training)
    PendulumV2[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](ctx, states)
    PendulumV2[DType.float32].step_kernel_gpu[BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM](
        ctx, states, actions, rewards, dones, obs
    )
"""

# V2 GPU-enabled components
from .constants import PConstants, PendulumLayout
from .state import PendulumV2State
from .action import PendulumV2Action
from .pendulum_v2 import PendulumV2
from .pendulum_v1 import PendulumEnv, PendulumState, PendulumAction
