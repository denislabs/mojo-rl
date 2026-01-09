"""Generic GPU REINFORCE Algorithm.

This module provides a composable REINFORCE implementation that works with
any environment implementing the GPUEnv trait.

Key design:
1. Training kernel is parameterized by Env type (compile-time polymorphism)
2. Multi-step kernel pattern: many steps per kernel launch
3. Policy network: simple 2-layer MLP with softmax output
4. All computation on GPU with minimal CPU interaction

Usage:
    from deep_rl.gpu.envs.cartpole import GPUCartPole
    from deep_rl.gpu.algorithms.reinforce import train_reinforce

    with DeviceContext() as ctx:
        train_reinforce[GPUCartPole, HIDDEN_DIM=32, NUM_ENVS=1024](
            ctx, num_updates=100, lr=0.01, gamma=0.99
        )
"""

from time import perf_counter_ns
from math import exp, log, sqrt
from random import random_float64

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from deep_rl.gpu import xorshift32, random_uniform
from core import GPUEnvDims


# =============================================================================
# Toy Agent Training Kernel (Generic over Env)
# =============================================================================

comptime NUM_ENVS: Int = 1024
comptime STEPS_PER_KERNEL: Int = 200
comptime HIDDEN_DIM: Int = 32
comptime OBS_DIM: Int = 4
comptime NUM_ACTIONS: Int = 2
comptime STATE_SIZE: Int = 4


fn toy_agent_kernel[
    out_layout: Layout,
    a_layout: Layout,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    state: LayoutTensor[dtype, state_layout, ImmutAnyOrigin],
    size: UInt,
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0
