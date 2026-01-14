"""GPU utilities for mojo-rl deep RL.

This module provides common GPU operations used across deep RL algorithms.

## Elementwise Operations (elementwise.mojo)
- gpu_add: output = a + b
- gpu_mul: output = a * b
- gpu_scale: output = input * scalar
- gpu_relu: output = max(input, 0)
- gpu_tanh: output = tanh(input)
- gpu_sigmoid: output = 1 / (1 + exp(-input))

## Matrix Operations (matmul.mojo)
- tiled_matmul_kernel: Optimized matmul using shared memory tiling

## Random Number Generation (random.mojo)
- xorshift32: Fast GPU-friendly PRNG
- random_uniform: Uniform random in [0, 1)
- random_range: Uniform random in [low, high)
- gaussian_noise: Standard Gaussian noise (CPU, uses stdlib random)
- gaussian_noise_pair: Two independent Gaussian samples (CPU)
- gaussian_noise_gpu: Standard Gaussian noise (GPU, maintains RNG state)
- gaussian_noise_pair_gpu: Two independent Gaussian samples (GPU)

## RL Kernels (rl_kernels.mojo)
- soft_update_kernel: Target network soft update
- zero_buffer_kernel: Zero out a buffer
- copy_buffer_kernel: Copy one buffer to another
- accumulate_rewards_kernel: Add step rewards to episode totals
- increment_steps_kernel: Increment step counters
- extract_completed_episodes_kernel: Extract completed episode data
- selective_reset_tracking_kernel: Reset tracking for done envs
- store_transitions_kernel: Store transitions to GPU replay buffer
- sample_indices_kernel: Generate random sample indices
- gather_batch_kernel: Gather sampled transitions into batch
"""


from .elementwise import (
    gpu_add,
    gpu_mul,
    gpu_scale,
    gpu_relu,
    gpu_tanh,
    gpu_sigmoid,
)
from .matmul import (
    tiled_matmul_kernel,
)

from .random import (
    xorshift32,
    random_uniform,
    random_range,
    gaussian_noise,
    gaussian_noise_pair,
    gaussian_noise_gpu,
    gaussian_noise_pair_gpu,
)

from .rl_kernels import (
    soft_update_kernel,
    zero_buffer_kernel,
    copy_buffer_kernel,
    accumulate_rewards_kernel,
    increment_steps_kernel,
    extract_completed_episodes_kernel,
    selective_reset_tracking_kernel,
    store_transitions_kernel,
    sample_indices_kernel,
    gather_batch_kernel,
)
