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

## Optimized Matmul Ops for Apple Silicon (matmul_ops.mojo)
- matmul_bias_kernel: Matmul + bias (inference)
- matmul_bias_cache_input_kernel: Matmul + bias with input caching (training)
- matmul_bias_tanh_kernel: Fused matmul + bias + tanh (inference)
- matmul_bias_tanh_cached_kernel: Fused matmul + bias + tanh with caching
- matmul_bias_relu_kernel: Fused matmul + bias + ReLU (inference)
- matmul_bias_relu_cached_kernel: Fused matmul + bias + ReLU with caching
- matmul_backward_dx_kernel: Backward pass for input gradient
- matmul_backward_dW_kernel: Backward pass for weight gradient

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

## Continuous Control Kernels (continuous_kernels.mojo)
- td_target_continuous_kernel: DDPG TD target r + γ*Q_t(s',a')*(1-done)
- td_target_min_twin_kernel: TD3/SAC TD target with min(Q1,Q2) and optional entropy
- actor_grad_from_critic_kernel: Extract ∂Q/∂a from critic input gradient
- add_gaussian_noise_kernel: Clipped Gaussian noise for TD3 exploration
- sac_reparameterize_kernel: SAC reparameterization trick with Jacobian log-prob
- a2c_gae_kernel: GAE advantages + returns for A2C GPU training
- a2c_softmax_sample_kernel: Softmax categorical sampling for parallel A2C envs
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
from .matmul_ops import (
    TILE_APPLE,
    matmul_bias_kernel,
    matmul_bias_cache_input_kernel,
    matmul_bias_tanh_cached_kernel,
    matmul_bias_tanh_kernel,
    matmul_bias_relu_cached_kernel,
    matmul_bias_relu_kernel,
    matmul_backward_dx_kernel,
    matmul_backward_dW_kernel,
    get_forward_grid,
    get_backward_dx_grid,
    get_backward_dW_grid,
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
    store_transitions_kernel_nd,
    gather_batch_kernel_nd,
)

from .continuous_kernels import (
    td_target_continuous_kernel,
    td_target_min_twin_kernel,
    actor_grad_from_critic_kernel,
    add_gaussian_noise_kernel,
    sac_reparameterize_kernel,
    a2c_gae_kernel,
    a2c_softmax_sample_kernel,
    concat_obs_action_kernel,
    scale_clip_actions_kernel,
    ddpg_exploration_kernel,
    td_mse_grad_kernel,
)

from .gpu_train_scratch import GPUTrainScratch
