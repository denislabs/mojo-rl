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

Note: RL-specific kernels have moved to deep_agents.core.kernels and
agent-specific kernels.mojo files (sac, td3, a2c).
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
)

from .gpu_train_scratch import GPUTrainScratch
