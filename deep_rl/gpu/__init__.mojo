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
