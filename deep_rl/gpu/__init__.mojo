"""GPU operations module for mojo-rl deep RL.

This module provides GPU-accelerated operations for neural network
training and inference on Apple Silicon (Metal) and NVIDIA GPUs (CUDA).

## Elementwise Operations (elementwise.mojo)
- gpu_add: output = a + b
- gpu_mul: output = a * b
- gpu_scale: output = input * scalar
- gpu_relu: output = max(input, 0)
- gpu_tanh: output = tanh(input)
- gpu_sigmoid: output = 1 / (1 + exp(-input))

## Reduction Operations (reduction.mojo) - Using block-wide primitives
- gpu_sum_kernel: Sum reduction using block.sum() (replaces 15+ lines!)
- gpu_max_kernel: Max reduction using block.max()
- gpu_mean_kernel: Mean using block.sum()
- gpu_normalize_kernel: Normalize by mean using block.sum() + block.broadcast()

## Matrix Operations (matmul.mojo)
- naive_matmul_kernel: Simple matmul, each thread computes one output element
- tiled_matmul_kernel: Optimized matmul using shared memory tiling (P16 pattern)
- gpu_matmul_naive: Wrapper for naive matmul
- gpu_matmul_tiled: Wrapper for tiled matmul

## Linear Layer Operations (linear.mojo)
- linear_forward_kernel: y = x @ W + b (fused matmul + bias)
- linear_forward_relu_kernel: y = max(0, x @ W + b) (fused with ReLU)
- linear_backward_dW_kernel: dW = x.T @ dy (weight gradient)
- linear_backward_db_kernel: db = sum(dy, axis=0) (bias gradient, using block.sum)
- linear_backward_dx_kernel: dx = dy @ W.T (input gradient)
- adam_update_kernel: Adam optimizer step
- soft_update_kernel: target = tau * source + (1 - tau) * target

## MLP Operations (mlp.mojo)
- linear_forward_tanh_kernel: y = tanh(x @ W + b) (fused for hidden layers)
- tanh_grad_kernel: grad = 1 - y^2 (given tanh output)
- elementwise_mul_kernel: output = a * b (for gradient chaining)
- Supports MLP2 (2-layer) and MLP3 (3-layer) architectures

## Actor-Critic Operations (actor_critic.mojo)
- relu_grad_kernel: grad = 1 if x > 0 else 0
- relu_grad_mul_kernel: fused upstream * relu_grad
- tanh_grad_mul_kernel: fused upstream * (1 - y^2)
- scale_action_kernel: output = tanh_output * action_scale
- concat_obs_action_kernel: output = [obs, action] concatenation
- split_grad_kernel: split gradient back to d_obs, d_action
- StochasticActor kernels for SAC:
  - split_mean_log_std_kernel: split combined output to mean/log_std
  - clamp_log_std_kernel: clamp log_std to valid range
  - sample_gaussian_kernel: action = mean + std * noise
  - compute_log_prob_kernel: Gaussian log probability
  - squash_action_kernel: tanh squashing for bounded actions
  - squash_log_prob_correction_kernel: correction for tanh squashing

## Usage
Run tests with:
    pixi run -e apple mojo run test_gpu_ops.mojo
    pixi run -e apple mojo run test_matmul.mojo
    pixi run -e apple mojo run test_gpu_linear.mojo
    pixi run -e apple mojo run test_gpu_mlp.mojo
    pixi run -e apple mojo run test_gpu_actor_critic.mojo

## Patterns Used
- Elementwise: algorithm.functional.elementwise (P23 pattern)
- Reduction: Block-wide primitives (P27 pattern) - block.sum(), block.max(), block.broadcast()
- Matmul: Shared memory tiling (P16 pattern)
- Linear: Fused tiled matmul + bias add, block.sum for db gradient
- MLP: Chained linear layers with tanh activations
"""
