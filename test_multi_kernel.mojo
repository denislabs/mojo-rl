"""Test importing multiple kernels from a2c_native."""

from deep_agents.gpu.a2c_native import (
    naive_matmul_kernel,
    tiled_matmul_kernel,
    tiled_matmul_bias_relu_kernel,
    tiled_matmul_bias_kernel,
    parallel_softmax_kernel,
    policy_gradient_kernel,
    value_loss_gradient_kernel,
    tiled_matmul_transA_kernel,
    tiled_matmul_transB_kernel,
    relu_backward_kernel,
    elementwise_add_kernel,
    bias_gradient_parallel_kernel,
    sgd_update_2d_kernel,
    sgd_update_1d_kernel,
)


fn main() raises:
    print("Testing multiple kernel imports from a2c_native")
    print("All imports succeeded!")
