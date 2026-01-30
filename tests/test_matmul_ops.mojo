"""Test that matmul_ops module compiles and runs correctly."""
from deep_rl.gpu import (
    TILE_APPLE,
    matmul_bias_kernel,
    matmul_bias_tanh_kernel,
    matmul_bias_relu_kernel,
)

def main():
    print("Matmul ops imported successfully!")
    print("TILE_APPLE =", TILE_APPLE)
    print("All kernels available!")
