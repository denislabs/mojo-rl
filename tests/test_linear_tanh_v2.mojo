"""Test LinearTanhV2 module compiles and runs."""
from mojo_rl.nn.model.linear_tanh_v2 import LinearTanhV2
from mojo_rl.nn.gpu.matmul_ops import TILE_APPLE


def main():
    print("Testing LinearTanhV2...")

    # Create a layer
    var layer = LinearTanhV2[64, 32]()

    print("LinearTanhV2[64, 32] created successfully!")
    print("  IN_DIM:", layer.IN_DIM)
    print("  OUT_DIM:", layer.OUT_DIM)
    print("  PARAM_SIZE:", layer.PARAM_SIZE)
    print("  CACHE_SIZE:", layer.CACHE_SIZE)
    print("  TILE_APPLE:", TILE_APPLE)
    print("All tests passed!")
