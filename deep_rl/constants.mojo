# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32
comptime TILE = 16  # Tile size for matmul kernels (optimal for Apple Silicon M1)
comptime TPB = 256  # Threads per block for elementwise ops
