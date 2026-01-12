# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32
comptime TILE = 16  # Tile size for matmul kernels
comptime TPB = 256  # Threads per block for elementwise ops
