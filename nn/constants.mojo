# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32
comptime TILE = 16  # Tile size for matmul kernels (optimal for Apple Silicon M1)
comptime TPB = 256  # Threads per block for elementwise ops

# MMA (Tensor Core) constants — shared across gpu/matmul.mojo and autodiff kernels
comptime MMA_M = 16  # Output rows per warp MMA op (m16n8k8)
comptime MMA_N = 8  # Output cols per warp MMA op
comptime MMA_K = 8  # Reduction dimension per MMA step
comptime MMA_BLOCK_M = 32  # 2 × MMA_M — block-level tile rows
comptime MMA_BLOCK_N = 32  # 4 × MMA_N — block-level tile cols
comptime MMA_WARPS_M = 2
comptime MMA_WARPS_N = 4
comptime MMA_NUM_WARPS = 8
comptime MMA_BLOCK_THREADS = MMA_NUM_WARPS * 32  # 256
