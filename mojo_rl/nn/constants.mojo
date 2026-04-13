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

# ─── GPU buffer alignment ───────────────────────────────────────────────
# TMA (Tensor Memory Accelerator) on SM100+ requires 16-byte alignment.
# For float32 (4 bytes), 4-element alignment = 16 bytes → OK.
# For bfloat16 (2 bytes), 4-element alignment = 8 bytes → misaligned!
# We align to max(4, 16 // sizeof(dtype)) elements so all dtypes get
# 16-byte alignment.  For float32 this is still 4 (no PARAM_SIZE change).


def gpu_align(x: Int) -> Int:
    """Round up element count for GPU buffer alignment.

    Uses 4-element alignment (16 bytes for float32).
    NOTE: bf16/fp16 need 8-element alignment for TMA on SM100.
    This will be increased when switching forward to eval_kernel_mma
    (which has bounds checking and doesn't require TMA alignment).
    """
    return (x + 3) & ~3
