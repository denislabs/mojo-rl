"""PCN-local kernel launch constant (vendored during the nn2
re-architecture, Phase B).

PCN's hand-written elementwise + RMSNorm-reduction GPU kernels assume a
256-thread block. nn2 deliberately uses `TPB = 128`, so PCN keeps its own
value locally rather than silently changing launch geometry by adopting
nn2's. Value is byte-for-byte the legacy `nn.constants` PCN previously
imported.

(The `MMA_BLOCK_*` tiling constants were removed in Phase D: PCN's GPU
matmuls now go through `linalg.matmul`/`max_matmul`, retiring the 2×2
register-tiled custom-MMA kernels they sized.)
"""

# Threads per block for PCN's elementwise + reduction kernels (legacy nn.TPB).
comptime TPB = 256
