"""Linear layer: y = x @ W + b.

Alias for the autodiff-based FusedMatMulBias layer, which provides
identical parameter layout [W_flat | b] with automatic kernel fusion.

PARAM_SIZE = in_dim * out_dim + out_dim
CACHE_SIZE = in_dim
"""

from ..autodiff import AutoFused, MatMul, BiasAdd

comptime Linear[in_dim: Int, out_dim: Int] = AutoFused[
    MatMul[in_dim, out_dim], BiasAdd[out_dim]
]
