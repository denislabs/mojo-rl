"""Linear layers: plain and fused with activations.

All are aliases for autodiff-based AutoFused layers, which provide
identical parameter layout [W_flat | b] with automatic kernel fusion.

Linear:         y = x @ W + b           (CACHE_SIZE = in_dim)
Linear + Act:   y = act(x @ W + b)      (CACHE_SIZE = in_dim + out_dim)
"""

from ..autodiff import (
    AutoFused,
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    MishOp,
    SwishOp,
)

comptime Linear[
    in_dim: Int,
    out_dim: Int,
    USE_MAX_KERNELS: Bool = False,
] = AutoFused[
    MatMul[in_dim, out_dim, USE_MAX_KERNELS],
    BiasAdd[out_dim],
    USE_MAX_KERNELS=USE_MAX_KERNELS,
]

comptime LinearReLU[
    in_dim: Int,
    out_dim: Int,
    USE_MAX_KERNELS: Bool = False,
] = AutoFused[
    MatMul[in_dim, out_dim, USE_MAX_KERNELS],
    BiasAdd[out_dim],
    ReLUOp[out_dim],
    USE_MAX_KERNELS=USE_MAX_KERNELS,
]

comptime LinearTanh[
    in_dim: Int,
    out_dim: Int,
    USE_MAX_KERNELS: Bool = False,
] = AutoFused[
    MatMul[in_dim, out_dim, USE_MAX_KERNELS],
    BiasAdd[out_dim],
    TanhOp[out_dim],
    USE_MAX_KERNELS=USE_MAX_KERNELS,
]

comptime LinearSigmoid[
    in_dim: Int,
    out_dim: Int,
    USE_MAX_KERNELS: Bool = False,
] = AutoFused[
    MatMul[in_dim, out_dim, USE_MAX_KERNELS],
    BiasAdd[out_dim],
    SigmoidOp[out_dim],
    USE_MAX_KERNELS=USE_MAX_KERNELS,
]

comptime LinearMish[
    in_dim: Int,
    out_dim: Int,
    USE_MAX_KERNELS: Bool = False,
] = AutoFused[
    MatMul[in_dim, out_dim, USE_MAX_KERNELS],
    BiasAdd[out_dim],
    MishOp[out_dim],
    USE_MAX_KERNELS=USE_MAX_KERNELS,
]

comptime LinearSwish[
    in_dim: Int,
    out_dim: Int,
    USE_MAX_KERNELS: Bool = False,
] = AutoFused[
    MatMul[in_dim, out_dim, USE_MAX_KERNELS],
    BiasAdd[out_dim],
    SwishOp[out_dim],
    USE_MAX_KERNELS=USE_MAX_KERNELS,
]

# NoisyLinear compositions (using Sequential since NoisyLinear is a Model, not DiffOp)
from .noisy_linear import NoisyLinear
from .sequential import Sequential
from .relu import ReLU
from .tanh import Tanh

comptime NoisyLinearReLU[in_dim: Int, out_dim: Int] = Sequential[
    NoisyLinear[in_dim, out_dim], ReLU[out_dim]
]

comptime NoisyLinearTanh[in_dim: Int, out_dim: Int] = Sequential[
    NoisyLinear[in_dim, out_dim], Tanh[out_dim]
]
