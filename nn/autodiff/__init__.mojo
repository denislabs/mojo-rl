from .op import DiffOp, FusedOp, OpID
from .chain import AutoDiffChain
from .primitives import (
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    MishOp,
    Scale,
    ElemMul,
    ReduceSum,
    ReduceMean,
    SoftmaxOp,
    LayerNormOp,
    RMSNormOp,
)
from .fused import (
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
    FusedMatMulBiasSigmoid,
    FusedMatMulBiasActivation,
    Activation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
)
from .fusion import FusionAnalyzer, FusedChain
from .combinators import Residual, Parallel, Repeat

# Convenience aliases: parameterized type aliases for common patterns.
# Usage: var model = LinearAD[4, 64]()
comptime LinearAD[in_d: Int, out_d: Int] = AutoDiffChain[
    MatMul[in_d, out_d], BiasAdd[out_d]
]
comptime LinearReLUAD[in_d: Int, out_d: Int] = AutoDiffChain[
    MatMul[in_d, out_d], BiasAdd[out_d], ReLUOp[out_d]
]
comptime LinearTanhAD[in_d: Int, out_d: Int] = AutoDiffChain[
    MatMul[in_d, out_d], BiasAdd[out_d], TanhOp[out_d]
]

# Fusion-aware aliases: single fused kernel per layer.
comptime Dense[in_d: Int, out_d: Int] = AutoDiffChain[
    FusedMatMulBias[in_d, out_d]
]
comptime DenseReLU[in_d: Int, out_d: Int] = AutoDiffChain[
    FusedMatMulBiasReLU[in_d, out_d]
]
comptime DenseTanh[in_d: Int, out_d: Int] = AutoDiffChain[
    FusedMatMulBiasTanh[in_d, out_d]
]
comptime DenseSigmoid[in_d: Int, out_d: Int] = AutoDiffChain[
    FusedMatMulBiasSigmoid[in_d, out_d]
]
