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
    DropoutOp,
    Flatten,
    Embedding,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    ScaledDotProductAttention,
)
from .fused import (
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
    FusedMatMulBiasSigmoid,
    FusedMatMulBiasMish,
    FusedMatMulBiasActivation,
    Activation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
    MishActivation,
)
from .fusion import FusionAnalyzer, FusedChain
from .auto_fused import AutoFused
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

# Fusion-aware aliases: AutoFused automatically detects and fuses patterns.
comptime Dense[in_d: Int, out_d: Int] = AutoFused[
    MatMul[in_d, out_d], BiasAdd[out_d]
]
comptime DenseReLU[in_d: Int, out_d: Int] = AutoFused[
    MatMul[in_d, out_d], BiasAdd[out_d], ReLUOp[out_d]
]
comptime DenseTanh[in_d: Int, out_d: Int] = AutoFused[
    MatMul[in_d, out_d], BiasAdd[out_d], TanhOp[out_d]
]
comptime DenseSigmoid[in_d: Int, out_d: Int] = AutoFused[
    MatMul[in_d, out_d], BiasAdd[out_d], SigmoidOp[out_d]
]
comptime DenseMish[in_d: Int, out_d: Int] = AutoFused[
    MatMul[in_d, out_d], BiasAdd[out_d], MishOp[out_d]
]

# ---------------------------------------------------------------------------
# Transformer composite aliases (Phase 8)
# ---------------------------------------------------------------------------
# Note: FFN, TransformerLayer, TransformerEncoder composites require
# Sequential which lives in nn.model — define them at the nn level or
# in user code. Example:
#   comptime FFN[d, ff] = Sequential[DenseReLU[d, ff], Dense[ff, d]]
#   comptime TransformerLayer[d, h, ff, s] = Sequential[
#       Residual[AutoDiffChain[
#           MatMul[s*d, s*d*3], BiasAdd[s*d*3],
#           ScaledDotProductAttention[d, h, s],
#       ]],
#       Residual[FFN[s*d, s*ff]],
#   ]
#   comptime TransformerEncoder[d, h, ff, s, n] = Repeat[n, TransformerLayer[d, h, ff, s]]
# ---------------------------------------------------------------------------
