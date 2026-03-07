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
# Composite model aliases (Phase 9)
# ---------------------------------------------------------------------------
# Pre-built architectures are in nn.composites (requires Sequential from
# nn.model, so can't be defined here). Available:
#   from nn.composites import ResBlock, ResNet, LeNet8x8, FFN
#
# ResBlock[dim]           = Residual[Sequential[DenseReLU[dim,dim], Dense[dim,dim]]]
# ResNet[in,dim,out,d]    = Sequential[DenseReLU, Repeat[d, ResBlock], Dense]
# LeNet8x8                = Conv->Pool->Conv->Dense pipeline for 8x8 input
# FFN[dim, ff_dim]        = Sequential[DenseReLU[dim,ff], Dense[ff,dim]]
#
# Transformer/GPT composites: define in user code using these building blocks:
#   comptime AttnBlock = AutoDiffChain[MatMul[SD, SD*3], BiasAdd[SD*3], SDPA[D,H,S]]
#   comptime TLayer = Sequential[Residual[AttnBlock], Residual[FFN[SD, FF]]]
#   comptime Encoder = Repeat[N, TLayer]
#   comptime GPT = Sequential[Embedding[V, SD], Encoder, Dense[SD, V]]
# ---------------------------------------------------------------------------
