from .op import DiffOp, FusedOp, OpID
from .chain import AutoDiffChain
from .primitives import (
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    MishOp,
    SwishOp,
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
    SymlogOp,
)
from .fused import (
    FusedMatMulBias,
    FusedMatMulBiasActivation,
    Activation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
    MishActivation,
    SwishActivation,
)
from .auto_fused import AutoFused
from .combinators import Residual, ProjectedResidual, Parallel, Repeat, Tokenwise
from .compute_graph import ComputeGraph, GNode, GraphNode
from .composite_params import CompositeParams

# Fusion-aware aliases: AutoFused automatically detects and fuses patterns.
# Dense/DenseReLU/etc. are identical to Linear/LinearReLU from mojo_rl.nn.model.
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
comptime DenseSwish[in_d: Int, out_d: Int] = AutoFused[
    MatMul[in_d, out_d], BiasAdd[out_d], SwishOp[out_d]
]
