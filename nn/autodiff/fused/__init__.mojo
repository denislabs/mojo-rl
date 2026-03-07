from .matmul_bias import FusedMatMulBias
from .matmul_bias_relu import FusedMatMulBiasReLU
from .matmul_bias_tanh import FusedMatMulBiasTanh
from .activation import (
    Activation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
)
from .matmul_bias_act import FusedMatMulBiasActivation

# New activation: Sigmoid fused kernel via parameterized type
comptime FusedMatMulBiasSigmoid[i: Int, o: Int] = FusedMatMulBiasActivation[
    i, o, SigmoidActivation
]
