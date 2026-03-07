from .matmul_bias import FusedMatMulBias
from .matmul_bias_relu import FusedMatMulBiasReLU
from .matmul_bias_tanh import FusedMatMulBiasTanh
from .activation import (
    Activation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
    MishActivation,
)
from .matmul_bias_act import FusedMatMulBiasActivation

# Convenience aliases via parameterized type
comptime FusedMatMulBiasSigmoid[i: Int, o: Int] = FusedMatMulBiasActivation[
    i, o, SigmoidActivation
]
comptime FusedMatMulBiasMish[i: Int, o: Int] = FusedMatMulBiasActivation[
    i, o, MishActivation
]
