"""Activation trait for parameterized fused matmul+bias+activation ops.

Each Activation defines:
- OP_ID: matches the standalone DiffOp OP_ID (RELU=10, TANH=11, SIGMOID=12, MISH=13)
- FUSED_OP_ID: OP_ID for the fused variant (101, 102, 103, 104)
- forward(): apply activation to pre-activation scalar
- cache(): what to store for backward (pre-act for ReLU, output for Tanh/Sigmoid, input for Mish)
- backward(): compute grad_out * activation_derivative from cached value
"""

from ...constants import dtype
from std.math import tanh, exp, log


trait Activation(Movable & ImplicitlyCopyable):
    """Trait for activation functions used in fused ops."""

    comptime OP_ID: Int  # Matches standalone DiffOp OP_ID (e.g. RELU=10)
    comptime FUSED_OP_ID: Int  # OP_ID for fused matmul+bias variant (e.g. 101)
    comptime FUSED_CONV_OP_ID: Int  # OP_ID for fused conv2d variant (e.g. 110)

    @staticmethod
    def forward(pre_act: Scalar[dtype]) -> Scalar[dtype]:
        """Apply activation function to pre-activation value."""
        ...

    @staticmethod
    def cache(pre_act: Scalar[dtype], output: Scalar[dtype]) -> Scalar[dtype]:
        """Return what to cache for backward. Either pre_act or output."""
        ...

    @staticmethod
    def backward(
        cache_val: Scalar[dtype], grad_out: Scalar[dtype]
    ) -> Scalar[dtype]:
        """Compute grad_out * activation_derivative from cached value."""
        ...


struct ReLUActivation(Activation):
    """ReLU: max(0, x). Caches pre-activation for backward."""

    comptime OP_ID: Int = 10  # OpID.RELU
    comptime FUSED_OP_ID: Int = 101  # OpID.FUSED_MATMUL_BIAS_RELU
    comptime FUSED_CONV_OP_ID: Int = 110  # OpID.FUSED_CONV2D_RELU

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    @staticmethod
    def forward(pre_act: Scalar[dtype]) -> Scalar[dtype]:
        return pre_act if pre_act > 0 else 0

    @staticmethod
    def cache(pre_act: Scalar[dtype], output: Scalar[dtype]) -> Scalar[dtype]:
        return pre_act

    @staticmethod
    def backward(
        cache_val: Scalar[dtype], grad_out: Scalar[dtype]
    ) -> Scalar[dtype]:
        return grad_out if cache_val > 0 else 0


struct TanhActivation(Activation):
    """Tanh activation. Caches output for backward."""

    comptime OP_ID: Int = 11  # OpID.TANH
    comptime FUSED_OP_ID: Int = 102  # OpID.FUSED_MATMUL_BIAS_TANH
    comptime FUSED_CONV_OP_ID: Int = 111  # OpID.FUSED_CONV2D_TANH

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    @staticmethod
    def forward(pre_act: Scalar[dtype]) -> Scalar[dtype]:
        return tanh(pre_act)

    @staticmethod
    def cache(pre_act: Scalar[dtype], output: Scalar[dtype]) -> Scalar[dtype]:
        return output

    @staticmethod
    def backward(
        cache_val: Scalar[dtype], grad_out: Scalar[dtype]
    ) -> Scalar[dtype]:
        return grad_out * (1 - cache_val * cache_val)


struct SigmoidActivation(Activation):
    """Sigmoid: 1/(1+exp(-x)). Caches output for backward."""

    comptime OP_ID: Int = 12  # OpID.SIGMOID
    comptime FUSED_OP_ID: Int = 103  # OpID.FUSED_MATMUL_BIAS_SIGMOID
    comptime FUSED_CONV_OP_ID: Int = 112  # OpID.FUSED_CONV2D_SIGMOID

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    @staticmethod
    def forward(pre_act: Scalar[dtype]) -> Scalar[dtype]:
        return 1.0 / (1.0 + exp(-pre_act))

    @staticmethod
    def cache(pre_act: Scalar[dtype], output: Scalar[dtype]) -> Scalar[dtype]:
        return output

    @staticmethod
    def backward(
        cache_val: Scalar[dtype], grad_out: Scalar[dtype]
    ) -> Scalar[dtype]:
        return grad_out * cache_val * (1 - cache_val)


struct MishActivation(Activation):
    """Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))). Caches input for backward.
    """

    comptime OP_ID: Int = 13  # OpID.MISH
    comptime FUSED_OP_ID: Int = 104  # OpID.FUSED_MATMUL_BIAS_MISH
    comptime FUSED_CONV_OP_ID: Int = 113  # OpID.FUSED_CONV2D_MISH

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    @staticmethod
    def forward(pre_act: Scalar[dtype]) -> Scalar[dtype]:
        # Clamp for numerical stability: tanh(softplus(x)) -> 1 for x>15, -> 0 for x<-15
        if pre_act > Scalar[dtype](15.0):
            return pre_act  # tanh(sp) ≈ 1, so mish(x) ≈ x
        if pre_act < Scalar[dtype](-15.0):
            return Scalar[dtype](0.0)  # tanh(sp) ≈ 0
        var sp = log(1.0 + exp(pre_act))
        return pre_act * tanh(sp)

    @staticmethod
    def cache(pre_act: Scalar[dtype], output: Scalar[dtype]) -> Scalar[dtype]:
        return pre_act  # Need input x for backward

    @staticmethod
    def backward(
        cache_val: Scalar[dtype], grad_out: Scalar[dtype]
    ) -> Scalar[dtype]:
        var x = cache_val
        # Clamp for numerical stability
        if x > Scalar[dtype](15.0):
            return grad_out  # dmish ≈ 1 for large x
        if x < Scalar[dtype](-15.0):
            return Scalar[dtype](0.0)  # dmish ≈ 0 for very negative x
        var sp = log(1.0 + exp(x))
        var tsp = tanh(sp)
        var sig = 1.0 / (1.0 + exp(-x))
        # d/dx[x * tanh(sp(x))] = tanh(sp) + x * sig * (1 - tanh²(sp))
        var dmish = tsp + x * sig * (1.0 - tsp * tsp)
        return grad_out * dmish
