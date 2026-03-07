"""Compile-time fusion pass infrastructure.

FusionAnalyzer examines a variadic DiffOp chain and detects fusible patterns
(MatMul+BiasAdd, MatMul+BiasAdd+Activation) at each position.

FusedChain provides pre-built fused AutoDiffChain aliases for common MLP
topologies (1-layer, 2-layer, 3-layer) with automatic pattern-based fusion.

Usage:
    # Manual fusion analysis
    comptime FA = FusionAnalyzer[MatMul[3,4], BiasAdd[4], ReLUOp[4]]
    comptime if FA._is_matmul_bias_activation_at[0]():
        # Use FusedMatMulBiasActivation instead

    # Automatic: fuse a 2-layer MLP
    comptime MyModel = FusedChain.two_layer_relu[3, 64, 2]
    # Equivalent to AutoDiffChain[FusedMatMulBiasReLU[3,64], FusedMatMulBias[64,2]]
"""

from ..constants import dtype
from .op import DiffOp, OpID
from .chain import AutoDiffChain
from .primitives import MatMul, BiasAdd, ReLUOp, TanhOp, SigmoidOp
from .fused import (
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
    FusedMatMulBiasSigmoid,
)
from std.builtin.variadics import Variadic


# =============================================================================
# FusionAnalyzer — compile-time pattern detection on DiffOp chains
# =============================================================================


struct FusionAnalyzer[*OPS: DiffOp]:
    """Analyze a DiffOp chain for fusible patterns using OP_ID matching.

    Pattern matchers use `comptime if in_bounds: return (...) else: return False`
    to avoid compiler crashes from variadic out-of-bounds access in dead code.
    """

    comptime op_types = Variadic.types[T=DiffOp, *Self.OPS]
    comptime N = Variadic.size(Self.op_types)

    @staticmethod
    fn _is_matmul_bias_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+2] is MatMul followed by BiasAdd."""
        comptime if idx + 1 < Self.N:
            return (
                Self.op_types[idx].OP_ID == OpID.MATMUL._value
                and Self.op_types[idx + 1].OP_ID == OpID.BIAS_ADD._value
            )
        else:
            return False

    @staticmethod
    fn _is_matmul_bias_relu_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+3] is MatMul, BiasAdd, ReLU."""
        comptime if idx + 2 < Self.N:
            return (
                Self.op_types[idx].OP_ID == OpID.MATMUL._value
                and Self.op_types[idx + 1].OP_ID == OpID.BIAS_ADD._value
                and Self.op_types[idx + 2].OP_ID == OpID.RELU._value
            )
        else:
            return False

    @staticmethod
    fn _is_matmul_bias_tanh_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+3] is MatMul, BiasAdd, Tanh."""
        comptime if idx + 2 < Self.N:
            return (
                Self.op_types[idx].OP_ID == OpID.MATMUL._value
                and Self.op_types[idx + 1].OP_ID == OpID.BIAS_ADD._value
                and Self.op_types[idx + 2].OP_ID == OpID.TANH._value
            )
        else:
            return False

    @staticmethod
    fn _is_matmul_bias_sigmoid_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+3] is MatMul, BiasAdd, Sigmoid."""
        comptime if idx + 2 < Self.N:
            return (
                Self.op_types[idx].OP_ID == OpID.MATMUL._value
                and Self.op_types[idx + 1].OP_ID == OpID.BIAS_ADD._value
                and Self.op_types[idx + 2].OP_ID == OpID.SIGMOID._value
            )
        else:
            return False

    @staticmethod
    fn _is_matmul_bias_activation_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+3] is MatMul, BiasAdd, any activation (OP_ID 10-19)."""
        comptime if idx + 2 < Self.N:
            return (
                Self.op_types[idx].OP_ID == OpID.MATMUL._value
                and Self.op_types[idx + 1].OP_ID == OpID.BIAS_ADD._value
                and Self.op_types[idx + 2].OP_ID >= 10
                and Self.op_types[idx + 2].OP_ID <= 19
            )
        else:
            return False

    @staticmethod
    fn _best_fusion_at[idx: Int]() -> String:
        """Return the best fusion pattern starting at idx.

        Returns "mbr" for MatMul+BiasAdd+ReLU (3 ops),
                "mbt" for MatMul+BiasAdd+Tanh (3 ops),
                "mbs" for MatMul+BiasAdd+Sigmoid (3 ops),
                "mb"  for MatMul+BiasAdd (2 ops),
                ""    for no fusion available.
        Greedy: tries 3-op patterns first, then 2-op.
        """
        comptime if Self._is_matmul_bias_relu_at[idx]():
            return "mbr"
        elif Self._is_matmul_bias_tanh_at[idx]():
            return "mbt"
        elif Self._is_matmul_bias_sigmoid_at[idx]():
            return "mbs"
        elif Self._is_matmul_bias_at[idx]():
            return "mb"
        else:
            return ""


# =============================================================================
# FusedChain — pre-built fused AutoDiffChain for common MLP topologies
# =============================================================================


struct FusedChain:
    """Factory for fused AutoDiffChain aliases.

    Provides parameterized type aliases for common MLP patterns with
    automatic fusion applied. Each layer is fused into a single kernel.

    Activation options: "relu", "tanh", "sigmoid", "none"
    - "relu"    -> FusedMatMulBiasReLU (single kernel: y = relu(x@W + b))
    - "tanh"    -> FusedMatMulBiasTanh (single kernel: y = tanh(x@W + b))
    - "sigmoid" -> FusedMatMulBiasSigmoid (single kernel: y = sigmoid(x@W + b))
    - "none"    -> FusedMatMulBias (single kernel: y = x@W + b)
    """

    # --- Single-layer fused ---

    comptime one_layer_relu[in_d: Int, out_d: Int] = AutoDiffChain[
        FusedMatMulBiasReLU[in_d, out_d]
    ]

    comptime one_layer_tanh[in_d: Int, out_d: Int] = AutoDiffChain[
        FusedMatMulBiasTanh[in_d, out_d]
    ]

    comptime one_layer_sigmoid[in_d: Int, out_d: Int] = AutoDiffChain[
        FusedMatMulBiasSigmoid[in_d, out_d]
    ]

    comptime one_layer_linear[in_d: Int, out_d: Int] = AutoDiffChain[
        FusedMatMulBias[in_d, out_d]
    ]

    # --- Two-layer fused ---
    # Common pattern: hidden layer with activation + output layer (linear)

    comptime two_layer_relu[in_d: Int, hid: Int, out_d: Int] = AutoDiffChain[
        FusedMatMulBiasReLU[in_d, hid],
        FusedMatMulBias[hid, out_d],
    ]

    comptime two_layer_tanh[in_d: Int, hid: Int, out_d: Int] = AutoDiffChain[
        FusedMatMulBiasTanh[in_d, hid],
        FusedMatMulBias[hid, out_d],
    ]

    # --- Three-layer fused (2 hidden + output) ---

    comptime three_layer_relu[
        in_d: Int, h1: Int, h2: Int, out_d: Int
    ] = AutoDiffChain[
        FusedMatMulBiasReLU[in_d, h1],
        FusedMatMulBiasReLU[h1, h2],
        FusedMatMulBias[h2, out_d],
    ]

    comptime three_layer_tanh[
        in_d: Int, h1: Int, h2: Int, out_d: Int
    ] = AutoDiffChain[
        FusedMatMulBiasTanh[in_d, h1],
        FusedMatMulBiasTanh[h1, h2],
        FusedMatMulBias[h2, out_d],
    ]

    # --- Common RL architectures ---

    comptime mlp_relu[in_d: Int, hid: Int, out_d: Int] = AutoDiffChain[
        FusedMatMulBiasReLU[in_d, hid],
        FusedMatMulBiasReLU[hid, hid],
        FusedMatMulBias[hid, out_d],
    ]

    comptime mlp_tanh[in_d: Int, hid: Int, out_d: Int] = AutoDiffChain[
        FusedMatMulBiasTanh[in_d, hid],
        FusedMatMulBiasTanh[hid, hid],
        FusedMatMulBias[hid, out_d],
    ]
