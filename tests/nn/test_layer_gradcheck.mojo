"""Level 1: CPU numerical gradcheck for all nn layer types.

Validates analytical backward() against central finite differences for both
param gradients and input gradients. This is the ground truth for backward
correctness -- if CPU gradcheck passes, the backward implementation is correct.

Tests leaf layers, combinators, and realistic architectures independently.
BatchNorm2D, NoisyLinear, and ResBlock (GPU-only backward) are excluded --
they have dedicated test files.

Usage:
    pixi run mojo run -I . tests/nn/test_layer_gradcheck.mojo
"""

from std.math import abs
from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Parallel,
    Linear,
    LinearReLU,
    LinearTanh,
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
    Mish,
    LayerNorm,
    Conv2DReLU,
    Conv2DLayer,
    Conv2DBatchNormReLU,
    FlattenLayer,
    ResBlockConv2D,
    Residual,
    Repeat,
    SkipConcat,
    DualPath,
    SplitApply,
    FanOut,
)
from mojo_rl.nn.model.resblock_conv2d_bn import ResBlockConv2DBN


def cpu_gradcheck[M: Model, BS: Int = 4](
    name: String,
    eps: Float64 = 1e-3,
    tol: Float64 = 0.01,
    max_params: Int = 200,
    max_inputs: Int = 100,
) raises:
    """CPU numerical gradcheck for any Model.

    Checks param gradients and input gradients via central finite differences.
    Uses eps=1e-3 (standard for float32: better signal-to-noise than 1e-4).
    Reports PASS/FAIL with max relative error.
    """
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE

    print("Gradcheck:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS, ")")

    # Initialize parameters
    var state = NetworkState[M, Adam[]]()
    state.initialize[Xavier[]]()

    # Allocate buffers (heap -- works for any size)
    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var output_ptr = alloc[Scalar[dtype]](BS * OUT)
    var cache_ptr = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)
    var grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    var out_plus_ptr = alloc[Scalar[dtype]](BS * OUT)
    var out_minus_ptr = alloc[Scalar[dtype]](BS * OUT)
    var fd_cache_ptr = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)

    # Fill input: deterministic, avoids ReLU dead zones
    for i in range(BS * IN):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)

    # Fill grad_output: varied, not all-same
    for i in range(BS * OUT):
        (grad_out_ptr + i)[] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )

    # Create LayoutTensor views
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_ptr
    )
    var output_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        output_ptr
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        cache_ptr
    )
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_ptr)
    var grad_in_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        grad_in_ptr
    )
    var out_plus_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](out_plus_ptr)
    var out_minus_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](out_minus_ptr)
    var fd_cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, CS), MutAnyOrigin
    ](fd_cache_ptr)

    # Forward + analytical backward
    M.forward[BS](input_t, output_t, state.params_view(), cache_t)
    state.zero_grads()
    memset(grad_in_ptr, 0, BS * IN)
    var grads = state.grads_view()
    M.backward[BS](
        grad_out_t, grad_in_t, state.params_view(), cache_t, grads
    )

    # === Check param gradients via finite differences ===
    var p_fail = 0
    var p_checked = 0
    var p_max_rel: Float64 = 0.0

    if PS > 0:
        var p_step = PS // max_params
        if p_step < 1:
            p_step = 1

        for p_idx in range(0, PS, p_step):
            var orig = (state.params + p_idx)[]

            # f(p + eps)
            (state.params + p_idx)[] = Scalar[dtype](Float64(orig) + eps)
            M.forward[BS](input_t, out_plus_t, state.params_view(), fd_cache_t)

            # f(p - eps)
            (state.params + p_idx)[] = Scalar[dtype](Float64(orig) - eps)
            M.forward[BS](input_t, out_minus_t, state.params_view(), fd_cache_t)

            # Restore
            (state.params + p_idx)[] = orig

            # Numerical gradient = sum_j(grad_out_j * (f_plus_j - f_minus_j) / (2*eps))
            var num_grad: Float64 = 0.0
            for j in range(BS * OUT):
                var go = Float64((grad_out_ptr + j)[])
                var fp = Float64((out_plus_ptr + j)[])
                var fm = Float64((out_minus_ptr + j)[])
                num_grad += go * (fp - fm) / (2.0 * eps)

            var ana_grad = Float64((state.grads + p_idx)[])
            var err = abs(ana_grad - num_grad)
            var denom = abs(ana_grad) + abs(num_grad)
            var rel: Float64 = 0.0
            if denom > 1e-5:
                rel = err / denom

            if rel > p_max_rel:
                p_max_rel = rel
            if rel > tol and denom > 1e-4:
                p_fail += 1
                if p_fail <= 3:
                    print(
                        "    PARAM p[",
                        p_idx,
                        "]: ana=",
                        ana_grad,
                        "num=",
                        num_grad,
                        "rel=",
                        rel,
                    )
            p_checked += 1

    # === Check input gradients via finite differences ===
    var i_fail = 0
    var i_checked = 0
    var i_max_rel: Float64 = 0.0

    var i_total = BS * IN
    var i_step = i_total // max_inputs
    if i_step < 1:
        i_step = 1

    for i_idx in range(0, i_total, i_step):
        var orig = (input_ptr + i_idx)[]

        # f(x + eps)
        (input_ptr + i_idx)[] = Scalar[dtype](Float64(orig) + eps)
        M.forward[BS](input_t, out_plus_t, state.params_view(), fd_cache_t)

        # f(x - eps)
        (input_ptr + i_idx)[] = Scalar[dtype](Float64(orig) - eps)
        M.forward[BS](input_t, out_minus_t, state.params_view(), fd_cache_t)

        # Restore
        (input_ptr + i_idx)[] = orig

        var num_grad: Float64 = 0.0
        for j in range(BS * OUT):
            var go = Float64((grad_out_ptr + j)[])
            var fp = Float64((out_plus_ptr + j)[])
            var fm = Float64((out_minus_ptr + j)[])
            num_grad += go * (fp - fm) / (2.0 * eps)

        var ana_grad = Float64((grad_in_ptr + i_idx)[])
        var err = abs(ana_grad - num_grad)
        var denom = abs(ana_grad) + abs(num_grad)
        var rel: Float64 = 0.0
        if denom > 1e-5:
            rel = err / denom

        if rel > i_max_rel:
            i_max_rel = rel
        if rel > tol and denom > 1e-4:
            i_fail += 1
            if i_fail <= 3:
                print(
                    "    INPUT i[",
                    i_idx,
                    "]: ana=",
                    ana_grad,
                    "num=",
                    num_grad,
                    "rel=",
                    rel,
                )
        i_checked += 1

    # === Report ===
    if PS > 0:
        if p_fail == 0:
            print(
                "  [PASS] params: max_rel=",
                p_max_rel,
                "(",
                p_checked,
                "checked)",
            )
        else:
            print(
                "  [FAIL] params:",
                p_fail,
                "/",
                p_checked,
                "max_rel=",
                p_max_rel,
            )
    else:
        print("  [SKIP] params: PARAM_SIZE=0")

    if i_fail == 0:
        print(
            "  [PASS] inputs: max_rel=",
            i_max_rel,
            "(",
            i_checked,
            "checked)",
        )
    else:
        print(
            "  [FAIL] inputs:",
            i_fail,
            "/",
            i_checked,
            "max_rel=",
            i_max_rel,
        )

    print()


def main() raises:
    print("=== NN Layer Gradcheck (CPU) ===")
    print()

    # ── Group A: Leaf layers with parameters ─────────────────
    print("--- Leaf layers (with params) ---")
    cpu_gradcheck[Linear[8, 4]]("Linear[8,4]")
    cpu_gradcheck[Linear[32, 1]]("Linear[32,1] (small output)")
    cpu_gradcheck[LinearReLU[16, 8]]("LinearReLU[16,8]")
    cpu_gradcheck[LinearTanh[8, 4]]("LinearTanh[8,4]")
    cpu_gradcheck[LayerNorm[16]]("LayerNorm[16]")
    cpu_gradcheck[Conv2DReLU[2, 4, 3, 1, 1, 5, 5]](
        "Conv2DReLU[2,4,3x3,5x5]"
    )
    cpu_gradcheck[Conv2DLayer[2, 4, 3, 1, 1, 5, 5]](
        "Conv2DLayer[2,4,3x3,5x5]"
    )

    # ── Group B: Param-free activations (grad_input only) ────
    print("--- Activations (param-free) ---")
    cpu_gradcheck[ReLU[8]]("ReLU[8]")
    cpu_gradcheck[Tanh[8]]("Tanh[8]")
    cpu_gradcheck[Sigmoid[8]]("Sigmoid[8]")
    cpu_gradcheck[Softmax[8]]("Softmax[8]")
    cpu_gradcheck[Mish[8]]("Mish[8]")

    # ── Group C: Combinators ─────────────────────────────────
    print("--- Combinators ---")
    cpu_gradcheck[Sequential[LinearReLU[8, 6], Linear[6, 4]]](
        "Sequential[LinearReLU[8,6], Linear[6,4]]"
    )
    cpu_gradcheck[Parallel[Linear[8, 4], Linear[8, 1]]](
        "Parallel[Linear(8->4), Linear(8->1)]"
    )
    cpu_gradcheck[Parallel[Linear[8, 4], Linear[8, 4]]](
        "Parallel[Linear(8->4), Linear(8->4)] (same size)"
    )
    cpu_gradcheck[Residual[LinearReLU[8, 8]]](
        "Residual[LinearReLU[8,8]]"
    )
    cpu_gradcheck[Repeat[2, LinearReLU[8, 8]]](
        "Repeat[2, LinearReLU[8,8]]"
    )
    cpu_gradcheck[SkipConcat[Linear[8, 4]]](
        "SkipConcat[Linear[8,4]]"
    )
    cpu_gradcheck[DualPath[Linear[8, 4], Linear[8, 1]]](
        "DualPath[Linear(8->4), Linear(8->1)]"
    )
    cpu_gradcheck[SplitApply[Linear[4, 3], Linear[4, 2], 4]](
        "SplitApply[Linear(4->3), Linear(4->2), split=4]"
    )
    cpu_gradcheck[FanOut[Linear[8, 4], 2]](
        "FanOut[Linear[8,4], N=2]"
    )

    # ── Group D: ResBlocks ─────────────────────────────────────
    print("--- ResBlocks ---")
    cpu_gradcheck[ResBlockConv2D[4, 3, 1, 5, 5]](
        "ResBlockConv2D[4ch,3x3,5x5]"
    )
    # BN gradcheck: eps=1e-3, tol=5% (BN batch stats add finite-diff noise)
    cpu_gradcheck[ResBlockConv2DBN[4, 3, 1, 5, 5]](
        "ResBlockConv2DBN[4ch,3x3,5x5]", tol=0.05,
    )

    # ── Group E: Realistic architectures ─────────────────────
    print("--- Realistic architectures ---")

    # TicTacToe-like MLP dual-head
    comptime MLP_DualHead = Sequential[
        LinearReLU[27, 64],
        Parallel[Linear[64, 9], Linear[64, 1]],
    ]
    cpu_gradcheck[MLP_DualHead]("MLP dual-head (TicTacToe-like)")

    # Conv trunk + FC dual-head (AlphaZero-like, small)
    comptime Conv_DualHead = Sequential[
        Conv2DReLU[2, 4, 3, 1, 1, 5, 5],
        Parallel[
            Sequential[FlattenLayer[4 * 5 * 5], Linear[100, 7]],
            Sequential[FlattenLayer[4 * 5 * 5], LinearReLU[100, 16], Linear[16, 1]],
        ],
    ]
    cpu_gradcheck[Conv_DualHead]("Conv+FC dual-head (AlphaZero-like)")

    print("=== Done ===")
