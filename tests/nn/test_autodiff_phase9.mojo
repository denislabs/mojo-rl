"""Phase 9 verification tests for Composite Model Library.

Pre-built model architectures using existing primitives and combinators.
No new DiffOps — purely compositional.

Tests:
  9.1 ResNet variants (ResBlock, ResNet)
  9.2 Multi-head architectures (Parallel + Dense)
  9.3 CNN architectures (Conv2D + Pool + Dense)
  9.4 Transformer architectures (Attention + FFN + Repeat)

Run with:
    pixi run mojo run -I . tests/test_autodiff_phase9.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs, sqrt, exp

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import (
    AutoDiffChain,
    AutoFused,
    Dense,
    DenseReLU,
    DenseTanh,
    DenseSigmoid,
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    LayerNormOp,
    Flatten,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    Embedding,
    ScaledDotProductAttention,
    Residual,
    Parallel,
    Repeat,
)
from mojo_rl.nn.model.sequential import Sequential
from layout import Layout, LayoutTensor


fn print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


fn check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


fn make_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(0)
    return lst^


fn make_rand_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(Scalar[dtype](random_float64(-0.5, 0.5)))
    return lst^


fn max_diff(a: List[Scalar[dtype]], b: List[Scalar[dtype]], n: Int) -> Float64:
    var md: Float64 = 0
    for i in range(n):
        var d = math_abs(Float64(a[i]) - Float64(b[i]))
        if d > md:
            md = d
    return md


# =============================================================================
# 9.1 ResNet Variants
# =============================================================================

# ResBlock: Residual[Sequential[DenseReLU[dim,dim], Dense[dim,dim]]]
comptime ResBlock8 = Residual[Sequential[DenseReLU[8, 8], Dense[8, 8]]]

# ResNet: DenseReLU[in,dim] -> Repeat[depth, ResBlock] -> Dense[dim, out]
comptime ResNet_2_8_1_2 = Sequential[
    DenseReLU[2, 8],
    Repeat[2, ResBlock8],
    Dense[8, 1],
]


fn test_resblock_dims() -> Int:
    print_header("9.1a ResBlock dimension checks")
    var fails = 0

    check(ResBlock8.IN_DIM == 8, "ResBlock8 IN_DIM == 8", fails)
    check(ResBlock8.OUT_DIM == 8, "ResBlock8 OUT_DIM == 8", fails)

    # ResBlock params = DenseReLU[8,8] + Dense[8,8]
    # DenseReLU[8,8]: MatMul(8*8=64) + BiasAdd(8) + ReLU(0) = 72
    # Dense[8,8]: MatMul(8*8=64) + BiasAdd(8) = 72
    # Total inner = 144
    comptime expected_inner_params = 8 * 8 + 8 + 8 * 8 + 8
    check(
        ResBlock8.PARAM_SIZE == expected_inner_params,
        "ResBlock8 PARAM_SIZE = " + String(ResBlock8.PARAM_SIZE),
        fails,
    )

    return fails


fn test_resnet_dims() -> Int:
    print_header("9.1b ResNet[2, 8, 1, depth=2] dimension checks")
    var fails = 0

    check(ResNet_2_8_1_2.IN_DIM == 2, "ResNet IN_DIM == 2", fails)
    check(ResNet_2_8_1_2.OUT_DIM == 1, "ResNet OUT_DIM == 1", fails)

    # DenseReLU[2,8] params: 2*8 + 8 = 24
    # Repeat[2, ResBlock8] params: same as ResBlock8 (shared weights) = 144
    # Dense[8,1] params: 8*1 + 1 = 9
    comptime dr_params = 2 * 8 + 8
    comptime rb_params = 8 * 8 + 8 + 8 * 8 + 8
    comptime d_params = 8 * 1 + 1
    check(
        ResNet_2_8_1_2.PARAM_SIZE == dr_params + rb_params + d_params,
        "ResNet PARAM_SIZE = "
        + String(ResNet_2_8_1_2.PARAM_SIZE)
        + " (expected "
        + String(dr_params + rb_params + d_params)
        + ")",
        fails,
    )

    return fails


fn test_resnet_forward_backward() -> Int:
    print_header("9.1c ResNet forward + backward (sanity)")
    var fails = 0
    seed(42)

    comptime M = ResNet_2_8_1_2
    comptime BATCH = 4

    var inp = make_rand_list(BATCH * M.IN_DIM)
    var params = make_rand_list(M.PARAM_SIZE)
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var out_data = make_list(BATCH * M.OUT_DIM)
    var cache_data = make_list(BATCH * M.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    M.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_output = False
    for i in range(BATCH * M.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_output = True
            break
    check(has_output, "ResNet forward produces non-zero output", fails)

    # Backward
    var go_data = make_rand_list(BATCH * M.OUT_DIM)
    var gi_data = make_list(BATCH * M.IN_DIM)
    var gp_data = make_list(M.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    M.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gi = False
    for i in range(BATCH * M.IN_DIM):
        if math_abs(Float64(gi_data[i])) > 1e-10:
            has_gi = True
            break
    check(has_gi, "ResNet backward produces non-zero grad_input", fails)

    var has_gp = False
    for i in range(M.PARAM_SIZE):
        if math_abs(Float64(gp_data[i])) > 1e-10:
            has_gp = True
            break
    check(has_gp, "ResNet backward produces non-zero grad_params", fails)

    return fails


fn test_resnet_xor_training() -> Int:
    print_header("9.1d ResNet XOR training convergence")
    var fails = 0

    comptime M = ResNet_2_8_1_2
    comptime BATCH = 4

    var inp_data = List[Scalar[dtype]](capacity=8)
    inp_data.append(0)
    inp_data.append(0)
    inp_data.append(0)
    inp_data.append(1)
    inp_data.append(1)
    inp_data.append(0)
    inp_data.append(1)
    inp_data.append(1)

    var target_data = List[Scalar[dtype]](capacity=4)
    target_data.append(0)
    target_data.append(1)
    target_data.append(1)
    target_data.append(0)

    seed(100)
    var params = make_rand_list(M.PARAM_SIZE)
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var grads = make_list(M.PARAM_SIZE)

    comptime LR: Float64 = 0.01
    comptime EPOCHS = 5000

    var final_loss: Float64 = 999.0

    for epoch in range(EPOCHS):
        for i in range(M.PARAM_SIZE):
            grads[i] = 0

        var out_data = make_list(BATCH * M.OUT_DIM)
        var cache_data = make_list(BATCH * M.CACHE_SIZE)

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
        ](inp_data.unsafe_ptr())
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](out_data.unsafe_ptr())
        var p_t = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var c_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
        ](cache_data.unsafe_ptr())

        M.forward[BATCH](inp_t, out_t, p_t, c_t)

        var loss: Float64 = 0.0
        var go_data = make_list(BATCH * M.OUT_DIM)
        for b_idx in range(BATCH):
            var diff = Float64(out_data[b_idx]) - Float64(target_data[b_idx])
            loss += diff * diff
            go_data[b_idx] = Scalar[dtype](2.0 * diff / Float64(BATCH))
        loss /= Float64(BATCH)

        var gi_data = make_list(BATCH * M.IN_DIM)
        var go_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](go_data.unsafe_ptr())
        var gi_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
        ](gi_data.unsafe_ptr())
        var g_t = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](grads.unsafe_ptr())

        M.backward[BATCH](go_t, gi_t, p_t, c_t, g_t)

        for i in range(M.PARAM_SIZE):
            params[i] = params[i] - Scalar[dtype](LR) * grads[i]

        final_loss = loss

    print("  Final loss: " + String(final_loss))
    check(final_loss < 0.05, "ResNet XOR converges (loss < 0.05)", fails)

    return fails


fn test_resnet_gradient_flow() -> Int:
    print_header("9.1e ResNet gradient flow through residual connections")
    var fails = 0
    seed(42)

    # Use a deeper ResNet: 4 residual blocks
    comptime ResBlock4 = Residual[Sequential[DenseReLU[4, 4], Dense[4, 4]]]
    comptime DeepResNet = Sequential[
        DenseReLU[2, 4],
        Repeat[4, ResBlock4],
        Dense[4, 1],
    ]

    comptime BATCH = 2
    var inp = make_rand_list(BATCH * DeepResNet.IN_DIM)
    var params = make_rand_list(DeepResNet.PARAM_SIZE)
    for i in range(DeepResNet.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var out_data = make_list(BATCH * DeepResNet.OUT_DIM)
    var cache_data = make_list(BATCH * DeepResNet.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DeepResNet.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DeepResNet.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(DeepResNet.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DeepResNet.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    DeepResNet.forward[BATCH](inp_t, out_t, p_t, c_t)

    var go_data = make_rand_list(BATCH * DeepResNet.OUT_DIM)
    var gi_data = make_list(BATCH * DeepResNet.IN_DIM)
    var gp_data = make_list(DeepResNet.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DeepResNet.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DeepResNet.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(DeepResNet.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    DeepResNet.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    # Check gradient magnitude at input (should NOT vanish through 4 residual blocks)
    var gi_norm: Float64 = 0.0
    for i in range(BATCH * DeepResNet.IN_DIM):
        gi_norm += Float64(gi_data[i]) * Float64(gi_data[i])
    gi_norm = sqrt(gi_norm)

    var go_norm: Float64 = 0.0
    for i in range(BATCH * DeepResNet.OUT_DIM):
        go_norm += Float64(go_data[i]) * Float64(go_data[i])
    go_norm = sqrt(go_norm)

    # Ratio should be reasonable (not vanishing)
    var ratio = gi_norm / (go_norm + 1e-10)
    print("  grad_input norm = " + String(gi_norm))
    print("  grad_output norm = " + String(go_norm))
    print("  ratio (gi/go) = " + String(ratio))
    check(
        ratio > 1e-4,
        "Gradient ratio > 1e-4 (no vanishing gradients, ratio="
        + String(ratio)
        + ")",
        fails,
    )

    return fails


# =============================================================================
# 9.2 Multi-Head Architectures
# =============================================================================

# MultiHead: Parallel[Dense[2,3], Dense[2,2]]
comptime MultiHead_2 = Parallel[Dense[2, 3], Dense[2, 2]]

# MultiHeadClassifier: MultiHead -> DenseReLU -> Dense
comptime MultiHeadClassifier_2_1 = Sequential[
    MultiHead_2,
    DenseReLU[5, 4],
    Dense[4, 1],
]


fn test_multihead_dims() -> Int:
    print_header("9.2a MultiHead dimension checks")
    var fails = 0

    check(MultiHead_2.IN_DIM == 2, "MultiHead IN_DIM == 2", fails)
    check(MultiHead_2.OUT_DIM == 5, "MultiHead OUT_DIM = 3+2 = 5", fails)

    check(MultiHeadClassifier_2_1.IN_DIM == 2, "Classifier IN_DIM == 2", fails)
    check(
        MultiHeadClassifier_2_1.OUT_DIM == 1, "Classifier OUT_DIM == 1", fails
    )

    return fails


fn test_multihead_forward_backward() -> Int:
    print_header("9.2b MultiHeadClassifier forward + backward")
    var fails = 0
    seed(42)

    comptime M = MultiHeadClassifier_2_1
    comptime BATCH = 4

    var inp = make_rand_list(BATCH * M.IN_DIM)
    var params = make_rand_list(M.PARAM_SIZE)
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var out_data = make_list(BATCH * M.OUT_DIM)
    var cache_data = make_list(BATCH * M.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    M.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_output = False
    for i in range(BATCH * M.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_output = True
            break
    check(has_output, "MultiHeadClassifier forward non-zero", fails)

    # Backward
    var go_data = make_rand_list(BATCH * M.OUT_DIM)
    var gi_data = make_list(BATCH * M.IN_DIM)
    var gp_data = make_list(M.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    M.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gi = False
    for i in range(BATCH * M.IN_DIM):
        if math_abs(Float64(gi_data[i])) > 1e-10:
            has_gi = True
            break
    check(has_gi, "MultiHeadClassifier backward grad_input non-zero", fails)

    var has_gp = False
    for i in range(M.PARAM_SIZE):
        if math_abs(Float64(gp_data[i])) > 1e-10:
            has_gp = True
            break
    check(has_gp, "MultiHeadClassifier backward grad_params non-zero", fails)

    return fails


fn test_multihead_xor_training() -> Int:
    print_header("9.2c MultiHeadClassifier XOR training convergence")
    var fails = 0

    # Use wider heads for more capacity
    comptime MH = Parallel[DenseReLU[2, 4], DenseReLU[2, 4]]
    comptime M = Sequential[MH, DenseReLU[8, 8], Dense[8, 1]]
    comptime BATCH = 4

    var inp_data = List[Scalar[dtype]](capacity=8)
    inp_data.append(0)
    inp_data.append(0)
    inp_data.append(0)
    inp_data.append(1)
    inp_data.append(1)
    inp_data.append(0)
    inp_data.append(1)
    inp_data.append(1)

    var target_data = List[Scalar[dtype]](capacity=4)
    target_data.append(0)
    target_data.append(1)
    target_data.append(1)
    target_data.append(0)

    seed(50)
    var params = make_rand_list(M.PARAM_SIZE)
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var grads = make_list(M.PARAM_SIZE)

    comptime LR: Float64 = 0.02
    comptime EPOCHS = 5000

    var final_loss: Float64 = 999.0

    for epoch in range(EPOCHS):
        for i in range(M.PARAM_SIZE):
            grads[i] = 0

        var out_data = make_list(BATCH * M.OUT_DIM)
        var cache_data = make_list(BATCH * M.CACHE_SIZE)

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
        ](inp_data.unsafe_ptr())
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](out_data.unsafe_ptr())
        var p_t = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var c_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
        ](cache_data.unsafe_ptr())

        M.forward[BATCH](inp_t, out_t, p_t, c_t)

        var loss: Float64 = 0.0
        var go_data = make_list(BATCH * M.OUT_DIM)
        for b_idx in range(BATCH):
            var diff = Float64(out_data[b_idx]) - Float64(target_data[b_idx])
            loss += diff * diff
            go_data[b_idx] = Scalar[dtype](2.0 * diff / Float64(BATCH))
        loss /= Float64(BATCH)

        var gi_data = make_list(BATCH * M.IN_DIM)
        var go_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](go_data.unsafe_ptr())
        var gi_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
        ](gi_data.unsafe_ptr())
        var g_t = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](grads.unsafe_ptr())

        M.backward[BATCH](go_t, gi_t, p_t, c_t, g_t)

        for i in range(M.PARAM_SIZE):
            params[i] = params[i] - Scalar[dtype](LR) * grads[i]

        final_loss = loss

    print("  Final loss: " + String(final_loss))
    check(final_loss < 0.05, "MultiHead XOR converges (loss < 0.05)", fails)

    return fails


# 3-branch multi-head variant
fn test_multihead_3branch() -> Int:
    print_header(
        "9.2d 3-branch MultiHead[Dense[4,3], DenseReLU[4,2], DenseTanh[4,1]]"
    )
    var fails = 0
    seed(42)

    comptime MH3 = Parallel[Dense[4, 3], DenseReLU[4, 2], DenseTanh[4, 1]]
    check(MH3.IN_DIM == 4, "3-branch IN_DIM == 4", fails)
    check(MH3.OUT_DIM == 6, "3-branch OUT_DIM = 3+2+1 = 6", fails)

    comptime M = Sequential[MH3, DenseReLU[6, 4], Dense[4, 1]]
    check(M.IN_DIM == 4, "3-branch classifier IN_DIM == 4", fails)
    check(M.OUT_DIM == 1, "3-branch classifier OUT_DIM == 1", fails)

    comptime BATCH = 2
    var inp = make_rand_list(BATCH * M.IN_DIM)
    var params = make_rand_list(M.PARAM_SIZE)
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.1
    var out_data = make_list(BATCH * M.OUT_DIM)
    var cache_data = make_list(BATCH * M.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    M.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_output = False
    for i in range(BATCH * M.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_output = True
            break
    check(has_output, "3-branch forward non-zero", fails)

    # Backward
    var go_data = make_rand_list(BATCH * M.OUT_DIM)
    var gi_data = make_list(BATCH * M.IN_DIM)
    var gp_data = make_list(M.PARAM_SIZE)
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())
    M.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gi = False
    for i in range(BATCH * M.IN_DIM):
        if math_abs(Float64(gi_data[i])) > 1e-10:
            has_gi = True
            break
    check(has_gi, "3-branch backward grad_input non-zero", fails)

    return fails


# =============================================================================
# 9.3 CNN Architectures
# =============================================================================

# Simple CNN: Conv2D -> ReLU -> MaxPool -> Flatten -> Dense
comptime SimpleCNN = Sequential[
    AutoDiffChain[
        Conv2D[1, 4, 3, 1, 0, 8, 8],
        ReLUOp[4 * 6 * 6],
        MaxPool2D[4, 6, 6, 2],
    ],
    AutoDiffChain[Flatten[4 * 3 * 3], MatMul[36, 10], BiasAdd[10]],
]

# LeNet-5 style (simplified for small input)
comptime LeNet = Sequential[
    AutoDiffChain[
        Conv2D[1, 6, 3, 1, 0, 8, 8],
        ReLUOp[6 * 6 * 6],
        MaxPool2D[6, 6, 6, 2],
    ],
    AutoDiffChain[
        Conv2D[6, 16, 3, 1, 0, 3, 3],
        ReLUOp[16 * 1 * 1],
    ],
    AutoDiffChain[
        Flatten[16],
        MatMul[16, 10],
        BiasAdd[10],
    ],
]


fn test_cnn_dims() -> Int:
    print_header("9.3a SimpleCNN dimension checks (1ch, 8x8 input)")
    var fails = 0

    check(SimpleCNN.IN_DIM == 1 * 8 * 8, "SimpleCNN IN_DIM = 64", fails)
    check(SimpleCNN.OUT_DIM == 10, "SimpleCNN OUT_DIM = 10", fails)

    return fails


fn test_lenet_dims() -> Int:
    print_header("9.3b LeNet dimension checks (1ch, 8x8 input)")
    var fails = 0

    check(LeNet.IN_DIM == 1 * 8 * 8, "LeNet IN_DIM = 64", fails)
    check(LeNet.OUT_DIM == 10, "LeNet OUT_DIM = 10", fails)

    return fails


fn test_cnn_forward_backward() -> Int:
    print_header("9.3c SimpleCNN forward + backward")
    var fails = 0
    seed(42)

    comptime M = SimpleCNN
    comptime BATCH = 2

    var inp = make_rand_list(BATCH * M.IN_DIM)
    var params = make_rand_list(M.PARAM_SIZE)
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var out_data = make_list(BATCH * M.OUT_DIM)
    var cache_data = make_list(BATCH * M.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    M.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_output = False
    for i in range(BATCH * M.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_output = True
            break
    check(has_output, "SimpleCNN forward produces non-zero output", fails)

    # Backward
    var go_data = make_rand_list(BATCH * M.OUT_DIM)
    var gi_data = make_list(BATCH * M.IN_DIM)
    var gp_data = make_list(M.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    M.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gi = False
    for i in range(BATCH * M.IN_DIM):
        if math_abs(Float64(gi_data[i])) > 1e-10:
            has_gi = True
            break
    check(has_gi, "SimpleCNN backward grad_input non-zero", fails)

    var has_gp = False
    for i in range(M.PARAM_SIZE):
        if math_abs(Float64(gp_data[i])) > 1e-10:
            has_gp = True
            break
    check(has_gp, "SimpleCNN backward grad_params non-zero", fails)

    return fails


fn test_lenet_forward_backward() -> Int:
    print_header("9.3d LeNet forward + backward")
    var fails = 0
    seed(99)

    comptime M = LeNet
    comptime BATCH = 2

    var inp = make_rand_list(BATCH * M.IN_DIM)
    var params = make_rand_list(M.PARAM_SIZE)
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var out_data = make_list(BATCH * M.OUT_DIM)
    var cache_data = make_list(BATCH * M.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    M.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_output = False
    for i in range(BATCH * M.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_output = True
            break
    check(has_output, "LeNet forward non-zero", fails)

    # Backward
    var go_data = make_rand_list(BATCH * M.OUT_DIM)
    var gi_data = make_list(BATCH * M.IN_DIM)
    var gp_data = make_list(M.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    M.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gp = False
    for i in range(M.PARAM_SIZE):
        if math_abs(Float64(gp_data[i])) > 1e-10:
            has_gp = True
            break
    check(has_gp, "LeNet backward grad_params non-zero", fails)

    return fails


fn test_cnn_synthetic_training() -> Int:
    print_header("9.3e SimpleCNN training on synthetic 8x8 data")
    var fails = 0

    # 2-class classification on random 8x8 images
    # Class 0: mostly negative pixels, Class 1: mostly positive
    comptime M = SimpleCNN
    comptime BATCH = 4

    seed(77)

    # Create simple training data
    var inp_data = make_list(BATCH * M.IN_DIM)
    var target_data = make_list(BATCH * M.OUT_DIM)

    # Sample 0: class 0 (target = [1,0,...])
    for i in range(M.IN_DIM):
        inp_data[0 * M.IN_DIM + i] = Scalar[dtype](random_float64(-1.0, -0.1))
    target_data[0 * M.OUT_DIM + 0] = 1.0

    # Sample 1: class 1 (target = [0,1,...])
    for i in range(M.IN_DIM):
        inp_data[1 * M.IN_DIM + i] = Scalar[dtype](random_float64(0.1, 1.0))
    target_data[1 * M.OUT_DIM + 1] = 1.0

    # Sample 2: class 0
    for i in range(M.IN_DIM):
        inp_data[2 * M.IN_DIM + i] = Scalar[dtype](random_float64(-1.0, -0.1))
    target_data[2 * M.OUT_DIM + 0] = 1.0

    # Sample 3: class 1
    for i in range(M.IN_DIM):
        inp_data[3 * M.IN_DIM + i] = Scalar[dtype](random_float64(0.1, 1.0))
    target_data[3 * M.OUT_DIM + 1] = 1.0

    var params = make_rand_list(M.PARAM_SIZE)
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.05

    var grads = make_list(M.PARAM_SIZE)

    comptime LR: Float64 = 0.005
    comptime EPOCHS = 500

    var final_loss: Float64 = 999.0

    for epoch in range(EPOCHS):
        for i in range(M.PARAM_SIZE):
            grads[i] = 0

        var out_data = make_list(BATCH * M.OUT_DIM)
        var cache_data = make_list(BATCH * M.CACHE_SIZE)

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
        ](inp_data.unsafe_ptr())
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](out_data.unsafe_ptr())
        var p_t = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var c_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
        ](cache_data.unsafe_ptr())

        M.forward[BATCH](inp_t, out_t, p_t, c_t)

        # MSE loss
        var loss: Float64 = 0.0
        var go_data = make_list(BATCH * M.OUT_DIM)
        for b_idx in range(BATCH):
            for o_idx in range(M.OUT_DIM):
                var idx = b_idx * M.OUT_DIM + o_idx
                var diff = Float64(out_data[idx]) - Float64(target_data[idx])
                loss += diff * diff
                go_data[idx] = Scalar[dtype](
                    2.0 * diff / Float64(BATCH * M.OUT_DIM)
                )
        loss /= Float64(BATCH * M.OUT_DIM)

        var gi_data = make_list(BATCH * M.IN_DIM)
        var go_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](go_data.unsafe_ptr())
        var gi_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
        ](gi_data.unsafe_ptr())
        var g_t = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](grads.unsafe_ptr())

        M.backward[BATCH](go_t, gi_t, p_t, c_t, g_t)

        for i in range(M.PARAM_SIZE):
            params[i] = params[i] - Scalar[dtype](LR) * grads[i]

        final_loss = loss

    print("  Final loss: " + String(final_loss))
    # Loss should decrease from initial (convergence doesn't need to be perfect)
    check(final_loss < 0.3, "CNN training loss decreased (loss < 0.3)", fails)

    return fails


# =============================================================================
# 9.4 Transformer Architectures
# =============================================================================


fn test_transformer_composite_dims() -> Int:
    print_header("9.4a Transformer composite dimension checks")
    var fails = 0

    comptime DIM = 4
    comptime SEQ = 2
    comptime HEADS = 2
    comptime FF_DIM = 8
    comptime SD = SEQ * DIM

    # FFN = Sequential[DenseReLU[SD, FF_DIM], Dense[FF_DIM, SD]]
    comptime FFN = Sequential[DenseReLU[SD, FF_DIM], Dense[FF_DIM, SD]]
    check(FFN.IN_DIM == SD, "FFN IN_DIM = " + String(SD), fails)
    check(FFN.OUT_DIM == SD, "FFN OUT_DIM = " + String(SD), fails)

    # TransformerLayer = Sequential[ResAttn, ResFFN]
    comptime AttnInner = AutoDiffChain[
        MatMul[SD, SD * 3],
        BiasAdd[SD * 3],
        ScaledDotProductAttention[DIM, HEADS, SEQ],
    ]
    comptime ResAttn = Residual[AttnInner]
    comptime ResFFN = Residual[FFN]
    comptime TLayer = Sequential[ResAttn, ResFFN]

    check(TLayer.IN_DIM == SD, "TransformerLayer IN_DIM = " + String(SD), fails)
    check(
        TLayer.OUT_DIM == SD, "TransformerLayer OUT_DIM = " + String(SD), fails
    )

    # TransformerEncoder = Repeat[2, TransformerLayer]
    comptime Encoder = Repeat[2, TLayer]
    check(Encoder.IN_DIM == SD, "Encoder IN_DIM = " + String(SD), fails)
    check(Encoder.OUT_DIM == SD, "Encoder OUT_DIM = " + String(SD), fails)
    check(
        Encoder.PARAM_SIZE == TLayer.PARAM_SIZE,
        "Encoder PARAM_SIZE = TLayer PARAM_SIZE (shared) = "
        + String(Encoder.PARAM_SIZE),
        fails,
    )
    check(
        Encoder.CACHE_SIZE == TLayer.CACHE_SIZE * 2,
        "Encoder CACHE_SIZE = 2x layer = " + String(Encoder.CACHE_SIZE),
        fails,
    )

    return fails


fn test_tiny_gpt() -> Int:
    print_header("9.4b Tiny GPT composite: Embedding + Encoder + Dense")
    var fails = 0
    seed(42)

    # Tiny GPT: vocab=8, dim=4, heads=2, ff=8, seq=2, layers=2
    comptime VOCAB = 8
    comptime DIM = 4
    comptime SEQ = 2
    comptime HEADS = 2
    comptime FF_DIM = 8
    comptime SD = SEQ * DIM

    comptime AttnInner = AutoDiffChain[
        MatMul[SD, SD * 3],
        BiasAdd[SD * 3],
        ScaledDotProductAttention[DIM, HEADS, SEQ],
    ]
    comptime ResAttn = Residual[AttnInner]
    comptime FFNInner = Sequential[DenseReLU[SD, FF_DIM], Dense[FF_DIM, SD]]
    comptime ResFFN = Residual[FFNInner]
    comptime TLayer = Sequential[ResAttn, ResFFN]
    comptime Encoder = Repeat[2, TLayer]

    # Full model: Embedding[VOCAB, SD] -> Encoder -> Dense[SD, VOCAB]
    comptime EmbedLayer = AutoDiffChain[Embedding[VOCAB, SD]]
    comptime GPT = Sequential[
        EmbedLayer,
        Encoder,
        Dense[SD, VOCAB],
    ]

    check(GPT.IN_DIM == VOCAB, "GPT IN_DIM = VOCAB = " + String(VOCAB), fails)
    check(GPT.OUT_DIM == VOCAB, "GPT OUT_DIM = VOCAB = " + String(VOCAB), fails)
    print("  GPT PARAM_SIZE = " + String(GPT.PARAM_SIZE))
    print("  GPT CACHE_SIZE = " + String(GPT.CACHE_SIZE))

    # Forward pass with one-hot input
    comptime BATCH = 1
    var inp = make_list(BATCH * GPT.IN_DIM)
    # Token 3 as one-hot
    inp[3] = 1.0

    var params = make_rand_list(GPT.PARAM_SIZE)
    for i in range(GPT.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var out_data = make_list(BATCH * GPT.OUT_DIM)
    var cache_data = make_list(BATCH * GPT.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(GPT.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    GPT.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_output = False
    for i in range(BATCH * GPT.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_output = True
            break
    check(has_output, "Tiny GPT forward produces non-zero output", fails)

    # Backward
    var go_data = make_rand_list(BATCH * GPT.OUT_DIM)
    var gi_data = make_list(BATCH * GPT.IN_DIM)
    var gp_data = make_list(GPT.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, GPT.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(GPT.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    GPT.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gp = False
    for i in range(GPT.PARAM_SIZE):
        if math_abs(Float64(gp_data[i])) > 1e-10:
            has_gp = True
            break
    check(has_gp, "Tiny GPT backward produces non-zero grad_params", fails)

    return fails


# =============================================================================
# Main
# =============================================================================


fn main():
    print("=" * 70)
    print("Phase 9: Composite Model Library — Test Suite")
    print("=" * 70)

    var total_fails = 0

    # 9.1 ResNet
    total_fails += test_resblock_dims()
    total_fails += test_resnet_dims()
    total_fails += test_resnet_forward_backward()
    total_fails += test_resnet_xor_training()
    total_fails += test_resnet_gradient_flow()

    # 9.2 Multi-head
    total_fails += test_multihead_dims()
    total_fails += test_multihead_forward_backward()
    total_fails += test_multihead_xor_training()
    total_fails += test_multihead_3branch()

    # 9.3 CNN
    total_fails += test_cnn_dims()
    total_fails += test_lenet_dims()
    total_fails += test_cnn_forward_backward()
    total_fails += test_lenet_forward_backward()
    total_fails += test_cnn_synthetic_training()

    # 9.4 Transformer
    total_fails += test_transformer_composite_dims()
    total_fails += test_tiny_gpt()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TESTS PASSED")
    else:
        print(String(total_fails) + " TESTS FAILED")
    print("=" * 70)
