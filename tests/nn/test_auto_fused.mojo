"""Comprehensive test suite for AutoFused automatic fusion.

Run: cd mojo-rl && pixi run mojo run tests/test_auto_fused.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import (
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    MishOp,
    AutoDiffChain,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
    FusedMatMulBiasSigmoid,
    FusedMatMulBiasMish,
)
from mojo_rl.nn.autodiff.auto_fused import AutoFused
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
    var lst = List[Scalar[dtype]](capacity=size if size > 0 else 1)
    for _ in range(size if size > 0 else 1):
        lst.append(0)
    return lst^


fn make_rand_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(Scalar[dtype](random_float64(-1.0, 1.0)))
    return lst^


fn max_diff(a: List[Scalar[dtype]], b: List[Scalar[dtype]], n: Int) -> Float64:
    var md: Float64 = 0
    for i in range(n):
        var d = math_abs(Float64(a[i]) - Float64(b[i]))
        if d > md:
            md = d
    return md


# =============================================================================
# Test 1: Compile-time dimensions
# =============================================================================


fn test_dimensions() -> Int:
    print_header("Compile-time dimensions")
    var fails = 0

    # 5-op chain: M+B+R + M+B
    comptime AF = AutoFused[
        MatMul[2, 4], BiasAdd[4], ReLUOp[4], MatMul[4, 1], BiasAdd[1]
    ]
    check(AF.IN_DIM == 2, "IN_DIM = 2", fails)
    check(AF.OUT_DIM == 1, "OUT_DIM = 1", fails)
    check(AF.PARAM_SIZE == 17, "PARAM_SIZE = 17 (12 + 5)", fails)
    check(AF.CACHE_SIZE == 10, "CACHE_SIZE = 10 (6 + 4)", fails)
    check(AF.WORKSPACE_SIZE_PER_SAMPLE == 4, "WORKSPACE = 4", fails)

    # 8-op chain: M+B+R + M+B+T + M+B
    comptime AF2 = AutoFused[
        MatMul[2, 8],
        BiasAdd[8],
        ReLUOp[8],
        MatMul[8, 4],
        BiasAdd[4],
        TanhOp[4],
        MatMul[4, 1],
        BiasAdd[1],
    ]
    check(AF2.IN_DIM == 2, "8-op IN_DIM = 2", fails)
    check(AF2.OUT_DIM == 1, "8-op OUT_DIM = 1", fails)
    # FusedMBR[2,8]: ps=24, cs=10; FusedMBT[8,4]: ps=36, cs=12; FusedMB[4,1]: ps=5, cs=4
    check(AF2.PARAM_SIZE == 65, "8-op PARAM_SIZE = 65", fails)
    check(AF2.CACHE_SIZE == 26, "8-op CACHE_SIZE = 26 (10+12+4)", fails)
    check(
        AF2.WORKSPACE_SIZE_PER_SAMPLE == 12, "8-op WORKSPACE = 12 (8+4)", fails
    )

    # Single op (no fusion)
    comptime AF3 = AutoFused[MatMul[3, 5]]
    check(AF3.IN_DIM == 3, "single op IN_DIM = 3", fails)
    check(AF3.OUT_DIM == 5, "single op OUT_DIM = 5", fails)
    check(AF3.PARAM_SIZE == 15, "single op PARAM_SIZE = 15", fails)

    return fails


# =============================================================================
# Test 2: Forward numerical match — AutoFused vs AutoDiffChain[FusedOps]
# =============================================================================


fn test_forward_5op() -> Int:
    print_header("Forward: 5-op AutoFused vs AutoDiffChain[FusedMBR, FusedMB]")
    var fails = 0
    seed(42)

    comptime IN_D = 2
    comptime HID = 4
    comptime OUT_D = 1
    comptime BATCH = 4

    comptime AF = AutoFused[
        MatMul[IN_D, HID],
        BiasAdd[HID],
        ReLUOp[HID],
        MatMul[HID, OUT_D],
        BiasAdd[OUT_D],
    ]
    comptime Ref = AutoDiffChain[
        FusedMatMulBiasReLU[IN_D, HID], FusedMatMulBias[HID, OUT_D]
    ]

    # Verify dimensions match
    check(AF.PARAM_SIZE == Ref.PARAM_SIZE, "PARAM_SIZE match", fails)
    check(AF.CACHE_SIZE == Ref.CACHE_SIZE, "CACHE_SIZE match", fails)

    var params = make_rand_list(AF.PARAM_SIZE)
    var input_data = make_rand_list(BATCH * IN_D)

    # AutoFused forward
    var af_out = make_list(BATCH * OUT_D)
    var af_cache = make_list(BATCH * AF.CACHE_SIZE)
    var af_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var af_o = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](af_out.unsafe_ptr())
    var af_p = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var af_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, AF.CACHE_SIZE), MutAnyOrigin
    ](af_cache.unsafe_ptr())
    AF.forward[BATCH](af_inp, af_o, af_p, af_c)

    # Reference forward
    var ref_out = make_list(BATCH * OUT_D)
    var ref_cache = make_list(BATCH * Ref.CACHE_SIZE)
    var ref_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var ref_o = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](ref_out.unsafe_ptr())
    var ref_p = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_o, ref_p, ref_c)

    var md = max_diff(af_out, ref_out, BATCH * OUT_D)
    print("  Max forward diff:", md)
    check(md < 1e-7, "forward match < 1e-7", fails)

    return fails


# =============================================================================
# Test 3: Backward numerical match
# =============================================================================


fn test_backward_5op() -> Int:
    print_header("Backward: 5-op AutoFused vs AutoDiffChain[FusedMBR, FusedMB]")
    var fails = 0
    seed(123)

    comptime IN_D = 2
    comptime HID = 4
    comptime OUT_D = 1
    comptime BATCH = 4

    comptime AF = AutoFused[
        MatMul[IN_D, HID],
        BiasAdd[HID],
        ReLUOp[HID],
        MatMul[HID, OUT_D],
        BiasAdd[OUT_D],
    ]
    comptime Ref = AutoDiffChain[
        FusedMatMulBiasReLU[IN_D, HID], FusedMatMulBias[HID, OUT_D]
    ]

    var params = make_rand_list(AF.PARAM_SIZE)
    var input_data = make_rand_list(BATCH * IN_D)

    # Forward pass (both need cache for backward)
    var af_out = make_list(BATCH * OUT_D)
    var af_cache = make_list(BATCH * AF.CACHE_SIZE)
    var af_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var af_o = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](af_out.unsafe_ptr())
    var af_p = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var af_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, AF.CACHE_SIZE), MutAnyOrigin
    ](af_cache.unsafe_ptr())
    AF.forward[BATCH](af_inp, af_o, af_p, af_c)

    var ref_out = make_list(BATCH * OUT_D)
    var ref_cache = make_list(BATCH * Ref.CACHE_SIZE)
    var ref_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var ref_o = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](ref_out.unsafe_ptr())
    var ref_p = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_o, ref_p, ref_c)

    # Backward pass
    var grad_out_data = make_rand_list(BATCH * OUT_D)

    var af_gi = make_list(BATCH * IN_D)
    var af_grads = make_list(AF.PARAM_SIZE)
    var af_go = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](grad_out_data.unsafe_ptr())
    var af_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](af_gi.unsafe_ptr())
    var af_g = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](af_grads.unsafe_ptr())
    AF.backward[BATCH](af_go, af_gi_t, af_p, af_c, af_g)

    var ref_gi = make_list(BATCH * IN_D)
    var ref_grads = make_list(Ref.PARAM_SIZE)
    var ref_go = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](grad_out_data.unsafe_ptr())
    var ref_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](ref_gi.unsafe_ptr())
    var ref_g = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](ref_grads.unsafe_ptr())
    Ref.backward[BATCH](ref_go, ref_gi_t, ref_p, ref_c, ref_g)

    var gi_diff = max_diff(af_gi, ref_gi, BATCH * IN_D)
    var gp_diff = max_diff(af_grads, ref_grads, AF.PARAM_SIZE)
    print("  Max grad_input diff:", gi_diff)
    print("  Max grad_params diff:", gp_diff)
    check(gi_diff < 1e-6, "grad_input match < 1e-6", fails)
    check(gp_diff < 1e-6, "grad_params match < 1e-6", fails)

    return fails


# =============================================================================
# Test 4: 8-op chain (mixed activations)
# =============================================================================


fn test_forward_8op() -> Int:
    print_header("Forward: 8-op M+B+R + M+B+T + M+B")
    var fails = 0
    seed(77)

    comptime IN_D = 3
    comptime H1 = 8
    comptime H2 = 4
    comptime OUT_D = 2
    comptime BATCH = 2

    comptime AF = AutoFused[
        MatMul[IN_D, H1],
        BiasAdd[H1],
        ReLUOp[H1],
        MatMul[H1, H2],
        BiasAdd[H2],
        TanhOp[H2],
        MatMul[H2, OUT_D],
        BiasAdd[OUT_D],
    ]
    comptime Ref = AutoDiffChain[
        FusedMatMulBiasReLU[IN_D, H1],
        FusedMatMulBiasTanh[H1, H2],
        FusedMatMulBias[H2, OUT_D],
    ]

    check(AF.PARAM_SIZE == Ref.PARAM_SIZE, "PARAM_SIZE match", fails)
    check(AF.CACHE_SIZE == Ref.CACHE_SIZE, "CACHE_SIZE match", fails)

    var params = make_rand_list(AF.PARAM_SIZE)
    var input_data = make_rand_list(BATCH * IN_D)

    var af_out = make_list(BATCH * OUT_D)
    var af_cache = make_list(BATCH * AF.CACHE_SIZE)
    var af_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var af_o = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](af_out.unsafe_ptr())
    var af_p = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var af_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, AF.CACHE_SIZE), MutAnyOrigin
    ](af_cache.unsafe_ptr())
    AF.forward[BATCH](af_inp, af_o, af_p, af_c)

    var ref_out = make_list(BATCH * OUT_D)
    var ref_cache = make_list(BATCH * Ref.CACHE_SIZE)
    var ref_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var ref_o = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](ref_out.unsafe_ptr())
    var ref_p = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_o, ref_p, ref_c)

    var md = max_diff(af_out, ref_out, BATCH * OUT_D)
    print("  Max forward diff:", md)
    check(md < 1e-6, "8-op forward match < 1e-6", fails)

    return fails


# =============================================================================
# Test 5: 8-op backward
# =============================================================================


fn test_backward_8op() -> Int:
    print_header("Backward: 8-op M+B+R + M+B+T + M+B")
    var fails = 0
    seed(88)

    comptime IN_D = 3
    comptime H1 = 8
    comptime H2 = 4
    comptime OUT_D = 2
    comptime BATCH = 2

    comptime AF = AutoFused[
        MatMul[IN_D, H1],
        BiasAdd[H1],
        ReLUOp[H1],
        MatMul[H1, H2],
        BiasAdd[H2],
        TanhOp[H2],
        MatMul[H2, OUT_D],
        BiasAdd[OUT_D],
    ]
    comptime Ref = AutoDiffChain[
        FusedMatMulBiasReLU[IN_D, H1],
        FusedMatMulBiasTanh[H1, H2],
        FusedMatMulBias[H2, OUT_D],
    ]

    var params = make_rand_list(AF.PARAM_SIZE)
    var input_data = make_rand_list(BATCH * IN_D)

    # Forward
    var af_out = make_list(BATCH * OUT_D)
    var af_cache = make_list(BATCH * AF.CACHE_SIZE)
    var af_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var af_o = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](af_out.unsafe_ptr())
    var af_p = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var af_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, AF.CACHE_SIZE), MutAnyOrigin
    ](af_cache.unsafe_ptr())
    AF.forward[BATCH](af_inp, af_o, af_p, af_c)

    var ref_out = make_list(BATCH * OUT_D)
    var ref_cache = make_list(BATCH * Ref.CACHE_SIZE)
    var ref_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var ref_o = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](ref_out.unsafe_ptr())
    var ref_p = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_o, ref_p, ref_c)

    # Backward
    var grad_out_data = make_rand_list(BATCH * OUT_D)

    var af_gi = make_list(BATCH * IN_D)
    var af_grads = make_list(AF.PARAM_SIZE)
    var af_go = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](grad_out_data.unsafe_ptr())
    var af_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](af_gi.unsafe_ptr())
    var af_g = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](af_grads.unsafe_ptr())
    AF.backward[BATCH](af_go, af_gi_t, af_p, af_c, af_g)

    var ref_gi = make_list(BATCH * IN_D)
    var ref_grads = make_list(Ref.PARAM_SIZE)
    var ref_go = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](grad_out_data.unsafe_ptr())
    var ref_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](ref_gi.unsafe_ptr())
    var ref_g = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](ref_grads.unsafe_ptr())
    Ref.backward[BATCH](ref_go, ref_gi_t, ref_p, ref_c, ref_g)

    var gi_diff = max_diff(af_gi, ref_gi, BATCH * IN_D)
    var gp_diff = max_diff(af_grads, ref_grads, AF.PARAM_SIZE)
    print("  Max grad_input diff:", gi_diff)
    print("  Max grad_params diff:", gp_diff)
    check(gi_diff < 1e-5, "8-op grad_input match < 1e-5", fails)
    check(gp_diff < 1e-5, "8-op grad_params match < 1e-5", fails)

    return fails


# =============================================================================
# Test 6: XOR training convergence
# =============================================================================


fn test_xor_training() -> Int:
    print_header("XOR training with AutoFused MLP")
    var fails = 0
    seed(42)

    comptime IN_D = 2
    comptime HID = 8
    comptime OUT_D = 1
    comptime BATCH = 4

    comptime M = AutoFused[
        MatMul[IN_D, HID],
        BiasAdd[HID],
        ReLUOp[HID],
        MatMul[HID, OUT_D],
        BiasAdd[OUT_D],
    ]

    # XOR data
    var input_data = List[Scalar[dtype]](capacity=BATCH * IN_D)
    # [0,0], [0,1], [1,0], [1,1]
    input_data.append(0)
    input_data.append(0)
    input_data.append(0)
    input_data.append(1)
    input_data.append(1)
    input_data.append(0)
    input_data.append(1)
    input_data.append(1)

    var target_data = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    target_data.append(0)  # 0 XOR 0
    target_data.append(1)  # 0 XOR 1
    target_data.append(1)  # 1 XOR 0
    target_data.append(0)  # 1 XOR 1

    # Init params
    var params = make_rand_list(M.PARAM_SIZE)
    # Scale down initial params
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.5

    var lr: Float64 = 0.05
    var final_loss: Float64 = 999.0

    for epoch in range(2000):
        var cache = make_list(BATCH * M.CACHE_SIZE)
        var output = make_list(BATCH * OUT_D)

        var inp = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
        ](input_data.unsafe_ptr())
        var out = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
        ](output.unsafe_ptr())
        var par = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var cch = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
        ](cache.unsafe_ptr())
        M.forward[BATCH](inp, out, par, cch)

        # MSE loss + grad
        var loss: Float64 = 0
        var grad_out_data = make_list(BATCH * OUT_D)
        for b in range(BATCH):
            var diff = Float64(output[b]) - Float64(target_data[b])
            loss += diff * diff
            grad_out_data[b] = Scalar[dtype](2.0 * diff / BATCH)
        loss /= BATCH

        # Backward
        var grad_input = make_list(BATCH * IN_D)
        var grads = make_list(M.PARAM_SIZE)
        var go = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
        ](grad_out_data.unsafe_ptr())
        var gi = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
        ](grad_input.unsafe_ptr())
        var g = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](grads.unsafe_ptr())
        M.backward[BATCH](go, gi, par, cch, g)

        # SGD update
        for i in range(M.PARAM_SIZE):
            params[i] = params[i] - Scalar[dtype](lr) * grads[i]

        final_loss = loss

    print("  Final loss:", final_loss)
    check(final_loss < 0.01, "XOR converges (loss < 0.01)", fails)

    return fails


# =============================================================================
# Test 7: Single op (no fusion) — passthrough
# =============================================================================


fn test_single_op() -> Int:
    print_header("Single op: AutoFused[MatMul[3,5]] passthrough")
    var fails = 0
    seed(99)

    comptime BATCH = 2
    comptime AF = AutoFused[MatMul[3, 5]]
    comptime Ref = AutoDiffChain[MatMul[3, 5]]

    check(AF.PARAM_SIZE == Ref.PARAM_SIZE, "PARAM_SIZE match", fails)
    check(AF.CACHE_SIZE == Ref.CACHE_SIZE, "CACHE_SIZE match", fails)
    check(AF.IN_DIM == 3, "IN_DIM = 3", fails)
    check(AF.OUT_DIM == 5, "OUT_DIM = 5", fails)

    var params = make_rand_list(AF.PARAM_SIZE)
    var input_data = make_rand_list(BATCH * 3)

    var af_out = make_list(BATCH * 5)
    var af_cache = make_list(BATCH * AF.CACHE_SIZE)
    var af_inp = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](
        input_data.unsafe_ptr()
    )
    var af_o = LayoutTensor[dtype, Layout.row_major(BATCH, 5), MutAnyOrigin](
        af_out.unsafe_ptr()
    )
    var af_p = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var af_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, AF.CACHE_SIZE), MutAnyOrigin
    ](af_cache.unsafe_ptr())
    AF.forward[BATCH](af_inp, af_o, af_p, af_c)

    var ref_out = make_list(BATCH * 5)
    var ref_cache = make_list(BATCH * Ref.CACHE_SIZE)
    var ref_inp = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](
        input_data.unsafe_ptr()
    )
    var ref_o = LayoutTensor[dtype, Layout.row_major(BATCH, 5), MutAnyOrigin](
        ref_out.unsafe_ptr()
    )
    var ref_p = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_o, ref_p, ref_c)

    var md = max_diff(af_out, ref_out, BATCH * 5)
    print("  Max forward diff:", md)
    check(md < 1e-7, "single op forward match", fails)

    return fails


# =============================================================================
# Test 8: M+B+S (sigmoid fusion)
# =============================================================================


fn test_sigmoid_fusion() -> Int:
    print_header("Sigmoid: AutoFused[M,B,S] vs FusedMatMulBiasSigmoid")
    var fails = 0
    seed(55)

    comptime BATCH = 3
    comptime AF = AutoFused[MatMul[4, 2], BiasAdd[2], SigmoidOp[2]]
    comptime Ref = AutoDiffChain[FusedMatMulBiasSigmoid[4, 2]]

    check(AF.PARAM_SIZE == Ref.PARAM_SIZE, "PARAM_SIZE match", fails)

    var params = make_rand_list(AF.PARAM_SIZE)
    var input_data = make_rand_list(BATCH * 4)

    var af_out = make_list(BATCH * 2)
    var af_cache = make_list(BATCH * AF.CACHE_SIZE)
    var af_inp = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](
        input_data.unsafe_ptr()
    )
    var af_o = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        af_out.unsafe_ptr()
    )
    var af_p = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var af_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, AF.CACHE_SIZE), MutAnyOrigin
    ](af_cache.unsafe_ptr())
    AF.forward[BATCH](af_inp, af_o, af_p, af_c)

    var ref_out = make_list(BATCH * 2)
    var ref_cache = make_list(BATCH * Ref.CACHE_SIZE)
    var ref_inp = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](
        input_data.unsafe_ptr()
    )
    var ref_o = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        ref_out.unsafe_ptr()
    )
    var ref_p = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_o, ref_p, ref_c)

    var md = max_diff(af_out, ref_out, BATCH * 2)
    print("  Max forward diff:", md)
    check(md < 1e-7, "sigmoid forward match", fails)

    return fails


# =============================================================================
# Test 9: Mish fusion
# =============================================================================


fn test_mish_fusion() -> Int:
    print_header("Mish: AutoFused[M,B,Mish] vs FusedMatMulBiasMish")
    var fails = 0
    seed(77)

    comptime BATCH = 3
    comptime AF = AutoFused[MatMul[4, 2], BiasAdd[2], MishOp[2]]
    comptime Ref = AutoDiffChain[FusedMatMulBiasMish[4, 2]]

    check(AF.PARAM_SIZE == Ref.PARAM_SIZE, "PARAM_SIZE match", fails)

    var params = make_rand_list(AF.PARAM_SIZE)
    var input_data = make_rand_list(BATCH * 4)

    # Forward
    var af_out = make_list(BATCH * 2)
    var af_cache = make_list(BATCH * AF.CACHE_SIZE)
    var af_inp = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](
        input_data.unsafe_ptr()
    )
    var af_o = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        af_out.unsafe_ptr()
    )
    var af_p = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var af_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, AF.CACHE_SIZE), MutAnyOrigin
    ](af_cache.unsafe_ptr())
    AF.forward[BATCH](af_inp, af_o, af_p, af_c)

    var ref_out = make_list(BATCH * 2)
    var ref_cache = make_list(BATCH * Ref.CACHE_SIZE)
    var ref_inp = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](
        input_data.unsafe_ptr()
    )
    var ref_o = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        ref_out.unsafe_ptr()
    )
    var ref_p = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_o, ref_p, ref_c)

    var md = max_diff(af_out, ref_out, BATCH * 2)
    print("  Max forward diff:", md)
    check(md < 1e-7, "mish forward match", fails)

    # Backward
    var af_grads = make_list(AF.PARAM_SIZE)
    var af_gi = make_list(BATCH * 4)
    var ref_grads = make_list(Ref.PARAM_SIZE)
    var ref_gi = make_list(BATCH * 4)
    var grad_out_data = make_rand_list(BATCH * 2)

    var af_go = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        grad_out_data.unsafe_ptr()
    )
    var af_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](
        af_gi.unsafe_ptr()
    )
    var af_g = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](af_grads.unsafe_ptr())
    AF.backward[BATCH](af_go, af_gi_t, af_p, af_c, af_g)

    var ref_go = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        grad_out_data.unsafe_ptr()
    )
    var ref_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 4), MutAnyOrigin
    ](ref_gi.unsafe_ptr())
    var ref_g = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](ref_grads.unsafe_ptr())
    Ref.backward[BATCH](ref_go, ref_gi_t, ref_p, ref_c, ref_g)

    var gi_diff = max_diff(af_gi, ref_gi, BATCH * 4)
    var gp_diff = max_diff(af_grads, ref_grads, AF.PARAM_SIZE)
    print("  Max grad_input diff:", gi_diff)
    print("  Max grad_params diff:", gp_diff)
    check(gi_diff < 1e-6, "mish grad_input match", fails)
    check(gp_diff < 1e-6, "mish grad_params match", fails)

    return fails


# =============================================================================
# Test 10: 11-op deep chain
# =============================================================================


fn test_deep_chain() -> Int:
    print_header("Deep: 11-op M+B+R x3 + M+B → 4 fused groups")
    var fails = 0
    seed(111)

    comptime BATCH = 2
    comptime AF = AutoFused[
        MatMul[3, 16],
        BiasAdd[16],
        ReLUOp[16],
        MatMul[16, 8],
        BiasAdd[8],
        ReLUOp[8],
        MatMul[8, 4],
        BiasAdd[4],
        ReLUOp[4],
        MatMul[4, 2],
        BiasAdd[2],
    ]
    comptime Ref = AutoDiffChain[
        FusedMatMulBiasReLU[3, 16],
        FusedMatMulBiasReLU[16, 8],
        FusedMatMulBiasReLU[8, 4],
        FusedMatMulBias[4, 2],
    ]

    check(AF.PARAM_SIZE == Ref.PARAM_SIZE, "PARAM_SIZE match", fails)
    check(AF.CACHE_SIZE == Ref.CACHE_SIZE, "CACHE_SIZE match", fails)

    var params = make_rand_list(AF.PARAM_SIZE)
    var input_data = make_rand_list(BATCH * 3)

    # Forward
    var af_out = make_list(BATCH * 2)
    var af_cache = make_list(BATCH * AF.CACHE_SIZE)
    var af_inp = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](
        input_data.unsafe_ptr()
    )
    var af_o = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        af_out.unsafe_ptr()
    )
    var af_p = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var af_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, AF.CACHE_SIZE), MutAnyOrigin
    ](af_cache.unsafe_ptr())
    AF.forward[BATCH](af_inp, af_o, af_p, af_c)

    var ref_out = make_list(BATCH * 2)
    var ref_cache = make_list(BATCH * Ref.CACHE_SIZE)
    var ref_inp = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](
        input_data.unsafe_ptr()
    )
    var ref_o = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        ref_out.unsafe_ptr()
    )
    var ref_p = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_c = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_o, ref_p, ref_c)

    var fwd_md = max_diff(af_out, ref_out, BATCH * 2)
    print("  Max forward diff:", fwd_md)
    check(fwd_md < 1e-6, "11-op forward match", fails)

    # Backward
    var grad_out_data = make_rand_list(BATCH * 2)
    var af_gi = make_list(BATCH * 3)
    var af_grads = make_list(AF.PARAM_SIZE)
    var af_go = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        grad_out_data.unsafe_ptr()
    )
    var af_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](
        af_gi.unsafe_ptr()
    )
    var af_g = LayoutTensor[
        dtype, Layout.row_major(AF.PARAM_SIZE), MutAnyOrigin
    ](af_grads.unsafe_ptr())
    AF.backward[BATCH](af_go, af_gi_t, af_p, af_c, af_g)

    var ref_gi = make_list(BATCH * 3)
    var ref_grads = make_list(Ref.PARAM_SIZE)
    var ref_go = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        grad_out_data.unsafe_ptr()
    )
    var ref_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3), MutAnyOrigin
    ](ref_gi.unsafe_ptr())
    var ref_g = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](ref_grads.unsafe_ptr())
    Ref.backward[BATCH](ref_go, ref_gi_t, ref_p, ref_c, ref_g)

    var gi_diff = max_diff(af_gi, ref_gi, BATCH * 3)
    var gp_diff = max_diff(af_grads, ref_grads, AF.PARAM_SIZE)
    print("  Max grad_input diff:", gi_diff)
    print("  Max grad_params diff:", gp_diff)
    check(gi_diff < 1e-5, "11-op grad_input match", fails)
    check(gp_diff < 1e-5, "11-op grad_params match", fails)

    return fails


# =============================================================================
# Main
# =============================================================================


fn main():
    print()
    var total_fails = 0
    total_fails += test_dimensions()
    total_fails += test_forward_5op()
    total_fails += test_backward_5op()
    total_fails += test_forward_8op()
    total_fails += test_backward_8op()
    total_fails += test_xor_training()
    total_fails += test_single_op()
    total_fails += test_sigmoid_fusion()
    total_fails += test_mish_fusion()
    total_fails += test_deep_chain()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TESTS PASSED")
    else:
        print("FAILURES:", total_fails)
    print("=" * 70)
