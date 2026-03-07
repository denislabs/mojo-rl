"""Tests for FusedChain production aliases and FusionAnalyzer.

Verifies that FusedChain aliases produce numerically identical results
to equivalent unfused AutoDiffChain compositions, for both forward and backward.

Run with:
    pixi run mojo run tests/test_fused_chain.mojo
"""

from nn.constants import dtype
from nn.autodiff import (
    AutoDiffChain,
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
    FusionAnalyzer,
    FusedChain,
)
from layout import Layout, LayoutTensor
from std.random import seed, random_float64
from std.math import abs as math_abs


# =============================================================================
# Helpers
# =============================================================================


fn rand_val() -> Scalar[dtype]:
    return Scalar[dtype](random_float64(-1.0, 1.0))


fn make_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(rand_val())
    return lst^


fn make_zeros(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(0)
    return lst^


fn max_diff(a: List[Scalar[dtype]], b: List[Scalar[dtype]], n: Int) -> Float64:
    var md: Float64 = 0
    for i in range(n):
        var d = math_abs(Float64(a[i]) - Float64(b[i]))
        if d > md:
            md = d
    return md


# =============================================================================
# Tests
# =============================================================================


fn test_one_layer_aliases() -> Int:
    print("\n" + "=" * 70)
    print("TEST: FusedChain one-layer aliases")
    print("=" * 70)
    var fails = 0
    comptime BATCH = 2

    # --- one_layer_relu[3,4] vs unfused ---
    comptime Fused1 = FusedChain.one_layer_relu[3, 4]
    comptime Unfused1 = AutoDiffChain[MatMul[3, 4], BiasAdd[4], ReLUOp[4]]
    comptime if Fused1.PARAM_SIZE != Unfused1.PARAM_SIZE:
        print("  FAIL: one_layer_relu PARAM_SIZE mismatch")
        fails += 1

    seed(42)
    var params1 = make_list(Fused1.PARAM_SIZE)
    var inp1 = make_list(BATCH * 3)

    var fo1 = make_zeros(BATCH * 4)
    var fc1 = make_zeros(BATCH * Fused1.CACHE_SIZE)
    var inp1_t = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](inp1.unsafe_ptr())
    var fo1_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](fo1.unsafe_ptr())
    var fp1_t = LayoutTensor[dtype, Layout.row_major(Fused1.PARAM_SIZE), MutAnyOrigin](params1.unsafe_ptr())
    var fc1_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused1.CACHE_SIZE), MutAnyOrigin](fc1.unsafe_ptr())
    Fused1.forward[BATCH](inp1_t, fo1_t, fp1_t, fc1_t)

    var uo1 = make_zeros(BATCH * 4)
    var uc1 = make_zeros(BATCH * Unfused1.CACHE_SIZE)
    var uo1_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](uo1.unsafe_ptr())
    var up1_t = LayoutTensor[dtype, Layout.row_major(Unfused1.PARAM_SIZE), MutAnyOrigin](params1.unsafe_ptr())
    var uc1_t = LayoutTensor[dtype, Layout.row_major(BATCH, Unfused1.CACHE_SIZE), MutAnyOrigin](uc1.unsafe_ptr())
    Unfused1.forward[BATCH](inp1_t, uo1_t, up1_t, uc1_t)

    var d1 = max_diff(fo1, uo1, BATCH * 4)
    if d1 < 1e-5:
        print("  PASS: one_layer_relu forward max_diff = " + String(d1))
    else:
        print("  FAIL: one_layer_relu forward max_diff = " + String(d1))
        fails += 1

    # --- one_layer_tanh[3,4] ---
    comptime Fused2 = FusedChain.one_layer_tanh[3, 4]
    comptime Unfused2 = AutoDiffChain[MatMul[3, 4], BiasAdd[4], TanhOp[4]]

    seed(42)
    var params2 = make_list(Fused2.PARAM_SIZE)
    var inp2 = make_list(BATCH * 3)

    var fo2 = make_zeros(BATCH * 4)
    var fc2 = make_zeros(BATCH * Fused2.CACHE_SIZE)
    var inp2_t = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](inp2.unsafe_ptr())
    var fo2_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](fo2.unsafe_ptr())
    var fp2_t = LayoutTensor[dtype, Layout.row_major(Fused2.PARAM_SIZE), MutAnyOrigin](params2.unsafe_ptr())
    var fc2_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused2.CACHE_SIZE), MutAnyOrigin](fc2.unsafe_ptr())
    Fused2.forward[BATCH](inp2_t, fo2_t, fp2_t, fc2_t)

    var uo2 = make_zeros(BATCH * 4)
    var uc2 = make_zeros(BATCH * Unfused2.CACHE_SIZE)
    var uo2_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](uo2.unsafe_ptr())
    var up2_t = LayoutTensor[dtype, Layout.row_major(Unfused2.PARAM_SIZE), MutAnyOrigin](params2.unsafe_ptr())
    var uc2_t = LayoutTensor[dtype, Layout.row_major(BATCH, Unfused2.CACHE_SIZE), MutAnyOrigin](uc2.unsafe_ptr())
    Unfused2.forward[BATCH](inp2_t, uo2_t, up2_t, uc2_t)

    var d2 = max_diff(fo2, uo2, BATCH * 4)
    if d2 < 1e-5:
        print("  PASS: one_layer_tanh forward max_diff = " + String(d2))
    else:
        print("  FAIL: one_layer_tanh forward max_diff = " + String(d2))
        fails += 1

    # --- one_layer_linear[3,4] ---
    comptime Fused3 = FusedChain.one_layer_linear[3, 4]
    comptime Unfused3 = AutoDiffChain[MatMul[3, 4], BiasAdd[4]]

    seed(42)
    var params3 = make_list(Fused3.PARAM_SIZE)
    var inp3 = make_list(BATCH * 3)

    var fo3 = make_zeros(BATCH * 4)
    var fc3 = make_zeros(BATCH * Fused3.CACHE_SIZE)
    var inp3_t = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](inp3.unsafe_ptr())
    var fo3_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](fo3.unsafe_ptr())
    var fp3_t = LayoutTensor[dtype, Layout.row_major(Fused3.PARAM_SIZE), MutAnyOrigin](params3.unsafe_ptr())
    var fc3_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused3.CACHE_SIZE), MutAnyOrigin](fc3.unsafe_ptr())
    Fused3.forward[BATCH](inp3_t, fo3_t, fp3_t, fc3_t)

    var uo3 = make_zeros(BATCH * 4)
    var uc3 = make_zeros(BATCH * Unfused3.CACHE_SIZE)
    var uo3_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](uo3.unsafe_ptr())
    var up3_t = LayoutTensor[dtype, Layout.row_major(Unfused3.PARAM_SIZE), MutAnyOrigin](params3.unsafe_ptr())
    var uc3_t = LayoutTensor[dtype, Layout.row_major(BATCH, Unfused3.CACHE_SIZE), MutAnyOrigin](uc3.unsafe_ptr())
    Unfused3.forward[BATCH](inp3_t, uo3_t, up3_t, uc3_t)

    var d3 = max_diff(fo3, uo3, BATCH * 4)
    if d3 < 1e-5:
        print("  PASS: one_layer_linear forward max_diff = " + String(d3))
    else:
        print("  FAIL: one_layer_linear forward max_diff = " + String(d3))
        fails += 1

    return fails


fn test_two_layer_relu() -> Int:
    """two_layer_relu[3,8,2]: hidden ReLU + output linear, forward + backward."""
    print("\n" + "=" * 70)
    print("TEST: FusedChain two_layer_relu (forward + backward)")
    print("=" * 70)
    var fails = 0
    comptime BATCH = 2
    comptime IN_D = 3
    comptime HID = 8
    comptime OUT_D = 2

    comptime Fused = FusedChain.two_layer_relu[IN_D, HID, OUT_D]
    comptime Unfused = AutoDiffChain[
        MatMul[IN_D, HID], BiasAdd[HID], ReLUOp[HID],
        MatMul[HID, OUT_D], BiasAdd[OUT_D],
    ]

    comptime if Fused.PARAM_SIZE != Unfused.PARAM_SIZE:
        print("  FAIL: PARAM_SIZE mismatch")
        fails += 1

    seed(42)
    var params = make_list(Fused.PARAM_SIZE)
    var inp = make_list(BATCH * IN_D)

    # Forward
    var fo = make_zeros(BATCH * OUT_D)
    var fc = make_zeros(BATCH * Fused.CACHE_SIZE)
    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](inp.unsafe_ptr())
    var fo_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](fo.unsafe_ptr())
    var fp_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](params.unsafe_ptr())
    var fc_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin](fc.unsafe_ptr())
    Fused.forward[BATCH](inp_t, fo_t, fp_t, fc_t)

    var uo = make_zeros(BATCH * OUT_D)
    var uc = make_zeros(BATCH * Unfused.CACHE_SIZE)
    var uo_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](uo.unsafe_ptr())
    var up_t = LayoutTensor[dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin](params.unsafe_ptr())
    var uc_t = LayoutTensor[dtype, Layout.row_major(BATCH, Unfused.CACHE_SIZE), MutAnyOrigin](uc.unsafe_ptr())
    Unfused.forward[BATCH](inp_t, uo_t, up_t, uc_t)

    var fwd_d = max_diff(fo, uo, BATCH * OUT_D)
    if fwd_d < 1e-5:
        print("  PASS: forward max_diff = " + String(fwd_d))
    else:
        print("  FAIL: forward max_diff = " + String(fwd_d))
        fails += 1

    # Backward
    seed(123)
    var grad_out = make_list(BATCH * OUT_D)
    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](grad_out.unsafe_ptr())

    var fgi = make_zeros(BATCH * IN_D)
    var fg = make_zeros(Fused.PARAM_SIZE)
    var fgi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](fgi.unsafe_ptr())
    var fg_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](fg.unsafe_ptr())
    Fused.backward[BATCH](go_t, fgi_t, fp_t, fc_t, fg_t)

    var ugi = make_zeros(BATCH * IN_D)
    var ug = make_zeros(Unfused.PARAM_SIZE)
    var ugi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](ugi.unsafe_ptr())
    var ug_t = LayoutTensor[dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin](ug.unsafe_ptr())
    Unfused.backward[BATCH](go_t, ugi_t, up_t, uc_t, ug_t)

    var bgi = max_diff(fgi, ugi, BATCH * IN_D)
    if bgi < 1e-5:
        print("  PASS: backward grad_input max_diff = " + String(bgi))
    else:
        print("  FAIL: backward grad_input max_diff = " + String(bgi))
        fails += 1

    var bpg = max_diff(fg, ug, Fused.PARAM_SIZE)
    if bpg < 1e-5:
        print("  PASS: backward param_grads max_diff = " + String(bpg))
    else:
        print("  FAIL: backward param_grads max_diff = " + String(bpg))
        fails += 1

    return fails


fn test_mlp_relu() -> Int:
    """mlp_relu[4,16,2]: 2 hidden ReLU + output, forward + backward."""
    print("\n" + "=" * 70)
    print("TEST: FusedChain mlp_relu (forward + backward)")
    print("=" * 70)
    var fails = 0
    comptime BATCH = 4
    comptime IN_D = 4
    comptime HID = 16
    comptime OUT_D = 2

    comptime Fused = FusedChain.mlp_relu[IN_D, HID, OUT_D]
    comptime Unfused = AutoDiffChain[
        MatMul[IN_D, HID], BiasAdd[HID], ReLUOp[HID],
        MatMul[HID, HID], BiasAdd[HID], ReLUOp[HID],
        MatMul[HID, OUT_D], BiasAdd[OUT_D],
    ]

    comptime if Fused.PARAM_SIZE != Unfused.PARAM_SIZE:
        print("  FAIL: PARAM_SIZE mismatch F=" + String(Fused.PARAM_SIZE) + " U=" + String(Unfused.PARAM_SIZE))
        fails += 1
    else:
        print("  PASS: PARAM_SIZE = " + String(Fused.PARAM_SIZE))

    seed(42)
    var params = make_list(Fused.PARAM_SIZE)
    var inp = make_list(BATCH * IN_D)

    # Forward
    var fo = make_zeros(BATCH * OUT_D)
    var fc = make_zeros(BATCH * Fused.CACHE_SIZE)
    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](inp.unsafe_ptr())
    var fo_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](fo.unsafe_ptr())
    var fp_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](params.unsafe_ptr())
    var fc_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin](fc.unsafe_ptr())
    Fused.forward[BATCH](inp_t, fo_t, fp_t, fc_t)

    var uo = make_zeros(BATCH * OUT_D)
    var uc = make_zeros(BATCH * Unfused.CACHE_SIZE)
    var uo_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](uo.unsafe_ptr())
    var up_t = LayoutTensor[dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin](params.unsafe_ptr())
    var uc_t = LayoutTensor[dtype, Layout.row_major(BATCH, Unfused.CACHE_SIZE), MutAnyOrigin](uc.unsafe_ptr())
    Unfused.forward[BATCH](inp_t, uo_t, up_t, uc_t)

    var fwd_d = max_diff(fo, uo, BATCH * OUT_D)
    if fwd_d < 1e-5:
        print("  PASS: forward max_diff = " + String(fwd_d))
    else:
        print("  FAIL: forward max_diff = " + String(fwd_d))
        fails += 1

    # Backward
    seed(123)
    var grad_out = make_list(BATCH * OUT_D)
    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](grad_out.unsafe_ptr())

    var fgi = make_zeros(BATCH * IN_D)
    var fg = make_zeros(Fused.PARAM_SIZE)
    var fgi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](fgi.unsafe_ptr())
    var fg_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](fg.unsafe_ptr())
    Fused.backward[BATCH](go_t, fgi_t, fp_t, fc_t, fg_t)

    var ugi = make_zeros(BATCH * IN_D)
    var ug = make_zeros(Unfused.PARAM_SIZE)
    var ugi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](ugi.unsafe_ptr())
    var ug_t = LayoutTensor[dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin](ug.unsafe_ptr())
    Unfused.backward[BATCH](go_t, ugi_t, up_t, uc_t, ug_t)

    var bgi = max_diff(fgi, ugi, BATCH * IN_D)
    if bgi < 1e-5:
        print("  PASS: backward grad_input max_diff = " + String(bgi))
    else:
        print("  FAIL: backward grad_input max_diff = " + String(bgi))
        fails += 1

    var bpg = max_diff(fg, ug, Fused.PARAM_SIZE)
    if bpg < 1e-5:
        print("  PASS: backward param_grads max_diff = " + String(bpg))
    else:
        print("  FAIL: backward param_grads max_diff = " + String(bpg))
        fails += 1

    return fails


fn test_fusion_analyzer_production() -> Int:
    print("\n" + "=" * 70)
    print("TEST: FusionAnalyzer (production module)")
    print("=" * 70)
    var fails = 0

    # Mixed chain: MBR at [0], MBT at [3], MB at [6]
    comptime FA = FusionAnalyzer[
        MatMul[3, 8], BiasAdd[8], ReLUOp[8],
        MatMul[8, 4], BiasAdd[4], TanhOp[4],
        MatMul[4, 2], BiasAdd[2],
    ]

    comptime if FA._is_matmul_bias_relu_at[0]():
        print("  PASS: MBR at [0]")
    else:
        print("  FAIL: should detect MBR at [0]")
        fails += 1

    comptime if FA._is_matmul_bias_tanh_at[3]():
        print("  PASS: MBT at [3]")
    else:
        print("  FAIL: should detect MBT at [3]")
        fails += 1

    comptime if FA._is_matmul_bias_at[6]():
        print("  PASS: MB at [6]")
    else:
        print("  FAIL: should detect MB at [6]")
        fails += 1

    # best_fusion_at greedy selection
    comptime if FA._best_fusion_at[0]() == "mbr":
        print("  PASS: best_fusion_at[0] = mbr")
    else:
        print("  FAIL: best_fusion_at[0] should be mbr")
        fails += 1

    comptime if FA._best_fusion_at[3]() == "mbt":
        print("  PASS: best_fusion_at[3] = mbt")
    else:
        print("  FAIL: best_fusion_at[3] should be mbt")
        fails += 1

    comptime if FA._best_fusion_at[6]() == "mb":
        print("  PASS: best_fusion_at[6] = mb")
    else:
        print("  FAIL: best_fusion_at[6] should be mb")
        fails += 1

    return fails


fn test_dimension_checks() -> Int:
    """Verify FusedChain aliases have correct IN_DIM/OUT_DIM."""
    print("\n" + "=" * 70)
    print("TEST: FusedChain dimension checks")
    print("=" * 70)
    var fails = 0

    # one_layer
    comptime if FusedChain.one_layer_relu[4, 8].IN_DIM == 4 and FusedChain.one_layer_relu[4, 8].OUT_DIM == 8:
        print("  PASS: one_layer_relu[4,8] dims 4->8")
    else:
        print("  FAIL: dim mismatch")
        fails += 1

    # two_layer
    comptime if FusedChain.two_layer_relu[4, 64, 2].IN_DIM == 4 and FusedChain.two_layer_relu[4, 64, 2].OUT_DIM == 2:
        print("  PASS: two_layer_relu[4,64,2] dims 4->2")
    else:
        print("  FAIL: dim mismatch")
        fails += 1

    # three_layer
    comptime if FusedChain.three_layer_relu[4, 64, 32, 2].IN_DIM == 4 and FusedChain.three_layer_relu[4, 64, 32, 2].OUT_DIM == 2:
        print("  PASS: three_layer_relu[4,64,32,2] dims 4->2")
    else:
        print("  FAIL: dim mismatch")
        fails += 1

    # mlp
    comptime if FusedChain.mlp_relu[8, 64, 4].IN_DIM == 8 and FusedChain.mlp_relu[8, 64, 4].OUT_DIM == 4:
        print("  PASS: mlp_relu[8,64,4] dims 8->4")
    else:
        print("  FAIL: dim mismatch")
        fails += 1

    comptime if FusedChain.mlp_tanh[8, 64, 4].IN_DIM == 8 and FusedChain.mlp_tanh[8, 64, 4].OUT_DIM == 4:
        print("  PASS: mlp_tanh[8,64,4] dims 8->4")
    else:
        print("  FAIL: dim mismatch")
        fails += 1

    return fails


# =============================================================================
# Main
# =============================================================================


fn main():
    print("=" * 70)
    print("FusedChain Production Tests")
    print("=" * 70)

    var total = 0
    total += test_one_layer_aliases()
    total += test_two_layer_relu()
    total += test_mlp_relu()
    total += test_fusion_analyzer_production()
    total += test_dimension_checks()

    print("\n" + "=" * 70)
    if total == 0:
        print("ALL FUSED CHAIN TESTS PASSED")
    else:
        print(String(total) + " TEST(S) FAILED")
    print("=" * 70)
