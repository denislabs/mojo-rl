"""Phase 2 verification tests for fused autodiff ops.

Run with:
    pixi run mojo run tests/test_autodiff_phase2.mojo
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
    AutoDiffChain,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
    FusedMatMulBiasSigmoid,
    FusedMatMulBiasActivation,
    SigmoidActivation,
    Dense,
    DenseReLU,
    DenseTanh,
    DenseSigmoid,
)
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
# Test 1+2: FusedMatMulBias forward + vjp
# =============================================================================


fn test_fused_bias() -> Int:
    print_header("FusedMatMulBias forward + vjp vs unfused MatMul->BiasAdd")
    var fails = 0

    comptime IN_D = 3
    comptime OUT_D = 4
    comptime BATCH = 2
    comptime Unfused = AutoDiffChain[MatMul[IN_D, OUT_D], BiasAdd[OUT_D]]
    comptime Fused = FusedMatMulBias[IN_D, OUT_D]

    check(Unfused.PARAM_SIZE == Fused.PARAM_SIZE, "PARAM_SIZE match", fails)

    seed(42)
    var params = make_rand_list(Fused.PARAM_SIZE)
    var inp = make_rand_list(BATCH * IN_D)
    var go_data = make_rand_list(BATCH * OUT_D)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](inp.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](go_data.unsafe_ptr())

    # --- Fused ---
    var f_out = make_list(BATCH * OUT_D)
    var f_cache = make_list(BATCH * Fused.CACHE_SIZE)
    var f_gi = make_list(BATCH * IN_D)
    var f_gp = make_list(Fused.PARAM_SIZE)

    var fo_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](f_out.unsafe_ptr())
    var fp_t = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var fc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin
    ](f_cache.unsafe_ptr())
    var fgi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](f_gi.unsafe_ptr())
    var fgp_t = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](f_gp.unsafe_ptr())

    Fused.eval[BATCH](inp_t, fo_t, fp_t, fc_t)
    Fused.vjp[BATCH](go_t, fgi_t, fp_t, fc_t, fgp_t)

    # --- Unfused ---
    var u_out = make_list(BATCH * OUT_D)
    var u_cache = make_list(BATCH * Unfused.CACHE_SIZE)
    var u_gi = make_list(BATCH * IN_D)
    var u_gp = make_list(Unfused.PARAM_SIZE)

    var uo_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](u_out.unsafe_ptr())
    # Use Unfused.PARAM_SIZE for unfused model's LayoutTensor types
    var up_t = LayoutTensor[
        dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var uc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Unfused.CACHE_SIZE), MutAnyOrigin
    ](u_cache.unsafe_ptr())
    var ugi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](u_gi.unsafe_ptr())
    var ugp_t = LayoutTensor[
        dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin
    ](u_gp.unsafe_ptr())

    var um = Unfused()
    um.forward[BATCH](inp_t, uo_t, up_t, uc_t)
    um.backward[BATCH](go_t, ugi_t, up_t, uc_t, ugp_t)

    var fwd_d = max_diff(f_out, u_out, BATCH * OUT_D)
    check(fwd_d < 1e-5, "forward max diff = " + String(fwd_d), fails)
    check(max_diff(f_gi, u_gi, BATCH * IN_D) < 1e-5, "grad_input match", fails)
    check(
        max_diff(f_gp, u_gp, Fused.PARAM_SIZE) < 1e-5,
        "grad_params match",
        fails,
    )
    return fails


# =============================================================================
# Test 3+4: FusedMatMulBiasReLU forward + vjp
# =============================================================================


fn test_fused_relu() -> Int:
    print_header("FusedMatMulBiasReLU forward + vjp vs unfused")
    var fails = 0

    comptime IN_D = 3
    comptime OUT_D = 4
    comptime BATCH = 2
    comptime Unfused = AutoDiffChain[
        MatMul[IN_D, OUT_D], BiasAdd[OUT_D], ReLUOp[OUT_D]
    ]
    comptime Fused = FusedMatMulBiasReLU[IN_D, OUT_D]

    check(Unfused.PARAM_SIZE == Fused.PARAM_SIZE, "PARAM_SIZE match", fails)

    seed(77)
    var params = make_rand_list(Fused.PARAM_SIZE)
    var inp = make_rand_list(BATCH * IN_D)
    var go_data = make_rand_list(BATCH * OUT_D)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](inp.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](go_data.unsafe_ptr())

    # Fused
    var f_out = make_list(BATCH * OUT_D)
    var f_cache = make_list(BATCH * Fused.CACHE_SIZE)
    var f_gi = make_list(BATCH * IN_D)
    var f_gp = make_list(Fused.PARAM_SIZE)

    var fo_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](f_out.unsafe_ptr())
    var fp_t = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var fc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin
    ](f_cache.unsafe_ptr())
    var fgi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](f_gi.unsafe_ptr())
    var fgp_t = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](f_gp.unsafe_ptr())

    Fused.eval[BATCH](inp_t, fo_t, fp_t, fc_t)
    Fused.vjp[BATCH](go_t, fgi_t, fp_t, fc_t, fgp_t)

    # Unfused
    var u_out = make_list(BATCH * OUT_D)
    var u_cache = make_list(BATCH * Unfused.CACHE_SIZE)
    var u_gi = make_list(BATCH * IN_D)
    var u_gp = make_list(Unfused.PARAM_SIZE)

    var uo_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](u_out.unsafe_ptr())
    var up_t = LayoutTensor[
        dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var uc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Unfused.CACHE_SIZE), MutAnyOrigin
    ](u_cache.unsafe_ptr())
    var ugi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](u_gi.unsafe_ptr())
    var ugp_t = LayoutTensor[
        dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin
    ](u_gp.unsafe_ptr())

    var um = Unfused()
    um.forward[BATCH](inp_t, uo_t, up_t, uc_t)
    um.backward[BATCH](go_t, ugi_t, up_t, uc_t, ugp_t)

    var fwd_d = max_diff(f_out, u_out, BATCH * OUT_D)
    check(fwd_d < 1e-5, "forward max diff = " + String(fwd_d), fails)
    check(max_diff(f_gi, u_gi, BATCH * IN_D) < 1e-5, "grad_input match", fails)
    check(
        max_diff(f_gp, u_gp, Fused.PARAM_SIZE) < 1e-5,
        "grad_params match",
        fails,
    )
    return fails


# =============================================================================
# Test 5+6: FusedMatMulBiasTanh forward + vjp
# =============================================================================


fn test_fused_tanh() -> Int:
    print_header("FusedMatMulBiasTanh forward + vjp vs unfused")
    var fails = 0

    comptime IN_D = 3
    comptime OUT_D = 4
    comptime BATCH = 2
    comptime Unfused = AutoDiffChain[
        MatMul[IN_D, OUT_D], BiasAdd[OUT_D], TanhOp[OUT_D]
    ]
    comptime Fused = FusedMatMulBiasTanh[IN_D, OUT_D]

    check(Unfused.PARAM_SIZE == Fused.PARAM_SIZE, "PARAM_SIZE match", fails)

    seed(55)
    var params = make_rand_list(Fused.PARAM_SIZE)
    var inp = make_rand_list(BATCH * IN_D)
    var go_data = make_rand_list(BATCH * OUT_D)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](inp.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](go_data.unsafe_ptr())

    # Fused
    var f_out = make_list(BATCH * OUT_D)
    var f_cache = make_list(BATCH * Fused.CACHE_SIZE)
    var f_gi = make_list(BATCH * IN_D)
    var f_gp = make_list(Fused.PARAM_SIZE)

    var fo_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](f_out.unsafe_ptr())
    var fp_t = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var fc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin
    ](f_cache.unsafe_ptr())
    var fgi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](f_gi.unsafe_ptr())
    var fgp_t = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](f_gp.unsafe_ptr())

    Fused.eval[BATCH](inp_t, fo_t, fp_t, fc_t)
    Fused.vjp[BATCH](go_t, fgi_t, fp_t, fc_t, fgp_t)

    # Unfused
    var u_out = make_list(BATCH * OUT_D)
    var u_cache = make_list(BATCH * Unfused.CACHE_SIZE)
    var u_gi = make_list(BATCH * IN_D)
    var u_gp = make_list(Unfused.PARAM_SIZE)

    var uo_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](u_out.unsafe_ptr())
    var up_t = LayoutTensor[
        dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var uc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Unfused.CACHE_SIZE), MutAnyOrigin
    ](u_cache.unsafe_ptr())
    var ugi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](u_gi.unsafe_ptr())
    var ugp_t = LayoutTensor[
        dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin
    ](u_gp.unsafe_ptr())

    var um = Unfused()
    um.forward[BATCH](inp_t, uo_t, up_t, uc_t)
    um.backward[BATCH](go_t, ugi_t, up_t, uc_t, ugp_t)

    var fwd_d = max_diff(f_out, u_out, BATCH * OUT_D)
    check(fwd_d < 1e-5, "forward max diff = " + String(fwd_d), fails)
    check(max_diff(f_gi, u_gi, BATCH * IN_D) < 1e-5, "grad_input match", fails)
    check(
        max_diff(f_gp, u_gp, Fused.PARAM_SIZE) < 1e-5,
        "grad_params match",
        fails,
    )
    return fails


# =============================================================================
# Test 7: Alias dimensions
# =============================================================================


fn test_alias_dims() -> Int:
    print_header("Fusion-aware alias dimensions")
    var fails = 0
    comptime I = 5
    comptime O = 8

    check(Dense[I, O].IN_DIM == I, "Dense IN_DIM", fails)
    check(Dense[I, O].OUT_DIM == O, "Dense OUT_DIM", fails)
    check(Dense[I, O].PARAM_SIZE == I * O + O, "Dense PARAM_SIZE", fails)
    check(DenseReLU[I, O].IN_DIM == I, "DenseReLU IN_DIM", fails)
    check(DenseReLU[I, O].OUT_DIM == O, "DenseReLU OUT_DIM", fails)
    check(DenseTanh[I, O].IN_DIM == I, "DenseTanh IN_DIM", fails)
    check(DenseTanh[I, O].OUT_DIM == O, "DenseTanh OUT_DIM", fails)
    return fails


# =============================================================================
# Test 8: XOR training with fused MLP
# =============================================================================


fn test_xor_fused() -> Int:
    print_header("Fused MLP XOR training convergence")
    var fails = 0

    comptime MLP = AutoDiffChain[
        FusedMatMulBiasReLU[2, 8], FusedMatMulBias[8, 1]
    ]
    comptime PS = MLP.PARAM_SIZE
    comptime CS = MLP.CACHE_SIZE
    comptime BATCH = 4

    var x = List[Scalar[dtype]](capacity=8)
    x.append(0)
    x.append(0)
    x.append(0)
    x.append(1)
    x.append(1)
    x.append(0)
    x.append(1)
    x.append(1)

    var y = List[Scalar[dtype]](capacity=4)
    y.append(0)
    y.append(1)
    y.append(1)
    y.append(0)

    var x_t = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        x.unsafe_ptr()
    )

    seed(42)
    var params = make_rand_list(PS)
    var grads = make_list(PS)
    var cache = make_list(BATCH * CS)
    var out = make_list(BATCH)
    var grad_out = make_list(BATCH)
    var grad_in = make_list(BATCH * 2)

    var model = MLP()
    comptime LR: Float64 = 0.05
    comptime EPOCHS = 2000
    var final_loss: Float64 = 0

    for epoch in range(EPOCHS):
        for i in range(PS):
            grads[i] = 0
        for i in range(BATCH * CS):
            cache[i] = 0
        for i in range(BATCH):
            out[i] = 0

        var p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
            params.unsafe_ptr()
        )
        var c_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CS), MutAnyOrigin
        ](cache.unsafe_ptr())
        var o_t = LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin](
            out.unsafe_ptr()
        )

        model.forward[BATCH](x_t, o_t, p_t, c_t)

        var loss: Float64 = 0
        for i in range(BATCH):
            var diff = Float64(out[i]) - Float64(y[i])
            loss += diff * diff
            grad_out[i] = Scalar[dtype](2.0 * diff / BATCH)
        loss /= BATCH

        if epoch == EPOCHS - 1:
            final_loss = loss

        for i in range(BATCH * 2):
            grad_in[i] = 0

        var go_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](grad_out.unsafe_ptr())
        var gi_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ](grad_in.unsafe_ptr())
        var g_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
            grads.unsafe_ptr()
        )

        model.backward[BATCH](go_t, gi_t, p_t, c_t, g_t)

        for i in range(PS):
            params[i] = params[i] - Scalar[dtype](LR) * grads[i]

    print("  Final loss: " + String(final_loss))
    check(final_loss < 0.01, "loss < 0.01 (converged)", fails)
    return fails


# =============================================================================
# Test: FusedMatMulBiasActivation[..., SigmoidActivation] forward + vjp
# =============================================================================


fn test_fused_sigmoid() -> Int:
    print_header("FusedMatMulBiasSigmoid forward + vjp vs unfused")
    var fails = 0

    comptime IN_D = 3
    comptime OUT_D = 4
    comptime BATCH = 2
    comptime Unfused = AutoDiffChain[
        MatMul[IN_D, OUT_D], BiasAdd[OUT_D], SigmoidOp[OUT_D]
    ]
    comptime Fused = FusedMatMulBiasActivation[IN_D, OUT_D, SigmoidActivation]

    check(Unfused.PARAM_SIZE == Fused.PARAM_SIZE, "PARAM_SIZE match", fails)

    seed(77)
    var params = make_rand_list(Fused.PARAM_SIZE)
    var inp = make_rand_list(BATCH * IN_D)
    var go_data = make_rand_list(BATCH * OUT_D)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](inp.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](go_data.unsafe_ptr())

    # Fused
    var f_out = make_list(BATCH * OUT_D)
    var f_cache = make_list(BATCH * Fused.CACHE_SIZE)
    var f_gi = make_list(BATCH * IN_D)
    var f_gp = make_list(Fused.PARAM_SIZE)

    var fo_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](f_out.unsafe_ptr())
    var fp_t = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var fc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin
    ](f_cache.unsafe_ptr())
    var fgi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](f_gi.unsafe_ptr())
    var fgp_t = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](f_gp.unsafe_ptr())

    Fused.eval[BATCH](inp_t, fo_t, fp_t, fc_t)
    Fused.vjp[BATCH](go_t, fgi_t, fp_t, fc_t, fgp_t)

    # Unfused
    var u_out = make_list(BATCH * OUT_D)
    var u_cache = make_list(BATCH * Unfused.CACHE_SIZE)
    var u_gi = make_list(BATCH * IN_D)
    var u_gp = make_list(Unfused.PARAM_SIZE)

    var uo_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](u_out.unsafe_ptr())
    var up_t = LayoutTensor[
        dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var uc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Unfused.CACHE_SIZE), MutAnyOrigin
    ](u_cache.unsafe_ptr())
    var ugi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](u_gi.unsafe_ptr())
    var ugp_t = LayoutTensor[
        dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin
    ](u_gp.unsafe_ptr())

    var um = Unfused()
    um.forward[BATCH](inp_t, uo_t, up_t, uc_t)
    um.backward[BATCH](go_t, ugi_t, up_t, uc_t, ugp_t)

    var fwd_d = max_diff(f_out, u_out, BATCH * OUT_D)
    check(fwd_d < 1e-5, "forward max diff = " + String(fwd_d), fails)
    check(max_diff(f_gi, u_gi, BATCH * IN_D) < 1e-5, "grad_input match", fails)
    check(
        max_diff(f_gp, u_gp, Fused.PARAM_SIZE) < 1e-5,
        "grad_params match",
        fails,
    )

    # Also verify the alias works
    check(
        FusedMatMulBiasSigmoid[IN_D, OUT_D].PARAM_SIZE == Fused.PARAM_SIZE,
        "alias PARAM_SIZE",
        fails,
    )
    check(
        DenseSigmoid[IN_D, OUT_D].IN_DIM == IN_D, "DenseSigmoid IN_DIM", fails
    )
    check(
        DenseSigmoid[IN_D, OUT_D].OUT_DIM == OUT_D,
        "DenseSigmoid OUT_DIM",
        fails,
    )
    return fails


# =============================================================================
# Main
# =============================================================================


fn main():
    print("=" * 70)
    print("Phase 2 Verification: Fused Autodiff Ops")
    print("=" * 70)

    var total = 0
    total += test_fused_bias()
    total += test_fused_relu()
    total += test_fused_tanh()
    total += test_alias_dims()
    total += test_xor_fused()
    total += test_fused_sigmoid()

    print("\n" + "=" * 70)
    if total == 0:
        print("ALL PHASE 2 TESTS PASSED")
    else:
        print(String(total) + " TEST(S) FAILED")
    print("=" * 70)
