"""Phase 1 verification tests for the autodiff system.

Tests:
1. MatMul forward + vjp
2. BiasAdd forward + vjp
3. ReLUOp forward + vjp
4. TanhOp forward + vjp
5. SigmoidOp forward + vjp
6. AutoDiffChain forward matches hand-coded Sequential[Linear, ReLU]
7. AutoDiffChain backward matches hand-coded Sequential[Linear, ReLU]
8. Training convergence test (XOR with AutoDiffChain MLP)
9. Convenience alias dimensions

Run with:
    pixi run mojo run -I . tests/test_autodiff_phase1.mojo
"""

from std.random import seed, random_float64
from std.math import exp, tanh as math_tanh, abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.autodiff import (
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    AutoDiffChain,
    Dense,
    DenseReLU,
    DenseTanh,
)
from mojo_rl.nn.initializer import Kaiming
from layout import Layout, LayoutTensor


# =============================================================================
# Test helpers
# =============================================================================


def print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


def check_close(
    actual: Float64, expected: Float64, tol: Float64, msg: String
) -> Bool:
    var diff = math_abs(actual - expected)
    if diff <= tol:
        print("  PASS: " + msg)
        return True
    else:
        print(
            "  FAIL: "
            + msg
            + " expected="
            + String(expected)
            + " actual="
            + String(actual)
            + " diff="
            + String(diff)
        )
        return False


# =============================================================================
# 1. MatMul forward + vjp
# =============================================================================


def test_matmul() -> Int:
    print_header("MatMul forward + vjp")
    var fails = 0

    comptime IN = 2
    comptime OUT = 3
    comptime BATCH = 2

    # W = [[1, 2, 3], [4, 5, 6]] (row-major, shape in×out)
    var params = List[Scalar[dtype]](capacity=IN * OUT)
    params.append(1.0)
    params.append(2.0)
    params.append(3.0)
    params.append(4.0)
    params.append(5.0)
    params.append(6.0)

    # x = [[1, 0], [0, 1]]  (identity-like)
    var inp = List[Scalar[dtype]](capacity=BATCH * IN)
    inp.append(1.0)
    inp.append(0.0)
    inp.append(0.0)
    inp.append(1.0)

    var out = List[Scalar[dtype]](capacity=BATCH * OUT)
    for _ in range(BATCH * OUT):
        out.append(0)
    var cache = List[Scalar[dtype]](capacity=BATCH * IN)
    for _ in range(BATCH * IN):
        cache.append(0)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](
        out.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(MatMul[IN, OUT].PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MatMul[IN, OUT].CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())

    # Forward: y = x @ W
    # Row 0: [1,0] @ W = [1,2,3]
    # Row 1: [0,1] @ W = [4,5,6]
    MatMul[IN, OUT].eval[BATCH](inp_t, out_t, p_t, c_t)

    if not check_close(Float64(out[0]), 1.0, 1e-5, "matmul y[0,0]=1"):
        fails += 1
    if not check_close(Float64(out[1]), 2.0, 1e-5, "matmul y[0,1]=2"):
        fails += 1
    if not check_close(Float64(out[2]), 3.0, 1e-5, "matmul y[0,2]=3"):
        fails += 1
    if not check_close(Float64(out[3]), 4.0, 1e-5, "matmul y[1,0]=4"):
        fails += 1
    if not check_close(Float64(out[4]), 5.0, 1e-5, "matmul y[1,1]=5"):
        fails += 1
    if not check_close(Float64(out[5]), 6.0, 1e-5, "matmul y[1,2]=6"):
        fails += 1

    # Cache stores input
    if not check_close(Float64(cache[0]), 1.0, 1e-5, "cache[0,0]=input"):
        fails += 1

    # VJP: grad_output = ones(BATCH, OUT)
    var go = List[Scalar[dtype]](capacity=BATCH * OUT)
    for _ in range(BATCH * OUT):
        go.append(1.0)
    var gi = List[Scalar[dtype]](capacity=BATCH * IN)
    for _ in range(BATCH * IN):
        gi.append(0)
    var gp = List[Scalar[dtype]](capacity=IN * OUT)
    for _ in range(IN * OUT):
        gp.append(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](
        go.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        gi.unsafe_ptr()
    )
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(MatMul[IN, OUT].PARAM_SIZE), MutAnyOrigin
    ](gp.unsafe_ptr())

    MatMul[IN, OUT].vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    # grad_input[0] = [1,1,1] @ W.T = [1+2+3, 4+5+6] = [6, 15]
    if not check_close(Float64(gi[0]), 6.0, 1e-4, "matmul gi[0,0]=6"):
        fails += 1
    if not check_close(Float64(gi[1]), 15.0, 1e-4, "matmul gi[0,1]=15"):
        fails += 1

    # dW = input.T @ grad_out  (input=identity, grad_out=ones)
    # dW = [[1,1,1],[1,1,1]]
    if not check_close(Float64(gp[0]), 1.0, 1e-4, "matmul dW[0,0]=1"):
        fails += 1
    if not check_close(Float64(gp[5]), 1.0, 1e-4, "matmul dW[1,2]=1"):
        fails += 1

    return fails


# =============================================================================
# 2. BiasAdd forward + vjp
# =============================================================================


def test_bias_add() -> Int:
    print_header("BiasAdd forward + vjp")
    var fails = 0

    comptime DIM = 3
    comptime BATCH = 2

    var params = List[Scalar[dtype]](capacity=DIM)
    params.append(10.0)
    params.append(20.0)
    params.append(30.0)

    var inp = List[Scalar[dtype]](capacity=BATCH * DIM)
    inp.append(1.0)
    inp.append(2.0)
    inp.append(3.0)
    inp.append(4.0)
    inp.append(5.0)
    inp.append(6.0)

    var out = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        out.append(0)

    var dummy = List[Scalar[dtype]](capacity=1)
    dummy.append(0)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        out.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(BiasAdd[DIM].PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BiasAdd[DIM].CACHE_SIZE), MutAnyOrigin
    ](dummy.unsafe_ptr())

    BiasAdd[DIM].eval[BATCH](inp_t, out_t, p_t, c_t)

    if not check_close(Float64(out[0]), 11.0, 1e-5, "bias y[0,0]=11"):
        fails += 1
    if not check_close(Float64(out[2]), 33.0, 1e-5, "bias y[0,2]=33"):
        fails += 1
    if not check_close(Float64(out[3]), 14.0, 1e-5, "bias y[1,0]=14"):
        fails += 1

    # VJP
    var go = List[Scalar[dtype]](capacity=BATCH * DIM)
    go.append(1.0)
    go.append(2.0)
    go.append(3.0)
    go.append(4.0)
    go.append(5.0)
    go.append(6.0)

    var gi = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        gi.append(0)
    var gp = List[Scalar[dtype]](capacity=DIM)
    for _ in range(DIM):
        gp.append(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        go.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        gi.unsafe_ptr()
    )
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(BiasAdd[DIM].PARAM_SIZE), MutAnyOrigin
    ](gp.unsafe_ptr())

    BiasAdd[DIM].vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    # grad_input = grad_output (identity)
    if not check_close(Float64(gi[0]), 1.0, 1e-5, "bias gi=go"):
        fails += 1
    # db = sum over batch: [1+4, 2+5, 3+6] = [5, 7, 9]
    if not check_close(Float64(gp[0]), 5.0, 1e-5, "bias db[0]=5"):
        fails += 1
    if not check_close(Float64(gp[1]), 7.0, 1e-5, "bias db[1]=7"):
        fails += 1
    if not check_close(Float64(gp[2]), 9.0, 1e-5, "bias db[2]=9"):
        fails += 1

    return fails


# =============================================================================
# 3. ReLUOp forward + vjp
# =============================================================================


def test_relu_op() -> Int:
    print_header("ReLUOp forward + vjp")
    var fails = 0

    comptime DIM = 4
    comptime BATCH = 1

    var inp = List[Scalar[dtype]](capacity=BATCH * DIM)
    inp.append(-2.0)
    inp.append(0.0)
    inp.append(1.5)
    inp.append(-0.5)

    var out = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        out.append(0)
    var cache = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        cache.append(0)
    var pdummy = List[Scalar[dtype]](capacity=1)
    pdummy.append(0)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        out.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(ReLUOp[DIM].PARAM_SIZE), MutAnyOrigin
    ](pdummy.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ReLUOp[DIM].CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())

    ReLUOp[DIM].eval[BATCH](inp_t, out_t, p_t, c_t)

    if not check_close(Float64(out[0]), 0.0, 1e-5, "relu(-2)=0"):
        fails += 1
    if not check_close(Float64(out[1]), 0.0, 1e-5, "relu(0)=0"):
        fails += 1
    if not check_close(Float64(out[2]), 1.5, 1e-5, "relu(1.5)=1.5"):
        fails += 1
    if not check_close(Float64(out[3]), 0.0, 1e-5, "relu(-0.5)=0"):
        fails += 1

    # VJP: grad = grad_out * (pre_act > 0)
    var go = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        go.append(1.0)
    var gi = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        gi.append(0)
    var gpdummy = List[Scalar[dtype]](capacity=1)
    gpdummy.append(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        go.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        gi.unsafe_ptr()
    )
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(ReLUOp[DIM].PARAM_SIZE), MutAnyOrigin
    ](gpdummy.unsafe_ptr())

    ReLUOp[DIM].vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    if not check_close(Float64(gi[0]), 0.0, 1e-5, "relu grad(-2)=0"):
        fails += 1
    if not check_close(Float64(gi[1]), 0.0, 1e-5, "relu grad(0)=0"):
        fails += 1
    if not check_close(Float64(gi[2]), 1.0, 1e-5, "relu grad(1.5)=1"):
        fails += 1
    if not check_close(Float64(gi[3]), 0.0, 1e-5, "relu grad(-0.5)=0"):
        fails += 1

    return fails


# =============================================================================
# 4. TanhOp forward + vjp
# =============================================================================


def test_tanh_op() -> Int:
    print_header("TanhOp forward + vjp")
    var fails = 0

    comptime DIM = 3
    comptime BATCH = 1

    var inp = List[Scalar[dtype]](capacity=BATCH * DIM)
    inp.append(0.0)
    inp.append(1.0)
    inp.append(-1.0)

    var out = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        out.append(0)
    var cache = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        cache.append(0)
    var pdummy = List[Scalar[dtype]](capacity=1)
    pdummy.append(0)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        out.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(TanhOp[DIM].PARAM_SIZE), MutAnyOrigin
    ](pdummy.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TanhOp[DIM].CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())

    TanhOp[DIM].eval[BATCH](inp_t, out_t, p_t, c_t)

    var t0 = math_tanh(Float64(0.0))
    var t1 = math_tanh(Float64(1.0))
    var tm1 = math_tanh(Float64(-1.0))

    if not check_close(Float64(out[0]), t0, 1e-4, "tanh(0)=0"):
        fails += 1
    if not check_close(Float64(out[1]), t1, 1e-3, "tanh(1)~=0.7616"):
        fails += 1
    if not check_close(Float64(out[2]), tm1, 1e-3, "tanh(-1)~=-0.7616"):
        fails += 1

    # VJP: grad = grad_out * (1 - tanh^2)
    var go = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        go.append(1.0)
    var gi = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        gi.append(0)
    var gpdummy = List[Scalar[dtype]](capacity=1)
    gpdummy.append(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        go.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        gi.unsafe_ptr()
    )
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(TanhOp[DIM].PARAM_SIZE), MutAnyOrigin
    ](gpdummy.unsafe_ptr())

    TanhOp[DIM].vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    # dtanh(0)/dx = 1
    if not check_close(Float64(gi[0]), 1.0, 1e-3, "tanh grad(0)=1"):
        fails += 1
    # dtanh(1)/dx = 1 - tanh(1)^2
    var eg1 = 1.0 - t1 * t1
    if not check_close(Float64(gi[1]), eg1, 1e-2, "tanh grad(1)~=0.42"):
        fails += 1

    return fails


# =============================================================================
# 5. SigmoidOp forward + vjp
# =============================================================================


def test_sigmoid_op() -> Int:
    print_header("SigmoidOp forward + vjp")
    var fails = 0

    comptime DIM = 3
    comptime BATCH = 1

    var inp = List[Scalar[dtype]](capacity=BATCH * DIM)
    inp.append(0.0)
    inp.append(2.0)
    inp.append(-2.0)

    var out = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        out.append(0)
    var cache = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        cache.append(0)
    var pdummy = List[Scalar[dtype]](capacity=1)
    pdummy.append(0)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        out.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(SigmoidOp[DIM].PARAM_SIZE), MutAnyOrigin
    ](pdummy.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SigmoidOp[DIM].CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())

    SigmoidOp[DIM].eval[BATCH](inp_t, out_t, p_t, c_t)

    var s0 = 1.0 / (1.0 + exp(Float64(-0.0)))
    var s2 = 1.0 / (1.0 + exp(Float64(-2.0)))
    var sm2 = 1.0 / (1.0 + exp(Float64(2.0)))

    if not check_close(Float64(out[0]), s0, 1e-4, "sigmoid(0)=0.5"):
        fails += 1
    if not check_close(Float64(out[1]), s2, 1e-3, "sigmoid(2)~=0.88"):
        fails += 1
    if not check_close(Float64(out[2]), sm2, 1e-3, "sigmoid(-2)~=0.12"):
        fails += 1

    # VJP: grad = grad_out * sig * (1-sig)
    var go = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        go.append(1.0)
    var gi = List[Scalar[dtype]](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        gi.append(0)
    var gpdummy = List[Scalar[dtype]](capacity=1)
    gpdummy.append(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        go.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        gi.unsafe_ptr()
    )
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(SigmoidOp[DIM].PARAM_SIZE), MutAnyOrigin
    ](gpdummy.unsafe_ptr())

    SigmoidOp[DIM].vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    # dsigmoid(0)/dx = 0.5 * 0.5 = 0.25
    if not check_close(Float64(gi[0]), 0.25, 1e-3, "sigmoid grad(0)=0.25"):
        fails += 1
    var eg2 = s2 * (1.0 - s2)
    if not check_close(Float64(gi[1]), eg2, 1e-2, "sigmoid grad(2)~=0.105"):
        fails += 1

    return fails


# =============================================================================
# 6. AutoDiffChain forward matches Sequential[Linear, ReLU]
# =============================================================================


def test_chain_forward_matches_sequential() -> Int:
    print_header("AutoDiffChain forward == Sequential[Linear, ReLU]")
    var fails = 0

    seed(123)

    comptime IN_D = 2
    comptime H = 4
    comptime BATCH = 2

    comptime ADC = AutoDiffChain[MatMul[IN_D, H], BiasAdd[H], ReLUOp[H]]
    comptime SEQ = Sequential[Linear[IN_D, H], ReLU[H]]

    # Verify dimensions
    comptime if ADC.IN_DIM != SEQ.IN_DIM:
        print("  FAIL: IN_DIM mismatch")
        fails += 1
    else:
        print("  PASS: IN_DIM=" + String(ADC.IN_DIM))
    comptime if ADC.OUT_DIM != SEQ.OUT_DIM:
        print("  FAIL: OUT_DIM mismatch")
        fails += 1
    else:
        print("  PASS: OUT_DIM=" + String(ADC.OUT_DIM))
    if ADC.PARAM_SIZE != SEQ.PARAM_SIZE:
        print(
            "  FAIL: PARAM_SIZE mismatch AD="
            + String(ADC.PARAM_SIZE)
            + " SEQ="
            + String(SEQ.PARAM_SIZE)
        )
        fails += 1
    else:
        print("  PASS: PARAM_SIZE=" + String(ADC.PARAM_SIZE))

    # Shared random params
    comptime PS = ADC.PARAM_SIZE
    var params = List[Scalar[dtype]](capacity=PS)
    for _ in range(PS):
        params.append(Scalar[dtype](random_float64() * 2.0 - 1.0))

    # Shared input
    var inp = List[Scalar[dtype]](capacity=BATCH * IN_D)
    for _ in range(BATCH * IN_D):
        inp.append(Scalar[dtype](random_float64() * 2.0 - 1.0))

    # ADC forward
    var ad_out = List[Scalar[dtype]](capacity=BATCH * H)
    for _ in range(BATCH * H):
        ad_out.append(0)
    var ad_cache = List[Scalar[dtype]](capacity=BATCH * ADC.CACHE_SIZE)
    for _ in range(BATCH * ADC.CACHE_SIZE):
        ad_cache.append(0)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](inp.unsafe_ptr())
    var ad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](ad_out.unsafe_ptr())
    var params_t = LayoutTensor[
        dtype, Layout.row_major(ADC.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ad_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ADC.CACHE_SIZE), MutAnyOrigin
    ](ad_cache.unsafe_ptr())

    ADC.forward[BATCH](inp_t, ad_out_t, params_t, ad_cache_t)

    # SEQ forward (same params — Linear stores [W, b] same as MatMul+BiasAdd)
    var seq_out = List[Scalar[dtype]](capacity=BATCH * H)
    for _ in range(BATCH * H):
        seq_out.append(0)
    var seq_cache = List[Scalar[dtype]](capacity=BATCH * SEQ.CACHE_SIZE)
    for _ in range(BATCH * SEQ.CACHE_SIZE):
        seq_cache.append(0)

    var seq_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](seq_out.unsafe_ptr())
    # Rebind params_t to Sequential's layout
    var seq_params_t = LayoutTensor[
        dtype, Layout.row_major(SEQ.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var seq_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ.CACHE_SIZE), MutAnyOrigin
    ](seq_cache.unsafe_ptr())

    SEQ.forward[BATCH](inp_t, seq_out_t, seq_params_t, seq_cache_t)

    # Compare
    var max_diff: Float64 = 0.0
    for i in range(BATCH * H):
        var d = math_abs(Float64(ad_out[i]) - Float64(seq_out[i]))
        if d > max_diff:
            max_diff = d

    if not check_close(
        max_diff, 0.0, 1e-4, "forward max_diff=" + String(max_diff)
    ):
        fails += 1

    return fails


# =============================================================================
# 7. AutoDiffChain backward matches Sequential backward
# =============================================================================


def test_chain_backward_matches_sequential() -> Int:
    print_header("AutoDiffChain backward == Sequential backward")
    var fails = 0

    seed(456)

    comptime IN_D = 2
    comptime H = 4
    comptime BATCH = 2

    comptime ADC = AutoDiffChain[MatMul[IN_D, H], BiasAdd[H], ReLUOp[H]]
    comptime SEQ = Sequential[Linear[IN_D, H], ReLU[H]]
    comptime PS = ADC.PARAM_SIZE

    # Shared params + input
    var params = List[Scalar[dtype]](capacity=PS)
    for _ in range(PS):
        params.append(Scalar[dtype](random_float64() * 2.0 - 1.0))
    var inp = List[Scalar[dtype]](capacity=BATCH * IN_D)
    for _ in range(BATCH * IN_D):
        inp.append(Scalar[dtype](random_float64() * 2.0 - 1.0))

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](inp.unsafe_ptr())

    # --- ADC forward + backward ---
    var ad_out = List[Scalar[dtype]](capacity=BATCH * H)
    for _ in range(BATCH * H):
        ad_out.append(0)
    var ad_cache = List[Scalar[dtype]](capacity=BATCH * ADC.CACHE_SIZE)
    for _ in range(BATCH * ADC.CACHE_SIZE):
        ad_cache.append(0)

    var ad_params_t = LayoutTensor[
        dtype, Layout.row_major(ADC.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](ad_out.unsafe_ptr())
    var ad_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ADC.CACHE_SIZE), MutAnyOrigin
    ](ad_cache.unsafe_ptr())

    ADC.forward[BATCH](inp_t, ad_out_t, ad_params_t, ad_cache_t)

    var go = List[Scalar[dtype]](capacity=BATCH * H)
    for _ in range(BATCH * H):
        go.append(1.0)
    var ad_gi = List[Scalar[dtype]](capacity=BATCH * IN_D)
    for _ in range(BATCH * IN_D):
        ad_gi.append(0)
    var ad_gp = List[Scalar[dtype]](capacity=PS)
    for _ in range(PS):
        ad_gp.append(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, H), MutAnyOrigin](
        go.unsafe_ptr()
    )
    var ad_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](ad_gi.unsafe_ptr())
    var ad_gp_t = LayoutTensor[
        dtype, Layout.row_major(ADC.PARAM_SIZE), MutAnyOrigin
    ](ad_gp.unsafe_ptr())

    ADC.backward[BATCH](go_t, ad_gi_t, ad_params_t, ad_cache_t, ad_gp_t)

    # --- SEQ forward + backward ---
    var seq_out = List[Scalar[dtype]](capacity=BATCH * H)
    for _ in range(BATCH * H):
        seq_out.append(0)
    var seq_cache = List[Scalar[dtype]](capacity=BATCH * SEQ.CACHE_SIZE)
    for _ in range(BATCH * SEQ.CACHE_SIZE):
        seq_cache.append(0)

    var seq_params_t = LayoutTensor[
        dtype, Layout.row_major(SEQ.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var seq_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H), MutAnyOrigin
    ](seq_out.unsafe_ptr())
    var seq_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SEQ.CACHE_SIZE), MutAnyOrigin
    ](seq_cache.unsafe_ptr())

    SEQ.forward[BATCH](inp_t, seq_out_t, seq_params_t, seq_cache_t)

    # Reset grad_output to ones
    for i in range(BATCH * H):
        go[i] = 1.0

    var seq_gi = List[Scalar[dtype]](capacity=BATCH * IN_D)
    for _ in range(BATCH * IN_D):
        seq_gi.append(0)
    var seq_gp = List[Scalar[dtype]](capacity=PS)
    for _ in range(PS):
        seq_gp.append(0)

    var seq_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](seq_gi.unsafe_ptr())
    var seq_gp_t = LayoutTensor[
        dtype, Layout.row_major(SEQ.PARAM_SIZE), MutAnyOrigin
    ](seq_gp.unsafe_ptr())

    SEQ.backward[BATCH](go_t, seq_gi_t, seq_params_t, seq_cache_t, seq_gp_t)

    # Compare grad_input
    var max_gi: Float64 = 0.0
    for i in range(BATCH * IN_D):
        var d = math_abs(Float64(ad_gi[i]) - Float64(seq_gi[i]))
        if d > max_gi:
            max_gi = d
    if not check_close(
        max_gi, 0.0, 1e-3, "grad_input max_diff=" + String(max_gi)
    ):
        fails += 1

    # Compare grad_params
    var max_gp: Float64 = 0.0
    for i in range(PS):
        var d = math_abs(Float64(ad_gp[i]) - Float64(seq_gp[i]))
        if d > max_gp:
            max_gp = d
    if not check_close(
        max_gp, 0.0, 1e-3, "grad_params max_diff=" + String(max_gp)
    ):
        fails += 1

    return fails


# =============================================================================
# 8. Training convergence (manual SGD loop, XOR)
# =============================================================================


def test_training_convergence() -> Int:
    print_header("Training convergence — XOR with AutoDiffChain MLP")
    var fails = 0

    seed(42)

    comptime BATCH = 4
    comptime IN_D = 2
    comptime H = 16
    comptime OUT_D = 1
    comptime EPOCHS = 2000
    comptime LR: Float64 = 0.05

    comptime MLP = AutoDiffChain[
        MatMul[IN_D, H],
        BiasAdd[H],
        ReLUOp[H],
        MatMul[H, OUT_D],
        BiasAdd[OUT_D],
    ]

    print("  MLP: MatMul[2,16] -> BiasAdd -> ReLU -> MatMul[16,1] -> BiasAdd")
    print("  PARAM_SIZE=" + String(MLP.PARAM_SIZE))

    comptime PS = MLP.PARAM_SIZE
    comptime CS = MLP.CACHE_SIZE

    # Init params with Kaiming-like small random values
    var params = List[Scalar[dtype]](capacity=PS)
    for _ in range(PS):
        params.append(Scalar[dtype](random_float64() * 0.5 - 0.25))

    # XOR data
    var inp = List[Scalar[dtype]](capacity=BATCH * IN_D)
    inp.append(0.0)
    inp.append(0.0)
    inp.append(0.0)
    inp.append(1.0)
    inp.append(1.0)
    inp.append(0.0)
    inp.append(1.0)
    inp.append(1.0)

    var target = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    target.append(0.0)
    target.append(1.0)
    target.append(1.0)
    target.append(0.0)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](inp.unsafe_ptr())
    var target_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](target.unsafe_ptr())

    var final_loss: Float64 = 999.0

    for epoch in range(EPOCHS):
        # Forward
        var out = List[Scalar[dtype]](capacity=BATCH * OUT_D)
        for _ in range(BATCH * OUT_D):
            out.append(0)
        var cache = List[Scalar[dtype]](capacity=BATCH * CS)
        for _ in range(BATCH * CS):
            cache.append(0)

        var params_t = LayoutTensor[
            dtype, Layout.row_major(MLP.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
        ](out.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, MLP.CACHE_SIZE), MutAnyOrigin
        ](cache.unsafe_ptr())

        MLP.forward[BATCH](inp_t, out_t, params_t, cache_t)

        # MSE loss
        var loss: Float64 = 0.0
        for i in range(BATCH * OUT_D):
            var diff = Float64(out[i]) - Float64(target[i])
            loss += diff * diff
        loss /= Float64(BATCH * OUT_D)

        if epoch % 100 == 0:
            print("  epoch " + String(epoch) + " loss=" + String(loss))
        final_loss = loss

        # Backward: grad_output = 2*(out - target) / N
        var go = List[Scalar[dtype]](capacity=BATCH * OUT_D)
        for i in range(BATCH * OUT_D):
            go.append(
                Scalar[dtype](
                    2.0
                    * (Float64(out[i]) - Float64(target[i]))
                    / Float64(BATCH * OUT_D)
                )
            )
        var gi = List[Scalar[dtype]](capacity=BATCH * IN_D)
        for _ in range(BATCH * IN_D):
            gi.append(0)
        var gp = List[Scalar[dtype]](capacity=PS)
        for _ in range(PS):
            gp.append(0)

        var go_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
        ](go.unsafe_ptr())
        var gi_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
        ](gi.unsafe_ptr())
        var gp_t = LayoutTensor[
            dtype, Layout.row_major(MLP.PARAM_SIZE), MutAnyOrigin
        ](gp.unsafe_ptr())

        MLP.backward[BATCH](go_t, gi_t, params_t, cache_t, gp_t)

        # SGD update
        for i in range(PS):
            params[i] = params[i] - Scalar[dtype](LR) * gp[i]

    print("  final loss=" + String(final_loss))
    if final_loss >= 0.05:
        print("  FAIL: XOR did not converge (loss=" + String(final_loss) + ")")
        fails += 1
    else:
        print("  PASS: XOR converged (loss=" + String(final_loss) + ")")

    return fails


# =============================================================================
# 9. Convenience alias dimensions
# =============================================================================


def test_convenience_aliases() -> Int:
    print_header("Convenience alias dimensions")
    var fails = 0

    comptime IN_D = 4
    comptime OUT_D = 8

    comptime if Dense[IN_D, OUT_D].IN_DIM != IN_D:
        print("  FAIL: Dense IN_DIM")
        fails += 1
    else:
        print("  PASS: Dense IN_DIM=" + String(IN_D))

    comptime if Dense[IN_D, OUT_D].OUT_DIM != OUT_D:
        print("  FAIL: Dense OUT_DIM")
        fails += 1
    else:
        print("  PASS: LinearAD OUT_DIM=" + String(OUT_D))

    comptime if Dense[IN_D, OUT_D].PARAM_SIZE != IN_D * OUT_D + OUT_D:
        print("  FAIL: LinearAD PARAM_SIZE")
        fails += 1
    else:
        print("  PASS: LinearAD PARAM_SIZE=" + String(IN_D * OUT_D + OUT_D))

    comptime if DenseReLU[IN_D, OUT_D].OUT_DIM != OUT_D:
        print("  FAIL: LinearReLUAD OUT_DIM")
        fails += 1
    else:
        print("  PASS: LinearReLUAD OUT_DIM=" + String(OUT_D))

    comptime if DenseTanh[IN_D, OUT_D].OUT_DIM != OUT_D:
        print("  FAIL: DenseTanh OUT_DIM")
        fails += 1
    else:
        print("  PASS: DenseTanh OUT_DIM=" + String(OUT_D))

    return fails


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Autodiff Phase 1 — Verification Tests")
    print("=" * 70)

    var total_fails = 0
    total_fails += test_matmul()
    total_fails += test_bias_add()
    total_fails += test_relu_op()
    total_fails += test_tanh_op()
    total_fails += test_sigmoid_op()
    total_fails += test_chain_forward_matches_sequential()
    total_fails += test_chain_backward_matches_sequential()
    total_fails += test_training_convergence()
    total_fails += test_convenience_aliases()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TESTS PASSED")
    else:
        print(String(total_fails) + " FAILURES")
    print("=" * 70)
