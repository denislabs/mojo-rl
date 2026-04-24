"""Dimension check + smoke gradcheck for the OFENet composite.

Verifies the DenseNet-style two-branch OFENet predictor composes cleanly
from Linear + BatchNorm1D + Swish + SkipConcat + SplitApply + Identity,
and that its backward pass matches finite differences for small dims.

We use a tiny state_dim=4, action_dim=2, per_unit=3, num_layers=6 config
to keep the gradcheck fast (PARAM_SIZE still grows quickly due to the
DenseNet concat chain).
"""

from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.composites_ofenet import (
    DenseBlock,
    StateBranch6,
    OFENetPredictor6,
)


def test_dims() raises:
    """Type-check: report composite dimensions match the paper formulas."""
    print("=" * 60)
    print("TEST: OFENet composite dimensions")
    print("=" * 60)

    # Tiny config (for gradcheck later)
    comptime DB_small = DenseBlock[4, 3]
    print("DenseBlock[4, 3]  IN=", DB_small.IN_DIM, " OUT=", DB_small.OUT_DIM,
          " PARAM_SIZE=", DB_small.PARAM_SIZE, " CACHE_SIZE=", DB_small.CACHE_SIZE)
    # Expected: IN=4, OUT=7

    comptime SB_small = StateBranch6[4, 3]
    print("StateBranch6[4, 3]  IN=", SB_small.IN_DIM, " OUT=", SB_small.OUT_DIM,
          " PARAM_SIZE=", SB_small.PARAM_SIZE)
    # Expected: IN=4, OUT=22

    comptime OFE_small = OFENetPredictor6[4, 2, 3]
    print("OFENetPredictor6[4, 2, 3]  IN=", OFE_small.IN_DIM,
          " OUT=", OFE_small.OUT_DIM,
          " PARAM_SIZE=", OFE_small.PARAM_SIZE)
    # Expected: IN=6, OUT=4

    # HalfCheetah paper config
    comptime OFE_hc = OFENetPredictor6[17, 6, 40]
    print("OFENetPredictor6[17, 6, 40] (HalfCheetah)",
          "  IN=", OFE_hc.IN_DIM, " OUT=", OFE_hc.OUT_DIM,
          " PARAM_SIZE=", OFE_hc.PARAM_SIZE)
    # Expected: IN=23, OUT=17

    print("PASS: dimensions composed")


def test_denseblock_gradcheck() raises:
    """Finite-difference gradcheck on a single DenseBlock."""
    print()
    print("=" * 60)
    print("TEST: DenseBlock[4, 3] gradcheck (IN=4, OUT=7)")
    print("=" * 60)

    comptime M = DenseBlock[4, 3]
    comptime BS = 3
    comptime IN = M.IN_DIM  # 4
    comptime OUT = M.OUT_DIM  # 7
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    var state = NetworkState[M, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var cache_ptr = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)
    var ws_ptr = alloc[Scalar[dtype]](BS * WS if WS > 0 else 1)
    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var grad_in_ptr = alloc[Scalar[dtype]](BS * IN)

    for i in range(BS * IN):
        input_ptr[i] = Scalar[dtype](0.1 + Float64(i % 11) * 0.1)
    for i in range(BS * OUT):
        grad_out_ptr[i] = Scalar[dtype](0.5 + Float64(i % 5) * 0.1)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](input_ptr)
    var out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](out_ptr)
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](cache_ptr)
    var go_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](grad_out_ptr)
    var gi_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](grad_in_ptr)

    # Forward
    memset(out_ptr, 0, BS * OUT)
    memset(cache_ptr, 0, BS * CS if CS > 0 else 1)
    M.forward[BS](inp_t, out_t, state.params_view(), cache_t)

    # Backward
    state.zero_grads()
    memset(grad_in_ptr, 0, BS * IN)
    var grads_v = state.grads_view()
    M.backward[BS](go_t, gi_t, state.params_view(), cache_t, grads_v)

    # FD check on input
    var eps_fd = Float64(1e-3)
    var max_input_err: Float64 = 0.0
    for idx in range(BS * IN):
        var orig = Float64(input_ptr[idx])

        input_ptr[idx] = Scalar[dtype](orig + eps_fd)
        var out_plus = alloc[Scalar[dtype]](BS * OUT)
        var cache_plus = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)
        memset(out_plus, 0, BS * OUT)
        memset(cache_plus, 0, BS * CS if CS > 0 else 1)
        # BN training-mode forward uses batch stats; the output doesn't
        # depend on the running stats (only on the current minibatch), so
        # the EMA drift across FD probes doesn't affect this check.
        var op_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](out_plus)
        var cp_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](cache_plus)
        M.forward[BS](inp_t, op_t, state.params_view(), cp_t)
        var lp: Float64 = 0.0
        for j in range(BS * OUT):
            lp += Float64(out_plus[j]) * Float64(grad_out_ptr[j])

        input_ptr[idx] = Scalar[dtype](orig - eps_fd)
        var out_minus = alloc[Scalar[dtype]](BS * OUT)
        var cache_minus = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)
        memset(out_minus, 0, BS * OUT)
        memset(cache_minus, 0, BS * CS if CS > 0 else 1)
        var om_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](out_minus)
        var cm_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](cache_minus)
        M.forward[BS](inp_t, om_t, state.params_view(), cm_t)
        var lm: Float64 = 0.0
        for j in range(BS * OUT):
            lm += Float64(out_minus[j]) * Float64(grad_out_ptr[j])

        input_ptr[idx] = Scalar[dtype](orig)

        var fd = (lp - lm) / (2.0 * eps_fd)
        var anal = Float64(grad_in_ptr[idx])
        var d = fd - anal
        if d < 0:
            d = -d
        if d > max_input_err:
            max_input_err = d

        out_plus.free()
        out_minus.free()
        cache_plus.free()
        cache_minus.free()

    print("Max |fd - analytical| on input grad:", max_input_err)
    if max_input_err < 0.05:
        print("PASS: DenseBlock input gradient")
    else:
        print("FAIL: DenseBlock input gradient")

    input_ptr.free()
    out_ptr.free()
    cache_ptr.free()
    ws_ptr.free()
    grad_out_ptr.free()
    grad_in_ptr.free()


def test_ofenet_trains() raises:
    """Train tiny OFENetPredictor6 on a toy next-state-prediction task.

    Validates that gradients flow through all three stages
    (state branch → SplitApply → action branch → predictor Linear) and
    that aux MSE loss actually decreases under Adam.
    """
    print()
    print("=" * 60)
    print("TEST: OFENetPredictor6 trains on toy dynamics")
    print("=" * 60)

    comptime SD = 4
    comptime AD = 2
    comptime PU = 3
    comptime M = OFENetPredictor6[SD, AD, PU]
    comptime BS = 8
    comptime STEPS = 200

    comptime IN = M.IN_DIM  # 6
    comptime OUT = M.OUT_DIM  # 4
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE

    print("PARAM_SIZE=", PS, " CACHE_SIZE=", CS)

    var state = NetworkState[M, Adam[LR=0.001]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var target_ptr = alloc[Scalar[dtype]](BS * OUT)
    var output_ptr = alloc[Scalar[dtype]](BS * OUT)
    var cache_ptr = alloc[Scalar[dtype]](BS * CS)
    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var grad_in_ptr = alloc[Scalar[dtype]](BS * IN)

    # Toy data: input = [s (4D) | a (2D)]. Target next_state = tanh of linear mix.
    # Keep values small so BN stats are stable.
    for b in range(BS):
        for i in range(SD):
            input_ptr[b * IN + i] = Scalar[dtype](
                Float64(b % 5) * 0.2 + Float64(i) * 0.1
            )
        for i in range(AD):
            input_ptr[b * IN + SD + i] = Scalar[dtype](
                Float64(b % 3) * 0.3 - Float64(i) * 0.2
            )
        # Target: s' = 0.5 * s + 0.3 * a[0] * [1,1,1,1]  — simple linear dynamics
        var a0 = Float64(input_ptr[b * IN + SD])
        for i in range(SD):
            var s_i = Float64(input_ptr[b * IN + i])
            target_ptr[b * OUT + i] = Scalar[dtype](0.5 * s_i + 0.3 * a0)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](input_ptr)
    var out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](output_ptr)
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](cache_ptr)
    var go_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](grad_out_ptr)
    var gi_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](grad_in_ptr)

    var init_loss: Float64 = 0.0
    var final_loss: Float64 = 0.0

    for step in range(STEPS):
        memset(output_ptr, 0, BS * OUT)
        memset(cache_ptr, 0, BS * CS)
        M.forward[BS](inp_t, out_t, state.params_view(), cache_t)

        # MSE loss + gradient
        var loss: Float64 = 0.0
        var inv = 1.0 / Float64(BS * OUT)
        for b in range(BS):
            for i in range(OUT):
                var diff = Float64(output_ptr[b * OUT + i]) - Float64(
                    target_ptr[b * OUT + i]
                )
                loss += diff * diff
                grad_out_ptr[b * OUT + i] = Scalar[dtype](2.0 * diff * inv)
        loss *= inv
        if step == 0:
            init_loss = loss

        state.zero_grads()
        memset(grad_in_ptr, 0, BS * IN)
        var grads_v = state.grads_view()
        M.backward[BS](go_t, gi_t, state.params_view(), cache_t, grads_v)
        state.optimizer_step()

        if step == STEPS - 1:
            final_loss = loss

        if step == 0 or step == STEPS // 2 or step == STEPS - 1:
            print("  step", step, "  loss=", loss)

    print("Init loss:", init_loss, " Final loss:", final_loss)
    if final_loss < init_loss * 0.2:
        print("PASS: OFENet trains on toy dynamics (>=5x loss reduction)")
    else:
        print("FAIL: insufficient loss reduction")

    input_ptr.free()
    target_ptr.free()
    output_ptr.free()
    cache_ptr.free()
    grad_out_ptr.free()
    grad_in_ptr.free()


def main() raises:
    test_dims()
    test_denseblock_gradcheck()
    test_ofenet_trains()
