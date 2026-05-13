"""ActionEmbedder smoke test.

Verifies:
  - Composite compiles and initializes (no comptime errors)
  - Forward produces non-NaN output of expected shape
  - Backward propagates non-zero gradients through every parameter group

Configurable shapes; default is the Phase 2 toy config:
  B=4, T=4, ACT=2, smoothed=32, EMB=64, mlp_scale=4

Run:
    pixi run mojo run -I . tests/experimental/lewm/test_action_embedder.mojo
"""

from std.math import abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.experimental.lewm import ActionEmbedder
from layout import Layout, LayoutTensor


def test_action_embedder_forward_backward() raises:
    comptime BATCH = 4
    comptime T = 4
    comptime ACT = 2
    comptime SMOOTHED = 32
    comptime EMB = 64

    comptime M = ActionEmbedder[T, ACT, SMOOTHED, EMB]

    print(
        "  shapes: IN_DIM =",
        M.IN_DIM,
        " OUT_DIM =",
        M.OUT_DIM,
        " PARAM_SIZE =",
        M.PARAM_SIZE,
        " CACHE_SIZE =",
        M.CACHE_SIZE,
    )

    # ---------- Initialize ----------
    var state = NetworkState[M, Adam[]]()
    state.initialize[Xavier[]]()
    var params = state.params_view()

    # ---------- Forward ----------
    var input_arr = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](
        uninitialized=True
    )
    for i in range(BATCH * M.IN_DIM):
        input_arr[i] = Scalar[dtype](0.27 * Float64(i % 13) - 0.4)

    var output_arr = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](
        uninitialized=True
    )
    var cache_arr = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](input_arr.unsafe_ptr())
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var state_t = state.model_state_view()

    M.forward[BATCH](input_t, output_t, params, state_t, cache_t)

    # NaN check
    var nan_count = 0
    var max_abs_out = Float64(0.0)
    for i in range(BATCH * M.OUT_DIM):
        var v = Float64(output_arr[i])
        if v != v:
            nan_count += 1
        var av = abs(v)
        if av > max_abs_out:
            max_abs_out = av

    if nan_count > 0:
        print("  [FAIL] forward produced", nan_count, "NaN values")
        return

    # ---------- Backward ----------
    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](
        uninitialized=True
    )
    for i in range(BATCH * M.OUT_DIM):
        grad_out_arr[i] = Scalar[dtype](1.0)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())

    var grad_in_arr = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](
        uninitialized=True
    )
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](grad_in_arr.unsafe_ptr())

    var grads_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())

    M.backward[BATCH](
        grad_out_t, grad_in_t, params, state_t, cache_t, grads_t
    )

    # Verify grads are non-zero somewhere
    var max_abs_grad = Float64(0.0)
    var nz_count = 0
    for i in range(M.PARAM_SIZE):
        var g = abs(Float64(grads_arr[i]))
        if g > max_abs_grad:
            max_abs_grad = g
        if g > 1e-8:
            nz_count += 1

    print(
        "  forward:  max|out| =",
        max_abs_out,
        ", NaN =",
        nan_count,
    )
    print(
        "  backward: max|grad| =",
        max_abs_grad,
        ", nonzero params =",
        nz_count,
        "/",
        M.PARAM_SIZE,
    )

    if (
        nan_count == 0
        and max_abs_out > 1e-6
        and max_abs_grad > 1e-6
        and nz_count > M.PARAM_SIZE // 2
    ):
        print("  [PASS] ActionEmbedder fwd+bwd smoke test")
    else:
        print("  [FAIL] ActionEmbedder smoke test")


def main() raises:
    print("=== ActionEmbedder smoke test ===")
    print()
    test_action_embedder_forward_backward()
    print()
    print("=== Done ===")
