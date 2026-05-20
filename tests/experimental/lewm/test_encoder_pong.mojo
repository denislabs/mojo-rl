"""LeWMEncoder smoke test at Pong pixel scale (Phase 3).

Verifies the encoder composite compiles + runs forward/backward on a
4-channel × 84×84 batch (Pong pixel obs).

Config:
  in_channels=4, img=84, patch=14, n_patches=36, hidden=64, heads=2,
  n_layers=2, embed=64, projector_hidden=128.

Run:
    pixi run mojo run -I . tests/experimental/lewm/test_encoder_pong.mojo
"""

from std.math import abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.experimental.lewm import LeWMEncoder
from layout import Layout, LayoutTensor


def test_encoder_pong_forward_backward() raises:
    comptime BATCH = 2
    comptime IN_CH = 4
    comptime IMG = 84
    comptime PATCH = 14
    comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)   # 36
    comptime HIDDEN = 64
    comptime HEADS = 2
    comptime LAYERS = 2
    comptime EMB = 64
    comptime PROJ_H = 128

    comptime M = LeWMEncoder[
        IN_CH, IMG, IMG, PATCH, HIDDEN, HEADS, LAYERS, N_PATCHES,
        EMB, 2, PROJ_H,
    ]

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

    var state = NetworkState[M, Adam[]]()
    state.initialize[Xavier[]]()
    var params = state.params_view()

    var input_arr = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](
        uninitialized=True
    )
    for i in range(BATCH * M.IN_DIM):
        # Mimic real Pong pixel obs: mostly zero with sparse 0..1 patches.
        var v = Float64((i * 7) % 256) / 255.0
        input_arr[i] = Scalar[dtype](v)

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
        and nz_count > M.PARAM_SIZE // 4
    ):
        print("  [PASS] LeWMEncoder Pong-scale fwd+bwd smoke")
    else:
        print("  [FAIL] LeWMEncoder Pong-scale smoke")


def main() raises:
    print("=== LeWMEncoder Pong-scale smoke ===")
    print()
    test_encoder_pong_forward_backward()
    print()
    print("=== Done ===")
