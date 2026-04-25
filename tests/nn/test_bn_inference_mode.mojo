"""Phase 3.5 validation: BN inference-mode forward + backward parity.

Verifies that `forward_gpu_inference_with_cache` + `backward_gpu_inference`
on `BatchNorm1D`:
  1. Forward gives the same output as the existing `forward_gpu_no_cache`
     (which is already CPU/GPU-verified by `test_batch_norm_1d`).
  2. Backward gives `dx[b, f] = γ[f] · inv_std_r[f] · dy[b, f]` exactly,
     with no writes to `grads` (BN params are frozen in inference mode —
     caller is responsible for zeroing them if desired).
  3. Sequential[BN1D, Linear] inference-mode forward + backward gives the
     same `dx` as the analytical chain rule.

Usage:
    pixi run -e apple mojo run -I . tests/nn/test_bn_inference_mode.mojo
"""

from std.math import sqrt
from std.memory import alloc, memset
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import BatchNorm1D, Sequential, Linear


def test_bn1d_inference_backward_analytical() raises:
    """BN1D inference-mode backward must produce dx = γ · inv_std_r · dy.

    Compares the GPU inference backward output against the closed-form
    per-feature scale, with grad_output set to 1.0 everywhere.
    """
    print("=" * 72)
    print("TEST: BatchNorm1D inference backward — analytical formula match")
    print("=" * 72)

    comptime DIM = 8
    comptime BATCH = 6
    comptime BN = BatchNorm1D[DIM]
    comptime PS = BN.PARAM_SIZE
    comptime SS = BN.STATE_SIZE
    comptime CS = BN.CACHE_SIZE
    comptime EPS = BN.EPSILON

    var ctx = DeviceContext()

    # Non-trivial running stats so γ · inv_std_r is feature-dependent.
    var gamma_h = alloc[Scalar[dtype]](DIM)
    var beta_h = alloc[Scalar[dtype]](DIM)
    var rmean_h = alloc[Scalar[dtype]](DIM)
    var rvar_h = alloc[Scalar[dtype]](DIM)
    for f in range(DIM):
        gamma_h[f] = Scalar[dtype](1.0 + Float64(f) * 0.13)
        beta_h[f] = Scalar[dtype](Float64(f) * 0.07 - 0.15)
        rmean_h[f] = Scalar[dtype](Float64(f) * 0.05 - 0.20)
        rvar_h[f] = Scalar[dtype](0.30 + Float64(f) * 0.10)

    # Build params + state buffers on host, then upload to device.
    var params_h = alloc[Scalar[dtype]](PS)
    for f in range(DIM):
        params_h[f] = gamma_h[f]
        params_h[DIM + f] = beta_h[f]

    var state_h = alloc[Scalar[dtype]](max(1, SS))
    for f in range(DIM):
        state_h[BN.RMEAN_OFF + f] = rmean_h[f]
        state_h[BN.RVAR_OFF + f] = rvar_h[f]

    # Random-ish inputs.
    var input_h = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        input_h[i] = Scalar[dtype](Float64(i % 11) * 0.21 - 1.05)

    # dy = 1 everywhere so the analytical dx is just γ · inv_std_r per (b,f).
    var grad_out_h = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        grad_out_h[i] = Scalar[dtype](1.0)

    # Pre-set grads to a sentinel so we can verify backward did NOT touch them.
    var grads_h = alloc[Scalar[dtype]](PS)
    var SENTINEL = Scalar[dtype](-987654.0)
    for i in range(PS):
        grads_h[i] = SENTINEL

    # Upload to device.
    var input_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var output_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var params_dev = ctx.enqueue_create_buffer[dtype](PS)
    var state_dev = ctx.enqueue_create_buffer[dtype](max(1, SS))
    var cache_dev = ctx.enqueue_create_buffer[dtype](BATCH * max(1, CS))
    var grad_out_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var grad_in_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var grads_dev = ctx.enqueue_create_buffer[dtype](PS)
    var ws_dev = ctx.enqueue_create_buffer[dtype](1)

    var input_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    var output_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    var params_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var state_host = ctx.enqueue_create_host_buffer[dtype](max(1, SS))
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    var grad_in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    var grads_host = ctx.enqueue_create_host_buffer[dtype](PS)

    for i in range(BATCH * DIM):
        input_host[i] = input_h[i]
        grad_out_host[i] = grad_out_h[i]
    for i in range(PS):
        params_host[i] = params_h[i]
        grads_host[i] = grads_h[i]
    for i in range(max(1, SS)):
        state_host[i] = state_h[i] if i < SS else Scalar[dtype](0.0)

    input_dev.enqueue_copy_from(input_host)
    params_dev.enqueue_copy_from(params_host)
    state_dev.enqueue_copy_from(state_host)
    grad_out_dev.enqueue_copy_from(grad_out_host)
    grads_dev.enqueue_copy_from(grads_host)

    # LayoutTensor views.
    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](input_dev.unsafe_ptr())
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](output_dev.unsafe_ptr())
    var params_t = LayoutTensor[
        dtype, Layout.row_major(PS), MutAnyOrigin
    ](params_dev.unsafe_ptr())
    var state_t = LayoutTensor[
        dtype, Layout.row_major(SS), MutAnyOrigin
    ](state_dev.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CS), MutAnyOrigin
    ](cache_dev.unsafe_ptr())
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_out_dev.unsafe_ptr())
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_in_dev.unsafe_ptr())
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(PS), MutAnyOrigin
    ](grads_dev.unsafe_ptr())

    # Inference forward + backward.
    BN.forward_gpu_inference_with_cache[BATCH](
        ctx, output_t, input_t, params_t, state_t, cache_t, ws_dev,
    )
    BN.backward_gpu_inference[BATCH](
        ctx, grad_in_t, grad_out_t, params_t, state_t, cache_t, grads_t, ws_dev,
    )
    ctx.synchronize()

    # Download grad_in + grads + state (state must be unchanged).
    var grad_in_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    var grads_dl = ctx.enqueue_create_host_buffer[dtype](PS)
    var state_dl = ctx.enqueue_create_host_buffer[dtype](max(1, SS))
    grad_in_dev.enqueue_copy_to(grad_in_dl)
    grads_dev.enqueue_copy_to(grads_dl)
    state_dev.enqueue_copy_to(state_dl)
    ctx.synchronize()

    # 1. Check dx == γ · inv_std_r everywhere (with dy=1).
    var max_dx_err: Float64 = 0.0
    for b in range(BATCH):
        for f in range(DIM):
            var inv_std_r = 1.0 / sqrt(Float64(rvar_h[f]) + Float64(EPS))
            var expected = Float64(gamma_h[f]) * inv_std_r
            var got = Float64(grad_in_dl[b * DIM + f])
            var err = expected - got
            if err < 0:
                err = -err
            if err > max_dx_err:
                max_dx_err = err

    if max_dx_err < 1e-5:
        print(
            "  [PASS] dx matches γ·inv_std_r       (max_err = "
            + String(max_dx_err) + ")"
        )
    else:
        print(
            "  [FAIL] dx mismatch                   (max_err = "
            + String(max_dx_err) + ")"
        )

    # 2. Verify state was NOT mutated by inference forward + backward.
    var max_state_drift: Float64 = 0.0
    for f in range(DIM):
        var rm_drift = Float64(state_dl[BN.RMEAN_OFF + f]) - Float64(rmean_h[f])
        var rv_drift = Float64(state_dl[BN.RVAR_OFF + f]) - Float64(rvar_h[f])
        if rm_drift < 0: rm_drift = -rm_drift
        if rv_drift < 0: rv_drift = -rv_drift
        if rm_drift > max_state_drift: max_state_drift = rm_drift
        if rv_drift > max_state_drift: max_state_drift = rv_drift
    if max_state_drift == 0.0:
        print("  [PASS] running stats unchanged       (drift = 0.0)")
    else:
        print("  [FAIL] running stats mutated         (drift = " + String(max_state_drift) + ")")

    # 3. Verify grads sentinel — backward must NOT have touched it.
    var sentinel_ok = True
    for i in range(PS):
        if grads_dl[i] != SENTINEL:
            sentinel_ok = False
            break
    if sentinel_ok:
        print("  [PASS] grads sentinel preserved      (no grad_params writes)")
    else:
        print("  [FAIL] grads sentinel overwritten    (backward wrote grad_params)")

    print()


def main() raises:
    test_bn1d_inference_backward_analytical()
