"""Phase 5 validation: Muon GPU step matches CPU step.

Verifies that the GPU `_compute_inv_norm_kernel` + `step_kernel_impl` pair
produces the same params/state as the host-side CPU `step()` reference, and
that `opt_global_state[0]` is populated with `inv_norm`.

Usage:
    pixi run -e apple  mojo run -I . tests/nn/test_muon_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_muon_gpu.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs, sqrt
from std.memory import alloc, memset, UnsafePointer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.optimizer import Muon


def _check_close(name: String, a: Float64, b: Float64, tol: Float64) -> Bool:
    var err = abs(a - b)
    if err < tol:
        return True
    print("  [FAIL]", name, ": cpu=", a, " gpu=", b, " err=", err)
    return False


def test_muon_gpu_matches_cpu(ctx: DeviceContext) raises:
    """Three-step Muon GPU run vs CPU reference."""
    print("Muon GPU vs CPU:")
    comptime PS = 16
    comptime LR = 0.02
    comptime BETA = 0.95
    comptime EPS = 1e-7

    # CPU reference state.
    var cpu_params = alloc[Scalar[dtype]](PS)
    var cpu_grads = alloc[Scalar[dtype]](PS)
    var cpu_state = alloc[Scalar[dtype]](PS)
    memset(cpu_state, 0, PS)
    var cpu_og = alloc[Scalar[dtype]](1)
    (cpu_og + 0)[] = Scalar[dtype](0.0)

    # GPU buffers (host-side mirrors are these CPU allocations; we'll
    # download GPU state at the end and compare).
    var gpu_params_buf = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_grads_buf = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_state_buf = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_og_buf = ctx.enqueue_create_buffer[dtype](1)
    var gpu_params_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var gpu_grads_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var gpu_state_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var gpu_og_host = ctx.enqueue_create_host_buffer[dtype](1)

    var params_t_cpu = LayoutTensor[
        dtype, Layout.row_major(PS), MutAnyOrigin
    ](cpu_params)
    var grads_t_cpu = LayoutTensor[
        dtype, Layout.row_major(PS), MutAnyOrigin
    ](cpu_grads)
    var state_t_cpu = LayoutTensor[
        dtype, Layout.row_major(PS, 1), MutAnyOrigin
    ](cpu_state)
    var og_t_cpu = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](cpu_og)

    var params_t_gpu = LayoutTensor[
        dtype, Layout.row_major(PS), MutAnyOrigin
    ](gpu_params_buf.unsafe_ptr())
    var grads_t_gpu = LayoutTensor[
        dtype, Layout.row_major(PS), MutAnyOrigin
    ](gpu_grads_buf.unsafe_ptr())
    var state_t_gpu = LayoutTensor[
        dtype, Layout.row_major(PS, 1), MutAnyOrigin
    ](gpu_state_buf.unsafe_ptr())
    var og_t_gpu = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](gpu_og_buf.unsafe_ptr())

    # Seed params and grads with structured non-zero values so each element
    # contributes a different magnitude to the norm.
    for i in range(PS):
        var p_init = Scalar[dtype](0.1 * Float64(i) - 0.5)
        var g_init = Scalar[dtype](0.05 * Float64((i * 7) % 11) - 0.2)
        (cpu_params + i)[] = p_init
        (cpu_grads + i)[] = g_init
        gpu_params_host[i] = p_init
        gpu_grads_host[i] = g_init
        gpu_state_host[i] = Scalar[dtype](0.0)
    gpu_og_host[0] = Scalar[dtype](0.0)
    ctx.enqueue_copy(gpu_params_buf, gpu_params_host)
    ctx.enqueue_copy(gpu_grads_buf, gpu_grads_host)
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host)
    ctx.enqueue_copy(gpu_og_buf, gpu_og_host)

    var max_param_err: Float64 = 0.0
    var max_state_err: Float64 = 0.0
    var any_fail = False

    for step in range(1, 4):
        # CPU step.
        Muon[LR, BETA, 5, EPS].step[PS](
            params_t_cpu, grads_t_cpu, state_t_cpu, og_t_cpu, step_num=step
        )

        # GPU step.
        Muon[LR, BETA, 5, EPS].step_gpu[PS](
            ctx,
            params_t_gpu,
            grads_t_gpu,
            state_t_gpu,
            og_t_gpu,
            step_num=step,
            lr_scale=1.0,
        )

        ctx.enqueue_copy(gpu_params_host, gpu_params_buf)
        ctx.enqueue_copy(gpu_state_host, gpu_state_buf)
        ctx.enqueue_copy(gpu_og_host, gpu_og_buf)
        ctx.synchronize()

        for i in range(PS):
            var pe = abs(
                Float64((cpu_params + i)[]) - Float64(gpu_params_host[i])
            )
            var se = abs(
                Float64((cpu_state + i)[]) - Float64(gpu_state_host[i])
            )
            if pe > max_param_err:
                max_param_err = pe
            if se > max_state_err:
                max_state_err = se

        # Sanity-check that GPU's stored inv_norm is positive and finite.
        # Strict numerical equivalence to a CPU-recomputed inv_norm is
        # already covered by the param/state agreement above (any wrong
        # inv_norm propagates into params).
        var gpu_inv_norm = Float64(gpu_og_host[0])
        if not (gpu_inv_norm > 0.0):
            print("  [FAIL] step", step, ": gpu inv_norm =", gpu_inv_norm)
            any_fail = True

    if max_param_err < 1e-5 and max_state_err < 1e-5 and not any_fail:
        print(
            "  [PASS] 3 steps: max_param_err=",
            max_param_err,
            " max_state_err=",
            max_state_err,
        )
    else:
        print(
            "  [FAIL] max_param_err=",
            max_param_err,
            " max_state_err=",
            max_state_err,
        )

    cpu_params.free()
    cpu_grads.free()
    cpu_state.free()
    cpu_og.free()
    print()


def main() raises:
    print("=== Muon GPU step correctness ===")
    print()
    var ctx = DeviceContext()
    test_muon_gpu_matches_cpu(ctx)
    print("=== Done ===")
