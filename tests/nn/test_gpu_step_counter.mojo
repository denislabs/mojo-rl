"""Phase 4 + Phase 6 validation: GPU optimizer step counter increments correctly
across N successive `step_gpu` calls (the same pattern that CUDA-graph replay
exercises — each replay re-runs the preamble bump kernel).

Validates:
  1. Adam: counter slot 0 holds the post-bump UInt32 step count after N calls.
  2. AdamW: same.
  3. `GPUNetworkState.download_to` syncs slot 0 back into `cpu.step_num`.
  4. CPU `optimizer_step` mirrors `step_num` into slot 0 (Phase 6 consistency
     fix), so a subsequent `upload_from` round-trips correctly.

Usage:
    pixi run -e apple  mojo run -I . tests/nn/test_gpu_step_counter.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_gpu_step_counter.mojo
"""

from std.gpu.host import DeviceContext
from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.optimizer import Adam, AdamW
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.training import NetworkState, GPUNetworkState


def test_adam_gpu_counter_advances(ctx: DeviceContext) raises:
    """N Adam.step_gpu calls leave opt_global_state[0] = N (UInt32)."""
    print("Adam GPU step counter:")
    comptime PS = 8
    comptime LR = 0.001
    comptime N_STEPS = 100

    var params_buf = ctx.enqueue_create_buffer[dtype](PS)
    var grads_buf = ctx.enqueue_create_buffer[dtype](PS)
    var state_buf = ctx.enqueue_create_buffer[dtype](PS * 2)
    var og_buf = ctx.enqueue_create_buffer[dtype](1)
    var og_host = ctx.enqueue_create_host_buffer[dtype](1)

    ctx.enqueue_memset(params_buf, 0)
    ctx.enqueue_memset(grads_buf, 0)
    ctx.enqueue_memset(state_buf, 0)
    ctx.enqueue_memset(og_buf, 0)

    var params_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var grads_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        grads_buf.unsafe_ptr()
    )
    var state_t = LayoutTensor[
        dtype, Layout.row_major(PS, 2), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var og_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
        og_buf.unsafe_ptr()
    )

    for _ in range(N_STEPS):
        Adam[LR].step_gpu[PS](
            ctx, params_t, grads_t, state_t, og_t, step_num=0, lr_scale=1.0
        )

    ctx.enqueue_copy(og_host, og_buf)
    ctx.synchronize()

    var counter_ptr = og_host.unsafe_ptr().bitcast[Scalar[DType.uint32]]()
    var observed = Int(counter_ptr[0])
    if observed == N_STEPS:
        print("  [PASS] counter advanced to", observed, "after", N_STEPS, "step_gpu calls")
    else:
        print("  [FAIL] counter=", observed, " expected=", N_STEPS)
    print()


def test_adamw_gpu_counter_advances(ctx: DeviceContext) raises:
    print("AdamW GPU step counter:")
    comptime PS = 8
    comptime LR = 0.001
    comptime N_STEPS = 50

    var params_buf = ctx.enqueue_create_buffer[dtype](PS)
    var grads_buf = ctx.enqueue_create_buffer[dtype](PS)
    var state_buf = ctx.enqueue_create_buffer[dtype](PS * 2)
    var og_buf = ctx.enqueue_create_buffer[dtype](1)
    var og_host = ctx.enqueue_create_host_buffer[dtype](1)

    ctx.enqueue_memset(params_buf, 0)
    ctx.enqueue_memset(grads_buf, 0)
    ctx.enqueue_memset(state_buf, 0)
    ctx.enqueue_memset(og_buf, 0)

    var params_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var grads_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        grads_buf.unsafe_ptr()
    )
    var state_t = LayoutTensor[
        dtype, Layout.row_major(PS, 2), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var og_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
        og_buf.unsafe_ptr()
    )

    for _ in range(N_STEPS):
        AdamW[LR].step_gpu[PS](
            ctx, params_t, grads_t, state_t, og_t, step_num=0, lr_scale=1.0
        )

    ctx.enqueue_copy(og_host, og_buf)
    ctx.synchronize()

    var counter_ptr = og_host.unsafe_ptr().bitcast[Scalar[DType.uint32]]()
    var observed = Int(counter_ptr[0])
    if observed == N_STEPS:
        print("  [PASS] counter advanced to", observed, "after", N_STEPS, "step_gpu calls")
    else:
        print("  [FAIL] counter=", observed, " expected=", N_STEPS)
    print()


def test_download_to_syncs_step_num(ctx: DeviceContext) raises:
    """After N optimizer_step calls, download_to sets cpu.step_num = N."""
    print("download_to syncs step_num from device counter:")
    comptime M = Sequential[Linear[4, 8], Linear[8, 4]]
    comptime OPT = Adam[]
    comptime N_STEPS = 25

    var cpu = NetworkState[M, OPT]()
    cpu.initialize()
    var gpu = GPUNetworkState[M, OPT](ctx)
    gpu.upload_from(cpu, ctx)

    # Need non-zero grads so optimizer_step actually mutates state, but the
    # counter advances regardless; just leave grads at 0.
    for _ in range(N_STEPS):
        gpu.optimizer_step(ctx)

    gpu.download_to(cpu, ctx)
    if cpu.step_num == N_STEPS and gpu.step_num == N_STEPS:
        print(
            "  [PASS] cpu.step_num=",
            cpu.step_num,
            " gpu.step_num=",
            gpu.step_num,
        )
    else:
        print(
            "  [FAIL] cpu.step_num=",
            cpu.step_num,
            " gpu.step_num=",
            gpu.step_num,
            " expected=",
            N_STEPS,
        )
    print()


def test_cpu_optimizer_step_mirrors_into_slot0() raises:
    """CPU optimizer_step writes step_num into opt_global_state[0]."""
    print("CPU optimizer_step mirrors step_num into opt_global_state[0]:")
    comptime M = Sequential[Linear[4, 8], Linear[8, 4]]
    comptime OPT = Adam[]
    comptime N_STEPS = 7

    var cpu = NetworkState[M, OPT]()
    cpu.initialize()

    for _ in range(N_STEPS):
        cpu.optimizer_step()

    var slot_ptr = cpu.opt_global_state.bitcast[Scalar[DType.uint32]]()
    var slot0 = Int(slot_ptr[0])
    if slot0 == N_STEPS and cpu.step_num == N_STEPS:
        print(
            "  [PASS] step_num=",
            cpu.step_num,
            " slot0=",
            slot0,
        )
    else:
        print(
            "  [FAIL] step_num=",
            cpu.step_num,
            " slot0=",
            slot0,
            " expected=",
            N_STEPS,
        )
    print()


def main() raises:
    print("=== GPU step counter validation (Phase 4 + 6) ===")
    print()
    var ctx = DeviceContext()
    test_adam_gpu_counter_advances(ctx)
    test_adamw_gpu_counter_advances(ctx)
    test_download_to_syncs_step_num(ctx)
    test_cpu_optimizer_step_mirrors_into_slot0()
    print("=== Done ===")
