"""GPU parity test for ScaledDotProductAttention.

Compares GPU forward + backward outputs against the validated CPU reference
on random inputs. Tests both causal and non-causal modes. The GPU kernels
should match the CPU implementation to within fp32 rounding noise.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_attention_gpu_parity.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import ScaledDotProductAttention
from layout import Layout, LayoutTensor


def check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


def print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


def make_rand_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(Scalar[dtype](random_float64(-0.5, 0.5)))
    return lst^


def make_zero_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(0)
    return lst^


def max_abs_diff(
    a: List[Scalar[dtype]], b_host: HostBuffer[dtype], n: Int
) -> Float64:
    var md: Float64 = 0
    for i in range(n):
        var d = math_abs(Float64(a[i]) - Float64(b_host[i]))
        if d > md:
            md = d
    return md


def _run_one[
    DIM: Int,
    HEADS: Int,
    SEQ: Int,
    BATCH: Int,
    causal: Bool,
](ctx: DeviceContext, mut fails: Int) raises:
    seed(31)
    comptime Attn = ScaledDotProductAttention[DIM, HEADS, SEQ, causal]

    print_header(
        "Attn parity (DIM="
        + String(DIM)
        + " HEADS="
        + String(HEADS)
        + " SEQ="
        + String(SEQ)
        + " BATCH="
        + String(BATCH)
        + " causal="
        + ("True" if causal else "False")
        + ")"
    )

    # ---------- Random input + grad_output ----------
    var inp = make_rand_list(BATCH * Attn.IN_DIM)
    var go = make_rand_list(BATCH * Attn.OUT_DIM)
    var params = make_zero_list(1)

    # ---------- CPU forward + backward (reference) ----------
    var out_cpu = make_zero_list(BATCH * Attn.OUT_DIM)
    var cache_cpu = make_zero_list(BATCH * Attn.CACHE_SIZE)
    var gi_cpu = make_zero_list(BATCH * Attn.IN_DIM)
    var gp_cpu = make_zero_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](out_cpu.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var cache_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
    ](cache_cpu.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var gi_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](gi_cpu.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](gp_cpu.unsafe_ptr())

    Attn.eval[BATCH, dtype](inp_t, out_cpu_t, p_t, cache_cpu_t)
    Attn.vjp[BATCH, dtype](go_t, gi_cpu_t, p_t, cache_cpu_t, gp_t)

    # ---------- GPU forward + backward ----------
    var inp_dev = ctx.enqueue_create_buffer[dtype](BATCH * Attn.IN_DIM)
    var out_dev = ctx.enqueue_create_buffer[dtype](BATCH * Attn.OUT_DIM)
    var cache_dev = ctx.enqueue_create_buffer[dtype](BATCH * Attn.CACHE_SIZE)
    var gi_dev = ctx.enqueue_create_buffer[dtype](BATCH * Attn.IN_DIM)
    var go_dev = ctx.enqueue_create_buffer[dtype](BATCH * Attn.OUT_DIM)
    var p_dev = ctx.enqueue_create_buffer[dtype](1)
    var gp_dev = ctx.enqueue_create_buffer[dtype](1)

    var inp_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Attn.IN_DIM)
    var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Attn.OUT_DIM)
    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Attn.OUT_DIM)
    var gi_host = ctx.enqueue_create_host_buffer[dtype](BATCH * Attn.IN_DIM)

    for i in range(BATCH * Attn.IN_DIM):
        inp_host[i] = inp[i]
    for i in range(BATCH * Attn.OUT_DIM):
        go_host[i] = go[i]
    ctx.enqueue_copy(inp_dev, inp_host)
    ctx.enqueue_copy(go_dev, go_host)
    ctx.enqueue_memset(cache_dev, 0)
    ctx.enqueue_memset(out_dev, 0)
    ctx.enqueue_memset(gi_dev, 0)

    var inp_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](inp_dev.unsafe_ptr())
    var out_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    var cache_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
    ](cache_dev.unsafe_ptr())
    var go_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](go_dev.unsafe_ptr())
    var gi_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](gi_dev.unsafe_ptr())
    var p_dev_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](p_dev.unsafe_ptr())
    var gp_dev_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](gp_dev.unsafe_ptr())

    var dummy_ws = ctx.enqueue_create_buffer[dtype](1)
    Attn.eval_gpu[BATCH, dtype](
        ctx, out_dev_t, inp_dev_t, p_dev_t, cache_dev_t, dummy_ws.unsafe_ptr()
    )
    Attn.vjp_gpu[BATCH, dtype](
        ctx,
        go_dev_t,
        gi_dev_t,
        p_dev_t,
        cache_dev_t,
        gp_dev_t,
        dummy_ws.unsafe_ptr(),
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()

    # ---------- Compare ----------
    var out_diff = max_abs_diff(out_cpu, out_host, BATCH * Attn.OUT_DIM)
    check(
        out_diff < 1e-4,
        "forward output max |GPU - CPU| = " + String(out_diff),
        fails,
    )
    var gi_diff = max_abs_diff(gi_cpu, gi_host, BATCH * Attn.IN_DIM)
    check(
        gi_diff < 1e-4,
        "grad_input max |GPU - CPU| = " + String(gi_diff),
        fails,
    )


def main() raises:
    var fails = 0
    var ctx = DeviceContext()

    _run_one[DIM=8, HEADS=2, SEQ=4, BATCH=2, causal=False](ctx, fails)
    _run_one[DIM=8, HEADS=2, SEQ=4, BATCH=2, causal=True](ctx, fails)
    _run_one[DIM=16, HEADS=4, SEQ=8, BATCH=2, causal=False](ctx, fails)
    _run_one[DIM=16, HEADS=4, SEQ=8, BATCH=2, causal=True](ctx, fails)
    _run_one[DIM=32, HEADS=4, SEQ=16, BATCH=4, causal=True](ctx, fails)

    print("\n" + "=" * 70)
    if fails == 0:
        print("ALL ATTENTION GPU PARITY TESTS PASSED")
    else:
        print("FAILED: " + String(fails) + " checks")
    print("=" * 70)
