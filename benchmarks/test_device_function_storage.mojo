"""Test: Can DeviceFunction be stored in a struct and reused?

This is the critical experiment — if we can store compiled kernels as
struct fields, we get true precompilation (compile once, dispatch many).

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/test_device_function_storage.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from std.time import perf_counter_ns

from mojo_rl.nn.constants import dtype


def add_kernel[
    dtype: DType, N: Int
](
    output: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = a[tid] + b[tid]


def main() raises:
    print("=== Test: DeviceFunction Storage ===\n")

    comptime N = 4096
    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)

    var ctx = DeviceContext()
    var stream = ctx.create_stream()

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var out_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(Scalar[dtype](1.0))
    b_buf.enqueue_fill(Scalar[dtype](2.0))
    out_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var a_t = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](a_buf)
    var b_t = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](b_buf)
    var out_t = LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin](out_buf)

    @always_inline
    def wrapper(
        output: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
        a: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
        b: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
    ):
        add_kernel[dtype, N](output, a, b)

    # Step 1: Compile once
    print("Compiling kernel...")
    var compiled = ctx.compile_function[wrapper, wrapper]()
    print("  Type compiled successfully")

    # Step 2: Use it multiple times from the stored var
    print("\nLaunch 1...")
    stream.enqueue_function(
        compiled, out_t, a_t, b_t, grid_dim=grid, block_dim=block
    )
    ctx.synchronize()

    # Verify result
    with out_buf.map_to_host() as h:
        print("  out[0] =", h[0], "(expected 3.0)")

    print("Launch 2 (reuse same compiled)...")
    stream.enqueue_function(
        compiled, out_t, a_t, b_t, grid_dim=grid, block_dim=block
    )
    ctx.synchronize()
    with out_buf.map_to_host() as h:
        print("  out[0] =", h[0], "(expected 3.0)")

    # Step 3: Benchmark — compile once, dispatch many
    print("\nBenchmark: 1000 dispatches from stored compiled function...")
    var total: UInt = 0
    for _ in range(1000):
        var start = perf_counter_ns()
        stream.enqueue_function(
            compiled, out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total += perf_counter_ns() - start

    var avg_ns = total // 1000
    print("  Avg dispatch: ", Float64(avg_ns) / 1000.0, " us")

    # Step 4: Compare with compile-every-time
    print("\nBenchmark: 1000 compile+dispatch (current pattern)...")
    var total2: UInt = 0
    for _ in range(1000):
        var start = perf_counter_ns()
        var c = ctx.compile_function[wrapper, wrapper]()
        stream.enqueue_function(
            c, out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total2 += perf_counter_ns() - start

    var avg_ns2 = total2 // 1000
    print("  Avg compile+dispatch: ", Float64(avg_ns2) / 1000.0, " us")

    if avg_ns > 0:
        print(
            "\n  Speedup from caching: ",
            Float64(avg_ns2) / Float64(avg_ns),
            "x",
        )

    print("\n=== DONE ===")
