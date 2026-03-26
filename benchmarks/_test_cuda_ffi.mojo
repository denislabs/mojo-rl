"""CUDA Graph capture of Mojo GPU kernels — PROVEN WORKING.

ctx.enqueue_function dispatches on CU_STREAM_PER_THREAD.
Graph capture on a fresh cuStreamCreate also captures Mojo kernels.

This test captures a chain of 5 kernels (simulating a mini forward pass)
and benchmarks graph replay vs direct dispatch.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/_test_cuda_ffi.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim
from std.ffi import OwnedDLHandle, c_int
from std.memory import alloc
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype


comptime CUptr = UnsafePointer[NoneType, MutAnyOrigin]


def add_kernel[N: Int](
    output: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = a[tid] + b[tid]


def scale_kernel[N: Int](
    output: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = input[tid] * Scalar[dtype](2.0)


def relu_kernel[N: Int](
    data: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        var v = data.ptr[tid]
        data.ptr[tid] = v if v > Scalar[dtype](0.0) else Scalar[dtype](0.0)


def main() raises:
    print("=== CUDA Graph Capture — Mojo Kernel Benchmark ===\n")

    var ctx = DeviceContext()
    ctx.synchronize()

    var cuda = OwnedDLHandle("libcuda.so")
    var cuStreamCreate = cuda.get_function[
        def (UnsafePointer[CUptr, MutAnyOrigin], UInt32) -> c_int
    ]("cuStreamCreate")
    var cuStreamBeginCapture = cuda.get_function[
        def (CUptr, c_int) -> c_int
    ]("cuStreamBeginCapture")
    var cuStreamEndCapture = cuda.get_function[
        def (CUptr, UnsafePointer[CUptr, MutAnyOrigin]) -> c_int
    ]("cuStreamEndCapture")
    var cuGraphInstantiate = cuda.get_function[
        def (UnsafePointer[CUptr, MutAnyOrigin], CUptr, UInt64) -> c_int
    ]("cuGraphInstantiate")
    var cuGraphLaunch = cuda.get_function[
        def (CUptr, CUptr) -> c_int
    ]("cuGraphLaunch")
    var cuStreamSynchronize = cuda.get_function[
        def (CUptr) -> c_int
    ]("cuStreamSynchronize")
    var cuGraphDestroy = cuda.get_function[
        def (CUptr) -> c_int
    ]("cuGraphDestroy")
    var cuGraphExecDestroy = cuda.get_function[
        def (CUptr) -> c_int
    ]("cuGraphExecDestroy")

    # Create capture stream
    var stream_buf = alloc[CUptr](1)
    stream_buf[] = CUptr()
    _ = cuStreamCreate(stream_buf, UInt32(0))
    var capture_stream = stream_buf[]

    # Buffers
    comptime N = 8192
    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    var d_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(Scalar[dtype](1.0))
    b_buf.enqueue_fill(Scalar[dtype](2.0))
    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var a_t = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](a_buf)
    var b_t = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](b_buf)
    var c_t = LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin](c_buf)
    var c_i = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](c_buf)
    var d_t = LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin](d_buf)
    var d_i = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](d_buf)

    comptime k_add = add_kernel[N]
    comptime k_scale = scale_kernel[N]
    comptime k_relu = relu_kernel[N]

    # Define a 5-kernel chain: add → scale → relu → scale → relu
    # Simulates a mini forward pass

    def run_chain() raises:
        ctx.enqueue_function[k_add, k_add](c_t, a_t, b_t, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_scale, k_scale](d_t, c_i, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_relu, k_relu](d_t, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_scale, k_scale](c_t, d_i, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_relu, k_relu](c_t, grid_dim=grid, block_dim=block)

    # Warmup
    run_chain()
    ctx.synchronize()
    with c_buf.map_to_host() as h:
        print("Direct dispatch result: c[0] =", h[0], "(expected 12.0)")

    # --- Capture the chain ---
    print("\n--- Capturing 5-kernel chain ---")
    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var r_begin = cuStreamBeginCapture(capture_stream, c_int(0))
    print("BeginCapture:", r_begin)

    run_chain()

    var graph_buf = alloc[CUptr](1)
    graph_buf[] = CUptr()
    var r_end = cuStreamEndCapture(capture_stream, graph_buf)
    var graph = graph_buf[]
    print("EndCapture:", r_end, "Graph:", Int(graph))

    var exec_buf = alloc[CUptr](1)
    exec_buf[] = CUptr()
    var r_inst = cuGraphInstantiate(exec_buf, graph, UInt64(0))
    print("Instantiate:", r_inst)

    if r_inst != 0:
        print("FAILED")
        return

    # Verify graph replay
    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var r_launch = cuGraphLaunch(exec_buf[], capture_stream)
    _ = cuStreamSynchronize(capture_stream)
    with c_buf.map_to_host() as h:
        print("Graph replay result: c[0] =", h[0], "(expected 12.0)")
        if h[0] != Scalar[dtype](12.0):
            print("MISMATCH — graph didn't capture all kernels")
            return

    print("\n--- Benchmark: 5 kernels × 1000 iterations ---\n")

    # Direct dispatch benchmark
    var warmup = 100
    var iters = 1000

    for _ in range(warmup):
        run_chain()
        ctx.synchronize()

    var total_direct: UInt = 0
    for _ in range(iters):
        var start = perf_counter_ns()
        run_chain()
        ctx.synchronize()
        total_direct += perf_counter_ns() - start

    # Graph replay benchmark
    for _ in range(warmup):
        _ = cuGraphLaunch(exec_buf[], capture_stream)
        _ = cuStreamSynchronize(capture_stream)

    var total_graph: UInt = 0
    for _ in range(iters):
        var start = perf_counter_ns()
        _ = cuGraphLaunch(exec_buf[], capture_stream)
        _ = cuStreamSynchronize(capture_stream)
        total_graph += perf_counter_ns() - start

    var avg_direct = Float64(total_direct // UInt(iters)) / 1000.0
    var avg_graph = Float64(total_graph // UInt(iters)) / 1000.0

    print("  Direct dispatch (5 kernels): ", avg_direct, " us")
    print("  Graph replay (5 kernels):    ", avg_graph, " us")
    if avg_graph > 0.0:
        print("  Speedup:                     ", avg_direct / avg_graph, "x")
        print("  Savings per call:            ", avg_direct - avg_graph, " us")

    # Cleanup
    _ = cuGraphExecDestroy(exec_buf[])
    _ = cuGraphDestroy(graph)
    stream_buf.free()
    graph_buf.free()
    exec_buf.free()

    print("\n=== CUDA Graph Capture of Mojo Kernels: WORKING ===")
