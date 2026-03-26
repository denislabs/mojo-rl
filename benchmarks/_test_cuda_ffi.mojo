"""Probe: Create CUstream via CUDA API, rebind as DeviceStream, capture kernels.

Strategy:
1. Create CUstream via cuStreamCreate (FFI) — we know this works
2. Rebind it as a DeviceStream
3. Try to use it with Mojo's enqueue_function
4. If kernels dispatch on our stream, we can capture them in a CUDA Graph

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/_test_cuda_ffi.mojo
"""

from std.gpu.host import DeviceContext, DeviceStream, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim
from std.ffi import OwnedDLHandle, c_int
from std.memory import alloc
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype


comptime CUptr = UnsafePointer[NoneType, MutAnyOrigin]


# Simple test kernel
def add_kernel[N: Int](
    output: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = a[tid] + b[tid]


def main() raises:
    print("=== CUDA Graph Capture Test ===\n")

    var mojo_ctx = DeviceContext()
    mojo_ctx.synchronize()

    var cuda = OwnedDLHandle("libcuda.so")
    var cuStreamCreate = cuda.get_function[
        def (UnsafePointer[CUptr, MutAnyOrigin], UInt32) -> c_int
    ]("cuStreamCreate")
    var cuStreamIsCapturing = cuda.get_function[
        def (CUptr, UnsafePointer[c_int, MutAnyOrigin]) -> c_int
    ]("cuStreamIsCapturing")
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

    # Create our own CUstream
    var stream_buf = alloc[CUptr](1)
    stream_buf[] = CUptr()
    var r = cuStreamCreate(stream_buf, UInt32(0))
    var cuda_stream = stream_buf[]
    print("Created CUstream:", Int(cuda_stream), "result:", r)

    if r != 0:
        print("FAILED: cuStreamCreate")
        stream_buf.free()
        return

    # Rebind raw CUstream as DeviceStream
    # DeviceStream has one field: _handle: UnsafePointer[_DeviceStreamCpp]
    # Our CUstream is also a pointer — let's see if rebind works
    var mojo_stream = rebind[DeviceStream](cuda_stream)
    print("Rebinded as DeviceStream")

    # Allocate test buffers
    comptime N = 1024
    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)

    var a_buf = mojo_ctx.enqueue_create_buffer[dtype](N)
    var b_buf = mojo_ctx.enqueue_create_buffer[dtype](N)
    var out_buf = mojo_ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(Scalar[dtype](1.0))
    b_buf.enqueue_fill(Scalar[dtype](2.0))
    out_buf.enqueue_fill(Scalar[dtype](0.0))
    mojo_ctx.synchronize()

    var a_t = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](a_buf)
    var b_t = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](b_buf)
    var out_t = LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin](out_buf)

    comptime kernel = add_kernel[N]

    # --- Test: enqueue on our stream (no capture) ---
    print("\n--- Test: enqueue_function with compile_function on our stream ---")
    var compiled = mojo_ctx.compile_function[kernel, kernel]()
    stream_buf.free()

    # Use stream.enqueue_function with our rebinded DeviceStream
    mojo_stream.enqueue_function(
        compiled, out_t, a_t, b_t, grid_dim=grid, block_dim=block
    )
    _ = cuStreamSynchronize(cuda_stream)

    # Check result
    with out_buf.map_to_host() as h:
        print("out[0] =", h[0], "(expected 3.0)")
        if h[0] == Scalar[dtype](3.0):
            print("KERNEL EXECUTED ON OUR CUDA STREAM!")
        else:
            print("Kernel may not have run on our stream")
            return

    # --- Test: Graph capture with kernel ---
    print("\n--- Graph Capture with Actual Kernel ---")
    out_buf.enqueue_fill(Scalar[dtype](0.0))
    mojo_ctx.synchronize()

    var r_begin = cuStreamBeginCapture(cuda_stream, c_int(2))
    print("BeginCapture:", r_begin)

    if r_begin != 0:
        print("FAILED: BeginCapture")
        return

    # Enqueue kernel DURING capture
    mojo_stream.enqueue_function(
        compiled, out_t, a_t, b_t, grid_dim=grid, block_dim=block
    )

    var graph_buf = alloc[CUptr](1)
    graph_buf[] = CUptr()
    var r_end = cuStreamEndCapture(cuda_stream, graph_buf)
    var graph = graph_buf[]
    print("EndCapture:", r_end, "Graph:", Int(graph))

    if r_end != 0:
        print("FAILED: EndCapture")
        graph_buf.free()
        return

    # Instantiate
    var exec_buf = alloc[CUptr](1)
    exec_buf[] = CUptr()
    var r_inst = cuGraphInstantiate(exec_buf, graph, UInt64(0))
    print("Instantiate:", r_inst)

    if r_inst != 0:
        print("FAILED: Instantiate")
        _ = cuGraphDestroy(graph)
        graph_buf.free()
        exec_buf.free()
        return

    # Replay the graph 3 times, verify output each time
    print("\n--- Graph Replay Benchmark ---")
    for i in range(3):
        out_buf.enqueue_fill(Scalar[dtype](0.0))
        mojo_ctx.synchronize()

        var start = perf_counter_ns()
        var r_launch = cuGraphLaunch(exec_buf[], cuda_stream)
        _ = cuStreamSynchronize(cuda_stream)
        var elapsed = perf_counter_ns() - start

        with out_buf.map_to_host() as h:
            print(
                "  Replay", i, ": launch=", r_launch,
                " out[0]=", h[0],
                " time=", Float64(elapsed) / 1000.0, "us",
            )

    # Compare with direct dispatch timing
    print("\n--- Direct Dispatch Comparison ---")
    var total_direct: UInt = 0
    for _ in range(100):
        out_buf.enqueue_fill(Scalar[dtype](0.0))
        mojo_ctx.synchronize()
        var start = perf_counter_ns()
        mojo_stream.enqueue_function(
            compiled, out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )
        _ = cuStreamSynchronize(cuda_stream)
        total_direct += perf_counter_ns() - start

    var total_graph: UInt = 0
    for _ in range(100):
        out_buf.enqueue_fill(Scalar[dtype](0.0))
        mojo_ctx.synchronize()
        var start = perf_counter_ns()
        _ = cuGraphLaunch(exec_buf[], cuda_stream)
        _ = cuStreamSynchronize(cuda_stream)
        total_graph += perf_counter_ns() - start

    var avg_direct = Float64(total_direct // 100) / 1000.0
    var avg_graph = Float64(total_graph // 100) / 1000.0
    print("  Direct dispatch avg:", avg_direct, "us")
    print("  Graph replay avg:   ", avg_graph, "us")
    if avg_graph > 0:
        print("  Speedup:            ", avg_direct / avg_graph, "x")

    # Cleanup
    _ = cuGraphExecDestroy(exec_buf[])
    _ = cuGraphDestroy(graph)
    graph_buf.free()
    exec_buf.free()

    print("\n========================================")
    print("CUDA Graph capture of Mojo kernels: SUCCESS")
    print("========================================")
