"""CUDA Graph capture — bypass AsyncRT, launch kernels via cuLaunchKernel.

Since we can't rebind a CUstream as DeviceStream (AsyncRT dereferences
the C++ wrapper), we try a different approach:
- Use ctx.enqueue_function on the DEFAULT stream for the capture
- Capture the default stream instead of a custom stream

The default CUDA stream can be captured if we use cuStreamBeginCapture
on the stream that Mojo's ctx.enqueue_function dispatches to.

Strategy: find which CUstream Mojo uses internally by checking if
the default stream (0) or a per-thread default stream works.

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
    print("=== CUDA Graph — Default Stream Capture ===\n")

    var mojo_ctx = DeviceContext()
    mojo_ctx.synchronize()

    var cuda = OwnedDLHandle("libcuda.so")
    var cuStreamIsCapturing = cuda.get_function[
        def (CUptr, UnsafePointer[c_int, MutAnyOrigin]) -> c_int
    ]("cuStreamIsCapturing")
    var cuStreamBeginCapture = cuda.get_function[
        def (CUptr, c_int) -> c_int
    ]("cuStreamBeginCapture")
    var cuStreamEndCapture = cuda.get_function[
        def (CUptr, UnsafePointer[CUptr, MutAnyOrigin]) -> c_int
    ]("cuStreamEndCapture")
    var cuStreamCreate = cuda.get_function[
        def (UnsafePointer[CUptr, MutAnyOrigin], UInt32) -> c_int
    ]("cuStreamCreate")
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
    var cuCtxGetStreamPriorityRange = cuda.get_function[
        def (UnsafePointer[c_int, MutAnyOrigin], UnsafePointer[c_int, MutAnyOrigin]) -> c_int
    ]("cuCtxGetStreamPriorityRange")

    # Allocate test buffers
    comptime N = 1024
    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)
    comptime kernel = add_kernel[N]

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

    # Test: dispatch a kernel via ctx.enqueue_function and see which stream it uses.
    # We check several candidate streams before and after dispatch.
    print("--- Probing: which CUstream does ctx.enqueue_function use? ---\n")

    # Candidate 1: Legacy default stream (CU_STREAM_LEGACY = 0x1)
    # Candidate 2: Per-thread default stream (CU_STREAM_PER_THREAD = 0x2)
    # Candidate 3: NULL stream (0x0)
    # Candidate 4: A fresh CUDA stream we create

    var status_buf = alloc[c_int](1)

    var candidates = alloc[CUptr](4)
    candidates[0] = CUptr()  # NULL (0)
    candidates[1] = CUptr(unsafe_from_address=1)  # CU_STREAM_LEGACY
    candidates[2] = CUptr(unsafe_from_address=2)  # CU_STREAM_PER_THREAD

    var fresh_buf = alloc[CUptr](1)
    fresh_buf[] = CUptr()
    _ = cuStreamCreate(fresh_buf, UInt32(0))
    candidates[3] = fresh_buf[]

    var names = ["NULL (0)", "LEGACY (1)", "PER_THREAD (2)", "fresh cuStreamCreate"]

    for i in range(4):
        status_buf[] = c_int(-1)
        var r = cuStreamIsCapturing(candidates[i], status_buf)
        print("  ", names[i], ": IsCapturing result=", r, " status=", status_buf[])

    # Try to begin capture on each, dispatch a Mojo kernel, end capture
    print("\n--- Attempting graph capture on each stream ---\n")

    for i in range(4):
        print("Stream:", names[i])
        out_buf.enqueue_fill(Scalar[dtype](0.0))
        mojo_ctx.synchronize()

        var r_begin = cuStreamBeginCapture(candidates[i], c_int(0))  # GLOBAL mode
        if r_begin != 0:
            print("  BeginCapture failed:", r_begin, "\n")
            continue

        # Dispatch kernel via Mojo's ctx.enqueue_function
        mojo_ctx.enqueue_function[kernel, kernel](
            out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )

        var graph_buf = alloc[CUptr](1)
        graph_buf[] = CUptr()
        var r_end = cuStreamEndCapture(candidates[i], graph_buf)
        var graph = graph_buf[]

        if r_end != 0:
            print("  EndCapture failed:", r_end, "\n")
            graph_buf.free()
            continue

        # Check if graph has nodes (kernel was captured)
        var exec_buf2 = alloc[CUptr](1)
        exec_buf2[] = CUptr()
        var r_inst = cuGraphInstantiate(exec_buf2, graph, UInt64(0))
        if r_inst != 0:
            print("  Instantiate failed:", r_inst, "\n")
            _ = cuGraphDestroy(graph)
            graph_buf.free()
            exec_buf2.free()
            continue

        # Try to launch
        var r_launch = cuGraphLaunch(exec_buf2[], candidates[i])
        _ = cuStreamSynchronize(candidates[i])

        with out_buf.map_to_host() as h:
            print("  EndCapture OK, Instantiate OK, Launch:", r_launch, "out[0]=", h[0])
            if h[0] == Scalar[dtype](3.0):
                print("  >>> KERNEL CAPTURED AND REPLAYED SUCCESSFULLY! <<<\n")
            else:
                print("  Graph empty or kernel not on this stream\n")

        _ = cuGraphExecDestroy(exec_buf2[])
        _ = cuGraphDestroy(graph)
        graph_buf.free()
        exec_buf2.free()

    status_buf.free()
    candidates.free()
    fresh_buf.free()
    print("=== Done ===")
