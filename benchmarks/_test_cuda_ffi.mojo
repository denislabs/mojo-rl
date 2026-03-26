"""Probe CUDA FFI feasibility — test graph capture with CUDA-created stream.

Instead of extracting CUstream from Mojo's DeviceStream (hard to probe),
create a CUDA stream directly via the driver API and test graph capture.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/_test_cuda_ffi.mojo
"""

from std.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle, c_int
from std.memory import alloc


comptime CUptr = UnsafePointer[NoneType, MutAnyOrigin]


def main() raises:
    print("=== CUDA Graph FFI Probe ===\n")

    # Ensure CUDA context is initialized by creating a DeviceContext first
    var ctx = DeviceContext()
    ctx.synchronize()
    print("Mojo DeviceContext initialized")

    # Load CUDA driver library
    var cuda = OwnedDLHandle("libcuda.so")
    print("Loaded libcuda.so")

    # Get function pointers
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
    var cuStreamDestroy = cuda.get_function[
        def (CUptr) -> c_int
    ]("cuStreamDestroy_v2")
    print("Got CUDA function pointers")

    # Create a CUDA stream directly via driver API
    var stream_buf = alloc[CUptr](1)
    stream_buf[] = CUptr()
    var r_create = cuStreamCreate(stream_buf, UInt32(0))
    var cuda_stream = stream_buf[]
    print("\ncuStreamCreate:", r_create, "handle:", Int(cuda_stream))

    if r_create != 0:
        print("FAILED: Could not create CUDA stream")
        stream_buf.free()
        return

    # Test 1: cuStreamIsCapturing
    print("\n--- Test 1: cuStreamIsCapturing ---")
    var status_buf = alloc[c_int](1)
    status_buf[] = c_int(-1)
    var r1 = cuStreamIsCapturing(cuda_stream, status_buf)
    print("Result:", r1, "Status:", status_buf[], "(0=none)")

    if r1 != 0:
        print("FAILED: IsCapturing error", r1)
        status_buf.free()
        stream_buf.free()
        return

    # Test 2: Begin capture
    print("\n--- Test 2: cuStreamBeginCapture ---")
    var r2 = cuStreamBeginCapture(cuda_stream, c_int(2))  # RELAXED
    print("Result:", r2)

    if r2 != 0:
        print("FAILED: BeginCapture error", r2)
        status_buf.free()
        stream_buf.free()
        return

    # Verify capture active
    status_buf[] = c_int(-1)
    _ = cuStreamIsCapturing(cuda_stream, status_buf)
    print("Capture active:", status_buf[], "(1=active)")

    # Test 3: End capture (empty graph — no kernels between begin/end)
    print("\n--- Test 3: cuStreamEndCapture ---")
    var graph_buf = alloc[CUptr](1)
    graph_buf[] = CUptr()
    var r3 = cuStreamEndCapture(cuda_stream, graph_buf)
    var graph = graph_buf[]
    print("Result:", r3, "Graph:", Int(graph))

    if r3 != 0:
        print("FAILED: EndCapture error", r3)
        status_buf.free()
        stream_buf.free()
        graph_buf.free()
        return

    # Test 4: Instantiate
    print("\n--- Test 4: cuGraphInstantiate ---")
    var exec_buf = alloc[CUptr](1)
    exec_buf[] = CUptr()
    var r4 = cuGraphInstantiate(exec_buf, graph, UInt64(0))
    print("Result:", r4)

    if r4 != 0:
        print("FAILED: Instantiate error", r4)
        _ = cuGraphDestroy(graph)
        status_buf.free()
        stream_buf.free()
        graph_buf.free()
        exec_buf.free()
        return

    # Test 5: Launch
    print("\n--- Test 5: cuGraphLaunch ---")
    var r5 = cuGraphLaunch(exec_buf[], cuda_stream)
    print("Result:", r5)
    _ = cuStreamSynchronize(cuda_stream)

    # Cleanup
    _ = cuGraphExecDestroy(exec_buf[])
    _ = cuGraphDestroy(graph)
    _ = cuStreamDestroy(cuda_stream)
    status_buf.free()
    stream_buf.free()
    graph_buf.free()
    exec_buf.free()

    if r5 == 0:
        print("\n========================================")
        print("ALL TESTS PASSED")
        print("CUDA Graph lifecycle works from Mojo!")
        print("========================================")
        print("\nNext: capture actual Mojo GPU kernels")
        print("between BeginCapture/EndCapture.")
    else:
        print("\nGraphLaunch failed with error", r5)
