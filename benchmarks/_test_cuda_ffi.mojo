"""Probe CUDA FFI feasibility.

Can we get a raw CUstream handle from DeviceStream and call CUDA driver API?

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/_test_cuda_ffi.mojo
"""

from std.gpu.host import DeviceContext, DeviceStream
from std.ffi import OwnedDLHandle, c_int
from std.memory import alloc

# All CUDA handles are opaque pointers
comptime CUptr = UnsafePointer[NoneType, MutAnyOrigin]


def main() raises:
    print("=== CUDA Graph FFI Probe ===\n")

    # Load CUDA driver library
    var cuda = OwnedDLHandle("libcuda.so")
    print("Loaded libcuda.so")

    var ctx = DeviceContext()
    var stream = ctx.create_stream()

    # DeviceStream → raw CUstream handle via rebind
    var raw_stream = rebind[CUptr](stream)
    print("Raw stream handle:", Int(raw_stream))

    # --- Test 1: cuStreamIsCapturing ---
    print("\n--- Test 1: cuStreamIsCapturing ---")
    var status_buf = alloc[c_int](1)
    status_buf[] = c_int(0)

    # Call via get_function with simple signature
    var fn_is_cap = cuda.get_function[def (CUptr, UnsafePointer[c_int, MutAnyOrigin]) -> c_int]("cuStreamIsCapturing")
    var r1 = fn_is_cap(raw_stream, status_buf)
    print("Result:", r1, "(0=CUDA_SUCCESS)")
    print("Status:", status_buf[], "(0=none)")

    if r1 != 0:
        print("FAILED: not a valid CUstream")
        status_buf.free()
        return

    print("OK: DeviceStream is a valid CUstream")

    # --- Test 2: cuStreamBeginCapture ---
    print("\n--- Test 2: cuStreamBeginCapture ---")
    var fn_begin = cuda.get_function[def (CUptr, c_int) -> c_int]("cuStreamBeginCapture")
    var r2 = fn_begin(raw_stream, c_int(2))  # CU_STREAM_CAPTURE_MODE_RELAXED
    print("Result:", r2)

    if r2 != 0:
        print("FAILED: BeginCapture error", r2)
        status_buf.free()
        return

    # Verify active
    status_buf[] = c_int(0)
    _ = fn_is_cap(raw_stream, status_buf)
    print("Capture active:", status_buf[], "(1=active)")

    # --- Test 3: cuStreamEndCapture ---
    print("\n--- Test 3: cuStreamEndCapture ---")
    var graph_buf = alloc[CUptr](1)
    graph_buf[] = CUptr()
    var fn_end = cuda.get_function[def (CUptr, UnsafePointer[CUptr, MutAnyOrigin]) -> c_int]("cuStreamEndCapture")
    var r3 = fn_end(raw_stream, graph_buf)
    var graph = graph_buf[]
    print("Result:", r3, "Graph:", Int(graph))

    if r3 != 0:
        print("FAILED: EndCapture error", r3)
        status_buf.free()
        graph_buf.free()
        return

    # --- Test 4: cuGraphInstantiate ---
    print("\n--- Test 4: cuGraphInstantiate ---")
    var exec_buf = alloc[CUptr](1)
    exec_buf[] = CUptr()
    var fn_inst = cuda.get_function[def (UnsafePointer[CUptr, MutAnyOrigin], CUptr, UInt64) -> c_int]("cuGraphInstantiate")
    var r4 = fn_inst(exec_buf, graph, UInt64(0))
    print("Result:", r4)

    if r4 != 0:
        print("FAILED: Instantiate error", r4)
        var fn_gd = cuda.get_function[def (CUptr) -> c_int]("cuGraphDestroy")
        _ = fn_gd(graph)
        status_buf.free()
        graph_buf.free()
        exec_buf.free()
        return

    # --- Test 5: cuGraphLaunch ---
    print("\n--- Test 5: cuGraphLaunch ---")
    var fn_launch = cuda.get_function[def (CUptr, CUptr) -> c_int]("cuGraphLaunch")
    var r5 = fn_launch(exec_buf[], raw_stream)
    print("Result:", r5)

    # Cleanup
    var fn_ged = cuda.get_function[def (CUptr) -> c_int]("cuGraphExecDestroy")
    var fn_gd = cuda.get_function[def (CUptr) -> c_int]("cuGraphDestroy")
    _ = fn_ged(exec_buf[])
    _ = fn_gd(graph)

    status_buf.free()
    graph_buf.free()
    exec_buf.free()

    if r5 == 0:
        print("\n========================================")
        print("ALL TESTS PASSED")
        print("CUDA Graph lifecycle works from Mojo!")
        print("========================================")
    else:
        print("\nGraphLaunch failed with error", r5)
