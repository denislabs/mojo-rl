"""Probe CUDA FFI feasibility.

Can we get a raw CUstream handle from DeviceStream and call CUDA driver API?

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/_test_cuda_ffi.mojo
"""

from std.gpu.host import DeviceContext, DeviceStream
from std.ffi import external_call
from std.memory import alloc


def main() raises:
    print("=== CUDA Graph FFI Probe ===\n")

    var ctx = DeviceContext()
    var stream = ctx.create_stream()

    # DeviceStream likely wraps a CUstream (void*) as its first field.
    # Use rebind to reinterpret the bits as a raw pointer.
    var raw_handle = rebind[UnsafePointer[NoneType, MutAnyOrigin]](stream)
    print("Raw stream handle:", Int(raw_handle))

    # Test: cuStreamIsCapturing(CUstream, CUstreamCaptureStatus*)
    # If this succeeds with result=0 (CUDA_SUCCESS), the handle is a valid CUstream.
    var cs_ptr = alloc[Int32](1)
    cs_ptr[] = Int32(0)
    var result = external_call[
        "cuStreamIsCapturing", Int32
    ](raw_handle, cs_ptr)
    var capture_status = cs_ptr[]
    print("cuStreamIsCapturing result:", result, "(0=CUDA_SUCCESS)")
    print("Capture status:", capture_status, "(0=none)")

    if result == 0:
        print("\n=== SUCCESS: CUDA FFI works! ===")
        print("DeviceStream handle is a valid CUstream.")

        # Test graph capture cycle
        var begin_result = external_call[
            "cuStreamBeginCapture", Int32
        ](raw_handle, Int32(2))  # CU_STREAM_CAPTURE_MODE_RELAXED
        print("\ncuStreamBeginCapture result:", begin_result, "(0=success)")

        if begin_result == 0:
            # Verify capture is active
            cs_ptr[] = Int32(0)
            _ = external_call[
                "cuStreamIsCapturing", Int32
            ](raw_handle, cs_ptr)
            print("Capture active:", cs_ptr[], "(1=active)")

            # End capture → get graph
            var graph_ptr = alloc[UnsafePointer[NoneType, MutAnyOrigin]](1)
            graph_ptr[] = UnsafePointer[NoneType, MutAnyOrigin]()
            var end_result = external_call[
                "cuStreamEndCapture", Int32
            ](raw_handle, graph_ptr)
            var graph_handle = graph_ptr[]
            print("cuStreamEndCapture result:", end_result)
            print("Graph handle:", Int(graph_handle))

            if end_result == 0:
                print("\n=== FULL SUCCESS: CUDA Graph capture works from Mojo! ===")

                # Instantiate the graph (even though it's empty)
                var exec_ptr = alloc[UnsafePointer[NoneType, MutAnyOrigin]](1)
                exec_ptr[] = UnsafePointer[NoneType, MutAnyOrigin]()
                var inst_result = external_call[
                    "cuGraphInstantiate", Int32
                ](exec_ptr, graph_handle, UInt64(0))
                print("cuGraphInstantiate result:", inst_result)

                if inst_result == 0:
                    # Launch the graph (empty, just proves the API works)
                    var launch_result = external_call[
                        "cuGraphLaunch", Int32
                    ](exec_ptr[], raw_handle)
                    print("cuGraphLaunch result:", launch_result)
                    print("\n=== COMPLETE: Full CUDA Graph lifecycle works! ===")

                    _ = external_call["cuGraphExecDestroy", Int32](exec_ptr[])

                _ = external_call["cuGraphDestroy", Int32](graph_handle)
                exec_ptr.free()
            graph_ptr.free()
        else:
            print("BeginCapture failed with error:", begin_result)
    else:
        print("\n=== FAILED: Handle is not a valid CUstream ===")
        print("Error code:", result)

    cs_ptr.free()
    print("\n=== Done ===")
