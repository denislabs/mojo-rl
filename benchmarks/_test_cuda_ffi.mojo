"""Probe CUDA FFI — find CUstream inside DeviceStream._handle (C++ object).

DeviceStream._handle is a UnsafePointer[_DeviceStreamCpp].
The _DeviceStreamCpp is a C++ object wrapping a CUstream.
We scan its memory to find the valid CUstream handle.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/_test_cuda_ffi.mojo
"""

from std.gpu.host import DeviceContext, DeviceStream
from std.ffi import OwnedDLHandle, c_int
from std.memory import alloc


comptime CUptr = UnsafePointer[NoneType, MutAnyOrigin]


def main() raises:
    print("=== CUDA Graph FFI — Find CUstream in DeviceStream ===\n")

    var ctx = DeviceContext()
    ctx.synchronize()

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

    var stream = ctx.create_stream()

    # DeviceStream has one field: _handle: UnsafePointer[_DeviceStreamCpp]
    # _handle is a pointer to a C++ object. Rebind to get that pointer.
    var cpp_obj_ptr = rebind[CUptr](stream)
    print("DeviceStream._handle (C++ obj ptr):", Int(cpp_obj_ptr))

    # The C++ object likely has the CUstream as one of its first fields.
    # Treat the C++ object as an array of pointer-sized values and probe each.
    var fields = cpp_obj_ptr.bitcast[CUptr]()

    var status_buf = alloc[c_int](1)
    var found_offset = -1
    var found_handle = CUptr()

    print("\nScanning _DeviceStreamCpp fields for valid CUstream:")
    for offset in range(16):  # probe first 16 pointer-sized slots
        var candidate = (fields + offset)[]
        status_buf[] = c_int(-1)
        var r = cuStreamIsCapturing(candidate, status_buf)
        var marker = "  <<<" if r == 0 else ""
        print(
            "  [", offset, "] ptr=", Int(candidate),
            " result=", r, " status=", status_buf[], marker,
        )
        if r == 0 and found_offset == -1:
            found_offset = offset
            found_handle = candidate

    if found_offset == -1:
        print("\nNo valid CUstream found in first 16 fields.")
        status_buf.free()
        return

    print("\n=== Found CUstream at offset", found_offset, "===")
    print("Handle:", Int(found_handle))

    # Now test full graph capture with the extracted handle
    print("\n--- Graph Capture Test ---")
    var r2 = cuStreamBeginCapture(found_handle, c_int(2))
    print("BeginCapture:", r2)

    if r2 != 0:
        print("BeginCapture failed:", r2)
        status_buf.free()
        return

    status_buf[] = c_int(-1)
    _ = cuStreamIsCapturing(found_handle, status_buf)
    print("Capture active:", status_buf[])

    var graph_buf = alloc[CUptr](1)
    graph_buf[] = CUptr()
    var r3 = cuStreamEndCapture(found_handle, graph_buf)
    var graph = graph_buf[]
    print("EndCapture:", r3, "Graph:", Int(graph))

    if r3 == 0:
        var exec_buf = alloc[CUptr](1)
        exec_buf[] = CUptr()
        var r4 = cuGraphInstantiate(exec_buf, graph, UInt64(0))
        print("Instantiate:", r4)

        if r4 == 0:
            var r5 = cuGraphLaunch(exec_buf[], found_handle)
            _ = cuStreamSynchronize(found_handle)
            print("Launch:", r5)

            if r5 == 0:
                print("\n========================================")
                print("FULL SUCCESS!")
                print("CUstream extracted from DeviceStream at offset", found_offset)
                print("CUDA Graph capture works on Mojo's own stream!")
                print("========================================")

            _ = cuGraphExecDestroy(exec_buf[])

        _ = cuGraphDestroy(graph)
        exec_buf.free()

    status_buf.free()
    graph_buf.free()
