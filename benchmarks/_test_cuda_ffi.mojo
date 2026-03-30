"""CUDA Graph Capture Diagnostic.

Tests which stream Mojo's ctx.enqueue_function actually dispatches on,
and whether CUDA graph capture can intercept those launches.

Tries 3 capture strategies:
  1. CU_STREAM_PER_THREAD (handle=2)
  2. NULL stream (handle=0) — the legacy default stream
  3. A freshly created stream

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
    print("=== CUDA Graph Capture — Diagnostic ===\n")

    var ctx = DeviceContext()
    ctx.synchronize()

    var cuda = OwnedDLHandle("libcuda.so")

    # --- Load all CUDA driver functions we need ---
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

    # Diagnostic: count nodes in captured graph
    # cuGraphGetNodes(graph, nodes_ptr, numNodes_ptr) -> CUresult
    # When nodes_ptr is null, just fills numNodes with the count
    var cuGraphGetNodes = cuda.get_function[
        def (CUptr, CUptr, UnsafePointer[UInt64, MutAnyOrigin]) -> c_int
    ]("cuGraphGetNodes")

    # Diagnostic: query capture status of a stream
    # cuStreamIsCapturing(stream, captureStatus_ptr) -> CUresult
    var cuStreamIsCapturing = cuda.get_function[
        def (CUptr, UnsafePointer[c_int, MutAnyOrigin]) -> c_int
    ]("cuStreamIsCapturing")

    # Diagnostic: get capture info (CUDA 10.1+)
    # cuStreamGetCaptureInfo(stream, captureStatus_ptr, id_ptr) -> CUresult
    var cuStreamGetCaptureInfo = cuda.get_function[
        def (CUptr, UnsafePointer[c_int, MutAnyOrigin], UnsafePointer[UInt64, MutAnyOrigin]) -> c_int
    ]("cuStreamGetCaptureInfo")

    # --- Buffers ---
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

    def run_chain() raises:
        ctx.enqueue_function[k_add, k_add](c_t, a_t, b_t, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_scale, k_scale](d_t, c_i, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_relu, k_relu](d_t, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_scale, k_scale](c_t, d_i, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_relu, k_relu](c_t, grid_dim=grid, block_dim=block)

    # Warmup + verify direct dispatch works
    run_chain()
    ctx.synchronize()
    with c_buf.map_to_host() as h:
        print("Direct dispatch result: c[0] =", h[0], "(expected 12.0)")

    # --- Helper: try capture on a given stream ---
    var replay_buf = alloc[CUptr](1)
    replay_buf[] = CUptr()
    _ = cuStreamCreate(replay_buf, UInt32(0))
    var replay_stream = replay_buf[]

    var graph_buf = alloc[CUptr](1)
    var exec_buf = alloc[CUptr](1)
    var num_nodes_buf = alloc[UInt64](1)
    var cap_status_buf = alloc[c_int](1)
    var cap_id_buf = alloc[UInt64](1)

    # ============================================================
    # Strategy 1: CU_STREAM_PER_THREAD (handle = 2)
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 1: CU_STREAM_PER_THREAD (handle=2)")
    print("=" * 60)

    var per_thread = CUptr(unsafe_from_address=2)

    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var r1 = cuStreamBeginCapture(per_thread, c_int(0))
    print("  BeginCapture:", r1)

    # Check capture status mid-capture
    cap_status_buf[] = c_int(0)
    var r1_status = cuStreamIsCapturing(per_thread, cap_status_buf)
    print("  IsCapturing (PER_THREAD):", r1_status, "status:", cap_status_buf[],
          "(1=capturing, 0=not)")

    run_chain()

    # Check capture status after kernels
    cap_status_buf[] = c_int(0)
    r1_status = cuStreamIsCapturing(per_thread, cap_status_buf)
    print("  IsCapturing after kernels:", r1_status, "status:", cap_status_buf[])

    graph_buf[] = CUptr()
    var r1_end = cuStreamEndCapture(per_thread, graph_buf)
    var graph1 = graph_buf[]
    print("  EndCapture:", r1_end, "Graph:", Int(graph1))

    if Int(graph1) != 0:
        num_nodes_buf[] = UInt64(0)
        var r1_nodes = cuGraphGetNodes(graph1, CUptr(), num_nodes_buf)
        print("  >>> Graph has", num_nodes_buf[], "nodes (expected 5) <<<")

        exec_buf[] = CUptr()
        var r1_inst = cuGraphInstantiate(exec_buf, graph1, UInt64(0))
        print("  Instantiate:", r1_inst)

        if r1_inst == 0:
            c_buf.enqueue_fill(Scalar[dtype](0.0))
            d_buf.enqueue_fill(Scalar[dtype](0.0))
            ctx.synchronize()
            _ = cuGraphLaunch(exec_buf[], replay_stream)
            _ = cuStreamSynchronize(replay_stream)
            with c_buf.map_to_host() as h:
                print("  Replay result: c[0] =", h[0], "(expected 12.0)")
            _ = cuGraphExecDestroy(exec_buf[])

        _ = cuGraphDestroy(graph1)
    else:
        print("  >>> No graph returned <<<")

    # ============================================================
    # Strategy 2: NULL stream (handle = 0) — legacy default
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 2: NULL stream (handle=0)")
    print("=" * 60)

    var null_stream = CUptr()

    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var r2 = cuStreamBeginCapture(null_stream, c_int(0))
    print("  BeginCapture:", r2)

    if r2 == 0:
        cap_status_buf[] = c_int(0)
        _ = cuStreamIsCapturing(null_stream, cap_status_buf)
        print("  IsCapturing (NULL):", cap_status_buf[])

        run_chain()

        graph_buf[] = CUptr()
        var r2_end = cuStreamEndCapture(null_stream, graph_buf)
        var graph2 = graph_buf[]
        print("  EndCapture:", r2_end, "Graph:", Int(graph2))

        if Int(graph2) != 0:
            num_nodes_buf[] = UInt64(0)
            _ = cuGraphGetNodes(graph2, CUptr(), num_nodes_buf)
            print("  >>> Graph has", num_nodes_buf[], "nodes <<<")

            exec_buf[] = CUptr()
            var r2_inst = cuGraphInstantiate(exec_buf, graph2, UInt64(0))
            print("  Instantiate:", r2_inst)

            if r2_inst == 0:
                c_buf.enqueue_fill(Scalar[dtype](0.0))
                d_buf.enqueue_fill(Scalar[dtype](0.0))
                ctx.synchronize()
                _ = cuGraphLaunch(exec_buf[], replay_stream)
                _ = cuStreamSynchronize(replay_stream)
                with c_buf.map_to_host() as h:
                    print("  Replay result: c[0] =", h[0], "(expected 12.0)")
                _ = cuGraphExecDestroy(exec_buf[])

            _ = cuGraphDestroy(graph2)
        else:
            print("  >>> No graph returned <<<")
    else:
        print("  BeginCapture FAILED (error", r2, ")")
        print("  (NULL stream may not support capture without CUDA_API_PER_THREAD_DEFAULT_STREAM)")

    # ============================================================
    # Strategy 3: Fresh created stream
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 3: Fresh created stream")
    print("=" * 60)

    var fresh_buf = alloc[CUptr](1)
    fresh_buf[] = CUptr()
    _ = cuStreamCreate(fresh_buf, UInt32(0))
    var fresh_stream = fresh_buf[]
    print("  Created stream:", Int(fresh_stream))

    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var r3 = cuStreamBeginCapture(fresh_stream, c_int(0))
    print("  BeginCapture:", r3)

    if r3 == 0:
        cap_status_buf[] = c_int(0)
        _ = cuStreamIsCapturing(fresh_stream, cap_status_buf)
        print("  IsCapturing (fresh):", cap_status_buf[])

        run_chain()

        graph_buf[] = CUptr()
        var r3_end = cuStreamEndCapture(fresh_stream, graph_buf)
        var graph3 = graph_buf[]
        print("  EndCapture:", r3_end, "Graph:", Int(graph3))

        if Int(graph3) != 0:
            num_nodes_buf[] = UInt64(0)
            _ = cuGraphGetNodes(graph3, CUptr(), num_nodes_buf)
            print("  >>> Graph has", num_nodes_buf[], "nodes <<<")

            exec_buf[] = CUptr()
            var r3_inst = cuGraphInstantiate(exec_buf, graph3, UInt64(0))
            print("  Instantiate:", r3_inst)

            if r3_inst == 0:
                c_buf.enqueue_fill(Scalar[dtype](0.0))
                d_buf.enqueue_fill(Scalar[dtype](0.0))
                ctx.synchronize()
                _ = cuGraphLaunch(exec_buf[], replay_stream)
                _ = cuStreamSynchronize(replay_stream)
                with c_buf.map_to_host() as h:
                    print("  Replay result: c[0] =", h[0], "(expected 12.0)")
                _ = cuGraphExecDestroy(exec_buf[])

            _ = cuGraphDestroy(graph3)
        else:
            print("  >>> No graph returned <<<")
    else:
        print("  BeginCapture FAILED (error", r3, ")")

    fresh_buf.free()

    # Cleanup
    graph_buf.free()
    exec_buf.free()
    num_nodes_buf.free()
    cap_status_buf.free()
    cap_id_buf.free()
    replay_buf.free()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If all graphs show 0 nodes, Mojo's AsyncRT dispatches on")
    print("a stream we cannot capture. Next step: LD_PRELOAD interception")
    print("to find which CUDA calls Mojo actually makes.")
    print("=" * 60)
