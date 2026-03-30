"""CUDA Graph capture and replay for Mojo GPU kernels.

Provides a clean API to capture sequences of GPU kernel launches into
a CUDA graph, then replay them with reduced launch overhead.

On Apple Silicon / non-NVIDIA platforms, all methods are compile-time
no-ops with zero overhead.

Usage:
    from mojo_rl.cuda.graph import CUDAGraph

    var graph = CUDAGraph(ctx)

    # Capture once (after warmup so stream is discovered)
    graph.begin_capture()
    my_gpu_kernels(ctx, ...)
    graph.end_capture()

    # Replay in hot loop
    for i in range(1000):
        graph.replay()

Requires: LD_PRELOAD with libcuda_intercept.so (set automatically
by pixi nvidia environment activation).
"""

from std.sys import has_nvidia_gpu_accelerator
from std.ffi import OwnedDLHandle, c_int
from std.memory import alloc
from std.gpu.host import DeviceContext


comptime _CUptr = UnsafePointer[NoneType, MutAnyOrigin]


struct CUDAGraph(Movable):
    """CUDA Graph capture and replay.

    Compile-time gated: all methods are no-ops on non-NVIDIA platforms.
    """

    # State: 0=init, 1=capturing, 2=captured (ready to replay)
    var _state: Int
    var _num_nodes: Int

    # CUDA handles (only meaningful on NVIDIA)
    var _graph: _CUptr
    var _exec: _CUptr
    var _mojo_stream: _CUptr
    var _replay_stream: _CUptr

    # Libraries kept alive for the lifetime of the graph
    var _cuda_ptr: UnsafePointer[OwnedDLHandle, MutAnyOrigin]
    var _intercept_ptr: UnsafePointer[OwnedDLHandle, MutAnyOrigin]

    def __init__(out self, ctx: DeviceContext) raises:
        """Initialize CUDA graph capture.

        On NVIDIA: loads libcuda.so + interceptor, discovers Mojo's
        internal CUDA stream (requires at least one prior kernel launch).

        On non-NVIDIA: initializes to disabled state.
        """
        self._state = 0
        self._num_nodes = 0
        self._graph = _CUptr()
        self._exec = _CUptr()
        self._mojo_stream = _CUptr()
        self._replay_stream = _CUptr()
        self._cuda_ptr = UnsafePointer[OwnedDLHandle, MutAnyOrigin]()
        self._intercept_ptr = UnsafePointer[OwnedDLHandle, MutAnyOrigin]()

        comptime if has_nvidia_gpu_accelerator():
            self._init_nvidia(ctx)

    def _init_nvidia(mut self, ctx: DeviceContext) raises:
        """Load CUDA driver + interceptor, discover stream."""
        self._cuda_ptr = alloc[OwnedDLHandle](1)
        self._cuda_ptr.init_pointee_move(OwnedDLHandle("libcuda.so"))

        self._intercept_ptr = alloc[OwnedDLHandle](1)
        self._intercept_ptr.init_pointee_move(
            OwnedDLHandle("./mojo_rl/cuda/libcuda_intercept.so")
        )

        # Get Mojo's internal stream from interceptor
        var get_stream = self._intercept_ptr[].get_function[
            def () -> _CUptr
        ]("intercept_get_mojo_stream")
        self._mojo_stream = get_stream()

        if Int(self._mojo_stream) == 0:
            print(
                "[CUDAGraph] WARNING: Mojo stream not yet discovered. "
                "Run at least one GPU kernel before creating CUDAGraph."
            )

        # Create replay stream
        var create_stream = self._cuda_ptr[].get_function[
            def (UnsafePointer[_CUptr, MutAnyOrigin], UInt32) -> c_int
        ]("cuStreamCreate")
        var stream_buf = alloc[_CUptr](1)
        stream_buf[] = _CUptr()
        _ = create_stream(stream_buf, UInt32(0))
        self._replay_stream = stream_buf[]
        stream_buf.free()

    def __del__(deinit self):
        """Clean up CUDA resources."""
        comptime if has_nvidia_gpu_accelerator():
            if self._state == 2 and self._cuda_ptr:
                var exec_destroy = self._cuda_ptr[].get_function[
                    def (_CUptr) -> c_int
                ]("cuGraphExecDestroy")
                _ = exec_destroy(self._exec)
                var graph_destroy = self._cuda_ptr[].get_function[
                    def (_CUptr) -> c_int
                ]("cuGraphDestroy")
                _ = graph_destroy(self._graph)
            if self._cuda_ptr:
                self._cuda_ptr.destroy_pointee()
                self._cuda_ptr.free()
            if self._intercept_ptr:
                self._intercept_ptr.destroy_pointee()
                self._intercept_ptr.free()

    def begin_capture(mut self) raises:
        """Begin capturing GPU kernel launches into a CUDA graph.

        All subsequent ctx.enqueue_function calls will be recorded
        until end_capture() is called. No-op on non-NVIDIA.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if Int(self._mojo_stream) == 0:
            raise Error(
                "[CUDAGraph] Cannot capture: Mojo stream not discovered. "
                "Ensure at least one GPU kernel ran before CUDAGraph init."
            )

        if self._state == 1:
            raise Error("[CUDAGraph] Already capturing.")

        # If we have a previous graph, destroy it
        if self._state == 2:
            var exec_destroy = self._cuda_ptr[].get_function[
                def (_CUptr) -> c_int
            ]("cuGraphExecDestroy")
            _ = exec_destroy(self._exec)
            var graph_destroy = self._cuda_ptr[].get_function[
                def (_CUptr) -> c_int
            ]("cuGraphDestroy")
            _ = graph_destroy(self._graph)
            self._exec = _CUptr()
            self._graph = _CUptr()

        var begin_fn = self._cuda_ptr[].get_function[
            def (_CUptr, c_int) -> c_int
        ]("cuStreamBeginCapture")
        var r = begin_fn(self._mojo_stream, c_int(0))
        if r != 0:
            raise Error(
                "[CUDAGraph] cuStreamBeginCapture failed: " + String(r)
            )
        self._state = 1

    def end_capture(mut self) raises:
        """End capture and instantiate the graph for replay.

        No-op on non-NVIDIA.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 1:
            raise Error("[CUDAGraph] Not capturing.")

        var graph_buf = alloc[_CUptr](1)
        graph_buf[] = _CUptr()
        var end_fn = self._cuda_ptr[].get_function[
            def (_CUptr, UnsafePointer[_CUptr, MutAnyOrigin]) -> c_int
        ]("cuStreamEndCapture")
        var r_end = end_fn(self._mojo_stream, graph_buf)
        self._graph = graph_buf[]
        graph_buf.free()

        if r_end != 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] cuStreamEndCapture failed: " + String(r_end)
            )

        # Count captured nodes
        var num_buf = alloc[UInt64](1)
        num_buf[] = UInt64(0)
        var get_nodes = self._cuda_ptr[].get_function[
            def (_CUptr, _CUptr, UnsafePointer[UInt64, MutAnyOrigin]) -> c_int
        ]("cuGraphGetNodes")
        _ = get_nodes(self._graph, _CUptr(), num_buf)
        self._num_nodes = Int(num_buf[])
        num_buf.free()

        if self._num_nodes == 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] Graph captured 0 nodes. "
                "Ensure LD_PRELOAD is set with libcuda_intercept.so."
            )

        # Instantiate
        var exec_buf = alloc[_CUptr](1)
        exec_buf[] = _CUptr()
        var instantiate = self._cuda_ptr[].get_function[
            def (UnsafePointer[_CUptr, MutAnyOrigin], _CUptr, UInt64) -> c_int
        ]("cuGraphInstantiate")
        var r_inst = instantiate(exec_buf, self._graph, UInt64(0))
        self._exec = exec_buf[]
        exec_buf.free()

        if r_inst != 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] cuGraphInstantiate failed: " + String(r_inst)
            )

        self._state = 2

    def replay(self) raises:
        """Replay the captured graph. No-op on non-NVIDIA.

        Launches on a dedicated replay stream and synchronizes.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error(
                "[CUDAGraph] No graph captured. "
                "Call begin_capture/end_capture first."
            )

        var launch = self._cuda_ptr[].get_function[
            def (_CUptr, _CUptr) -> c_int
        ]("cuGraphLaunch")
        _ = launch(self._exec, self._replay_stream)

        var sync_fn = self._cuda_ptr[].get_function[
            def (_CUptr) -> c_int
        ]("cuStreamSynchronize")
        _ = sync_fn(self._replay_stream)

    def replay_async(self) raises:
        """Replay without synchronizing. Call sync() later.

        No-op on non-NVIDIA.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        var launch = self._cuda_ptr[].get_function[
            def (_CUptr, _CUptr) -> c_int
        ]("cuGraphLaunch")
        _ = launch(self._exec, self._replay_stream)

    def sync(self) raises:
        """Synchronize the replay stream. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        var sync_fn = self._cuda_ptr[].get_function[
            def (_CUptr) -> c_int
        ]("cuStreamSynchronize")
        _ = sync_fn(self._replay_stream)

    def is_captured(self) -> Bool:
        """Whether a graph has been captured and is ready for replay."""
        comptime if not has_nvidia_gpu_accelerator():
            return False
        return self._state == 2

    def num_nodes(self) -> Int:
        """Number of kernel nodes in the captured graph."""
        return self._num_nodes
