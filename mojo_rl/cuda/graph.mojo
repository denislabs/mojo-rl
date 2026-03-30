"""CUDA Graph capture and replay for Mojo GPU kernels.

On Apple Silicon / non-NVIDIA platforms, all methods are compile-time
no-ops with zero overhead.

Usage:
    from mojo_rl.cuda import CUDAGraph

    var graph = CUDAGraph(ctx)
    graph.begin_capture()
    my_gpu_kernels(ctx, ...)
    graph.end_capture()

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
    All CUDA driver calls go through the interceptor library to avoid
    re-entrant dlsym issues.
    """

    var _state: Int       # 0=init, 1=capturing, 2=captured
    var _num_nodes: Int
    var _graph: _CUptr
    var _exec: _CUptr
    var _mojo_stream: _CUptr
    var _replay_stream: _CUptr
    var _lib: UnsafePointer[OwnedDLHandle, MutAnyOrigin]

    def __init__(out self, ctx: DeviceContext) raises:
        """Initialize CUDA graph capture.

        On NVIDIA: loads interceptor, discovers Mojo's internal stream.
        Requires at least one prior kernel launch for stream discovery.
        On non-NVIDIA: disabled state, all methods are no-ops.
        """
        self._state = 0
        self._num_nodes = 0
        self._graph = _CUptr()
        self._exec = _CUptr()
        self._mojo_stream = _CUptr()
        self._replay_stream = _CUptr()
        self._lib = UnsafePointer[OwnedDLHandle, MutAnyOrigin]()

        comptime if has_nvidia_gpu_accelerator():
            self._init_nvidia(ctx)

    def _init_nvidia(mut self, ctx: DeviceContext) raises:
        ctx.synchronize()

        self._lib = alloc[OwnedDLHandle](1).bitcast[
            OwnedDLHandle, origin=MutAnyOrigin
        ]()
        self._lib.init_pointee_move(
            OwnedDLHandle("./mojo_rl/cuda/libcuda_intercept.so")
        )

        # Get Mojo's internal stream
        var get_stream = self._lib[].get_function[
            def () -> _CUptr
        ]("intercept_get_mojo_stream")
        self._mojo_stream = get_stream()

        if Int(self._mojo_stream) == 0:
            print(
                "[CUDAGraph] WARNING: Mojo stream not yet discovered. "
                "Run at least one GPU kernel before creating CUDAGraph."
            )
            return

        # Create replay stream via interceptor wrapper
        var stream_create = self._lib[].get_function[
            def (UnsafePointer[_CUptr, MutAnyOrigin]) -> c_int
        ]("intercept_stream_create")
        var stream_buf = alloc[_CUptr](1)
        stream_buf[] = _CUptr()
        var r = stream_create(stream_buf)
        if r != 0:
            stream_buf.free()
            raise Error("[CUDAGraph] cuStreamCreate failed: " + String(r))
        self._replay_stream = stream_buf[]
        stream_buf.free()

    def __del__(deinit self):
        comptime if has_nvidia_gpu_accelerator():
            if self._state == 2 and self._lib:
                var exec_destroy = self._lib[].get_function[
                    def (_CUptr) -> c_int
                ]("intercept_graph_exec_destroy")
                _ = exec_destroy(self._exec)
                var graph_destroy = self._lib[].get_function[
                    def (_CUptr) -> c_int
                ]("intercept_graph_destroy")
                _ = graph_destroy(self._graph)
            if self._lib:
                self._lib.destroy_pointee()
                self._lib.bitcast[OwnedDLHandle]().free()

    def begin_capture(mut self) raises:
        """Begin capturing GPU kernel launches. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        if Int(self._mojo_stream) == 0:
            raise Error(
                "[CUDAGraph] Mojo stream not discovered. "
                "Ensure at least one GPU kernel ran before init."
            )
        if self._state == 1:
            raise Error("[CUDAGraph] Already capturing.")

        # Destroy previous graph if re-capturing
        if self._state == 2:
            var ed = self._lib[].get_function[
                def (_CUptr) -> c_int
            ]("intercept_graph_exec_destroy")
            _ = ed(self._exec)
            var gd = self._lib[].get_function[
                def (_CUptr) -> c_int
            ]("intercept_graph_destroy")
            _ = gd(self._graph)
            self._exec = _CUptr()
            self._graph = _CUptr()

        var begin_fn = self._lib[].get_function[
            def (_CUptr) -> c_int
        ]("intercept_stream_begin_capture")
        var r = begin_fn(self._mojo_stream)
        if r != 0:
            raise Error(
                "[CUDAGraph] cuStreamBeginCapture failed: " + String(r)
            )
        self._state = 1

    def end_capture(mut self) raises:
        """End capture and instantiate the graph. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 1:
            raise Error("[CUDAGraph] Not capturing.")

        # End capture
        var graph_buf = alloc[_CUptr](1)
        graph_buf[] = _CUptr()
        var end_fn = self._lib[].get_function[
            def (_CUptr, UnsafePointer[_CUptr, MutAnyOrigin]) -> c_int
        ]("intercept_stream_end_capture")
        var r_end = end_fn(self._mojo_stream, graph_buf)
        self._graph = graph_buf[]
        graph_buf.free()

        if r_end != 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] cuStreamEndCapture failed: " + String(r_end)
            )

        # Count nodes
        var num_buf = alloc[UInt64](1)
        num_buf[] = UInt64(0)
        var get_nodes = self._lib[].get_function[
            def (_CUptr, UnsafePointer[UInt64, MutAnyOrigin]) -> c_int
        ]("intercept_graph_get_nodes")
        _ = get_nodes(self._graph, num_buf)
        self._num_nodes = Int(num_buf[])
        num_buf.free()

        if self._num_nodes == 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] Captured 0 nodes. "
                "Ensure LD_PRELOAD is set with libcuda_intercept.so."
            )

        # Instantiate
        var exec_buf = alloc[_CUptr](1)
        exec_buf[] = _CUptr()
        var inst = self._lib[].get_function[
            def (UnsafePointer[_CUptr, MutAnyOrigin], _CUptr) -> c_int
        ]("intercept_graph_instantiate")
        var r_inst = inst(exec_buf, self._graph)
        self._exec = exec_buf[]
        exec_buf.free()

        if r_inst != 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] cuGraphInstantiate failed: " + String(r_inst)
            )

        self._state = 2

    def replay(self) raises:
        """Replay the captured graph (launch + sync). No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        var launch = self._lib[].get_function[
            def (_CUptr, _CUptr) -> c_int
        ]("intercept_graph_launch")
        _ = launch(self._exec, self._replay_stream)

        var sync_fn = self._lib[].get_function[
            def (_CUptr) -> c_int
        ]("intercept_stream_synchronize")
        _ = sync_fn(self._replay_stream)

    def replay_async(self) raises:
        """Replay without sync. Call sync() later. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        var launch = self._lib[].get_function[
            def (_CUptr, _CUptr) -> c_int
        ]("intercept_graph_launch")
        _ = launch(self._exec, self._replay_stream)

    def sync(self) raises:
        """Synchronize the replay stream. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        var sync_fn = self._lib[].get_function[
            def (_CUptr) -> c_int
        ]("intercept_stream_synchronize")
        _ = sync_fn(self._replay_stream)

    def is_captured(self) -> Bool:
        """Whether a graph is ready for replay."""
        comptime if not has_nvidia_gpu_accelerator():
            return False
        return self._state == 2

    def num_nodes(self) -> Int:
        """Number of kernel nodes in the captured graph."""
        return self._num_nodes
