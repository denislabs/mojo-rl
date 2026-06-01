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


@always_inline
def _uninit[T: AnyType](out value: T):
    """Returns uninitialized data."""
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(value))


struct CUDAGraph(Movable):
    """CUDA Graph capture and replay.

    Compile-time gated: all methods are no-ops on non-NVIDIA platforms.
    All CUDA driver calls go through the interceptor library wrappers.
    """

    var _state: Int  # 0=init, 1=capturing, 2=captured
    var _num_nodes: Int
    var _graph: _CUptr
    var _exec: _CUptr
    var _mojo_stream: _CUptr
    var _replay_stream: _CUptr
    var _lib: OwnedDLHandle
    # Cached `intercept_graph_launch` pointer — resolved once in `__init__`
    # instead of re-`dlsym`'d on every replay (replays happen tens of
    # thousands of times per run). NVIDIA-only; uninit + never called on
    # non-NVIDIA (the replay methods comptime-return there).
    var _launch_fn: def (_CUptr, _CUptr) thin -> c_int

    def __init__(out self, ctx: DeviceContext) raises:
        """Initialize CUDA graph capture.

        On NVIDIA: loads interceptor, discovers Mojo's internal stream.
        Requires at least one prior kernel launch for stream discovery.
        On non-NVIDIA: disabled state, all methods are no-ops.
        """
        self._state = 0
        self._num_nodes = 0
        # Raw _CUptr fields: NVIDIA path overwrites via interceptor calls
        # below; non-NVIDIA paths never read these (compile-time guarded).
        self._graph = _uninit[_CUptr]()
        self._exec = _uninit[_CUptr]()
        self._mojo_stream = _uninit[_CUptr]()
        self._replay_stream = _uninit[_CUptr]()

        self._launch_fn = _uninit[def (_CUptr, _CUptr) thin -> c_int]()

        comptime if has_nvidia_gpu_accelerator():
            ctx.synchronize()
            self._lib = OwnedDLHandle("./mojo_rl/cuda/libcuda_intercept.so")

            # Resolve the hot graph-launch symbol once (used by every replay).
            self._launch_fn = self._lib.get_function[
                def (_CUptr, _CUptr) thin -> c_int
            ]("intercept_graph_launch")

            # Get Mojo's internal stream
            var get_stream = self._lib.get_function[def() thin -> _CUptr](
                "intercept_get_mojo_stream"
            )
            self._mojo_stream = get_stream()

            if Int(self._mojo_stream) == 0:
                print(
                    "[CUDAGraph] WARNING: Mojo stream not discovered. "
                    "Run at least one GPU kernel before creating CUDAGraph."
                )
            else:
                # Create replay stream
                var stream_create = self._lib.get_function[
                    def(UnsafePointer[_CUptr, MutAnyOrigin]) thin -> c_int
                ]("intercept_stream_create")
                var stream_buf = alloc[_CUptr](1)
                stream_buf[] = _uninit[_CUptr]()
                _ = stream_create(stream_buf)
                self._replay_stream = stream_buf[]
                stream_buf.free()
        else:
            self._lib = _uninit[OwnedDLHandle]()

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
            _ = self._lib.get_function[def(_CUptr) thin -> c_int](
                "intercept_graph_exec_destroy"
            )(self._exec)
            _ = self._lib.get_function[def(_CUptr) thin -> c_int](
                "intercept_graph_destroy"
            )(self._graph)
            self._exec = _uninit[_CUptr]()
            self._graph = _uninit[_CUptr]()

        var r = self._lib.get_function[def(_CUptr) thin -> c_int](
            "intercept_stream_begin_capture"
        )(self._mojo_stream)
        if r != 0:
            raise Error("[CUDAGraph] cuStreamBeginCapture failed: " + String(r))
        self._state = 1

    def end_capture(mut self) raises:
        """End capture and instantiate the graph. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 1:
            raise Error("[CUDAGraph] Not capturing.")

        # End capture
        var graph_buf = alloc[_CUptr](1)
        graph_buf[] = _uninit[_CUptr]()
        var r_end = self._lib.get_function[
            def(_CUptr, UnsafePointer[_CUptr, MutAnyOrigin]) thin -> c_int
        ]("intercept_stream_end_capture")(self._mojo_stream, graph_buf)
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
        _ = self._lib.get_function[
            def(_CUptr, UnsafePointer[UInt64, MutAnyOrigin]) thin -> c_int
        ]("intercept_graph_get_nodes")(self._graph, num_buf)
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
        exec_buf[] = _uninit[_CUptr]()
        var r_inst = self._lib.get_function[
            def(UnsafePointer[_CUptr, MutAnyOrigin], _CUptr) thin -> c_int
        ]("intercept_graph_instantiate")(exec_buf, self._graph)
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

        _ = self._launch_fn(self._exec, self._replay_stream)

        _ = self._lib.get_function[def(_CUptr) thin -> c_int](
            "intercept_stream_synchronize"
        )(self._replay_stream)

    def replay_async(self) raises:
        """Replay without sync on replay stream. Call sync() later. No-op on non-NVIDIA.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        _ = self._launch_fn(self._exec, self._replay_stream)

    def replay_on_mojo_stream(self) raises:
        """Replay on Mojo's main stream (implicit ordering with other kernels).

        Unlike replay_async() which uses a separate replay stream,
        this launches the graph on the same stream used by ctx.enqueue_function.
        This ensures correct ordering: kernels enqueued before this call
        (e.g. select_actions_gpu) complete before the graph starts,
        and the graph completes before kernels enqueued after.

        No explicit sync needed between pre-graph and post-graph operations.
        No-op on non-NVIDIA.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        _ = self._launch_fn(self._exec, self._mojo_stream)

    def sync(self) raises:
        """Synchronize the replay stream. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        _ = self._lib.get_function[def(_CUptr) thin -> c_int](
            "intercept_stream_synchronize"
        )(self._replay_stream)

    def is_captured(self) -> Bool:
        """Whether a graph is ready for replay."""
        comptime if not has_nvidia_gpu_accelerator():
            return False
        return self._state == 2

    def num_nodes(self) -> Int:
        """Number of kernel nodes in the captured graph."""
        return self._num_nodes


# ──────────────────────────────────────────────────────────────────────
# maybe_capture_replay — capture-lifecycle harness behind a closure.
# ──────────────────────────────────────────────────────────────────────


def maybe_capture_replay[
    STEP: def () capturing raises -> None,
](mut graph: Optional[CUDAGraph], ctx: DeviceContext) raises:
    """Capture `STEP` into `graph` on first call; replay it thereafter.

    One generic helper owns the whole capture lifecycle (warmup → begin →
    capture → end → replay) behind a comptime *capturing closure*, so a
    training loop never inlines a `comptime if USE_CUDA_GRAPH` maze and the
    caller's trainer never learns about `CUDAGraph`.

    `STEP` is a comptime capturing closure that mutably captures the caller's
    state (e.g. the trainer + ctx) and enqueues the *pure device-kernel* step
    — NO host work. It must enqueue the SAME kernel sequence every call so the
    captured graph stays valid on replay (sampling included, so each replay
    draws a fresh minibatch via the device RNG counter).

    `graph` is the caller-owned capture slot (None until first capture). On
    NVIDIA: the first call runs `STEP` once to settle the stream, then
    captures a second run; later calls replay on the Mojo stream (implicit
    ordering with kernels enqueued before/after). On non-NVIDIA: `CUDAGraph`
    is a compile-time no-op, so this just runs `STEP()` every call —
    bit-identical to the non-captured path (the Apple-Silicon verification
    path; real capture/replay needs an NVIDIA run).

    Host bookkeeping (step counters, metric flush cadence) is intentionally
    NOT here — keep it in the caller's loop, advanced once per logical update,
    so it stays correct whether the step ran directly or via replay."""
    comptime if has_nvidia_gpu_accelerator():
        if not graph:
            STEP()
            ctx.synchronize()
            var g = CUDAGraph(ctx)
            g.begin_capture()
            STEP()
            g.end_capture()
            graph = g^
        else:
            graph.value().replay_on_mojo_stream()
    else:
        STEP()
