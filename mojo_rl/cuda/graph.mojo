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
from std.memory import alloc, dealloc
from max.gpu.host import DeviceContext


comptime _CUptr = Pointer[NoneType, MutUntrackedOrigin]


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
    # `intercept_graph_launch` used to be cached here as a raw function
    # pointer. Mojo 1.0's `get_function` returns a `_DLCallable` whose origin
    # is `origin_of(self._lib)` — it deliberately borrows the handle so the
    # library cannot be unloaded mid-call — and that origin cannot be named in
    # a field type, so the callable is resolved per replay instead. `dlsym` on
    # an already-open handle is a hash lookup; against a graph launch plus a
    # stream synchronize it is not measurable.

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

        comptime if has_nvidia_gpu_accelerator():
            ctx.synchronize()
            self._lib = OwnedDLHandle("./mojo_rl/cuda/libcuda_intercept.so")

            # ⚠ `get_function`'s parameter is the symbol's RETURN type, and
            # the returned `_DLCallable` IS the callable — there is no unwrap
            # step. This file previously used the pre-1.0 spelling
            # `get_function[def(A, B) thin -> R](name)()`, where the parameter
            # was the whole function type and the trailing `()` unwrapped it.
            # Under 1.0 that spelling still COMPILES — a function type is a
            # valid `RegisterPassable` return type — but the trailing `()`
            # CALLS the C function with no arguments and types its return value
            # as a function pointer, which the next call then jumps into. See
            # `tests/cuda/test_dlhandle_get_function_arity.mojo`.
            #
            # Get Mojo's internal stream
            var get_stream = self._lib.get_function[_CUptr]("intercept_get_mojo_stream")
            self._mojo_stream = get_stream()

            if Int(self._mojo_stream) == 0:
                print(
                    "[CUDAGraph] WARNING: Mojo stream not discovered. "
                    "Run at least one GPU kernel before creating CUDAGraph."
                )
            else:
                # Create replay stream
                var stream_create = self._lib.get_function[c_int](
                    "intercept_stream_create"
                )
                var stream_alloc = alloc[_CUptr]({count = 1})
                var stream_buf = stream_alloc.unsafe_ptr()
                stream_buf.unsafe_write(_uninit[_CUptr]())
                _ = stream_create(stream_buf)
                self._replay_stream = stream_buf[]
                dealloc(stream_alloc^)
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
            var exec_destroy = self._lib.get_function[c_int]("intercept_graph_exec_destroy")
            _ = exec_destroy(self._exec)
            var graph_destroy = self._lib.get_function[c_int]("intercept_graph_destroy")
            _ = graph_destroy(self._graph)
            self._exec = _uninit[_CUptr]()
            self._graph = _uninit[_CUptr]()

        # ⚠ A PRE-CAPTURE `intercept_stream_synchronize` USED TO BE HERE AND
        # WAS REMOVED — IT IS THE CALL THAT CRASHED. It was added to keep the
        # interceptor's sync-suppression honest (drain before the window so
        # answering "already idle" is true). The empty-capture probe then
        # faulted inside the driver ON THAT VERY CALL, with no capture active,
        # which is what proved the failure is not capture-specific: this shim
        # cannot safely call the driver with MAX's stream handle at all.
        # Do not reinstate it without fixing that first.
        var begin_capture = self._lib.get_function[c_int]("intercept_stream_begin_capture")
        var r = begin_capture(self._mojo_stream)
        if r != 0:
            raise Error("[CUDAGraph] cuStreamBeginCapture failed: " + String(r))
        self._state = 1

    def end_capture(mut self) raises:
        """End capture and instantiate the graph. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 1:
            raise Error("[CUDAGraph] Not capturing.")

        # End capture.
        #
        # Symbol resolution is hoisted ABOVE each allocation throughout this
        # method. `get_function` raises if the symbol is missing, and an
        # `Allocation` must be consumed on every path out of the scope — so a
        # raising call sandwiched between the allocation and its `dealloc`
        # leaks the buffer (and does not compile). Resolving first leaves no
        # raising call in the window, which is why none of these need a
        # `try`/`except`.
        var end_capture = self._lib.get_function[c_int](
            "intercept_stream_end_capture"
        )
        var graph_alloc = alloc[_CUptr]({count = 1})
        var graph_buf = graph_alloc.unsafe_ptr()
        graph_buf.unsafe_write(_uninit[_CUptr]())
        var r_end = end_capture(self._mojo_stream, graph_buf)
        self._graph = graph_buf[]
        dealloc(graph_alloc^)

        if r_end != 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] cuStreamEndCapture failed: " + String(r_end)
            )

        # Count nodes
        var get_nodes = self._lib.get_function[c_int](
            "intercept_graph_get_nodes"
        )
        var num_alloc = alloc[UInt64]({count = 1})
        var num_buf = num_alloc.unsafe_ptr()
        num_buf.unsafe_write(UInt64(0))
        _ = get_nodes(self._graph, num_buf)
        self._num_nodes = Int(num_buf[])
        dealloc(num_alloc^)

        if self._num_nodes == 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] Captured 0 nodes. "
                "Ensure LD_PRELOAD is set with libcuda_intercept.so."
            )

        # Instantiate
        var instantiate = self._lib.get_function[c_int](
            "intercept_graph_instantiate"
        )
        var exec_alloc = alloc[_CUptr]({count = 1})
        var exec_buf = exec_alloc.unsafe_ptr()
        exec_buf.unsafe_write(_uninit[_CUptr]())
        var r_inst = instantiate(exec_buf, self._graph)
        self._exec = exec_buf[]
        dealloc(exec_alloc^)

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

        var launch = self._lib.get_function[c_int]("intercept_graph_launch")
        _ = launch(self._exec, self._replay_stream)

        var stream_sync = self._lib.get_function[c_int]("intercept_stream_synchronize")
        _ = stream_sync(self._replay_stream)

    def replay_async(self) raises:
        """Replay without sync on replay stream. Call sync() later. No-op on non-NVIDIA.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        var launch = self._lib.get_function[c_int]("intercept_graph_launch")
        _ = launch(self._exec, self._replay_stream)

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

        var launch = self._lib.get_function[c_int]("intercept_graph_launch")
        _ = launch(self._exec, self._mojo_stream)

    def sync(self) raises:
        """Synchronize the replay stream. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        var stream_sync = self._lib.get_function[c_int]("intercept_stream_synchronize")
        _ = stream_sync(self._replay_stream)

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
            # Diagnostic: confirm the harness actually captured kernels (a
            # tiny/zero node count means capture silently failed — e.g. the
            # closure enqueued on a different stream than the one being
            # captured). Printed once per graph (first call only).
            print(
                "[CUDA Graph] maybe_capture_replay captured",
                g.num_nodes(),
                "nodes",
            )
            graph = g^
        else:
            graph.value().replay_on_mojo_stream()
    else:
        STEP()
