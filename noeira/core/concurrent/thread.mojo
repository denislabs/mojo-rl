# +--------------------------------------------------------------------------+ #
# | pthread, because Mojo has no async/await and does not need it
# +--------------------------------------------------------------------------+ #
"""One OS thread, spawned and joined. The bottom of the concurrency backbone.

    def my_entry(arg: OpaquePtr) -> OpaquePtr:
        ...                                   # must NOT raise
        return null_opaque()

    var t = ThreadHandle.spawn[my_entry](arg)
    t.join()

`runtime.asyncrt.Task` is documented at `docs.modular.com` but is NOT in the
standalone Mojo package — re-verified on Mojo 1.0.0 (`ed45d567`), where
`from runtime.asyncrt import ...` is `unable to locate module 'runtime'` and no
`runtime.mojoc` exists in `.pixi/envs/*/lib/mojo/` even with MAX installed. It
is also not what this tree wants: see `docs/CONCURRENCY_BACKBONE.md`.

`references/flare-main/flare/runtime/_thread.mojo` is the source of this shape,
and it pins `mojo >=1.0.0,<1.1.0` — the compiler we run — so its idioms are
known to compile here. This file is deliberately a near-transcription.

⚠ THE START ROUTINE IS A COMPTIME PARAMETER, NOT AN ARGUMENT. C's
`pthread_create` takes `void *(*)(void *)` — a bare function pointer with no
environment — and Mojo has no closure that can cross a thread boundary. So the
entry is passed as a parameter and all its state arrives through the single
opaque `arg`. `worker.mojo` builds the generic-thunk trick on top of this.

⚠ THE START ROUTINE MUST NOT RAISE. pthread has no exception channel; a `raise`
crossing the boundary is undefined. Catch everything inside and encode failure
in shared state.

⚠ A THREAD THAT OUTLIVES ITS `ThreadHandle` IS UNDEFINED BEHAVIOUR, and so is
freeing anything it still reads. Always `join()` before dropping either.
"""

from std.ffi import external_call, c_int
from std.memory import Pointer


comptime OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]
"""The `void*` that crosses the thread boundary. `MutUntrackedOrigin` because
the borrow checker cannot see the other thread's use of it — that is the whole
point, and it is why everything here is unsafe by construction."""


@always_inline
def null_opaque() -> OpaquePtr:
    """A C `NULL` of the flavour used throughout.

    `Pointer` is non-nullable and rejects a comptime-literal address of 0, but
    pthread genuinely needs NULL for the attr argument, the retval slot, and a
    start routine's return. Building it from a *runtime* zero sidesteps the
    non-null constraint."""
    var zero = 0
    return OpaquePtr(unsafe_from_address=zero)


@always_inline
def opaque_from_address(addr: Int) -> OpaquePtr:
    """Address -> `void*`. The only sanctioned way to hand a shared block to a
    thread: an `Int` carries no origin, so nothing about the receiving side's
    use of it can be checked, or needs to be."""
    return OpaquePtr(unsafe_from_address=addr)


struct ThreadHandle(Movable):
    """Owning handle to one live OS thread.

    Move-only on purpose. A `pthread_t` names one thread and POSIX forbids
    joining it twice; making the handle non-`Copyable` puts "exactly one owner,
    exactly one join" in the type system rather than in a comment.

    `pthread_t` is `unsigned long` on Linux and an opaque pointer on macOS —
    both 64 bits, both stored here as `UInt64`. Do not read the bit pattern.
    """

    var _tid: UInt64
    """Opaque `pthread_t`. Zeroed by a successful `join()` so a second call on
    *this handle* short-circuits instead of joining a stale id (which is UB)."""

    def __init__(out self, tid: UInt64):
        self._tid = tid

    def __init__(out self, *, deinit move: Self):
        self._tid = move._tid

    @staticmethod
    def spawn[
        start: def (OpaquePtr) thin -> OpaquePtr
    ](arg: OpaquePtr) raises -> ThreadHandle:
        """Start a thread running `start(arg)`.

        Parameters:
            start: Entry point. Must not raise, and must not capture — a plain
                top-level `def` with this signature is ABI-compatible with C's
                `void *(*)(void *)` on both Linux x86_64 and macOS arm64.

        Args:
            arg: The single opaque pointer the entry receives. Everything the
                thread needs must be reachable from it.

        Returns:
            A handle the caller MUST `join()`.

        Raises:
            Error: `pthread_create` returned non-zero.
        """
        var tid = UInt64(0)
        var tid_ptr = Pointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=Int(Pointer(to=tid))
        )
        var rc = external_call[
            "pthread_create",
            c_int,
            Pointer[UInt64, MutUntrackedOrigin],  # pthread_t *thread
            OpaquePtr,  # const pthread_attr_t *attr (NULL = joinable)
            def (OpaquePtr) thin -> OpaquePtr,  # void *(*start)(void *)
            OpaquePtr,  # void *arg
        ](tid_ptr, null_opaque(), start, arg)
        if rc != c_int(0):
            raise Error("pthread_create failed, rc=" + String(Int(rc)))
        return ThreadHandle(tid)

    def join(mut self) raises:
        """Block until the thread returns, discarding its value.

        Idempotent: after a successful join `_tid` is zeroed, so calling again
        on the same handle returns immediately rather than running
        `pthread_join` on a stale id.

        ⚠ THIS CANNOT BE BOUNDED. There is no portable `pthread_timedjoin`
        (Linux has `_np`, macOS has nothing), so a worker that never returns
        hangs the caller forever. Bounding the wait is the WORKER's job — see
        `WorkerCtl.drain_deadline_passed` in `worker.mojo`, and the measured
        15.0s -> 5.0s in `docs/design_spikes/spike_bounded_close_hung_dashboard.mojo`.

        Raises:
            Error: `pthread_join` returned non-zero. The handle is left intact
                so the caller can retry or propagate.
        """
        if self._tid == 0:
            return
        var rc = external_call["pthread_join", c_int, UInt64, OpaquePtr](
            self._tid, null_opaque()
        )
        if rc != c_int(0):
            raise Error("pthread_join failed, rc=" + String(Int(rc)))
        self._tid = UInt64(0)

    def joined(self) -> Bool:
        """Whether this handle has already been successfully joined."""
        return self._tid == 0


@always_inline
def sleep_us(microseconds: Int) -> Int:
    """Sleep this thread for at least `microseconds`.

    ⚠ THE RETURN TYPE MUST BE `Int32`, NOT `std.ffi.c_int`. With `c_int` the
    compiler reports "existing function with conflicting signature" and then
    "failed to legalize operation 'pop.external_call'" — a LOWERING failure
    with no source line pointing at the cause.

    ⚠ DO NOT DECLARE `nanosleep` ANYWHERE IN A MODULE THAT ALSO IMPORTS
    `std.time.sleep`. `sleep` emits its own `nanosleep` prototype and the two
    collide the same way.

    `flare/runtime/_libc_time.mojo` opens with a warning that `usleep` through
    `external_call` overshoots by 1000-1500x. That does NOT reproduce here:
    measured at 1.5x of a 500us request on Mojo 1.0.0 / macOS arm64, which is
    ordinary OS wakeup latency (`docs/design_spikes/probe_sleep_primitives.mojo`).
    Do not architect around it.

    Returns:
        0 on success, -1 if a signal cut the sleep short.
    """
    if microseconds <= 0:
        return 0
    return Int(external_call["usleep", Int32, Int32](Int32(microseconds)))
