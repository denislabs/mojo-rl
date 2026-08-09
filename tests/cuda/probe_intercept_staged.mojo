"""Staged interceptor probe — which call in `CUDAGraph.__init__` dies?

`test_cuda_graph_minimal` crashes on NVIDIA, so the graph harness is broken
independently of SAC / replay / physics. This walks `CUDAGraph.__init__`'s
interceptor calls ONE AT A TIME, printing before and after each, so a single
run names the failing call instead of leaving us with an unwind whose frames
#3 and #6 are the same libc trampoline.

It does NOT use `CUDAGraph` — it loads the .so directly, so nothing is hidden
behind the wrapper.

⚠ RUN ON A TTY, NOT INTO A FILE. Mojo's `print` is block-buffered when stdout
is a pipe, and a SIGSEGV never flushes it — you would lose exactly the lines
that matter. If you must redirect, expect the tail to be missing.

THE HYPOTHESIS IT TESTS. `graph.mojo` unwraps every symbol with a trailing
`()`, per its own note that `_DLCallable.__call__` is variadic and discards
arguments. That works for symbols WITH arguments, where unwrap and call are
distinguishable. `intercept_get_mojo_stream` is the only ZERO-ARGUMENT one:

    var get_stream = lib.get_function[def() thin -> _CUptr](name)()
    self._mojo_stream = get_stream()

For a 0-arg signature both spellings are `()`, so if the first one CALLS
rather than unwraps, `get_stream` holds the STREAM VALUE typed as a function
pointer and the second `()` jumps to it. The reported `#7` equalled the
stream address the interceptor printed, which is what that would look like.

READ THE OUTPUT LIKE THIS:
  crash before "get_stream unwrapped"   -> dlopen/dlsym, not the ABI
  crash between unwrap and "stream ="   -> the 0-arg ambiguity above; the
                                           `_ARG` variant below is the fix
  "stream = 0x..." matches the value the
  interceptor printed                   -> unwrap is fine, look further down
  crash at stream_create / begin_capture-> the ambiguity is NOT the bug;
                                           suspect the interceptor vs the
                                           current Mojo runtime (stream
                                           discovery, cuGetProcAddress)
"""

from std.ffi import OwnedDLHandle, c_int
from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator
from std.memory import alloc
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT

comptime _CUptr = UnsafePointer[NoneType, MutUntrackedOrigin]


def main() raises:
    comptime if not has_nvidia_gpu_accelerator():
        print("non-NVIDIA: the interceptor path does not exist here. Skipped.")
        return

    var ctx = DeviceContext()

    @parameter
    @always_inline
    def _bump(buf: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin]):
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var buf = ctx.enqueue_create_buffer[DT](1)
    var lt = LayoutTensor[DT, Layout.row_major(1)](buf)

    # Stream discovery needs a prior launch (see CUDAGraph.__init__).
    print("[probe] warmup launch ...")
    ctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)
    ctx.synchronize()
    print("[probe] warmup done")

    print("[probe] dlopen libcuda_intercept.so ...")
    var lib = OwnedDLHandle("./mojo_rl/cuda/libcuda_intercept.so")
    print("[probe] dlopen ok")

    # ── 1. the zero-argument symbol, the suspect ──────────────────────────
    print("[probe] resolving intercept_get_mojo_stream (0-arg) ...")
    var get_stream = lib.get_function[def () thin -> _CUptr](
        "intercept_get_mojo_stream"
    )()
    print("[probe] get_stream unwrapped")
    var stream = get_stream()
    print("[probe] stream =", Int(stream))

    # ── 2. the same symbol through a ONE-ARGUMENT signature ───────────────
    # Calling a 0-arg C function through a 1-arg pointer is ABI-safe on
    # x86-64 SysV (the extra argument sits in a register the callee ignores),
    # and it makes unwrap and call textually distinct — so this reaches the C
    # function even if the 0-arg spelling above does not.
    print("[probe] resolving the same symbol as 1-arg ...")
    var get_stream1 = lib.get_function[def (_CUptr) thin -> _CUptr](
        "intercept_get_mojo_stream"
    )()
    print("[probe] get_stream1 unwrapped")
    # A null pointer as the ignored argument.
    var null_arg = alloc[_CUptr](1)
    var stream1 = get_stream1(null_arg.bitcast[NoneType]())
    null_arg.free()
    print("[probe] stream via 1-arg =", Int(stream1))
    print(
        "[probe] the two agree:",
        Int(stream) == Int(stream1),
        " (disagreement means the 0-arg spelling is the bug)",
    )

    if Int(stream1) == 0:
        print("[probe] STREAM NOT DISCOVERED — the interceptor never saw a")
        print("        launch. That is the dlsym-hook-vs-Mojo-runtime break,")
        print("        not an ABI problem.")
        return

    # ── 3. replay-stream creation ─────────────────────────────────────────
    print("[probe] intercept_stream_create ...")
    var stream_create = lib.get_function[
        def (UnsafePointer[_CUptr, MutUntrackedOrigin]) thin -> c_int
    ]("intercept_stream_create")()
    var sbuf = alloc[_CUptr](1)
    var rc_create = stream_create(sbuf)
    print("[probe] stream_create rc =", Int(rc_create),
          " stream =", Int(sbuf[]))
    sbuf.free()

    # ── 4. begin / end capture around one launch ──────────────────────────
    print("[probe] intercept_stream_begin_capture ...")
    var begin_cap = lib.get_function[def (_CUptr) thin -> c_int](
        "intercept_stream_begin_capture"
    )()
    var rc_begin = begin_cap(stream1)
    print("[probe] begin_capture rc =", Int(rc_begin))
    if Int(rc_begin) != 0:
        print("[probe] begin_capture FAILED — stop here.")
        return

    ctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)
    print("[probe] kernel enqueued under capture")

    var graph_buf = alloc[_CUptr](1)
    var end_cap = lib.get_function[
        def (_CUptr, UnsafePointer[_CUptr, MutUntrackedOrigin]) thin -> c_int
    ]("intercept_stream_end_capture")()
    var rc_end = end_cap(stream1, graph_buf)
    print("[probe] end_capture rc =", Int(rc_end),
          " graph =", Int(graph_buf[]))
    graph_buf.free()
    print("[probe] SURVIVED the whole sequence.")
