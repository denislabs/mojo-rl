"""Minimal gate for `mojo_rl/cuda/device_graph.mojo`: record, replay, latch.

The successor to `test_cuda_graph_minimal.mojo`, and much smaller, because
most of what that file gates does not exist here: there is no interceptor to
be unhooked, no `LD_PRELOAD` to be missing, no borrowed stream to be destroyed
under us, and no silent zero-node capture. `DeviceGraph` is a supported API
that either records or raises.

## The arithmetic is the assertion, and it holds on BOTH platforms

N calls to `maybe_record_replay` must produce EXACTLY N increments, whichever
path ran:

    NVIDIA   call 1 runs STEP once (the warm-up) and RECORDS a second,
             non-executing pass; calls 2..N replay.        1 + (N-1) = N
    Apple    `DeviceGraph.create` raises, the slot latches disabled, and
             STEP runs directly every call.                    N

That equality is the whole contract — the caller's loop must not care which
happened — and it makes a Mac a real correctness check rather than a build
check. It also catches the two failures that would otherwise be silent: a
`build` that EXECUTES lands on N+1, and a replay that runs nothing lands on 1.

⚠ `test_platform_contract` is what stops the arithmetic test passing
vacuously. On Apple every arm of this file can pass with nothing recorded at
all, so the platform contract is pinned separately: NVIDIA must end up
RECORDED, everything else DISABLED. Without it, "N increments" only proves
`STEP` was called N times, which the disabled path does by definition.

Run with:
    pixi run -e nvidia mojo run -I . tests/cuda/test_device_graph_minimal.mojo
    pixi run -e apple  mojo run -I . tests/cuda/test_device_graph_minimal.mojo

    # bisect a suspected graph problem against a known-good run:
    MOJO_RL_DEVICE_GRAPH=0 pixi run -e nvidia mojo run -I . \
        tests/cuda/test_device_graph_minimal.mojo
"""

from std.gpu import thread_idx
from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_true, TestSuite
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.cuda import GraphSlot, maybe_record_replay

comptime CALLS = 5


def _read(ctx: DeviceContext, buf: DeviceBuffer[DT]) raises -> Float64:
    var h = ctx.enqueue_create_host_buffer[DT](1)
    ctx.enqueue_copy(h, buf)
    ctx.synchronize()
    return Float64(h[0])


def test_n_calls_make_n_increments() raises:
    """Recorded or not, `CALLS` calls advance the counter by exactly `CALLS`."""
    # ⚠ NESTED, not module-level. `TestSuite.discover_tests` instantiates every
    # top-level def for the HOST, and a GPU kernel fails there with "target
    # does not support operation: _get_intrinsic_name".
    @parameter
    @always_inline
    def _bump(buf: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin]):
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DT](1)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    # ⚠⚠ THE VIEW IS BUILT INSIDE THE STEP, NOT CAPTURED FROM A LOCAL, AND THE
    # FIRST DRAFT OF THIS TEST GOT IT WRONG.
    #
    #     var lt = LayoutTensor[DT, Layout.row_major(1)](buf)   # <- WRONG
    #     def _step(...): gctx.enqueue_function[_bump](lt, ...)
    #
    # Mojo destroys a value at its LAST USE, and a mention inside a closure
    # that is only ever passed as a COMPTIME PARAMETER does not count as one.
    # So `lt` died on its own construction line, every launch wrote through a
    # dead view, and the counter came back 0.0 with nothing raised anywhere.
    # `buf` is safe to capture precisely because `_read` mentions it below.
    #
    # ⚠ THE COMPILER TELLS YOU: "assignment to 'lt' was never used". In a file
    # whose next line hands `lt` to a kernel that warning reads like a false
    # positive. It is not — it is the whole bug.
    #
    # The production shape is the same rule one level up: one struct owns the
    # context and the buffers, and the closure mentions only that struct.
    #
    # ⚠ AND THE ARGUMENT IS THE POINT. A step that reaches for an outer `ctx`
    # instead of using `gctx` enqueues EAGERLY during the recording pass: the
    # work is still correct, the graph is still built, and it records NOTHING.
    # Nothing raises. The counter is what tells you — it lands on CALLS+1.
    def _step(gctx: DeviceContext) capturing raises -> None:
        var v = LayoutTensor[DT, Layout.row_major(1)](buf)
        gctx.enqueue_function[_bump](v, grid_dim=1, block_dim=1)

    var slot = GraphSlot()
    for _ in range(CALLS):
        maybe_record_replay[_step](slot, ctx)
    ctx.synchronize()

    var got = _read(ctx, buf)
    assert_true(
        got == Float64(CALLS),
        "expected exactly "
        + String(CALLS)
        + " increments, got "
        + String(got)
        + ". "
        + String(CALLS + 1)
        + " means the recording pass EXECUTED (the step enqueued on the outer"
        + " context, not the recording one). 1 means replay ran nothing.",
    )

    # ⚠ ORDER MATTERS: SLOT FIRST, BUFFER SECOND. The graph holds RAW POINTERS
    # into `buf`, so releasing the buffer while a live graph still references
    # it is a use-after-free waiting for the next replay. Mojo destroys a value
    # at its LAST USE, so these two lines ARE the ordering — reversing them is
    # a real bug, not a style choice. (Reading the counter above is also what
    # keeps `buf` alive that far; a timing arm of the DeviceGraph spike
    # crashed on exactly that omission.)
    _ = slot^
    _ = buf^


def test_platform_contract() raises:
    """NVIDIA records; every other device latches disabled.

    ⚠ THIS IS WHAT KEEPS THE ARITHMETIC TEST HONEST. "N calls, N increments"
    is satisfied by the disabled path trivially, so without pinning which path
    ran, an NVIDIA regression that silently stopped recording would still pass.
    """
    @parameter
    @always_inline
    def _bump(buf: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin]):
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DT](1)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    # View built inside the step — see `test_n_calls_make_n_increments`.
    def _step(gctx: DeviceContext) capturing raises -> None:
        var v = LayoutTensor[DT, Layout.row_major(1)](buf)
        gctx.enqueue_function[_bump](v, grid_dim=1, block_dim=1)

    var slot = GraphSlot()
    maybe_record_replay[_step](slot, ctx)
    maybe_record_replay[_step](slot, ctx)
    ctx.synchronize()

    comptime if has_nvidia_gpu_accelerator():
        assert_true(
            slot.is_recorded() and not slot.is_disabled(),
            "NVIDIA must RECORD. A latched-disabled slot here means"
            " DeviceGraph.create raised — read the printed reason. It is not"
            " an LD_PRELOAD problem: this path has no interceptor.",
        )
    else:
        assert_true(
            slot.is_disabled() and not slot.is_recorded(),
            "off CUDA/HIP, DeviceGraph.create must raise and the slot must"
            " latch disabled. Recording here would mean MAX grew a backend"
            " and this contract needs updating.",
        )
    _ = slot^
    _ = buf^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
