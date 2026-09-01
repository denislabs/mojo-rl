"""Does MAX's own `DeviceGraph` work here — and can it replace our LD_PRELOAD shim?

    pixi run -e nvidia mojo run -I . \\
        benchmarks/bench_device_graph_spike.mojo

    # the point of arm 0: NO interceptor, NO activation, NO LD_PRELOAD.
    pixi run -e nvidia mojo build -I . -o /tmp/dg_spike \\
        benchmarks/bench_device_graph_spike.mojo
    /tmp/dg_spike                      # <- run it DIRECTLY, bare shell

    # skip the arm that may kill the process (see arm D):
    MOJO_RL_SPIKE_SPLITK=0 pixi run -e nvidia mojo run -I . \\
        benchmarks/bench_device_graph_spike.mojo

## Why this exists

`docs/MODULAR_SOURCE_DEEP_DIVE.md` found that MAX 26.5 SHIPS the API our
`mojo_rl/cuda/graph.mojo` reimplements by hand:

    from max.gpu.host import DeviceGraph, DeviceGraphBuilder

1616 lines in `max.mojoc`, exported from `max/gpu/host/__init__.mojo`, with
MAX's own test at
`references/modular-max-v26.5/max/kernels/test/gpu/device_context/test_device_graph_builder.mojo`
(this file's syntax is copied from it, deliberately).

What we have instead is ~400 lines of `LD_PRELOAD` interceptor that discovers
MAX's INTERNAL stream by watching `cuLaunchKernelEx` and then calls
`cuStreamBeginCapture` on a stream we do not own. That file's own comments are
the bill for it: `cuStreamDestroy` on our handle, driver SIGSEGVs in
BeginCapture / EndCapture / GetCaptureInfo, a load-bearing `_ctx` field, and a
failure mode where capture "succeeds" with ZERO nodes and nothing says so. Plus
[[_the_preload_is_a_property_of_the_process_not_the_binary]]: a binary you
BUILD and then run directly silently loses capture entirely.

`DeviceGraph` is a different mechanism — explicit DAG construction, not stream
capture — so none of that applies. It is a supported API, it needs no preload,
and `recording_context()` claims that ordinary `DeviceContext` code records
unmodified, which would mean our existing `STEP()` closures port as-is.

⚠ THAT IS A DOCSTRING, NOT A MEASUREMENT. This file is the measurement.

## The arms, in order, each independently reportable

  0. is it even here?      imports resolve, `DeviceContext` alive, device named.
  A. add_function          the explicit-DAG path: one node, build must NOT
                           execute, replay must execute exactly once.
  B. recording_context     OUR kernel style (LayoutTensor args) through the
                           ordinary `enqueue_function` API, four launches, order
                           preserved. If this passes, a `STEP()` closure that
                           takes a `DeviceContext` records with NO shim.
  C. a MAX library GEMM    `max_matmul` recorded through `recording_context()`.
                           This is the arm that decides whether our real
                           training steps port, because they are mostly library
                           GEMMs, not hand kernels.
  D. a SPLIT-K GEMM        the known blocker. Predicted to FAIL, and to fail
                           SILENTLY — see below.
  E. timing                eager enqueue vs `graph.replay()`, us/iter.

## What arm D is really testing, and why it is last

Stream capture cannot contain a split-K GEMM at all: MAX's allocator refuses to
serve `enqueue_create_buffer` while a graph is capturing and ABORTS the process
(`benchmarks/bench_cuda_graph_splitk_capture.mojo`;
[[_split_k_cannot_be_cuda_graph_captured]]). `DeviceGraph` has a documented
capture-scoped pool for exactly this — `builder.create_buffer`, "all allocations
created for a device graph should go through this method". But `multistage_gemm`
calls `ctx.enqueue_create_buffer`, and `recording_context()`'s docstring says
buffer allocation FORWARDS TO THE BACKING CONTEXT.

So the predicted failure is worse than an abort: the workspace is allocated
EAGERLY at record time, `multistage_gemm` frees it at `_ = work_space_data^`
when the call returns, and the recorded node keeps the raw pointer. That is a
dangling device pointer on replay — **a wrong answer, not a crash**, and quite
possibly a RIGHT answer on the first replay while the freed block is still
untouched in the pool.

Hence arm D replays TWICE with an allocator CHURN in between, and checks an
analytic checksum both times. A pass on replay 1 and a fail on replay 2 is the
signature of the dangling workspace and is the most likely outcome.

⚠ It may also fault or abort. It runs LAST so that A–C have already printed,
and `MOJO_RL_SPIKE_SPLITK=0` skips it.

⚠ AND IT MUST PROVE IT IS SPLIT-K ON *THIS* CARD. `[256 x 2592] @ [2592 x 256]`
partitions K on a 5090; `select_config` hardcodes `A100.sm_count = 108` in its
wave estimate, so another part can decide otherwise and the arm would be
measuring nothing. It prints `num_k_partitions` and says so when it is 1.
([[_a_capture_abort_names_only_the_first_blocker]]: a size is not an identity.)

## Reading the result

    0,A,B pass                 DeviceGraph is live here. The shim is replaceable
                               for hand-written kernel sequences at minimum.
    C passes                   library calls record. Port `maybe_capture_replay`
                               onto `DeviceGraph` — the closure signature has to
                               grow a `DeviceContext` argument (today it captures
                               one), which is the whole migration.
    C fails                    the interesting failure. Report WHICH call raised;
                               a `synchronize()` inside a library path would do
                               it, and `recording_context()` raises on host waits
                               by design.
    D fails (as predicted)     unchanged conclusion: own the split-K workspace.
                               DeviceGraph does not rescue split-K by itself, it
                               just fails differently (silently, not loudly).
    D passes                   then MAX routes that allocation somewhere better
                               than its docstring implies — re-check
                               `_split_k_cannot_be_cuda_graph_captured`.

⚠ Off NVIDIA this file compiles and runs but there is NO VERDICT: Metal supports
neither stream capture nor device graphs ([[_the_metal_launch_floor_is_command_
buffer_retirement]]). Every arm is expected to RAISE on Apple, and the arm
harness reports that as "unsupported here", not as a failure. Run it on the
5090 for an answer.
"""

from std.math import ceildiv
from std.os import getenv
from std.sys import has_nvidia_gpu_accelerator
from std.time import perf_counter_ns

from layout import Layout, LayoutTensor, TileTensor, row_major
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)
from linalg.matmul import matmul as max_matmul
from linalg.utils_gpu import select_config

comptime DT = DType.float32

comptime N = 1024
comptime BLOCK = 256
comptime GRID = ceildiv(N, BLOCK)

# Same shapes as `bench_cuda_graph_splitk_capture.mojo`, so the two files are
# directly comparable. Both PASS the dispatch gate (n % 128 == 0, k % 32 == 0,
# k >= 128) — neither can reach the vendor fallback, so the ONLY difference
# between them is whether `select_config` partitions K.
comptime OK_M = 2592  # plain multistage
comptime OK_K = 256
comptime OK_N = 256

comptime SK_M = 256  # multistage_gemm_split_k on a 5090
comptime SK_K = 2592
comptime SK_N = 256

comptime AVAL = 0.01
comptime BVAL = 0.02

comptime CHAIN = 32  # launches per iteration in the timing arm
comptime TIMED_ITERS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Kernels.
#
# ⚠ TWO STYLES ON PURPOSE. `_fill_ptr` takes raw `UnsafePointer`s because that
# is what MAX's own device-graph test uses and arm A must not be the thing that
# fails. `_fill_lt` / `_bump_lt` take `LayoutTensor`, which is what EVERY kernel
# in this repo takes — arm B exists to prove that our style records too, and not
# only MAX's.
# ─────────────────────────────────────────────────────────────────────────────

from std.gpu import global_idx


def _fill_ptr(
    buf: Pointer[Float32, MutAnyOrigin], val: Int32, length: Int32
):
    var n = Int(length)
    var tid = global_idx.x
    if tid >= n:
        return
    buf[unsafe_offset=tid] = Float32(Int(val))


def _fill_lt(
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin], val: Int32
):
    var tid = Int(global_idx.x)
    if tid >= N:
        return
    buf[tid] = Scalar[DT](Int(val))


def _bump_lt(
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin], delta: Int32
):
    var tid = Int(global_idx.x)
    if tid >= N:
        return
    buf[tid] = buf[tid] + Scalar[DT](Int(delta))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_first(ctx: DeviceContext, buf: DeviceBuffer[DT]) raises -> Float64:
    """First element, host-side. The arms assert on an ANALYTIC value.

    ⚠ Reading one element is enough ONLY because every kernel here writes the
    whole buffer uniformly. Do not copy this into a test of a real kernel.
    """
    var host = ctx.enqueue_create_host_buffer[DT](1)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    return Float64(host[0])


def _sum_first(
    ctx: DeviceContext, buf: DeviceBuffer[DT], total: Int, n: Int
) raises -> Float64:
    var host = ctx.enqueue_create_host_buffer[DT](total)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var s = Float64(0.0)
    for i in range(n):
        s += Float64(host[i])
    return s


def _close(got: Float64, want: Float64) -> Bool:
    """Relative, and loose enough for TF32.

    A 5090 runs fp32 matmul through TF32 tensor cores by default (~10 mantissa
    bits); `bench_cuda_graph_splitk_capture.mojo` saw 0.40914977 against 0.4096
    and a 1e-3 gate called that "the step did not run". The job of this check is
    to catch a step that computed NOTHING or read freed memory — both off by
    ~100% — not to police the last bits.
    """
    if want == 0.0:
        return abs(got) <= 1e-9
    return abs(got - want) <= 5e-3 * abs(want)


def _verdict(label: String, ok: Bool, detail: String):
    print("    " + ("PASS  " if ok else "FAIL  ") + label + "   " + detail)


# ─────────────────────────────────────────────────────────────────────────────
# Arm A — the explicit-DAG path
# ─────────────────────────────────────────────────────────────────────────────


def arm_a(ctx: DeviceContext) raises -> Bool:
    """`add_function`: build must NOT execute; each replay must execute once.

    ⚠ "BUILD MUST NOT EXECUTE" IS HALF THE ASSERTION, and it is the half a
    careless harness drops. `mojo_rl/cuda/graph.mojo`'s minimal test learned
    this: a graph layer that silently ran the work eagerly and replayed NOTHING
    passes any check that only looks at the final value.
    """
    print("  A. add_function — explicit node, no stream capture")

    var buf = ctx.enqueue_create_buffer[DT](N)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()

    var fill = ctx.compile_function[_fill_ptr]()

    # ⚠ `{imm}` is the capture list, and this spelling is copied verbatim from
    # MAX's own test. `DeviceGraph.create` takes
    # `def[o: ImmOrigin](mut DeviceGraphBuilder[o]) raises` — the builder's
    # origin is scoped to the call, so a node handle cannot escape it.
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            fill,
            buf,
            Int32(7),
            Int32(N),
            grid_dim=GRID,
            block_dim=BLOCK,
        )

    var graph = DeviceGraph.create(ctx, build)
    ctx.synchronize()
    var after_build = _read_first(ctx, buf)

    graph.replay()
    ctx.synchronize()
    var after_1 = _read_first(ctx, buf)

    # Zero it so the second replay has to recompute rather than inherit.
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    graph.replay()
    ctx.synchronize()
    var after_2 = _read_first(ctx, buf)

    var ok = _close(after_build, 0.0) and _close(after_1, 7.0) and _close(
        after_2, 7.0
    )
    _verdict(
        "one-node graph",
        ok,
        "build->"
        + String(after_build)
        + " (want 0)  replay1->"
        + String(after_1)
        + "  replay2(after wipe)->"
        + String(after_2)
        + " (want 7)",
    )
    if not ok and not _close(after_build, 0.0):
        print("      ^ the BUILD executed. That is eager enqueue wearing a")
        print("        graph's name, and every timing number below is void.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Arm B — recording_context with OUR kernel style
# ─────────────────────────────────────────────────────────────────────────────


def arm_b(ctx: DeviceContext) raises -> Bool:
    """Four LayoutTensor launches through the ordinary `enqueue_function` API.

    THE ARM THAT DECIDES THE MIGRATION SHAPE. If unmodified `DeviceContext`
    code records, then `maybe_capture_replay`'s `STEP()` closure records too —
    it only has to take the context as an ARGUMENT instead of capturing one.

    ⚠ THE ARITHMETIC IS THE ASSERTION, not just the total. fill(5) then +1, +2,
    +4 lands on 12 only if all four recorded AND stayed in order. A graph that
    dropped the fill, or reordered the bumps against it, lands elsewhere — and
    on CUDA that ordering is a recorded dependency EDGE, not an implicit stream
    FIFO, so it is a real thing to check.
    """
    print("  B. recording_context — our LayoutTensor kernels, unmodified")

    var buf = ctx.enqueue_create_buffer[DT](N)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    var lt = LayoutTensor[DT, Layout.row_major(N)](buf)

    var fill = ctx.compile_function[_fill_lt]()
    var bump = ctx.compile_function[_bump_lt]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            rec.enqueue_function(
                fill, lt, Int32(5), grid_dim=GRID, block_dim=BLOCK
            )
            rec.enqueue_function(
                bump, lt, Int32(1), grid_dim=GRID, block_dim=BLOCK
            )
            rec.enqueue_function(
                bump, lt, Int32(2), grid_dim=GRID, block_dim=BLOCK
            )
            rec.enqueue_function(
                bump, lt, Int32(4), grid_dim=GRID, block_dim=BLOCK
            )

    var graph = DeviceGraph.create(ctx, build)
    ctx.synchronize()
    var after_build = _read_first(ctx, buf)

    graph.replay()
    ctx.synchronize()
    var after_1 = _read_first(ctx, buf)

    buf.enqueue_fill(Scalar[DT](99))
    ctx.synchronize()
    graph.replay()
    ctx.synchronize()
    var after_2 = _read_first(ctx, buf)

    var ok = _close(after_build, 0.0) and _close(after_1, 12.0) and _close(
        after_2, 12.0
    )
    _verdict(
        "4-launch chain",
        ok,
        "build->"
        + String(after_build)
        + " (want 0)  replay1->"
        + String(after_1)
        + "  replay2(from 99)->"
        + String(after_2)
        + " (want 12)",
    )
    if not ok and _close(after_2, 111.0):
        print("      ^ 99+1+2+4: the FILL did not record, the bumps did.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Arms C / D — a MAX library GEMM through recording_context
# ─────────────────────────────────────────────────────────────────────────────


struct _Gemm[M_: Int, K_: Int, N_: Int](Movable):
    """Owns the context AND the operands.

    ⚠⚠ LOAD-BEARING, and two rounds of crashes in
    `bench_cuda_graph_splitk_capture.mojo` came from not doing it. Mojo
    destroys a value at its LAST USE, and MAX's `DeviceContext` destructor
    SYNCHRONIZES AND DESTROYS the stream. Separately, A GRAPH HOLDS RAW
    POINTERS: every operand must outlive the LAST REPLAY, not the build. Both
    rules point the same way — one struct owns everything, the closure mentions
    only that struct.
    """

    var ctx: DeviceContext
    var a: DeviceBuffer[DT]
    var b: DeviceBuffer[DT]
    var c: DeviceBuffer[DT]

    def __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.a = ctx.enqueue_create_buffer[DT](Self.M_ * Self.K_)
        self.b = ctx.enqueue_create_buffer[DT](Self.K_ * Self.N_)
        self.c = ctx.enqueue_create_buffer[DT](Self.M_ * Self.N_)
        self.a.enqueue_fill(Scalar[DT](AVAL))
        self.b.enqueue_fill(Scalar[DT](BVAL))
        self.c.enqueue_fill(Scalar[DT](0))
        ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx^
        self.a = move.a^
        self.b = move.b^
        self.c = move.c^

    def gemm(mut self, ctx: DeviceContext) raises:
        """The recorded work. Takes the context as an ARGUMENT — that is the
        one change a `STEP()` closure needs to become recordable."""
        var av = TileTensor(self.a, row_major[Self.M_, Self.K_]())
        var bv = TileTensor(self.b, row_major[Self.K_, Self.N_]())
        var cv = TileTensor(self.c, row_major[Self.M_, Self.N_]())
        max_matmul[target="gpu"](cv, av, bv, ctx)

    def wipe(self) raises:
        self.c.enqueue_fill(Scalar[DT](0))
        self.ctx.synchronize()

    def checksum(self, n: Int) raises -> Float64:
        return _sum_first(self.ctx, self.c, Self.M_ * Self.N_, n)

    @staticmethod
    def want(n: Int) -> Float64:
        return Float64(Self.K_) * AVAL * BVAL * Float64(n)


def _churn(ctx: DeviceContext) raises:
    """Make the allocator hand out — and dirty — whatever a freed split-K
    workspace left behind.

    ⚠ THIS IS THE WHOLE POINT OF ARM D. A dangling pointer into a pool block
    nobody has touched yet still reads the right bytes; the first replay can
    pass on a graph that is already wrong. `P * M * N * 4` for the arm's shape
    is 512KB at P=2 — the exact `request=512KB` in the capture abort — so this
    allocates several multiples of that and writes a poison value through them.
    """
    comptime POISON_ELEMS = 512 * 1024 // 4
    var junk = List[DeviceBuffer[DT]]()
    for _ in range(8):
        var b = ctx.enqueue_create_buffer[DT](POISON_ELEMS)
        b.enqueue_fill(Scalar[DT](-12345.0))
        junk.append(b^)
    ctx.synchronize()
    _ = junk^


def _gemm_arm[
    M_: Int, K_: Int, N_: Int
](ctx: DeviceContext, label: String, churn: Bool) raises -> Bool:
    var g = _Gemm[M_, K_, N_](ctx)

    # Warm the library OUTSIDE the graph: first-call autotuning, module load
    # and any one-off allocation happen here, not at record time. Then WIPE, so
    # the warm-up's correct output cannot stand in for a graph that recorded
    # nothing.
    g.gemm(ctx)
    ctx.synchronize()
    g.wipe()

    def build(mut builder: DeviceGraphBuilder) raises {mut g}:
        with builder.recording_context() as rec:
            g.gemm(rec)

    var graph = DeviceGraph.create(ctx, build)
    ctx.synchronize()

    var after_build = g.checksum(8)
    var want = _Gemm[M_, K_, N_].want(8)

    graph.replay()
    ctx.synchronize()
    var r1 = g.checksum(8)

    var r2 = Float64(0.0)
    if churn:
        g.wipe()
        _churn(ctx)
        graph.replay()
        ctx.synchronize()
        r2 = g.checksum(8)

    var ok = _close(after_build, 0.0) and _close(r1, want)
    if churn:
        ok = ok and _close(r2, want)

    var detail = (
        "build->"
        + String(after_build)
        + " (want 0)  replay1->"
        + String(r1)
        + "  want "
        + String(want)
    )
    if churn:
        detail += "  replay2(after churn)->" + String(r2)
    _verdict(label, ok, detail)

    if churn and _close(r1, want) and not _close(r2, want):
        print("      ^ RIGHT then WRONG across an allocator churn. That is the")
        print("        dangling split-K workspace, exactly as predicted: the")
        print("        recorded node holds a pointer MAX already freed.")
    return ok


def arm_c(ctx: DeviceContext) raises -> Bool:
    print("  C. max_matmul through recording_context — plain multistage")
    return _gemm_arm[OK_M, OK_K, OK_N](
        ctx, "library GEMM, no split-K", churn=False
    )


def arm_d(ctx: DeviceContext) raises -> Bool:
    print("  D. max_matmul through recording_context — SPLIT-K")
    var picked = select_config[DT, DT, DT, False](SK_M, SK_N, SK_K, ctx)
    var parts = Int(picked.num_k_partitions)
    print(
        "     select_config: num_k_partitions = "
        + String(parts)
        + "   block_tile "
        + String(picked.block_tile_shape)
    )
    if parts <= 1:
        # ⚠ A SHAPE IS NOT AN IDENTITY. `select_config` hardcodes A100's
        # sm_count in its wave estimate, so this shape partitioning K is a
        # property of the 5090 it was chosen on, not of the numbers.
        print("     ⚠ K IS NOT PARTITIONED ON THIS DEVICE — arm D would be a")
        print("       second copy of arm C and proves nothing about split-K.")
        print("       Find a shape where this prints > 1 before believing a")
        print("       PASS here. SKIPPED.")
        return True
    return _gemm_arm[SK_M, SK_K, SK_N](
        ctx, "library GEMM, split-K", churn=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Arm E — what it buys
# ─────────────────────────────────────────────────────────────────────────────


def arm_e(ctx: DeviceContext) raises:
    """Eager enqueue of `CHAIN` launches vs one `graph.replay()`, both synced.

    ⚠ THIS IS A LAUNCH-OVERHEAD MEASUREMENT AND NOTHING ELSE. The kernels are
    deliberately trivial, so the ratio here is an upper bound that a real step
    will not see — [[_a_per_call_sweep_is_an_upper_bound_on_a_step]]. Quote it
    as a ceiling.
    """
    print("  E. timing — " + String(CHAIN) + " launches, eager vs replay")

    var buf = ctx.enqueue_create_buffer[DT](N)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    var lt = LayoutTensor[DT, Layout.row_major(N)](buf)
    var bump = ctx.compile_function[_bump_lt]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            for _ in range(CHAIN):
                rec.enqueue_function(
                    bump, lt, Int32(1), grid_dim=GRID, block_dim=BLOCK
                )

    var graph = DeviceGraph.create(ctx, build)
    ctx.synchronize()

    # Warm both paths before either is timed.
    for _ in range(10):
        for _ in range(CHAIN):
            ctx.enqueue_function(
                bump, lt, Int32(1), grid_dim=GRID, block_dim=BLOCK
            )
        ctx.synchronize()
        graph.replay()
        ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(TIMED_ITERS):
        for _ in range(CHAIN):
            ctx.enqueue_function(
                bump, lt, Int32(1), grid_dim=GRID, block_dim=BLOCK
            )
        ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(TIMED_ITERS):
        graph.replay()
        ctx.synchronize()
    var t2 = perf_counter_ns()

    var eager_us = Float64(t1 - t0) / 1000.0 / Float64(TIMED_ITERS)
    var replay_us = Float64(t2 - t1) / 1000.0 / Float64(TIMED_ITERS)
    print(
        "     eager  "
        + String(eager_us)
        + " us/iter      replay "
        + String(replay_us)
        + " us/iter      "
        + String(eager_us / replay_us)
        + "x"
    )
    print("     (trivial kernels — a CEILING on launch overhead, not a step)")


# ─────────────────────────────────────────────────────────────────────────────


def _run[
    arm: def (DeviceContext) raises thin -> Bool
](name: String, ctx: DeviceContext) -> Int:
    """Run one arm. Returns 1 pass, 0 fail, -1 raised.

    ⚠ A RAISE IS ITS OWN OUTCOME, not a failure. Off NVIDIA every arm is
    EXPECTED to raise (Metal has neither stream capture nor device graphs), and
    on NVIDIA the exception TEXT is the finding — `recording_context()` raises
    by design on host-visible waits, so a library path that synchronizes shows
    up here and nowhere else.
    """
    try:
        return 1 if arm(ctx) else 0
    except e:
        print("    RAISED  " + name + ": " + String(e))
        return -1


def main() raises:
    var ctx = DeviceContext()
    print("=" * 76)
    print("MAX DeviceGraph spike —", ctx.name())
    print("=" * 76)
    print(
        "  0. import + link:  max.gpu.host.DeviceGraph resolved, context live."
    )
    print("     NO libcuda_intercept.so, NO LD_PRELOAD, NO stream borrowing.")
    comptime if not has_nvidia_gpu_accelerator():
        print("")
        print("  ⚠ NOT AN NVIDIA DEVICE. Metal supports neither stream capture")
        print("    nor device graphs, so every arm below is expected to RAISE")
        print("    and NO VERDICT IS POSSIBLE. This run only proves the file")
        print("    builds and links. Run it on the 5090 for the answer.")
    print("")

    var a = _run[arm_a]("A", ctx)
    var b = _run[arm_b]("B", ctx)
    var c = _run[arm_c]("C", ctx)

    var d = 2  # 2 = not attempted
    if getenv("MOJO_RL_SPIKE_SPLITK", "1") == "0":
        print("  D. SKIPPED by MOJO_RL_SPIKE_SPLITK=0")
    else:
        print("  (arm D may fault or abort the process — everything above has")
        print("   already printed. MOJO_RL_SPIKE_SPLITK=0 skips it.)")
        d = _run[arm_d]("D", ctx)

    print("")
    try:
        arm_e(ctx)
    except e:
        print("  E. RAISED: " + String(e))

    print("")
    print("=" * 76)
    comptime if not has_nvidia_gpu_accelerator():
        print("  NO VERDICT — not an NVIDIA device.")
        return
    if a != 1:
        print("  A FAILED -> DeviceGraph is not usable here at all; B/C/D say")
        print("  nothing. Check the MAX version exports it:")
        print("    grep -n DeviceGraph .pixi/envs/nvidia/lib/mojo/max.mojoc")
    elif b != 1:
        print("  A passes, B fails -> the explicit DAG works but")
        print("  recording_context does not, so a STEP() closure cannot be")
        print("  recorded as-is and every node would have to be hand-added.")
        print("  That is a much larger migration than the shim is worth.")
    elif c != 1:
        print("  A,B pass, C fails -> hand kernels record, MAX library calls")
        print("  do not. Our steps are mostly library GEMMs, so read the RAISED")
        print("  text above: that names the call DeviceGraph will not record.")
    else:
        print("  A,B,C PASS -> DeviceGraph records our kernels AND MAX's")
        print("  library calls, with no interceptor. `mojo_rl/cuda/graph.mojo`")
        print("  is replaceable; the migration is giving `maybe_capture_replay`")
        print("  a STEP(ctx) signature instead of a captured context.")
        if d == 0:
            print("")
            print("  D FAILED as predicted -> split-K is STILL not capturable.")
            print("  DeviceGraph does not rescue it, it just fails silently")
            print("  instead of aborting. Own the workspace (that work is")
            print("  already landed for Linear/Conv2D dW) before recording a")
            print("  step that contains one.")
        elif d == 1:
            print("")
            print("  D PASSED -> then the split-K workspace survives recording,")
            print("  which contradicts _split_k_cannot_be_cuda_graph_captured.")
            print("  Re-run it; a pass here needs the churn arm to be real.")
