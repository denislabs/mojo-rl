"""Does `linalg.matmul` allocate a device buffer PER CALL?

    pixi run -e nvidia nsys profile --stats=true mojo run -I . \
        benchmarks/bench_matmul_alloc_probe.mojo

## Why

An nsys capture of `examples/dm_control/tdmpc2_dm_walker_profile_gpu.mojo` on
an RTX 5090 showed:

    cuMemFree_v2    2.59 s   36.7%   21,464 calls   avg 121 us
    cuMemAlloc_v2   2.13 s   30.1%   21,464 calls   avg  99 us

4.72 s — 67% of all CUDA API time, and MORE than the 3.10 s of GPU kernel time
in the same run. The counts are exactly equal, so something creates and
destroys a buffer per use. It is not the repo's own allocation sites: every
`alloc_gpu` / `enqueue_create_buffer` under `tdmpc2/`, `planners/trajectory/`
and `nn/` sits in a constructor, and `Tensor.ensure_gpu` only ever grows
(`if not self.dev or self.n < n`), so per-call scratch does not reallocate.

That leaves the GEMM. This file calls `max_matmul` `REPS` times on
PREALLOCATED buffers and does nothing else in the loop — no Tensor, no Module,
no agent. So every `cuMemAlloc_v2` nsys reports beyond the handful at startup
belongs to `linalg.matmul`.

## Reading it

Compare `cuMemAlloc_v2`'s `Num Calls` against `REPS` (printed at the end):

    ~= REPS or a small multiple  ->  CONFIRMED: the GEMM allocates per call.
                                     Fixing it is a MAX-level concern (a
                                     workspace cache), and worth an upstream
                                     report with these numbers.
    ~= a handful (< 50)          ->  REFUTED: the GEMM is not the source, and
                                     the allocations come from somewhere else
                                     in the agent. Fall back to a backtrace
                                     capture of the full profile:

        nsys profile --trace=cuda,osrt --cudabacktrace=memory:10000 \
             --sample=cpu --backtrace=dwarf -o alloc --force-overwrite=true \
             mojo run -I . examples/dm_control/tdmpc2_dm_walker_profile_gpu.mojo

    then open `alloc.nsys-rep` in the Nsight GUI and read the backtrace on any
    `cuMemAlloc_v2` row. (`--cudabacktrace` needs `--sample`, and it is slow —
    shrink COLLECT_STEPS/TRAIN_STEPS in that script before using it.)

The shape is the TD-MPC2 walker planner's dominant one at N_ENVS=8:
`[BATCH_TOTAL x K_PAD] @ [K_PAD x MLP]`, K already padded to a multiple of 32
so this measures the FAST path (25.3 us/call in the capture above).

⚠ Two loops run: a `splitK`-prone tall-skinny shape and a square-ish one. If
only one of them allocates, the workspace is shape-dependent — worth knowing,
because then the fix might be a shape choice rather than an upstream ask.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32

# The walker planner's dominant GEMM: BATCH_TOTAL = N_ENVS(8) * (256 + 12).
comptime M = 2144
comptime K = 544          # (LATENT 512 + ACT 6) padded to a multiple of 32
comptime N = 512
comptime REPS = 2000

# A second shape: the two-hot head, whose narrow N triggers split-K on NVIDIA
# (14,500 `splitKreduce_kernel` instances appeared in the walker capture).
comptime M2 = 2144
comptime K2 = 512
comptime N2 = 101
comptime REPS2 = 2000


def run[M_: Int, K_: Int, N_: Int](
    ctx: DeviceContext, reps: Int, label: String
) raises:
    var a = ctx.enqueue_create_buffer[DT](M_ * K_)
    var b = ctx.enqueue_create_buffer[DT](K_ * N_)
    var c = ctx.enqueue_create_buffer[DT](M_ * N_)
    a.enqueue_fill(Scalar[DT](0.01))
    b.enqueue_fill(Scalar[DT](0.01))
    ctx.synchronize()

    var av = TileTensor(a, row_major[M_, K_]())
    var bv = TileTensor(b, row_major[K_, N_]())
    var cv = TileTensor(c, row_major[M_, N_]())

    for _ in range(10):                      # warm up / one-off allocations
        max_matmul[target="gpu"](cv, av, bv, ctx)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(reps):                    # <- ONLY max_matmul in here
        max_matmul[target="gpu"](cv, av, bv, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    print(
        "  ", label, ": [", M_, "x", K_, "]@[", K_, "x", N_, "]  x", reps,
        " calls   ", Float64(t1 - t0) / 1000.0 / Float64(reps), " us/call",
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("=" * 70)
    print("max_matmul per-call allocation probe —", ctx.name())
    print("=" * 70)
    run[M, K, N](ctx, REPS, "planner GEMM ")
    run[M2, K2, N2](ctx, REPS2, "two-hot head")
    print("=" * 70)
    print("  total max_matmul calls in the timed loops:", REPS + REPS2)
    print("  (plus 20 warm-up calls before them)")
    print()
    print("  Now read `cuMemAlloc_v2 / Num Calls` from the CUDA API Summary.")
    print("  ~= 4020 (or a multiple)  -> the GEMM allocates per call.")
    print("  < 50                     -> it does not; backtrace the agent run")
    print("                              (see this file's header).")
    print("=" * 70)
