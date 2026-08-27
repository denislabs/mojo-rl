"""LeWM PushT sampler-only micro-benchmark.

No model, no GPU. Just constructs the HDF5-backed sampler and hammers
`sample_batch_uint8` N times to measure libhdf5/disk cost in isolation.

Reports mean / median / p10 / p90 / min / max per call (ms), plus an
effective MB/s based on the uint8 payload moved (B * T * H * W * 3 bytes).

Run (Apple, no GPU env needed since we don't import any kernel):

    pixi run mojo run -I . benchmarks/bench_lewm_pusht_sampler.mojo
"""

from std.time import perf_counter_ns
from std.memory import alloc

from mojo_rl.envs.pusht.offline_sampler import PushTOfflineSampler


def insertion_sort_u64(
    buf: Pointer[Scalar[DType.uint64], MutAnyOrigin], n: Int
):
    for i in range(1, n):
        var x = buf[i]
        var j = i - 1
        while j >= 0 and buf[j] > x:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = x


def main() raises:
    comptime BATCH = 16
    comptime T = 4
    comptime H = 224
    comptime W = 224
    comptime FRAMESKIP = 5
    comptime ACTION_DIM = 2

    comptime PIX_PER_CALL = BATCH * T * H * W * 3
    comptime ACT_PER_CALL = BATCH * T * FRAMESKIP * ACTION_DIM

    var n_warmup = 3
    var n_iters = 30

    print("LeWM PushT sampler benchmark")
    print(
        "  BATCH=",
        BATCH,
        " T=",
        T,
        " H=",
        H,
        " W=",
        W,
        " frameskip=",
        FRAMESKIP,
    )
    print("  warmup=", n_warmup, " iters=", n_iters)
    print(
        "  payload/call: ",
        PIX_PER_CALL,
        " uint8 pixels (",
        Float64(PIX_PER_CALL) / 1e6,
        " MB) +",
        ACT_PER_CALL,
        " fp32 actions",
    )

    var sampler = PushTOfflineSampler(
        frameskip=FRAMESKIP, num_steps=T, path=String("")
    )

    var pixels = alloc[Scalar[DType.uint8]](PIX_PER_CALL)
    var actions = alloc[Scalar[DType.float32]](ACT_PER_CALL)

    # Warmup (also primes any libhdf5 caches).
    for _ in range(n_warmup):
        sampler.sample_batch_uint8(BATCH, T, pixels, actions)

    # Timed loop.
    var times_ns = alloc[Scalar[DType.uint64]](n_iters)
    var sum_ns = UInt(0)
    for i in range(n_iters):
        var t0 = perf_counter_ns()
        sampler.sample_batch_uint8(BATCH, T, pixels, actions)
        var t1 = perf_counter_ns()
        var dt = t1 - t0
        sum_ns += dt
        times_ns[i] = Scalar[DType.uint64](dt)

    insertion_sort_u64(times_ns, n_iters)

    var mean_ms = Float64(sum_ns) / Float64(n_iters) / 1e6
    var med_ms = Float64(times_ns[n_iters // 2]) / 1e6
    var p10_ms = Float64(times_ns[n_iters // 10]) / 1e6
    var p90_ms = Float64(times_ns[(n_iters * 9) // 10]) / 1e6
    var min_ms = Float64(times_ns[0]) / 1e6
    var max_ms = Float64(times_ns[n_iters - 1]) / 1e6

    var bytes_per_call = Float64(PIX_PER_CALL)
    var mbps_mean = (bytes_per_call / 1e6) / (mean_ms / 1e3)

    print("")
    print("=== sample_batch_uint8 timing (ms per call) ===")
    print("  mean   = ", mean_ms)
    print("  median = ", med_ms)
    print("  p10    = ", p10_ms)
    print("  p90    = ", p90_ms)
    print("  min    = ", min_ms)
    print("  max    = ", max_ms)
    print("")
    print("  effective uint8 throughput (mean): ", mbps_mean, " MB/s")
