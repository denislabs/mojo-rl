"""R.5 GPU kernel parity test for `redq_ensemble_target_gpu`.

Mirrors the CPU kernel unit test: at N=2 / N_MIN=2 / MODE=MIN, the
GPU kernel must produce y values bit-equal (within FP rounding tol)
to the CPU kernel for the same inputs. This gates the GPU port of
the combine + α·logp + γ + terminal-mask math.

Apple Metal: this is the "compiles + matches CPU" gate. Real
numeric convergence on NVIDIA is HW-gated (legacy GPU-only kernel
math is well-understood; the port is mechanical).
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.redq.kernels import (
    redq_ensemble_target_cpu,
    redq_ensemble_target_gpu,
    REDQ_TARGET_MIN,
)


comptime N = 2
comptime N_MIN = 2
comptime BATCH = 8


def test_redq_ensemble_target_gpu_matches_cpu() raises:
    print("--- redq_ensemble_target_gpu vs CPU at N=2 N_MIN=2 MIN ---")
    var ctx = DeviceContext()

    # ── Build inputs on host, run both CPU and GPU, compare.
    var q_buf = List[Scalar[DT]](length=N * BATCH, fill=Scalar[DT](0.0))
    # Deterministic values; Q1 and Q2 alternate which is smaller.
    var q1_vals = List[Float64](length=BATCH, fill=0.0)
    var q2_vals = List[Float64](length=BATCH, fill=0.0)
    for b in range(BATCH):
        q1_vals[b] = 0.3 * Float64(b) - 1.0
        q2_vals[b] = -0.2 * Float64(b) + 0.5
        q_buf[0 * BATCH + b] = Scalar[DT](q1_vals[b])
        q_buf[1 * BATCH + b] = Scalar[DT](q2_vals[b])

    var subset_host = List[UInt32](length=N_MIN, fill=UInt32(0))
    subset_host[0] = UInt32(0)
    subset_host[1] = UInt32(1)

    var subset_cpu_int = List[Int](length=N_MIN, fill=0)
    subset_cpu_int[0] = 0
    subset_cpu_int[1] = 1

    var rewards = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    var terms   = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    var lps     = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    for b in range(BATCH):
        rewards[b] = Scalar[DT](-0.1 + 0.2 * Float64(b))
        terms[b]   = Scalar[DT](1.0) if b == 3 else Scalar[DT](0.0)
        lps[b]     = Scalar[DT](-0.5 + 0.15 * Float64(b))

    var gamma = Scalar[DT](0.97)
    var alpha = Scalar[DT](0.18)

    # CPU reference.
    var y_cpu = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        rewards.unsafe_ptr(),
        q_buf.unsafe_ptr(),
        terms.unsafe_ptr(),
        lps.unsafe_ptr(),
        subset_cpu_int.unsafe_ptr(),
        gamma, alpha,
        y_cpu.unsafe_ptr(),
    )

    # GPU run.
    var out_y_dev = ctx.enqueue_create_buffer[DT](BATCH)
    var rewards_dev = ctx.enqueue_create_buffer[DT](BATCH)
    var q_next_dev = ctx.enqueue_create_buffer[DT](N * BATCH)
    var terms_dev = ctx.enqueue_create_buffer[DT](BATCH)
    var lps_dev = ctx.enqueue_create_buffer[DT](BATCH)
    var subset_dev = ctx.enqueue_create_buffer[DType.uint32](N_MIN)

    ctx.enqueue_copy(rewards_dev, rewards.unsafe_ptr())
    ctx.enqueue_copy(q_next_dev, q_buf.unsafe_ptr())
    ctx.enqueue_copy(terms_dev, terms.unsafe_ptr())
    ctx.enqueue_copy(lps_dev, lps.unsafe_ptr())
    ctx.enqueue_copy(subset_dev, subset_host.unsafe_ptr())

    redq_ensemble_target_gpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        ctx,
        out_y_dev.unsafe_ptr(),
        rewards_dev.unsafe_ptr(),
        q_next_dev.unsafe_ptr(),
        terms_dev.unsafe_ptr(),
        lps_dev.unsafe_ptr(),
        subset_dev.unsafe_ptr(),
        gamma, alpha,
    )

    var y_gpu_host = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    ctx.enqueue_copy(y_gpu_host.unsafe_ptr(), out_y_dev)
    ctx.synchronize()

    var max_dev: Float64 = 0.0
    for b in range(BATCH):
        var d = Float64(y_gpu_host[b]) - Float64(y_cpu[b])
        if d < 0.0:
            d = -d
        if d > max_dev:
            max_dev = d
        print(
            "  b=", b,
            " cpu=", y_cpu[b],
            " gpu=", y_gpu_host[b],
        )
    print("  max |gpu - cpu| =", max_dev)
    assert_true(
        max_dev < 1e-5,
        "GPU kernel must match CPU within FP rounding tolerance",
    )
    print("PASS — redq_ensemble_target GPU == CPU at N=2 N_MIN=2 MIN.")


def main() raises:
    test_redq_ensemble_target_gpu_matches_cpu()
