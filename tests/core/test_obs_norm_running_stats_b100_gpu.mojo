"""ObsNormStats running-stat test at BATCH=100.

Guards the same NVIDIA store-drop class that hit BatchNorm: update_obs_norm_kernel
is a per-thread SERIAL reduction over the batch (two `for e in range(BATCH)`
loops) followed by read-modify-write store-backs — mean[d]/var_[d] (unconditional)
and count[0] (conditional `if d == 0`). With count_prior=0 and a FIXED batch fed
N times, working stats give count = N*BATCH, mean = batch_mean, var = batch_var.
A dropped count[0] store leaves count stuck at 0; a dropped mean/var store leaves
them at init.

Run (Apple):  pixi run -e apple  mojo run -I . tests/core/test_obs_norm_running_stats_b100_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . tests/core/test_obs_norm_running_stats_b100_gpu.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT as gpu_dtype
from mojo_rl.core.obs_norm import ObsNormStats


def main() raises:
    comptime OBS_DIM = 6
    comptime BATCH = 100
    comptime N_UPDATES = 3
    var ctx = DeviceContext()

    # Fixed batch with a distinct per-dim mean/scale.
    var host = ctx.enqueue_create_host_buffer[gpu_dtype](BATCH * OBS_DIM)
    ctx.synchronize()
    for e in range(BATCH):
        for d in range(OBS_DIM):
            var noise = Float32(((e * 7 + d * 13) % 11) - 5) * 0.1 * Float32(d + 1)
            host[e * OBS_DIM + d] = Scalar[gpu_dtype](noise + Float32(d + 1))
    var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH * OBS_DIM)
    ctx.enqueue_copy(obs_buf, host)
    ctx.synchronize()

    # Host batch mean/var per dim (the convergence target).
    var hmean = List[Float64](length=OBS_DIM, fill=0.0)
    var hvar = List[Float64](length=OBS_DIM, fill=0.0)
    for d in range(OBS_DIM):
        var m: Float64 = 0.0
        for e in range(BATCH):
            m += Float64(host[e * OBS_DIM + d])
        m /= Float64(BATCH)
        var v: Float64 = 0.0
        for e in range(BATCH):
            var diff = Float64(host[e * OBS_DIM + d]) - m
            v += diff * diff
        v /= Float64(BATCH)
        hmean[d] = m
        hvar[d] = v

    # count_prior=0 → after N updates of the SAME batch: count=N*BATCH,
    # mean=batch_mean, var=batch_var (exactly, modulo fp32).
    var stats = ObsNormStats[OBS_DIM](ctx, count_prior=0.0)
    for _ in range(N_UPDATES):
        stats._update[BATCH](ctx, obs_buf)
    ctx.synchronize()
    stats.sync_host(ctx)

    var ok = True
    for d in range(OBS_DIM):
        var dm = abs(stats.host_mean[d] - hmean[d])
        var dv = abs(stats.host_var[d] - hvar[d])
        print(
            "d", d,
            "| mean", stats.host_mean[d], "(host", hmean[d], ") d=", dm,
            "| var", stats.host_var[d], "(host", hvar[d], ") d=", dv,
        )
        if dm > 1e-2 or dv > 1e-2:
            ok = False

    var expected_count = Float64(N_UPDATES * BATCH)
    print("count =", stats.host_count, "(expected", expected_count, ")")
    var count_ok = abs(stats.host_count - expected_count) < 0.5

    assert_true(
        ok,
        "ObsNorm running mean/var did NOT match batch stats (mean/var store"
        " dropped in the per-thread reduction kernel).",
    )
    assert_true(
        count_ok,
        "ObsNorm count did NOT accumulate to N*BATCH (the conditional count[0]"
        " store-back was dropped — the BatchNorm store-drop class).",
    )
    print("PASS")
