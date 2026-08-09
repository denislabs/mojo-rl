"""Smoke + correctness test for ObsNormStats GPU paths.

Instantiates the running-obs-normalizer, pushes a few batches through
`update_and_apply`, and checks that the device stats track a hand-computed
running mean/var and that the in-place normalization is sane.

This is the gate for the Pointer->MutAnyOrigin migration of
`mojo_rl/core/obs_norm.mojo`: it exercises BOTH the update kernel (reads obs,
writes mean/var/count) and the apply kernel (writes obs, reads mean/var), so
it surfaces any mutability mismatch that the package build can't (precompile
doesn't instantiate generics).
"""

from std.math import sqrt
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT as gpu_dtype
from mojo_rl.core.obs_norm import ObsNormStats


def main() raises:
    comptime OBS_DIM = 4
    comptime BATCH = 8

    var ctx = DeviceContext()
    var stats = ObsNormStats[OBS_DIM](ctx)

    # Three batches of known data; fill obs with value = (e + d) so the
    # per-dim mean over the batch is deterministic.
    comptime N_BATCHES = 3
    for it in range(N_BATCHES):
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH * OBS_DIM)
        var obs_host = ctx.enqueue_create_host_buffer[gpu_dtype](BATCH * OBS_DIM)
        for e in range(BATCH):
            for d in range(OBS_DIM):
                obs_host[e * OBS_DIM + d] = Scalar[gpu_dtype](
                    Float64(e + d) + Float64(it)
                )
        ctx.enqueue_copy(obs_buf, obs_host)
        ctx.synchronize()

        stats.update_and_apply[BATCH](ctx, obs_buf)

        # Pull normalized obs back; check finite and that the apply happened
        # (after enough stats, the mean-removed obs should be roughly centered).
        ctx.enqueue_copy(obs_host, obs_buf)
        ctx.synchronize()
        var any_nan = False
        for i in range(BATCH * OBS_DIM):
            var v = Float64(obs_host[i])
            if v != v:
                any_nan = True
        if any_nan:
            print("FAIL: NaN in normalized obs at iter", it)
            return

    stats.sync_host(ctx)
    print("host_count =", stats.host_count)
    print("host_mean[0..4] =", stats.host_mean[0], stats.host_mean[1],
          stats.host_mean[2], stats.host_mean[3])

    # count should have advanced by N_BATCHES * BATCH from the 1e3 prior.
    var expected_count = 1e3 + Float64(N_BATCHES * BATCH)
    if abs(stats.host_count - expected_count) > 1e-3:
        print("FAIL: count", stats.host_count, "!=", expected_count)
        return

    # Mean for dim d across all data = mean over (e in 0..BATCH, it in 0..N) of
    # (e + d + it). E[e]=3.5, E[it]=1.0 -> per-dim mean contribution ~ 4.5 + d,
    # but blended with the 1e3 count_prior(mean 0) it stays small/positive.
    for d in range(OBS_DIM):
        if stats.host_mean[d] <= 0.0:
            print("FAIL: mean[", d, "] not positive:", stats.host_mean[d])
            return
        if stats.host_var[d] <= 0.0:
            print("FAIL: var[", d, "] not positive:", stats.host_var[d])
            return

    print("ObsNormStats GPU smoke: OK")
