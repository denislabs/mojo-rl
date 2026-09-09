"""`ObsEma` against torch's `BatchNorm1d(affine=False, momentum=0.01)` — the
running observation normaliser's gate (G3.2).

    pixi run -e act-ref mojo run -I . tests/fb/test_obs_ema_vs_batchnorm.mojo

Feeds the same K batches to both in train mode (ours through
`ema_update_kernel` on the GPU, torch's through its forward), compares
the running mean and variance after every batch, then normalises a fresh
batch in eval mode on both sides and compares the rows. Float32 on both
sides (torch is run in float32 too, so the comparison is exact
arithmetic, not an accumulation band): `TOL` 1e-5.

What a failure would mean: a biased instead of unbiased batch variance
(off by n/(n−1), 1.6 % at 64 rows), the wrong momentum sign or
placement, or the epsilon inside instead of outside the square root — the
mistakes a from-memory BatchNorm makes. The sidecar round trip is checked
too: `save` then `load` restores the statistics to float32 precision.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.fb.obs_ema import ObsEma, OBS_EMA_EPS
from mojo_rl.deep_agents.fb.kernels import ensure_t


comptime OBS = 9
comptime ROWS = 64
comptime K = 7
comptime TOL = 1e-5


def _fill(mut t: Tensor, n: Int, seed: Int, scale: Float64, shift: Float64):
    for i in range(n):
        var k = (i * 7919 + seed * 104729) % 1000
        t.data[i] = Scalar[DT]((Float64(k) / 999.0 * 2.0 - 1.0) * scale + shift)


def _py_rows(builtins: PythonObject, t: Tensor, n: Int) raises -> PythonObject:
    var out = builtins.list()
    for i in range(n):
        _ = out.append(Float64(t.data[i]))
    return out


def test_obs_ema_matches_batchnorm1d() raises:
    var builtins = Python.import_module("builtins")
    var torch = Python.import_module("torch")
    var bn = torch.nn.BatchNorm1d(OBS, affine=False, momentum=0.01)
    _ = bn.train()

    var ctx = DeviceContext()
    var ema = ObsEma[OBS].make(ctx)
    var batch = Tensor()
    ensure_t["gpu"](batch, ROWS * OBS, Optional(ctx))

    var worst_stats = 0.0
    for k in range(K):
        # a batch with a moving mean and scale, so the EMA has something to track
        _fill(batch, ROWS * OBS, 100 + k, 1.0 + 0.3 * Float64(k), 0.5 * Float64(k))
        batch.upload(ctx)
        var xb = torch.tensor(_py_rows(builtins, batch, ROWS * OBS), dtype=torch.float32).reshape(ROWS, OBS)
        _ = bn(xb)
        ema.update[ROWS](batch)
        ema.sync_host()
        var rm = bn.running_mean
        var rv = bn.running_var
        for d in range(OBS):
            var em = abs(Float64(ema.mean.data[d]) - Float64(py=rm[d]))
            var ev = abs(Float64(ema.var_.data[d]) - Float64(py=rv[d]))
            if em > worst_stats:
                worst_stats = em
            if ev > worst_stats:
                worst_stats = ev
    print("  running mean / var after", K, "batches: worst |d| vs BatchNorm1d", worst_stats)
    assert_true(worst_stats < TOL, "running statistics differ from BatchNorm1d by " + String(worst_stats))

    # eval-mode apply on a fresh batch, in place on ours
    _ = bn.eval()
    _fill(batch, ROWS * OBS, 777, 2.0, -0.7)
    batch.upload(ctx)
    var xe = torch.tensor(_py_rows(builtins, batch, ROWS * OBS), dtype=torch.float32).reshape(ROWS, OBS)
    var ye = bn(xe).reshape(-1)
    ema.apply[ROWS](batch)
    batch.download(ctx)
    ctx.synchronize()
    var worst_apply = 0.0
    for i in range(ROWS * OBS):
        var e = abs(Float64(batch.data[i]) - Float64(py=ye[i]))
        if e > worst_apply:
            worst_apply = e
    print("  eval-mode normalisation of a fresh batch: worst |d|", worst_apply)
    assert_true(worst_apply < TOL, "normalised rows differ from BatchNorm1d by " + String(worst_apply))

    # sidecar round trip
    var path = String("/tmp/obs_ema_gate.norm")
    ema.save(path)
    var ema2 = ObsEma[OBS].make(ctx)
    ema2.load(path)
    ema2.sync_host()
    var worst_rt = 0.0
    for d in range(OBS):
        var em = abs(Float64(ema2.mean.data[d]) - Float64(ema.mean.data[d]))
        var ev = abs(Float64(ema2.var_.data[d]) - Float64(ema.var_.data[d]))
        if em > worst_rt:
            worst_rt = em
        if ev > worst_rt:
            worst_rt = ev
    print("  sidecar save/load round trip: worst |d|", worst_rt)
    assert_true(worst_rt < 1e-6, "sidecar round trip lost precision: " + String(worst_rt))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
