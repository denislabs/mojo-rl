"""The device sampler gathers the same batch the host sampler does.

⚠ The two RNGs CANNOT agree. The device draws with Philox and the host with a
xorshift, so no seed makes `sample` reproduce `sample_batch` and a "same batch"
gate is not a thing that exists. What IS supposed to agree — and what the whole
point of `gather_at` is — is the part downstream of the draw: given the SAME
`(row, n_real)`, the device gather must produce the same tensors as
`ACTDataset.fill_at`, including the ImageNet normalization, the per-joint qpos
and action normalization, and the padding rule past an episode's end.

Gating the draw itself would need a device-side reimplementation of the host
RNG, which would be testing a coincidence rather than the thing that can break.
"""

from std.sys import exit
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.act.config import (
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from mojo_rl.deep_agents.act.data import ACTDataset
from mojo_rl.deep_agents.act.data_gpu import ACTDeviceDataset

from std.python import Python, PythonObject

comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W
comptime K = 8
comptime B = 3
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W

comptime HDS = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W]
comptime DDS = ACTDeviceDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W]


def store_path() raises -> String:
    """`$ACT_STORE`, else the newest store at this resolution — same
    resolution rule as `test_act_dataset.mojo`."""
    var os = Python.import_module("os")
    var env = os.environ.get(PythonObject("ACT_STORE"), PythonObject(""))
    var envs = String(env)
    if envs.byte_length() > 0:
        return envs
    var glob = Python.import_module("glob")
    var home = String(os.path.expanduser(PythonObject("~")))
    var pat = (
        home + "/.cache/mojo_rl/act_so101/*_" + String(IMG_H) + "x"
        + String(IMG_W) + ".h5"
    )
    var hits = glob.glob(PythonObject(pat))
    var builtins = Python.import_module("builtins")
    var n_hits = Int(String(builtins.len(hits)))
    if n_hits == 0:
        raise Error("no ACT store at " + pat + " — set ACT_STORE")
    var best = String(hits[0])
    var best_t = Float64(0.0)
    for i in range(n_hits):
        var cand = String(hits[i])
        var mt = Float64(String(os.path.getmtime(PythonObject(cand))))
        if mt > best_t:
            best_t = mt
            best = cand
    return best


def _maxdiff(ref a: List[Scalar[DT]], ref b: Tensor, n: Int) -> Float64:
    var m = Float64(0.0)
    for i in range(n):
        var d = abs(Float64(a[i]) - Float64(b.data[i]))
        if d > m:
            m = d
    return m


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def main() raises:
    var fails = 0
    var ctx = DeviceContext()
    print("ACT device-sampler parity gate")
    print("  device: " + String(ctx.name()))

    var path = store_path()
    var host = HDS(String(path), seed=11, max_image_bytes=0)  # streamed
    var dev = DDS.upload_from[B](host, ctx, seed=11)
    print("  uploaded " + String(dev.n_rows) + " rows ("
          + String(Float64(dev.n_rows) * Float64(IMG_ELEMS) / 1e9)
          + " GB of uint8)")

    # Rows chosen to include one that RUNS OFF the end of its episode, so the
    # normalized-zero padding rule is exercised and not just the happy path.
    var eps = host.train_eps.copy()
    if len(eps) == 0:
        print("  (no training episodes — nothing to compare)")
        exit(1)
    var ep0 = eps[0]
    var len0 = host.store.episodes.length_of(ep0)
    var st0 = host.store.episodes.start_of(ep0)
    var starts = List[Int]()
    starts.append(0)
    starts.append(len0 // 2)
    starts.append(len0 - 2)          # last one runs off the episode end
    var rows = List[Int]()
    var nreals = List[Int]()
    for i in range(B):
        var ts = starts[i]
        rows.append(st0 + ts)
        var rem = len0 - ts
        nreals.append(K if rem > K else rem)

    # host reference
    # ⚠ `fill_at` requires PRE-SIZED buffers (only `sample_batch` sizes them).
    var h_q = List[Scalar[DT]](length=B * QPOS, fill=Scalar[DT](0))
    var h_i = List[Scalar[DT]](length=B * IMG_ELEMS, fill=Scalar[DT](0))
    var h_a = List[Scalar[DT]](length=B * K * ADIM, fill=Scalar[DT](0))
    var h_v = List[Scalar[DT]](length=B * K, fill=Scalar[DT](0))
    for b in range(B):
        host.fill_at[K](b, ep0, starts[b], h_q, h_i, h_a, h_v)

    # device
    var d_q = Tensor()
    var d_i = Tensor()
    var d_a = Tensor()
    var d_v = Tensor()
    dev.gather_at[B, K](rows, nreals, d_q, d_i, d_a, d_v, ctx)
    ctx.synchronize()
    d_q.download(ctx)
    d_i.download(ctx)
    d_a.download(ctx)
    d_v.download(ctx)

    var dq = _maxdiff(h_q, d_q, B * QPOS)
    var di = _maxdiff(h_i, d_i, B * IMG_ELEMS)
    var da = _maxdiff(h_a, d_a, B * K * ADIM)
    var dv = _maxdiff(h_v, d_v, B * K)
    # fp32 elementwise on both sides; the only difference is the order of two
    # multiplies, so this is a tight bound, not a parity band.
    check(fails, "qpos", dq < 1e-5, "maxdiff " + String(dq))
    check(fails, "images", di < 1e-5, "maxdiff " + String(di))
    check(fails, "actions (incl. normalized-zero padding)", da < 1e-5,
          "maxdiff " + String(da))
    check(fails, "valid mask", dv == 0.0, "maxdiff " + String(dv))

    # the batch must not be trivially zero, or the comparison proves nothing
    var s = Float64(0.0)
    for i in range(B * IMG_ELEMS):
        s += abs(Float64(d_i.data[i]))
    check(fails, "the gathered images are non-trivial", s > 1.0,
          "sum|img| = " + String(s))

    # the truncated slot must actually be padded
    var pad_seen = False
    for t in range(K):
        if h_v[(B - 1) * K + t] == Scalar[DT](0.0):
            pad_seen = True
    check(fails, "slot 2 really is truncated (the pad path ran)", pad_seen,
          "n_real = " + String(nreals[B - 1]) + " of K=" + String(K))

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)
