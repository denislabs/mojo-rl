# +--------------------------------------------------------------------------+ #
# | norm.json — the deployment's statistics, identical to the dataset's
# +--------------------------------------------------------------------------+ #
"""Gate `deep_agents/act/norm_file.mojo`.

    pixi run mojo run -I . tests/deep_agents/act/test_act_norm_file.mojo

Uses the cached 5-episode store (`ACT_STORE` overrides), with images NOT
loaded. ⚠ The comparison is BIT equality of every statistic against the
`ACTDataset` it came from, after a save and a load: the file exists so a
deployment can use these numbers instead of the store's, and "close" is a
policy fed shifted inputs.
"""

from std.os import getenv
from std.os.path import exists

from noeira.deep_agents.act.data import ACTDataset
from noeira.deep_agents.act.norm_file import ACTNorm, act_norm_from
from noeira.nn.constants import DT


comptime QPOS = 6
comptime ADIM = 6
comptime N_CAM = 2
comptime H = 240
comptime W = 320


def main() raises:
    print("[act-norm-file] gate")
    var store = getenv("ACT_STORE")
    if store.byte_length() == 0:
        store = getenv("HOME") + "/.cache/noeira/act_so101/DenisLabs__record-test_20260825_094319_240x320.h5"
    if not exists(store):
        raise Error("no store at " + store + " — set ACT_STORE (see test_act_dataset.mojo)")
    var ds = ACTDataset[QPOS, ADIM, N_CAM, H, W](store.copy(), seed=7, max_image_bytes=0)
    var cams = List[String]()
    cams.append(String("observation.images.overhead"))
    cams.append(String("observation.images.wrist"))
    var n = act_norm_from(
        ds.qpos_raw, ds.action_raw, ds.n_rows(), ds.n_episodes(),
        ds.qpos_mean, ds.qpos_std, ds.action_mean, ds.action_std,
        cams, H, W, store,
    )
    var path = String("/tmp/noeira_act_norm_gate.json")
    n.save(path)
    var back = ACTNorm.load(path, QPOS, ADIM)

    var checks = 0
    for j in range(QPOS):
        if back.qpos_mean[j] != ds.qpos_mean[j] or back.qpos_std[j] != ds.qpos_std[j]:
            raise Error("qpos stat " + String(j) + " did not round-trip bit-exact")
        checks += 2
    for j in range(ADIM):
        if back.action_mean[j] != ds.action_mean[j] or back.action_std[j] != ds.action_std[j]:
            raise Error("action stat " + String(j) + " did not round-trip bit-exact")
        checks += 2
    # The boxes against a loop written HERE over the raw columns, not the
    # module's own loop: min/max is simple, but it is the one piece of the file
    # the module computes rather than copies.
    for j in range(ADIM):
        var lo = Float64(1e18)
        var hi = Float64(-1e18)
        for r in range(ds.n_rows()):
            var v = Float64(ds.action_raw[r * ADIM + j])
            lo = v if v < lo else lo
            hi = v if v > hi else hi
        if Float32(back.action_min[j]) != Float32(lo) or Float32(back.action_max[j]) != Float32(hi):
            raise Error("action box " + String(j) + " differs from the columns")
        checks += 2
    for j in range(QPOS):
        var lo = Float64(1e18)
        for r in range(ds.n_rows()):
            var v = Float64(ds.qpos_raw[r * QPOS + j])
            lo = v if v < lo else lo
        if Float32(back.qpos_min[j]) != Float32(lo):
            raise Error("qpos box " + String(j) + " differs from the columns")
        checks += 1
    if len(back.cameras) != 2 or back.cameras[1] != "observation.images.wrist" or back.img_h != H or back.n_rows != ds.n_rows():
        raise Error("cameras / shape / provenance did not round-trip")
    checks += 1
    print("  " + String(ds.n_rows()) + " rows: 24 statistics bit-exact after save+load, boxes match the raw columns, cameras kept")

    var refused = False
    try:
        _ = ACTNorm.load(path, 7, ADIM)
    except:
        refused = True
    if not refused:
        raise Error("a 6-joint norm file must be refused by a 7-joint build")
    checks += 1
    print("  a norm file for another joint count is refused")
    print("  " + String(checks) + " checks, 0 failures")
    print("[PASS] act-norm-file")
