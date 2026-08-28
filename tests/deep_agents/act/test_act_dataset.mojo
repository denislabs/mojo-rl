# +--------------------------------------------------------------------------+ #
# | M0 gate — the ACT dataset path
# +--------------------------------------------------------------------------+ #
"""Gates `ACTDataset` against facts established OUTSIDE this code path.

Every reference number here comes from somewhere else: the episode lengths from
LeRobot's `meta/episodes` parquet, the normalization statistics from the
converter's numpy pass (read back out of the store's `norm_*` datasets, which
`ACTDataset` deliberately does NOT read — it recomputes them). A gate that
shares its reference implementation cannot see a shared mistake.

    pixi run mojo run -I . tests/deep_agents/act/test_act_dataset.mojo

Requires a store; point `ACT_STORE` at one, or let it find the newest under
`~/.cache/mojo_rl/act_so101/`:

    pixi run python tools/act/lerobot_v3_to_store.py \
        --repo <hf-dataset> --height 240 --width 320

⚠ DATASET-AGNOSTIC BY CONSTRUCTION. An earlier version pinned 1997 rows / 5
episodes / a literal length list from one recording, which broke the moment a
second dataset arrived — and the pin was weak anyway: both the literals and the
store's `ep_len`/`ep_offset` trace back to the same `meta/episodes` parquet, so
agreement proved little. What carries the gate is the CROSS-IMPLEMENTATION work
that holds for any dataset: statistics recomputed in Mojo against the
converter's independent numpy pass, the padding semantics, the normalization
round-trip, and the structural invariants below.
"""

from std.python import Python, PythonObject

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.io.hdf5 import H5File
from mojo_rl.deep_agents.act.config import (
    IMAGENET_MEAN_R,
    IMAGENET_STD_R,
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from mojo_rl.deep_agents.act.data import ACTDataset


comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime H = SO101_IMG_H
comptime W = SO101_IMG_W
comptime HW = H * W
comptime CAM_ELEMS = 3 * HW
comptime IMG_ELEMS = N_CAM * CAM_ELEMS
comptime K = 100
comptime BATCH = 4

def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def store_path() raises -> String:
    """`$ACT_STORE`, else the most recently written store at this resolution."""
    var os = Python.import_module("os")
    var env = os.environ.get(PythonObject("ACT_STORE"), PythonObject(""))
    var envs = String(env)
    if envs.byte_length() > 0:
        return envs
    var glob = Python.import_module("glob")
    var home = String(os.path.expanduser(PythonObject("~")))
    var pat = (
        home + "/.cache/mojo_rl/act_so101/*_" + String(H) + "x" + String(W)
        + ".h5"
    )
    var hits = glob.glob(PythonObject(pat))
    var builtins = Python.import_module("builtins")
    var n_hits = Int(String(builtins.len(hits)))
    if n_hits == 0:
        raise Error(
            "no ACT store at " + pat + " — build one with"
            " tools/act/lerobot_v3_to_store.py, or set ACT_STORE"
        )
    var best = String(hits[0])
    var best_t = Float64(0.0)
    for i in range(n_hits):
        var cand = String(hits[i])
        var mt = Float64(String(os.path.getmtime(PythonObject(cand))))
        if mt > best_t:
            best_t = mt
            best = cand
    return best


def read_f32(path: String, name: String, n: Int) raises -> List[Scalar[DT]]:
    """Read a small float32 dataset the store carries but does not declare."""
    var f = H5File(String(path))
    var ds = f.open_dataset(String(name))
    if ds.n_elements() != n:
        raise Error(
            "gate: '" + name + "' has " + String(ds.n_elements())
            + " elements, expected " + String(n)
        )
    var out = List[Scalar[DT]](unsafe_uninit_length=n)
    ds.read_all[DT](out.unsafe_ptr().as_unsafe_any_origin())
    return out^


def main() raises:
    var fails = 0
    var path = store_path()
    var os = Python.import_module("os")
    if not Bool(os.path.exists(PythonObject(path))):
        print("MISSING STORE: " + path)
        print("build it with tools/act/lerobot_v3_to_store.py — see the header")
        raise Error("store not found")

    print("ACT dataset gate")
    print("  " + path)
    print("")

    var ds = ACTDataset[QPOS, ADIM, N_CAM, H, W](String(path), seed=1234)

    print(
        "  images: "
        + ("RESIDENT" if ds.images_resident else "STREAMED")
        + " ("
        + String(
            (ds.n_rows() * IMG_ELEMS) // (1 << 20)
        )
        + " MiB column)"
    )

    # ── 1. structural invariants (any dataset) ───────────────────────────
    print(
        "  " + String(ds.n_episodes()) + " episodes, "
        + String(ds.n_rows()) + " rows"
    )
    check(
        fails,
        "enough episodes for a train/val split",
        ds.n_episodes() >= 2,
        String(ds.n_episodes()) + " episodes",
    )
    # Episodes must TILE the row axis exactly: start at 0, each starting where
    # the last ended, summing to n_rows. A sampler reading a row outside any
    # episode, or two episodes overlapping, silently trains on transitions that
    # never happened.
    var contiguous = ds.n_episodes() > 0
    var total = 0
    var min_len = 1 << 30
    for e in range(ds.n_episodes()):
        if ds.store.episodes.start_of(e) != total:
            contiguous = False
        var le = ds.store.episodes.length_of(e)
        if le <= 0:
            contiguous = False
        min_len = min(min_len, le)
        total += le
    check(
        fails,
        "episodes tile the row axis exactly",
        contiguous and total == ds.n_rows(),
        "sum=" + String(total) + " vs n_rows=" + String(ds.n_rows()),
    )
    check(
        fails,
        "every episode is longer than one chunk boundary probe",
        min_len >= 2,
        "shortest episode = " + String(min_len),
    )

    # ── 2. statistics: Mojo (ddof=1) vs the converter's numpy (ddof=1) ───
    # `ACTDataset` recomputes; these datasets hold what numpy produced. Two
    # independent implementations of `utils.py:get_norm_stats`.
    var qm = read_f32(path, String("norm_qpos_mean"), QPOS)
    var qs = read_f32(path, String("norm_qpos_std"), QPOS)
    var am = read_f32(path, String("norm_action_mean"), ADIM)
    var as_ = read_f32(path, String("norm_action_std"), ADIM)

    # Scored in units of the column's own std, not in absolute degrees. An
    # absolute threshold here is a threshold on how wide the operator happened
    # to swing that joint, and it also scales with the row count: the earlier
    # absolute 1e-3 passed at 1997 rows and failed at 15447 for two
    # implementations that were both computing the right quantity. The std is
    # the scale everything downstream divides by, so an error small in those
    # units is small where it is actually used.
    var worst_stat = Float64(0.0)
    for j in range(QPOS):
        var sc = Float64(qs[j])
        worst_stat = max(
            worst_stat, abs(Float64(ds.qpos_mean[j]) - Float64(qm[j])) / sc
        )
        worst_stat = max(
            worst_stat, abs(Float64(ds.qpos_std[j]) - Float64(qs[j])) / sc
        )
    for j in range(ADIM):
        var sc = Float64(as_[j])
        worst_stat = max(
            worst_stat, abs(Float64(ds.action_mean[j]) - Float64(am[j])) / sc
        )
        worst_stat = max(
            worst_stat, abs(Float64(ds.action_std[j]) - Float64(as_[j])) / sc
        )
    # Both sides accumulate in float64 and the store rounds to float32, so
    # what is left is storage rounding — parts in 1e7, not 1e5. Anything
    # larger means one of the two reductions changed, which is the whole
    # point of keeping two of them.
    check(
        fails,
        "norm stats vs numpy (12 values, in std units)",
        worst_stat < 1e-5,
        "max rel = " + String(worst_stat),
    )
    # The stats must also be non-degenerate — a std pinned at the 1e-2 floor
    # everywhere would pass the comparison above and destroy training.
    var min_std = Float64(1e30)
    for j in range(ADIM):
        min_std = min(min_std, Float64(ds.action_std[j]))
    check(
        fails,
        "action std is off the 1e-2 floor",
        min_std > 1.0,
        "min std = " + String(min_std),
    )

    # ── 3. split is a partition ──────────────────────────────────────────
    var split_ok = (
        len(ds.train_eps) + len(ds.val_eps) == ds.n_episodes()
        and len(ds.val_eps) >= 1
        and len(ds.train_eps) >= 1
    )
    var seen = List[Bool](length=ds.n_episodes(), fill=False)
    for i in range(len(ds.train_eps)):
        if seen[ds.train_eps[i]]:
            split_ok = False
        seen[ds.train_eps[i]] = True
    for i in range(len(ds.val_eps)):
        if seen[ds.val_eps[i]]:
            split_ok = False
        seen[ds.val_eps[i]] = True
    for i in range(ds.n_episodes()):
        if not seen[i]:
            split_ok = False
    check(
        fails,
        "80/20 split is a partition",
        split_ok,
        String(len(ds.train_eps)) + " train / " + String(len(ds.val_eps))
        + " val",
    )

    # ── 4. chunk padding ─────────────────────────────────────────────────
    var qbuf = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var ibuf = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var abuf = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var vbuf = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)

    var ep0 = 0
    var ep0_len = ds.store.episodes.length_of(ep0)

    ds.fill_at[K](0, ep0, ep0_len - 1, qbuf, ibuf, abuf, vbuf)
    var n_valid_last = 0
    for t in range(K):
        if vbuf[t] > Scalar[DT](0.5):
            n_valid_last += 1
    check(
        fails,
        "final step of an episode -> 1 valid action",
        n_valid_last == 1,
        String(n_valid_last) + " valid",
    )

    ds.fill_at[K](0, ep0, 0, qbuf, ibuf, abuf, vbuf)
    var n_valid_first = 0
    for t in range(K):
        if vbuf[t] > Scalar[DT](0.5):
            n_valid_first += 1
    check(
        fails,
        "step 0 of a " + String(ep0_len) + "-step episode -> K valid",
        n_valid_first == K,
        String(n_valid_first) + " valid",
    )

    # Partial chunk: K-1 steps from the end leaves exactly 1 padded slot.
    ds.fill_at[K](0, ep0, ep0_len - K + 1, qbuf, ibuf, abuf, vbuf)
    var n_valid_mid = 0
    for t in range(K):
        if vbuf[t] > Scalar[DT](0.5):
            n_valid_mid += 1
    check(
        fails,
        "K-1 steps from the end -> K-1 valid",
        n_valid_mid == K - 1,
        String(n_valid_mid) + " valid",
    )

    # Padded slots hold the NORMALIZED zero, `(0-mean)/std`, not 0 — the
    # reference pads before normalizing (utils.py:52).
    ds.fill_at[K](0, ep0, ep0_len - 1, qbuf, ibuf, abuf, vbuf)
    var want_pad = -Float64(ds.action_mean[0]) / Float64(ds.action_std[0])
    var got_pad = Float64(abuf[(K - 1) * ADIM + 0])
    check(
        fails,
        "padded action == (0-mean)/std, not 0",
        abs(got_pad - want_pad) < 1e-5,
        "got " + String(got_pad) + ", want " + String(want_pad),
    )

    # ── 5. normalization inverts to the recorded units ───────────────────
    ds.fill_at[K](0, ep0, 7, qbuf, ibuf, abuf, vbuf)
    var g = ds.store.episodes.start_of(ep0) + 7
    var worst_inv = Float64(0.0)
    for j in range(QPOS):
        var back = (
            Float64(qbuf[j]) * Float64(ds.qpos_std[j])
            + Float64(ds.qpos_mean[j])
        )
        worst_inv = max(
            worst_inv, abs(back - Float64(ds.qpos_raw[g * QPOS + j]))
        )
    check(
        fails,
        "qpos normalization round-trips to the raw row",
        worst_inv < 1e-3,
        "max|diff| = " + String(worst_inv),
    )

    # Action t=0 must be the action recorded AT the observation step — the
    # check that catches an off-by-one in the chunk window.
    var worst_a0 = Float64(0.0)
    for j in range(ADIM):
        var back = (
            Float64(abuf[j]) * Float64(ds.action_std[j])
            + Float64(ds.action_mean[j])
        )
        worst_a0 = max(
            worst_a0, abs(back - Float64(ds.action_raw[g * ADIM + j]))
        )
    check(
        fails,
        "chunk[0] is the action AT the observation step (no shift)",
        worst_a0 < 1e-3,
        "max|diff| = " + String(worst_a0),
    )

    # ── 6. images ────────────────────────────────────────────────────────
    # Row `g` read here, by this test, straight off the store — not through
    # the sampler that produced `ibuf`. That is what makes this a check and
    # not a restatement, and it is the same reference for both residency
    # modes.
    var row = List[Scalar[DType.uint8]](unsafe_uninit_length=IMG_ELEMS)
    var img_ds = ds.store.open_column[DType.uint8](String("images"))
    img_ds.read_range[DType.uint8](g, g + 1, mptr(row))

    var worst_img = Float64(0.0)
    for p in range(0, HW, 4099):  # coprime stride across the R plane
        var v = (Float64(ibuf[p]) * IMAGENET_STD_R + IMAGENET_MEAN_R) * 255.0
        worst_img = max(worst_img, abs(v - Float64(Int(row[p]))))
    check(
        fails,
        "image normalization round-trips to the stored uint8",
        worst_img < 1e-2,
        "max|diff| = " + String(worst_img),
    )

    # The two camera slots must not hold the same bytes — a slot-indexing bug
    # would duplicate one feed and pass every check above.
    var cam_diff = Float64(0.0)
    for p in range(0, HW, 997):
        cam_diff += abs(Float64(ibuf[p]) - Float64(ibuf[CAM_ELEMS + p]))
    check(
        fails,
        "camera slots carry different images",
        cam_diff > 1.0,
        "sum|front-side| = " + String(cam_diff),
    )

    # ── 6b. the streamed image path == the resident one ──────────────────
    # `max_image_bytes=0` forces streaming regardless of the store's size, so
    # this runs on any dataset. Against a store small enough to be resident it
    # is a genuine cross-path comparison (whole-column `load_column` read vs
    # per-row `read_range`); against one too large to be resident both sides
    # stream and it degrades to checking that the staging buffer is not
    # reused across slots. Bit-exact either way — no arithmetic differs.
    var dstream = ACTDataset[QPOS, ADIM, N_CAM, H, W](
        String(path), seed=1234, max_image_bytes=0
    )
    check(
        fails,
        "max_image_bytes=0 forces the streamed path",
        not dstream.images_resident,
    )
    var sbuf_q = List[Scalar[DT]](unsafe_uninit_length=2 * QPOS)
    var sbuf_i = List[Scalar[DT]](unsafe_uninit_length=2 * IMG_ELEMS)
    var sbuf_a = List[Scalar[DT]](unsafe_uninit_length=2 * K * ADIM)
    var sbuf_v = List[Scalar[DT]](unsafe_uninit_length=2 * K)
    # Two different steps into two different slots: a staging buffer that is
    # not refilled per sample would leave slot 1 holding slot 0's frame.
    dstream.fill_at[K](0, ep0, 7, sbuf_q, sbuf_i, sbuf_a, sbuf_v)
    dstream.fill_at[K](1, ep0, ep0_len - 1, sbuf_q, sbuf_i, sbuf_a, sbuf_v)

    var worst_stream = Float64(0.0)
    for pi in range(0, IMG_ELEMS, 1021):
        worst_stream = max(
            worst_stream, abs(Float64(sbuf_i[pi]) - Float64(ibuf[pi]))
        )
    check(
        fails,
        "streamed images == the images the live path produced",
        worst_stream == 0.0,
        "max|diff| = " + String(worst_stream),
    )
    var slot_diff = Float64(0.0)
    for pi in range(0, IMG_ELEMS, 1021):
        slot_diff += abs(
            Float64(sbuf_i[pi]) - Float64(sbuf_i[IMG_ELEMS + pi])
        )
    check(
        fails,
        "each streamed slot holds its OWN frame",
        slot_diff > 1.0,
        "sum|slot0-slot1| = " + String(slot_diff),
    )

    # ── 7. batch sampling ────────────────────────────────────────────────
    ds.sample_batch[K, BATCH](False, qbuf, ibuf, abuf, vbuf)
    var every_slot = True
    for b in range(BATCH):
        # step 0 of any chunk is always a real action
        if vbuf[b * K] <= Scalar[DT](0.5):
            every_slot = False
    check(fails, "sample_batch(train) fills every slot", every_slot)

    ds.sample_batch[K, BATCH](True, qbuf, ibuf, abuf, vbuf)
    check(fails, "sample_batch(val) runs on the 1-episode split", True)

    # ── 8. reproducible RNG stream ───────────────────────────────────────
    var d1 = ACTDataset[QPOS, ADIM, N_CAM, H, W](String(path), seed=777)
    var d2 = ACTDataset[QPOS, ADIM, N_CAM, H, W](String(path), seed=777)
    var q1 = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var q2 = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var i1 = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var i2 = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var a1 = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var a2 = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var v1 = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)
    var v2 = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)
    d1.sample_batch[K, BATCH](False, q1, i1, a1, v1)
    d2.sample_batch[K, BATCH](False, q2, i2, a2, v2)
    var same = True
    var moved = False
    for i in range(BATCH * QPOS):
        if q1[i] != q2[i]:
            same = False
        if q1[i] != qbuf[i]:
            moved = True  # guards against "all zeros compare equal"
    check(fails, "same seed -> same batch", same)
    check(fails, "the sampler actually varies with the seed", moved)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("act dataset gate failed")
