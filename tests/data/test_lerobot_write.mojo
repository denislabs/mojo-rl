# +--------------------------------------------------------------------------+ #
# | The LeRobot v3 WRITER, read back by the importer that predates it
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/data/lerobot_write.mojo` — Leg B of the recording plan.

    pixi run mojo run -I . tests/data/test_lerobot_write.mojo

Writes a dataset, then reads it with `import_lerobot_v3` — code written months
earlier, against files `lerobot-record` produced, which knows nothing about
this writer. If the writer's `meta/info.json`, `meta/episodes/*.parquet`,
`meta/tasks.parquet`, `data/*.parquet` or its mp4s are wrong in any way the
importer looks at, this fails.

⚠ THAT IS A DIFFERENT QUESTION FROM `test_parquet_write.mojo`. That gate asks
whether the BYTES are a valid Parquet file, against a fixture Arrow wrote.
This one asks whether the DATASET is a valid LeRobot v3 dataset — the episode
index, the `from_timestamp` routing into packed videos, the feature schema.
Neither subsumes the other, and this one needs no fixture at all.

## What the shape of the test data is for

⚠ **UNEQUAL EPISODE LENGTHS (4, 7, 3).** Equal lengths make `ep_offset` a
multiple of a constant, so an offset computed by multiplication instead of by
accumulation passes.

⚠ **TWO CAMERAS, AND `state_dim != action_dim`.** A single camera cannot catch
a camera-slot ordering bug, and equal dims cannot catch a state/action swap.

⚠ **THE FRAMES ARE FLAT COLOURS THAT ENCODE (episode, t, camera).** H.264 at
crf 30 is lossy, so a byte comparison is impossible — but a routing bug is not
a small error, it is the WRONG FRAME. Each frame is a solid colour carrying
its own coordinates, so after import every row can be asked "which frame are
you", and the answer must be exact even though the pixels are approximate.

⚠ **THE DATASET IS WRITTEN TWICE, WITH TWO VIDEO PACKINGS.** With the default
size limit all episodes land in ONE mp4 per camera, so episodes are located by
`from_timestamp` and the file index never moves. With `video_mb=0` every
episode rolls into its OWN file, so `from_timestamp` is always 0 and the file
index does all the work. Those are two different routing paths through
`CameraStream`, and a dataset that exercises one says nothing about the other.
"""

from std.os import makedirs
from std.os.path import exists
from std.memory import Pointer

from mojo_rl.data.lerobot import free_bytes, import_lerobot_v3
from mojo_rl.data.lerobot_rejected import (
    kept_rows,
    load_rejected_episodes,
    refuse_existing_dataset,
    reject_episode,
    save_rejected_episodes,
)
from mojo_rl.data.lerobot_write import LeRobotWriter
from mojo_rl.io.fileio import remove_file
from mojo_rl.io.proc import run_capture
from mojo_rl.data.store import TrajectoryStore


comptime H = 48
comptime W = 64
comptime FPS = 30
comptime SDIM = 3
comptime ADIM = 2
comptime N_EP = 3

comptime EP_LENS: Array[Int, N_EP] = [4, 7, 3]
comptime N_ROWS = 14


def _signature(ep: Int, t: Int, cam: Int) -> Tuple[Int, Int, Int]:
    """The flat colour a frame carries, as (r, g, b).

    Spread well apart so lossy compression cannot move one into another's
    neighbourhood: the nearest pair is 20 apart per channel.
    """
    return (20 + ep * 60, 20 + t * 20, 40 + cam * 120)


def _write_dataset(root: String, video_mb: Int) raises:
    var sn = List[String]()
    for n in [
        String("shoulder_pan.pos"), String("shoulder_lift.pos"),
        String("elbow_flex.pos"),
    ]:
        sn.append(n)
    var an = List[String]()
    for n in [String("gripper.pos"), String("wrist_flex.pos")]:
        an.append(n)
    var cams = List[String]()
    cams.append(String("observation.images.front"))
    cams.append(String("observation.images.side"))

    var w = LeRobotWriter(
        root, FPS, sn^, an^, cams^, H, W, video_mb=video_mb
    )
    var lens = materialize[EP_LENS]()
    for ep in range(N_EP):
        w.begin_episode(String("episode task ") + String(ep % 2))
        for t in range(lens[ep]):
            var st = List[Float64]()
            var ac = List[Float64]()
            # Quarters: exactly representable in float32, so the comparison
            # below can be EXACT rather than tolerant.
            for d in range(SDIM):
                st.append(Float64(ep * 16 + t * 4 + d) + 0.25)
            for d in range(ADIM):
                ac.append(Float64(ep * 16 + t * 4 + d) * -1.0 - 0.5)
            var frames = List[List[UInt8]]()
            for cam in range(2):
                var sig = _signature(ep, t, cam)
                var f = List[UInt8](unsafe_uninit_length = W * H * 3)
                for p in range(W * H):
                    f[p * 3] = UInt8(sig[0])
                    f[p * 3 + 1] = UInt8(sig[1])
                    f[p * 3 + 2] = UInt8(sig[2])
                frames.append(f^)
            w.add_frame(st, ac, frames)
        w.end_episode()
    w.close(verbose=False)


def _check(root: String, h5: String, label: String) raises -> Int:
    """Import `root` and verify the store against what was written."""
    import_lerobot_v3(root, h5, H, W, verbose=False)
    var s = TrajectoryStore(h5)
    var lens = materialize[EP_LENS]()

    if s.n_rows() != N_ROWS:
        raise Error(
            label + ": the store has " + String(s.n_rows()) + " rows, wrote "
            + String(N_ROWS)
        )
    if s.n_episodes() != N_EP:
        raise Error(
            label + ": the store has " + String(s.n_episodes())
            + " episodes, wrote " + String(N_EP)
        )
    var off = 0
    for e in range(N_EP):
        if s.episodes.start_of(e) != off or s.episodes.length_of(e) != lens[e]:
            raise Error(
                label + ": episode " + String(e) + " is at ("
                + String(s.episodes.start_of(e)) + ", "
                + String(s.episodes.length_of(e)) + "), wrote (" + String(off)
                + ", " + String(lens[e]) + ")"
            )
        off += lens[e]

    # ── qpos / action, exactly ────────────────────────────────────────
    var qpos = List[Float32](unsafe_uninit_length = N_ROWS * SDIM)
    var act = List[Float32](unsafe_uninit_length = N_ROWS * ADIM)
    s.read_range[DType.float32](
        String("qpos"), 0, N_ROWS,
        qpos.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin(),
    )
    s.read_range[DType.float32](
        String("action"), 0, N_ROWS,
        act.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin(),
    )

    var compared = 0
    var row = 0
    for ep in range(N_EP):
        for t in range(lens[ep]):
            for d in range(SDIM):
                var want = Float32(Float64(ep * 16 + t * 4 + d) + 0.25)
                if qpos[row * SDIM + d] != want:
                    raise Error(
                        label + ": qpos[" + String(row) + "][" + String(d)
                        + "] is " + String(qpos[row * SDIM + d]) + ", wrote "
                        + String(want)
                    )
                compared += 1
            for d in range(ADIM):
                var want = Float32(
                    Float64(ep * 16 + t * 4 + d) * -1.0 - 0.5
                )
                if act[row * ADIM + d] != want:
                    raise Error(
                        label + ": action[" + String(row) + "][" + String(d)
                        + "] is " + String(act[row * ADIM + d]) + ", wrote "
                        + String(want)
                    )
                compared += 1
            row += 1

    # ── images: which frame is in this row? ───────────────────────────
    # The store's images column is [n_cam, 3, H, W], channel-major.
    var per_cam = 3 * H * W
    var img = List[UInt8](unsafe_uninit_length = 2 * per_cam)
    var routed = 0
    row = 0
    for ep in range(N_EP):
        for t in range(lens[ep]):
            s.read_range[DType.uint8](
                String("images"), row, row + 1,
                img.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
                .as_unsafe_any_origin(),
            )
            for cam in range(2):
                var base = cam * per_cam
                # Mean of each channel plane.
                var got = Array[Int, 3](fill=0)
                for ch in range(3):
                    var acc = 0
                    for p in range(H * W):
                        acc += Int(img[base + ch * H * W + p])
                    got[ch] = acc // (H * W)
                var want = _signature(ep, t, cam)
                # ⚠ TOLERANT PER CHANNEL, EXACT AS AN IDENTITY. crf 30 moves a
                # flat colour by a few levels; it cannot move it 20, which is
                # the spacing between two different frames' signatures.
                var dr = got[0] - want[0]
                var dg = got[1] - want[1]
                var db = got[2] - want[2]
                if dr < 0: dr = -dr
                if dg < 0: dg = -dg
                if db < 0: db = -db
                if dr > 8 or dg > 8 or db > 8:
                    raise Error(
                        label + ": row " + String(row) + " camera "
                        + String(cam) + " decoded rgb(" + String(got[0]) + ","
                        + String(got[1]) + "," + String(got[2])
                        + "), episode " + String(ep) + " t=" + String(t)
                        + " wrote rgb(" + String(want[0]) + ","
                        + String(want[1]) + "," + String(want[2])
                        + ") — the wrong frame is in this row"
                    )
                routed += 1
            row += 1

    if routed != N_ROWS * 2:
        raise Error(
            label + ": checked " + String(routed) + " frames, expected "
            + String(N_ROWS * 2)
        )
    print(
        "  " + label + ": " + String(compared) + " state/action values exact, "
        + String(routed) + " frames routed to the right row"
    )
    return compared + routed


def _names() -> Tuple[List[String], List[String], List[String]]:
    var sn = List[String]()
    for n in [
        String("shoulder_pan.pos"), String("shoulder_lift.pos"),
        String("elbow_flex.pos"),
    ]:
        sn.append(n)
    var an = List[String]()
    for n in [String("gripper.pos"), String("wrist_flex.pos")]:
        an.append(n)
    var cams = List[String]()
    cams.append(String("observation.images.front"))
    cams.append(String("observation.images.side"))
    return (sn^, an^, cams^)


def _record_episode(mut w: LeRobotWriter, ep: Int, n: Int, end: Bool = True) raises:
    """Episode `ep`'s deterministic rows and frames — the same formulas as
    `_write_dataset`, so `_verify` can recognise every row by its source."""
    w.begin_episode(String("episode task ") + String(ep % 2))
    for t in range(n):
        var st = List[Float64]()
        var ac = List[Float64]()
        for d in range(SDIM):
            st.append(Float64(ep * 16 + t * 4 + d) + 0.25)
        for d in range(ADIM):
            ac.append(Float64(ep * 16 + t * 4 + d) * -1.0 - 0.5)
        var frames = List[List[UInt8]]()
        for cam in range(2):
            var sig = _signature(ep, t, cam)
            var f = List[UInt8](unsafe_uninit_length = W * H * 3)
            for p in range(W * H):
                f[p * 3] = UInt8(sig[0])
                f[p * 3 + 1] = UInt8(sig[1])
                f[p * 3 + 2] = UInt8(sig[2])
            frames.append(f^)
        w.add_frame(st, ac, frames)
    if end:
        w.end_episode()


def _verify(h5: String, ref eps: List[Int], ref lens: List[Int], label: String) raises -> Int:
    """The store holds exactly source episodes `eps`, in order: row counts,
    qpos values and which frame sits in every row."""
    var s = TrajectoryStore(h5)
    var rows = 0
    for l in lens:
        rows += l
    if s.n_episodes() != len(eps) or s.n_rows() != rows:
        raise Error(
            label + ": store has " + String(s.n_episodes()) + " episodes / "
            + String(s.n_rows()) + " rows, expected " + String(len(eps))
            + " / " + String(rows)
        )
    var qpos = List[Float32](unsafe_uninit_length = rows * SDIM)
    s.read_range[DType.float32](
        String("qpos"), 0, rows,
        qpos.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin(),
    )
    var img = List[UInt8](unsafe_uninit_length = 2 * 3 * H * W)
    var n = 0
    var row = 0
    for k in range(len(eps)):
        var ep = eps[k]
        for t in range(lens[k]):
            if qpos[row * SDIM] != Float32(Float64(ep * 16 + t * 4) + 0.25):
                raise Error(label + ": row " + String(row) + " is not episode " + String(ep) + " t=" + String(t))
            s.read_range[DType.uint8](
                String("images"), row, row + 1,
                img.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
                .as_unsafe_any_origin(),
            )
            for cam in range(2):
                var base = cam * 3 * H * W
                var acc = 0
                for p in range(H * W):
                    acc += Int(img[base + p])
                var d = acc // (H * W) - _signature(ep, t, cam)[0]
                if d < -8 or d > 8:
                    raise Error(label + ": row " + String(row) + " camera " + String(cam) + " holds another episode's frame")
            n += 3
            row += 1
    return n


def _check_checkpoint_resume() raises -> Int:
    """Crash mid-episode, import what survived, resume, finish, import again.

    ⚠⚠ THE CRASH IS SIMULATED BY NEVER CALLING `close()`. The writer is
    abandoned with episode 2 half-recorded, exactly the state a killed
    recorder leaves: its video file exists and is unreferenced, and all
    metadata is whatever the last `end_episode` wrote.
    """
    var root = String("/tmp/mojo_rl_lw_ckpt")
    _ = run_capture(String("rm -rf ") + root)
    var n = 0
    var names = _names()

    # ── session 1: two episodes, then a crash inside the third ────────
    var w = LeRobotWriter(
        root, FPS, names[0].copy(), names[1].copy(), names[2].copy(), H, W,
        checkpoint=True,
    )
    _record_episode(w, 0, 4)
    _record_episode(w, 1, 7)
    _record_episode(w, 2, 3, end=False)
    for c in range(2):
        _ = w._enc[c].stop()  # release ffmpeg; nothing else is finished

    if not exists(root + "/meta/mojo_rl_writer_stats.json"):
        raise Error("checkpoint: no resume state after two episodes")
    import_lerobot_v3(root, String("/tmp/mojo_rl_lw_ckpt_crash.h5"), H, W, verbose=False)
    var e1 = List[Int]()
    e1.append(0)
    e1.append(1)
    var l1 = List[Int]()
    l1.append(4)
    l1.append(7)
    n += _verify(String("/tmp/mojo_rl_lw_ckpt_crash.h5"), e1, l1, String("after crash"))
    print("  checkpoint: crash inside episode 3 -> the dataset imports with episodes 0 and 1 intact")

    # ── session 2: resume, record two more, finish ────────────────────
    var r = LeRobotWriter.resume(
        root, FPS, names[0].copy(), names[1].copy(), names[2].copy(), H, W
    )
    if r.n_episodes() != 2 or r.n_rows() != 11:
        raise Error("resume: expected 2 episodes / 11 rows, got " + String(r.n_episodes()) + " / " + String(r.n_rows()))
    _record_episode(r, 2, 3)
    _record_episode(r, 3, 5)
    r.close(verbose=False)
    import_lerobot_v3(root, String("/tmp/mojo_rl_lw_ckpt_resumed.h5"), H, W, verbose=False)
    var e2 = List[Int]()
    var l2 = List[Int]()
    for pair in [(0, 4), (1, 7), (2, 3), (3, 5)]:
        e2.append(pair[0])
        l2.append(pair[1])
    n += _verify(String("/tmp/mojo_rl_lw_ckpt_resumed.h5"), e2, l2, String("resumed"))
    print("  resume: 2 recorded + 2 resumed -> 4 episodes, every row and frame from its own episode")

    # ── refusals ──────────────────────────────────────────────────────
    var refused = 0
    try:
        _ = LeRobotWriter.resume(root, FPS + 1, names[0].copy(), names[1].copy(), names[2].copy(), H, W)
    except:
        refused += 1
    var one_cam = List[String]()
    one_cam.append(String("observation.images.front"))
    try:
        _ = LeRobotWriter.resume(root, FPS, names[0].copy(), names[1].copy(), one_cam^, H, W)
    except:
        refused += 1
    try:
        _ = LeRobotWriter.resume(String("/tmp/mojo_rl_lw_packed"), FPS, names[0].copy(), names[1].copy(), names[2].copy(), H, W)
    except:
        refused += 1
    if refused != 3:
        raise Error("resume: expected 3 refusals (fps, camera set, non-checkpointed), got " + String(refused))
    print("  resume refuses: another fps, another camera set, a dataset recorded without checkpoints")
    return n + 3


def _check_rejected(root: String, h5: String) raises -> Int:
    """Discard episode 1 of `root`, import, and verify the store holds exactly
    episodes 0 and 2 — rows, values and frames.

    ⚠ THE MIDDLE EPISODE, on purpose. Dropping the last one is satisfied by
    an importer that merely stops early; dropping the middle one needs every
    later row and frame to shift, which is where an off-by-an-episode lives.
    """
    var n = 0
    var lens = materialize[EP_LENS]()

    # ── the list itself: sorted, unique, round-trips ──────────────────
    var messy = List[Int]()
    messy.append(1)
    save_rejected_episodes(root, messy)
    var again = reject_episode(root, 1)
    if len(again) != 1 or again[0] != 1:
        raise Error("rejecting the same episode twice must not duplicate it")
    if len(load_rejected_episodes(root)) != 1:
        raise Error("rejected_episodes.json did not round-trip")
    n += 1

    import_lerobot_v3(root, h5, H, W, verbose=False)
    var s = TrajectoryStore(h5)
    var want_rows = lens[0] + lens[2]
    if s.n_episodes() != 2 or s.n_rows() != want_rows:
        raise Error(
            "rejected: the store has " + String(s.n_episodes()) + " episodes / "
            + String(s.n_rows()) + " rows, expected 2 / " + String(want_rows)
        )
    if s.episodes.length_of(0) != lens[0] or s.episodes.length_of(1) != lens[2]:
        raise Error("rejected: the kept episodes have the wrong lengths")
    n += 1

    var qpos = List[Float32](unsafe_uninit_length = want_rows * SDIM)
    s.read_range[DType.float32](
        String("qpos"), 0, want_rows,
        qpos.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin(),
    )
    var per_cam = 3 * H * W
    var img = List[UInt8](unsafe_uninit_length = 2 * per_cam)
    var row = 0
    for ep in [0, 2]:
        for t in range(lens[ep]):
            var want = Float32(Float64(ep * 16 + t * 4 + 0) + 0.25)
            if qpos[row * SDIM] != want:
                raise Error(
                    "rejected: store row " + String(row) + " holds qpos "
                    + String(qpos[row * SDIM]) + ", expected episode "
                    + String(ep) + " t=" + String(t) + " (" + String(want) + ")"
                )
            s.read_range[DType.uint8](
                String("images"), row, row + 1,
                img.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
                .as_unsafe_any_origin(),
            )
            var acc = 0
            for p in range(H * W):
                acc += Int(img[p])  # camera 0, red plane
            var dr = acc // (H * W) - _signature(ep, t, 0)[0]
            if dr < -8 or dr > 8:
                raise Error(
                    "rejected: store row " + String(row) + " holds a frame of"
                    " the wrong episode (red " + String(acc // (H * W)) + ")"
                )
            n += 2
            row += 1
    print(
        "  rejected: episode 1 of 3 discarded -> 2 episodes, " + String(want_rows)
        + " rows, values and frames of episodes 0 and 2"
    )

    # ── the statistics exclude the rejected rows ──────────────────────
    var x = List[Float32]()
    var starts = List[Int]()
    var ls = List[Int]()
    var off = 0
    for ep in range(N_EP):
        starts.append(off)
        ls.append(lens[ep])
        for t in range(lens[ep]):
            x.append(Float32(ep * 100 + t))
        off += lens[ep]
    var rej = List[Int]()
    rej.append(1)
    var k = kept_rows(x, 1, starts, ls, rej)
    if len(k) != lens[0] + lens[2] or k[lens[0] - 1] != Float32(lens[0] - 1) or k[lens[0]] != Float32(200):
        raise Error("kept_rows: wrong rows kept for the statistics")
    print("  kept_rows: the stats see episodes 0 and 2 only")
    n += 1

    # ── a rejection past the end is refused, not ignored ──────────────
    _ = reject_episode(root, N_EP)
    var raised = False
    try:
        import_lerobot_v3(root, h5 + ".bad", H, W, verbose=False)
    except:
        raised = True
    if not raised:
        raise Error("rejecting episode " + String(N_EP) + " of " + String(N_EP) + " must be refused")
    print("  out-of-range rejection refused")
    if free_bytes(String("/tmp")) <= 0:
        raise Error("free_bytes(/tmp) must report free space — the import's disk preflight would be blind")
    print("  free_bytes: /tmp reports " + String(free_bytes(String("/tmp")) // 1_000_000_000) + " GB free")
    n += 1

    # ── a recorder refuses to write over an existing recording ────────
    var refused = False
    try:
        refuse_existing_dataset(root)
    except:
        refused = True
    if not refused:
        raise Error("refuse_existing_dataset accepted a directory holding a dataset")
    print("  an existing recording is refused as --out")
    n += 1

    remove_file(root + "/meta/rejected_episodes.json")
    return n


def main() raises:
    print("[lerobot-write] gate")

    var total = 0

    # One mp4 per camera: episodes located by `from_timestamp`.
    var r1 = String("/tmp/mojo_rl_lw_packed")
    # A run that died inside the rejected leg leaves its list behind, and the
    # writer does not clear the directory.
    if exists(r1 + "/meta/rejected_episodes.json"):
        remove_file(r1 + "/meta/rejected_episodes.json")
    _write_dataset(r1, 100)
    total += _check(r1, String("/tmp/mojo_rl_lw_packed.h5"), String("packed"))

    # One mp4 per episode: located by `file_index`, `from_timestamp` always 0.
    var r2 = String("/tmp/mojo_rl_lw_rolled")
    _write_dataset(r2, 0)
    total += _check(r2, String("/tmp/mojo_rl_lw_rolled.h5"), String("rolled"))

    if not exists(
        r2 + "/videos/observation.images.front/chunk-000/file-002.mp4"
    ):
        raise Error(
            "the `rolled` dataset has no file-002.mp4 — video rolling did not"
            " happen, so that leg tested the same packing as the first"
        )
    print("  rolling: the rolled dataset really produced one file per episode")

    total += _check_rejected(r1, String("/tmp/mojo_rl_lw_rejected.h5"))
    total += _check_checkpoint_resume()

    if total < 100:
        raise Error("only " + String(total) + " checks ran")
    print("  " + String(total) + " checks, 0 failures")
    print("[PASS] lerobot-write")
