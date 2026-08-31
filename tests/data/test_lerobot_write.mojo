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

from mojo_rl.data.lerobot import import_lerobot_v3
from mojo_rl.data.lerobot_write import LeRobotWriter
from mojo_rl.data.store import TrajectoryStore


comptime H = 48
comptime W = 64
comptime FPS = 30
comptime SDIM = 3
comptime ADIM = 2
comptime N_EP = 3

comptime EP_LENS: InlineArray[Int, N_EP] = [4, 7, 3]
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
                var got = InlineArray[Int, 3](fill=0)
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


def main() raises:
    print("[lerobot-write] gate")

    var total = 0

    # One mp4 per camera: episodes located by `from_timestamp`.
    var r1 = String("/tmp/mojo_rl_lw_packed")
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

    if total < 100:
        raise Error("only " + String(total) + " checks ran")
    print("  " + String(total) + " checks, 0 failures")
    print("[PASS] lerobot-write")
