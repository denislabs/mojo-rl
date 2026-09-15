# +--------------------------------------------------------------------------+ #
# | A recording mirrored on the platform while it is still being recorded
# +--------------------------------------------------------------------------+ #
"""Gate `data/recording_files.mojo` (what is safe to send) and
`data/dataset_sync.mojo` (sending it, and bringing it back).

    pixi run mojo run -I . tests/data/test_dataset_sync.mojo

Hermetic: `tools/io/mock_monitor_server.py` plays the Worker and R2.

⚠⚠ THE WRITER IS STILL RECORDING DURING THE FIRST PUSH. Episode 2 is begun and
has frames in its mp4 when `push_dataset` runs: that is the `--watch` case,
and the one where a naive mirror uploads a half-written video. The checks
assert the bytes on the far side, not the counts the report prints.
"""

from std.os import makedirs
from std.os.path import exists
from std.time import sleep

from mojo_rl.data.dataset_sync import (
    HashCache,
    dataset_path_refusal,
    pull_dataset,
    push_dataset,
    watch_is_done,
)
from mojo_rl.data.lerobot import import_lerobot_v3
from mojo_rl.data.lerobot_push import dataset_files
from mojo_rl.data.lerobot_write import LeRobotWriter
from mojo_rl.data.recording_files import plan_upload, recording_finished, video_file_index
from mojo_rl.data.remote import RemoteCatalog
from mojo_rl.data.store import TrajectoryStore
from mojo_rl.io.fileio import remove_file
from mojo_rl.io.http import HttpClient, http_shim_available
from mojo_rl.io.proc import run_capture
from mojo_rl.io.sha256 import sha256_file


comptime WORK = "/tmp/mojo_rl_dataset_sync_gate"
comptime PORT_FILE = "/tmp/mojo_rl_dataset_sync_gate_port"
comptime LOG_FILE = "/tmp/mojo_rl_dataset_sync_gate_log"
comptime SLUG = "so101-tower"
comptime NAME = "cube-in-bowl"
comptime H = 48
comptime W = 64


def _check(cond: Bool, what: String) raises:
    if not cond:
        raise Error(what)


def _episode(mut w: LeRobotWriter, ep: Int, n: Int, end: Bool = True) raises:
    w.begin_episode(String("Grab the cube and put it in the bowl"))
    for t in range(n):
        var st = List[Float64]()
        var ac = List[Float64]()
        for d in range(6):
            st.append(Float64(ep * 16 + t * 4 + d) + 0.25)
            ac.append(Float64(ep * 16 + t * 4 + d) - 0.5)
        var frames = List[List[UInt8]]()
        for cam in range(2):
            var f = List[UInt8](unsafe_uninit_length = W * H * 3)
            for p in range(W * H * 3):
                f[p] = UInt8((20 + ep * 60 + t * 20 + cam * 40) & 0xFF)
            frames.append(f^)
        w.add_frame(st, ac, frames)
    if end:
        w.end_episode()


def _start_server() raises -> String:
    for p in [String(PORT_FILE), String(LOG_FILE)]:
        try:
            remove_file(p)
        except:
            pass
    _ = run_capture(
        "python3 tools/io/mock_monitor_server.py " + String(PORT_FILE) + " "
        + String(LOG_FILE) + " 240 > /tmp/mojo_rl_dataset_sync_gate_server.log 2>&1 &"
    )
    for _ in range(100):
        if exists(PORT_FILE):
            var f = open(String(PORT_FILE), "r")
            var port = String(f.read().strip())
            f.close()
            if port.byte_length() > 0:
                return "http://127.0.0.1:" + port
        sleep(0.1)
    raise Error("the mock monitor never wrote " + String(PORT_FILE))


def _object_puts(needle: String) raises -> Int:
    """Object PUTs whose path contains `needle` — bytes that moved."""
    var f = open(String(LOG_FILE), "r")
    var text = String(f.read())
    f.close()
    var n = 0
    for line in text.split("\n"):
        var parts = String(line).split(" ")
        if len(parts) >= 3 and String(parts[1]) == "PUT" and String(parts[2]).startswith("/r2/ds/"):
            if needle.byte_length() == 0 or String(parts[2]).find(needle) >= 0:
                n += 1
    return n


def _post(url: String, body: String) raises:
    var http = HttpClient(5000, 5000)
    var payload = List[UInt8]()
    for i in range(body.byte_length()):
        payload.append(body.as_bytes()[i])
    _ = http.request(String("POST"), url, payload^, String("application/json"), -1)


def part_a() raises -> Int:
    var n = 0
    _check(video_file_index(String("videos/observation.images.wrist/chunk-000/file-004.mp4")) == 4, "file index 4")
    _check(video_file_index(String("videos/cam/chunk-000/file-120.mp4")) == 120, "file index 120")
    _check(video_file_index(String("data/chunk-000/file-000.parquet")) == -1, "parquet is not a video")
    _check(video_file_index(String("videos/cam/chunk-000/file-00x.mp4")) == -1, "non-digit index")
    n += 4
    for ok in [String("meta/info.json"), String("videos/observation.images.overhead/chunk-000/file-049.mp4"), String("meta/rejected_episodes.json")]:
        _check(dataset_path_refusal(ok) == "", "accepted: " + ok)
        n += 1
    var refused = 0
    var bad = [String("../x"), String("meta/../../x"), String(".hidden"), String("a b"), String(""), String("a/b/c/d/e/f/g/h/i")]
    for b in bad:
        if dataset_path_refusal(b).byte_length() > 0:
            refused += 1
    _check(refused == 6, "dataset_path_refusal refused " + String(refused) + " of 6")
    n += 1
    print("  A dataset paths and video file indices: " + String(n) + " checks")
    return n


def part_b(url: String) raises -> Int:
    var n = 0
    var A = String(WORK) + "/box_a/" + NAME
    var B = String(WORK) + "/box_b/" + NAME
    var cat = RemoteCatalog(url, String("gate-key"))
    cat.upsert_project(String(SLUG), String(""))

    var sn = List[String]()
    for i in range(6):
        sn.append(String("j") + String(i) + ".pos")
    var cams = List[String]()
    cams.append(String("observation.images.overhead"))
    cams.append(String("observation.images.wrist"))
    var w = LeRobotWriter(A, 30, sn.copy(), sn.copy(), cams.copy(), H, W, checkpoint=True)
    _episode(w, 0, 4)
    _episode(w, 1, 7)
    _episode(w, 2, 120, end=False)  # ⚠ STILL RECORDING
    # ⚠ NOT VACUOUS: the open episode's mp4 must EXIST, half-written, before
    # the push. ffmpeg only creates the file once its encoder emits output,
    # which a handful of frames does not trigger — the first version of this
    # check held back zero files because there were none to hold.
    var open_mp4 = A + "/videos/observation.images.wrist/chunk-000/file-002.mp4"
    for _ in range(100):
        if exists(open_mp4):
            break
        sleep(0.05)
    _check(exists(open_mp4), "B0: the open episode's mp4 never appeared on disk — the hold-back check would be vacuous")

    # ── plan: the open episode's videos are held back, info.json goes last ─
    var files = dataset_files(A)
    var plan = plan_upload(files, A)
    var held_here = 0
    for h in plan.held:
        if video_file_index(h) == 2:
            held_here += 1
    _check(held_here == len(plan.held) and len(plan.held) >= 1, "B0: expected the open episode's video(s) held back, got " + String(len(plan.held)))
    _check(plan.send[len(plan.send) - 1] == "meta/info.json", "B0: meta/info.json must be sent last")
    print("  B0 plan while recording: " + String(len(plan.held)) + " open video(s) held, info.json last of " + String(len(plan.send)))
    n += 2

    # ── 1. push while recording ────────────────────────────────────────
    var cache = HashCache()
    var r1 = push_dataset(cat, String(SLUG), String(NAME), A, cache, verbose=False)
    _check(r1.uploaded == len(plan.send), "B1: uploaded " + String(r1.uploaded) + " of " + String(len(plan.send)))
    _check(_object_puts(String("file-002.mp4")) == 0, "B1: a half-written video was uploaded")
    print("  B1 push during episode 3: " + String(r1.uploaded) + " files up, the open video not")
    n += 2

    # ── 2. a second push moves no bytes ────────────────────────────────
    var before = _object_puts(String(""))
    var r2 = push_dataset(cat, String(SLUG), String(NAME), A, cache, verbose=False)
    _check(r2.uploaded == 0 and _object_puts(String("")) == before, "B2: a no-change push moved bytes")
    print("  B2 re-push: 0 uploaded, " + String(r2.unchanged) + " unchanged, no object PUTs")
    n += 1

    # ── 3. the episode ends: only what it changed is sent ──────────────
    w.end_episode()
    var before_old_videos = _object_puts(String("file-000.mp4")) + _object_puts(String("file-001.mp4"))
    var r3 = push_dataset(cat, String(SLUG), String(NAME), A, cache, verbose=False)
    _check(_object_puts(String("file-002.mp4")) == 2, "B3: episode 3's two videos should now be sent")
    _check(
        _object_puts(String("file-000.mp4")) + _object_puts(String("file-001.mp4")) == before_old_videos,
        "B3: finished episodes' videos were re-uploaded",
    )
    _check(r3.n_episodes == 3 and r3.held == 0, "B3: expected 3 episodes and nothing held")
    print("  B3 episode 3 ends: " + String(r3.uploaded) + " files re-sent (its videos + rewritten metadata), old videos untouched")
    n += 3
    _check(not recording_finished(A), "B3b: a checkpoint must not mark the recording finished")
    var r3b = push_dataset(cat, String(SLUG), String(NAME), A, cache, verbose=False)
    _check(not watch_is_done(r3b, A), "B3b: --watch must not stop before the recorder finishes")
    w.close(verbose=False)
    _check(recording_finished(A), "B3b: close() must mark the recording finished")
    var r3c = push_dataset(cat, String(SLUG), String(NAME), A, cache, verbose=False)
    # close() rewrote the metadata, so this pass still had work: not done yet.
    _check(r3c.uploaded > 0 and not watch_is_done(r3c, A), "B3b: the pass that sends close()'s rewrite is not the last")
    var r3d = push_dataset(cat, String(SLUG), String(NAME), A, cache, verbose=False)
    _check(watch_is_done(r3d, A), "B3b: finished + nothing left to send must stop --watch")
    print("  B3b --watch: idle mid-session keeps going; stops only after close() and its rewrite are pushed")
    n += 5

    # ── 4. pull on another box: byte for byte, and it imports ──────────
    var p4 = pull_dataset(cat, String(SLUG), String(NAME), B)
    _check(p4.pending == 0 and len(p4.refused) == 0, "B4: unexpected pending/refused files")
    var fa = dataset_files(A)
    var fb = dataset_files(B)
    var same = 0
    for f in fa:
        if exists(B + "/" + f) and sha256_file(A + "/" + f) == sha256_file(B + "/" + f):
            same += 1
    _check(same == len(fa) and len(fb) == len(fa), "B4: pulled copy differs: " + String(same) + " of " + String(len(fa)) + " files identical")
    import_lerobot_v3(A, String(WORK) + "/a.h5", H, W, verbose=False)
    import_lerobot_v3(B, String(WORK) + "/b.h5", H, W, verbose=False)
    var sa = TrajectoryStore(String(WORK) + "/a.h5")
    var sb = TrajectoryStore(String(WORK) + "/b.h5")
    _check(sb.n_episodes() == 3 and sb.n_rows() == 131 and sa.n_rows() == sb.n_rows(), "B4: the pulled dataset imports differently")
    print("  B4 pull: " + String(len(fa)) + " files byte-identical; both copies import as 3 episodes / 131 rows")
    n += 3

    # ── 5. a path the platform should never have sent is refused ───────
    _post(url + "/__inject_dataset_file", String("{\"slug\":\"") + SLUG + "\",\"name\":\"" + NAME + "\",\"path\":\"../escape.txt\",\"sha256\":\"" + "0" * 64 + "\"}")
    var p5 = pull_dataset(cat, String(SLUG), String(NAME), B)
    _check(len(p5.refused) == 1 and not exists(String(WORK) + "/box_b/escape.txt"), "B5: a traversal path was written")
    print("  B5 server-sent ../escape.txt: refused, not written")
    n += 1

    # ── 6. a file mid-upload is not pulled ─────────────────────────────
    _post(url + "/projects/" + SLUG + "/datasets/" + NAME + "/files", String("{\"path\":\"meta/late.json\",\"sha256\":\"") + "1" * 64 + "\",\"size_bytes\":5}")
    var p6 = pull_dataset(cat, String(SLUG), String(NAME), B)
    _check(p6.pending == 1 and not exists(B + "/meta/late.json"), "B6: a pending file was pulled")
    print("  B6 a file still uploading: reported pending, not pulled")
    n += 1
    return n


def main() raises:
    print("[dataset-sync] gate")
    _ = run_capture(String("rm -rf ") + WORK)
    makedirs(String(WORK), exist_ok=True)
    var a = part_a()
    if not http_shim_available():
        raise Error("the http shim is not built — pixi run build-http")
    var url = _start_server()
    var b = 0
    try:
        b = part_b(url)
    finally:
        try:
            var h = HttpClient(2000, 2000)
            _ = h.request(String("GET"), url + "/__shutdown")
        except:
            pass
    print("  " + String(a + b) + " checks, 0 failures")
    print("[PASS] dataset sync (" + String(a + b) + " checks)")
