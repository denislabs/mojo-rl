# +--------------------------------------------------------------------------+ #
# | Mirroring a recording on the platform — push while recording, pull to train
# +--------------------------------------------------------------------------+ #
"""`dataset-push` / `dataset-pull`: a LeRobot v3 recording, file by file.

    pixi run dataset-push -- --project so101-tower --dataset cube-in-bowl --watch
    pixi run dataset-pull -- --project so101-tower --dataset cube-in-bowl

⚠⚠ A RECORDING IN PROGRESS IS PUSHABLE, AND THAT SHAPES EVERYTHING HERE. The
recorder checkpoints after every episode (`LeRobotWriter(checkpoint=True)`):
each finished episode's mp4 is closed and every metadata file rewritten. Two
consequences:

1. THE VIDEO BEING RECORDED IS SKIPPED. In a checkpointed dataset, episode `e`
   lives in `file-{e:03d}.mp4`, so any file index `>= total_episodes` is the
   one ffmpeg is still writing — unreadable, and referenced by nothing.
2. METADATA IS PUSHED LAST, AND THE PASS REPEATS IF AN EPISODE ENDED DURING
   IT. The server must never hold an `info.json` naming an episode whose video
   it does not have yet, and an episode can end mid-push. `total_episodes` is
   read before and after; if it moved, the pass runs again.

⚠ THE SERVER DECIDES WHAT IS RE-SENT (decision 18): every file is offered with
its digest and the Worker answers `unchanged` for a ready file with the same
sha and size. A `--watch` cycle therefore costs one small request per file;
this module only avoids RE-HASHING the large, immutable videos (`HashCache`).
"""

from std.collections import Dict
from std.os import makedirs
from std.os.path import exists

from mojo_rl.data.lerobot_push import dataset_files
from mojo_rl.data.recording_files import PushPlan, plan_upload, total_episodes
from mojo_rl.data.remote import RemoteCatalog
from mojo_rl.io.fetch import fetch_to_cache
from mojo_rl.io.fileio import file_size
from mojo_rl.io.json import load_json
from mojo_rl.io.sha256 import sha256_file


comptime HASH_CACHE_MIN_BYTES = 4_000_000
"""Files at least this big are hashed once per (path, size). Below it — every
metadata file — they are hashed on every pass: a rewritten parquet can keep its
size exactly."""


def dataset_path_refusal(rel: String) -> String:
    """Why a dataset path must not be written under a directory, or ""."""
    if rel.byte_length() == 0 or rel.byte_length() > 512:
        return String("path length outside 1..512 bytes")
    var parts = rel.split("/")
    if len(parts) > 8:
        return String("more than 8 directories deep")
    for i in range(len(parts)):
        var seg = String(parts[i])
        var n = seg.byte_length()
        if n == 0:
            return String("empty segment")
        var b = seg.as_bytes()
        for k in range(n):
            var c = Int(b[k])
            var alnum = (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (c >= 97 and c <= 122)
            if k == 0 and not alnum:
                return String("segment '" + seg + "' must start with a letter or digit")
            if not (alnum or c == 46 or c == 95 or c == 45):
                return String("segment '" + seg + "' has a character outside [A-Za-z0-9._-]")
    return String("")


struct HashCache(Movable):
    """sha256 of large files, keyed by path and size, across watch cycles."""

    var sizes: Dict[String, Int]
    var shas: Dict[String, String]

    def __init__(out self):
        self.sizes = Dict[String, Int]()
        self.shas = Dict[String, String]()

    def __init__(out self, *, deinit move: Self):
        self.sizes = move.sizes^
        self.shas = move.shas^

    def sha(mut self, path: String, size: Int) raises -> String:
        if size < HASH_CACHE_MIN_BYTES:
            return sha256_file(path)
        if path in self.sizes and self.sizes.get(path, -1) == size:
            return self.shas.get(path, String(""))
        var h = sha256_file(path)
        self.sizes[path] = size
        self.shas[path] = h
        return h^


struct PushReport(Movable):
    var uploaded: Int
    var unchanged: Int
    var held: Int
    var bytes: Int
    var passes: Int
    var n_episodes: Int

    def __init__(out self):
        self.uploaded = 0
        self.unchanged = 0
        self.held = 0
        self.bytes = 0
        self.passes = 0
        self.n_episodes = -1

    def __init__(out self, *, deinit move: Self):
        self.uploaded = move.uploaded
        self.unchanged = move.unchanged
        self.held = move.held
        self.bytes = move.bytes
        self.passes = move.passes
        self.n_episodes = move.n_episodes


def push_dataset(
    mut cat: RemoteCatalog,
    slug: String,
    name: String,
    root: String,
    mut cache: HashCache,
    verbose: Bool = True,
) raises -> PushReport:
    """One sync of `root` to the platform. Repeats while episodes end under it."""
    var rep = PushReport()
    if not exists(root + "/meta/info.json"):
        raise Error(
            root + " holds no finished episode yet (no meta/info.json) — nothing"
            " to push"
        )
    for _ in range(4):
        var files = dataset_files(root)
        var plan = plan_upload(files, root)
        rep.passes += 1
        rep.held = len(plan.held)
        cat.upsert_dataset(slug, name, plan.n_episodes, _total_frames(root), _fps(root))
        for rel in plan.send:
            var local = root + "/" + rel
            if not exists(local):
                continue  # rotated away between the listing and now
            var size = file_size(local)
            var sha = cache.sha(local, size)
            var got = cat.push_dataset_file(slug, name, rel, local, sha, size)
            if got == "uploaded":
                rep.uploaded += 1
                rep.bytes += size
                if verbose:
                    print("  uploaded  " + rel + "  (" + String(size // 1000) + " KB)")
            else:
                rep.unchanged += 1
        rep.n_episodes = plan.n_episodes
        # ⚠ An episode that ended during the pass rewrote the metadata and
        # closed a video this pass held back. Go again, so what the server holds
        # is one checkpoint, not half of two.
        if total_episodes(root) == plan.n_episodes:
            break
    return rep^


def _total_frames(root: String) raises -> Int:
    var doc = load_json(root + "/meta/info.json")
    var n = doc.field(doc.root(), String("total_frames"))
    return doc.integer(n) if n >= 0 else 0


def _fps(root: String) raises -> Int:
    var doc = load_json(root + "/meta/info.json")
    var n = doc.field(doc.root(), String("fps"))
    return doc.integer(n) if n >= 0 else 0


struct PullReport(Movable):
    var downloaded: Int
    var present: Int
    var pending: Int
    var refused: List[String]

    def __init__(out self):
        self.downloaded = 0
        self.present = 0
        self.pending = 0
        self.refused = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.downloaded = move.downloaded
        self.present = move.present
        self.pending = move.pending
        self.refused = move.refused^


def pull_dataset(
    mut cat: RemoteCatalog, slug: String, name: String, dest: String
) raises -> PullReport:
    """Bring every READY file of a dataset into `dest`, verified.

    ⚠ `meta/info.json` IS FETCHED LAST, mirroring the push: an interrupted pull
    must not leave an `info.json` newer than the videos and data beside it.
    """
    var doc = cat.dataset_files(slug, name)
    var rep = PullReport()
    var files = -1
    try:
        files = doc.field(doc.root(), String("files"))
    except:
        files = -1  # an empty doc: the platform has no such dataset
    if files < 0:
        raise Error(
            "dataset '" + name + "' of project '" + slug + "' is not on the"
            " platform — push it: pixi run dataset-push -- --project " + slug
            + " --dataset " + name
        )
    makedirs(dest, exist_ok=True)
    for pass_info in [False, True]:
        for i in range(doc.size(files)):
            var row = doc.at(files, i)
            var rel = doc.string(doc.field(row, String("path")))
            var is_info = rel == "meta/info.json"
            if is_info != pass_info:
                continue
            if doc.string(doc.field(row, String("status"))) != "ready":
                rep.pending += 1
                continue
            var why = dataset_path_refusal(rel)
            if why.byte_length() > 0:
                rep.refused.append(rel + " — " + why)
                continue
            var out = dest + "/" + rel
            var sha = doc.string(doc.field(row, String("sha256")))
            var size = doc.integer(doc.field(row, String("sizeBytes")))
            if exists(out) and file_size(out) == size and sha256_file(out) == sha:
                rep.present += 1
                continue
            _ = fetch_to_cache(
                doc.string(doc.field(row, String("download_url"))), out, sha, size, rel
            )
            rep.downloaded += 1
    return rep^
