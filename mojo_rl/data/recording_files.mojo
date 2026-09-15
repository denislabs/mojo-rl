# +--------------------------------------------------------------------------+ #
# | Which files of a recording are safe to send — while it is still recording
# +--------------------------------------------------------------------------+ #
"""`plan_upload`: the one rule for pushing a LeRobot v3 recording anywhere.

A checkpointed recording (`LeRobotWriter(checkpoint=True)`) keeps episode `e`
in `file-{e:03d}.mp4`, so a video whose index is `>= total_episodes` is the one
ffmpeg is still writing: unreadable, and referenced by no metadata. It is held
back. Everything else is sent with `meta/` last and `meta/info.json` very
last, so a destination never holds an index naming a video it lacks.

⚠ NO IMPORTS FROM THE PUSHERS. `lerobot_push` (the Hub) and `dataset_sync`
(the platform) both depend on this module; taking the file list as an argument
is what keeps it below both.
"""

from std.os.path import exists

from mojo_rl.io.json import load_json


comptime WRITER_STATS = "meta/mojo_rl_writer_stats.json"
"""Present only in a checkpointed recording — see `LeRobotWriter`."""


def total_episodes(root: String) raises -> Int:
    """`meta/info.json`'s episode count, or -1 when there is no info yet."""
    if not exists(root + "/meta/info.json"):
        return -1
    var doc = load_json(root + "/meta/info.json")
    var n = doc.field(doc.root(), String("total_episodes"))
    return doc.integer(n) if n >= 0 else -1


def recording_finished(root: String) raises -> Bool:
    """True once the recorder has exited through `finish`.

    A dataset without the writer's side file was only ever written at close, so
    it is finished by construction.
    """
    if not exists(root + "/" + WRITER_STATS):
        return exists(root + "/meta/info.json")
    var doc = load_json(root + "/" + WRITER_STATS)
    var node = doc.field(doc.root(), String("finished"))
    return node >= 0 and doc.boolean(node)


def video_file_index(rel: String) -> Int:
    """`videos/<cam>/chunk-000/file-004.mp4` -> 4; -1 for anything else."""
    if not rel.startswith("videos/") or not rel.endswith(".mp4"):
        return -1
    var cut = rel.rfind("/file-")
    if cut < 0:
        return -1
    var digits = String(rel[byte = cut + 6 : rel.byte_length() - 4])
    if digits.byte_length() == 0:
        return -1
    var v = 0
    for i in range(digits.byte_length()):
        var c = Int(digits.as_bytes()[i])
        if c < 48 or c > 57:
            return -1
        v = v * 10 + (c - 48)
    return v


struct PushPlan(Movable):
    """The files a pass sends, in order, and those it holds back."""

    var send: List[String]
    """Data and videos first, then `meta/`, with `meta/info.json` LAST."""
    var held: List[String]
    """The video(s) still being written."""
    var n_episodes: Int

    def __init__(out self):
        self.send = List[String]()
        self.held = List[String]()
        self.n_episodes = -1

    def __init__(out self, *, deinit move: Self):
        self.send = move.send^
        self.held = move.held^
        self.n_episodes = move.n_episodes


def plan_upload(ref files: List[String], root: String) raises -> PushPlan:
    """Which of a recording's `files` to send now, and in which order.

    ⚠ ONE RULE FOR EVERY DESTINATION: `hf-push-dataset` and `dataset-push`
    both call this. A backup to the Hub made mid-session used to upload the
    half-written mp4 of the episode being recorded.
    """
    var out = PushPlan()
    out.n_episodes = total_episodes(root)
    var checkpointed = exists(root + "/" + WRITER_STATS)
    var meta = List[String]()
    var info = False
    for f in files:
        var rel = String(f)
        if rel.endswith(".tmp"):
            continue
        var idx = video_file_index(rel)
        # ⚠ Only a CHECKPOINTED dataset has one file per episode; in a packed
        # one, file indices say nothing about which episode is open.
        if checkpointed and idx >= 0 and idx >= out.n_episodes:
            out.held.append(rel)
            continue
        if rel == "meta/info.json":
            info = True
        elif rel.startswith("meta/"):
            meta.append(rel)
        else:
            out.send.append(rel)
    for m in meta:
        out.send.append(m)
    if info:
        out.send.append(String("meta/info.json"))
    return out^


