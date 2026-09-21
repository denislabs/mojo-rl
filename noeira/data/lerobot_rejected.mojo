# +--------------------------------------------------------------------------+ #
# | Discarded episodes — a demonstration the operator rejected
# +--------------------------------------------------------------------------+ #
"""`meta/rejected_episodes.json`: the episodes a recording keeps but no one
should train on.

    {"rejected_episodes": [1, 4], "note": "..."}

⚠⚠ THE EPISODE STAYS IN THE DATASET; IT IS SKIPPED ON IMPORT. `LeRobotWriter`
is append-only: by the time an operator judges a demonstration, its frames are
already inside an ffmpeg pipe and its rows are counted. Rewriting the videos
and the parquet to cut one out would be a second writer. So the rejection is a
list beside the data, and `import_lerobot_v3` honours it — frames AND
normalisation statistics — which is the only consumer that feeds training.

⚠ A FILE IN `meta/`, SO IT TRAVELS. `push_lerobot_dataset` uploads every file
and `hf_download_dataset` fetches every file, so a training box that pulls the
repo gets the list with the data. LeRobot's own loader ignores the file: a
dataset opened there still contains the rejected episodes, and says so here.

⚠ WRITTEN AT THE MOMENT OF REJECTION, not when the recorder finishes. A crash
after a rejection must not bring the bad demonstration back.
"""

from std.os import makedirs
from std.os.path import exists

from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.io.json import J_ARRAY, JsonWriter, parse_json


comptime REJECTED_FILE = "meta/rejected_episodes.json"


def load_rejected_episodes(root: String) raises -> List[Int]:
    """The rejected episode indices, sorted and unique. Empty when absent."""
    var out = List[Int]()
    var path = root + "/" + REJECTED_FILE
    if not exists(path):
        return out^
    var doc = parse_json(read_file_bytes(path))
    var arr = doc.field(doc.root(), String("rejected_episodes"))
    if arr < 0 or doc.kind_of(arr) != J_ARRAY:
        raise Error(path + ": expected {\"rejected_episodes\": [ ... ]}")
    for i in range(doc.size(arr)):
        var e = doc.integer(doc.at(arr, i))
        if e < 0:
            raise Error(path + ": negative episode index " + String(e))
        _insert_sorted_unique(out, e)
    return out^


def save_rejected_episodes(root: String, ref episodes: List[Int]) raises:
    """Write the list atomically, sorted and unique."""
    var clean = List[Int]()
    for e in episodes:
        _insert_sorted_unique(clean, e)
    var w = JsonWriter()
    w.begin_object()
    w.key(String("rejected_episodes"))
    w.begin_array()
    for e in clean:
        w.integer(e)
    w.end_array()
    w.member(
        String("note"),
        String(
            "Episodes the operator rejected while recording. They remain in"
            " the videos and parquet; noeira's importer skips them."
        ),
    )
    w.end_object()
    makedirs(root + "/meta", exist_ok=True)
    write_text_atomic(root + "/" + REJECTED_FILE, w.done())


def reject_episode(root: String, episode: Int) raises -> List[Int]:
    """Add one episode to the list on disk. Returns the full list."""
    var cur = load_rejected_episodes(root)
    _insert_sorted_unique(cur, episode)
    save_rejected_episodes(root, cur)
    return cur^


def refuse_existing_dataset(root: String) raises:
    """Raise if `root` already holds a recording.

    ⚠⚠ `LeRobotWriter` WRITES INTO WHATEVER IS THERE. A mistyped `--out`
    pointing at yesterday's 50 demonstrations would overwrite them, and a
    leftover `rejected_episodes.json` would silently reject episodes of the
    NEW recording by index. A recorder calls this before it opens a writer.
    """
    for rel in [String("meta/info.json"), String(REJECTED_FILE)]:
        if exists(root + "/" + rel):
            raise Error(
                root + " already holds a recording (" + rel + " exists)."
                " Add --resume to continue it, or choose another dataset name;"
                " this refuses rather than overwrite it."
            )


def is_rejected(ref rejected: List[Int], episode: Int) -> Bool:
    for e in rejected:
        if e == episode:
            return True
    return False


def kept_rows(
    ref x: List[Float32],
    dim: Int,
    ref starts: List[Int],
    ref lengths: List[Int],
    ref rejected: List[Int],
) -> List[Float32]:
    """The rows of `x` (flat, `dim` wide) belonging to episodes NOT rejected.

    ⚠ FOR THE NORMALISATION STATISTICS. A rejected demonstration is often the
    one that went somewhere strange — an arm flung to a joint limit, a gripper
    left open — which is exactly the kind of row that moves a mean and a std.
    Skipping its frames but keeping it in the statistics would still train on
    it, one division at a time.
    """
    var out = List[Float32]()
    for e in range(len(starts)):
        if is_rejected(rejected, e):
            continue
        for r in range(starts[e], starts[e] + lengths[e]):
            for d in range(dim):
                out.append(x[r * dim + d])
    return out^


def _insert_sorted_unique(mut xs: List[Int], v: Int):
    for i in range(len(xs)):
        if xs[i] == v:
            return
        if xs[i] > v:
            xs.insert(i, v)
            return
    xs.append(v)
