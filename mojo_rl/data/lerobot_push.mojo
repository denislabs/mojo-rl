# +--------------------------------------------------------------------------+ #
# | Pushing a LeRobot v3 dataset to the Hub, or to our own platform
# +--------------------------------------------------------------------------+ #
"""Walk a dataset directory and upload it.

    var url = push_lerobot_dataset(
        String("/tmp/my-recording"), String("DenisLabs/my-recording")
    )

The last link: `LeRobotWriter` produces the directory, `io/hf_push.mojo`
speaks the Hub's write API, and this is the twenty lines between them.

## Why this is a walk and not a manifest

The writer knows exactly which files it produced, so handing them straight to
the pusher looks tidier. It would also mean the two could disagree — a video
that rolled to `file-003.mp4` and a pusher that thought there were three
files, and the difference is a dataset that is missing an episode's frames and
imports perfectly right up until it doesn't. Walking what is actually on disk
cannot drift from what is actually on disk.

⚠ `.part` FILES ARE SKIPPED, AND SO IS ANYTHING HIDDEN. An interrupted
download leaves `x.parquet.part` next to `x.parquet`; uploading it would put a
half file in the repo under a name nothing reads and every later `hf_tree`
listing would carry it.

## Ordering

`HubPush.push` does preupload -> LFS upload -> commit, so every LFS object is
in the store before the commit that references it. A commit naming an object
nobody uploaded is accepted by the Hub and then 404s on download, which is why
the order is not an implementation detail.

⚠ **RE-PUSHING IS CHEAP AND THAT IS NOT AN ACCIDENT.** The LFS batch answers
with no upload action for content the Hub already has, so a dataset re-pushed
after one changed episode moves one file. Measured — see
`docs/SO101_RECORDING_PLAN.md` phase 0. Do not add a "skip unchanged files"
layer here; the protocol already has one, keyed on content rather than on
mtime.
"""

from std.os import listdir
from std.os.path import exists, isdir

from mojo_rl.io.hf_push import HubPush, HubUpload
from mojo_rl.io.fileio import write_file_atomic
from mojo_rl.io.json import JsonDoc, load_json


def _walk(root: String, rel: String, mut out: List[String]) raises:
    """Collect every file under `root/rel`, as paths relative to `root`."""
    var here = root + "/" + rel if rel != "" else root.copy()
    var names = listdir(here)
    _sort(names)
    for i in range(len(names)):
        var name = names[i]
        if name.startswith("."):
            continue
        if name.endswith(".part"):
            continue
        var child = name if rel == "" else rel + "/" + name
        if isdir(here + "/" + name):
            _walk(root, child, out)
        else:
            out.append(child)


def _sort(mut xs: List[String]):
    """Insertion sort, so a push lists files in a stable order.

    Not required by the Hub. It matters for the PROGRESS OUTPUT and for
    diffing two runs' logs — `listdir` order is filesystem order, which is
    stable on one box and not across two.
    """
    for i in range(1, len(xs)):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v


def dataset_files(root: String) raises -> List[String]:
    """Every file of a dataset directory, relative to it, sorted."""
    if not isdir(root):
        raise Error("lerobot_push: no dataset directory at " + root)
    var out = List[String]()
    _walk(root, String(""), out)
    if len(out) == 0:
        raise Error("lerobot_push: " + root + " holds no files")
    # A dataset without these is not a dataset, and the failure to catch it
    # here is a repo that looks fine and cannot be imported.
    for required in [
        String("meta/info.json"), String("meta/tasks.parquet"),
    ]:
        var found = False
        for i in range(len(out)):
            if out[i] == required:
                found = True
                break
        if not found:
            raise Error(
                "lerobot_push: " + root + " has no " + required
                + " — refusing to push something that is not a v3 dataset"
            )
    return out^


def write_dataset_card(root: String, repo: String) raises:
    """A minimal `README.md`, so the Hub page is not blank.

    ⚠ FACTS ONLY, AND ONLY ONES READ BACK OFF DISK. A generated card is a
    place where invented numbers become permanent, so everything here comes
    from `meta/info.json`. It claims nothing about what the data is good for.
    """
    var doc = load_json(root + "/meta/info.json")
    var r = doc.root()
    var fps = doc.integer(doc.field(r, String("fps")))
    var eps = doc.integer(doc.field(r, String("total_episodes")))
    var frames = doc.integer(doc.field(r, String("total_frames")))
    var robot = String("")
    var rn = doc.field(r, String("robot_type"))
    if rn >= 0:
        robot = doc.string(rn)

    var text = String("---\n")
    text += "task_categories:\n- robotics\n"
    text += "tags:\n- LeRobot\n- mojo-rl\n"
    if robot != "":
        text += "- " + robot + "\n"
    text += "---\n\n"
    text += "# " + repo + "\n\n"
    text += "A [LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset.\n\n"
    text += "| | |\n|---|---|\n"
    if robot != "":
        text += "| robot | `" + robot + "` |\n"
    text += "| episodes | " + String(eps) + " |\n"
    text += "| frames | " + String(frames) + " |\n"
    text += "| fps | " + String(fps) + " |\n\n"
    text += (
        "Recorded and written with [mojo-rl](https://github.com/DenisLaboureyras/mojo-rl)"
        " — no Python in the capture or the upload path.\n"
    )

    var bytes = List[UInt8]()
    for i in range(text.byte_length()):
        bytes.append(text.as_bytes()[i])
    write_file_atomic(root + "/README.md", bytes)


def push_lerobot_dataset(
    root: String,
    var repo: String,
    var message: String = String(""),
    private: Bool = True,
    var token: String = String(""),
    var revision: String = String("main"),
    card: Bool = True,
    verbose: Bool = True,
) raises -> String:
    """Create the repo if needed, upload every file, commit. Returns the URL.

    ⚠ `private=True` BY DEFAULT. A robot recording is of someone's room, and
    the Hub's own default is public. Making the safe direction the default
    means publishing is a decision rather than an omission — and note the Hub
    IGNORES this on a repo that already exists, so it does not un-publish
    anything.
    """
    if card and not exists(root + "/README.md"):
        write_dataset_card(root, repo)

    var rels = dataset_files(root)
    if message == "":
        message = String("Add ") + String(len(rels)) + " files"

    var p = HubPush(
        repo.copy(), revision=revision^, token=token^, verbose=verbose
    )
    _ = p.create_repo(private=private)

    var files = List[HubUpload]()
    var total = 0
    for i in range(len(rels)):
        var u = HubUpload(rels[i], root + "/" + rels[i])
        total += u.size
        files.append(u^)
    if verbose:
        print(
            "  [push] " + String(len(files)) + " files, "
            + String(total // 1000000) + " MB"
        )

    var url = p.push(files, message^)
    if verbose:
        print("  [push] " + url)
    return url^
