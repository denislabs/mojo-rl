# +--------------------------------------------------------------------------+ #
# | Private projects — which files are the definition, and who wins a sync
# +--------------------------------------------------------------------------+ #
"""The pure half of `project-push` / `project-pull` for a project's DEFINITION.

    projects/so101-tower/
        project.kv                    synced
        calibration/follower.json     synced
        policies/act.kv               synced
        policies/act.ckpt             NOT synced — weights, see `sync_refusal`
        runs/...                      NOT synced — run data goes as artifacts
        datasets/...                  NOT synced — recordings go to the Hub
        .sync.kv                      NOT synced — this box's sync record

⚠⚠ PROJECTS ARE NOT IN THE CORE GIT REPO. `projects/` is ignored: a project is
private, and creating one must not mean touching mojo_rl. So the definition
needs its own way onto a second box — a rented GPU starting a training run is
the case that forces it — and that is the monitor, file by file.

## ⚠⚠ The one failure a sync must not have

Box B pushing over an edit box A made after B last pulled, and nobody finding
out. So every file carries THREE digests:

| name   | where it comes from                                  |
|--------|------------------------------------------------------|
| local  | the file on this box, hashed now                     |
| remote | the platform's current row                           |
| base   | what this box last synced, from `.sync.kv`           |

and `plan_push` / `plan_pull` decide from all three. Two digests cannot tell
"I edited it" from "they edited it" — both look like `local != remote` — and
that is exactly the distinction between a safe upload and an erased edit. The
server enforces the same rule on its side (`base_sha256`, a compare-and-swap),
so a race between the plan and the write loses there instead.

⚠ NO MERGING, AND NO DELETION ACROSS BOXES. A conflict is reported and left for
the person (`--force` picks a side). A file deleted locally is not deleted on
the platform, and a pull does not resurrect it; removing a definition file
from the platform is a dashboard act. Both are deliberately the conservative
choice for a first version whose users are one person on several machines.

This module does no I/O over the network; `data/project_sync.mojo` executes
the plans against a `RemoteCatalog`.
"""

from std.collections import Dict

from mojo_rl.core.kv import KvWriter, kv_lines
from mojo_rl.io.fileio import file_size, write_text_atomic
from mojo_rl.io.proc import quote_arg, run_capture
from mojo_rl.io.sha256 import sha256_file


comptime SYNC_FILE = ".sync.kv"

# ── actions ────────────────────────────────────────────────────────────────
comptime ACT_SAME = "same"
"""Local and remote already agree."""
comptime ACT_UPLOAD = "upload"
comptime ACT_DOWNLOAD = "download"
comptime ACT_CONFLICT = "conflict"
"""Both sides changed since the last sync. Nothing moves without `--force`."""
comptime ACT_BEHIND = "behind"
"""Push: the platform is newer and this box did not edit it — pull first."""
comptime ACT_AHEAD = "ahead"
"""Pull: this box edited it and the platform did not — push it."""
comptime ACT_REMOTE_ONLY = "remote-only"
"""Push: on the platform, absent here and never synced here — pull brings it."""
comptime ACT_LOCAL_ONLY = "local-only"
"""Pull: here and not on the platform — push sends it."""
comptime ACT_DELETED_LOCALLY = "deleted-locally"
"""Synced once, since removed here. Not propagated, not resurrected."""


comptime _BULK_SUFFIXES = (
    ".ckpt .h5 .hdf5 .mp4 .safetensors .pt .pth .bin .npz .npy .part .tmp .zip"
    " .tar .zst .gz"
)


def _segment_ok(s: String) -> Bool:
    """`[A-Za-z0-9][A-Za-z0-9._-]*` — the monitor's `relPathParts` rule."""
    var n = s.byte_length()
    if n == 0:
        return False
    var b = s.as_bytes()
    for i in range(n):
        var c = Int(b[i])
        var alnum = (
            (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (c >= 97 and c <= 122)
        )
        if i == 0 and not alnum:
            return False
        if not (alnum or c == 46 or c == 95 or c == 45):
            return False
    return True


def sync_refusal(rel: String) -> String:
    """Why `rel` is not part of the synced definition, or "" if it is.

    ⚠⚠ A REASON RATHER THAN A BOOL, because every skipped file is PRINTED. A
    calibration file named `follower (copy).json` that silently never reaches
    the training box is the bug; "skipped: name the platform refuses" is the
    fix.

    ⚠ THIS RUNS ON PULL TOO, against the paths the SERVER sends. `project-pull`
    writes each one under a directory on this box, so a path the server should
    never have accepted — `../.env`, `runs/x` — is refused here as well rather
    than trusted because it came from our own Worker.
    """
    if rel.byte_length() == 0 or rel.byte_length() > 512:
        return String("path length outside 1..512 bytes")
    var parts = rel.split("/")
    if len(parts) > 8:
        return String("more than 8 directories deep")
    for i in range(len(parts)):
        var seg = String(parts[i])
        if seg.startswith("."):
            return String("hidden file")
        if not _segment_ok(seg):
            return String(
                "name the platform refuses (use [A-Za-z0-9._-], starting with"
                " a letter or digit)"
            )
    var first = String(parts[0])
    if len(parts) > 1 and (first == "runs" or first == "artifacts"):
        return String("run data — pushed as run artifacts, not as definition")
    if len(parts) > 1 and first == "datasets":
        # ⚠ A RECORDING IS NOT DEFINITION, even though most of its files are
        # small `.json` / `.parquet` that pass every other rule here. Its home
        # is the Hub (`hf-push-dataset`); pushed here it would be half a
        # dataset — the videos refused as bulk, the metadata accepted.
        return String("recorded dataset — pushed with hf-push-dataset, not as definition")
    var dot = rel.rfind(".")
    var slash = rel.rfind("/")
    if dot > slash and dot >= 0:
        var ext = String(rel[byte=dot:])
        for sfx in String(_BULK_SUFFIXES).split(" "):
            if ext == String(sfx):
                return String(
                    "weights or bulk data (" + ext + ") — those go through"
                    " run artifacts and policies"
                )
    return String("")


def sort_strings(mut xs: List[String]):
    """Insertion sort: plans and reports list files in one stable order."""
    for i in range(1, len(xs)):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v


# =============================================================================
# The local side
# =============================================================================


struct LocalScan(Movable):
    """This box's definition files, hashed, and what was left out and why."""

    var shas: Dict[String, String]
    var sizes: Dict[String, Int]
    var skipped_paths: List[String]
    var skipped_reasons: List[String]

    def __init__(out self):
        self.shas = Dict[String, String]()
        self.sizes = Dict[String, Int]()
        self.skipped_paths = List[String]()
        self.skipped_reasons = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.shas = move.shas^
        self.sizes = move.sizes^
        self.skipped_paths = move.skipped_paths^
        self.skipped_reasons = move.skipped_reasons^

    def skip(mut self, path: String, reason: String):
        self.skipped_paths.append(path)
        self.skipped_reasons.append(reason)


def scan_definition(project_dir: String, max_bytes: Int = 0) raises -> LocalScan:
    """Hash every definition file under `project_dir`.

    `max_bytes` is the PLATFORM'S cap, read from the file list the server
    returns — not a copy of it here — and 0 means no cap (a pull's scan).

    ⚠ `.sync.kv` IS NOT REPORTED AS SKIPPED. It is this module's own record,
    and listing it on every push would teach a person to skim the skip list,
    which is the list that has to be read.
    """
    var out = LocalScan()
    var prefix = project_dir + "/"
    var txt = run_capture(
        String("find ") + quote_arg(project_dir) + " -type f 2>/dev/null | LC_ALL=C sort",
        1 << 24,
    )
    for line in txt.split("\n"):
        var full = String(line)
        if full.byte_length() == 0 or not full.startswith(prefix):
            continue
        var rel = String(full[byte = prefix.byte_length() :])
        if rel == SYNC_FILE:
            continue
        # ⚠ `runs/`, `artifacts/` AND `datasets/` ARE SKIPPED AS A WHOLE.
        # A project with 40 runs would otherwise print a thousand "run data"
        # lines above the three that matter.
        if (
            rel.startswith("runs/")
            or rel.startswith("artifacts/")
            or rel.startswith("datasets/")
        ):
            continue
        var why = sync_refusal(rel)
        if why.byte_length() > 0:
            out.skip(rel, why)
            continue
        var size = file_size(full)
        if max_bytes > 0 and size > max_bytes:
            out.skip(
                rel,
                String(size) + " bytes, over the platform's " + String(max_bytes)
                + "-byte cap for definition files",
            )
            continue
        out.shas[rel] = sha256_file(full)
        out.sizes[rel] = size
    return out^


def load_base(project_dir: String) raises -> Dict[String, String]:
    """`.sync.kv`: the digest of each file as this box last synced it.

    ⚠ ABSENT IS EMPTY, NOT AN ERROR. A box that has never synced has no base,
    and every file then plans as "new on both sides" — which uploads a file the
    platform lacks, downloads one this box lacks, and calls two DIFFERENT
    copies a conflict. That is the correct first-sync behaviour, not a
    degraded one.
    """
    var out = Dict[String, String]()
    var path = project_dir + "/" + SYNC_FILE
    var txt: String
    try:
        with open(path, "r") as fh:
            txt = fh.read()
    except:
        return out^
    for line in kv_lines(txt, path):
        if line.key != "file":
            continue
        var sp = line.value.find(" ")
        if sp != 64:
            raise Error(
                path + " line " + String(line.lineno) + ": expected"
                " 'file=<sha256> <path>'"
            )
        out[String(line.value[byte = 65 :])] = String(line.value[byte=0:64])
    return out^


def save_base(project_dir: String, ref base: Dict[String, String]) raises:
    """Write `.sync.kv`, sorted by path so it diffs."""
    var keys = List[String]()
    for k in base.keys():
        keys.append(k)
    sort_strings(keys)
    var w = KvWriter(String(SYNC_FILE))
    w.comment(
        String(
            "What this box last synced with the platform. Written by"
            " project-push/pull; not synced itself."
        )
    )
    for k in keys:
        w.add(String("file"), base[k] + " " + k)
    write_text_atomic(project_dir + "/" + SYNC_FILE, w^.done())


# =============================================================================
# The decision
# =============================================================================


@fieldwise_init
struct SyncAction(Copyable, Movable):
    var path: String
    var action: String
    var local_sha: String
    """"" when the file is not on this box."""
    var remote_sha: String
    """"" when the file is not on the platform."""
    var base_sha: String
    """"" when this box never synced the file."""


def _get(ref d: Dict[String, String], k: String) -> String:
    var v = d.get(k)
    if v:
        return v.value()
    return String("")


def _union_sorted(
    ref a: Dict[String, String],
    ref b: Dict[String, String],
    ref c: Dict[String, String],
) -> List[String]:
    var seen = Dict[String, Bool]()
    var out = List[String]()
    for k in a.keys():
        if k not in seen:
            seen[k] = True
            out.append(k)
    for k in b.keys():
        if k not in seen:
            seen[k] = True
            out.append(k)
    for k in c.keys():
        if k not in seen:
            seen[k] = True
            out.append(k)
    sort_strings(out)
    return out^


def push_action(l: String, r: String, b: String, force: Bool) -> String:
    """What `project-push` does with one file. See the module header.

    ⚠⚠ THE WHOLE RULE IS "UPLOAD ONLY OVER WHAT YOU LAST SAW". `r == b` means
    the platform has not moved since this box synced, so replacing it erases
    nothing; any other `r` belongs to someone else's edit.
    """
    if l.byte_length() == 0:
        if r.byte_length() == 0:
            return String(ACT_SAME)  # only in the base: gone everywhere
        if b.byte_length() > 0 and b == r:
            return String(ACT_DELETED_LOCALLY)
        return String(ACT_REMOTE_ONLY)
    if l == r:
        return String(ACT_SAME)
    if force:
        return String(ACT_UPLOAD)
    if r == b:
        return String(ACT_UPLOAD)
    if r.byte_length() == 0:
        # ⚠ The platform lost a file this box synced — removed from the
        # dashboard. Sending it again erases nothing, and "behind" here would
        # leave push and pull pointing at each other forever: pull sees a
        # `local-only` file and tells you to push.
        return String(ACT_UPLOAD)
    if l == b:
        return String(ACT_BEHIND)
    return String(ACT_CONFLICT)


def pull_action(l: String, r: String, b: String, force: Bool) -> String:
    """What `project-pull` does with one file — `push_action`'s mirror.

    ⚠ `l == b` IS WHAT MAKES A DOWNLOAD SAFE: this box has not edited the file
    since it last synced, so overwriting it loses nothing.
    """
    if r.byte_length() == 0:
        if l.byte_length() == 0:
            return String(ACT_SAME)
        return String(ACT_LOCAL_ONLY)
    if l == r:
        return String(ACT_SAME)
    if force:
        return String(ACT_DOWNLOAD)
    if l.byte_length() == 0:
        if b.byte_length() > 0 and b == r:
            # ⚠ NOT RESURRECTED. The person removed it after the last sync, and
            # the platform still holding the synced copy is not a reason to
            # put it back.
            return String(ACT_DELETED_LOCALLY)
        return String(ACT_DOWNLOAD)
    if l == b:
        return String(ACT_DOWNLOAD)
    if r == b:
        return String(ACT_AHEAD)
    return String(ACT_CONFLICT)


def plan_push(
    ref local: Dict[String, String],
    ref remote: Dict[String, String],
    ref base: Dict[String, String],
    force: Bool = False,
) -> List[SyncAction]:
    var out = List[SyncAction]()
    for k in _union_sorted(local, remote, base):
        var l = _get(local, k)
        var r = _get(remote, k)
        var b = _get(base, k)
        out.append(SyncAction(k, push_action(l, r, b, force), l, r, b))
    return out^


def plan_pull(
    ref local: Dict[String, String],
    ref remote: Dict[String, String],
    ref base: Dict[String, String],
    force: Bool = False,
) -> List[SyncAction]:
    var out = List[SyncAction]()
    for k in _union_sorted(local, remote, base):
        var l = _get(local, k)
        var r = _get(remote, k)
        var b = _get(base, k)
        out.append(SyncAction(k, pull_action(l, r, b, force), l, r, b))
    return out^
