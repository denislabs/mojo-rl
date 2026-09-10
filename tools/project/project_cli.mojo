"""`project` — init, list, show, tag and prune, over `projects/<name>/`.

    pixi run project-init so101 --description "SO-101 tabletop manipulation"
    pixi run project-list
    pixi run project-show so101
    pixi run run-tag <run_id> "meilleur reach à ce jour, testé 8/10"
    pixi run project-prune so101 --older-than 30 [--apply]
    pixi run project-push so101 [<run_id>] [--kind checkpoint]
    pixi run project-pull so101 <run_id> [--kind checkpoint]

⚠⚠ EVERYTHING HERE IS LOCAL AND COMPLETE. `docs/PROJECT_LAYER_PLAN.md` §10: the
OSS half is a project on ONE machine with nothing crippled, and the paid half is
the same project on SEVERAL. A local-only user who cannot tag, find or prune
their runs will not become a paying one — they will keep their own directory
convention and never adopt the layer at all.

⚠ `project-push` / `project-pull` ARE THE ONE PAID-HALF PAIR (§10, decision 6):
everything above works on a single machine with no account. They need
`RL_MONITOR_URL` + `RL_MONITOR_API_KEY` in `.env` — no new credential.

⚠ `project-prune` IS DRY-RUN BY DEFAULT and needs `--apply`. It deletes run
directories; the one thing worse than 219 flat checkpoints is a tool that
removes the one you wanted.
"""

from std.sys import argv
from std.os.path import exists

from mojo_rl.core.project import (
    ProjectSpec,
    check_refs,
    load_project,
    projects_root,
    project_exists,
)
from mojo_rl.core.run import RunRecord, epoch_seconds, iso8601_utc, load_run
from mojo_rl.data.remote import RemoteCatalog
from mojo_rl.io.fetch import fetch_to_cache
from mojo_rl.io.fileio import file_size
from mojo_rl.io.json import JsonDoc
from mojo_rl.io.proc import quote_arg, run_capture
from mojo_rl.io.sha256 import sha256_file


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _has(name: String) -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def _positional(n: Int) -> String:
    """The n-th argument that is not a flag or a flag's value."""
    var av = argv()
    var seen = 0
    var i = 1
    while i < len(av):
        var a = String(av[i])
        if a.startswith("--"):
            i += 2
            continue
        if seen == n:
            return a
        seen += 1
        i += 1
    return String("")


def _ls(path: String) raises -> List[String]:
    """Entries of a directory, sorted. Empty when it does not exist."""
    var out = List[String]()
    if not exists(path):
        return out^
    var txt = run_capture(
        String("ls -1 ") + quote_arg(path) + " 2>/dev/null", 1 << 20
    )
    for line in txt.split("\n"):
        var s = String(line.strip())
        if s.byte_length() > 0:
            out.append(s)
    return out^


# =============================================================================


def cmd_init() raises:
    var name = _positional(1)
    if name.byte_length() == 0:
        raise Error("usage: project-init <name> [--description ...]")
    var root = projects_root()
    if project_exists(name, root):
        raise Error(
            "project '" + name + "' already exists at " + root + "/" + name
            + "/project.kv — refusing to overwrite a definition"
        )
    _ = run_capture(
        String("mkdir -p ") + quote_arg(root + "/" + name + "/runs")
        + " " + quote_arg(root + "/" + name + "/policies") + " 2>&1", 4096
    )
    var p = ProjectSpec(name, root)
    p.created = iso8601_utc(epoch_seconds())
    p.description = _flag(String("--description"), String(""))
    p.write()
    print("created", p.kv_path())
    print()
    print("⚠ The DEFINITION is tracked; the runs are not. `git add` it:")
    print("    git add", p.kv_path())
    print()
    print("Add what the project uses, as pinned references:")
    print("    ref=family:<name>:<path>@<commit>")
    print("    ref=task:<name>:<path>@<commit>")


def cmd_list() raises:
    var root = projects_root()
    var names = _ls(root)
    var shown = 0
    for n in names:
        if not project_exists(n, root):
            continue
        var p = load_project(n, root)
        var runs = _ls(p.runs_dir())
        shown += 1
        print(
            "  " + p.name, "—", String(len(runs)), "runs,",
            String(len(p.refs)), "refs" ,
            ("  " + p.description) if p.description else "",
        )
    if shown == 0:
        print("  no projects under " + root + "/ — `pixi run project-init <name>`")
    else:
        print()
        print(" ", shown, "project(s) under", root + "/")


def cmd_show() raises:
    var name = _positional(1)
    if name.byte_length() == 0:
        raise Error("usage: project-show <name>")
    var p = load_project(name, projects_root())
    print("project", p.name, "  created", p.created)
    if p.description:
        print("  ", p.description)
    print()
    print("  refs:")
    var rep = check_refs(p, verbose=True)
    for i in range(len(p.refs)):
        var r = p.refs[i].copy()
        print("   ", r.kind, r.name, "->", r.path, "@" + r.commit if r.commit else " (unpinned)")
    print(
        "   ", rep.checked, "checked,", rep.missing, "missing,", rep.drifted,
        "drifted",
    )
    print()
    var runs = _ls(p.runs_dir())
    print("  runs:", len(runs))
    var stale = 0
    for rid in runs:
        var kv = p.runs_dir() + "/" + rid + "/run.kv"
        if not exists(kv):
            continue
        var rec = load_run(kv)
        if rec.is_stale():
            stale += 1
        print(
            "   ", rec.status, rid,
            ("  " + rec.outcome) if rec.outcome else "",
            ("  #" + rec.tag) if rec.tag else "",
        )
    if stale > 0:
        # ⚠ A RECORD STILL SAYING `running` IS A CRASHED RUN — the process died
        # before `close()`, so nothing local could have said otherwise.
        print()
        print("   ⚠", stale, "run(s) still say status=running: they crashed or")
        print("     are still going. Nothing infers this on their behalf.")


def cmd_tag() raises:
    var rid = _positional(1)
    var text = _positional(2)
    if rid.byte_length() == 0 or text.byte_length() == 0:
        raise Error('usage: run-tag <run_id> "free text"')
    var root = projects_root()
    # ⚠ SEARCH BOTH ROOTS. A run made before its project existed is under the
    # flat `runs/`, and the id is the same either way.
    var candidates = List[String]()
    candidates.append(String("runs/") + rid + "/run.kv")
    for n in _ls(root):
        candidates.append(root + "/" + n + "/runs/" + rid + "/run.kv")
    for c in candidates:
        if exists(c):
            _tag_file(c, text)
            print("tagged", c)
            return
    raise Error("no run '" + rid + "' under runs/ or " + root + "/*/runs/")


def _tag_file(path: String, text: String) raises:
    """⚠ REWRITE THE RECORD, DO NOT APPEND A LINE. Two `tag=` lines is a record
    that disagrees with itself and a reader that believes the first."""
    from mojo_rl.core.kv import KvWriter, kv_lines
    from mojo_rl.io.fileio import write_text_atomic
    var txt: String
    with open(path, "r") as fh:
        txt = fh.read()
    var ls = kv_lines(txt, path)
    var w = KvWriter(String("run spec"))
    var wrote = False
    for i in range(len(ls)):
        if ls[i].key == "tag":
            w.add(String("tag"), text)
            wrote = True
        else:
            w.add(ls[i].key, ls[i].value)
    if not wrote:
        w.add(String("tag"), text)
    write_text_atomic(path, w^.done())


def cmd_prune() raises:
    """Delete run directories that nothing is holding on to.

    ⚠⚠ DRY RUN BY DEFAULT. It needs `--apply`. `runs/` will rot exactly as
    `checkpoints/` did, but the one thing worse than 219 flat checkpoints is a
    tool that removed the one you wanted.

    What is NEVER deleted, and why each one:
      * `status=running` — it may be a LIVE run. Deleting under a training
        process is how a six-hour job dies at hour five.
      * a non-empty `tag=` — a human wrote it down, which is the only signal
        anywhere that a run mattered.
      * a `:local` artifact — ⚠ that is precisely the "lost the files" pain
        this layer exists to fix, reintroduced by the fix for it (§14 Q2).
        Once P3 uploads them the artifact line says `:uploaded` and the bytes
        survive the delete.
    """
    var name = _positional(1)
    if name.byte_length() == 0:
        raise Error("usage: project-prune <name> [--older-than DAYS] [--apply]")
    var older = atol(_flag(String("--older-than"), String(30)))
    var apply = _has("--apply")
    var p = load_project(name, projects_root())
    var now = epoch_seconds()
    var cutoff = now - older * 86400

    var scanned = 0
    var kept = 0
    var removed = 0
    var bytes_freed = 0
    for rid in _ls(p.runs_dir()):
        var d = p.runs_dir() + "/" + rid
        var kv = d + "/run.kv"
        if not exists(kv):
            continue
        scanned += 1
        var rec = load_run(kv)
        var why = String("")
        if rec.is_stale():
            why = String("status=running (may be live)")
        elif rec.tag.byte_length() > 0:
            why = String("tagged: ") + rec.tag
        else:
            for a in rec.artifacts:
                if a.endswith(":local"):
                    why = String("has a :local artifact")
                    break
        if why.byte_length() == 0 and _mtime(kv) >= cutoff:
            why = String("newer than ") + String(older) + "d"
        if why.byte_length() > 0:
            kept += 1
            print("  keep  ", rid, "—", why)
            continue
        var sz = _du_bytes(d)
        removed += 1
        bytes_freed += sz
        print("  DELETE", rid, "—", String(sz // 1048576), "MB")
        if apply:
            _ = run_capture(String("rm -rf ") + quote_arg(d) + " 2>&1", 4096)
    print()
    print(
        " ", scanned, "scanned,", kept, "kept,", removed,
        "to delete —", bytes_freed // 1048576, "MB",
    )
    if removed > 0 and not apply:
        print("  DRY RUN. Re-run with --apply to actually delete.")


def _mtime(path: String) raises -> Int:
    var s = run_capture(
        String("stat -f %m ") + quote_arg(path) + " 2>/dev/null"
        + " || stat -c %Y " + quote_arg(path) + " 2>/dev/null", 64
    )
    var v = String(s.strip())
    return atol(v) if v.byte_length() > 0 else 0


def _du_bytes(path: String) raises -> Int:
    var s = run_capture(
        String("du -sk ") + quote_arg(path) + " 2>/dev/null | cut -f1", 64
    )
    var v = String(s.strip())
    return (atol(v) * 1024) if v.byte_length() > 0 else 0


# =============================================================================
# push / pull — the two commands that need an account
# =============================================================================


def _artifact_rel(line: String) -> String:
    """`checkpoints/best.ckpt` out of `checkpoints/best.ckpt:sha256:ab..:215:local`.

    ⚠ `split_once` CUTS AT THE FIRST SEPARATOR, which is why the path comes
    first in the record: a path may not contain `:` (the monitor refuses one
    too), so the first colon is always the end of it.
    """
    var cut = line.find(":")
    return String(line) if cut < 0 else String(line[byte=0:cut])


def _run_dirs(project: String, only: String) raises -> List[String]:
    var out = List[String]()
    var base = projects_root() + "/" + project + "/runs"
    for n in _ls(base):
        if only.byte_length() > 0 and n != only:
            continue
        if exists(base + "/" + n + "/run.kv"):
            out.append(base + "/" + n)
    return out^


def _ready_digests(mut cat: RemoteCatalog, run_id: String) -> List[String]:
    """The sha256 of every artifact the monitor already holds for this run.

    ⚠⚠ THE SERVER IS THE AUTHORITY ON WHAT ARRIVED, NOT THE LOCAL `:local`
    MARKER. `ArtifactSink` uploads from a background thread and never rewrites
    `run.kv` — plumbing a completion back across the thread boundary would be
    a second source of truth for a fact the monitor already has. So a push
    asks, and skips what is already there with a matching digest. That also
    makes a push after a CRASH correct for free: whatever completed is
    skipped, whatever did not is re-sent.

    A failure to ask returns empty, which re-sends. Re-sending is wasteful;
    skipping something that is not there is wrong.
    """
    var out = List[String]()
    try:
        var doc = cat.artifacts_of(run_id)
        var root = doc.root()
        var n = doc.size(root)
        for i in range(n):
            var row = doc.at(root, i)
            var st = doc.string(doc.field(row, String("status")))
            if st != "ready":
                continue
            var sha = doc.field(row, String("sha256"))
            if sha >= 0:
                out.append(doc.string(sha))
    except:
        pass
    return out^


def cmd_push() raises:
    """Send this box's artifacts for a project's runs to the monitor.

    ⚠ IT IS A RESUME, NOT A RE-SEND. Anything the monitor already holds with a
    matching sha256 is skipped, so running it twice costs two catalog reads and
    no bytes — the same "cache, not toll booth" rule `RemoteCatalog.pull` uses
    for datasets.
    """
    var project = _positional(1)
    if project.byte_length() == 0:
        raise Error("usage: project-push <project> [<run_id>] [--kind K]")
    var only = _positional(2)
    var kind = _flag(String("--kind"), String("checkpoint"))
    var dirs = _run_dirs(project, only)
    if len(dirs) == 0:
        raise Error("no runs with a run.kv under " + project)

    var cat = RemoteCatalog.from_env()
    var sent = 0
    var skipped = 0
    var missing = 0
    var considered = 0

    for d in dirs:
        var rec = load_run(String(d) + "/run.kv")
        var have = _ready_digests(cat, rec.run_id)
        for line in rec.artifacts:
            considered += 1
            var rel = _artifact_rel(String(line))
            var local = String(d) + "/" + rel
            if not exists(local):
                missing += 1
                print("  missing  " + rec.run_id + "  " + rel)
                continue
            var sha = sha256_file(local)
            var known = False
            for h in have:
                if String(h) == sha:
                    known = True
                    break
            if known:
                skipped += 1
                continue
            print(
                "  push     "
                + rec.run_id
                + "  "
                + rel
                + "  ("
                + String(file_size(local) // 1_000_000)
                + " MB)"
            )
            _ = cat.push_artifact(rec.run_id, rel, local, kind)
            sent += 1

    # ⚠ PRINT WHAT WAS CONSIDERED BESIDE WHAT WAS SENT. "0 pushed" is also what
    # a run with no `artifact=` lines prints, and those are different facts.
    print(
        String(considered)
        + " artifacts considered across "
        + String(len(dirs))
        + " runs: "
        + String(sent)
        + " pushed, "
        + String(skipped)
        + " already there, "
        + String(missing)
        + " missing locally"
    )


def cmd_pull() raises:
    """Bring a run's artifacts back onto this box.

    ⚠ NO BYTES MOVE for a file that is already here with the right digest —
    `fetch_to_cache` compares before it transfers.
    """
    var project = _positional(1)
    var rid = _positional(2)
    if project.byte_length() == 0 or rid.byte_length() == 0:
        raise Error("usage: project-pull <project> <run_id> [--kind K]")
    var kind = _flag(String("--kind"), String(""))
    var dest_root = projects_root() + "/" + project + "/runs/" + rid
    _ = run_capture("mkdir -p " + quote_arg(dest_root))

    var cat = RemoteCatalog.from_env()
    var doc = cat.artifacts_of(rid)
    var root = doc.root()
    var n = doc.size(root)
    var pulled = 0
    var pending = 0
    for i in range(n):
        var row = doc.at(root, i)
        var rel = doc.string(doc.field(row, String("path")))
        var st = doc.string(doc.field(row, String("status")))
        var k = doc.string(doc.field(row, String("kind")))
        if kind.byte_length() > 0 and k != kind:
            continue
        if st != "ready":
            # ⚠ A pending row is an upload that never completed. Its bytes are
            # not there, and offering a download would 404.
            pending += 1
            print("  pending  " + rel + " (never completed; not pulled)")
            continue
        var meta = cat.describe_artifact(doc.string(doc.field(row, String("id"))))
        var dest = dest_root + "/" + rel
        var cut = dest.rfind("/")
        if cut > 0:
            _ = run_capture("mkdir -p " + quote_arg(String(dest[byte=0:cut])))
        _ = fetch_to_cache(
            meta.download_url, dest, meta.sha256, meta.size_bytes, rel
        )
        pulled += 1
    print(
        String(n)
        + " artifacts listed for "
        + rid
        + ": "
        + String(pulled)
        + " pulled, "
        + String(pending)
        + " still pending on the server"
    )


def main() raises:
    var cmd = _positional(0)
    if cmd == "init":
        cmd_init()
    elif cmd == "list":
        cmd_list()
    elif cmd == "show":
        cmd_show()
    elif cmd == "tag":
        cmd_tag()
    elif cmd == "prune":
        cmd_prune()
    elif cmd == "push":
        cmd_push()
    elif cmd == "pull":
        cmd_pull()
    else:
        print("usage: project_cli <init|list|show|tag|prune|push|pull> ...")
        print("  see the module docstring, or docs/PROJECT_LAYER_PLAN.md §10")
