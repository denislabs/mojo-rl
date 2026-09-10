# +--------------------------------------------------------------------------+ #
# | Promotion — the only way a checkpoint leaves runs/
# +--------------------------------------------------------------------------+ #
"""`projects/<project>/policies/<name>.kv` and its materialised weights.

    projects/so101/policies/reach.kv
        name=reach
        run=2026-09-02_act-reach_a3f21c8b
        checkpoint=best
        sha256=9f1c...
        promoted=2026-09-02T14:40:00Z
        note=8/10 on the real arm, gripper ok
        supersedes=2026-08-30_act-reach_91b0f2aa

## ⚠⚠ Promotion is an explicit, recorded HUMAN act

§8, and it is the answer to decision 2: **deployment never knows a run id.**
`deploy_reach_real.mojo` points at `policies/reach.ckpt`, which is a ROLE. Which
run currently fills that role is a fact recorded here, by a person, with a
sentence saying why — and the sentence is the only thing that ever knew whether
a checkpoint was actually good. No metric knows "8/10 on the real arm".

## ⚠⚠ COPY OR HARD-LINK, NEVER SYMLINK

A symlink into `runs/` breaks three ways that all look like the same
confusing failure much later:

* `project-pull` onto another machine brings the policy and not its target;
* rsync/sftp either follow it (silently duplicating) or copy a dangling link;
* `project-prune` deletes the run and the policy becomes a dead pointer.

A hard link costs nothing on the same filesystem and survives all three. When
the link cannot be made — a different filesystem — the answer is a COPY, not a
symlink.

## The invariant

Nothing outside `runs/` and `policies/` is ever a policy. That is what stops a
second flat 218-file directory from accreting beside the first one.
"""

from .kv import KvWriter, kv_lines
from .project import projects_root
from ..io.fileio import write_text_atomic


comptime SCHEMA_VERSION = "1"

comptime _KNOWN_KEYS = (
    "schema_version name run checkpoint sha256 bytes promoted note supersedes"
)


struct PolicyRecord(Movable):
    """`policies/<name>.kv`, as read or as being written."""

    var name: String
    var run: String
    """The run id that produced these weights. ⚠ RECORDED, NOT USED BY THE
    DEPLOY PATH — that is the whole point of the role."""
    var checkpoint: String
    """`best`, `last`, `step_400000` — which of the run's checkpoints."""
    var sha256: String
    var bytes: Int
    var promoted: String
    var note: String
    """⚠ THE HUMAN'S JUDGEMENT, and the only field a metric cannot supply."""
    var supersedes: String
    """The run id this replaced, or empty for the first promotion. Gives a
    history for free — each record names its predecessor."""

    def __init__(out self, name: String = String("")):
        self.name = name
        self.run = String("")
        self.checkpoint = String("")
        self.sha256 = String("")
        self.bytes = 0
        self.promoted = String("")
        self.note = String("")
        self.supersedes = String("")

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.run = move.run^
        self.checkpoint = move.checkpoint^
        self.sha256 = move.sha256^
        self.bytes = move.bytes
        self.promoted = move.promoted^
        self.note = move.note^
        self.supersedes = move.supersedes^

    def render(self) raises -> String:
        var w = KvWriter(String("policy record"))
        w.add(String("schema_version"), String(SCHEMA_VERSION))
        w.add(String("name"), self.name)
        w.add(String("run"), self.run)
        w.add(String("checkpoint"), self.checkpoint)
        w.add(String("sha256"), self.sha256)
        w.add(String("bytes"), String(self.bytes))
        w.add(String("promoted"), self.promoted)
        if self.note.byte_length() > 0:
            w.add(String("note"), self.note)
        if self.supersedes.byte_length() > 0:
            w.add(String("supersedes"), self.supersedes)
        return w^.done()


def parse_policy(text: String, what: String) raises -> PolicyRecord:
    """Read a `policies/<name>.kv`.

    ⚠ AN UNKNOWN KEY IS KEPT SILENTLY, NOT REFUSED. A record written by a newer
    version must still be readable by an older one — the alternative is a tool
    that refuses to show you a policy because it gained a field.
    """
    var p = PolicyRecord()
    for line in kv_lines(text, what):
        var k = line.key
        var v = line.value
        if k == "name":
            p.name = v
        elif k == "run":
            p.run = v
        elif k == "checkpoint":
            p.checkpoint = v
        elif k == "sha256":
            p.sha256 = v
        elif k == "bytes":
            try:
                p.bytes = atol(v)
            except:
                p.bytes = 0
        elif k == "promoted":
            p.promoted = v
        elif k == "note":
            p.note = v
        elif k == "supersedes":
            p.supersedes = v
    return p^


def load_policy(path: String) raises -> PolicyRecord:
    var txt: String
    with open(path, "r") as fh:
        txt = fh.read()
    return parse_policy(txt, path)


def policy_kv_path(project_dir: String, name: String) -> String:
    return project_dir + "/policies/" + name + ".kv"


def policy_ckpt_path(project_dir: String, name: String) -> String:
    """⚠ THE PATH A DEPLOY SCRIPT POINTS AT. It names a ROLE, never a run, and
    that is decision 2: nothing on the deploy path has to be edited when a
    better checkpoint arrives — only the policy is re-promoted."""
    return project_dir + "/policies/" + name + ".ckpt"


def write_policy(project_dir: String, ref rec: PolicyRecord) raises:
    write_text_atomic(policy_kv_path(project_dir, rec.name), rec.render())


def resolve_policy(
    project: String,
    name: String,
    fallback: String = String(""),
    root: String = String(""),
) raises -> String:
    """The weights a deploy path should load for a ROLE, or `fallback`.

    ⚠⚠ THIS IS DECISION 2, AND IT IS THE ONLY THING A DEPLOY SCRIPT SHOULD
    KNOW. It names a role — "reach" — never a run id, so a better checkpoint is
    one `project-promote` away and no deploy script is edited, rebuilt, or
    re-flashed.

    ⚠ IT FALLS BACK RATHER THAN RAISING, and the fallback is the constant the
    script used before. A deploy path is the one place in this tree where
    failing closed is worse than failing open: an arm that will not start
    because a project layer is not set up yet is a regression for someone who
    only wanted to run the thing that worked yesterday.

    ⚠ `root` OVERRIDES `projects_root()`, for a gate that must not depend on
    the ambient `MOJO_RL_PROJECTS`. Deploy paths leave it empty.

    ⚠ THE CALLER MUST PRINT WHICH ONE IT GOT. Silently loading different
    weights than the operator expects is the failure this whole layer exists to
    prevent — see `describe_policy`.
    """
    from std.os.path import exists

    var base = root if root.byte_length() > 0 else projects_root()
    var role = policy_ckpt_path(base + "/" + project, name)
    if exists(role):
        return role
    return fallback


def describe_policy(
    project: String, name: String, root: String = String("")
) raises -> String:
    """One line saying WHICH run's weights a role currently holds, or "".

    ⚠ MEANT TO BE PRINTED BEFORE THE ARM MOVES. "reach -> 2026-09-02_... (best),
    promoted 2026-09-02, note: 8/10 on the real arm" is the sentence that lets
    an operator stop a deploy they did not intend.
    """
    from std.os.path import exists

    var base = root if root.byte_length() > 0 else projects_root()
    var kv = policy_kv_path(base + "/" + project, name)
    if not exists(kv):
        return String("")
    var rec = load_policy(kv)
    var line = (
        rec.name + " -> " + rec.run + " (" + rec.checkpoint + "), promoted "
        + rec.promoted
    )
    if rec.note.byte_length() > 0:
        line += ("\n    note: " + rec.note)
    return line^
