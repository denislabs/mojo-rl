# +--------------------------------------------------------------------------+ #
# | A promoted policy's weights, on a machine where it was not promoted
# +--------------------------------------------------------------------------+ #
"""`project-pull <project> --weights`: fill `policies/<role>.ckpt` from the run.

Promotion happens on the box that trained: it hard-links the run's checkpoint
to `policies/<role>.ckpt` and writes `policies/<role>.kv` (+ `.norm.json`).
Project sync carries the `.kv` and `.norm.json` — they are definition — but
never the weights, which are hundreds of megabytes and already uploaded as
the RUN's artifact. This closes the gap: for every policy record whose weights
are missing here, find `checkpoints/<checkpoint>.ckpt` among the run's
artifacts and download it to the role's path.

⚠⚠ THE DIGEST IS THE ONE RECORDED AT PROMOTION, NOT THE ARTIFACT'S. A run's
`best.ckpt` is a role inside the run and is overwritten every time validation
improves; if training went on after the promotion, the artifact at that path
is a DIFFERENT model than the one a person judged and promoted. The download
is verified against `policies/<role>.kv`'s `sha256=`, and a mismatch is
refused with that explanation rather than deployed.
"""

from std.os.path import exists

from mojo_rl.core.policy import load_policy, policy_ckpt_path
from mojo_rl.data.remote import RemoteCatalog
from mojo_rl.io.fetch import fetch_to_cache
from mojo_rl.io.fileio import file_size
from mojo_rl.io.proc import quote_arg, run_capture
from mojo_rl.io.sha256 import sha256_file


struct WeightsReport(Movable):
    var lines: List[String]
    var pulled: Int
    var present: Int
    var failed: Int

    def __init__(out self):
        self.lines = List[String]()
        self.pulled = 0
        self.present = 0
        self.failed = 0

    def __init__(out self, *, deinit move: Self):
        self.lines = move.lines^
        self.pulled = move.pulled
        self.present = move.present
        self.failed = move.failed


def pull_policy_weights(mut cat: RemoteCatalog, project_dir: String) raises -> WeightsReport:
    var rep = WeightsReport()
    var listing = run_capture(
        String("ls -1 ") + quote_arg(project_dir + "/policies") + " 2>/dev/null", 1 << 16
    )
    for line in listing.split("\n"):
        var f = String(line.strip())
        if not f.endswith(".kv"):
            continue
        var role = String(f[byte=0 : f.byte_length() - 3])
        var pol = load_policy(project_dir + "/policies/" + f)
        var dst = policy_ckpt_path(project_dir, role)
        if exists(dst) and sha256_file(dst) == pol.sha256:
            rep.present += 1
            rep.lines.append("present   " + role + "  (" + pol.run + " " + pol.checkpoint + ")")
            continue
        var want = String("checkpoints/") + pol.checkpoint + ".ckpt"
        var doc = cat.artifacts_of(pol.run)
        var root = doc.root()
        var id = String("")
        var status = String("")
        for i in range(doc.size(root)):
            var row = doc.at(root, i)
            if doc.string(doc.field(row, String("path"))) == want:
                id = doc.string(doc.field(row, String("id")))
                status = doc.string(doc.field(row, String("status")))
        if id.byte_length() == 0:
            rep.failed += 1
            rep.lines.append(
                "MISSING   " + role + "  — run " + pol.run + " has no uploaded " + want
                + " on the platform (push it from the box that trained: project-push)"
            )
            continue
        if status != "ready":
            rep.failed += 1
            rep.lines.append("PENDING   " + role + "  — " + want + " of run " + pol.run + " is still uploading")
            continue
        var meta = cat.describe_artifact(id)
        if pol.sha256.byte_length() == 64 and meta.sha256 != pol.sha256:
            rep.failed += 1
            rep.lines.append(
                "REFUSED   " + role + "  — the platform's " + want + " of run " + pol.run
                + " is not the checkpoint that was promoted (sha " + String(meta.sha256[byte=0:12])
                + " vs promoted " + String(pol.sha256[byte=0:12])
                + "). Training overwrote it after the promotion: promote again on the box,"
                " or promote a step_* checkpoint that does not move."
            )
            continue
        _ = fetch_to_cache(meta.download_url, dst, pol.sha256, pol.bytes, role + ".ckpt")
        rep.pulled += 1
        rep.lines.append("pulled    " + role + "  (" + pol.run + " " + pol.checkpoint + ", " + String(file_size(dst) // 1_000_000) + " MB)")
    return rep^
