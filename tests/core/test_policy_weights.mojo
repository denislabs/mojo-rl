# +--------------------------------------------------------------------------+ #
# | A promoted policy's weights, fetched where it was not promoted
# +--------------------------------------------------------------------------+ #
"""Gate `data/policy_weights.mojo` against the mock monitor.

    pixi run mojo run -I . tests/core/test_policy_weights.mojo

⚠⚠ CHECK 2 IS THE ONE THAT MATTERS: a run's `best.ckpt` overwritten AFTER the
promotion must be refused, because the artifact at that path is then a model
nobody judged. It is asserted by the bytes on disk (nothing written), not by
the report's count.
"""

from std.os import makedirs
from std.os.path import exists
from std.time import sleep

from noeira.core.policy import PolicyRecord, policy_ckpt_path, write_policy
from noeira.data.policy_weights import pull_policy_weights
from noeira.data.remote import RemoteCatalog
from noeira.io.fileio import file_size, remove_file, write_file_atomic
from noeira.io.http import HttpClient, http_shim_available
from noeira.io.proc import run_capture
from noeira.io.sha256 import sha256_file


comptime WORK = "/tmp/noeira_policy_weights_gate"
comptime PORT_FILE = "/tmp/noeira_policy_weights_gate_port"
comptime LOG_FILE = "/tmp/noeira_policy_weights_gate_log"


def _blob(path: String, seed: Int, n: Int) raises:
    makedirs(String(path[byte=0 : path.rfind("/")]), exist_ok=True)
    var b = List[UInt8]()
    for i in range(n):
        b.append(UInt8((i * 13 + seed * 7 + 1) & 0xFF))
    write_file_atomic(path, b)


def _start_server() raises -> String:
    for p in [String(PORT_FILE), String(LOG_FILE)]:
        try:
            remove_file(p)
        except:
            pass
    _ = run_capture(
        "python3 tools/io/mock_monitor_server.py " + String(PORT_FILE) + " "
        + String(LOG_FILE) + " 120 > /tmp/noeira_policy_weights_gate_server.log 2>&1 &"
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


def _check(c: Bool, what: String) raises:
    if not c:
        raise Error(what)


def main() raises:
    print("[policy-weights] gate")
    if not http_shim_available():
        raise Error("the http shim is not built — pixi run build-http")
    _ = run_capture(String("rm -rf ") + WORK)
    var url = _start_server()
    var n = 0
    try:
        var cat = RemoteCatalog(url, String("gate-key"))
        var box = String(WORK) + "/box"           # where it was trained and promoted
        var mac = String(WORK) + "/mac/so101-tower"  # where it is deployed

        # The training box uploads two runs' best checkpoints.
        var rid = String("2026-09-15_act-so101_aaaa0001")
        _blob(box + "/best_a.ckpt", 1, 50_000)
        _ = cat.push_artifact(rid, String("checkpoints/best.ckpt"), box + "/best_a.ckpt", String("checkpoint"))

        # The promotion record, as synced to the Mac (no weights).
        makedirs(mac + "/policies", exist_ok=True)
        var pol = PolicyRecord(String("act"))
        pol.run = rid
        pol.checkpoint = String("best")
        pol.sha256 = sha256_file(box + "/best_a.ckpt")
        pol.bytes = 50_000
        pol.promoted = String("2026-09-15T18:00:00Z")
        write_policy(mac, pol)

        # ── 1. weights pulled to the role, byte for byte ───────────────
        var r1 = pull_policy_weights(cat, mac)
        var dst = policy_ckpt_path(mac, String("act"))
        _check(r1.pulled == 1 and exists(dst) and sha256_file(dst) == pol.sha256, "1: weights not materialised at the role")
        var r1b = pull_policy_weights(cat, mac)
        _check(r1b.present == 1 and r1b.pulled == 0, "1: a second pull re-downloaded present weights")
        print("  1 pulled to policies/act.ckpt, identical; a second pull downloads nothing")
        n += 2

        # ── 2. ⚠⚠ best.ckpt overwritten after the promotion: REFUSED ────
        remove_file(dst)
        _blob(box + "/best_b.ckpt", 2, 50_000)
        _ = cat.push_artifact(rid, String("checkpoints/best.ckpt"), box + "/best_b.ckpt", String("checkpoint"))
        var r2 = pull_policy_weights(cat, mac)
        _check(r2.failed == 1 and r2.pulled == 0, "2: expected a refusal")
        _check(not exists(dst), "2: A CHECKPOINT NOBODY PROMOTED WAS WRITTEN TO THE ROLE")
        _check(r2.lines[0].find("not the checkpoint that was promoted") >= 0, "2: the refusal must say why")
        print("  2 run's best.ckpt overwritten after promotion: refused, nothing written")
        n += 3

        # ── 3. a run with no uploaded checkpoint is reported, not raised ──
        var pol2 = PolicyRecord(String("reach"))
        pol2.run = String("2026-09-15_sac-reach_bbbb0002")
        pol2.checkpoint = String("best")
        pol2.sha256 = String("0") * 64
        pol2.bytes = 10
        pol2.promoted = String("2026-09-15T18:00:00Z")
        write_policy(mac, pol2)
        var r3 = pull_policy_weights(cat, mac)
        var missing = 0
        for l in r3.lines:
            if l.startswith("MISSING   reach"):
                missing += 1
        _check(missing == 1, "3: a policy whose run uploaded nothing must be reported MISSING")
        print("  3 a policy whose run has no uploaded checkpoint: reported MISSING, others still processed")
        n += 1
    finally:
        try:
            var h = HttpClient(2000, 2000)
            _ = h.request(String("GET"), url + "/__shutdown")
        except:
            pass
    print("  " + String(n) + " checks, 0 failures")
    print("[PASS] policy weights (" + String(n) + " checks)")
