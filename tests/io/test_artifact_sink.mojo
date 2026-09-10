# +--------------------------------------------------------------------------+ #
# | Artifacts leaving the box: what got sent, and what deliberately did not
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/io/artifact_sink.mojo` — §7 of the project layer plan.

    pixi run build-http                              # ONCE
    pixi run mojo run -I . tests/io/test_artifact_sink.mojo

Hermetic: `tools/io/mock_monitor_server.py` plays BOTH the monitor Worker and
R2 on a loopback port, recording every request. Nothing here needs the network,
a credential, or a live service.

## What could be wrong, and what each check is for

* **Nothing is uploaded.** The whole point is that a box dying at 3am does not
  take the weights with it.
* **⚠⚠ The bytes arrive corrupt or truncated.** An artifact store that returns
  a checkpoint which will not load is worse than one that returns nothing, so
  the gate hashes what the fixture RECEIVED and compares it to the file on
  disk. A status code cannot see this.
* **⚠⚠ Every save is uploaded.** This is the defect §7's supersede rule exists
  to prevent: ACT validates every N steps, so a naive sink transfers gigabytes
  of weights that a later save has already replaced. The check is a COUNT OF
  PUTs, and it must be strictly less than the number of offers.
* **The policy is not applied, or is applied on the wrong side.** A declined
  request must not even be queued — the queue is not where the saving happens.
* **`close()` returns before the transfer finished.** Then the process exits
  and the artifact is gone, which is the failure this whole file is about.
* **A failure is silent.** An artifact that never left must be COUNTED, or a
  reassuring summary hides the one fact the operator needed.
"""

from std.os.path import exists
from std.time import sleep

from mojo_rl.io.artifact_sink import (
    ArtifactSink,
    KIND_CHECKPOINT,
    KIND_EVAL,
    KIND_VIDEO,
)
from mojo_rl.io.fileio import remove_file, write_file_atomic
from mojo_rl.io.http import http_shim_available
from mojo_rl.io.proc import run_capture
from mojo_rl.io.sha256 import sha256_file


comptime PORT_FILE = "/tmp/mojo_rl_art_gate_port"
comptime LOG_FILE = "/tmp/mojo_rl_art_gate_log"
comptime RUN_DIR = "/tmp/mojo_rl_art_gate_run"
comptime RUN_ID = "2026-09-10_art-gate_c0ffee01"


def _start_server() raises -> String:
    for p in [String(PORT_FILE), String(LOG_FILE)]:
        try:
            remove_file(p)
        except:
            pass
    _ = run_capture(
        "python3 tools/io/mock_monitor_server.py "
        + String(PORT_FILE)
        + " "
        + String(LOG_FILE)
        + " 180 > /tmp/mojo_rl_art_gate_server.log 2>&1 &"
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


def _log() raises -> List[String]:
    var out = List[String]()
    if not exists(LOG_FILE):
        return out^
    var f = open(String(LOG_FILE), "r")
    var text = String(f.read())
    f.close()
    for line in text.split("\n"):
        var s = String(line).strip()
        if String(s).byte_length() > 0:
            out.append(String(s))
    return out^


def _count(method: String, needle: String) raises -> Int:
    var n = 0
    for line in _log():
        var parts = String(line).split(" ")
        if len(parts) >= 3 and String(parts[1]) == method:
            if needle.byte_length() == 0 or String(parts[2]).find(needle) >= 0:
                n += 1
    return n


def _clear() raises:
    var f = open(String(LOG_FILE), "w")
    f.write(String(""))
    f.close()


def _write_blob(rel: String, seed: Int, n: Int) raises -> String:
    """A deterministic file under the run dir. Returns its sha256."""
    var cut = rel.rfind("/")
    if cut > 0:
        _ = run_capture(
            "mkdir -p " + String(RUN_DIR) + "/" + String(rel[byte=0:cut])
        )
    var bytes = List[UInt8]()
    for i in range(n):
        bytes.append(UInt8((i * 31 + seed * 7 + 11) & 0xFF))
    write_file_atomic(String(RUN_DIR) + "/" + rel, bytes)
    return sha256_file(String(RUN_DIR) + "/" + rel)


def _sink(base: String, run_id: String = String(RUN_ID)) raises -> ArtifactSink:
    return ArtifactSink(
        run_id=run_id,
        run_dir=String(RUN_DIR),
        base_url=base,
        api_key=String("gate-key"),
    )


def main() raises:
    print("=== artifact sink (§7) ===")
    if not http_shim_available():
        raise Error(
            "the HTTP shim is not built — run `pixi run build-http` first"
        )
    _ = run_capture("rm -rf " + String(RUN_DIR) + " && mkdir -p " + String(RUN_DIR))
    var base = _start_server()
    print("  fixture at " + base + " (monitor + R2)")
    var checks = 0

    # ── 1. one artifact, end to end, BYTE FOR BYTE ──────────────────
    #
    # ⚠ The decisive part is not the 200s, it is that what the fixture
    # RECEIVED hashes to the file on disk. An artifact store that returns a
    # checkpoint which will not load is worse than one that returns nothing.
    _clear()
    var sha = _write_blob(String("checkpoints/best.ckpt"), 1, 64 * 1024)
    var a = _sink(base)
    if not a.offer(String("checkpoints/best.ckpt"), String(KIND_CHECKPOINT)):
        raise Error("offer() refused the first best.ckpt")
    a.close(drain_ms=20000)
    if a.uploaded() != 1 or a.failed() != 0:
        raise Error(
            "one artifact: uploaded="
            + String(a.uploaded())
            + " failed="
            + String(a.failed())
        )
    var regs = _count(String("POST"), String("/artifacts"))
    var puts = _count(String("PUT"), String(""))
    if regs != 2 or puts != 1:
        # register + complete are both POSTs to /artifacts*
        raise Error(
            "expected register+complete and one PUT, saw "
            + String(regs)
            + " POSTs and "
            + String(puts)
            + " PUTs"
        )
    var got = String("")
    for line in _log():
        if String(line).find(" PUT ") >= 0:
            got = String(line)
    if got.find(String(sha[byte=0:16])) < 0:
        raise Error(
            "the bytes that arrived do not hash to the file on disk:\n    "
            + got
            + "\n    want sha prefix "
            + String(sha[byte=0:16])
        )
    print(
        "  one artifact: register + PUT + complete, and the received bytes"
        " hash to the file on disk"
    )
    checks += 3

    # ── 2. ⚠⚠ supersede: many offers, FEWER transfers ───────────────
    #
    # THE DECISIVE CHECK OF THIS FILE. A sink without the dedup PUTs once per
    # offer, which is exactly the "gigabytes of superseded weights" §7 names.
    # The file is overwritten between offers, as a real driver does.
    # ⚠ THE RUN ID IS `slow-...` ON PURPOSE. The fixture sleeps 150 ms on a PUT
    # whose key contains `slow`, which is the ONLY way to reproduce the
    # condition supersede exists for: requests arriving faster than transfers
    # complete. Over loopback a PUT finishes before the next offer is made, so
    # a sink with NO dedup at all passes a naive version of this check — which
    # is exactly what the first draft of this file did.
    _clear()
    var b = _sink(base, String("slow-run-c0ffee"))
    var offers = 12
    var last_sha = String("")
    for i in range(offers):
        last_sha = _write_blob(String("checkpoints/best.ckpt"), i + 2, 8 * 1024)
        if not b.offer(String("checkpoints/best.ckpt"), String(KIND_CHECKPOINT)):
            raise Error("offer() refused best.ckpt at " + String(i))
    b.close(drain_ms=20000)
    var transfers = _count(String("PUT"), String(""))
    if b.superseded() == 0:
        raise Error(
            "12 offers of the same path collapsed NOTHING — the dedup is not"
            " running"
        )
    if transfers >= offers:
        raise Error(
            String(offers)
            + " offers produced "
            + String(transfers)
            + " transfers; supersede must make this strictly smaller"
        )
    if b.uploaded() + b.superseded() != offers:
        raise Error(
            "accounting does not close: uploaded="
            + String(b.uploaded())
            + " superseded="
            + String(b.superseded())
            + " offers="
            + String(offers)
        )
    # ⚠ AND THE SURVIVOR IS THE NEWEST. Collapsing to an OLD upload would be
    # worse than collapsing nothing: it would advertise stale weights as the
    # run's best.
    var last_put = String("")
    for line in _log():
        if String(line).find(" PUT ") >= 0:
            last_put = String(line)
    if last_put.find(String(last_sha[byte=0:16])) < 0:
        raise Error(
            "the last transfer was not the newest bytes — supersede kept an"
            " older file:\n    "
            + last_put
        )
    print(
        "  supersede: "
        + String(offers)
        + " offers -> "
        + String(transfers)
        + " transfers ("
        + String(b.uploaded())
        + " uploaded, "
        + String(b.superseded())
        + " collapsed), and the last one is the newest bytes"
    )
    checks += 4

    # ── 3. the policy declines BEFORE the queue ─────────────────────
    _clear()
    _ = _write_blob(String("checkpoints/last.ckpt"), 1, 8 * 1024)
    _ = _write_blob(String("checkpoints/step_100.ckpt"), 1, 8 * 1024)
    var c = _sink(base)
    # `last` is allowed once, then rate-limited for the default 15 minutes.
    var last_ok = 0
    for _ in range(5):
        if c.offer(String("checkpoints/last.ckpt"), String(KIND_CHECKPOINT)):
            last_ok += 1
    var step_ok = c.offer(
        String("checkpoints/step_100.ckpt"), String(KIND_CHECKPOINT)
    )
    c.close(drain_ms=20000)
    if last_ok != 1:
        raise Error(
            "`last` was accepted "
            + String(last_ok)
            + " times in a row; the default is once per 15 minutes"
        )
    if step_ok:
        raise Error("`step_100.ckpt` was accepted without --keep-every")
    if _count(String("PUT"), String("step_")) != 0:
        raise Error("a step_ checkpoint reached the wire")
    print(
        "  policy: `last` accepted 1 of 5, `step_` refused — and refused at"
        " offer(), so nothing was queued"
    )
    checks += 3

    # ── 4. ...and --keep-every is what turns step_ back on ──────────
    _clear()
    var d = ArtifactSink(
        run_id=String(RUN_ID),
        run_dir=String(RUN_DIR),
        base_url=base,
        api_key=String("gate-key"),
        keep_every=True,
    )
    if not d.offer(String("checkpoints/step_100.ckpt"), String(KIND_CHECKPOINT)):
        raise Error("--keep-every did not re-enable step_ checkpoints")
    d.close(drain_ms=20000)
    if d.uploaded() != 1:
        raise Error("keep_every: uploaded " + String(d.uploaded()) + " of 1")
    print("  policy: --keep-every re-enables step_, and it lands")
    checks += 1

    # ── 5. ⚠ close() does not return before the bytes are gone ──────
    #
    # A close that returns early is indistinguishable from a working one until
    # the process exits and the artifact is not there. Six files, all of which
    # must be on the far side by the time close() comes back.
    # ⚠⚠ THE RUN ID IS `slow-` HERE TOO, AND CHECK 5 WAS VACUOUS WITHOUT IT.
    # Over loopback the worker had already uploaded all six before `close()`
    # was even called, so a mutant that passed `drain_ms = 0` — abandoning the
    # backlog outright — still passed this check. With 150 ms per PUT the
    # backlog is real when close() runs, and only a close that WAITS clears it.
    _clear()
    var e = _sink(base, String("slow-close-c0ffee"))
    for i in range(6):
        _ = _write_blob(
            String("eval/rollout_") + String(i) + ".mp4", i + 40, 32 * 1024
        )
        if not e.offer(
            String("eval/rollout_") + String(i) + ".mp4", String(KIND_VIDEO)
        ):
            raise Error("offer() refused rollout " + String(i))
    e.close(drain_ms=30000)
    var landed = _count(String("PUT"), String("rollout_"))
    if landed != 6 or e.uploaded() != 6:
        raise Error(
            "close() returned with "
            + String(landed)
            + " of 6 on the wire (sink says "
            + String(e.uploaded())
            + ")"
        )
    if e.abandoned() != 0:
        raise Error("close() abandoned " + String(e.abandoned()) + " artifacts")
    print("  close: all 6 distinct artifacts were on the far side before it returned")
    checks += 2

    # ── 6. a failure is COUNTED, and does not stop the run ──────────
    #
    # ⚠ The sink must survive a dashboard that is down. The failure must also
    # be VISIBLE: an artifact that never left is the exact condition this sink
    # exists to end, so a reassuring summary would hide the one fact that
    # matters.
    _clear()
    _ = _write_blob(String("checkpoints/best.ckpt"), 99, 16 * 1024)
    var f = _sink(String("http://127.0.0.1:9"))
    if not f.offer(String("checkpoints/best.ckpt"), String(KIND_CHECKPOINT)):
        raise Error("offer() refused against a dead dashboard")
    f.close(drain_ms=8000)
    if f.uploaded() != 0 or f.failed() + f.abandoned() != 1:
        raise Error(
            "dead dashboard: uploaded="
            + String(f.uploaded())
            + " failed="
            + String(f.failed())
            + " abandoned="
            + String(f.abandoned())
            + " (want 0 uploaded and exactly 1 accounted for)"
        )
    # ⚠ ALL FOUR LOSS COUNTS, NOT ANY ONE OF THEM. The first version of this
    # check accepted the line as long as it said "failed" somewhere — so a
    # report that silently dropped `abandoned` and `dropped` passed it, and
    # those two are precisely the artifacts still sitting only on this box.
    var rep = f.report()
    var missing = List[String]()
    for word in [
        String("uploaded"),
        String("superseded"),
        String("failed"),
        String("abandoned"),
        String("dropped"),
    ]:
        if rep.find(word) < 0:
            missing.append(String(word))
    if len(missing) > 0:
        raise Error(
            "report() omits "
            + String(len(missing))
            + " of the 5 counts (first: "
            + missing[0]
            + "): "
            + rep
        )
    print("  dead dashboard: 0 uploaded, 1 accounted for, and report() says so")
    print("    " + rep)
    checks += 2

    # ── 7. a missing file fails that ONE artifact, not the sink ─────
    _clear()
    _ = _write_blob(String("metrics.csv"), 5, 4 * 1024)
    var g = _sink(base)
    _ = g.offer(String("checkpoints/does_not_exist.ckpt"), String(KIND_CHECKPOINT))
    _ = g.offer(String("metrics.csv"), String(KIND_EVAL))
    g.close(drain_ms=20000)
    if g.uploaded() != 1 or g.failed() != 1:
        raise Error(
            "a missing file should fail alone: uploaded="
            + String(g.uploaded())
            + " failed="
            + String(g.failed())
        )
    print("  a missing file fails alone; the artifact behind it still lands")
    checks += 1

    # ── 8. an inert sink costs nothing and admits it ────────────────
    var h = _sink(String(""))
    if h.enabled():
        raise Error("a sink with no base_url reported itself enabled")
    for _ in range(500):
        _ = h.offer(String("checkpoints/best.ckpt"), String(KIND_CHECKPOINT))
    h.close(drain_ms=500)
    if h.uploaded() != 0 or h.dropped() != 0 or h.report() != "":
        raise Error("an inert sink did something: " + h.report())
    print("  inert sink: 500 offers, nothing queued, nothing reported")
    checks += 1

    _ = run_capture(
        "curl -s -X POST " + base + "/__shutdown > /dev/null 2>&1 || true"
    )
    _ = run_capture("rm -rf " + String(RUN_DIR))
    print("[PASS] artifact sink (" + String(checks) + " checks)")


# MUTANTS THIS FILE WAS CHECKED AGAINST (each must turn it red):
#   B1  _drain does not dedup (always append)      -> check 2
#   B2  _drain keeps the FIRST kind, not the last  -> (kind is not asserted;
#                                                     see the note below)
#   B3  _drain returns after ONE pop               -> check 2
#   B4  policy: `last` has no rate limit           -> check 3
#   B5  policy: `step_` allowed without keep_every -> check 3
#   B6  policy: `step_` refused even WITH it       -> check 4
#   B7  close() does not wait (drain_ms = 0)       -> check 5 (needs the
#                                                      SLOW fixture; see below)
#   B8  a failed upload is not counted             -> check 6
#   B9  report() omits abandoned/dropped           -> check 6 (needs ALL
#                                                      five counts asserted)
#   B10 a failure latches the whole sink dead      -> check 7
#   B11 offer() queues even when disabled          -> check 8
#
# ⚠⚠ B7 AND B9 SURVIVED THE FIRST SWEEP, and checks 5 and 6 were rewritten
# because of it — neither failure was visible by reading. Check 5 held because
# the worker had already finished before `close()` was called, so it measured
# nothing; check 6 asked whether the report said "failed" and a report that
# dropped `abandoned` and `dropped` still said it. A check whose outcome does
# not depend on the code under test is not a gate.
#
# ⚠ B2 IS NOT GATED AND THAT IS RECORDED RATHER THAN PAPERED OVER. Which
# `kind` survives a collapse is unobservable from the fixture's log, because
# the register body is written before the PUT and the gate matches on the PUT.
# Asserting it would need the fixture to correlate the two, which is more
# machinery than the fact is worth: `kind` is a grouping label, and a
# `best.ckpt` filed as "other" is findable either way.
