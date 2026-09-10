# +--------------------------------------------------------------------------+ #
# | Every checkpoint a driver writes must be offered to the artifact sink
# +--------------------------------------------------------------------------+ #
"""A SOURCE gate over `mojo_rl/deep_agents/training/`.

    pixi run mojo run -I . tests/deep_agents/test_checkpoints_announce.mojo

## ⚠⚠ Why a source gate rather than a behavioural one

`trainer.save_state(checkpoint_path)` appears eighteen times across four driver
files. A site that saves and forgets to announce produces an artifact that
simply never leaves the box — and there is NOTHING TO SEE: the run trains, the
checkpoint is on disk, the dashboard shows the run, and only the box dying
reveals that the weights were never uploaded.

That is not a defect a behavioural test can reach. Covering it that way would
mean running all eighteen code paths — several of which are NVIDIA-only, and
one of which is a CUDA-graph capture. So the gate reads the source and checks
the one mechanical rule that makes the drift impossible.

This is the same shape as `tests/core/test_drivers_use_runcontext.mojo`, and it
exists for the same reason that one does: the previous version of this mistake
— five hand-rolled `--tag` blocks in the FB family, of which a patch touched
two — was found by accident rather than by a gate.

## ⚠ The file list is IN THE GATE, not derived

A gate that globbed `training/*.mojo` would pass forever the day someone adds a
fifth driver, because a file that is not scanned cannot fail. The list is
written down, and the count of scanned save sites is PRINTED beside the count
of failures — "0 violations" is also what scanning nothing prints.
"""

from mojo_rl.io.fileio import read_file_bytes


def drivers() -> List[String]:
    """⚠ A LIST, NOT A `comptime` TUPLE: a tuple cannot be indexed by a runtime
    loop variable. Written down rather than globbed — see the header."""
    var out = List[String]()
    out.append(String("mojo_rl/deep_agents/training/driver_offpolicy.mojo"))
    out.append(
        String("mojo_rl/deep_agents/training/driver_offpolicy_discrete.mojo")
    )
    out.append(String("mojo_rl/deep_agents/training/driver_onpolicy.mojo"))
    out.append(
        String("mojo_rl/deep_agents/training/driver_onpolicy_discrete.mojo")
    )
    return out^

comptime SAVE = "trainer.save_state(checkpoint_path)"
comptime ANNOUNCE = "announce_checkpoint(checkpoint_path, artifacts, run_dir)"

comptime EXPECT_SITES = 18
"""⚠ PINNED. If a driver gains or loses a save site this gate FAILS rather than
silently scanning a different amount of code — the number is the evidence that
the scan reached what it was written against."""


def _read(path: String) raises -> String:
    var bytes = read_file_bytes(path)
    bytes.append(0)
    return String(unsafe_from_utf8_ptr=bytes.unsafe_ptr())


def _lines(text: String) raises -> List[String]:
    var out = List[String]()
    for line in text.split("\n"):
        out.append(String(line))
    return out^


def main() raises:
    print("=== every save_state is announced ===")

    var sites = 0
    var violations = 0
    var files_scanned = 0

    var files = drivers()
    for i in range(len(files)):
        var path = String(files[i])
        var text = _read(path)
        files_scanned += 1
        var lines = _lines(text)

        for n in range(len(lines)):
            var line = String(lines[n]).strip()
            # ⚠ The docstring in driver_offpolicy MENTIONS the call. Only a
            # line that IS the call counts, or the gate would demand an
            # announce inside a comment.
            if String(line) != String(SAVE):
                continue
            sites += 1

            # The next line that is not blank must be the announce.
            var found = String("")
            for m in range(n + 1, len(lines)):
                var nxt = String(lines[m]).strip()
                if String(nxt).byte_length() == 0:
                    continue
                found = String(nxt)
                break
            if found.find(String(ANNOUNCE)) < 0:
                violations += 1
                print(
                    "    "
                    + path
                    + ":"
                    + String(n + 1)
                    + " saves a checkpoint and does not announce it"
                )
                print("      next line was: " + found)

    # ⚠ THE SCANNED COUNT IS PRINTED BESIDE THE FAILURE COUNT. "0 violations"
    # is also what a gate that read nothing reports.
    print(
        "  "
        + String(files_scanned)
        + " drivers scanned, "
        + String(sites)
        + " save sites, "
        + String(violations)
        + " unannounced"
    )

    if files_scanned != len(files):
        raise Error("a driver in the list was not read")
    if sites != EXPECT_SITES:
        raise Error(
            "expected "
            + String(EXPECT_SITES)
            + " save sites, found "
            + String(sites)
            + " — a driver gained or lost one, so update EXPECT_SITES"
            " DELIBERATELY rather than letting the scan drift"
        )
    if violations != 0:
        raise Error(
            String(violations)
            + " checkpoint(s) are written and never offered to the sink."
            " The artifact would stay on the box with nothing to see."
        )

    # ⚠ And the rule itself must live in ONE place. A driver that inlined the
    # offer instead of calling the helper would pass the check above while
    # reintroducing exactly the drift it exists to prevent.
    var inlined = 0
    for i in range(len(files)):
        var text = _read(String(files[i]))
        if text.find(String(".offer(")) >= 0:
            inlined += 1
            print("    " + String(files[i]) + " calls sink.offer() directly")
    if inlined != 0:
        raise Error(
            String(inlined)
            + " driver(s) inline the offer instead of calling"
            " announce_checkpoint — that is the rule written twice"
        )
    print("  and no driver inlines the offer; the rule has one home")

    print("[PASS] checkpoints announce (" + String(sites) + " sites)")
