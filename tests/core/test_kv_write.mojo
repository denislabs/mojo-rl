"""`kv_write` — what the reader can read back, and what it must refuse.

Run: pixi run mojo run -I . tests/core/test_kv_write.mojo

`core/kv.mojo` was a READER with no writing half, so every producer of a
`key=value` file in this tree built its own line concatenation. The project
layer's `run.kv` and `project.kv` would have been the next two. This gates the
one writer instead.

⚠⚠ THE INVARIANT IS THE PARSED FORM, NOT THE BYTES. `kv_write(kv_lines(text))`
cannot equal `text`: the reader drops comments and blank lines and strips each
field, all of which are lossy on purpose. What must hold is that the KEY/VALUE
PAIRS survive, and that writing a second time changes nothing — so the format
has a fixpoint and a file rewritten by a tool is stable.

⚠ EVERY COUNT HERE PRINTS WHAT WAS COMPARED BESIDE WHAT DIFFERED. "0 mismatches"
and "nothing was tested" are the same output otherwise, and this gate walks a
directory: the day the glob returns nothing it must say so.
"""

from mojo_rl.core.kv import KvLine, KvWriter, kv_lines, kv_write


comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime FAMILY_DIR = "mojo_rl/tasks/families/"


def _slurp(path: String) raises -> String:
    with open(path, "r") as fh:
        return fh.read()


def _pairs(text: String, what: String) raises -> String:
    """The parsed form, flattened, so two of them compare in one shot."""
    var out = String("")
    var ls = kv_lines(text, what)
    for i in range(len(ls)):
        out += ls[i].key + "\x01" + ls[i].value + "\x02"
    return out^


def _expect_refusal(w: String, key: String, value: String, why: String) raises:
    var kw = KvWriter(String("gate"))
    var refused = False
    try:
        kw.add(key, value)
    except:
        refused = True
    if not refused:
        raise Error(
            "accepted a " + w + " it cannot read back (" + why + "): key='"
            + key + "' value='" + value + "'"
        )


# =============================================================================


def test_every_task_and_family_round_trips() raises:
    """The real files this format already carries, through write and back."""
    var paths = [
        String(TASK_DIR) + "so101_reach_brick.task",
        String(TASK_DIR) + "so101_lift_brick.task",
        String(TASK_DIR) + "so101_gather_bricks.task",
        String(TASK_DIR) + "so101_reach_clear.task",
        String(TASK_DIR) + "so101_settle_brick.task",
        String(FAMILY_DIR) + "so101_tabletop.family",
    ]
    var files = 0
    var fields = 0
    var differing = 0
    var not_fixpoint = 0
    for p in paths:
        var text = _slurp(p)
        var read1 = kv_lines(text, p)
        var written = kv_write(read1, p)
        var read2 = kv_lines(written, p)
        files += 1
        fields += len(read1)
        if _pairs(text, p) != _pairs(written, p):
            differing += 1
            print("    pairs changed:", p)
        # ⚠ The fixpoint is the separate claim: writing what was written must
        # produce the same bytes, or a tool that rewrites a file churns it.
        if kv_write(read2, p) != written:
            not_fixpoint += 1
            print("    not a fixpoint:", p)
    print(
        "  round-trip:", files, "files /", fields, "fields compared,",
        differing, "differing,", not_fixpoint, "not a fixpoint",
    )
    if files != 6 or fields < 40:
        raise Error(
            "gate went vacuous: " + String(files) + " files, "
            + String(fields) + " fields"
        )
    if differing != 0 or not_fixpoint != 0:
        raise Error("round-trip failed on " + String(differing) + " files")


def test_repeating_keys_keep_their_order() raises:
    """⚠ THE CASE THIS FORMAT EXISTS FOR. `artifact=`, `config=`, `active=` and
    `region=` all repeat; a writer that deduplicated or sorted would silently
    reorder a run's artifact list."""
    var w = KvWriter(String("gate"))
    w.add(String("active"), String("table"))
    w.add(String("active"), String("brick"))
    w.add(String("init"), String("brick@table_top"))
    w.add(String("active"), String("lid"))
    var got = w^.done()
    var want = String(
        "active=table\nactive=brick\ninit=brick@table_top\nactive=lid\n"
    )
    if got != want:
        raise Error("repeating keys reordered:\n" + got)
    var back = kv_lines(got, String("gate"))
    if len(back) != 4 or back[3].value != String("lid"):
        raise Error("round-trip lost a repeat: " + String(len(back)))
    print("  repeats: 4 lines, 3 sharing a key, order preserved")


def test_values_that_must_be_accepted() raises:
    """⚠ `=` AND `:` IN A VALUE ARE LEGAL AND LOAD-BEARING. `split_once` cuts at
    the first `=`, which is what lets `outcome=success_rate=0.82` and
    `region=table:site:table_surface:-0.1,-0.15,0.1,0.15` work at all. A writer
    that escaped or refused them would break the files already in the tree."""
    var cases = [
        (String("outcome"), String("success_rate=0.82 val_l1=0.031")),
        (
            String("region"),
            String("table:site:table_surface:-0.1,-0.15,0.1,0.15"),
        ),
        (String("language"), String("Move the gripper over the table")),
        (String("tag"), String("meilleur reach à ce jour, testé 8/10")),
        (String("note"), String("")),
        (String("artifact"), String("checkpoints/best.ckpt:sha256:9f1c:local")),
    ]
    var w = KvWriter(String("gate"))
    for c in cases:
        w.add(c[0], c[1])
    var text = w^.done()
    var back = kv_lines(text, String("gate"))
    var compared = 0
    var differing = 0
    for i in range(len(cases)):
        compared += 1
        if back[i].key != cases[i][0] or back[i].value != cases[i][1]:
            differing += 1
            print("    lost:", cases[i][0], "->", back[i].value)
    print("  legal values:", compared, "compared,", differing, "differing")
    if compared != 6 or differing != 0:
        raise Error("a legal value did not survive: " + String(differing))


def test_refusals() raises:
    """⚠ EACH ONE IS A WAY THE READER WOULD DISAGREE WITH THE WRITER.

    A newline breaks the FILE — the second line has no `=`, so parsing stops at
    a line the author never wrote. The rest break one FIELD, silently, and only
    the field that had them.
    """
    _expect_refusal(
        String("value"), String("tag"), String("two\nlines"),
        String("newline splits the record"),
    )
    _expect_refusal(
        String("value"), String("tag"), String("trailing "),
        String("the reader strips it"),
    )
    _expect_refusal(
        String("value"), String("tag"), String(" leading"),
        String("the reader strips it"),
    )
    _expect_refusal(
        String("value"), String("tag"), String("tabbed\t"),
        String("strip() takes tabs too"),
    )
    _expect_refusal(
        String("value"), String("tag"), String("cr\r"),
        String("a stray \\r is stripped"),
    )
    _expect_refusal(
        String("key"), String(""), String("v"), String("empty key")
    )
    _expect_refusal(
        String("key"), String("a=b"), String("v"),
        String("'=' in a key moves the cut"),
    )
    _expect_refusal(
        String("key"), String("#tag"), String("v"),
        String("reads back as a comment"),
    )
    _expect_refusal(
        String("key"), String("two\nkeys"), String("v"),
        String("newline in a key"),
    )
    _expect_refusal(
        String("key"), String(" spaced"), String("v"),
        String("the reader strips the key too"),
    )
    print("  refusals: 10 of 10 rejected")


def test_a_comment_is_written_but_not_recovered() raises:
    """⚠ SAID OUT LOUD BECAUSE IT IS A TRAP. `comment()` exists for a file
    header a human reads. Nothing in `kv_lines` returns comments, so a program
    that stores state in one loses it on the next read."""
    var w = KvWriter(String("gate"))
    w.comment(String("written by the gate"))
    w.add(String("k"), String("v"))
    var text = w^.done()
    if not text.startswith("# written by the gate\n"):
        raise Error("comment not written: " + text)
    var back = kv_lines(text, String("gate"))
    if len(back) != 1:
        raise Error("comment came back as data: " + String(len(back)))
    print("  comment: written, and correctly absent from the parse")


def main() raises:
    print("=" * 62)
    print("kv_write — the round trip, the repeats, and the refusals")
    print("=" * 62)
    test_every_task_and_family_round_trips()
    test_repeating_keys_keep_their_order()
    test_values_that_must_be_accepted()
    test_refusals()
    test_a_comment_is_written_but_not_recovered()
    print("[PASS] kv_write")
