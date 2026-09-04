"""The pre-tokenised instruction table. No network, no dump, no weights.

The table is checked in, so this gate reads what deployment reads. What it
pins:

  1. The recording's one task parses, and `n_lang()` is **6** — 128 image
     tokens + 6 language + 1 state = P = 135, which is what every mask in the
     prefill is built from.
  2. **The trailing newline survived.** The last id must be 198 (`'\\n'`), the
     token `NewLineTaskProcessorStep` adds. Five ids would be a working policy
     attending to a truncated instruction.
  3. A declared count that disagrees with the ids raises — that is the field a
     hand edit gets wrong.
  4. Two tasks of different lengths raise rather than picking one, because a
     comptime `N_LANG` cannot represent both.

Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_tasks.mojo
"""

from std.testing import assert_true, assert_equal

from mojo_rl.deep_agents.smolvla.tasks import TaskTokens
from mojo_rl.deep_agents.smolvla.attn_mask import smolvla_ar

comptime TABLE = "tools/vla/smolvla_tasks_record-test_20260828_092736.tsv"
comptime NEWLINE_ID = 198
comptime N_CAM = 2
comptime IMG_TOK = 64


def main() raises:
    print("=" * 70)
    print("SmolVLA pre-tokenised instructions")
    print("=" * 70)

    var t = TaskTokens(String(TABLE))
    print("  tasks:", t.size())
    assert_true(t.size() > 0, "the task table is empty")

    var n = t.n_lang()
    var ids = t.for_index(0)
    print("  [1] task 0:", t.texts[0])
    print("      ", n, "tokens:", ids[0], ids[1], ids[2], ids[3], ids[4],
          ids[5])
    assert_equal(n, 6, "expected 6 tokens for 'Grab the green cube\\n'")
    assert_equal(len(ids), n, "for_index disagrees with n_lang")

    # [2] the newline the processor appends
    assert_equal(
        ids[n - 1], NEWLINE_ID,
        "the last id is not the newline — NewLineTaskProcessorStep's '\\n' was"
        " dropped, so every index after the language block shifts by one",
    )
    print("  [2] last id is", NEWLINE_ID, "— the appended newline survived")

    # [3] this is what P is, and the mask is built from the same numbers
    var P = N_CAM * IMG_TOK + n + 1
    var ar = smolvla_ar(N_CAM * IMG_TOK, n, 1, 0)
    print("  [3] P =", N_CAM, "x", IMG_TOK, "+", n, "+ 1 =", P)
    assert_equal(
        len(ar), P,
        "smolvla_ar and the token count disagree on P — the prefill mask would"
        " not match the prefix",
    )

    # [4] a count that disagrees with the ids must raise
    var bad = String("/tmp/vla_bad_tasks.tsv")
    with open(bad, "w") as f:
        f.write(String("0\t6\t55,4183,260\tshort\n"))
    var raised = False
    try:
        var _t2 = TaskTokens(bad)
    except:
        raised = True
    assert_true(raised, "a declared count that disagrees with the ids must raise")
    print("  [4] a miscounted row raises")

    # [5] two different lengths must raise rather than pick one
    var mixed = String("/tmp/vla_mixed_tasks.tsv")
    with open(mixed, "w") as f:
        f.write(String("0\t3\t1,2,3\tone\n1\t4\t1,2,3,4\ttwo\n"))
    var t3 = TaskTokens(mixed)
    raised = False
    try:
        var _n = t3.n_lang()
    except:
        raised = True
    assert_true(
        raised,
        "two instruction lengths must raise — a comptime N_LANG cannot hold"
        " both, and picking one silently is the multi-task trap",
    )
    print("  [5] two lengths raise instead of picking one")

    print("PASSED — the instruction table is what the prefix will embed")
