"""The task table round-trips BYTE-EXACT — the VLA binding gate's foundation.

## WHY BYTE-EXACT IS THE REQUIREMENT

A consumer tokenises the instruction, and the tokenisation is sensitive to the
exact bytes AND to a step after: lerobot's `NewLineTaskProcessorStep` appends
"\\n" before tokenising, so "Grab the green cube" becomes six ids ending in 198,
not five. The point of carrying the text in the store is an equality gate —
`store.task_text[k] == tsv.task[k]` — that turns a drifted token table into an
error instead of a policy quietly attending to the wrong sentence.

⚠⚠ A STORE THAT NORMALISED THE TEXT WOULD DEFEAT THAT GATE IN BOTH DIRECTIONS:
it would fail on a cosmetic difference, or PASS while the ids came from
different bytes than the store records. The second is the one that matters.

## ⚠ THE STRINGS HERE ARE HOSTILE ON PURPOSE

The manifest is `key=value` LINES and `parse_manifest` calls `.strip()` on
every value, so the two things that break it are a LINE BREAK and EDGE
WHITESPACE. A gate that only round-tripped "Grab the green cube" would pass on
an escaper that handled neither. Every case below is a way the format could
eat a byte.

Run: pixi run mojo run -I . tests/data/test_manifest_tasks.mojo
"""

from mojo_rl.data.manifest import (
    Manifest, TaskEntry, parse_manifest,
    escape_task_text, unescape_task_text,
)
from mojo_rl.data.column import ColumnSpec


def _hostile() -> List[String]:
    var o = List[String]()
    o.append(String("Grab the green cube"))       # the ordinary one
    o.append(String(" leading space"))            # strip() would eat it
    o.append(String("trailing space "))           # strip() would eat it
    o.append(String("  both  "))
    o.append(String("two\nlines"))                # would SPLIT the record
    o.append(String("tab\there"))                 # the field separator
    o.append(String("carriage\rreturn"))
    o.append(String("a=b=c"))                     # the key/value separator
    o.append(String('quote " inside'))            # the quoting character
    o.append(String("back\\slash"))               # the escape character
    o.append(String("\\\"tricky\\\""))            # escape + quote together
    o.append(String(""))                          # empty
    o.append(String("ünïcøde ✓ 日本語"))            # multi-byte
    o.append(String("ends with backslash\\"))
    return o^


def main() raises:
    print("=== manifest task table — byte-exact round trip ===")
    var cases = _hostile()
    var bad = 0

    # ── 1. escape -> unescape is the identity ─────────────────────────────
    for i in range(len(cases)):
        var e = escape_task_text(cases[i])
        var back = unescape_task_text(e)
        if back != cases[i]:
            print("  FAIL escape/unescape:", i, "->", e)
            bad += 1
    print("  ok: escape->unescape identity on", len(cases), "hostile strings")

    # ── 2. through the WHOLE manifest, which is where strip() lives ───────
    var m = Manifest()
    m.env_id = String("lerobot/test")
    m.n_rows = 10
    m.n_episodes = 2
    m.seed = 7
    m.source_commit = String("abc1234")
    var shp = List[Int]()
    shp.append(6)
    m.columns.append(ColumnSpec(String("qpos"), DType.float32, shp^))
    for i in range(len(cases)):
        m.tasks.append(TaskEntry(i, cases[i]))

    var text = m.encode()
    var m2 = parse_manifest(text)
    if len(m2.tasks) != len(cases):
        print("  FAIL: parsed", len(m2.tasks), "tasks, wrote", len(cases))
        bad += 1
    for i in range(len(cases)):
        var got = m2.task_text(i)
        if got != cases[i]:
            print("  FAIL manifest round trip at", i)
            bad += 1
    print("  ok: every task survives encode->parse through the manifest")

    # ⚠ AND THE MANIFEST ITSELF IS STILL A FIXED POINT. A task line that broke
    # the format would show up here even if `task_text` happened to agree.
    if m2.encode() != text:
        print("  FAIL: encode->parse->encode is not a fixed point")
        bad += 1
    else:
        print("  ok: encode->parse->encode is a fixed point")

    # ⚠ THE COMMON CASE STAYS READABLE — the whole reason for quoting rather
    # than hex. If this ever fails, the format got harder to eyeball and the
    # `bytes(f['__manifest__'][:]).decode()` promise in manifest.mojo is worth
    # less than it claims.
    var plain = Manifest()
    plain.tasks.append(TaskEntry(0, String("Grab the green cube")))
    if plain.encode().find('task=0\t"Grab the green cube"') < 0:
        print("  FAIL: the ordinary case is not readable in the manifest")
        bad += 1
    else:
        print('  ok: reads as  task=0\t"Grab the green cube"')

    # ── 3. refusals ───────────────────────────────────────────────────────
    var raised = 0
    try:
        _ = unescape_task_text(String('"bad \\q escape"'))
    except e:
        raised += 1
    try:
        _ = unescape_task_text(String("not quoted"))
    except e:
        raised += 1
    try:
        _ = m2.task_text(999)
    except e:
        raised += 1
    if raised != 3:
        print("  FAIL: only", raised, "of 3 refusals fired")
        bad += 1
    else:
        print("  ok: unknown escape, unquoted text and unknown index all RAISE")

    print()
    if bad != 0:
        raise Error("manifest tasks: " + String(bad) + " failure(s)")
    print("=== PASS ===")
