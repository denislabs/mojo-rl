"""LIBERO's fifty frozen inits, per task, as one of OUR init tables — G12.

    pixi run libero-init-dump          # the Python leg: pickle, remap, verify
    pixi run libero-init-freeze        # this: mask, key, store

`tools/tasks/libero_init_table.py` reads the `.pruned_init` pickles, remaps
each row into our joint order BY NAME and verifies the result by driving both
models through MuJoCo. It deliberately stops there. Everything below is a RULE
this tree already owns exactly once:

    the active mask     `tasks/active.active_mask`
    the family key      `tasks/init_table.family_key`
    the store format    `tasks/init_table.write_init_table`

and a second copy of any of them in a Python tool is the shape
`_a_rule_written_inline_twice_drifts` names — the more so for the KEY, which is
the string every future load is refused against.

## ⚠ WHY THE MASK IS NOT READ OUT OF THE DUMP

`init_table.mojo`'s header: the mask is STORED, not derived, so that an eval
run months from now does not change its answer because someone edited an
`active=` line. That argument is about the READER. The WRITER must derive it,
from the `.task` file, through `active_mask` — which is also what rejects an
`active=` naming a slot the family does not declare.

## THE THREE CHECKS AFTER THE WRITE

1. **round trip** — `load_init_table` under the real key, then every one of the
   `rows x (1 + nq + nv)` floats compared against what went in. A store that
   wrote the right shape and the wrong bytes reads back as a run that works.
2. **the key refuses** — the same file loaded under a wrong `nq` must RAISE.
   The refusal is the feature (`init_table.mojo`), and a feature nothing
   exercises is a feature that has never run.
3. **the rows are not all the same** — the per-task spread of each free slot's
   xy, printed. Fifty copies of one draw would round-trip perfectly and freeze
   an eval into a single episode repeated fifty times; the spread is the number
   that says LIBERO's randomisation survived the conversion.

⚠ 3 IS THE ANTI-VACUITY CHECK AND IT IS NOT DECORATION. Every other assertion
here passes on a table of fifty identical rows.
"""

from std.sys import argv
from std.os import makedirs
from std.os.path import dirname
from std.math import sqrt

from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat,
)
from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, FamilySpec,
)
from noeira.tasks.family import scene_path
from noeira.tasks.active import active_mask
from noeira.tasks.reset import free_slot_addresses
from noeira.tasks.init_table import (
    write_init_table, load_init_table, family_key, INIT_TIME_WORDS,
)
from noeira.tasks.libero_fixtures import dump_path_from_index


comptime TASK_DIR = "noeira/tasks/tasks/"
comptime OUT_DIR = "build/init"


def _floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    var toks = s.split(" ")
    for k in range(len(toks)):
        var t = String(String(toks[k]).strip())
        if t.byte_length() > 0:
            out.append(Float64(t))
    return out^


def _pad(s: String, n: Int) -> String:
    var out = String(s)
    while out.byte_length() < n:
        out += " "
    return out^


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error(
            "usage: libero_init_freeze.mojo <index.txt> [out.h5]   (the index is"
            " written by tools/tasks/libero_init_table.py)"
        )
    var index_path = String(a[1])
    var index_text: String
    try:
        with open(index_path, "r") as fh:
            index_text = fh.read()
    except e:
        # ⚠ SKIPPED, LOUDLY, AND NOT A PASS — the same contract as
        # `libero_demo_success.mojo`. THEIR joint order comes from the recorded
        # `model_file`, so the dump needs the ~6 GB of gitignored demos.
        print("  SKIPPED: no index at", index_path)
        print("  Fetch the suite's demos into references/libero_demos/<suite>/")
        print("  (HF yifengzhu-hf/LIBERO-datasets) and run")
        print("     pixi run libero-init-dump --suite <suite>")
        print("=== SKIPPED (no dumps — this is not a pass) ===")
        return

    var task_names = List[String]()
    var dump_paths = List[String]()
    var lines = index_text.split("\n")
    for i in range(len(lines)):
        var l = String(String(lines[i]).strip())
        if l.byte_length() == 0:
            continue
        var parts = l.split(" ")
        if len(parts) < 3:
            raise Error("malformed index line: " + l)
        task_names.append(String(parts[0]))
        dump_paths.append(dump_path_from_index(index_path, String(parts[2])))
    if len(task_names) == 0:
        raise Error("empty index: " + index_path)

    var suite = String(task_names[0])
    var cut = suite.find("__")
    if cut < 0:
        raise Error("task name without a suite prefix: " + suite)
    var suite_cut = String(suite[byte=0:cut])
    suite = suite_cut^

    var f = load_family("noeira/tasks/families/" + suite + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var dims = dims_from_flat(fmd)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var words = INIT_TIME_WORDS + nq + nv
    var out_path = String(OUT_DIR) + "/" + suite + ".init.h5"
    if len(a) > 2:
        out_path = String(a[2])

    makedirs(dirname(out_path), exist_ok=True)
    print("==============================================================================")
    print("LIBERO's frozen inits ->", out_path)
    print("==============================================================================")
    print("  family:", f.name, "| nq", nq, "nv", nv, "| key", family_key(f.name, nq, nv))
    print("  ", len(task_names), "tasks from", index_path)
    print()

    var state = List[Float64]()
    var task_ix = List[Int32]()
    var mask = List[Float64]()
    var names = List[String]()

    # Free slots, so the spread below can be reported per object rather than
    # per qpos address. `free_slot_addresses` is the same walk `reset_slots`
    # uses, so the ordinals agree with the ones a reset writes.
    #
    # ⚠ IT RETURNS ONE ENTRY PER FAMILY SLOT AND `-1` FOR A STATIC ONE. A
    # fixture's `qadr` of -1 read as an offset lands on `row[0]`, the time
    # word, and prints a spread for a body that has no state at all — which is
    # what the first run of this file did.
    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)

    print("  task                                                rows  mask   "
          + "free-slot xy spread (mm)")
    for ti in range(len(task_names)):
        var t = load_task(TASK_DIR + String(task_names[ti]) + ".task")
        validate_task_against_family(t, f)
        var mk = active_mask(t, f)

        var body: String
        with open(String(dump_paths[ti]), "r") as fh:
            body = fh.read()
        var dl = body.split("\n")
        var hdr = _floats(String(dl[0]))
        if len(hdr) != 2:
            raise Error("dump header is not '<rows> <words>': " + String(dl[0]))
        var n_rows = Int(hdr[0])
        if Int(hdr[1]) != words:
            raise Error(
                "dump '" + String(dump_paths[ti]) + "' has " + String(Int(hdr[1]))
                + "-word rows but this family is 1 + nq " + String(nq) + " + nv "
                + String(nv) + " = " + String(words) + ". The Python leg and this"
                " one disagree about the scene; re-run libero-init-dump."
            )

        # per free slot: min/max of x and y over this task's rows
        var lo = List[Float64]()
        var hi = List[Float64]()
        for _ in range(len(addrs) * 2):
            lo.append(1.0e30)
            hi.append(-1.0e30)

        var seen = 0
        for k in range(1, len(dl)):
            var line = String(String(dl[k]).strip())
            if line.byte_length() == 0:
                continue
            if not line.startswith("ROW "):
                raise Error("unexpected dump line: " + line)
            var row = _floats(String(line[byte=4:]))
            if len(row) != words:
                raise Error(
                    "dump row " + String(seen) + " of '" + String(dump_paths[ti])
                    + "' is " + String(len(row)) + " floats, not " + String(words)
                )
            for w in range(words):
                state.append(row[w])
            for s in range(len(addrs)):
                if addrs[s].qadr < 0:
                    continue
                var qa = INIT_TIME_WORDS + addrs[s].qadr
                for c in range(2):
                    var v = row[qa + c]
                    if v < lo[s * 2 + c]:
                        lo[s * 2 + c] = v
                    if v > hi[s * 2 + c]:
                        hi[s * 2 + c] = v
            task_ix.append(Int32(ti))
            mask.append(mk)
            seen += 1
        if seen != n_rows:
            raise Error(
                "dump '" + String(dump_paths[ti]) + "' declares " + String(n_rows)
                + " rows and carries " + String(seen)
            )
        names.append(String(t.language))

        var spread = String("")
        for s in range(len(addrs)):
            if addrs[s].qadr < 0:
                continue
            spread += (
                " " + String(Int((hi[s * 2] - lo[s * 2]) * 1000.0))
                + "x" + String(Int((hi[s * 2 + 1] - lo[s * 2 + 1]) * 1000.0))
            )
        print(
            "  " + _pad(String(t.name), 50) + " " + _pad(String(seen), 5)
            + " " + _pad(String(Int(mk)), 6) + spread
        )

    var n = len(task_ix)
    write_init_table(
        out_path, f.name, nq, nv, state, task_ix, mask, names,
        seed=0, source_commit=String("libero .pruned_init via ") + index_path,
    )

    # 1. round trip
    var tbl = load_init_table(out_path, f.name, nq, nv)
    if tbl.n_rows() != n:
        raise Error(
            "round trip: wrote " + String(n) + " rows and read back "
            + String(tbl.n_rows())
        )
    var qpos = List[Float64](length=nq, fill=0.0)
    var qvel = List[Float64](length=nv, fill=0.0)
    var worst = 0.0
    for i in range(n):
        tbl.apply(i, qpos, qvel)
        var b = i * words + INIT_TIME_WORDS
        for k in range(nq):
            var d = abs(qpos[k] - state[b + k])
            if d > worst:
                worst = d
        for k in range(nv):
            var d = abs(qvel[k] - state[b + nq + k])
            if d > worst:
                worst = d
        if Int(tbl.task_index[i]) != Int(task_ix[i]):
            raise Error("round trip: row " + String(i) + " changed task index")
        if tbl.mask[i] != mask[i]:
            raise Error("round trip: row " + String(i) + " changed mask")
        _ = tbl.task_label(i)
    if worst != 0.0:
        raise Error(
            "round trip: worst |written - read| is " + String(worst)
            + ", and float64 through HDF5 is EXACT. A store that wrote the right"
            " shape and the wrong bytes reads back as a run that works."
        )

    # 2. the key refuses
    var refused = False
    try:
        var _bad = load_init_table(out_path, f.name, nq + 1, nv)
    except e:
        refused = True
    if not refused:
        raise Error(
            "the family key did NOT refuse a table loaded under nq " + String(nq + 1)
            + ". The refusal is what stops a table from another family loading as"
            " coordinates that mean something else here."
        )

    # 3. the rows are not all the same
    var moved = 0
    var n_free = 0
    for s in range(len(addrs)):
        if addrs[s].qadr < 0:
            continue
        n_free += 1
        var qa = INIT_TIME_WORDS + addrs[s].qadr
        var mn = 1.0e30
        var mx = -1.0e30
        for i in range(n):
            var v = state[i * words + qa]
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        if mx - mn > 1.0e-4:
            moved += 1
    if moved == 0:
        raise Error(
            "no free slot's x varies by more than 0.1 mm across " + String(n)
            + " rows. Every assertion above passes on fifty copies of one draw;"
            " this is the one that does not."
        )

    print()
    print("  round trip: ", n, "rows,", nq + nv, "floats each, exact")
    print("  the key refuses a table loaded under the wrong nq")
    print("  ", moved, "of", n_free, "free slots vary across the table")
    print("  wrote", out_path)
    print()
    print("=== " + String(n) + " frozen inits, " + String(len(task_names))
          + " tasks, keyed " + family_key(f.name, nq, nv) + " ===")
