"""LIBERO's demonstrations into a `TrajectoryStore`, and the gate on the states.

    pixi run libero-demo-import                          # the whole suite, with images
    pixi run libero-demo-import --no-images --demos 2     # a smoke run, 1/30 the size
    pixi run libero-demo-import libero_goal --out build/demos/libero_goal.h5

The importer is `noeira/envs/libero/demos.mojo` — native, no h5py and no
MuJoCo. This file supplies the task list and then CHECKS the one column only we
can produce.

## ⚠⚠ THE GATE: OUR NATIVE REMAP AGAINST THE PYTHON ONE, ON REAL DATA

`state` is their `(time, qpos, qvel)` rewritten into our joint addresses. Nothing
about the store's shape can tell you it is right — a wrong remap writes the same
number of the same-typed floats, and the file opens.

So the rows are compared against the dumps `tools/libero/libero_demo_success.py`
already writes: the trailing window of every demo, in OUR joint order, produced
by a COMPLETELY DIFFERENT path — Python, `mujoco.MjModel` on the recorded
`model_file`, and `libero_demo_common.build_remap`. That Python path is the one
`libero_init_table.py` verified against MuJoCo by driving both models and
comparing body poses (worst 2.18e-07 m, which is the recorded XML's own rounding).
So this is a two-implementation cross-check whose reference is itself gated —
not `_a_gate_that_shares_its_reference_implementation_is_blind`.

## ⚠⚠ THE TOLERANCE IS ONE ULP, AND THE REASON IS OUR OWN DECIMAL PARSER

A remap is a PERMUTATION and permutations do not round, so the first version of
this gate demanded an exact zero. It failed at 3.85e-34 on three of ten tasks —
which is not a wrong address (that differs by O(1), thirty orders away) but the
TRANSPORT: the dump is text, and `Float64(String)` does not round it correctly.

MEASURED, by parsing every small token of three dumps in both languages and
comparing bit patterns: **16 of 67 875 strings differ, all by exactly one ULP,
all of them negative with tiny exponents** (`-3.366416907697219e-18` reads
`0xbc4f0cba28dee6b7` here against CPython's `0xbc4f0cba28dee6b8`). CPython's
`float()` is correctly rounded; ours is one ULP low. That is a fact about the
stdlib and not about this store, and it is recorded rather than tuned around.

So the comparison is RELATIVE and in ULPs: `|a - b| <= 2 * 2^-52 * max(|a|,|b|)`,
with an exact zero required against an exact zero. A wrong address still fails
by thirty orders of magnitude, and the run prints how many values differed at
all so the transport's cost stays visible instead of hiding inside a tolerance.

⚠ AND THE COMPARISON COUNT IS PRINTED. "0 mismatches" over zero comparisons is
this tree's default failure mode; the dumps are gitignored, so the run says how
many values it actually compared and refuses if a task contributed none.

## ⚠ NO `RunContext`, DELIBERATELY

L6's plan says to run the importer under one "so the store is an artifact".
`core/run.mojo`'s own header scopes `RunContext` to "one execution of a learning
algorithm" and says in as many words that a probe or a diagnostic is not a run;
`examples/so101/act_so101_import_dataset.mojo` — the importer this one is modelled
on — uses none either. Wiring one in here would contradict both the precedent and
the struct's stated contract, so the store is an ordinary output with its
provenance in its own manifest (`source_commit` names the remap table). Recorded
here rather than silently diverged from.
"""

from std.os import listdir
from std.sys import argv

from std.memory.alloc import unsafe_alloc

from noeira.io.hdf5.reader import H5File
from noeira.envs.libero.demos import (
    import_libero_demos, CAM_H, CAM_W, N_CAMS, CAM_ELEMS, COL_STATE, COL_IMAGES,
    COL_QPOS, COL_JOINTS, COL_GRIPPER, QPOS_PROPRIO, QPOS_WORDS, JOINT_DIM,
    GRIPPER_DIM,
)
from noeira.data.store import TrajectoryStore
from noeira.tasks.spec import load_task
from noeira.envs.libero.state_remap import load_state_remap
from noeira.envs.libero.fixtures import dump_path_from_index


comptime TASK_DIR = "noeira/envs/libero/tasks/"
comptime DEMO_DIR = "references/libero_demos/"

comptime MAX_ULPS: Float64 = 2.0
"""What the dump's decimal round trip costs — see the header. One ULP is what
`Float64(String)` was measured to lose on 16 of 67 875 tokens; two is that with
a factor of headroom, and still thirty orders of magnitude below a swapped
address."""

comptime F64_EPS: Float64 = 2.220446049250313e-16
"""`2^-52`, one ULP relative at the top of a binade."""


def _ulps(a: Float64, b: Float64) -> Float64:
    """`|a - b|` measured in ULPs of the larger operand.

    ⚠ AN EXACT ZERO MUST MEET AN EXACT ZERO. Scaling by `max(|a|,|b|)` makes
    every comparison against zero pass for free, and a remap that dropped a
    joint writes zeros — the one failure this gate most needs to catch.
    """
    var m = abs(a) if abs(a) > abs(b) else abs(b)
    if m == 0.0:
        return 0.0
    return abs(a - b) / (m * F64_EPS)


def _floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    var toks = s.split(" ")
    for k in range(len(toks)):
        var t = String(String(toks[k]).strip())
        if t.byte_length() > 0:
            out.append(Float64(t))
    return out^


def task_stems(suite: String) raises -> List[String]:
    """Every `<suite>__*.task` on disk, sorted, with the suite prefix stripped.

    ⚠ THE SAME ORDER `libero_eval` AND `libero_init_freeze` USE, and for the
    same reason: `task_index` is an index into it. A demo file whose stem is not
    among them RAISES below rather than being imported under a guessed index.
    """
    var out = List[String]()
    var want = suite + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = want.byte_length() : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def main() raises:
    var args = argv()
    var suite = String("libero_goal")
    var out_path = String("")
    var images = True
    var max_demos = 0
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--no-images":
            images = False
        elif s == "--out" and i + 1 < len(args):
            out_path = String(args[i + 1])
            i += 1
        elif s == "--demos" and i + 1 < len(args):
            max_demos = Int(String(args[i + 1]))
            i += 1
        elif not s.startswith("--"):
            suite = s
        else:
            raise Error("libero demo import: unknown argument '" + s + "'")
        i += 1
    if out_path.byte_length() == 0:
        out_path = String("build/demos/") + suite + (
            ".h5" if images else ".lowdim.h5"
        )

    print("=" * 78)
    print("LIBERO demonstrations ->", out_path)
    print("=" * 78)

    var stems = task_stems(suite)
    if len(stems) == 0:
        raise Error(
            "no `" + suite + "__*.task` under " + TASK_DIR
            + " — generate them with `pixi run libero-family " + suite + "`"
        )
    var remap = load_state_remap(suite)
    print("  remap :", len(remap.joints), "joints, nq", remap.nq, "nv", remap.nv,
          "(noeira/envs/libero/tables/state_remap_" + suite + ".kv)")

    var names = List[String]()
    var texts = List[String]()
    var files = List[String]()
    var missing = 0
    for k in range(len(stems)):
        var path = String(DEMO_DIR) + suite + "/" + String(stems[k]) + "_demo.hdf5"
        var t = load_task(String(TASK_DIR) + suite + "__" + String(stems[k]) + ".task")
        try:
            with open(path, "r") as _probe:
                pass
        except e:
            missing += 1
            continue
        names.append(suite + "__" + String(stems[k]))
        texts.append(String(t.language))
        files.append(path^)
    if len(files) == 0:
        # ⚠ SKIPPED, LOUDLY, AND NOT A PASS — the demos are ~6 GB and gitignored.
        print("  SKIPPED: no demo files under", String(DEMO_DIR) + suite)
        print("  Fetch them (HF yifengzhu-hf/LIBERO-datasets) into")
        print("   ", String(DEMO_DIR) + suite + "/<task>_demo.hdf5")
        print("=== SKIPPED (no demonstrations — this is not a pass) ===")
        return
    if missing > 0:
        # ⚠ A PARTIAL SUITE IS ALLOWED AND SAID OUT LOUD. `task_index` stays the
        # index into the FULL sorted list, so a store built from eight of ten
        # tasks is still comparable with one built from all ten.
        print("  ⚠", missing, "of", len(stems),
              "tasks have no demo file; their task_index is skipped, not shifted")

    print("  images:", "2 x " + String(CAM_H) + "x" + String(CAM_W)
          + " CHW per row" if images else "NONE (--no-images)")
    if max_demos > 0:
        print("  demos :  capped at", max_demos, "per task")
    print()

    # ⚠ THE INDEX MUST BE THE FULL LIST'S. `import_libero_demos` writes
    # `task_index = i` for `files[i]`, so a suite with a missing task would
    # renumber. Pad the lists back out to the full stem order.
    var full_names = List[String]()
    var full_texts = List[String]()
    var full_files = List[String]()
    for k in range(len(stems)):
        var path = String(DEMO_DIR) + suite + "/" + String(stems[k]) + "_demo.hdf5"
        var present = False
        for j in range(len(files)):
            if files[j] == path:
                present = True
        if not present:
            continue
        full_names.append(suite + "__" + String(stems[k]))
        var t = load_task(String(TASK_DIR) + suite + "__" + String(stems[k]) + ".task")
        full_texts.append(String(t.language))
        full_files.append(path^)

    var rep = import_libero_demos(
        String(DEMO_DIR) + suite, suite, out_path,
        full_names, full_texts, full_files,
        images=images, max_demos=max_demos,
    )
    print()
    print("  wrote", out_path, "—", rep.n_episodes, "episodes,", rep.n_rows,
          "rows,", rep.n_tasks, "tasks")

    # ── the gate: our `state` column against the Python remap's dumps ──────
    var st = TrajectoryStore(out_path)
    if st.n_rows() != rep.n_rows:
        raise Error(
            "the store reports " + String(st.n_rows()) + " rows and the import "
            + String(rep.n_rows)
        )
    var state_dim = remap.nq + remap.nv
    var spec = st.column(String(COL_STATE))
    if spec.row_dim() != state_dim:
        raise Error(
            "the `state` column is " + String(spec.row_dim()) + " wide, not nq "
            + String(remap.nq) + " + nv " + String(remap.nv) + " = "
            + String(state_dim)
        )

    # ── `qpos` is `joint_states` ++ `gripper_states` ++ their one-step
    # difference (zero on an episode's first row) ++ onehot(task), row
    # for row. Four columns read back and compared: the concatenation is
    # trivial, the thing that can go wrong is a ROW misalignment between the
    # appends, and that is what an exact per-row comparison over the whole
    # store catches. The one-hot is checked against `task_index` and must
    # light EXACTLY one word.
    var QPOS_DIM = QPOS_WORDS + len(full_names)
    var q_all = st.load_column[DType.float32](String(COL_QPOS))
    var ep_first = List[Bool](length=rep.n_rows, fill=False)
    for e in range(st.n_episodes()):
        ep_first[st.episodes.start_of(e)] = True
    var dq_bad = 0
    var dq_moving = 0
    var j_all = st.load_column[DType.float32](String(COL_JOINTS))
    var g_all = st.load_column[DType.float32](String(COL_GRIPPER))
    var t_all = st.load_column[DType.int32](String("task_index"))
    if len(q_all) != rep.n_rows * QPOS_DIM:
        raise Error(
            "the `qpos` column is " + String(len(q_all)) + " words for "
            + String(rep.n_rows) + " rows, not " + String(QPOS_DIM) + " per row"
        )
    var q_bad = 0
    var q_moving = 0
    var oh_bad = 0
    for r in range(rep.n_rows):
        for k in range(JOINT_DIM):
            if q_all[r * QPOS_DIM + k] != j_all[r * JOINT_DIM + k]:
                q_bad += 1
        for k in range(GRIPPER_DIM):
            if q_all[r * QPOS_DIM + JOINT_DIM + k] != g_all[r * GRIPPER_DIM + k]:
                q_bad += 1
        for k in range(QPOS_PROPRIO):
            var want = (
                Scalar[DType.float32](0)
                if ep_first[r] else
                q_all[r * QPOS_DIM + k] - q_all[(r - 1) * QPOS_DIM + k]
            )
            var got = q_all[r * QPOS_DIM + QPOS_PROPRIO + k]
            if got != want:
                dq_bad += 1
            if got != 0.0:
                dq_moving += 1
        var lit = 0
        for k in range(len(full_names)):
            var v = q_all[r * QPOS_DIM + QPOS_WORDS + k]
            if v == 1.0:
                lit += 1
                if k != Int(t_all[r]):
                    oh_bad += 1
            elif v != 0.0:
                oh_bad += 1
        if lit != 1:
            oh_bad += 1
        if r > 0 and q_all[r * QPOS_DIM] != q_all[(r - 1) * QPOS_DIM]:
            q_moving += 1
    print("  qpos  :", rep.n_rows, "rows == joint_states ++ gripper_states ++"
          " diff ++ onehot(task);", q_bad, "proprio words differ;", dq_bad,
          "difference words differ;", oh_bad, "one-hot faults;", q_moving,
          "rows where joint 1 moved;", dq_moving, "non-zero difference words")
    if dq_bad > 0:
        raise Error("the one-step difference in `qpos` is wrong on "
                    + String(dq_bad) + " words (row r - row r-1 of the same"
                    " episode, zero on the first row)")
    if dq_moving == 0:
        raise Error("the one-step difference is zero everywhere — checked nothing")
    if oh_bad > 0:
        raise Error("the task one-hot in `qpos` is wrong on " + String(oh_bad)
                    + " counts — it must light exactly the row's task_index")
    if q_bad > 0:
        raise Error(
            "the `qpos` column is not joint_states ++ gripper_states: "
            + String(q_bad) + " words differ"
        )
    if q_moving == 0:
        raise Error("qpos never moves — a constant column checked nothing")

    # ── the images: are the three planes the three CHANNELS? ──────────────
    #
    # ⚠⚠ 98 304 OF EACH ROW'S 98 616 BYTES ARE PIXELS, AND A TRANSPOSE THAT IS
    # WRONG STILL WRITES ALL OF THEM. Their datasets are HWC and the store is
    # CHW (`lerobot_v3_to_store.py`'s convention, and what every consumer here
    # expects); reading HWC as if it were CHW yields three "planes" that are
    # interleaved strips of the top third of the image, which renders as
    # something and trains into nothing.
    #
    # ⚠ CHECKED BY PLANE MEANS, NOT BY RE-INDEXING. Spelling the same
    # permutation a second time and comparing is
    # `_a_gate_that_shares_its_reference_implementation_is_blind`; a mean over
    # each stored plane against a mean over each SOURCE CHANNEL is a different
    # arithmetic path, and any plane scramble moves it. MEASURED on
    # `turn_on_the_stove` demo 0: the true channel means are (127.1, 118.0,
    # 108.1) and reading the HWC buffer as CHW gives (149.0, 103.0, 101.3) — a
    # 21.8 shift against a tolerance of exactly zero.
    if images:
        var im_col = st.column(String(COL_IMAGES))
        if im_col.row_dim() != N_CAMS * CAM_ELEMS:
            raise Error(
                "the `images` column is " + String(im_col.row_dim())
                + " wide, not " + String(N_CAMS) + " x " + String(CAM_ELEMS)
            )
        var worst_mean = 0.0
        var checked = 0
        var flip_bad = 0
        var flip_distinct = 0
        var ep_at = 0
        for ti in range(len(full_names)):
            var src = H5File(String(full_files[ti]))
            var ds = src.open_dataset(String("data/demo_0/obs/agentview_rgb"))
            var T = Int(ds.dims[0])
            var raw = unsafe_alloc[Scalar[DType.uint8]](
                T * CAM_ELEMS
            ).as_unsafe_any_origin()
            ds.read_all[DType.uint8](raw)
            var row0 = st.episodes.start_of(ep_at)
            var stored = unsafe_alloc[Scalar[DType.uint8]](
                N_CAMS * CAM_ELEMS
            ).as_unsafe_any_origin()
            st.read_range[DType.uint8](
                String(COL_IMAGES), row0, row0 + 1, stored
            )
            for c in range(3):
                var a = 0.0
                var b = 0.0
                for y in range(CAM_H):
                    for x in range(CAM_W):
                        # source: HWC, row 0
                        a += Float64(Int(raw[unsafe_offset = (y * CAM_W + x) * 3 + c]))
                        # store: CHW, camera 0
                        b += Float64(Int(stored[
                            unsafe_offset = c * CAM_H * CAM_W + y * CAM_W + x
                        ]))
                var n = Float64(CAM_H * CAM_W)
                var e = abs(a / n - b / n)
                if e > worst_mean:
                    worst_mean = e
                checked += 1
            # ⚠⚠ THE ORIENTATION, THROUGH A DIFFERENT INDEX PATH. A plane mean
            # is the same whichever way up the rows are, so it cannot see the
            # flip the importer applies (row 0 of the STORE is the top of the
            # picture, row 0 of the RECORDING is the bottom). The store's first
            # row of plane 0 must be the source's LAST row, channel 0 — and
            # the source's first and last rows must DIFFER, or the comparison
            # would pass on an importer that stopped flipping.
            for x in range(CAM_W):
                var src_last = Int(raw[unsafe_offset = ((CAM_H - 1) * CAM_W + x) * 3])
                var src_first = Int(raw[unsafe_offset = x * 3])
                var sto_first = Int(stored[unsafe_offset = x])
                if src_last != sto_first:
                    flip_bad += 1
                if src_last != src_first:
                    flip_distinct += 1
            raw.unsafe_free()
            stored.unsafe_free()
            ep_at += rep.eps_per_task[ti]
        if checked == 0:
            raise Error("no image plane was checked")
        print("  images:", checked, "channel planes checked against their HWC"
              " source; worst mean difference", worst_mean)
        print("  rows  : store row 0 vs source row " + String(CAM_H - 1) + ":",
              flip_bad, "pixels differ;", flip_distinct,
              "pixels where the source's top and bottom rows differ")
        if worst_mean != 0.0:
            raise Error(
                "a stored plane's mean differs from its source CHANNEL's by "
                + String(worst_mean) + ". The HWC -> CHW transpose in"
                " noeira/envs/libero/demos.mojo is wrong; the planes are not"
                " the channels."
            )
        if flip_bad > 0:
            raise Error(
                "the store's first image row is not the recording's last: "
                + String(flip_bad) + " pixels differ. The importer must write"
                " the picture top row first (see libero_demos.mojo's header)."
            )
        if flip_distinct == 0:
            raise Error(
                "the source's top and bottom rows are identical on every task"
                " checked — the orientation check is vacuous"
            )

    var index_path = String(DEMO_DIR) + "_dumps/" + suite + "/index.txt"
    var index_text: String
    try:
        with open(index_path, "r") as fh:
            index_text = fh.read()
    except e:
        print()
        print("  ⚠ NOT GATED: no dump index at", index_path)
        print("    The `state` column is the one thing this store carries that")
        print("    nothing else could produce, and it is UNCHECKED without the")
        print("    Python remap to compare against. Run:")
        print("      pixi run libero-demo-dump --suite", suite)
        print("    then this again.")
        print("=== IMPORTED, NOT GATED (this is not a pass) ===")
        return

    var states = st.load_column[DType.float64](String(COL_STATE))


    var dump_of = List[String]()
    var dump_task = List[String]()
    var lines = index_text.split("\n")
    for k in range(len(lines)):
        var l = String(String(lines[k]).strip())
        if l.byte_length() == 0:
            continue
        var parts = l.split(" ")
        dump_task.append(String(parts[0]))
        dump_of.append(dump_path_from_index(index_path, String(parts[1])))

    print()
    print("  task                                                demos  values"
          + "  differing  worst ULPs")
    var total_cmp = 0
    var total_diff = 0
    var worst_all = 0.0
    var ep_base = 0
    for ti in range(len(full_names)):
        var which = -1
        for k in range(len(dump_task)):
            if dump_task[k] == full_names[ti]:
                which = k
        if which < 0:
            raise Error(
                "no dump for '" + full_names[ti] + "' in " + index_path
                + " — re-run libero-demo-dump for the whole suite; a task"
                " compared against nothing passes."
            )
        var body: String
        with open(String(dump_of[which]), "r") as fh:
            body = fh.read()
        var dl = body.split("\n")
        var n_demos_dump = Int(String(String(dl[0]).strip()))

        var demo = -1
        var win = 0
        var total_states = 0
        var seen_in_demo = 0
        var q_row = List[Float64]()
        var n_cmp = 0
        var n_diff = 0
        var worst = 0.0
        var k = 1
        while k < len(dl):
            var line = String(String(dl[k]).strip())
            k += 1
            if line.byte_length() == 0:
                continue
            if line.startswith("DEMO "):
                var p = line.split(" ")
                demo += 1
                win = Int(String(p[2]))
                total_states = Int(String(p[3]))
                seen_in_demo = 0
                continue
            if line.startswith("FIX "):
                continue
            if demo < 0:
                raise Error("dump line before any DEMO: " + line)
            # ⚠ THE CAP IS PER TASK, NOT GLOBAL. `--demos N` gives every task
            # N episodes, so the dump's 50 must be cut against THIS task's
            # count; against the store's total, demo 5 of task 0 would be read
            # as episode 5, which belongs to task 2.
            if demo >= rep.eps_per_task[ti]:
                break
            var vals = _floats(String(line[byte=5:]))
            if line.startswith("QPOS "):
                q_row = vals^
                continue
            if not line.startswith("QVEL "):
                raise Error("unexpected dump line: " + line)
            # this pair is state index (total_states - win + seen_in_demo)
            var idx = total_states - win + seen_in_demo
            seen_in_demo += 1
            var row0 = st.episodes.start_of(ep_base + demo)
            var T = st.episodes.length_of(ep_base + demo)
            if idx < 0 or idx >= T:
                raise Error(
                    full_names[ti] + " demo " + String(demo) + ": the dump's"
                    " state " + String(idx) + " is outside the store's "
                    + String(T) + " rows"
                )
            var b = (row0 + idx) * state_dim
            for c in range(remap.nq):
                var u = _ulps(Float64(states[b + c]), q_row[c])
                if u > worst:
                    worst = u
                if u > 0.0:
                    n_diff += 1
                n_cmp += 1
            for c in range(remap.nv):
                var u = _ulps(Float64(states[b + remap.nq + c]), vals[c])
                if u > worst:
                    worst = u
                if u > 0.0:
                    n_diff += 1
                n_cmp += 1
        var n_here = rep.eps_per_task[ti]
        if n_here > n_demos_dump:
            raise Error(
                full_names[ti] + ": the store has " + String(n_here)
                + " episodes and the dump only " + String(n_demos_dump)
                + " demos — the dump is stale; re-run libero-demo-dump."
            )
        if n_cmp == 0:
            raise Error(
                full_names[ti] + ": zero values compared against its dump."
                " A task whose gate compares nothing reports a perfect result."
            )
        if worst > worst_all:
            worst_all = worst
        total_cmp += n_cmp
        total_diff += n_diff
        var pad = String(full_names[ti])
        while pad.byte_length() < 52:
            pad += " "
        print("  " + pad + String(n_here) + "      " + String(n_cmp)
              + "     " + String(n_diff) + "        " + String(worst))
        ep_base += n_here

    print()
    print("  ", total_cmp, "state values compared against the Python remap's"
          " dumps;", total_diff, "differ at all, worst", worst_all, "ULPs")
    if worst_all > MAX_ULPS:
        raise Error(
            "our native remap and the Python one disagree by " + String(worst_all)
            + " ULPs, over the " + String(MAX_ULPS) + " our decimal parser costs"
            " (see the header). A permutation does not round, so a residual"
            " this large is a WRONG ADDRESS. Compare"
            " noeira/envs/libero/tables/state_remap_" + suite + ".kv against"
            " tools/libero/libero_demo_common.build_remap."
        )
    print()
    print("=== " + String(rep.n_rows) + " rows, " + String(rep.n_episodes)
          + " episodes; the `state` column matches the Python remap to "
          + String(worst_all) + " ULPs (" + String(total_diff) + " of "
          + String(total_cmp) + " values differ at all) ===")
