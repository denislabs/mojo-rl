"""FROZEN INIT STATES — freeze, load, refuse. P4a.

`TASK_LAYER_PLAN.md` §6.2. A success rate over states the run sampled for
itself is not comparable with anything. This gates the mechanism that makes it
comparable.

## WHAT IT ASSERTS

1. **THE ROWS SURVIVE THE DISK BIT-EXACTLY.** `float64` in, `float64` out, ==
   not `almost_equal`: a frozen init that came back to 1e-16 would be a
   DIFFERENT episode, and the whole point of the table is that it is not.

2. **THE TABLE IS WHAT DETERMINES THE EPISODE.** Applying row `i` reproduces
   the exact `qpos`/`qvel` the freeze pass sampled, from a DIFFERENT starting
   vector — so the state comes from the table and not from what happened to be
   in the buffer.

3. ⚠⚠ **THE CONTROL: A DIFFERENT SEED IS A DIFFERENT TABLE.** Checks 1 and 2
   pass on a table whose rows are all identical, or on a freeze that ignored
   its seed. A second table frozen at another seed must DIFFER — otherwise the
   reproducibility being gated is the reproducibility of a constant.

4. **A MISMATCHED FAMILY IS REFUSED**, not adapted. Three ways: the wrong
   family name, the wrong `nq`, the wrong `nv`. Each must raise.

5. **THE PER-TASK BREAKDOWN IS RESOLVABLE.** Every row's `task_index` resolves
   to its instruction through the manifest's task table, byte-exactly — that
   is what "the monitor breaks it down per task" needs, and the table is
   self-describing so it needs no second file.

⚠ THIS GATE DOES NOT STEP THE PHYSICS. It gates the FREEZE/LOAD boundary; that
two eval runs over one table report the same success rate is
`examples/tasks/task_eval_frozen.mojo`, which needs a GPU. Keeping them apart
is what makes a failure here an I/O bug and a failure there a physics one.

Run: pixi run mojo run -I . tests/tasks/test_init_table.mojo
"""

from std.pathlib import Path

from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.eval import region_sites
from mojo_rl.tasks.active import active_mask
from mojo_rl.tasks.sampler import RegionFrame, SampleReport
from mojo_rl.tasks.reset import free_slot_addresses
from mojo_rl.tasks.init_table import (
    InitTable, append_init_rows, write_init_table, load_init_table,
    family_key, INIT_TIME_WORDS,
)
from mojo_rl.tasks.so101_tabletop_xml import So101TabletopModel


comptime NQ = So101TabletopModel.NQ
comptime NV = So101TabletopModel.NV
comptime N_PER_TASK = 8

comptime OUT_A = "/tmp/mojo_rl_init_a.h5"
comptime OUT_B = "/tmp/mojo_rl_init_b.h5"


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def main() raises:
    print("=== frozen init states — freeze, load, refuse ===")
    var ta = Tally()

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)

    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)

    # ⚠ THE REGION FRAME IS A **WORLD** POSITION AND MUST COME FROM FK. A
    # `SiteData.pos` is BODY-LOCAL; the sampler draws around where the site
    # actually is, and on a fixture composed at (0.25, 0, 0.01) the two differ
    # by exactly the fixture's placement. So build the model and run forward
    # kinematics, as `test_task_reset_steps` does.
    var dims = dims_from_flat(
        fmd, max_contacts=32, nmesh_verts=SO_ARM101_NMESH_VERTS
    )
    var m = Model[DType.float64, DynDims](dims)
    build_model_runtime[DType.float64](fmd, dims, m)
    var d = Data[DType.float64, DynDims, 1](dims)
    forward_kinematics["cpu", DType.float64, DynDims, 1](d, m)
    var frames = List[RegionFrame]()
    for i in range(len(f.regions)):
        var s = rsites[i]
        frames.append(RegionFrame(
            Float64(d.site_xpos.data[s * 3]),
            Float64(d.site_xpos.data[s * 3 + 1]),
            Float64(d.site_xpos.data[s * 3 + 2]),
        ))
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(0.02)

    var base = List[Float64]()
    for _ in range(NQ):
        base.append(0.0)

    var tg = load_task("mojo_rl/tasks/tasks/so101_gather_bricks.task")
    var tr = load_task("mojo_rl/tasks/tasks/so101_reach_brick.task")
    validate_task_against_family(tg, f)
    validate_task_against_family(tr, f)

    # ── freeze table A ────────────────────────────────────────────────────
    print()
    print("--- 1. freeze", 2 * N_PER_TASK, "episodes across 2 tasks ---")
    var st = List[Float64]()
    var tix = List[Int32]()
    var mk = List[Float64]()
    var rep = SampleReport()
    append_init_rows(
        tg, f, 0, frames, radii, addrs, base, NQ, NV,
        N_PER_TASK, UInt64(7), 0, st, tix, mk, rep,
    )
    append_init_rows(
        tr, f, 1, frames, radii, addrs, base, NQ, NV,
        N_PER_TASK, UInt64(7), N_PER_TASK, st, tix, mk, rep,
    )
    var names = List[String]()
    names.append(String(tg.language))
    names.append(String(tr.language))
    write_init_table(
        String(OUT_A), f.name, NQ, NV, st, tix, mk, names,
        seed=7, source_commit=String("p4a-gate"),
    )
    print("    sampler:", rep.accepted, "placements in", rep.attempts, "draws")
    print("    wrote", OUT_A)

    var tbl = load_init_table(String(OUT_A), f.name, NQ, NV)
    ta.check(tbl.n_rows() == 2 * N_PER_TASK,
             "loaded " + String(tbl.n_rows()) + " rows")
    ta.check(tbl.row_words() == INIT_TIME_WORDS + NQ + NV,
             "row is 1 + nq + nv = " + String(tbl.row_words()) + " words")
    ta.check(tbl.key == family_key(f.name, NQ, NV),
             "keyed '" + tbl.key + "'")

    # ── 2. bit-exact round trip, and the table IS the episode ─────────────
    print()
    print("--- 2. the rows survive the disk, bit for bit ---")
    var exact = True
    var differing = 0
    var compared = 0
    for i in range(tbl.n_rows()):
        # ⚠ A POISONED DESTINATION. Starting from the freeze-time vector would
        # make `apply` a no-op that passes; starting from a value that appears
        # nowhere in the table means every word compared was WRITTEN.
        var qpos = List[Float64]()
        for _ in range(NQ):
            qpos.append(-7777.0)
        var qvel = List[Float64]()
        for _ in range(NV):
            qvel.append(-8888.0)
        tbl.apply(i, qpos, qvel)
        var b = i * tbl.row_words() + INIT_TIME_WORDS
        for k in range(NQ):
            compared += 1
            if qpos[k] != st[b + k]:
                exact = False
            if qpos[k] != -7777.0:
                differing += 1
        for k in range(NV):
            compared += 1
            if qvel[k] != st[b + NQ + k]:
                exact = False
    print("    ", compared, "words compared,", differing,
          "qpos words actually overwritten")
    ta.check(exact, "every frozen word round-trips == (not almost_equal)")
    # ⚠ ANTI-VACUITY: `apply` writing nothing would leave the poison in place
    # and `exact` would be False — but only if the table is non-degenerate.
    # This says the writes happened at all.
    ta.check(differing == tbl.n_rows() * NQ,
             "apply() wrote EVERY qpos word, not just the free slots")

    # ⚠ THE ROWS MUST DIFFER FROM EACH OTHER, not just from another seed's.
    # Eight frozen episodes that are all the same state is a table whose
    # success rate is one episode measured eight times — and it passes every
    # round-trip check above, because a constant round-trips perfectly.
    var distinct = 0
    for i in range(1, N_PER_TASK):
        var b0 = INIT_TIME_WORDS
        var bi = i * tbl.row_words() + INIT_TIME_WORDS
        var same = True
        for k in range(NQ + NV):
            if tbl._state[b0 + k] != tbl._state[bi + k]:
                same = False
        if not same:
            distinct += 1
    ta.check(
        distinct == N_PER_TASK - 1,
        String(distinct) + " of " + String(N_PER_TASK - 1)
        + " later rows of task 0 differ from its first — the table is not a"
        " constant repeated",
    )

    var masks_ok = True
    for i in range(tbl.n_rows()):
        var want = active_mask(tg, f) if Int(tbl.task_index[i]) == 0 \
            else active_mask(tr, f)
        if tbl.mask[i] != want:
            masks_ok = False
    ta.check(masks_ok, "every row's stored active mask is its task's")

    # ── 3. the control: another seed is another table ─────────────────────
    print()
    print("--- 3. CONTROL: a different sampling seed is a different table ---")
    var st2 = List[Float64]()
    var tix2 = List[Int32]()
    var mk2 = List[Float64]()
    var rep2 = SampleReport()
    append_init_rows(
        tg, f, 0, frames, radii, addrs, base, NQ, NV,
        N_PER_TASK, UInt64(99), 0, st2, tix2, mk2, rep2,
    )
    append_init_rows(
        tr, f, 1, frames, radii, addrs, base, NQ, NV,
        N_PER_TASK, UInt64(99), N_PER_TASK, st2, tix2, mk2, rep2,
    )
    write_init_table(
        String(OUT_B), f.name, NQ, NV, st2, tix2, mk2, names,
        seed=99, source_commit=String("p4a-gate"),
    )
    var tbl2 = load_init_table(String(OUT_B), f.name, NQ, NV)
    # ⚠ COMPARED THROUGH BOTH **LOADED** TABLES, not through `st` and `st2`.
    # Reading the in-memory freeze would compare the sampler against itself
    # and would pass on a loader that ignored the file and returned a
    # constant — the two arms would never touch the disk. Same shape as
    # `feedback_a_gate_that_shares_its_reference_implementation_is_blind`:
    # ask what code BOTH sides pass through on the way to the comparison.
    var diff_words = 0
    var cmp_words = 0
    var diff_rows = 0
    for i in range(tbl.n_rows()):
        var qa = List[Float64]()
        var qb = List[Float64]()
        for _ in range(NQ):
            qa.append(0.0)
            qb.append(0.0)
        var va = List[Float64]()
        var vb = List[Float64]()
        for _ in range(NV):
            va.append(0.0)
            vb.append(0.0)
        tbl.apply(i, qa, va)
        tbl2.apply(i, qb, vb)
        var row_diff = False
        for k in range(NQ):
            cmp_words += 1
            if qa[k] != qb[k]:
                diff_words += 1
                row_diff = True
        for k in range(NV):
            cmp_words += 1
            if va[k] != vb[k]:
                diff_words += 1
                row_diff = True
        if row_diff:
            diff_rows += 1
    print("    ", diff_words, "of", cmp_words, "words differ, in",
          diff_rows, "of", tbl.n_rows(), "rows")
    # ⚠ THE COUNT IS EXPLICABLE, WHICH IS THE POINT OF PRINTING IT. Only an
    # ACTIVE free slot's x and y move between seeds: the sampler pins z to the
    # surface, the quaternion is identity, a parked slot is at a fixed park
    # pose and the arm starts from `base_qpos`. gather runs 2 active slots and
    # reach 1, so 8*(2*2) + 8*(1*2) = 48. A number that does NOT decompose
    # like that is the interesting failure.
    # ⚠⚠ WITHOUT THIS CHECK THE WHOLE FILE IS VACUOUS. Checks 1 and 2 hold on
    # a freeze that ignores its seed and writes one state N times.
    ta.check(diff_words > 0,
             "the two seeds produce DIFFERENT frozen states")
    ta.check(diff_rows == tbl.n_rows(),
             "and EVERY row differs, not just one — the seed moved them all")
    ta.check(tbl2.n_rows() == tbl.n_rows(),
             "with the same row count, so the difference is the STATE")

    # ── 4. a mismatched table is refused, three ways ──────────────────────
    print()
    print("--- 4. a table from elsewhere is REFUSED, not adapted ---")
    var refused = 0
    try:
        var _bad = load_init_table(String(OUT_A), String("other_family"), NQ, NV)
        print("    NOT REFUSED: wrong family name")
    except e:
        refused += 1
        print("    refused wrong family name")
    try:
        var _bad = load_init_table(String(OUT_A), f.name, NQ + 1, NV)
        print("    NOT REFUSED: wrong nq")
    except e:
        refused += 1
        print("    refused wrong nq")
    try:
        var _bad = load_init_table(String(OUT_A), f.name, NQ, NV - 1)
        print("    NOT REFUSED: wrong nv")
    except e:
        refused += 1
        print("    refused wrong nv")
    ta.check(refused == 3, String(refused) + " of 3 mismatches refused")
    # ⚠ AND THE MATCHING ONE STILL LOADS — a loader that refused everything
    # would score 3 of 3 above.
    var _ok = load_init_table(String(OUT_A), f.name, NQ, NV)
    ta.check(_ok.n_rows() == tbl.n_rows(),
             "and the MATCHING key still loads (the refusal is not blanket)")

    # ── 5. the per-task breakdown resolves ────────────────────────────────
    print()
    print("--- 5. every row names its task, byte-exactly ---")
    var labelled = 0
    var label_ok = True
    for i in range(tbl.n_rows()):
        var want = tg.language if Int(tbl.task_index[i]) == 0 else tr.language
        if tbl.task_label(i) != want:
            label_ok = False
        else:
            labelled += 1
    print("    ", labelled, "of", tbl.n_rows(), "rows resolved:",
          '"' + tbl.task_label(0) + '"', "|",
          '"' + tbl.task_label(tbl.n_rows() - 1) + '"')
    ta.check(label_ok and labelled == tbl.n_rows(),
             "every task_index resolves to its instruction through the manifest")
    # ⚠ ANTI-VACUITY: one task would make "every index resolves" trivially
    # true and would not show that the mapping is per-ROW.
    var seen0 = False
    var seen1 = False
    for i in range(tbl.n_rows()):
        if Int(tbl.task_index[i]) == 0:
            seen0 = True
        if Int(tbl.task_index[i]) == 1:
            seen1 = True
    ta.check(seen0 and seen1,
             "the table carries BOTH tasks, so the mapping is per-row")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "init table: " + String(ta.failures) + " of " + String(ta.checks)
            + " failed"
        )
    print("=== PASS ===")
