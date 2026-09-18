"""Our goals, evaluated on LIBERO's OWN successful demonstrations.

    pixi run python tools/tasks/libero_demo_success.py --suite libero_goal
    pixi run mojo run -I . examples/tasks/libero_demo_success.mojo \
        references/libero_demos/_dumps/libero_goal/index.txt

Every demonstration in the corpus is a successful human teleoperation of the
task whose `.task` file we generated, so the state it ends in is a state the
benchmark calls SOLVED. This drives our engine to each of those states —
that demo's own fixture placement patched in, the recorded `qpos`/`qvel` set,
forward kinematics and collision run — and asks our bound goal.

**Every demo of every task must fire SOMEWHERE IN ITS WINDOW.** A task that
scores below 50 of 50 has a wrong predicate, and the failure names which.

⚠⚠ "SOMEWHERE", NOT "AT THE END" — AND THAT CORRECTION COST TWO DEMOS. Asking
only about the FINAL recorded state scored 498 of 500: one
`put_the_bowl_on_the_plate` and one `put_the_wine_bottle_on_top_of_the_cabinet`
came out False with correct predicates. LIBERO's protocol
(`lifelong/metric.py`) checks EVERY step and succeeds at ANY step, and the
recording does not stop there — the stove demos satisfy their goal about
eleven steps before they end — so an operator can knock the bowl off the
plate after the episode has already been scored. The window is the
benchmark's question; the final frame was mine.

⚠⚠ THIS IS THE GATE `test_libero_goal_eval` CANNOT BE. That file builds
eleven states by hand and checks the ten goals against them; I wrote both,
so they agree by construction and a wrong threshold, an inverted argument or
a mis-anchored box would pass. Here the states come from the benchmark and
the answer is not mine to choose.

⚠ FIXTURE POSES ARE PATCHED INTO THE MODEL RECORD, NOT REBUILT. LIBERO
redraws each fixture's xy inside its region rect at every reset (the
assessment's §6e), so each demo has its own placement and the family's
static slot sits at the rect centre. `bodies[b, BODY_IDX_POS_*]` is what
forward kinematics reads, so writing those seven columns is the whole patch
and costs nothing per demo.

⚠ COLLISION IS RUN, NOT STEPPED. `On(obj, obj)` and a box region with a
contact slot read the contact list, and a step would move the state we were
asked about. `detect_contacts` fills `d.contacts` and the lane's contact
count from the FK products at exactly this state.
"""

from std.sys import argv
from std.math import sqrt

from mojo_rl.physics3d.fields import Data, Model, DynDims, DynamicsScratch
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_POS_X, BODY_IDX_QUAT_X, BODY_IDX_QUAT_W,
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
)
from mojo_rl.tasks.spec import load_family, load_task, validate_task_against_family
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses,
)
from mojo_rl.tasks.eval import (
    eval_goal, HostState, region_sites, region_contact_bodies,
)
from mojo_rl.tasks.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS
from mojo_rl.tasks.libero_fixtures import dump_path_from_index


comptime DT = DType.float64
comptime TASK_DIR = "mojo_rl/tasks/tasks/"


def _floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    var toks = s.split(" ")
    for k in range(len(toks)):
        var t = String(String(toks[k]).strip())
        if t.byte_length() > 0:
            out.append(Float64(t))
    return out^


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error(
            "usage: libero_demo_success.mojo <index.txt>   (written by"
            " tools/tasks/libero_demo_success.py)"
        )
    var index_path = String(a[1])
    var index_text: String
    try:
        with open(index_path, "r") as fh:
            index_text = fh.read()
    except e:
        # ⚠ SKIPPED, LOUDLY, AND NOT A PASS. The demonstrations are ~6 GB per
        # suite and gitignored; a clone without them cannot run this.
        print("  SKIPPED: no index at", index_path)
        print("  Fetch the suite's demos into references/libero_demos/<suite>/")
        print("  (HF yifengzhu-hf/LIBERO-datasets) and run")
        print("     pixi run libero-demo-dump --suite <suite>")
        print("=== SKIPPED (no demonstrations — this is not a pass) ===")
        return

    var task_names = List[String]()
    var dump_paths = List[String]()
    var lines = index_text.split("\n")
    for i in range(len(lines)):
        var l = String(String(lines[i]).strip())
        if l.byte_length() == 0:
            continue
        var parts = l.split(" ")
        task_names.append(String(parts[0]))
        dump_paths.append(dump_path_from_index(index_path, String(parts[1])))
    if len(task_names) == 0:
        raise Error("empty index: " + index_path)

    # every task in the index belongs to one family
    var suite = String(task_names[0])
    var cut = suite.find("__")
    if cut < 0:
        raise Error("task name without a suite prefix: " + suite)
    # ⚠ A TEMPORARY: `x = String(x[...])` aliases the value being replaced.
    var suite_cut = String(suite[byte=0:cut])
    suite = suite_cut^
    var f = load_family("mojo_rl/tasks/families/" + suite + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
    )
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(
                fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nb = dims.get_nbody()
    var ns = dims.get_nsite()
    var mc = LIBERO_GOAL_MAX_CONTACTS

    var nqs = List[Int]()
    for i in range(len(fmd.joints)):
        nqs.append(fmd.joints[i].nq)
    var jadr = joint_qpos_addresses(nqs)
    var rsites = region_sites(f, fmd.site_names)
    var rcontact = region_contact_bodies(f, fmd.body_names)
    var site_body = List[Int]()
    var site_quat = List[Float64]()
    for i in range(len(fmd.sites)):
        site_body.append(fmd.sites[i].body_id)
        site_quat.append(fmd.sites[i].quat_x)
        site_quat.append(fmd.sites[i].quat_y)
        site_quat.append(fmd.sites[i].quat_z)
        site_quat.append(fmd.sites[i].quat_w)
    var body_parent = List[Int]()
    body_parent.append(-1)
    for i in range(len(fmd.bodies)):
        body_parent.append(fmd.bodies[i].parent)

    print("=" * 78)
    print("OUR goals on LIBERO's OWN demonstrations —", suite)
    print("=" * 78)
    print("  scene:", scene_path(f), "| nq", nq, "nv", nv, "nbody", nb,
          "nsite", ns, "| max contacts", mc)
    print()

    var total = 0
    var total_ok = 0
    var failing = List[String]()
    for ti in range(len(task_names)):
        var t = load_task(String(TASK_DIR) + task_names[ti] + ".task")
        validate_task_against_family(t, f)
        var g = bind_goal(
            parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
            fmd.joint_names, jadr,
        )
        require_tier_a(g, t.name)

        var dump: String
        with open(dump_paths[ti], "r") as fh:
            dump = fh.read()
        var dl = dump.split("\n")
        var n_demos = Int(String(String(dl[0]).strip()))

        var solved = 0
        var seen = 0
        var min_con = 1 << 30
        var max_con = 0
        var earliest = 0      # steps before the end, summed for the mean
        var latest_fire = 0   # the worst (closest to the end) over the task
        var i = 1
        while i < len(dl):
            var l = String(String(dl[i]).strip())
            if not l.startswith("DEMO "):
                i += 1
                continue
            var head = l.split(" ")
            var n_win = Int(String(head[2]))
            # the window's states, oldest first
            var win_q = List[List[Float64]]()
            var win_v = List[List[Float64]]()
            for w in range(n_win):
                var qp = _floats(String(String(dl[i + 1 + 2 * w])[byte=5:]))
                var qv = _floats(String(String(dl[i + 2 + 2 * w])[byte=5:]))
                if len(qp) != nq or len(qv) != nv:
                    raise Error(
                        "demo " + l + ": " + String(len(qp)) + " qpos / "
                        + String(len(qv)) + " qvel, scene wants " + String(nq)
                        + " / " + String(nv)
                    )
                win_q.append(qp^)
                win_v.append(qv^)
            i += 1 + 2 * n_win
            while i < len(dl):
                var fl = String(String(dl[i]).strip())
                if not fl.startswith("FIX "):
                    break
                var toks = fl.split(" ")
                var want = String(toks[1])
                var bi = -1
                for b in range(len(fmd.body_names)):
                    if String(fmd.body_names[b]) == want:
                        bi = b
                if bi <= 0:
                    raise Error("no body '" + want + "' in the scene")
                # ⚠ THE MODEL RECORD, ROW `bi`. `fmd.body_names` includes the
                # world at 0 and `Model.bodies` is indexed the same way, so
                # the row is `bi` and NOT `bi - 1` (which is the FlatModelDef
                # list's index — the off-by-one that reads a real, wrong body).
                var o = bi * MODEL_BODY_SIZE
                m.bodies.data[o + BODY_IDX_POS_X + 0] = Scalar[DT](Float64(String(toks[2])))
                m.bodies.data[o + BODY_IDX_POS_X + 1] = Scalar[DT](Float64(String(toks[3])))
                m.bodies.data[o + BODY_IDX_POS_X + 2] = Scalar[DT](Float64(String(toks[4])))
                m.bodies.data[o + BODY_IDX_QUAT_W] = Scalar[DT](Float64(String(toks[5])))
                m.bodies.data[o + BODY_IDX_QUAT_X + 0] = Scalar[DT](Float64(String(toks[6])))
                m.bodies.data[o + BODY_IDX_QUAT_X + 1] = Scalar[DT](Float64(String(toks[7])))
                m.bodies.data[o + BODY_IDX_QUAT_X + 2] = Scalar[DT](Float64(String(toks[8])))
                i += 1

            # ⚠ WALK THE WINDOW FROM THE END BACKWARDS and stop at the first
            # True. The benchmark succeeds at ANY step, so one hit settles the
            # demo; going backwards also measures HOW LATE the goal still
            # holds, which is the number that says whether our predicate and
            # theirs agree about when the episode was won.
            var fired = -1
            for w in range(n_win - 1, -1, -1):
                for k in range(nq):
                    d.qpos.data[k] = Scalar[DT](win_q[w][k])
                for k in range(nv):
                    d.qvel.data[k] = Scalar[DT](win_v[w][k])
                forward_kinematics["cpu", DT, DynDims, 1](d, m)
                detect_contacts["cpu", DT, DynDims, 1](d, m)

                var st = HostState(
                    List[Float64](), List[Float64](), List[Float64]()
                )
                for k in range(nb * 3):
                    st.xpos.append(Float64(d.xpos.data[k]))
                for k in range(nb * 4):
                    st.xquat.append(Float64(d.xquat.data[k]))
                for k in range(ns * 3):
                    st.site_xpos.append(Float64(d.site_xpos.data[k]))
                for k in range(nq):
                    st.qpos.append(Float64(d.qpos.data[k]))
                st.site_body = site_body.copy()
                st.site_quat = site_quat.copy()
                st.body_parent = body_parent.copy()
                var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
                if ncon > mc:
                    ncon = mc
                st.ncon = ncon
                for k in range(ncon):
                    st.con_a.append(Int(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_A]))
                    st.con_b.append(Int(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_B]))
                if ncon < min_con:
                    min_con = ncon
                if ncon > max_con:
                    max_con = ncon
                if eval_goal(g, f, st, rsites, rcontact):
                    fired = n_win - 1 - w
                    break
            if fired >= 0:
                solved += 1
                earliest += fired
                if fired > latest_fire:
                    latest_fire = fired
            seen += 1
        total += seen
        total_ok += solved
        var stem = String(task_names[ti])
        var c2 = stem.find("__")
        var stem_cut = String(stem[byte = c2 + 2 :])
        stem = stem_cut^
        var verdict = "ok  " if solved == seen else "FAIL"
        var mean_lag = 0
        if solved > 0:
            mean_lag = earliest // solved
        print("  " + verdict, String(solved) + " / " + String(seen),
              " fires", String(mean_lag) + " steps before the end (worst "
              + String(latest_fire) + ")",
              " contacts", String(min_con) + "-" + String(max_con), " ", stem)
        print("        goal:", t.goal)
        if solved != seen:
            failing.append(stem + " (" + String(solved) + "/" + String(seen) + "): " + t.goal)
        _ = n_demos

    print()
    print("  TOTAL:", total_ok, "of", total,
          "recorded demonstrations satisfy our goal in their window")
    if len(failing) > 0:
        print()
        print("  ⚠ THESE PREDICATES DISAGREE WITH THE BENCHMARK:")
        for i in range(len(failing)):
            print("      ", failing[i])
        raise Error(
            String(total - total_ok) + " of " + String(total) + " recorded"
            " SUCCESSFUL demonstrations never satisfy our goal anywhere in"
            " their trailing window. The demonstrations are ground truth; the"
            " predicate is what is wrong. (If a task is close, re-dump with a"
            " larger --window before touching a threshold.)"
        )
    print("=== PASS — every demonstration reaches a state our goal calls solved ===")
