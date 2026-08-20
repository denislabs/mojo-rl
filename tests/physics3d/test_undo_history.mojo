"""Undo ACROSS a structural edit — the document snapshot stack. V2.9.

WHY THIS EXISTS
===============
V2 shipped `delete body`, the most destructive operation in the studio, and
reset the `EditLog` after it: there was no undo for a delete, an add, a rename
or a reparent. `EditLog` replays dims-preserving edits, and a delete is not
one of them.

The document is now authoritative (V2.4), so the snapshot is a `String` and
one stack covers both tiers. This gates that stack.

⚠⚠ THE ARM THAT MATTERS IS "UNDO RESTORES THE ORIGINAL **RECORD**", not "undo
restores the original text". Text equality would pass on a stack that never
stored anything and returned the input; and it would FAIL on a correct undo
that had round-tripped through the writer. So every restore here is PARSED and
compared as a model fingerprint — counts plus every name, in order.

⚠ AND NON-VACUITY IS THE DEFAULT FAILURE HERE. "the restored model equals the
original" is trivially true if the edit changed nothing. Each arm prints the
BEFORE fingerprint beside the AFTER and asserts they DIFFER first.

The documents this test walks through are dumped to /tmp/undo_history/ and
`scripts/check_undo_history_vs_mujoco.py` asserts MuJoCo loads every one of
them: an undo that produces text the reference refuses is a bug no internal
comparison can see.

Run: pixi run mojo run -I . tests/physics3d/test_undo_history.mojo
     pixi run python scripts/check_undo_history_vs_mujoco.py
"""

from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.studio.history import (
    History, HISTORY_CAP, edit_key,
)
from mojo_rl.physics3d.studio.scene import SceneDoc, scene_from_base
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.studio.remap import (
    remap_state, pose_snapshot, apply_pose_snapshot, joint_qpos_adr,
)
from mojo_rl.physics3d.studio.structure import (
    delete_body, delete_geom, rename_element, add_body, reparent_body,
)

comptime DT = DType.float64
comptime ZOO = "tests/physics3d/assets/structural_edit_zoo.xml"
comptime CHEETAH = "mojo_rl/envs/half_cheetah/assets/half_cheetah.xml"
comptime DUMP = "/tmp/undo_history"


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def fingerprint(xml: String) raises -> String:
    """Counts AND every name, in order — the model the text describes.

    ⚠ NAMES, NOT JUST COUNTS. `test_body_tree_vs_mujoco` exists because nbody
    agreed while the tree was a different robot; a fingerprint of five integers
    would repeat that mistake one level down. Names in order also catch a
    rename that landed on the wrong element, which counts cannot.
    """
    var fmd = parse_xml_full(xml, String(""))
    var s = String(
        "nb=", len(fmd.bodies), " ng=", len(fmd.geoms),
        " nj=", len(fmd.joints), " ns=", len(fmd.sites),
        " na=", len(fmd.actuators), " nt=", len(fmd.tendons),
        " ne=", len(fmd.equalities), " |",
    )
    for n in fmd.body_names:
        s += " " + n
    s += " |"
    for n in fmd.geom_names:
        s += " " + n
    s += " |"
    for n in fmd.joint_names:
        s += " " + n
    return s^


def dumped(mut n: Int, tag: String, xml: String) raises:
    """Every state the stack hands back goes to disk for MuJoCo to judge."""
    var p = String(DUMP, "/", n, "_", tag, ".xml")
    with open(p, "w") as f:
        f.write(xml)
    n += 1


def main() raises:
    var t = Tally()
    print("=== undo across a structural edit ===")

    # ⚠ THE DIRECTORY MUST EXIST — `open(.., "w")` does not create one, and a
    # dump that silently failed would leave the reference judging nothing.
    var dumps = 0

    var scene = scene_from_base(String(ZOO))
    var doc0 = expand_mjcf(read_file(String(ZOO)), String(""))
    var fp0 = fingerprint(doc0)

    var h = History()
    h.push(doc0, String(""), scene, String("opened"))
    t.truth(h.depth() == 1 and not h.can_undo() and not h.can_redo(),
            String("a fresh stack has one entry and neither direction (depth ",
                   h.depth(), ")"))
    dumped(dumps, String("opened"), h.doc())

    # ── a delete, then an undo ────────────────────────────────────────────
    print("--- delete 'arm', then undo ---")
    var r1 = delete_body(doc0, String("arm"))
    t.truth(r1.ok, "delete_body('arm') succeeded")
    h.push(r1.xml, String(""), scene, String("deleted 'arm'"))
    var fp1 = fingerprint(h.doc())
    dumped(dumps, String("deleted_arm"), h.doc())

    # ⚠ NON-VACUITY FIRST. Everything below is trivially true if the delete
    # did nothing — and `find_named` matching inside a comment made exactly
    # that failure mode real once (it took nbody 5 to 1 the other way).
    t.truth(fp0 != fp1, "the delete CHANGED the model (non-vacuity)")
    print("      before:", fp0)
    print("      after :", fp1)

    t.truth(h.can_undo() and not h.can_redo(),
            "after one edit: undo available, redo not")
    t.truth(h.undo_label() == "deleted 'arm'",
            String("the menu can name what will come back: '",
                   h.undo_label(), "'"))
    t.truth(h.undo(), "undo() reports it moved")
    t.truth(fingerprint(h.doc()) == fp0,
            "undo restored the ORIGINAL record — counts and every name")
    dumped(dumps, String("undone"), h.doc())

    # ── redo ──────────────────────────────────────────────────────────────
    print("--- redo ---")
    t.truth(h.can_redo(), "redo is available after an undo")
    t.truth(h.redo(), "redo() reports it moved")
    t.truth(fingerprint(h.doc()) == fp1, "redo restored the DELETED state")
    t.truth(not h.can_redo(), "and there is nothing further to redo")

    # ── the floor, and the control ────────────────────────────────────────
    print("--- the floor ---")
    _ = h.undo()
    t.truth(not h.can_undo(), "at entry 0 there is nothing left to undo")
    # ⚠ THE RETURN VALUE IS LOAD-BEARING. The studio rebuilds only on True; an
    # undo at the floor that reported success would rebuild the same document
    # and throw the live pose away for nothing.
    t.truth(not h.undo(), "undo() at the floor reports False (control)")
    t.truth(fingerprint(h.doc()) == fp0, "and the document did not move")

    # ── a new edit after an undo discards the redo tail ───────────────────
    print("--- a new edit discards the redo tail ---")
    t.truth(h.can_redo(), "the delete is still sitting in the tail")
    var r2 = rename_element(doc0, String("body"), String("post"),
                            String("mast"))
    t.truth(r2.ok, "rename post -> mast succeeded")
    h.push(r2.xml, String(""), scene, String("renamed 'post'"))
    t.truth(not h.can_redo(),
            "pushing after an undo dropped the redo tail")
    t.truth(h.depth() == 2, String("depth is 2, not 3 — got ", h.depth()))
    var fp2 = fingerprint(h.doc())
    t.truth(fp2 != fp0 and fp2 != fp1,
            "and the live state is the RENAME, not the delete (non-vacuity)")
    # ⚠ ASK THE RECORD, NOT THE FINGERPRINT STRING. The first draft tested
    # `fp2.find(" post") == -1` and failed on a correct rename: the geom is
    # `post_geom`, so the body's old name is a PREFIX of a name that legally
    # survives. A substring search over a flattened list is a match on the
    # wrong element waiting to happen.
    var renamed = parse_xml_full(h.doc(), String(""))
    var has_mast = False
    var has_post = False
    for n in renamed.body_names:
        if n == "mast":
            has_mast = True
        if n == "post":
            has_post = True
    t.truth(has_mast and not has_post,
            "the renamed BODY carries its new name and not its old one")
    dumped(dumps, String("renamed"), h.doc())

    # ── coalescing: a drag is ONE step ────────────────────────────────────
    print("--- coalescing ---")
    var h2 = History()
    h2.push(doc0, String(""), scene, String("opened"))
    var k = edit_key(0, 3, 5)
    for i in range(40):
        # The text differs every frame, as a real drag's does.
        h2.push(doc0 + String("<!--", i, "-->"), String(""), scene,
                String("size"), k)
    t.truth(h2.depth() == 2,
            String("40 frames of one drag are ONE undo step (depth ",
                   h2.depth(), ")"))
    t.truth(h2.doc().find(String("<!--39-->")) != -1,
            "and the LAST frame is what is live, not the first")
    _ = h2.undo()
    t.truth(fingerprint(h2.doc()) == fp0,
            "one undo takes the whole drag back")

    # ⚠ THE CONTROL: distinct keys must NOT fold together, or dragging five
    # different sliders would leave one undo step and lose four edits.
    var h3 = History()
    h3.push(doc0, String(""), scene, String("opened"))
    for i in range(5):
        h3.push(doc0 + String("<!--", i, "-->"), String(""), scene,
                String("size"), edit_key(0, i, 5))
    t.truth(h3.depth() == 6,
            String("five DIFFERENT fields are five steps (depth ",
                   h3.depth(), ")"))

    # ⚠ AND THE OTHER CONTROL: "" is the ABSENCE of a key, not a key. Two
    # structural edits in a row must both survive.
    var h4 = History()
    h4.push(doc0, String(""), scene, String("opened"))
    h4.push(doc0 + String("<!--a-->"), String(""), scene, String("a"))
    h4.push(doc0 + String("<!--b-->"), String(""), scene, String("b"))
    t.truth(h4.depth() == 3,
            String("two keyless pushes do not coalesce (depth ", h4.depth(),
                   ")"))

    # ── the cap evicts the OLDEST ─────────────────────────────────────────
    print("--- the cap ---")
    var h5 = History()
    for i in range(HISTORY_CAP + 20):
        h5.push(String("<mujoco/><!--", i, "-->"), String(""), scene,
                String("e", i))
    t.truth(h5.depth() == HISTORY_CAP,
            String("depth is capped at ", HISTORY_CAP, " — got ", h5.depth()))
    t.truth(h5.doc().find(String("<!--", HISTORY_CAP + 19, "-->")) != -1,
            "the NEWEST entry is live")
    for _ in range(HISTORY_CAP * 2):
        _ = h5.undo()
    t.truth(h5.doc().find(String("<!--20-->")) != -1,
            "and undoing to the floor lands on entry 20 — the oldest KEPT")

    # ── base_dir and the scene ride along ─────────────────────────────────
    print("--- base_dir and the scene ---")
    # ⚠⚠ BOTH ARE EASY TO DROP AND NEITHER IS RECONSTRUCTIBLE FROM THE TEXT.
    # `base_dir` differs between a structural edit (the model's directory) and
    # a prop edit (the CWD), and `SceneDoc.to_mjcf` is one-way.
    var h6 = History()
    var s0 = scene_from_base(String(ZOO))
    h6.push(doc0, String("dir/a"), s0, String("opened"))
    var s1 = scene_from_base(String(ZOO))
    _ = s1.add_prop(0, 0.05, 0.05, 0.05, 0.0, 0.0, 0.5)
    h6.push(doc0, String("dir/b"), s1, String("added a prop"))
    t.truth(h6.base_dir() == "dir/b", "the live entry's base_dir")
    t.truth(len(h6.scene().props) == 1, "the live entry's scene has the prop")
    _ = h6.undo()
    t.truth(h6.base_dir() == "dir/a",
            String("undo restored the OTHER base_dir — got '", h6.base_dir(),
                   "'"))
    t.truth(len(h6.scene().props) == 0,
            "and the scene came back without the prop")

    # ── a real robot, and a two-deep undo ─────────────────────────────────
    print("--- half_cheetah: two edits deep ---")
    var c0 = expand_mjcf(read_file(String(CHEETAH)), String(""))
    var cfp0 = fingerprint(c0)
    var hc = History()
    hc.push(c0, String(""), scene, String("opened"))
    dumped(dumps, String("cheetah_opened"), hc.doc())

    var cr1 = delete_body(c0, String("bthigh"))
    t.truth(cr1.ok, "delete_body('bthigh') succeeded")
    hc.push(cr1.xml, String(""), scene, String("deleted 'bthigh'"))
    var cfp1 = fingerprint(hc.doc())
    t.truth(cfp0 != cfp1, "the delete changed the cheetah (non-vacuity)")
    dumped(dumps, String("cheetah_no_bthigh"), hc.doc())

    var cr2 = add_body(hc.doc(), String("torso"), String("fin"),
                       0.0, 0.0, 0.2)
    t.truth(cr2.ok, "add_body('fin' under 'torso') succeeded")
    hc.push(cr2.xml, String(""), scene, String("added 'fin'"))
    var cfp2 = fingerprint(hc.doc())
    t.truth(cfp2 != cfp1, "the add changed it again (non-vacuity)")
    dumped(dumps, String("cheetah_with_fin"), hc.doc())

    _ = hc.undo()
    t.truth(fingerprint(hc.doc()) == cfp1,
            "one undo is back at the delete")
    _ = hc.undo()
    t.truth(fingerprint(hc.doc()) == cfp0,
            "two undos are back at the file as opened — ACROSS two structural"
            " edits, which V2 could not do at all")
    dumped(dumps, String("cheetah_undone_twice"), hc.doc())

    # ── reparent, undone ──────────────────────────────────────────────────
    print("--- reparent, undone ---")
    var rp = reparent_body(doc0, String("hand"), String("trunk"))
    t.truth(rp.ok, String("reparent hand -> trunk: ", rp.notes[0]))
    var hr = History()
    hr.push(doc0, String(""), scene, String("opened"))
    hr.push(rp.xml, String(""), scene, String("reparented 'hand'"))
    t.truth(fingerprint(hr.doc()) != fp0,
            "the reparent changed the model (non-vacuity)")
    dumped(dumps, String("reparented"), hr.doc())
    _ = hr.undo()
    t.truth(fingerprint(hr.doc()) == fp0, "and it undid cleanly")

    # ── a geom delete, undone ─────────────────────────────────────────────
    print("--- geom delete, undone ---")
    var gd = delete_geom(doc0, String("post_geom"))
    t.truth(gd.ok, "delete_geom('post_geom') succeeded")
    var hg = History()
    hg.push(doc0, String(""), scene, String("opened"))
    hg.push(gd.xml, String(""), scene, String("deleted 'post_geom'"))
    var gfp = fingerprint(hg.doc())
    t.truth(gfp != fp0, "the geom delete changed the model (non-vacuity)")
    dumped(dumps, String("geom_deleted"), hg.doc())
    _ = hg.undo()
    t.truth(fingerprint(hg.doc()) == fp0, "and it undid cleanly")

    # ── the POSE of what came back ────────────────────────────────────────
    # ⚠⚠ THE COUNT IS NOT THE GATE. The studio prints "3 of the reset came
    # back from the snapshot", which reads identically whether the values are
    # the ones the joints were deleted at or three zeros written confidently.
    # This compares the NUMBERS.
    print("--- the deleted subtree comes back where it was ---")
    var pb = String("mojo_rl/envs/ant/assets")
    var asrc = expand_mjcf(read_file(String("mojo_rl/envs/ant/assets/ant.xml")),
                           pb)
    var afmd = parse_xml_full(asrc, pb)
    var adims = dims_from_flat(afmd)
    var am = Model[DT, DynDims](adims)
    build_model_runtime[DT](afmd, adims, am)
    var ad = Data[DT, DynDims, 1](adims)
    # ⚠ A DISTINCTIVE POSE, derived from each slot's own address, so a value
    # landing one slot over shows up as a wrong NUMBER and not as a plausible
    # pose. Same reasoning as `test_state_remap`.
    for i in range(adims.get_nq()):
        ad.qpos.data[i] = Scalar[DT](0.1 + 0.01 * Float64(i))
    for i in range(adims.get_nv()):
        ad.qvel.data[i] = Scalar[DT](-0.5 - 0.01 * Float64(i))

    # ⚠ NOT THE LAST LEG — `front_left_leg`'s joints sit in the MIDDLE of
    # ant's joint list, so restoring them has to place values into addresses
    # that everything after them also occupies.
    var vic = String("front_left_leg")
    var want_hip = Float64(0.0)
    var want_ankle = Float64(0.0)
    var aq = joint_qpos_adr(afmd)
    for j in range(len(afmd.joints)):
        if afmd.joint_names[j] == "hip_1":
            want_hip = Float64(ad.qpos.data[aq[j]])
        if afmd.joint_names[j] == "ankle_1":
            want_ankle = Float64(ad.qpos.data[aq[j]])
    t.truth(want_hip != 0.0 and want_ankle != 0.0,
            String("the two victim joints start at ", want_hip, " / ",
                   want_ankle, " (non-vacuity: not already zero)"))

    var snap = pose_snapshot(afmd, ad)
    t.truth(len(snap.names) == len(afmd.joints),
            String("the snapshot holds every joint (", len(snap.names), ")"))

    var adel = delete_body(asrc, vic)
    t.truth(adel.ok, String("deleted '", vic, "' from ant"))
    var bfmd = parse_xml_full(adel.xml, pb)
    var bdims = dims_from_flat(bfmd)
    var bm = Model[DT, DynDims](bdims)
    build_model_runtime[DT](bfmd, bdims, bm)
    var bsf = spec_fields_runtime[DT](bfmd, bdims, bm)
    var bd = Data[DT, DynDims, 1](bdims)
    for i in range(bdims.get_nq()):
        bd.qpos.data[i] = bsf.qpos0.data[i]
    var brep = remap_state(afmd, ad, bfmd, bd)
    t.truth(brep.dropped == 2,
            String("two joints went with the leg — got ", brep.dropped))

    # Now UNDO: rebuild the original and fill from the live (reduced) state.
    var cdims = dims_from_flat(afmd)
    var cm = Model[DT, DynDims](cdims)
    build_model_runtime[DT](afmd, cdims, cm)
    var csf = spec_fields_runtime[DT](afmd, cdims, cm)
    var cd = Data[DT, DynDims, 1](cdims)
    for i in range(cdims.get_nq()):
        cd.qpos.data[i] = csf.qpos0.data[i]
    var crep = remap_state(bfmd, bd, afmd, cd)
    t.truth(crep.reset == 2,
            String("the live state cannot account for the two resurrected"
                   " joints — got ", crep.reset))

    # ⚠ WITHOUT THE SNAPSHOT THEY SIT AT qpos0 — measured, not asserted, so
    # the arm below is known to be testing something.
    var hip_i = -1
    var ank_i = -1
    for j in range(len(afmd.joints)):
        if afmd.joint_names[j] == "hip_1":
            hip_i = j
        if afmd.joint_names[j] == "ankle_1":
            ank_i = j
    var before_hip = Float64(cd.qpos.data[aq[hip_i]])
    t.truth(before_hip != want_hip,
            String("before the snapshot the hip is at ", before_hip,
                   ", not ", want_hip, " (non-vacuity)"))

    apply_pose_snapshot(snap, afmd, cd, crep)
    t.truth(crep.restored == 2,
            String("both came back from the snapshot — got ", crep.restored))
    t.truth(Float64(cd.qpos.data[aq[hip_i]]) == want_hip,
            String("hip_1 is back at ", want_hip, " — got ",
                   Float64(cd.qpos.data[aq[hip_i]])))
    t.truth(Float64(cd.qpos.data[aq[ank_i]]) == want_ankle,
            String("ankle_1 is back at ", want_ankle, " — got ",
                   Float64(cd.qpos.data[aq[ank_i]])))

    # ⚠⚠ AND THE LIVE STATE MUST NOT HAVE BEEN OVERWRITTEN. The snapshot is
    # older than the running sim; applying it to slots the live state already
    # answered would rewind the whole robot on every undo. `remap_state` set
    # the root's qpos from `bd` — which is where `ad` put it — so a joint the
    # first pass carried must still hold the LIVE value, and the way to prove
    # the snapshot did not simply agree is to make them differ first.
    var d2 = Data[DT, DynDims, 1](cdims)
    for i in range(cdims.get_nq()):
        d2.qpos.data[i] = csf.qpos0.data[i]
    for i in range(bdims.get_nq()):
        bd.qpos.data[i] = Scalar[DT](7.0)
    var rep2 = remap_state(bfmd, bd, afmd, d2)
    var live_j = -1
    for j in range(len(afmd.joints)):
        if afmd.joint_names[j] == "hip_4":
            live_j = j
    t.truth(live_j >= 0 and Float64(d2.qpos.data[aq[live_j]]) == 7.0,
            "a joint the LIVE state answered for holds the live value")
    apply_pose_snapshot(snap, afmd, d2, rep2)
    t.truth(Float64(d2.qpos.data[aq[live_j]]) == 7.0,
            String("and the snapshot did NOT overwrite it — got ",
                   Float64(d2.qpos.data[aq[live_j]]),
                   " (the snapshot holds ", want_hip, "-ish)"))

    print("--- the dump the reference will judge ---")
    t.truth(dumps >= 10, String("documents written for MuJoCo: ", dumps))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_undo_history: " + String(t.fails) + " failed")
