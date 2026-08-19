"""Deleting a body leaves a model that LOADS — ours, and MuJoCo's.

WHY THIS EXISTS
===============
`delete_body` is the first edit in this tool that can produce a document the
loader refuses. Cutting the span is the easy half; the model is a GRAPH, and
what is left behind is an actuator driving a joint that no longer exists, an
`<exclude>` naming a body that does not, a tendon routed through a deleted
site — and a keyframe whose `qpos` is now the wrong length.

FOUR ARMS, AND THE FOURTH IS THE ONE THAT CANNOT BE FAKED:

  1. the edit APPLIES and the counts actually drop (a no-op passes every
     other arm);
  2. `leftover_dangling` is EMPTY — the post-condition of every operation in
     `structure.mojo`;
  3. our own loader parses and builds the result;
  4. ⚠⚠ MuJoCo loads the same text and agrees on every count. Arms 1-3 are a
     CLOSED LOOP: a reference we prune wrongly in a way we also read wrongly
     cancels out perfectly. Mojo cannot call MuJoCo, so arm 4 is a second
     step — this test writes the edited documents to /tmp with a manifest,
     and `scripts/check_structural_edits_vs_mujoco.py` judges them.

⚠ THE ZOO FIXTURE EXISTS FOR THE CASCADE. `assets/structural_edit_zoo.xml`
puts a reference to the deleted body in EVERY section that can hold one, and
one of them is INDIRECT: the actuator `cable_m` drives the tendon `cable`,
which is removed only because the site it routes through went with the arm.
`cable` was never inside the deleted span, so a single-pass prune leaves
`tendon='cable'` dangling — a file the editor called done and MuJoCo refuses.

⚠ MESH MODELS ARE ARMS 1-3 ONLY, and this is stated rather than hidden: their
`file=` paths are relative to the model's own directory, so the edited text
cannot be written to /tmp and loaded by MuJoCo without moving the assets. They
still exercise the paths that matter — go2's keyframe, aloha's equality,
softfoot's fifty equalities and five tendons.

Run: pixi run mojo run -I . tests/physics3d/test_structural_edit.mojo
     pixi run python scripts/check_structural_edits_vs_mujoco.py
"""

from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.studio.structure import (
    delete_body, delete_joint, delete_geom, leftover_dangling,
)
from mojo_rl.physics3d.studio.validate import (
    validate_model, SEV_ERROR,
)


comptime DT = DType.float64
comptime OUT_DIR = String("/tmp/physics3d_structural")


struct Case(Copyable, Movable):
    var path: String
    var tag: String
    var target: String
    var judged_by_mujoco: Bool
    """False for mesh models — see the header."""
    var want_notes: Int
    """Minimum prunes this edit must REPORT. 0 means "any"."""
    var breaks_model: String
    """"" when the result must still LOAD; else the diagnostic code it earns.

    ⚠⚠ AN EDIT IS ALLOWED TO PRODUCE AN UNLOADABLE MODEL, and pretending
    otherwise would be the wrong design. Deleting the only geom of a moving
    body is a perfectly reasonable thing to do halfway through a repair, and
    MuJoCo refuses the result — so the requirement is not "every edit stays
    valid", it is "the tool SAYS which one it is". Where this is non-empty the
    gate asserts BOTH halves: our validator names the defect, and MuJoCo
    really does refuse the same text.
    """

    def __init__(out self, path: String, tag: String, target: String,
                 judged: Bool, want_notes: Int,
                 breaks_model: String = String("")):
        self.path = path
        self.tag = tag
        self.target = target
        self.judged_by_mujoco = judged
        self.want_notes = want_notes
        self.breaks_model = breaks_model


def cases() -> List[Case]:
    var c = List[Case]()
    # ── mesh-free, so MuJoCo can judge the edited text ────────────────────
    c.append(Case(String("tests/physics3d/assets/structural_edit_zoo.xml"),
                  String("body"), String("arm"), True, 9))
    c.append(Case(String("mojo_rl/envs/ant/assets/ant.xml"),
                  String("body"), String("front_left_leg"), True, 2))
    c.append(Case(String("mojo_rl/envs/half_cheetah/assets/half_cheetah.xml"),
                  String("body"), String("bthigh"), True, 3))
    # humanoid carries TWO tendons and FIVE keyframes.
    c.append(Case(String("mojo_rl/envs/humanoid/assets/humanoid.xml"),
                  String("body"), String("left_thigh"), True, 4))
    c.append(Case(String("mojo_rl/envs/walker2d/assets/walker2d.xml"),
                  String("joint"), String("thigh_joint"), True, 1))
    # ⚠ THIS ONE BREAKS THE MODEL ON PURPOSE. `bshin` is the shin's only
    # geom, so the body keeps its joint and loses its mass — which MuJoCo
    # refuses. The edit is still the right thing to perform; the validator is
    # what has to say so.
    c.append(Case(String("mojo_rl/envs/half_cheetah/assets/half_cheetah.xml"),
                  String("geom"), String("bshin"), True, 0,
                  String("zero-mass-moving-body")))
    # ── mesh models: arms 1-3 only ────────────────────────────────────────
    c.append(Case(String("references/mujoco_menagerie-main/unitree_go2/scene.xml"),
                  String("body"), String("FL_thigh"), False, 3))
    c.append(Case(String("references/mujoco_menagerie-main/aloha/scene.xml"),
                  String("body"), String("left/left_finger_link"), False, 1))
    c.append(Case(String("references/mujoco_menagerie-main/iit_softfoot/scene.xml"),
                  String("body"), String("sf_phalanx_1_1"), False, 1))
    return c^


struct Counts(Copyable, Movable):
    var nbody: Int
    var njnt: Int
    var ngeom: Int
    var nu: Int
    var nq: Int
    var nv: Int
    var nsite: Int
    var neq: Int
    var ntendon: Int

    def __init__(out self, fmd: FlatModelDef, dims: DynDims):
        self.nbody = dims.get_nbody()
        self.njnt = dims.get_njoint()
        self.ngeom = dims.get_ngeom()
        self.nu = dims.get_nact()
        self.nq = dims.get_nq()
        self.nv = dims.get_nv()
        self.nsite = dims.get_nsite()
        self.neq = len(fmd.equalities)
        self.ntendon = dims.get_ntendon()

    def row(self) -> String:
        return (
            String(self.nbody) + " " + String(self.njnt) + " "
            + String(self.ngeom) + " " + String(self.nu) + " "
            + String(self.nq) + " " + String(self.nv) + " "
            + String(self.nsite) + " " + String(self.neq) + " "
            + String(self.ntendon)
        )


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


def _dir_of(p: String) -> String:
    var i = p.rfind("/")
    return String(p[byte=0:i]) if i > 0 else String(".")


def _read(p: String) raises -> String:
    var f = open(p, "r")
    var s = f.read()
    f.close()
    return s^


def _verts_budget(fmd: FlatModelDef) raises -> Int:
    """The studio's retry-on-raise mesh budget — see the same note there."""
    var verts = 0
    var tries = 0
    while True:
        var dims = dims_from_flat(fmd, nmesh_verts=verts)
        var m = Model[DT, DynDims](dims)
        try:
            build_model_runtime[DT](fmd, dims, m)
            return verts
        except e:
            if String(e).find("mesh vertex capacity") == -1:
                raise e
            tries += 1
            if tries > 24:
                raise e
            verts = 4096 if verts == 0 else verts * 2


def _load_counts(xml: String, base: String) raises -> Counts:
    """Parse AND build — a document that parses but cannot build is not loaded."""
    var fmd = parse_xml_full(xml, base)
    var dims = dims_from_flat(fmd, nmesh_verts=_verts_budget(fmd))
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    return Counts(fmd, dims)


def _error_codes(xml: String, base: String) raises -> String:
    """The ERROR codes `validate_model` reports for this document."""
    var fmd = parse_xml_full(xml, base)
    var dims = dims_from_flat(fmd, nmesh_verts=_verts_budget(fmd))
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var out = String("")
    for d in validate_model(fmd, m):
        if d.severity >= SEV_ERROR:
            out += d.code + " "
    return out^


def main() raises:
    var t = Tally()
    print("=== structural edits: delete, prune, and still load ===")

    var manifest = String("")
    var n_judged = 0

    for c in cases():
        print("---", c.path, ": delete", c.tag, "'" + c.target + "' ---")
        var base = _dir_of(c.path)
        # ⚠ THE EXPANDED TEXT, WHICH IS WHAT THE STUDIO EDITS. A scene file's
        # own source has the robot behind an `<attach>`; deleting a body by
        # name in it would find nothing.
        var src = expand_mjcf(_read(c.path), base)
        var before = _load_counts(src, base)

        var r = delete_body(src, c.target) if c.tag == "body" else (
            delete_joint(src, c.target) if c.tag == "joint"
            else delete_geom(src, c.target)
        )
        t.truth(r.ok, String("the target was found"))

        # ── arm 2: nothing dangles ────────────────────────────────────────
        var left = leftover_dangling(r.xml)
        if len(left) > 0:
            for x in left:
                print("       DANGLING:", x)
        t.truth(len(left) == 0, String("no dangling reference is left behind"))

        # ── the prune reported itself ─────────────────────────────────────
        for note in r.notes:
            print("       note:", note)
        t.truth(len(r.notes) >= c.want_notes,
                String("reported ", len(r.notes), " prune(s), wanted >= ",
                       c.want_notes))

        # ── arm 3: our loader takes it ────────────────────────────────────
        var after = _load_counts(r.xml, base)

        # ── arm 1: it actually changed something ──────────────────────────
        # ⚠ NON-VACUITY. Every other arm is true of a no-op edit: nothing
        # dangles, the model loads, the counts agree with MuJoCo. Only this
        # says the deletion happened.
        var shrank = (
            after.nbody < before.nbody
            or after.njnt < before.njnt
            or after.ngeom < before.ngeom
        )
        t.truth(shrank,
                String("the model SHRANK: nbody ", before.nbody, "->",
                       after.nbody, " njnt ", before.njnt, "->", after.njnt,
                       " ngeom ", before.ngeom, "->", after.ngeom))

        # ── the edit that BREAKS the model, and the tool saying so ────────
        if c.breaks_model.byte_length() > 0:
            var codes = _error_codes(r.xml, base)
            t.truth(codes.find(c.breaks_model) != -1,
                    String("the validator names '", c.breaks_model,
                           "' (got: '", codes, "')"))

        if c.judged_by_mujoco:
            var out_path = OUT_DIR + "/" + String(n_judged) + ".xml"
            var wf = open(out_path, "w")
            wf.write(r.xml)
            wf.close()
            # ⚠ THE MANIFEST CARRIES WHETHER MuJoCo SHOULD ACCEPT IT. Without
            # that column the python half would have to guess, and a case
            # that broke the model would read as a failure of the EDIT.
            var expect_load = String("0") if c.breaks_model.byte_length() > 0 \
                else String("1")
            manifest += out_path + " " + expect_load + " " + after.row() + "\n"
            n_judged += 1

    # ── the cascade, named ────────────────────────────────────────────────
    # ⚠⚠ THE INDIRECT PRUNE IS THE ONE A SINGLE PASS MISSES, so it gets its
    # own arm rather than hiding inside a count. `cable_m` drives the tendon
    # `cable`, which is removed only because the site it routes through went
    # with the arm — `cable` is nowhere in the deleted span.
    print("--- the prune cascades to a SECOND order reference ---")
    var zoo = expand_mjcf(
        _read(String("tests/physics3d/assets/structural_edit_zoo.xml")),
        String("tests/physics3d/assets"),
    )
    var zr = delete_body(zoo, String("arm"))
    var saw_cascade = False
    for note in zr.notes:
        if note.find("tendon='cable'") != -1:
            saw_cascade = True
    t.truth(saw_cascade,
            "the actuator driving the removed TENDON was pruned too")

    # ── the negative control ──────────────────────────────────────────────
    # ⚠ WITHOUT THIS, a `delete_body` that deleted the FIRST body regardless
    # of the name would pass every arm above.
    print("--- a name that is not there ---")
    var ant = expand_mjcf(_read(String("mojo_rl/envs/ant/assets/ant.xml")),
                          String("mojo_rl/envs/ant/assets"))
    var miss = delete_body(ant, String("no_such_body"))
    t.truth(not miss.ok, "a missing target reports ok=False")
    t.truth(miss.xml == ant, "and leaves the document BYTE-IDENTICAL")

    # ⚠ AND THE OTHER HALF OF THE CONTROL: a name that exists as one KIND is
    # not a target of another. Names are scoped per element kind.
    var wrong_kind = delete_joint(ant, String("front_left_leg"))
    t.truth(not wrong_kind.ok,
            "a BODY's name is not found as a <joint> (kinds are scoped)")

    var mf = open(OUT_DIR + "/manifest.txt", "w")
    mf.write(manifest)
    mf.close()

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    print("    wrote", n_judged, "edited documents to", OUT_DIR)
    print("    NOW RUN THE MuJoCo HALF — arms 1-3 are a closed loop:")
    print("    pixi run python scripts/check_structural_edits_vs_mujoco.py")
    if t.fails != 0:
        raise Error(
            "test_structural_edit: " + String(t.fails) + " failed"
        )
