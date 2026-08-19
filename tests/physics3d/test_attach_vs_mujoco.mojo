"""`<attach>` + `<frame>` expansion, against MuJoCo — studio S2.

WHY THE ORACLE IS MuJoCo AND NOT US
===================================
The studio's scene file IS MJCF: `<asset><model>` declares an asset,
`<attach model= prefix=>` instantiates it, `<frame pos quat>` places it. That
choice was made precisely so **MuJoCo can load the scene file unchanged**, and
this gate is what cashes it in — MuJoCo compiles the composed scene properly
(each sub-model separately, then attached), while we splice TEXT. Two entirely
different routes to the same `mjModel`, which is the only kind of comparison
that can catch a prefixer that misses a reference or a frame folded twice.

Checking our expansion against our own parser would prove nothing; see
`feedback_a_gate_that_shares_its_reference_implementation_is_blind`.

WHAT THE FIXTURE EXERCISES (`fixtures/attach/`)
* TWO INSTANCES OF ONE ASSET — the case prefixing exists for. `cube1_` and
  `cube2_` must not collide on `root`, `box`, `tip`, `free`, the `skin`
  material or the `prop` default class.
* A ROTATED FRAME on cube2, so a frame folded into the wrong operand order
  (`q_child * q_frame` instead of `q_frame * q_child`) shows up.
* A MULTI-BODY asset with a joint and an ACTUATOR (`arm`), because splicing
  only the bodies is the obvious mistake: it yields a model that LOADS, with
  every geom present and no actuator — a prop that is grey and limp rather
  than absent.
* NESTED BODIES, so `<frame>` must touch DIRECT children only. A grandchild's
  pose is already relative to its parent, and transforming it too is the
  classic double-application — which reads as a scaling error, not a
  duplicated rotation.

⚠ THE GOLDEN IS REGENERATED, NOT REMEMBERED:

    pixi run python -c "import mujoco; m=mujoco.MjModel.from_xml_path(P); ..."

and `mj_saveLastXML` prints MuJoCo's own flattening of the same scene, which
is the reference for what the expander should produce.

Run: pixi run mojo run -I . tests/physics3d/test_attach_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.expander import (
    expand_mjcf, check_references,
)
from mojo_rl.physics3d.parser.runtime_load import read_model_source
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.studio.scene import SceneDoc, Instance
from mojo_rl.physics3d.studio.panel import _f

comptime SCENE = String("tests/physics3d/fixtures/attach/scene.xml")


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def eq(mut self, got: Int, want: Int, msg: String):
        self.checks += 1
        if got == want:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "— MuJoCo", want, "we", got)

    def near(mut self, got: Float64, want: Float64, msg: String):
        self.checks += 1
        if abs(got - want) <= 1e-9:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg, "— MuJoCo", want, "we", got)

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def main() raises:
    var t = Tally()
    print("=== <attach> + <frame> vs MuJoCo 3.10.0 ===")

    var src = read_model_source(SCENE)
    var flat = expand_mjcf(src[0], src[1])
    var fmd = parse_xml_full(flat, src[1])
    var nq = 0
    var nv = 0
    for j in fmd.joints:
        nq += j.nq
        nv += j.nv

    # ── counts, against `MjModel.from_xml_path(scene.xml)` ────────────────
    print("--- counts ---")
    t.eq(len(fmd.bodies) + 1, 5, "nbody")
    t.eq(len(fmd.geoms), 5, "ngeom")
    t.eq(nq, 15, "nq")           # free 7 + hinge 1 + free 7
    t.eq(nv, 13, "nv")           # free 6 + hinge 1 + free 6
    t.eq(len(fmd.actuators), 1, "nact")
    t.eq(len(fmd.sites), 3, "nsite")
    t.eq(len(fmd.materials), 2, "nmat")   # ONE per instance, not shared

    # ── names, which is what prefixing is FOR ─────────────────────────────
    print("--- prefixed names ---")
    var want = List[String]()
    want.append(String("world"))
    want.append(String("arm1_base"))
    want.append(String("arm1_link"))
    want.append(String("cube1_root"))
    want.append(String("cube2_root"))
    t.eq(len(fmd.body_names), len(want), "body name count")
    var wrong = 0
    for i in range(len(want)):
        if i >= len(fmd.body_names) or fmd.body_names[i] != want[i]:
            wrong += 1
            print("      body", i, "want", want[i], "got",
                  fmd.body_names[i] if i < len(fmd.body_names) else "-")
    t.eq(wrong, 0, "body names in MuJoCo's order")

    # ⚠ TWO INSTANCES OF ONE ASSET IS THE POINT. If the prefix were dropped
    # both cubes would be named `root`, and the count arm above would still
    # pass — a model with two identically-named bodies loads.
    var c1 = False
    var c2 = False
    for n in fmd.body_names:
        if n == "cube1_root":
            c1 = True
        if n == "cube2_root":
            c2 = True
    t.truth(c1 and c2, "both instances of cube.xml are present and DISTINCT")

    # ── the frame transforms ──────────────────────────────────────────────
    print("--- <frame> folded into the bodies ---")
    # cube1: frame (0.3, 0, 0.5) + body (0, 0, 0.05)
    t.near(fmd.bodies[2].pos_x, 0.3, "cube1 pos.x")
    t.near(fmd.bodies[2].pos_z, 0.55, "cube1 pos.z (frame + body)")
    # cube2: frame (-0.3, 0.1, 0.5), rotated 90 deg about Z
    t.near(fmd.bodies[3].pos_x, -0.3, "cube2 pos.x")
    t.near(fmd.bodies[3].pos_y, 0.1, "cube2 pos.y")
    t.near(fmd.bodies[3].pos_z, 0.55, "cube2 pos.z")
    # ⚠ THE ROTATION IS THE ARM THAT CATCHES AN OPERAND SWAP. `q_frame *
    # q_child` and `q_child * q_frame` agree whenever one of them is identity,
    # which is every UNROTATED fixture — so a scene without this arm would
    # pass with the multiplication backwards.
    # ⚠⚠ THE GOLDEN IS `mjModel.body_quat`, NOT THE FIXTURE'S TEXT. The XML
    # writes `0.7071068` — seven digits — and both MuJoCo and our parser
    # NORMALISE it to 0.70710678118654757, because 0.7071068^2 * 2 is
    # 1.00000004. Transcribing the input as the expected value is how a gate
    # comes to disagree with the very oracle it claims to check: the first
    # draft of this file failed here while the expander was correct.
    t.near(fmd.bodies[3].quat_w, 0.7071067811865476,
           "cube2 quat.w (frame rotation, MuJoCo-normalised)")
    t.near(fmd.bodies[3].quat_z, 0.7071067811865476, "cube2 quat.z")

    # ⚠ NESTED BODY: the frame must NOT reach it. arm1_link is a CHILD of
    # arm1_base, so its pos stays (0, 0, 0.3) — applying the frame twice is
    # the classic double-transform and reads as a scaling error.
    t.near(fmd.bodies[1].pos_z, 0.3, "the nested body is NOT re-transformed")

    # ── the sections that ride along ──────────────────────────────────────
    print("--- non-worldbody sections came too ---")
    t.truth(len(fmd.actuators) == 1,
            "the arm's ACTUATOR survived the splice (bodies-only is the"
            " obvious mistake, and it still loads)")
    var ref_ok = False
    for i in range(len(fmd.actuators)):
        # its transmission must resolve to the arm's ONE hinge, not to -1
        if fmd.actuators[i].joint_id >= 0:
            ref_ok = True
    t.truth(ref_ok,
            "the actuator's joint= reference RESOLVED after prefixing"
            " (an unresolved one is a ZERO-FORCE actuator, silently)")
    t.truth(flat.find("cube1_skin") != -1 and flat.find("cube2_skin") != -1,
            "materials are per-instance, not shared")
    t.truth(flat.find('class="cube1_prop"') != -1,
            "default CLASSES are prefixed, and so are the references to them")

    # ── the reference validator ───────────────────────────────────────────
    # ⚠⚠ NEGATIVE CONTROLS ONLY, because the positive case is every arm above:
    # the scene expands and validates. What needs proving is that the check
    # FIRES — a validator that never raises passes every real model while
    # protecting nothing, and this one guards the failure `full_parser` is
    # measured to be silent about (an actuator whose joint= resolves to -1
    # applies ZERO FORCE and says nothing).
    print("--- the reference validator ---")
    var dangling = String(
        '<mujoco><worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" type="sphere" size="0.1"/></body></worldbody>'
        '<actuator><motor name="m" joint="NOPE"/></actuator></mujoco>'
    )
    var fired = False
    var named = False
    try:
        check_references(dangling)
    except e:
        fired = True
        named = String(e).find("NOPE") != -1
    t.truth(fired, "a dangling joint= reference RAISES")
    t.truth(named, "the message names the reference that dangles")

    # ⚠ AND IT MUST NOT FIRE ON A VALID DOCUMENT, or it would simply refuse
    # every model — the other way a validator can be useless.
    var good = String(
        '<mujoco><default><default class="c"><geom rgba="1 0 0 1"/></default>'
        '</default><worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" class="c" type="sphere" size="0.1"/></body>'
        '</worldbody><actuator><motor name="m" joint="j"/></actuator></mujoco>'
    )
    var clean = True
    try:
        check_references(good)
    except e:
        clean = False
        print("      unexpected:", e)
    t.truth(clean,
            "a valid document passes, INCLUDING its <default class=> which is"
            " the one element that declares with class= rather than name=")

    # ── the scene DOCUMENT round-trips ────────────────────────────────────
    # ⚠⚠ THE WRITER'S OUTPUT MUST BE A FILE MuJoCo LOADS, not merely one our
    # expander reads. That is the whole reason the document is MJCF: it keeps
    # `mjModel` available as the oracle for every later slice. A writer gated
    # only against our own parser would drift from MuJoCo silently and take
    # the oracle with it.
    #
    # This builds the SAME scene as the fixture, programmatically, and
    # requires the same model out the far side.
    print("--- SceneDoc -> MJCF -> expand ---")
    var doc = SceneDoc()
    doc.base_xml = String(
        '  <compiler angle="radian"/>\n'
        '  <worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
        "</worldbody>"
    )
    doc.add_asset(String("cube"), String("cube.xml"))
    doc.add_asset(String("arm"), String("arm.xml"))
    var p_arm = doc.place(String("arm"), 0.0, 0.0, 0.0)
    var p_c1 = doc.place(String("cube"), 0.3, 0.0, 0.5)
    var p_c2 = doc.place(String("cube"), -0.3, 0.1, 0.5)
    # ⚠ THE PREFIXES MUST BE UNIQUE PER INSTANCE, not per asset. Two cubes
    # sharing one prefix is the exact collision prefixing exists to prevent,
    # and it produces a model that LOADS with duplicate names.
    t.truth(p_c1 != p_c2, String("two instances get distinct prefixes ('",
                                 p_c1, "' vs '", p_c2, "')"))
    t.truth(p_arm == "arm1_" and p_c1 == "cube1_" and p_c2 == "cube2_",
            "prefixes are asset-numbered with the trailing underscore MuJoCo"
            " concatenates verbatim")
    doc.instances[2].qw = 0.7071068
    doc.instances[2].qz = 0.7071068

    var text = doc.to_mjcf(String("scene"))
    var dir = String("tests/physics3d/fixtures/attach")
    var flat2 = expand_mjcf(text, dir)
    var fmd2 = parse_xml_full(flat2, dir)
    var nq2 = 0
    for j in fmd2.joints:
        nq2 += j.nq
    t.eq(len(fmd2.bodies) + 1, 5, "round-trip nbody")
    t.eq(len(fmd2.geoms), 5, "round-trip ngeom")
    t.eq(nq2, 15, "round-trip nq")
    t.eq(len(fmd2.actuators), 1, "round-trip nact")
    t.near(fmd2.bodies[2].pos_z, 0.55, "round-trip cube1 pos.z")
    t.near(fmd2.bodies[3].quat_w, 0.7071067811865476, "round-trip cube2 quat")

    # ⚠⚠ NEGATIVE COORDINATES, AND THE ARM EXISTS BECAUSE THEY WERE WRONG.
    # cube2 is at x = -0.3 and the writer emitted **-1.7**: the display
    # formatter split the scaled integer with `//` and `%`, which FLOOR, so
    # -3000 // 10000 is -1 and -3000 % 10000 is 7000. Every negative
    # coordinate in the inspector read as a different, plausible number.
    # Found only by writing a scene to disk and comparing it with what was
    # asked for — no in-memory test would have looked at the text.
    t.near(fmd2.bodies[3].pos_x, -0.3, "a NEGATIVE coordinate survives the writer")
    t.truth(_f(-0.3) == "-0.3000",
            String("the display formatter handles negatives ('", _f(-0.3),
                   "', not '-1.7000')"))
    t.truth(_f(-1.25) == "-1.2500" and _f(1.25) == "1.2500"
            and _f(0.0) == "0.0000",
            "and still handles the cases that always worked")

    # remove() is what S3's delete needs, and it must not disturb the rest.
    doc.remove(p_c1)
    t.eq(len(doc.instances), 2, "remove() drops exactly one instance")
    t.truth(doc.find(p_c1) < 0 and doc.find(p_c2) >= 0,
            "the REMAINING instance keeps its identity after a delete")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_attach_vs_mujoco: " + String(t.fails) + " failed")
