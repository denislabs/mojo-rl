"""A section may appear MORE THAN ONCE, and MuJoCo merges the repeats.

WHY THIS EXISTS
===============
MJCF allows `<worldbody>`, `<asset>`, `<actuator>`, `<equality>`, `<contact>`,
`<tendon>` and `<keyframe>` to appear several times; MuJoCo merges them. Our
`parse_xml_full` took the FIRST occurrence and stopped, so everything in a
second block was discarded — silently, because a smaller model is not an
error.

Found when the studio's prop writer emitted two `<worldbody>` sections (the
base's and the props'): a five-prop scene loaded as a bare floor, **nbody 1**.
The writer was changed to emit one, which fixed the symptom and left the
parser gap; this is the gap.

⚠⚠ THE FIXTURE HAS TO REPEAT THE SECTIONS ON PURPOSE, because nothing in the
tree does any more. A gate built from our own writers could not fail — the
same reason the mesh-index gate needs an external fixture.

⚠ AND MuJoCo IS THE ORACLE, not our own expectations: "how many bodies should
a two-worldbody model have" is exactly the question we got wrong, so the
answer has to come from the implementation that defines it.

Run: pixi run mojo run -I . tests/physics3d/test_repeated_sections.mojo
"""

from mojo_rl.physics3d.parser.full_parser import parse_xml_full

comptime XML = String(
    '<mujoco model="repeat">'
    '<compiler angle="radian"/>'
    '<asset><material name="m1" rgba="1 0 0 1"/></asset>'
    '<worldbody>'
    '  <geom name="floor" type="plane" size="5 5 0.1"/>'
    '  <body name="a" pos="0 0 1"><freejoint name="ja"/>'
    '    <geom name="ga" type="sphere" size="0.1"/></body>'
    "</worldbody>"
    '<actuator><motor name="m_a" joint="ja" gear="1"/></actuator>'
    # ── everything below is a SECOND block of a section already seen ──
    '<asset><material name="m2" rgba="0 1 0 1"/></asset>'
    '<worldbody>'
    '  <body name="b" pos="1 0 1"><joint name="jb" type="hinge" axis="0 1 0"/>'
    '    <geom name="gb" type="box" size="0.1 0.1 0.1"/>'
    '    <site name="sb" pos="0 0 0.2"/></body>'
    '  <body name="c" pos="2 0 1"><freejoint name="jc"/>'
    '    <geom name="gc" type="capsule" size="0.05 0.2"/></body>'
    "</worldbody>"
    '<actuator><motor name="m_b" joint="jb" gear="2"/></actuator>'
    "</mujoco>"
)


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


def main() raises:
    var t = Tally()
    print("=== repeated sections, against MuJoCo 3.10.0 ===")
    var fmd = parse_xml_full(XML, String(""))
    var nq = 0
    var nv = 0
    for j in fmd.joints:
        nq += j.nq
        nv += j.nv

    # MuJoCo on this exact string: nbody 4, ngeom 4, njnt 3, nu 2, nsite 1,
    # nmat 2, nq 15, nv 13.
    t.eq(len(fmd.bodies) + 1, 4, "nbody (a, b, c + world)")
    t.eq(len(fmd.geoms), 4, "ngeom (floor + 3)")
    t.eq(len(fmd.joints), 3, "njoint")
    t.eq(nq, 15, "nq")
    t.eq(nv, 13, "nv")
    t.eq(len(fmd.actuators), 2, "nact — BOTH <actuator> blocks")
    t.eq(len(fmd.sites), 1, "nsite — declared only in the SECOND worldbody")
    t.eq(len(fmd.materials), 2, "nmat — BOTH <asset> blocks")

    # ⚠ NON-VACUITY, AND IT IS THE WHOLE TEST. Taking only the first block
    # gives nbody 2 / nact 1 / nmat 1 — all still plausible numbers. Naming
    # the elements that exist ONLY in the second block is what distinguishes
    # "merged" from "happened to be right".
    t.checks += 1
    var have_b = False
    var have_c = False
    for n in fmd.body_names:
        if n == "b":
            have_b = True
        if n == "c":
            have_c = True
    if have_b and have_c:
        print("  ok: bodies from the SECOND <worldbody> are present")
    else:
        t.fails += 1
        print("  FAIL: the second <worldbody> was dropped (b", have_b,
              " c", have_c, ")")

    # The second block's actuator must also RESOLVE, not just exist — its
    # `joint="jb"` names a joint declared in the second worldbody.
    t.checks += 1
    var resolved = False
    for i in range(len(fmd.actuators)):
        if fmd.actuator_names[i] == "m_b" and fmd.actuators[i].joint_id >= 0:
            resolved = True
    if resolved:
        print("  ok: the second block's actuator RESOLVED across blocks")
    else:
        t.fails += 1
        print("  FAIL: m_b did not resolve — a zero-force actuator")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_repeated_sections: " + String(t.fails) + " failed")
