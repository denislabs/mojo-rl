"""An unresolved name reference must RAISE, not resolve to -1 — the parser gap.

WHY THIS EXISTS
===============
`full_parser`'s behaviour was MIXED, and the silent paths were the dangerous
ones. Measured in the studio plan's §3.2:

    <contact><pair> geom1/geom2   RAISED, naming the geom
    <equality> joint names        RAISED
    <fixed> tendon joint          RAISED
    actuator joint=               **SILENT** -> joint_id = -1
    <equality> body1/body2        **SILENT** -> -1 into the record
    <contact><exclude>            **SILENTLY SKIPPED**

Each silent one produces a model that LOADS and is wrong:

* an actuator with an unresolved `joint=` applies **ZERO FORCE**, and it is
  unrecoverable downstream because -1 is a LEGAL sentinel there ("no joint
  transmission") — nothing after the parser can tell a typo from a
  tendon-driven actuator. The symptom is a limp robot, which reads as a
  control or gain problem;
* an equality naming a missing body silently welds to the WORLDBODY;
* a skipped `<exclude>` leaves a pair COLLIDING where MuJoCo excludes it, and
  `nexclude == 0` against MuJoCo's count has already read as a solver
  divergence in this tree.

⚠⚠ THE WORLDBODY IS WHY TWO OF THESE HID SO LONG. `_find_body_index_by_name`
returns **0** both for the worldbody and for a name it never saw, so `>= 0` —
the obvious test, and the one the joint paths use — is always true for a body.
The check has to be on the NAME.

⚠ CAMERA `target=` IS DELIBERATELY EXEMPT: its resolution to -1 is documented
degradation and MuJoCo accepts a model that names a missing target.

⚠ VERIFIED NON-BREAKING BEFORE LANDING: 180 XML files across `mojo_rl/envs`
and `mujoco_menagerie` were parsed, and none hit a new raise. The strictness
catches genuinely broken references, not a convention this tree relies on.

Run: pixi run mojo run -I . tests/physics3d/test_unresolved_references_raise.mojo
"""

from mojo_rl.physics3d.parser.full_parser import parse_xml_full

comptime _BODY = String(
    '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
    '<body name="a" pos="0 0 1"><joint name="ja" type="hinge" axis="0 1 0"/>'
    '<geom name="ga" type="sphere" size="0.1"/></body>'
    '<body name="b" pos="1 0 1"><joint name="jb" type="hinge" axis="0 1 0"/>'
    '<geom name="gb" type="sphere" size="0.1"/></body></worldbody>'
)


def _model(extra: String) -> String:
    return String("<mujoco>") + _BODY + extra + "</mujoco>"


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

    def raises_naming(mut self, xml: String, needle: String, msg: String):
        """Must raise, AND the message must name the offending reference.

        ⚠ NAMING IS HALF THE FIX. "physics3d: bad model" sends the user back
        to a 50 KB file; the whole point of raising here is that the parser
        still has the string in hand.
        """
        self.checks += 1
        var fired = False
        var named = False
        try:
            _ = parse_xml_full(xml, String(""))
        except e:
            fired = True
            named = String(e).find(needle) != -1
        if fired and named:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg, "— raised:", fired, " named:", named)

    def loads(mut self, xml: String, msg: String):
        self.checks += 1
        try:
            _ = parse_xml_full(xml, String(""))
            print("  ok:", msg)
        except e:
            self.fails += 1
            print("  FAIL:", msg, "—", e)


def main() raises:
    var t = Tally()
    print("=== unresolved references raise ===")

    # ── POSITIVE CONTROLS FIRST. A parser that refused everything would pass
    # every arm below, which is the other way a strictness check can be
    # useless.
    print("--- valid models still load ---")
    t.loads(_model(String(
        '<actuator><motor name="m" joint="ja" gear="1"/></actuator>'
    )), "an actuator naming a real joint")
    t.loads(_model(String(
        '<equality><weld body1="a" body2="b"/></equality>'
    )), "an equality naming real bodies")
    t.loads(_model(String(
        '<contact><exclude body1="a" body2="b"/></contact>'
    )), "an exclude naming real bodies")
    # ⚠ `world` IS A LEGAL TARGET and must not be mistaken for "not found" —
    # the two share index 0, which is the whole reason these were silent.
    t.loads(_model(String(
        '<equality><weld body1="a" body2="world"/></equality>'
    )), "an equality welding to the WORLDBODY by name")

    print("--- and a typo raises ---")
    t.raises_naming(
        _model(String(
            '<actuator><motor name="m" joint="NOPE" gear="1"/></actuator>'
        )),
        String("NOPE"),
        "actuator joint= (was a ZERO-FORCE actuator, silently)",
    )
    t.raises_naming(
        _model(String('<equality><weld body1="NOPE" body2="b"/></equality>')),
        String("NOPE"),
        "equality body1= (would weld to the worldbody instead)",
    )
    t.raises_naming(
        _model(String('<equality><weld body1="a" body2="NOPE"/></equality>')),
        String("NOPE"),
        "equality body2=",
    )
    t.raises_naming(
        _model(String('<contact><exclude body1="NOPE" body2="b"/></contact>')),
        String("NOPE"),
        "contact exclude (the pair would COLLIDE where MuJoCo excludes it)",
    )

    # ── the ones that already raised, so the class stays covered ──────────
    print("--- the three that already raised ---")
    t.raises_naming(
        _model(String('<contact><pair geom1="ga" geom2="NOPE"/></contact>')),
        String("NOPE"),
        "contact pair geom2=",
    )
    t.raises_naming(
        _model(String('<equality><joint joint1="NOPE" joint2="jb"/></equality>')),
        String("NOPE"),
        "equality joint1=",
    )

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_unresolved_references_raise: " + String(t.fails) + " failed"
        )
