"""What `parse_model_runtime` actually loads — the composition, or a stub.

WHY THIS EXISTS
===============
`read_model_source` resolved `<include>` and stopped. `<attach>` was left in
the text, the parser did not know the element, and the model that came back
was whatever the HOST file happened to declare on its own — with no
diagnostic. Measured across every XML in the tree plus Menagerie's scenes:

    iit_softfoot/scene.xml              1 body    (MuJoCo: 51)
    tests/.../fixtures/attach/scene     0 bodies  (its three attaches: 5)
    mujoco/model/hammock                0 bodies  (MuJoCo: 112)
    mujoco/model/humanoid/100_humanoids 0 bodies  (MuJoCo: 1601)

A model with ZERO BODIES is not a subtle degradation. It was found only
because a tendon-wrap gate loaded softfoot through this entry point and got
nothing back.

THE SAME SHAPE, THREE MORE TIMES. `<replicate>`, `<composite>` and
`<flexcomp>` GENERATE bodies from a description. The text walk saw an unknown
element and built only what was written out:

    replicate/bunnies.xml      2 bodies    (MuJoCo: 127)
    sleep/dominos.xml          5 bodies    (MuJoCo: 97)
    hammock/hammock.xml       16 bodies    (MuJoCo: 112)

⚠⚠ AND THE BODY COUNT IS NOT ENOUGH TO SEE IT. Three of them — `helix`,
`bowl`, `container` — matched MuJoCo's nbody EXACTLY while replicating GEOMS
inside a single body: helix 3 geoms against MuJoCo's 101, bowl 7 against 324.
One count agreeing is not the model agreeing.

They now RAISE, naming the element. Nothing in Menagerie uses any of the
three, and nothing outside `.pixi` does either — every file affected is one of
MuJoCo's own samples, which is why this sat unnoticed.

Run: pixi run mojo run -I . tests/physics3d/test_loader_composition.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime
from mojo_rl.physics3d.parser.expander import expand_mjcf


comptime ATTACH_SCENE = String("tests/physics3d/fixtures/attach/scene.xml")
comptime SOFTFOOT = String(
    "references/mujoco_menagerie-main/iit_softfoot/scene.xml"
)
comptime MS_HUMAN = String(
    "references/mujoco_menagerie-main/ms_human_700/scene.xml"
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
            print("  FAIL:", msg, "got", got, "want", want)

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _mesh_paths(path: String) raises -> List[String]:
    """Every DISTINCT `mesh_filename` a model's geoms name, as stored."""
    var fmd = parse_model_runtime(path)
    var seen = List[String]()
    for g in fmd.geoms:
        if g.mesh_filename == "":
            continue
        var dup = False
        for s in seen:
            if s == g.mesh_filename:
                dup = True
        if not dup:
            seen.append(g.mesh_filename)
    return seen^


def _mesh_count(path: String) raises -> Int:
    return len(_mesh_paths(path))


def _missing_meshes(path: String) raises -> Int:
    var n = 0
    for f in _mesh_paths(path):
        try:
            var fh = open(f, "r")
            _ = fh.read_bytes(4)
            fh.close()
        except:
            n += 1
            if n <= 3:
                print("    MISSING:", f)
    return n


def _raises_naming(xml: String, needle: String) -> Bool:
    """Does expanding `xml` raise with `needle` in the message?"""
    try:
        _ = expand_mjcf(xml, String("."))
        return False
    except e:
        return String(e).find(needle) != -1


def main() raises:
    var t = Tally()
    print("=== what parse_model_runtime loads ===")

    # ── the loader expands `<attach>`, not just `<include>` ───────────────
    # ⚠ THROUGH `parse_model_runtime`, deliberately. Calling `expand_mjcf`
    # here would test the expander, which already has its own gate, and would
    # have passed the whole time this bug was live. The regression was in the
    # ENTRY POINT, so the entry point is what this loads through.
    print("--- fixtures/attach/scene.xml through the runtime loader ---")
    var a = parse_model_runtime(ATTACH_SCENE)
    # arm (2 bodies) + cube + cube = 4 declared bodies, 5 geoms with the floor.
    t.eq(len(a.bodies), 4, "nbody-1")
    t.eq(len(a.geoms), 5, "ngeom")
    t.eq(len(a.actuators), 1, "nact — the arm's, which rides along")

    # ⚠ NON-VACUITY. Before the fix this file loaded as the floor alone:
    # `nbody-1 == 0`, `ngeom == 1`, and no error. Asserting the counts are
    # RIGHT is the same arm as asserting the attach happened, but only
    # because the stub is a model, not a failure.
    t.truth(len(a.bodies) > 0,
            "the attached bodies exist at all (the stub had ZERO)")

    print("--- iit_softfoot/scene.xml through the runtime loader ---")
    var s = parse_model_runtime(SOFTFOOT)
    t.eq(len(s.bodies), 50, "nbody-1 (MuJoCo: 51)")
    t.eq(len(s.tendons), 5, "ntendon")
    t.eq(len(s.geoms), 197, "ngeom")

    # ── asset paths survive the composition boundary ─────────────────────
    # ⚠⚠ A `file=` IS RELATIVE TO THE FILE THAT WROTE IT, and both composition
    # paths lost that. `ms_human_700`'s `assets/asset/*.xml` write
    # `file="../geometry/sacrum.stl"` — right from `assets/asset/`, and
    # `ms_human_700/../geometry/` from the model root; all 189 meshes resolved
    # to nothing and the model drew ONLY its tendons. `<attach>` had the
    # mirror-image bug: it rebased by a CWD-relative directory that already
    # contained the base, so every attached mesh came out `base/base/...` and
    # softfoot drew as bare sites.
    #
    # ⚠ THE ORACLE IS THE FILESYSTEM. A path that "looks right" is exactly
    # what both bugs produced; only opening the file settles it.
    print("--- every mesh path resolves to a file that exists ---")
    t.eq(_missing_meshes(SOFTFOOT), 0, "iit_softfoot missing meshes (<attach>)")
    t.eq(_missing_meshes(MS_HUMAN), 0, "ms_human_700 missing meshes (<include>)")
    # ⚠ NON-VACUITY: a model with no mesh assets would report 0 missing while
    # proving nothing.
    t.truth(_mesh_count(MS_HUMAN) > 100,
            String("ms_human_700 declares meshes at all: ",
                   _mesh_count(MS_HUMAN)))
    t.truth(_mesh_count(SOFTFOOT) > 0,
            String("iit_softfoot declares meshes at all: ",
                   _mesh_count(SOFTFOOT)))

    # ── the body generators are refused, each by name ─────────────────────
    print("--- <replicate> / <composite> / <flexcomp> refuse loudly ---")
    var body = String(
        '<worldbody><body name="b"><geom type="sphere" size="0.1"/></body>'
        '</worldbody>'
    )
    t.truth(
        _raises_naming(
            String('<mujoco><worldbody>'
                   '<replicate count="8" offset="0 0 0.1">'
                   '<body name="b"><geom type="sphere" size="0.1"/></body>'
                   '</replicate></worldbody></mujoco>'),
            String("<replicate>"),
        ),
        "<replicate> raises, naming itself",
    )
    t.truth(
        _raises_naming(
            String('<mujoco><worldbody>'
                   '<composite type="rope" count="10 1 1" spacing="0.05"/>'
                   '</worldbody></mujoco>'),
            String("<composite>"),
        ),
        "<composite> raises, naming itself",
    )
    t.truth(
        _raises_naming(
            String('<mujoco><worldbody>'
                   '<flexcomp name="f" type="grid" count="3 3 1"/>'
                   '</worldbody></mujoco>'),
            String("<flexcomp>"),
        ),
        "<flexcomp> raises, naming itself",
    )

    # ⚠⚠ THE NEGATIVE CONTROL. Without it, a check that refused EVERY
    # document would pass all three arms above. The same document minus the
    # generator must go through. (The first draft of these four fixtures
    # carried an `<include file="none.xml"/>` to force the full expansion
    # pass — which made `resolve_includes` raise "cannot open" and ALL FOUR
    # "pass", control included. `_refuse_generators` runs unconditionally, so
    # no include is needed.)
    var clean = String("<mujoco>") + body + "</mujoco>"
    var went_through = False
    try:
        _ = expand_mjcf(clean, String("."))
        went_through = True
    except:
        went_through = False
    t.truth(went_through,
            "the SAME document without a generator expands fine (control)")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_loader_composition: " + String(t.fails) + " failed"
        )
