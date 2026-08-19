"""`<include>` resolution and NAMELESS mesh assets — a Menagerie scene.

WHY THIS EXISTS
===============
Every model in `mujoco_menagerie` ships as a pair: `robot.xml` with the
kinematics, and `scene.xml` with a floor, lights, cameras, a `<contact>`
section and ONE `<include>` of the robot. `scene.xml` is the file you are
meant to open.

Our parser used to STRIP `<include>` (`_strip_include_tags`) rather than
resolve it, and the failure did not look like a missing feature:

    physics3d: <contact><pair> references unknown geom2='torso_collision_0'.

A reference error naming a geom that plainly exists, from a robot that never
loaded — the scene's own `<contact>` survived while the file declaring those
geoms did not. Anyone opening a Menagerie model hits it immediately.

`merge_mjcf` was written for exactly this and had been unreferenced since
phase 1b.5; its docstring names ToddlerBot and this scene/robot split as the
reason `<contact>` and `<keyframe>` are accumulated. `resolve_includes` is
the follow-up it was kept for, and it is also the groundwork for S2's
`<attach>` expander.

⚠⚠ THE ORACLE IS MuJoCo, AND IT HAS TO BE. A merge is a text transformation
whose only observable is what the parser then counts, so checking it against
our own parser proves nothing about whether the merge was faithful — the
"gate sharing its reference implementation is blind" failure. Regenerate with:

    pixi run python -c "import mujoco; m = mujoco.MjModel.from_xml_path(P); \\
        print(m.nbody, m.ngeom, m.nq, m.nv, m.nu, m.nsite, m.ntendon, m.neq)"

⚠ THE MERGE IS STRICT, and that is the point of the second half of this file.
`merge_mjcf` drops any section not in its accumulator list WITHOUT a
diagnostic — three have been lost that way (`<tendon>`, `<option>`'s
`<flag>`, `<contact>`), each to a docstring claim that outlived its
limitation. `resolve_includes` therefore checks every section present in any
input against the output and raises naming the missing one.

Run: pixi run mojo run -I . tests/physics3d/test_include_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, read_model_source,
)
from mojo_rl.physics3d.parser.xml_parser import resolve_includes

comptime SCENE = String(
    "references/mujoco_menagerie-main/toddlerbot_2xc/scene.xml"
)
comptime ROBOT = String(
    "references/mujoco_menagerie-main/toddlerbot_2xc/toddlerbot_2xc.xml"
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

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def main() raises:
    var t = Tally()
    print("=== <include> resolution vs MuJoCo (3.10.0) ===")

    # ── the robot alone, which always worked ──────────────────────────────
    # ⚠ THE CONTROL. Without it, a scene that matched MuJoCo would not say
    # whether the INCLUDE did anything — the two files could have been
    # identical for all this test knows.
    var robot = parse_model_runtime(ROBOT)
    var r_geom = len(robot.geoms)
    print("--- the robot file alone (the control) ---")
    t.eq(len(robot.bodies) + 1, 46, "robot nbody")
    t.eq(r_geom, 71, "robot ngeom")

    # ── the scene, which is what a user opens ─────────────────────────────
    print("--- scene.xml, with the <include> resolved ---")
    var s = parse_model_runtime(SCENE)
    var nq = 0
    var nv = 0
    for j in s.joints:
        nq += j.nq
        nv += j.nv
    t.eq(len(s.bodies) + 1, 46, "nbody")
    t.eq(len(s.geoms), 72, "ngeom")      # 71 robot + the scene's floor
    t.eq(nq, 51, "nq")
    t.eq(nv, 50, "nv")
    t.eq(len(s.actuators), 30, "nact")
    t.eq(len(s.sites), 12, "nsite")
    t.eq(len(s.tendons), 2, "ntendon")
    t.eq(len(s.equalities), 13, "nequality")
    t.eq(len(s.pairs), 91, "npair")      # 65 in the robot + 26 in the scene
    t.eq(s.nkey, 1, "nkey")

    # ⚠ NON-VACUITY, AND IT IS THE WHOLE TEST. If the scene merged to the same
    # thing as the robot, every arm above would pass while the include did
    # nothing. The floor geom and the scene's 26 pairs are what the merge
    # ADDED, and they are the two sections most easily dropped.
    t.truth(len(s.geoms) == r_geom + 1,
            String("the merge ADDED the scene's floor (", r_geom, " -> ",
                   len(s.geoms), ")"))
    t.truth(len(s.pairs) > len(robot.pairs),
            String("the merge ADDED the scene's <contact> pairs (",
                   len(robot.pairs), " -> ", len(s.pairs), ")"))
    t.truth(s.nkey == 1,
            "the ROBOT's <keyframe> survived the merge (it is in the"
            " included file, and the scene is what gets loaded)")

    # ── nameless <mesh> assets ────────────────────────────────────────────
    # ⚠⚠ THE ROBOT WAS INVISIBLE WITHOUT THIS, and nothing raised. MuJoCo:
    # "If omitted, the mesh name equals the file name without the path and
    # extension" (XMLreference, asset-mesh-name, 3.10.0 — the RUNTIME
    # version). ToddlerBot writes the bare `<mesh file="head_visual.stl"/>`
    # form for all 47 of its assets, and requiring `name=` skipped every one:
    # each `mesh="head_visual"` on a geom then resolved to `mesh_id = -1`, and
    # a mesh geom with no mesh DRAWS NOTHING and carries no collision
    # geometry. The robot rendered as its 12 sites and nothing else.
    #
    # The form is common across Menagerie, so this made most of that library
    # silently unloadable — the same shape as the 16-asset cap that once left
    # SO-ARM100's jaw without contact surfaces.
    print("--- <mesh> with no name= ---")
    t.eq(len(s.mesh_asset_names), 47, "nmesh assets")
    var named_by_stem = 0
    for i in range(len(s.mesh_asset_names)):
        var nm = s.mesh_asset_names[i]
        var f = s.mesh_asset_files[i]
        # every one of ToddlerBot's is nameless, so every name must be the
        # file's stem — present in the path, and without the extension.
        if f.find("/" + nm + ".stl") != -1 and nm.find(".") == -1:
            named_by_stem += 1
    t.eq(named_by_stem, 47, "names derived from the file STEM (no path, no ext)")
    var have_head = False
    for i in range(len(s.mesh_asset_names)):
        if s.mesh_asset_names[i] == "head_visual":
            have_head = True
    t.truth(have_head,
            "the name a geom actually references ('head_visual') exists")
    # ⚠ NON-VACUITY: the whole point is that the GEOMS resolve. A mesh table
    # that is right while every geom still holds -1 would pass the arms above.
    var mesh_geoms = 0
    var unresolved = 0
    for g in s.geoms:
        if g.geom_type == 5:
            mesh_geoms += 1
            if g.mesh_id < 0:
                unresolved += 1
    t.truth(mesh_geoms > 40,
            String("the scene has ", mesh_geoms, " mesh geoms (non-vacuous)"))
    t.eq(unresolved, 0, "mesh geoms left at mesh_id = -1")

    # ── the strictness check itself ───────────────────────────────────────
    print("--- strictness ---")
    # A file with no include is returned untouched, not round-tripped through
    # the merger — cheap, and it keeps every single-file model on the exact
    # text it always had.
    var src = read_model_source(ROBOT)
    var same = resolve_includes(src[0], src[1])
    t.truth(same.byte_length() == src[0].byte_length(),
            "a file with no <include> is returned VERBATIM")

    # ⚠ NEGATIVE CONTROL: a missing file must RAISE, naming the file. Silently
    # yielding the host is how "the robot never loaded" became a confusing
    # error about a geom instead of a clear one about a path.
    var raised = False
    try:
        _ = resolve_includes(
            String('<mujoco><include file="nope_does_not_exist.xml"/></mujoco>'),
            String("."),
        )
    except e:
        raised = String(e).find("nope_does_not_exist.xml") != -1
    t.truth(raised, "a missing <include> file raises, NAMING the file")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_include_vs_mujoco: " + String(t.fails) + " failed")
