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
from mojo_rl.physics3d.parser.xml_parser import (
    resolve_includes, merge_mjcf,
)

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

    # ── every top-level section survives the merge ────────────────────────
    # ⚠⚠ THIS IS THE REGRESSION GUARD FOR A FOUR-TIME BUG. `merge_mjcf`
    # accumulates a HAND-MAINTAINED list, and every entry on it was added the
    # same way: a section was dropped, nothing raised, and it surfaced months
    # later through a model that needed it — <tendon> via fish, <option>'s
    # <flag> via cartpole, <contact> via humanoid_CMU, <keyframe> via
    # ToddlerBot. Four more were still missing on 2026-08-19 (<statistic> in
    # 24 Menagerie files and 11 of ours, <size>, <custom>, <extension>), so
    # the fifth instance was already written and waiting.
    #
    # One fragment carrying EVERY section, merged with a second, and every one
    # must come out the other side.
    print("--- every section survives a merge ---")
    var host = String(
        '<mujoco>'
        '<compiler angle="radian"/><option timestep="0.001"/>'
        '<statistic extent="2" center="0 0 0.5"/><size njmax="500"/>'
        '<visual><global azimuth="90"/></visual>'
        '<default><geom rgba="1 0 0 1"/></default>'
        '<asset><material name="m1" rgba="0 1 0 1"/></asset>'
        '<worldbody><body name="b1"><joint name="j1" type="hinge"/>'
        '<geom name="g1" type="sphere" size="0.1"/>'
        '<site name="s1" pos="0 0 0"/></body></worldbody>'
        '<tendon><fixed name="t1"><joint joint="j1" coef="1"/></fixed></tendon>'
        '<actuator><motor name="a1" joint="j1"/></actuator>'
        '<equality><joint joint1="j1" polycoef="0 1 0 0 0"/></equality>'
        '<sensor><jointpos name="sp1" joint="j1"/></sensor>'
        '<contact><exclude body1="b1" body2="b1"/></contact>'
        '<keyframe><key name="home" qpos="0"/></keyframe>'
        '<custom><numeric name="n1" data="1 2 3"/></custom>'
        '<extension><plugin plugin="mujoco.elasticity.cable"/></extension>'
        '</mujoco>'
    )
    var other = String('<mujoco><worldbody><geom name="floor" type="plane"'
                       ' size="1 1 1"/></worldbody></mujoco>')
    var merged = merge_mjcf(host, other)
    var sections = List[String]()
    sections.append(String("compiler")); sections.append(String("option"))
    sections.append(String("statistic")); sections.append(String("size"))
    sections.append(String("visual")); sections.append(String("default"))
    sections.append(String("asset")); sections.append(String("worldbody"))
    sections.append(String("tendon")); sections.append(String("actuator"))
    sections.append(String("equality")); sections.append(String("sensor"))
    sections.append(String("contact")); sections.append(String("keyframe"))
    sections.append(String("custom")); sections.append(String("extension"))
    var lost = 0
    for sec in sections:
        var open_a = "<" + sec + ">"
        var open_b = "<" + sec + " "
        if merged.find(open_a) == -1 and merged.find(open_b) == -1:
            lost += 1
            print("    LOST: <", sec, ">")
    t.eq(lost, 0, String("sections lost of ", len(sections)))
    # ⚠ NON-VACUITY: a merge that echoed its input verbatim would pass the
    # arm above while merging nothing. The floor comes from the OTHER
    # fragment, so its presence proves both inputs reached the output.
    t.truth(merged.find("floor") != -1,
            "the second fragment's content is in the merge (not an echo)")
    t.truth(merged.find("<numeric") != -1 and merged.find("<plugin") != -1,
            "section CHILDREN survive, not just the wrapper tags")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_include_vs_mujoco: " + String(t.fails) + " failed")
