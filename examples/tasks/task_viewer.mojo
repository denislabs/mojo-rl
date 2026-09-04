"""Watch a `.task` run — the task layer, on screen.

    pixi run build-imgui                                          # ONCE
    pixi run mojo run -I . examples/tasks/task_viewer.mojo
    pixi run mojo run -I . examples/tasks/task_viewer.mojo so101_lift_brick 7
    pixi run mojo run -I . examples/tasks/task_viewer.mojo <task> --check

argv only picks which task opens FIRST; **every task is in the sidebar** and
switching is instant. Both argv arguments pick DATA, not code: the same binary
runs every task in `mojo_rl/tasks/tasks/`, which is the claim the whole layer
exists to make.

⚠⚠ A TASK SWITCH DOES NOT REBUILD THE MODEL, and that is the fixed scene
budget paying off in the one place you can watch it. Every task in a family
instantiates every slot, so `nq`/`nv`/`ngeom` are constant across the list —
switching reloads a `.task`, rebinds a goal, and resets. `manipulation`'s
viewer has to tear the renderer down for its switch because each of its tasks
is a different MODEL; this one does not.

⚠⚠ IT OPENS ON THE FREE CAMERA. `so_arm101.xml` ships exactly one camera —
`wrist_cam`, bolted to the wrist — and the renderer opens on `active_camera
= 0`, so without `request_free_camera()` you look down the gripper at whatever
the gripper faces, and dragging cannot fix it: a body-attached camera is
re-aimed EVERY frame, so the mouse fights the model and loses.
`sac_so_arm101_reach_policy_viewer.mojo` records the same trap on the same
asset.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it. CPU physics on purpose: one env at 60 Hz needs no GPU.

## WHAT IT IS FOR, AND IT IS NOT A DEMO

Every gate in `tests/tasks/` either constructs geometry or counts records.
Exactly one steps the physics, and it was written after this viewer's own
question — "is a viewer premature?" — turned up a static fixture welded 50 m in
the air with its region on top and the sampler placing bricks into the sky,
while `nq`, `nv`, `ngeom` and every contact count stayed correct.

So this is the cheapest instrument for the class of defect a task layer
generates: things that are dimensionally right and physically absurd. Watch
the reset, not the motion.

⚠ THE ARM DOES NOT MOVE, AND THAT IS THE POINT OF THE DEFAULT. `zero` drive
leaves the props to settle under gravity from their sampled poses, which is
what is worth looking at. Reaching a goal needs a controller or a trained
policy — see `docs/TASK_LAYER_IMPLEMENTATION.md`'s note on P2's gate — so the
GOAL READOUT here tells you whether the goal is SATISFIABLE and correctly
wired, not whether anything is solving it.

⚠ THE GOAL IS RE-EVALUATED EVERY FRAME and printed on transition. A goal that
holds at reset and then flickers is the sampler and the predicate disagreeing
about the surface — which is a real defect and the reason `On`'s lower bound
is slightly negative (a settled object sits microscopically inside what it
rests on).
"""

from std.random import seed as seed_rng
from std.sys import argv

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
    spec_fields_runtime, read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.render_fields import build_render_fields
from mojo_rl.physics3d.parser.model_def_from_xml import RfOnlyModelDef
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.model.model_renderer import ModelRenderer
from mojo_rl.render.imgui import (
    imgui_shim_available, ig_begin_panel, ig_end, ig_text, ig_text_colored,
    ig_separator_text, ig_selectable, ig_button, ig_spacing,
)
from mojo_rl.physics3d.studio.stepping import StudioRk4Pyr

from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import parse_goal, bind_goal, require_tier_a
from mojo_rl.tasks.eval import eval_goal, region_sites
from mojo_rl.tasks.sampler import (
    sample_placements, RegionFrame, SampleReport,
)
from mojo_rl.tasks.reset import free_slot_addresses, reset_slots


comptime DT = DType.float64

# ⚠ `math3d.Vec3`/`Quat` ARE GENERIC OVER DTYPE and the renderer takes the
# bound ones. `physics_studio.mojo` does the same two lines for the same
# reason; an unbound `Vec3` in a `List` fails with "is not concrete", which
# names the list rather than the import.
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]
comptime FAMILY = "mojo_rl/tasks/families/so101_tabletop.family"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"

# ⚠ THE ARM'S OWN COLLISION HULLS, FROM THE ARM'S OWN CONSTANT. 0 means
# "mesh geoms do not collide" and `fields_build` RAISES rather than letting the
# arm quietly stop colliding. ⚠ Do NOT hand-copy the figure an error quotes:
# `parse_model_runtime` and the batched env's `ModelDims` disagree by ONE hull
# vertex on this very scene, so a number tuned to one path fails on the other.
comptime NMESH_VERTS = SO_ARM101_NMESH_VERTS
comptime MAX_CONTACTS = 32
comptime SIDEBAR_W: Float32 = 300.0


def task_names() -> List[String]:
    """Every `.task` the sidebar offers.

    ⚠ A FUNCTION, NOT A COMPTIME ARRAY — `Array[String, N]` is not
    `ImplicitlyCopyable`, so a comptime table cannot be indexed at runtime and
    the error names materialisation rather than the lookup
    (`studio/scene.mojo`'s `_prop_mjcf_type` records the same trap).

    ⚠ THE LIST IS THE ONLY THING THAT KNOWS ABOUT TASKS AT COMPILE TIME, and
    it holds NAMES. Everything downstream reads the `.task` file, so adding an
    entry here plus a file is the whole cost of a new task — no new type, no
    new config, no rebuild of anything but this list.
    """
    var out = List[String]()
    out.append(String("so101_reach_brick"))
    out.append(String("so101_lift_brick"))
    out.append(String("so101_gather_bricks"))
    return out^


def main() raises:
    var args = argv()
    var task_name = (
        String(args[1]) if len(args) > 1 else String("so101_gather_bricks")
    )
    var run_seed = UInt64(0)
    # ⚠ `--check` IS A HEADLESS SMOKE PATH, not a debug flag. It runs
    # everything except the window: load, compose-check, build, reset, step,
    # evaluate. That is the whole task-layer data path, and it is the part
    # that can be run in CI or over ssh — `physics_studio` carries the same
    # idea as `max_frames` for the same reason.
    var check_only = False
    var seed_arg = -1
    for i in range(1, len(args)):
        if String(args[i]) == "--check":
            check_only = True
        elif i >= 2:
            seed_arg = i
    if seed_arg > 0:
        run_seed = UInt64(Int(String(args[seed_arg])))
    seed_rng(0)

    print("=" * 68)
    print("task viewer —", task_name, " seed", run_seed)
    print("=" * 68)
    if not check_only and not imgui_shim_available():
        print("  ⚠ no Dear ImGui shim — the sidebar will be absent and the")
        print("    task fixed to argv. Build it once with:")
        print("       pixi run build-imgui")
        print("    (the window still opens; only the picker is missing)")

    # ── the task layer: all data, no code ─────────────────────────────────
    var f = load_family(String(FAMILY))
    var names = task_names()
    var cur = 0
    for i in range(len(names)):
        if names[i] == task_name:
            cur = i
    var t = load_task(String(TASK_DIR) + names[cur] + ".task")
    validate_task_against_family(t, f)
    print("  family :", f.name, "|", len(f.slots), "slots,",
          f.n_free_slots(), "free")
    print("  task   :", t.name)
    print("  says   :", t.language)
    print("  goal   :", t.goal)

    # ── the model, loaded at RUN TIME from the composed scene ─────────────
    var path = scene_path(f)
    var fmd = parse_model_runtime(path)
    var dims = dims_from_flat(
        fmd, max_contacts=MAX_CONTACTS, nmesh_verts=NMESH_VERTS
    )
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var integ = StudioRk4Pyr(dims)
    var nb = dims.get_nbody()
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var ns = dims.get_nsite()
    print("  scene  :", path)
    print("           nq", nq, " nv", nv, " nbody", nb, " nsite", ns)

    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
    require_tier_a(g, t.name)
    var rsites = region_sites(f, fmd.site_names)

    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)

    if check_only:
        # One reset + one step + the goal, then out. No window.
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        var sp = List[Float64]()
        for i in range(ns * 3):
            sp.append(Float64(d.site_xpos.data[i]))
        var frames = List[RegionFrame]()
        for i in range(len(f.regions)):
            var s = rsites[i]
            frames.append(RegionFrame(sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2]))
        var radii = List[Float64]()
        for _ in range(len(f.slots)):
            radii.append(0.02)
        var rep = SampleReport()
        var placed = sample_placements(t, f, frames, radii, run_seed, 0, rep)
        var qpos = List[Float64]()
        for i in range(nq):
            qpos.append(Float64(d.qpos.data[i]))
        var qvel = List[Float64]()
        for _ in range(nv):
            qvel.append(0.0)
        reset_slots(t, f, placed, addrs, qpos, qvel)
        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](qpos[i])
        for i in range(nv):
            d.qvel.data[i] = Scalar[DT](qvel[i])
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        print("  reset  :", rep.accepted, "placed in", rep.attempts, "draws")
        for i in range(len(f.slots)):
            if f.slots[i].kind != SLOT_FREE:
                continue
            var a = addrs[i].qadr
            print("           ", f.slots[i].name, "at (", qpos[a], ",",
                  qpos[a + 1], ",", qpos[a + 2], ")",
                  "ACTIVE" if t.is_active(f.slots[i].name) else "parked")
        var xb = List[Float64]()
        for i in range(nb * 3):
            xb.append(Float64(d.xpos.data[i]))
        var xq = List[Float64]()
        for i in range(nb * 4):
            xq.append(Float64(d.xquat.data[i]))
        var sp2 = List[Float64]()
        for i in range(ns * 3):
            sp2.append(Float64(d.site_xpos.data[i]))
        print("  goal at reset ->", eval_goal(g, f, xb, xq, sp2, rsites))
        integ.step["cpu"](d, m)
        print("  one step OK — the composed family simulates")
        _ = sf
        return

    # ── the renderer, over the SAME composed file ─────────────────────────
    # ⚠ `RfOnlyModelDef` + `adopt_rf` is the runtime-model path: the render
    # fields come from the parse we already did, so the picture and the
    # physics cannot be two different scenes.
    var src = read_model_source(path)
    var rf = build_render_fields(parse_xml_full(src[0], src[1]), src[0], src[1])
    var renderer = ModelRenderer[RfOnlyModelDef](
        width=1280, height=800, visual_radius_scale=1.0,
        show_velocity=False,
        title=String("task viewer — ") + f.name,
        adopt_rf=Optional(rf.copy()),
    )
    renderer.init(None)

    # ⚠⚠ FREE CAMERA, AND IT IS NOT A PREFERENCE. `so_arm101.xml` ships exactly
    # one camera — `wrist_cam`, bolted to the wrist — and the renderer opens on
    # `active_camera = 0`, so without this you look down the gripper at
    # whatever the gripper faces. Dragging cannot fix it: a body-attached
    # camera is re-aimed EVERY frame in `render`, so the mouse fights the model
    # and loses. `sac_so_arm101_reach_policy_viewer.mojo` documents the same
    # trap for the same asset.
    renderer.request_free_camera()

    var have_ui = renderer.imgui_init()
    if have_ui:
        renderer.set_ui_sidebar_width(Int(SIDEBAR_W))
        renderer.set_show_hud(False)
    else:
        print("  (no ImGui shim — `pixi run build-imgui` for the sidebar;")
        print("   the task is fixed to argv without it)")

    var positions = List[Vec3]()
    var quats = List[Quat]()
    var episode = 0
    var step = 0
    var last_goal = False
    var lane = 0
    var held = 0          # frames the goal has held, for the readout
    var paused = False

    while renderer.is_open():
        if renderer.check_quit():
            break

        # ── reset at the start of an episode ──────────────────────────────
        if step == 0:
            forward_kinematics["cpu", DT, DynDims, 1](d, m)
            var sp = List[Float64]()
            for i in range(ns * 3):
                sp.append(Float64(d.site_xpos.data[i]))
            # ⚠ REGION FRAMES ARE RESOLVED AFTER FK, EVERY EPISODE. A region
            # rides a site, and a site attached to a movable slot moves — that
            # is the whole reason regions are site-relative.
            var frames = List[RegionFrame]()
            for i in range(len(f.regions)):
                var si = rsites[i]
                frames.append(
                    RegionFrame(sp[si * 3], sp[si * 3 + 1], sp[si * 3 + 2])
                )
            var radii = List[Float64]()
            for _ in range(len(f.slots)):
                radii.append(0.02)

            var rep = SampleReport()
            var placed = sample_placements(
                t, f, frames, radii, run_seed, lane, rep
            )
            var qpos = List[Float64]()
            for i in range(nq):
                qpos.append(Float64(d.qpos.data[i]))
            var qvel = List[Float64]()
            for _ in range(nv):
                qvel.append(0.0)
            reset_slots(t, f, placed, addrs, qpos, qvel)
            for i in range(nq):
                d.qpos.data[i] = Scalar[DT](qpos[i])
            for i in range(nv):
                d.qvel.data[i] = Scalar[DT](qvel[i])
            forward_kinematics["cpu", DT, DynDims, 1](d, m)
            episode += 1
            last_goal = False
            held = 0

        # ⚠ ZERO DRIVE. See the header: the reset is the subject, and an arm
        # sweeping through the scene knocks the props over within a second.
        if not paused:
            integ.step["cpu"](d, m)
            step += 1

        # ── the goal, every frame ─────────────────────────────────────────
        var xb = List[Float64]()
        for i in range(nb * 3):
            xb.append(Float64(d.xpos.data[i]))
        var xq = List[Float64]()
        for i in range(nb * 4):
            xq.append(Float64(d.xquat.data[i]))
        var sp2 = List[Float64]()
        for i in range(ns * 3):
            sp2.append(Float64(d.site_xpos.data[i]))
        var holds = eval_goal(g, f, xb, xq, sp2, rsites)
        if holds:
            held += 1
        if holds != last_goal:
            print("    ep", episode, "step", step, "goal ->", holds)
            last_goal = holds

        # ── the sidebar ───────────────────────────────────────────────────
        var want_task = cur
        var want_reset = False
        if have_ui:
            renderer.imgui_new_frame()
            ig_begin_panel(
                String("tasks"), 0.0, 0.0, SIDEBAR_W,
                Float32(renderer.renderer.height),
            )
            ig_separator_text(String("family"))
            ig_text(f.name + "  (" + String(len(f.slots)) + " slots, "
                    + String(f.n_free_slots()) + " free)")
            ig_text(String("nq ") + String(nq) + "   nv " + String(nv)
                    + "   nbody " + String(nb))

            # ⚠⚠ THIS LIST IS THE WHOLE CLAIM. Every entry is a `.task` FILE;
            # switching reloads data and rebinds a goal. The MODEL is not
            # rebuilt, because the family's scene budget is fixed — which is
            # exactly why a switch is instant here and a `manipulation` task
            # switch has to tear the renderer down.
            ig_separator_text(String("task  (data, no rebuild)"))
            for i in range(len(names)):
                if ig_selectable(names[i], i == cur):
                    want_task = i
            ig_spacing()
            ig_text(String("says: ") + t.language)
            ig_text(String("goal: ") + t.goal)

            ig_separator_text(String("goal"))
            if holds:
                ig_text_colored(
                    String("HOLDS  (") + String(held) + " frames)",
                    0.3, 0.9, 0.4,
                )
            else:
                ig_text_colored(String("not met"), 0.9, 0.5, 0.3)
            # ⚠ THE READOUT SAYS SATISFIABLE AND WIRED, NOT SOLVED. Nothing
            # drives the arm here; a goal that is False for a whole episode is
            # the expected state of `reach` and `lift`.
            ig_text(String("⚠ zero drive — nothing is solving this"))

            ig_separator_text(String("episode"))
            ig_text(String("ep ") + String(episode) + "   step "
                    + String(step) + " / " + String(f.horizon))
            ig_text(String("seed ") + String(run_seed) + "   lane "
                    + String(lane))
            if ig_button(String("reset (next lane)"), -1.0):
                want_reset = True
            if ig_button(String("pause / run"), -1.0):
                paused = not paused
            if ig_button(String("free camera"), -1.0):
                renderer.request_free_camera()
            ig_end()

        # ── draw ──────────────────────────────────────────────────────────
        positions.clear()
        quats.clear()
        for b in range(nb):
            positions.append(Vec3(
                Float64(d.xpos.data[b * 3 + 0]),
                Float64(d.xpos.data[b * 3 + 1]),
                Float64(d.xpos.data[b * 3 + 2]),
            ))
            # ⚠ `Data.xquat` IS (x, y, z, W) AND `Quat` TAKES (W, x, y, z).
            # Two orders, one line. Reading either the other way tips every
            # body by an arbitrary rotation that looks like a physics bug.
            quats.append(Quat(
                Float64(d.xquat.data[b * 4 + 3]),
                Float64(d.xquat.data[b * 4 + 0]),
                Float64(d.xquat.data[b * 4 + 1]),
                Float64(d.xquat.data[b * 4 + 2]),
            ))
        renderer.render(positions, quats)

        # ⚠⚠ THE SWITCH HAPPENS AFTER `render`, NEVER BEFORE. `imgui_new_frame`
        # opened an ImGui frame that only `render` closes; doing work that can
        # raise between them leaves that frame open and the NEXT `NewFrame`
        # asserts. `physics_studio` and `viewer_core` both document the same
        # constraint, and it is why the model swap there is at the loop's end.
        if want_task != cur:
            cur = want_task
            t = load_task(String(TASK_DIR) + names[cur] + ".task")
            validate_task_against_family(t, f)
            g = bind_goal(
                parse_goal(t.goal), f, fmd.body_names, fmd.site_names
            )
            require_tier_a(g, t.name)
            print("  -> task:", t.name, "|", t.goal)
            step = 0
            lane += 1
        elif want_reset:
            step = 0
            lane += 1
        elif step >= f.horizon:
            # A new lane each episode, so consecutive resets show the
            # DISTRIBUTION rather than one draw — the sampler is seeded by
            # (seed, lane).
            step = 0
            lane += 1

    renderer.close()
    print("  closed after", episode, "episode(s)")
    _ = sf
