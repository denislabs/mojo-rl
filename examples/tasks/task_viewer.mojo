"""Watch a `.task` run — the task layer, on screen.

    pixi run mojo run -I . examples/tasks/task_viewer.mojo
    pixi run mojo run -I . examples/tasks/task_viewer.mojo so101_lift_brick
    pixi run mojo run -I . examples/tasks/task_viewer.mojo so101_gather_bricks 7

argv is the TASK NAME and an optional SEED. Both pick DATA, not code: the same
binary runs every task in `mojo_rl/tasks/tasks/`, which is the claim the whole
layer exists to make. Add a `.task` file and it is selectable with no rebuild.

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
from mojo_rl.physics3d.studio.stepping import StudioRk4Pyr

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

# ⚠ THE ARM'S OWN COLLISION HULLS. 0 means "mesh geoms do not collide" and
# `fields_build` RAISES rather than letting the arm quietly stop colliding —
# set it to 1 and read the error, which quotes the number it needs.
comptime NMESH_VERTS = 26198
comptime MAX_CONTACTS = 32


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

    # ── the task layer: all data, no code ─────────────────────────────────
    var f = load_family(String(FAMILY))
    var t = load_task(String(TASK_DIR) + task_name + ".task")
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
        title=String("task viewer — ") + task_name,
        adopt_rf=Optional(rf.copy()),
    )
    renderer.init(None)

    var positions = List[Vec3]()
    var quats = List[Quat]()
    var episode = 0
    var step = 0
    var last_goal = False
    var lane = 0

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
                var s = rsites[i]
                frames.append(
                    RegionFrame(sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2])
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
            print("  ep", episode, "lane", lane, "— placed", rep.accepted,
                  "in", rep.attempts, "draws")
            last_goal = False

        # ⚠ ZERO DRIVE. See the header: the reset is the subject, and an arm
        # sweeping through the scene knocks the props over within a second.
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
        if holds != last_goal:
            print("    step", step, "goal ->", holds)
            last_goal = holds

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

        # A new lane each episode, so consecutive resets show the DISTRIBUTION
        # rather than one draw — the sampler is seeded by (seed, lane).
        if step >= f.horizon:
            step = 0
            lane += 1

    renderer.close()
    print("  closed after", episode, "episode(s)")
    _ = sf
