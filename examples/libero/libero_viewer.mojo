"""Watch a LIBERO task run, and DRIVE IT BY HAND — the port, on screen.

    pixi run build-imgui                                        # ONCE
    pixi run libero-viewer
    pixi run mojo run -I . examples/tasks/libero_viewer.mojo libero_spatial
    pixi run mojo run -I . examples/tasks/libero_viewer.mojo libero_goal put_the_bowl_on_the_plate 7
    pixi run mojo run -I . examples/tasks/libero_viewer.mojo libero_goal --check
    pixi run mojo run -I . examples/tasks/libero_viewer.mojo libero_goal --shot 30

argv picks the FAMILY (first argument) and which of its tasks opens first;
every task of that family is in the sidebar and switching is instant,
because a family's scene budget is fixed — a switch reloads a `.task` and
rebinds a goal, it does not rebuild the model. Switching FAMILY does rebuild
everything, so it is an argument rather than a button.

## ⚠⚠ THE ARM IS HELD BY THE REAL CONTROLLER, NOT BY ZERO DRIVE

`task_viewer.mojo` drives its SO-101 with zero torque on purpose: that arm
holds itself and the reset is the subject. **A Panda does not.** With zero
ctrl it collapses under gravity in about a second and you are watching a
dropped robot, not a task. So this viewer runs `OSC_POSE` — the same
`osc_pose_gpu` kernel functions the batched controller uses, one lane — at
the benchmark's clocks: 20 Hz policy, 2 ms physics, 25 substeps, `set_goal`
on the first substep of each.

That makes the viewer an instrument for L4 as well as L3. The **DRIVE**
buttons send a delta action exactly as a policy would (±1 on one axis,
which `scale_action` maps to 5 cm), so you can walk the gripper to the bowl
and watch the goal readout flip. `hold` sends the zero action, which is the
controller cancelling gravity and holding the reset pose — if THAT drifts,
the controller is wrong, and you can see it without a gate.

⚠ THE GRIPPER RAMP IS PER SUBSTEP. `close`/`open` set the seventh action
component; the ramp moves 0.01 per sim step, so a policy step is 0.25 and
the fingers take four to go from neutral to shut. Holding the button for a
few frames is the correct way to close it.

## ⚠ WHAT IT IS FOR

Same as `task_viewer`: the class of defect a task layer generates is things
that are dimensionally right and physically absurd, and the cheapest
instrument for those is an eye. `nq`, `nv` and every contact count were
correct while a fixture hung 50 m in the air.

For LIBERO specifically, the things worth watching are the ones the gates
can only assert numerically: whether a sampled bowl lands ON the table
rather than inside it, whether the drawer's box region sits where the
drawer is, and whether the goal flickers as a prop settles.

⚠ THE GOAL READOUT IS THE FULL L3 LANGUAGE. Box regions, `Joint`, and
`On(obj, obj)` all read state a pose alone does not carry — the site table,
the contact list, `qpos` — so this evaluates through `HostState` with
contacts from `detect_contacts`, exactly as
`examples/tasks/libero_demo_success.mojo` does against the benchmark's own
demonstrations.

⚠ RUN THIS ON THE LAPTOP. It opens an SDL3 window and blocks on it. CPU
physics: one env at 60 Hz needs no GPU.

## ⚠ WHAT IT DRAWS IS ROBOSUITE'S PICTURE, NOT MuJoCo's DEFAULT

`renderer.set_group_shown(0, False)`: robosuite renders with
`render_collision_mesh=False` (`environments/base.py`: `vopt.geomgroup[0] = 0`),
and so does every LIBERO camera observation. It matters here more than it
sounds — the Panda's group-0 collision meshes are the SAME surfaces as its
visual meshes, so with both drawn every link z-fights into a speckle, and the
bowl's convex hull is forty translucent boxes that the solid pass draws opaque.
The MuJoCo default (groups 0-2) is what `physics_studio` shows, with a
checkbox per group.

`--shot N` renders N frames, writes `screenshot_<k>.jpg` in the cwd and quits
— the instrument for comparing against `mujoco.Renderer` on the same scene
without sitting at the window. `--cam-free ex ey ez tx ty tz` puts the free
camera at an exact eye/target, so the two images share a pose: a MuJoCo free
camera (`lookat L, distance d, azimuth a, elevation e`) has
`eye = L - d * (cos e cos a, cos e sin a, sin e)` and `target = L`.
"""

from std.os import listdir
from std.random import seed as seed_rng
from std.sys import argv

from noeira.math3d import Vec3 as Vec3G, Quat as QuatG
from noeira.physics3d.fields import Data, Model, DynDims, DynamicsScratch
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
    spec_fields_runtime, read_model_source,
)
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.render_fields import build_render_fields
from noeira.physics3d.parser.model_def_from_xml import RfOnlyModelDef
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.model.model_renderer import ModelRenderer
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
)
from noeira.render.imgui import (
    imgui_shim_available, ig_begin_panel, ig_end, ig_text, ig_text_colored,
    ig_separator_text, ig_selectable, ig_button, ig_spacing, ig_same_line,
    ig_toggle_button,
)
from noeira.physics3d.studio.stepping import StudioIntegEll
from noeira.physics3d.dynamics.actuation import apply_actions_fields
from noeira.physics3d.dynamics.osc_pose import (
    OscPose, OscPoseConfig, ARM_DOF, OSC_ACTION_DIM,
)

from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
    TaskSpec, FamilySpec,
)
from noeira.tasks.family import scene_path
from noeira.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses,
)
from noeira.tasks.eval import (
    eval_goal, HostState, region_sites, region_contact_bodies,
)
from noeira.tasks.sampler import (
    sample_placements, sample_joint_inits, RegionFrame, SampleReport,
)
from noeira.tasks.reset import (
    free_slot_addresses, reset_slots, SlotAddress,
    joint_init_addresses, joint_init_dof_addresses, apply_joint_inits,
)
from noeira.tasks.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime DT = DType.float64
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]
comptime TASK_DIR = "noeira/tasks/tasks/"
comptime FAMILY_DIR = "noeira/tasks/families/"
comptime SIDEBAR_W: Float32 = 340.0
comptime SUBSTEPS = 25
"""LIBERO's clocks: `control_freq` 20 against a 2 ms `timestep`."""
comptime PROP_RADIUS: Float64 = 0.02


def task_names(suite: String) raises -> List[String]:
    """Every `<suite>__*.task` on disk, sorted — the sidebar's list.

    ⚠ READ FROM THE DIRECTORY, NOT A COMPTIME TABLE. `task_viewer` hardcodes
    three names because three is all its family has; a LIBERO suite's task
    list is GENERATED (`pixi run libero-family <suite>`) and changes when the
    goal language grows, so a hardcoded list here would go stale silently and
    show fewer tasks than exist.
    """
    var out = List[String]()
    var want = suite + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("libero viewer: no '" + want + "' in the composed scene")


def do_reset(
    mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims],
    mut osc: OscPose,
    mut scratch: DynamicsScratch[DT, DynDims, 1],
    t: TaskSpec, f: FamilySpec,
    addrs: List[SlotAddress], rsites: List[Int],
    jq: List[Int], jv: List[Int],
    nq: Int, nv: Int, ns: Int, run_seed: UInt64, lane: Int,
) raises -> SampleReport:
    """One episode's reset: the family's base pose, a sampled placement per
    free slot, and the controller re-anchored on the result.

    ⚠ A TOP-LEVEL FUNCTION, NOT A NESTED ONE. A nested `def` in Mojo cannot
    infer a capture convention and the error names the loop rather than the
    closure — `task_viewer.mojo`'s `fresh_state` records the same trap.
    """
    # ⚠⚠ `base_qpos` FIRST. The Panda at qpos 0 self-collides (18 contacts at
    # rest, L2's finding); the family carries the reset pose and the MuJoCo
    # oracle reports 0 contacts only when it is applied.
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    # ⚠⚠ THE JOINT INITS COME FIRST, BEFORE FK AND BEFORE THE REGION FRAMES.
    # Opening a drawer MOVES the region an object is placed into: the top
    # drawer's interior site rides the sliding body. Sampling first and opening
    # afterwards leaves the bowl at the CLOSED drawer's interior position and
    # the drawer slides out from under it — 32 contacts, and only at some seeds.
    #
    # `bddl_base_domain._reset_internal` does exactly this and says why: it runs
    # every `OpenCloseSampler`, then calls `mujoco.mj_step1` — "we manually do
    # this stepping" — and only then `placement_initializer.sample()`.
    var jvals0 = sample_joint_inits(t, run_seed, lane)
    for k in range(len(jvals0)):
        d.qpos.data[jq[k]] = Scalar[DT](jvals0[k])
        d.qvel.data[jv[k]] = Scalar[DT](0)
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var sp = List[Float64]()
    for i in range(ns * 3):
        sp.append(Float64(d.site_xpos.data[i]))
    # ⚠ REGION FRAMES AFTER FK, EVERY EPISODE — a region rides a site, and a
    # site on a movable slot moves.
    var frames = List[RegionFrame]()
    for i in range(len(f.regions)):
        var si = rsites[i]
        frames.append(RegionFrame(sp[si * 3], sp[si * 3 + 1], sp[si * 3 + 2]))
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(PROP_RADIUS)
    var rep = SampleReport()
    var placed = sample_placements(t, f, frames, radii, run_seed, lane, rep)
    var qpos = List[Float64]()
    for i in range(nq):
        qpos.append(Float64(d.qpos.data[i]))
    var qvel = List[Float64]()
    for _ in range(nv):
        qvel.append(0.0)
    reset_slots(t, f, placed, addrs, qpos, qvel)
    # ⚠ RE-APPLIED, because `reset_slots` rebuilt `qpos` from `d` BEFORE the
    # placements and the joint values must survive into the vector it returns.
    apply_joint_inits(t, jq, jvals0, qpos, qvel, jv)
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](qpos[i])
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](qvel[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    # the controller's goal pose and nullspace target are the RESET pose
    osc.update(d, m, scratch)
    osc.reset(d, m)
    return rep^


def read_state(
    d: Data[DT, DynDims, 1], nb: Int, ns: Int, nq: Int,
    site_body_tab: List[Int], site_quat_tab: List[Float64],
    body_parent_tab: List[Int],
) -> HostState:
    """Everything the L3 language reads, out of `Data` — poses, `qpos`, the
    site table and the lane's contact list. `detect_contacts` must have run."""
    var st = HostState(List[Float64](), List[Float64](), List[Float64]())
    for k in range(nb * 3):
        st.xpos.append(Float64(d.xpos.data[k]))
    for k in range(nb * 4):
        st.xquat.append(Float64(d.xquat.data[k]))
    for k in range(ns * 3):
        st.site_xpos.append(Float64(d.site_xpos.data[k]))
    for k in range(nq):
        st.qpos.append(Float64(d.qpos.data[k]))
    st.site_body = site_body_tab.copy()
    st.site_quat = site_quat_tab.copy()
    st.body_parent = body_parent_tab.copy()
    var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    if ncon > LIBERO_GOAL_MAX_CONTACTS:
        ncon = LIBERO_GOAL_MAX_CONTACTS
    st.ncon = ncon
    for k in range(ncon):
        st.con_a.append(Int(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_A]))
        st.con_b.append(Int(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_B]))
    return st^


def main() raises:
    var args = argv()
    var suite = String("libero_goal")
    var first_task = String("")
    var check_only = False
    var run_seed = UInt64(0)
    var positional = 0
    var shot_after = -1
    var cam_free = List[Float64]()
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--check":
            check_only = True
        elif s == "--shot":
            if i + 1 >= len(args):
                raise Error("libero viewer: --shot takes a frame count")
            shot_after = Int(String(args[i + 1]))
            i += 1
        elif s == "--cam-free":
            if i + 6 >= len(args):
                raise Error("libero viewer: --cam-free takes ex ey ez tx ty tz")
            for k in range(6):
                cam_free.append(Float64(String(args[i + 1 + k])))
            i += 6
        elif s.startswith("--"):
            # ⚠ REFUSED, NOT SKIPPED: a mistyped flag that falls through to
            # the positionals runs a different task for an hour.
            raise Error("libero viewer: unknown option '" + s + "'")
        elif positional == 0:
            suite = s
            positional += 1
        elif positional == 1:
            first_task = s
            positional += 1
        else:
            run_seed = UInt64(Int(s))
        i += 1
    seed_rng(0)

    print("=" * 72)
    print("LIBERO viewer —", suite, " seed", run_seed)
    print("=" * 72)
    if not check_only and not imgui_shim_available():
        print("  ⚠ no Dear ImGui shim — no sidebar and no DRIVE buttons.")
        print("    Build it once:  pixi run build-imgui")

    var f = load_family(String(FAMILY_DIR) + suite + ".family")
    var names = task_names(suite)
    if len(names) == 0:
        raise Error(
            "libero viewer: no tasks for '" + suite + "' under " + TASK_DIR
            + " — generate them with `pixi run libero-family " + suite + "`"
        )
    var cur = 0
    if first_task.byte_length() > 0:
        for i in range(len(names)):
            if names[i] == first_task or names[i] == suite + "__" + first_task:
                cur = i
    var t = load_task(String(TASK_DIR) + names[cur] + ".task")
    validate_task_against_family(t, f)
    print("  family :", f.name, "|", len(f.slots), "slots,",
          f.n_free_slots(), "free |", len(f.regions), "regions")
    print("  tasks  :", len(names), "in the sidebar")
    print("  task   :", t.name)
    print("  says   :", t.language)
    print("  goal   :", t.goal)

    # ── the model ─────────────────────────────────────────────────────────
    var path = scene_path(f)
    var fmd = parse_model_runtime(path)
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
    )
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(
                fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var scratch = DynamicsScratch[DT, DynDims, 1](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    # ⚠ ELLIPTIC, like the family. robosuite's `base.xml` sets
    # `cone="elliptic"`; a pyramidal viewer would show different friction
    # from every gate in the tree.
    var integ = StudioIntegEll(dims)
    var nb = dims.get_nbody()
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var ns = dims.get_nsite()
    var nact = dims.get_nact()
    print("  scene  :", path)
    print("           nq", nq, " nv", nv, " nbody", nb, " nsite", ns,
          " mesh verts", verts)

    # ── the goal, and everything its evaluation reads ─────────────────────
    var nqs = List[Int]()
    for i in range(len(fmd.joints)):
        nqs.append(fmd.joints[i].nq)
    var jadr = joint_qpos_addresses(nqs)
    var g = bind_goal(
        parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
        fmd.joint_names, jadr,
    )
    require_tier_a(g, t.name)
    var rsites = region_sites(f, fmd.site_names)
    var rcontact = region_contact_bodies(f, fmd.body_names)
    var site_body_tab = List[Int]()
    var site_quat_tab = List[Float64]()
    for i in range(len(fmd.sites)):
        site_body_tab.append(fmd.sites[i].body_id)
        site_quat_tab.append(fmd.sites[i].quat_x)
        site_quat_tab.append(fmd.sites[i].quat_y)
        site_quat_tab.append(fmd.sites[i].quat_z)
        site_quat_tab.append(fmd.sites[i].quat_w)
    var body_parent_tab = List[Int]()
    body_parent_tab.append(-1)
    for i in range(len(fmd.bodies)):
        body_parent_tab.append(fmd.bodies[i].parent)

    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)
    # ⚠ RESOLVED ONCE, AND IT RAISES HERE rather than at the first reset: a
    # `jinit=` naming a joint the scene does not have is a shut drawer with a
    # bowl in it, and the cause is nowhere near the symptom.
    var jq_adr = joint_init_addresses(t, fmd.joint_names, jq)
    var jv_adr = joint_init_dof_addresses(t, fmd.joint_names, jv)

    # ── the controller ────────────────────────────────────────────────────
    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for i in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[i].nq
        da += fmd.joints[i].nv
    var dof = List[Int]()
    var qadr = List[Int]()
    var jidx = List[Int]()
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ji = _index(fmd.joint_names, String("robot_joint") + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
        jidx.append(ji)
        var ai = _index(fmd.actuator_names, String("robot_torq_j") + String(j + 1))
        act_idx.append(ai)
        tmin.append(fmd.actuators[ai].ctrl_min)
        tmax.append(fmd.actuators[ai].ctrl_max)
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var site_body = fmd.sites[site].body_id
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))
    var osc = OscPose(
        dof^, qadr^, jidx^, tmin^, tmax^, act_idx.copy(), site, site_body,
        ga1, ga2,
        fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
        fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
        OscPoseConfig(), nact, nq, nv,
    )
    var act = List[Scalar[DT]](length=nact if nact > 0 else 1, fill=Scalar[DT](0))

    var lane = 0

    # ── headless smoke: reset, one policy step, the goal ──────────────────
    if check_only:
        var rep = do_reset(d, m, osc, scratch, t, f, addrs, rsites, jq_adr, jv_adr, nq, nv, ns, run_seed, 0)
        print("  reset  :", rep.accepted, "placed in", rep.attempts, "draws")
        for i in range(len(f.slots)):
            if f.slots[i].kind != SLOT_FREE:
                continue
            var a = addrs[i].qadr
            print("           ", f.slots[i].name, "at (",
                  Float64(d.qpos.data[a]), ",", Float64(d.qpos.data[a + 1]),
                  ",", Float64(d.qpos.data[a + 2]), ")",
                  "ACTIVE" if t.is_active(f.slots[i].name) else "parked")
        var zero = List[Float64](length=OSC_ACTION_DIM, fill=0.0)
        zero[6] = -1.0
        var p0 = osc.ee_pos(d)
        for s in range(SUBSTEPS):
            osc.update(d, m, scratch)
            if s == 0:
                osc.set_goal(zero, d, m)
            var ctrl = osc.run(zero, d, m, scratch)
            for i in range(nv):
                d.qfrc.data[i] = Scalar[DT](0)
            apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
            integ.step["cpu"](d, m)
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        detect_contacts["cpu", DT, DynDims, 1](d, m)
        var p1 = osc.ee_pos(d)
        var drift = 0.0
        for k in range(3):
            drift += (p1[k] - p0[k]) * (p1[k] - p0[k])
        print("  one policy step under the HOLD action:")
        var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        print("           grip site moved", drift ** 0.5, "m;", ncon, "contacts")
        # ⚠ THE PAIRS, NOT JUST THE COUNT, AND FROM THE FIRST ONE. A count
        # tells you a reset is contact-heavy and not WHY: when the
        # fixture-region inits first landed, two tasks jumped from 4 contacts to
        # 56 and 64 (the cap) and only the pair list said which body was buried
        # in which. MuJoCo reports ZERO for both composed scenes with the arm at
        # base_qpos and the props parked, so any contact here is worth a name.
        if ncon > 0:
            print("           contact pairs (first 12):")
            for ci in range(ncon if ncon < 12 else 12):
                var ba = Int(d.contacts.data[ci * CONTACT_SIZE + CONTACT_IDX_BODY_A])
                var bb = Int(d.contacts.data[ci * CONTACT_SIZE + CONTACT_IDX_BODY_B])
                # ⚠ NO OFF-BY-ONE. `fmd.body_names` INCLUDES the worldbody at
                # index 0 (`body_names_in_order`: "index 0 is the worldbody"),
                # so a contact's body id indexes it DIRECTLY. Subtracting one —
                # which the first version of this did, by analogy with
                # `body_parent_tab` — names the body before the real one, and
                # every pair reads plausibly: it blamed a fingertip and a stove
                # knob for contacts that were somewhere else entirely.
                var na = String(fmd.body_names[ba]) if ba < len(fmd.body_names) else String("?")
                var nb = String(fmd.body_names[bb]) if bb < len(fmd.body_names) else String("?")
                print("             ", na, "<->", nb)
        var st = read_state(d, nb, ns, nq, site_body_tab, site_quat_tab, body_parent_tab)
        print("  goal   ->", eval_goal(g, f, st, rsites, rcontact))
        print("  ok: the family resets, the controller holds, the goal reads")
        return

    # ── the window ────────────────────────────────────────────────────────
    var src = read_model_source(path)
    var rf = build_render_fields(parse_xml_full(src[0], src[1]), src[0], src[1])
    var renderer = ModelRenderer[RfOnlyModelDef](
        width=1440, height=900, visual_radius_scale=1.0,
        show_velocity=False,
        title=String("LIBERO — ") + f.name,
        adopt_rf=Optional(rf.copy()),
    )
    renderer.init(None)
    # robosuite's `render_collision_mesh=False` — see the module docstring.
    renderer.set_group_shown(0, False)
    # ⚠ FREE CAMERA. The composed scene ships LIBERO's `agentview` and
    # `frontview` plus the Panda's two, and the renderer opens on camera 0 —
    # a body-attached one is re-aimed every frame, so the mouse would fight
    # it. `task_viewer` records the same trap on the SO-101's wrist cam.
    renderer.request_free_camera()
    if len(cam_free) == 6:
        renderer.set_free_camera(
            Vec3(cam_free[0], cam_free[1], cam_free[2]),
            Vec3(cam_free[3], cam_free[4], cam_free[5]),
        )

    var have_ui = renderer.imgui_init()
    if have_ui:
        renderer.set_ui_sidebar_width(Int(SIDEBAR_W))
        renderer.set_show_hud(False)

    var positions = List[Vec3]()
    var quats = List[Quat]()
    var episode = 0
    var step = 0
    var last_goal = False
    var held = 0
    var paused = False
    var grip_close = False
    # the DRIVE action, held until the button is released (one frame)
    var drive = List[Float64](length=OSC_ACTION_DIM, fill=0.0)
    var ever_held = False

    _ = do_reset(d, m, osc, scratch, t, f, addrs, rsites, jq_adr, jv_adr, nq, nv, ns, run_seed, lane)
    episode = 1

    var frame = 0
    while renderer.is_open():
        if renderer.check_quit():
            break
        if shot_after >= 0 and frame == shot_after:
            renderer.request_screenshot()
        if shot_after >= 0 and frame > shot_after:
            break
        frame += 1

        # ── one POLICY step per frame, at the benchmark's clocks ──────────
        if not paused:
            var action = List[Float64](length=OSC_ACTION_DIM, fill=0.0)
            for k in range(6):
                action[k] = drive[k]
            action[6] = 1.0 if grip_close else -1.0
            for s in range(SUBSTEPS):
                osc.update(d, m, scratch)
                if s == 0:
                    osc.set_goal(action, d, m)
                var ctrl = osc.run(action, d, m, scratch)
                for i in range(nv):
                    d.qfrc.data[i] = Scalar[DT](0)
                apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
                integ.step["cpu"](d, m)
            step += 1
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        detect_contacts["cpu", DT, DynDims, 1](d, m)
        # ⚠ THE DRIVE IS ONE POLICY STEP, NOT A LATCH. A held button re-arms
        # it next frame; releasing it stops the arm where it is, which is
        # what makes the buttons usable for aiming.
        for k in range(6):
            drive[k] = 0.0

        var st = read_state(d, nb, ns, nq, site_body_tab, site_quat_tab, body_parent_tab)
        var holds = eval_goal(g, f, st, rsites, rcontact)
        if holds:
            held += 1
            ever_held = True
        if holds != last_goal:
            print("    ep", episode, "step", step, "goal ->", holds)
            last_goal = holds

        var want_task = cur
        var want_reset = False
        if have_ui:
            renderer.imgui_new_frame()
            ig_begin_panel(
                String("libero"), 0.0, 0.0, SIDEBAR_W,
                Float32(renderer.renderer.height),
            )
            ig_separator_text(String("family"))
            ig_text(f.name + "  (" + String(len(f.slots)) + " slots, "
                    + String(f.n_free_slots()) + " free)")
            ig_text(String("nq ") + String(nq) + "   nv " + String(nv)
                    + "   contacts " + String(st.ncon))

            ig_separator_text(String("task  (data, no rebuild)"))
            for i in range(len(names)):
                var short = String(names[i])
                var c2 = short.find("__")
                if c2 >= 0:
                    var cut2 = String(short[byte = c2 + 2 :])
                    short = cut2^
                if ig_selectable(short, i == cur):
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
            elif ever_held:
                ig_text_colored(
                    String("not met  (held ") + String(held)
                    + " frames earlier)", 0.9, 0.8, 0.3,
                )
            else:
                ig_text_colored(String("not met"), 0.9, 0.5, 0.3)

            # ⚠⚠ THE DRIVE IS A REAL POLICY ACTION. Each button is +-1 on one
            # axis of the OSC delta, which `scale_action` maps to 5 cm — the
            # same number a trained policy's saturated output would produce.
            ig_separator_text(String("drive  (OSC_POSE, +-5 cm / step)"))
            if ig_button(String("-X"), 70.0):
                drive[0] = -1.0
            ig_same_line()
            if ig_button(String("+X"), 70.0):
                drive[0] = 1.0
            ig_same_line()
            if ig_button(String("-Y"), 70.0):
                drive[1] = -1.0
            ig_same_line()
            if ig_button(String("+Y"), 70.0):
                drive[1] = 1.0
            if ig_button(String("-Z (down)"), 145.0):
                drive[2] = -1.0
            ig_same_line()
            if ig_button(String("+Z (up)"), 145.0):
                drive[2] = 1.0
            if ig_toggle_button(
                String("close gripper") if not grip_close
                else String("opening..."),
                grip_close, -1.0,
            ):
                grip_close = not grip_close
            var ee = osc.ee_pos(d)
            ig_text(String("grip site  ") + String(ee[0])[byte=0:6] + "  "
                    + String(ee[1])[byte=0:6] + "  " + String(ee[2])[byte=0:6])

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

        positions.clear()
        quats.clear()
        for b in range(nb):
            positions.append(Vec3(
                Float64(d.xpos.data[b * 3 + 0]),
                Float64(d.xpos.data[b * 3 + 1]),
                Float64(d.xpos.data[b * 3 + 2]),
            ))
            # ⚠ `Data.xquat` IS (x, y, z, W) AND `Quat` TAKES (W, x, y, z).
            quats.append(Quat(
                Float64(d.xquat.data[b * 4 + 3]),
                Float64(d.xquat.data[b * 4 + 0]),
                Float64(d.xquat.data[b * 4 + 1]),
                Float64(d.xquat.data[b * 4 + 2]),
            ))
        renderer.render(positions, quats)

        # ⚠⚠ THE SWITCH IS AFTER `render`, NEVER BEFORE. `imgui_new_frame`
        # opened a frame that only `render` closes; work that can raise in
        # between leaves it open and the next `NewFrame` asserts.
        if want_task != cur:
            cur = want_task
            t = load_task(String(TASK_DIR) + names[cur] + ".task")
            validate_task_against_family(t, f)
            g = bind_goal(
                parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
                fmd.joint_names, jadr,
            )
            require_tier_a(g, t.name)
            print("  -> task:", t.name, "|", t.goal)
            lane += 1
            _ = do_reset(d, m, osc, scratch, t, f, addrs, rsites, jq_adr, jv_adr, nq, nv, ns, run_seed, lane)
            step = 0
            episode += 1
            held = 0
            ever_held = False
            last_goal = False
            grip_close = False
        elif want_reset or step >= f.horizon:
            lane += 1
            _ = do_reset(d, m, osc, scratch, t, f, addrs, rsites, jq_adr, jv_adr, nq, nv, ns, run_seed, lane)
            step = 0
            episode += 1
            held = 0
            ever_held = False
            last_goal = False
            grip_close = False

    renderer.close()
    print("  closed after", episode, "episode(s)")
