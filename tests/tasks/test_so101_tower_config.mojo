"""`So101TowerConfig` and its three tasks agree with the files they restate.

    pixi run mojo run -I . tests/tasks/test_so101_tower_config.mojo

The `so101_tower` family is `So101FamilyConfig[So101TowerPlacement, ...]`, a
comptime type over a GENERATED table, and neither can read the `.family` or
the composed scene. This is the CPU gate that says the restated numbers are
the files' numbers — the shape `tests/tasks/test_libero_task_hooks.mojo` has
for the 23 LIBERO families and `test_active_mask.mojo` for the tabletop:

1. the config's `MAX_STEPS` is the family's `horizon=`, and `FRAME_SKIP`
   times the scene's timestep is one `control_freq=` period to within a
   substep (16 x 2 ms against 1/30 s);
2. the table's nq/nv/nbody/nsite are the model def's, its free-slot
   addresses are `reset.free_slot_addresses` on the parsed scene (an
   INDEPENDENT derivation), and its gripper site is `robot_grasp_center` —
   the PINCH CENTRE, not the jaw tip (the bake's step 4c);
3. the model def's `OBS_DIM` is `NQ + NV + N_FREE + TASK_GOAL_WORDS` — the
   width `task_hooks.write_task_obs` writes;
4. the runtime model BUILDS under the config's hull budget
   (`NMESH_VERTS`), which `fields_build` otherwise refuses;
5. every `so101_tower_*.task` validates against the family, binds, is Tier A,
   names only device-evaluable regions, and is device-placeable — the four
   refusals a training driver would hit on its first step.

⚠ NO GPU, NO PHYSICS STEP. What a policy sees and earns on this family is
gated on the tabletop family by `test_active_mask` and
`test_tape_gpu_parity`; the hooks are the same code. What is specific here is
the table, the budget and the tasks, and those are what this checks.
"""

from std.os import listdir
from std.math import abs
from std.testing import assert_true, assert_equal

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.reset import free_slot_addresses
from mojo_rl.tasks.predicates import parse_goal, bind_goal, require_tier_a
from mojo_rl.tasks.gpu_eval import require_gpu_regions
from mojo_rl.tasks.placement.check import require_device_placement
from mojo_rl.tasks.placement.so101_tower import So101TowerPlacement
from mojo_rl.tasks.family_config import So101TowerConfig
from mojo_rl.tasks.task_hooks import TASK_GOAL_WORDS
from mojo_rl.tasks.so101_tower_xml import (
    So101TowerModel, SO101_TOWER_MAX_CONTACTS, SO101_TOWER_N_FREE_SLOTS,
)

comptime DT = DType.float64
comptime FAMILY = "mojo_rl/tasks/families/so101_tower.family"
comptime TASK_DIR = "mojo_rl/tasks/tasks"
comptime P = So101TowerPlacement
comptime CFG = So101TowerConfig


def main() raises:
    print("=== so101_tower: config + table + tasks vs the files ===")
    var f = load_family(String(FAMILY))
    var fmd = parse_model_runtime(scene_path(f))

    # ── 1. horizon and cadence ────────────────────────────────────────────
    assert_equal(CFG.MAX_STEPS, f.horizon, "MAX_STEPS is the family's horizon")
    var dt = CFG.get_timestep()
    var period = Float64(CFG.FRAME_SKIP) * dt
    var want = 1.0 / Float64(f.control_freq)
    var off = period - want
    if off < 0.0:
        off = -off
    print(
        "  cadence: FRAME_SKIP", CFG.FRAME_SKIP, "x", dt, "=", period,
        "s vs 1/control_freq", want, "(off by", off, ")",
    )
    assert_true(
        off < dt,
        "FRAME_SKIP x timestep is more than a substep from 1/control_freq",
    )

    # ── 2. the table vs the model def and the parsed scene ────────────────
    assert_equal(P.NQ, So101TowerModel.NQ, "table NQ")
    assert_equal(P.NV, So101TowerModel.NV, "table NV")
    assert_equal(P.NBODY, So101TowerModel.NBODY, "table NBODY")
    assert_equal(P.NSITE, So101TowerModel.NSITE, "table NSITE")
    assert_equal(P.N_SLOTS, len(f.slots), "table N_SLOTS")
    assert_equal(P.N_FREE, f.n_free_slots(), "table N_FREE")
    assert_equal(P.N_FREE, SO101_TOWER_N_FREE_SLOTS, "model def N_FREE")
    assert_equal(P.N_REGIONS, len(f.regions), "table N_REGIONS")
    var jt = List[Int]()
    var jqn = List[Int]()
    var jvn = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jqn.append(fmd.joints[i].nq)
        jvn.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jqn, jvn)
    var j = 0
    for si in range(len(f.slots)):
        if f.slots[si].kind != SLOT_FREE:
            continue
        assert_equal(P.free_slot(j), si, "free slot index " + String(j))
        assert_equal(P.free_qadr(j), addrs[si].qadr, "qadr " + f.slots[si].name)
        assert_equal(P.free_dadr(j), addrs[si].dadr, "dadr " + f.slots[si].name)
        assert_true(P.free_has_geom(j), f.slots[si].name + " has slot_geom")
        print(
            "  free", j, f.slots[si].name, "slot", si, "qadr",
            P.free_qadr(j), "dadr", P.free_dadr(j), "radius",
            P.free_radius[DT](j),
        )
        j += 1
    var grip = -1
    for i in range(len(fmd.site_names)):
        if String(fmd.site_names[i]) == "robot_grasp_center":
            grip = i
    assert_true(grip >= 0, "the scene has robot_grasp_center")
    assert_equal(P.GRIPPER_SITE, grip, "GRIPPER_SITE is the pinch centre")
    assert_equal(CFG.GRIPPER_SITE, grip, "the config reads it from the table")
    print("  table: nq", P.NQ, "nv", P.NV, "nbody", P.NBODY, "nsite",
          P.NSITE, "gripper site", P.GRIPPER_SITE)
    # the grasp term's two bodies, BY NAME — a renumbered scene would pay the
    # rung for touching the wrong link
    assert_equal(
        String(fmd.body_names[CFG.GRIPPER_BODY]), String("robot_gripper"),
        "GRIPPER_BODY is robot_gripper",
    )
    assert_equal(
        String(fmd.body_names[CFG.JAW_BODY]),
        String("robot_moving_jaw_so101_v1"), "JAW_BODY is the moving jaw",
    )
    print("  grasp term: weight", CFG.SHAPE_W_GRASP, "on bodies",
          CFG.GRIPPER_BODY, "+", CFG.JAW_BODY)
    # the closing bonus's joint, by name and by range
    assert_equal(
        String(fmd.joint_names[CFG.GRIPPER_QADR]), String("robot_gripper"),
        "GRIPPER_QADR is the gripper hinge (the first six joints are 1-dof)",
    )
    var gj = fmd.joints[CFG.GRIPPER_QADR]
    assert_true(
        abs(gj.range_min - CFG.GRIPPER_CLOSED) < 1e-9
        and abs(gj.range_max - CFG.GRIPPER_OPEN) < 1e-9,
        "GRIPPER_CLOSED/OPEN are the scene's gripper range",
    )
    print("  closing bonus: weight", CFG.SHAPE_W_CLOSE, "within",
          CFG.CLOSE_RADIUS, "m; gripper qpos", CFG.GRIPPER_QADR)

    # ── 3. the observation width ──────────────────────────────────────────
    assert_equal(
        So101TowerModel.OBS_DIM,
        P.NQ + P.NV + P.N_FREE + TASK_GOAL_WORDS,
        "OBS_DIM = NQ + NV + N_FREE + TASK_GOAL_WORDS",
    )
    assert_equal(CFG.OBS_MASK_BASE, P.NQ + P.NV, "mask base")
    assert_equal(CFG.OBS_GOAL_BASE, P.NQ + P.NV + P.N_FREE, "goal base")
    print("  obs: dim", So101TowerModel.OBS_DIM, "mask at", CFG.OBS_MASK_BASE,
          "goal words at", CFG.OBS_GOAL_BASE)

    # ── 4. the hull budget builds ─────────────────────────────────────────
    var dims = dims_from_flat(
        fmd, max_contacts=SO101_TOWER_MAX_CONTACTS, nmesh_verts=CFG.NMESH_VERTS
    )
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    print("  hull budget: NMESH_VERTS", CFG.NMESH_VERTS, "builds")

    # ── 5. every tower task ───────────────────────────────────────────────
    var n_tasks = 0
    var names = List[String]()
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith("so101_tower_") and n.endswith(".task"):
            names.append(n)
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            if names[k] < names[i]:
                names[i], names[k] = names[k], names[i]
    for n in names:
        var t = load_task(String(TASK_DIR) + "/" + n)
        assert_equal(t.family, f.name, n + " belongs to so101_tower")
        validate_task_against_family(t, f)
        var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
        require_tier_a(g, t.name)
        require_gpu_regions(g, t.name)
        require_device_placement[P](t, f)
        print("  task", t.name, "| goal", t.goal, "| active", len(t.active),
              "| inits", len(t.inits), "— binds, Tier A, device-placeable")
        n_tasks += 1
    assert_true(n_tasks >= 3, "the three tower tasks were found")
    print("=== PASS ===")
