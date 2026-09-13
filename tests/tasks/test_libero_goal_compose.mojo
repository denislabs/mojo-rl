"""`libero_goal.family` composes to the checked-in scene, and OUR parser
carries every slot — L2's Mojo half of the compose gate.

    pixi run mojo run -I . tests/tasks/test_libero_goal_compose.mojo

The ORACLE half is `pixi run python tools/tasks/check_family.py
mojo_rl/tasks/scenes/libero_goal.xml mojo_rl/tasks/families/libero_goal.family`
(MuJoCo loads the composed scene at `base_qpos`, counts ZERO contacts at
rest, and sums the nine assets independently: nbody 38, njnt 17, nq 41,
nv 37, ngeom 190). This file checks the same sums through our runtime
parser and that the comptime model def (`libero_goal_xml.mojo`) agrees
with the MuJoCo-generated dims. Needs the pulled pack.

⚠ NOT A COPY OF `test_family_compose_vs_mujoco.mojo`. That gate is the SO-101
family's and assumes the composer's own floor; this family has `floor=0`
(the arena brings the ground), `inherit_option=1`, a `base_pos`, and a
`base_qpos`, and each of those is asserted here rather than assumed.
"""

from std.os.path import exists

from mojo_rl.tasks.spec import load_family, SLOT_FREE, SLOT_STATIC
from mojo_rl.tasks.family import compose_family, scene_path, SCENE_DIR
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime
from mojo_rl.tasks.libero_goal_dims import LIBERO_GOAL_DIMS
from mojo_rl.tasks.libero_goal_xml import LiberoGoalModel, LIBERO_GOAL_N_FREE_SLOTS


comptime FAMILY = String("mojo_rl/tasks/families/libero_goal.family")
comptime PACK = String("mojo_rl/tasks/libero/assets")


def main() raises:
    print("=== libero_goal.family -> composed scene, vs MuJoCo — L2 ===")
    if not exists(PACK):
        print("  SKIPPED: no LIBERO pack at", PACK,
              "— run `pixi run assets-pull libero`")
        print("=== SKIPPED (no pack — this is not a pass) ===")
        return
    var f = load_family(FAMILY)
    print("  family:", f.name, "| slots:", len(f.slots), "| free:",
          f.n_free_slots())
    var bad = 0

    # the L2 keys, as generated
    if f.base != "mojo_rl/envs/robots/assets/panda_robosuite.xml":
        print("  FAIL: base is", f.base)
        bad += 1
    if f.floor or not f.inherit_option:
        print("  FAIL: floor must be 0 and inherit_option 1")
        bad += 1
    if f.base_x != -0.66 or f.base_z != 0.912:
        print("  FAIL: base_pos", f.base_x, f.base_y, f.base_z)
        bad += 1
    if len(f.base_qpos) != 9:
        print("  FAIL: base_qpos has", len(f.base_qpos), "entries")
        bad += 1
    if len(f.slots) < 2 or f.slots[0].name != "arena" or f.slots[0].kind != SLOT_STATIC:
        print("  FAIL: slot 0 must be the static arena")
        bad += 1
    if f.n_free_slots() != LIBERO_GOAL_N_FREE_SLOTS:
        print("  FAIL: free slots", f.n_free_slots(), "vs model def",
              LIBERO_GOAL_N_FREE_SLOTS)
        bad += 1

    var out = scene_path(f)
    var xml = compose_family(f, SCENE_DIR)
    var on_disk = String("")
    with open(out, "r") as fh:
        on_disk = fh.read()
    if on_disk != xml:
        raise Error(
            "libero_goal: '" + out + "' is STALE. Run `pixi run"
            " gen-family-scenes`, then `pixi run gen-dims`."
        )
    print("  ok:", out, "is up to date with the .family")
    if xml.find('cone="elliptic"') < 0 or xml.find('inertiagrouprange="0 0"') < 0:
        print("  FAIL: the composed scene did not inherit robosuite's option/compiler")
        bad += 1
    if xml.find('<geom name="floor"') >= 0:
        print("  FAIL: the composer added its own floor on top of the arena's")
        bad += 1

    var fmd = parse_model_runtime(out)
    var base = parse_model_runtime(f.base)
    var exp_bodies = len(base.bodies)
    var exp_joints = len(base.joints)
    var exp_geoms = len(base.geoms)
    for i in range(len(f.slots)):
        var a = parse_model_runtime(f.slots[i].asset)
        exp_bodies += len(a.bodies)
        exp_joints += len(a.joints)
        exp_geoms += len(a.geoms)
    print("  ours   : nbody", len(fmd.bodies), " njoint", len(fmd.joints),
          " ngeom", len(fmd.geoms))
    print("  expect : nbody", exp_bodies, " njoint", exp_joints, " ngeom",
          exp_geoms, " (base + every slot's asset, no floor of our own)")
    if len(fmd.bodies) != exp_bodies or len(fmd.joints) != exp_joints or len(fmd.geoms) != exp_geoms:
        print("  FAIL: the composed model does not carry every slot")
        bad += 1
    # ⚠ AGAINST MuJoCo's NUMBERS, not only our own sums: `nbody` in
    # `FlatModelDef` excludes the world, MuJoCo's includes it.
    if len(fmd.bodies) + 1 != LIBERO_GOAL_DIMS.NBODY:
        print("  FAIL: nbody+1", len(fmd.bodies) + 1, "vs MuJoCo", LIBERO_GOAL_DIMS.NBODY)
        bad += 1
    if len(fmd.joints) != LIBERO_GOAL_DIMS.NJOINT or len(fmd.geoms) != LIBERO_GOAL_DIMS.NGEOM:
        print("  FAIL: njoint/ngeom vs MuJoCo", LIBERO_GOAL_DIMS.NJOINT, LIBERO_GOAL_DIMS.NGEOM)
        bad += 1
    print("  ok: our parser and MuJoCo agree: nbody", LIBERO_GOAL_DIMS.NBODY,
          "njnt", LIBERO_GOAL_DIMS.NJOINT, "nq", LIBERO_GOAL_DIMS.NQ,
          "nv", LIBERO_GOAL_DIMS.NV, "ngeom", LIBERO_GOAL_DIMS.NGEOM)
    if LIBERO_GOAL_DIMS.NQ != 41 or LIBERO_GOAL_DIMS.NV != 37:
        print("  FAIL: expected nq 41 (9 + 4 fixture joints + 4x7), nv 37")
        bad += 1
    if LIBERO_GOAL_DIMS.TIMESTEP != 0.002:
        print("  FAIL: timestep", LIBERO_GOAL_DIMS.TIMESTEP, "— the <option> was not inherited")
        bad += 1

    # the comptime model def instantiates and its dims are MuJoCo's
    comptime M = LiberoGoalModel
    if M.NQ != LIBERO_GOAL_DIMS.NQ or M.NV != LIBERO_GOAL_DIMS.NV:
        print("  FAIL: comptime model def NQ/NV", M.NQ, M.NV)
        bad += 1
    print("  ok: LiberoGoalModel NQ", M.NQ, "NV", M.NV, "NGEOM", M.NGEOM)

    if bad != 0:
        raise Error(String(bad) + " checks failed")
    print("=== PASS (Mojo half) ===")
    print("⚠ The MuJoCo half of this gate is `tools/tasks/check_family.py`,")
    print("  run it on:", out, FAMILY)
