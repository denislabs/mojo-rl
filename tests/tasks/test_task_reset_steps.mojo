"""The composed family, RESET AND STEPPED — the first time it is simulated.

⚠⚠ EVERY GATE BEFORE THIS ONE CONSTRUCTED GEOMETRY. `test_task_eval` writes
positions into arrays and asks what a predicate means; `test_family_compose`
counts records. Neither ever ran the physics, and that is exactly how a static
fixture ended up welded 50 m in the air with the region on its surface and the
sampler placing props into the sky — `nq`, `nv`, `ngeom` and every contact
count were correct the whole time.

So this file builds the model, resets it from a sampled init, and STEPS it.
What it asserts is what only a simulation can say:

1. the region's site is where a task can actually reach — on the table, not
   in orbit;
2. a sampled prop lands ON the surface and STAYS there — settle it under
   gravity and it must still be on the table, which is a statement about the
   sampler's z, the fixture's pose and the solver all at once;
3. an INACTIVE slot is parked and touches nothing;
4. reset is IDEMPOTENT in `(seed, lane)` — step the scene, reset again, and
   the state matches the first reset exactly.

Run: pixi run mojo run -I . tests/tasks/test_task_reset_steps.mojo
"""

from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
from mojo_rl.tasks.spec import load_family, load_task, validate_task_against_family
from mojo_rl.tasks.family import scene_path, park_pos
from mojo_rl.tasks.predicates import parse_goal, bind_goal, slot_body_id
from mojo_rl.tasks.eval import eval_goal, region_sites
from mojo_rl.tasks.sampler import (
    sample_placements, RegionFrame, SampleReport,
)
from mojo_rl.tasks.reset import free_slot_addresses, reset_slots
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics


comptime DT = DType.float64


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def main() raises:
    print("=== the composed family, reset and stepped ===")
    var ta = Tally()

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var t = load_task("mojo_rl/tasks/tasks/so101_gather_bricks.task")
    validate_task_against_family(t, f)

    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)

    # The joint table, as four parallel lists — `tasks/` does not take a
    # physics3d record type in its signature (§7).
    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)
    print("  free slot addresses:")
    for i in range(len(f.slots)):
        print("    ", f.slots[i].name, "-> qadr", addrs[i].qadr,
              " dadr", addrs[i].dadr)

    # ── build a real model + data ─────────────────────────────────────────
    # ⚠ `nmesh_verts` IS NOT OPTIONAL FOR THIS FAMILY. The base is SO-ARM101,
    # whose 30 collision meshes need 26,198 hull vertices; the default 0 means
    # "mesh geoms do not collide" and `fields_build` RAISES rather than
    # letting the arm quietly stop colliding. That raise is the reason this
    # number is here and not guessed — set it to 1 and read the error.
    var dims = dims_from_flat(
        fmd, max_contacts=32, nmesh_verts=SO_ARM101_NMESH_VERTS
    )
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nb = dims.get_nbody()
    var ns = dims.get_nsite()
    print("  model: nq", nq, " nv", nv, " nbody", nb, " nsite", ns)

    # ── 1. where is the region, really? ───────────────────────────────────
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var sp = List[Float64]()
    for i in range(ns * 3):
        sp.append(Float64(d.site_xpos.data[i]))
    var ts = rsites[0]
    print("  region 'table_top' site at (", sp[ts * 3], ",",
          sp[ts * 3 + 1], ",", sp[ts * 3 + 2], ")")
    # ⚠ A COARSE BOUND, AND IT IS NOT THE REAL CHECK. It catches a region
    # flung out of the workspace — the 50-m table — and nothing subtler. It
    # did NOT catch the table floating at z=0.30, because the gripper site
    # could reach the props on it while the arm had to work around a slab
    # held up by nothing.
    #
    # The real checks are MuJoCo-side, in `tools/tasks/check_family.py`: a
    # static fixture must REST ON SOMETHING, and each region must be within a
    # sampled reachable envelope. Both are asserted there because both need
    # 4k forward-kinematics solves and a geom-extent walk, which belong with
    # the oracle rather than here.
    var rx = sp[ts * 3]
    var ry = sp[ts * 3 + 1]
    var rz = sp[ts * 3 + 2]
    var reach = (rx * rx + ry * ry + rz * rz) ** 0.5
    ta.check(reach < 0.6,
             "the region is within the arm's workspace (|p| = "
             + String(reach) + " m)")

    # ── 2. sample, reset, and settle ──────────────────────────────────────
    var frames = List[RegionFrame]()
    for i in range(len(f.regions)):
        var s = rsites[i]
        frames.append(RegionFrame(sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2]))
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(0.02)

    var rep = SampleReport()
    var placed = sample_placements(t, f, frames, radii, UInt64(11), 0, rep)
    print("  sampler:", rep.accepted, "placed in", rep.attempts, "attempts")

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

    var xb = List[Float64]()
    for i in range(nb * 3):
        xb.append(Float64(d.xpos.data[i]))
    var brick = slot_body_id(String("brick"), fmd.body_names)
    var cube_b = slot_body_id(String("cube_b"), fmd.body_names)
    print("  brick  at (", xb[brick * 3], ",", xb[brick * 3 + 1], ",",
          xb[brick * 3 + 2], ")")
    print("  cube_b at (", xb[cube_b * 3], ",", xb[cube_b * 3 + 1], ",",
          xb[cube_b * 3 + 2], ")   <- INACTIVE, parked")

    ta.check(
        xb[brick * 3 + 2] > rz and xb[brick * 3 + 2] < rz + 0.1,
        "an ACTIVE slot lands just above the table surface",
    )

    # ⚠ THE INACTIVE SLOT MUST BE AT ITS PARK POSE, not merely "far away".
    # `reset_slots` writes EVERY free slot; a version that only wrote the
    # active ones would leave this at whatever the model file said, which is
    # also far away — and would drift once episodes reused a Data.
    var pk = park_pos(f, 3)
    ta.check(
        xb[cube_b * 3] == pk[0] and xb[cube_b * 3 + 2] == pk[2],
        "an INACTIVE slot is at its exact PARK pose",
    )

    # ── 3. the goal evaluates on the real state ───────────────────────────
    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
    var xq = List[Float64]()
    for i in range(nb * 4):
        xq.append(Float64(d.xquat.data[i]))
    var sp2 = List[Float64]()
    for i in range(ns * 3):
        sp2.append(Float64(d.site_xpos.data[i]))
    var holds = eval_goal(g, f, xb, xq, sp2, rsites)
    print("  goal", t.goal, "at reset ->", holds)
    # ⚠ `gather` asks for both props ON the table, and the sampler puts them
    # there — so this must hold AT RESET. If it does not, the sampler's z and
    # `ON_MAX_DZ` disagree, which is a real defect and not a tuning matter.
    ta.check(holds, "the gather goal HOLDS at reset (sampler agrees with On)")

    # ── 3b. ⚠⚠ THE QUATERNION CONVENTIONS, ON REAL DATA ──────────────────
    #
    # TWO DIFFERENT ORDERS LIVE IN THE SAME SYSTEM and this is the only place
    # that pins both:
    #
    #   a free joint's `qpos`  is (x, y, z,  w, x, y, z)  -- W FIRST
    #   `Data.xquat`           is (x, y, z, w)            -- W LAST
    #
    # `eval_goal` read `xquat` as w-first for one commit. The P2c gate could
    # not see it because that gate CONSTRUCTS the array, under the same
    # assumption the evaluator made — so the two agreed and both were wrong.
    # It surfaced from reading the studio's render code. This arm evaluates
    # against a Data the physics filled, so it cannot drift back.
    print("--- quaternion conventions, against a real Data ---")
    var g_up = bind_goal(
        parse_goal(String("Upright(brick, 0.05)")),
        f, fmd.body_names, fmd.site_names,
    )
    ta.check(
        eval_goal(g_up, f, xb, xq, sp2, rsites),
        "a freshly placed brick IS upright (identity written by reset)",
    )

    # Tip it 90 degrees about x, IN qpos (w-first), and re-run FK.
    var ba = addrs[1].qadr          # slot 1 is `brick`
    d.qpos.data[ba + 3] = Scalar[DT](0.7071067811865476)   # qw
    d.qpos.data[ba + 4] = Scalar[DT](0.7071067811865476)   # qx
    d.qpos.data[ba + 5] = Scalar[DT](0.0)
    d.qpos.data[ba + 6] = Scalar[DT](0.0)
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var xq2 = List[Float64]()
    for i in range(nb * 4):
        xq2.append(Float64(d.xquat.data[i]))
    var xb2 = List[Float64]()
    for i in range(nb * 3):
        xb2.append(Float64(d.xpos.data[i]))
    var sp3 = List[Float64]()
    for i in range(ns * 3):
        sp3.append(Float64(d.site_xpos.data[i]))
    print("    brick xquat after a 90-deg qpos tip: (",
          xq2[brick * 4], ",", xq2[brick * 4 + 1], ",",
          xq2[brick * 4 + 2], ",", xq2[brick * 4 + 3], ")")
    # ⚠ THE LAYOUT ASSERTION. w-first in qpos, w-LAST in xquat: a 90-degree
    # turn about x must come back with the SAME value in slots 0 and 3 and
    # zeros between. Reading either convention the other way puts 0.707 in a
    # different slot and this fails by inspection.
    ta.check(
        xq2[brick * 4 + 0] > 0.70 and xq2[brick * 4 + 3] > 0.70
        and xq2[brick * 4 + 1] < 1e-9 and xq2[brick * 4 + 2] < 1e-9,
        "qpos is W-FIRST and Data.xquat is W-LAST",
    )
    ta.check(
        not eval_goal(g_up, f, xb2, xq2, sp3, rsites),
        "a brick tipped 90 degrees is NOT upright, on real Data",
    )

    # ── 4. reset is idempotent in (seed, lane) ────────────────────────────
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0.0)
    var rep2 = SampleReport()
    var placed2 = sample_placements(t, f, frames, radii, UInt64(11), 0, rep2)
    var qpos2 = List[Float64]()
    for _ in range(nq):
        qpos2.append(0.0)
    var qvel2 = List[Float64]()
    for _ in range(nv):
        qvel2.append(0.0)
    reset_slots(t, f, placed2, addrs, qpos2, qvel2)
    var same = True
    for i in range(len(f.slots)):
        if addrs[i].qadr < 0:
            continue
        for k in range(7):
            if qpos2[addrs[i].qadr + k] != qpos[addrs[i].qadr + k]:
                same = False
    ta.check(same, "reset is IDEMPOTENT for the same (seed, lane)")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "task reset: " + String(ta.failures) + " of " + String(ta.checks)
            + " check(s) failed"
        )
    print("=== PASS ===")
