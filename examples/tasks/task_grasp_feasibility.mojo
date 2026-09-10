"""CAN THE ARM ACTUALLY PICK THIS UP? — the question that has to be answered
before "the task is hard" means anything.

    pixi run mojo run -I . examples/tasks/task_grasp_feasibility.mojo so101_lift_brick

## ⚠⚠ WHY THIS EXISTS

Two 1M-step SAC runs on `so101_lift_brick`, 38 GPU-minutes each, returned a
success rate of 0.0 with textbook critics (peak `mean_q` 0.99x and 0.94x their
fixed points). The second ran at the gradient-optimal margins, so the shaping
was not the limit: the reach term saturated in both and the goal term never
moved. That is the signature of a policy that reaches the brick and cannot
lift it — and "cannot" has two very different causes:

  a. the grasp is a discrete contact event with no partial credit, so no
     `tolerance` term leads a policy to it; or
  b. nothing in the sim can lift this brick at all.

They are not distinguishable from a training curve, and (b) makes every run
on this task a measurement of nothing. Every collision geom on SO-101 is a
MESH and the brick is a BOX, so (b) is a live possibility and not a paranoid
one.

⚠ THIS PROBE DOES NOT USE A POLICY OR IK. It places the brick where the
answer must be yes and drives the actuators directly, so a failure is the
sim's and not the controller's.
"""

from std.random import random_float64, seed as seed_rng
from std.sys import argv

from mojo_rl.core.cont_action import ContAction
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.family_config import So101TabletopConfig
from mojo_rl.tasks.so101_tabletop_xml import So101TabletopModel
from mojo_rl.tasks.predicates import parse_goal, bind_goal
from mojo_rl.tasks.eval import region_sites
from mojo_rl.tasks.active import active_mask
from mojo_rl.tasks.tape import encode_goal, TAPE_WORDS
from mojo_rl.tasks.sampler import sample_placements, RegionFrame, SampleReport
from mojo_rl.tasks.reset import free_slot_addresses, reset_slots
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE, META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B, CONTACT_IDX_DIST, CONTACT_SIZE,
    CONTACT_IDX_CONDIM, CONTACT_IDX_FRICTION,
    MODEL_GEOM_SIZE, GEOM_IDX_BODY, GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y, GEOM_IDX_HALF_Z, GEOM_IDX_RBOUND,
)
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime
from mojo_rl.envs.phyics3d_env import Phyics3dEnv

comptime CFG = So101TabletopConfig
comptime DT = DType.float64
comptime EnvT = Phyics3dEnv[So101TabletopModel, CFG, DT]
comptime NQ = So101TabletopModel.NQ
comptime NV = So101TabletopModel.NV
comptime SEED: UInt64 = 12345


def brick_contacts(mut env: EnvT, body: Int) -> Tuple[Int, Float64]:
    """`(contacts touching `body`, the deepest penetration among them)`.

    ⚠ CONTACTS CARRY BODY IDS, NOT GEOM IDS (`CONTACT_IDX_BODY_A/B`), so this
    counts the brick's contacts with ANYTHING — table included. The caller
    separates arm from table by what it has moved the brick next to.
    """
    var n = Int(Float64(env.d.meta.data[META_IDX_NUM_CONTACTS]))
    var hits = 0
    var deepest = 0.0
    for c in range(n):
        var o = c * CONTACT_SIZE
        var a = Int(Float64(env.d.contacts.data[o + CONTACT_IDX_BODY_A]))
        var b = Int(Float64(env.d.contacts.data[o + CONTACT_IDX_BODY_B]))
        if a == body or b == body:
            hits += 1
            var d = Float64(env.d.contacts.data[o + CONTACT_IDX_DIST])
            if d < deepest:
                deepest = d
    return (hits, deepest)


def place_brick(mut env: EnvT, qadr: Int, dadr: Int,
                x: Float64, y: Float64, z: Float64):
    """Teleport the brick, upright and at rest."""
    env.d.qpos.data[qadr] = Scalar[DT](x)
    env.d.qpos.data[qadr + 1] = Scalar[DT](y)
    env.d.qpos.data[qadr + 2] = Scalar[DT](z)
    env.d.qpos.data[qadr + 3] = Scalar[DT](1.0)
    env.d.qpos.data[qadr + 4] = Scalar[DT](0.0)
    env.d.qpos.data[qadr + 5] = Scalar[DT](0.0)
    env.d.qpos.data[qadr + 6] = Scalar[DT](0.0)
    for i in range(6):
        env.d.qvel.data[dadr + i] = Scalar[DT](0.0)


def act(g: Float64, a0: Float64, a1: Float64, a2: Float64,
        a3: Float64, a4: Float64) raises -> ContAction[6]:
    var v = List[Float64]()
    v.append(a0)
    v.append(a1)
    v.append(a2)
    v.append(a3)
    v.append(a4)
    v.append(g)
    return ContAction[6].from_list(v)


def hold_actions(
    mut e: EnvT, ctrl_min: List[Float64], ctrl_max: List[Float64],
    qadr_of_act: List[Int],
) raises -> List[Float64]:
    """The action that commands each joint to STAY WHERE IT IS.

    ⚠⚠ ACTION 0 IS NOT "DO NOTHING" FOR A POSITION ACTUATOR. Under
    `NORMALIZED_ACTIONS` a zero maps to the MIDDLE of the ctrl range, so a
    zero vector is a command to swing to the mid-range pose. A hold is
    `a = (q - mid) / halfrange`.

    ⚠ NESTED `def`s CANNOT CAPTURE THESE, so the tables come in as arguments
    rather than being read off `fmd` in place — the same Mojo limitation that
    put `task_reset` and `measure` at module scope in the policy viewer.
    """
    var out = List[Float64]()
    for i in range(len(ctrl_min)):
        var mid = 0.5 * (ctrl_min[i] + ctrl_max[i])
        var half = 0.5 * (ctrl_max[i] - ctrl_min[i])
        var q = Float64(e.d.qpos.data[qadr_of_act[i]])
        var a = 0.0 if half <= 0.0 else (q - mid) / half
        if a > 1.0:
            a = 1.0
        if a < -1.0:
            a = -1.0
        out.append(a)
    return out^


def gripper_contacts(mut e: EnvT, brick: Int, g0: Int, g1: Int) -> Int:
    """Contacts between the brick and the GRIPPER bodies specifically.

    ⚠⚠ "THE BRICK IS HIGH" IS NOT "THE BRICK IS HELD". A cube the closing jaw
    has FLICKED is airborne, and sampled at the wrong instant it reads as a
    successful grasp with a perfectly good height — the first version of this
    probe passed on exactly that, reporting z 0.18 with ZERO contacts, which
    is a state no settled object can be in. A grasp is: touching the gripper,
    above the table, and still there later.
    """
    var n = Int(Float64(e.d.meta.data[META_IDX_NUM_CONTACTS]))
    var hits = 0
    for c in range(n):
        var o = c * CONTACT_SIZE
        var a = Int(Float64(e.d.contacts.data[o + CONTACT_IDX_BODY_A]))
        var b = Int(Float64(e.d.contacts.data[o + CONTACT_IDX_BODY_B]))
        var other = -1
        if a == brick:
            other = b
        elif b == brick:
            other = a
        if other == g0 or other == g1:
            hits += 1
    return hits


def brick_speed(mut e: EnvT, dadr: Int) -> Float64:
    """|linear velocity| of the brick — an airborne cube is not at rest."""
    var vx = Float64(e.d.qvel.data[dadr])
    var vy = Float64(e.d.qvel.data[dadr + 1])
    var vz = Float64(e.d.qvel.data[dadr + 2])
    return (vx * vx + vy * vy + vz * vz) ** 0.5


def step_hold(mut e: EnvT, h: List[Float64], grip: Float64) raises:
    """One step holding the arm and commanding the gripper."""
    var v = List[Float64]()
    for i in range(5):
        v.append(h[i])
    v.append(grip)
    _ = e.step(ContAction[6].from_list(v))


def main() raises:
    seed_rng(7)
    var args = argv()
    var task_name = String("so101_lift_brick")
    if len(args) > 1:
        task_name = String(args[1])

    print("=" * 72)
    print("grasp feasibility —", task_name)
    print("=" * 72)

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var t = load_task("mojo_rl/tasks/tasks/" + task_name + ".task")
    validate_task_against_family(t, f)
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
    var tape = encode_goal(g)
    var mask = active_mask(t, f)
    var brick = g.terms[0].a

    # The jaw is where a contact MUST appear if mesh-vs-box works at all.
    var jaw = -1
    for i in range(len(fmd.body_names)):
        if "moving_jaw" in fmd.body_names[i]:
            jaw = i
    print("  brick body", brick, "=", fmd.body_names[brick])
    print("  jaw   body", jaw, "=", fmd.body_names[jaw] if jaw >= 0 else "NOT FOUND")
    if jaw < 0:
        raise Error("grasp feasibility: no moving_jaw body in the composed"
                    " scene — the probe cannot locate the gripper.")
    var grip_body = -1
    for i in range(len(fmd.body_names)):
        if fmd.body_names[i] == "robot_gripper":
            grip_body = i
    print("  grip  body", grip_body,
          "=", fmd.body_names[grip_body] if grip_body >= 0 else "NOT FOUND")

    var env = EnvT()
    _ = env.reset()
    for w in range(TAPE_WORDS):
        env.d.meta.data[META_IDX_TASK_PARAM_0 + w] = Scalar[DT](tape[w])
    env.d.meta.data[META_IDX_TASK_ACTIVE] = Scalar[DT](mask)

    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)
    var sp0 = List[Float64]()
    for i in range(len(fmd.site_names) * 3):
        sp0.append(Float64(env.d.site_xpos.data[i]))
    var frames = List[RegionFrame]()
    for i in range(len(f.regions)):
        var rs = rsites[i]
        frames.append(RegionFrame(sp0[rs * 3], sp0[rs * 3 + 1], sp0[rs * 3 + 2]))
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(CFG.SLOT_RADIUS)
    var rep = SampleReport()
    var placed = sample_placements(t, f, frames, radii, SEED, 0, rep)
    var q0 = List[Float64]()
    for i in range(NQ):
        q0.append(Float64(env.d.qpos.data[i]))
    var v0 = List[Float64]()
    for _ in range(NV):
        v0.append(0.0)
    reset_slots(t, f, placed, addrs, q0, v0)
    for i in range(NQ):
        env.d.qpos.data[i] = Scalar[DT](q0[i])
    for i in range(NV):
        env.d.qvel.data[i] = Scalar[DT](v0[i])

    # The brick's slot, by family slot order.
    var bslot = -1
    for i in range(len(f.slots)):
        if f.slots[i].name == "brick":
            bslot = i
    var qadr = addrs[bslot].qadr
    var dadr = addrs[bslot].dadr
    print("  brick free joint at qpos", qadr, " qvel", dadr)
    print()

    var fails = 0

    # ── LEG 1: does a mesh jaw collide with a box brick AT ALL? ───────────
    # ⚠ THE BRICK IS PUT INSIDE THE JAW, which is not a pose any policy would
    # produce — that is the point. If the solver reports nothing here, no
    # contact between the arm and a prop is possible anywhere, and every
    # `lift` run so far measured a brick the arm cannot touch.
    print("LEG 1 — a mesh jaw against a box brick")
    _ = env.step(act(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    var jx = Float64(env.d.xpos.data[jaw * 3])
    var jy = Float64(env.d.xpos.data[jaw * 3 + 1])
    var jz = Float64(env.d.xpos.data[jaw * 3 + 2])
    print("  jaw at", jx, jy, jz)
    place_brick(env, qadr, dadr, jx, jy, jz)
    _ = env.step(act(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    var c1 = brick_contacts(env, brick)
    print("  brick placed AT the jaw origin -> contacts", c1[0],
          " deepest penetration", c1[1], "m")
    # ⚠⚠ CONDIM AND FRICTION, NOT JUST THE COUNT. A contact with `condim 1`
    # is FRICTIONLESS: the jaw can press on the brick as hard as its actuator
    # allows and the brick still slides straight out, which reads in LEG 3 as
    # "the gripper cannot hold it" and in a training curve as "the task is
    # hard". A zero friction coefficient does the same thing.
    var ncon = Int(Float64(env.d.meta.data[META_IDX_NUM_CONTACTS]))
    var shown = 0
    for c in range(ncon):
        var o = c * CONTACT_SIZE
        var a = Int(Float64(env.d.contacts.data[o + CONTACT_IDX_BODY_A]))
        var b = Int(Float64(env.d.contacts.data[o + CONTACT_IDX_BODY_B]))
        if (a == brick or b == brick) and shown < 4:
            print("     contact", a, "<->", b, " condim",
                  Float64(env.d.contacts.data[o + CONTACT_IDX_CONDIM]),
                  " friction",
                  Float64(env.d.contacts.data[o + CONTACT_IDX_FRICTION]))
            shown += 1
    if c1[0] == 0:
        print("  ⚠⚠ FAIL: NO CONTACT. A box centred on the jaw body reports"
              " nothing, so mesh-vs-box does not collide in this scene and"
              " the arm passes through every prop. `lift` is not a hard"
              " task, it is an impossible one.")
        fails += 1
    else:
        print("  OK — the arm and the brick collide.")
    print()

    # ── LEG 2: anti-vacuity — the count must be able to be ZERO. ──────────
    # ⚠ A COUNTER THAT ALWAYS RETURNS NONZERO WOULD PASS LEG 1 ON A BROKEN
    # SIM. Park the brick far from everything and require silence.
    print("LEG 2 — the contact counter can read zero (anti-vacuity)")
    place_brick(env, qadr, dadr, 3.0, 3.0, 3.0)
    _ = env.step(act(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    var c2 = brick_contacts(env, brick)
    print("  brick 3 m away -> contacts", c2[0])
    if c2[0] != 0:
        print("  FAIL: a brick 3 m from anything still reports contacts —"
              " the counter does not discriminate and LEG 1 proves nothing.")
        fails += 1
    else:
        print("  OK — silent when it should be.")
    print()

    # ── LEG 3: can the gripper HOLD it against gravity? ───────────────────
    # ⚠⚠ THE ARM HAS TO BE COMMANDED TO STAND STILL, NOT SENT ZERO. These are
    # POSITION actuators under `NORMALIZED_ACTIONS`, so action 0 means "go to
    # the middle of your ctrl range" — the arm swings across the table and
    # whatever it was holding is flung off it. The first version of this leg
    # did exactly that and reported the brick at z 0.0199: not a dropped
    # grasp, the FLOOR beside the table (the gripper sits at x 0.386 and the
    # table spans 0.15..0.35), which reads identically in the z alone.
    #
    # The hold action per joint is the one that maps back to the pose the arm
    # is already in: `a = (q - mid) / halfrange`.
    print("LEG 3 — the gripper closed on the brick, against gravity")
    comptime GS = CFG.GRIPPER_SITE

    var jadr = List[Int]()
    var acc = 0
    for i in range(len(fmd.joints)):
        jadr.append(acc)
        acc += fmd.joints[i].nq
    var a_lo = List[Float64]()
    var a_hi = List[Float64]()
    var a_qa = List[Int]()
    for i in range(6):
        a_lo.append(fmd.actuators[i].ctrl_min)
        a_hi.append(fmd.actuators[i].ctrl_max)
        a_qa.append(jadr[fmd.actuators[i].joint_id])

    # ⚠⚠ THE PLACEMENT IS SWEPT, NOT GUESSED. "I put the brick where I think
    # the jaws are and it fell" is not evidence about the gripper — it is
    # evidence about my guess. The brick is offered at every point of a grid
    # spanning the gripper opening, in both closing directions, and the leg
    # fails only if NONE of them holds.
    var best_hold = -1.0
    var best_dir = 0.0
    var best_contacts = 0
    var best_off = List[Float64]()
    best_off.append(0.0)
    best_off.append(0.0)
    best_off.append(0.0)
    var trials = 0
    var best_any = -1.0
    var best_any_gc = 0
    var best_any_spd = 0.0
    comptime STEP_M = 0.015          # 1.5 cm — the brick is 4 cm across
    for k in range(2):
        var close = -1.0 if k == 0 else 1.0
        for ix in range(-2, 3):
            for iy in range(-2, 3):
                for iz in range(-2, 3):
                    for i in range(NQ):
                        env.d.qpos.data[i] = Scalar[DT](q0[i])
                    for i in range(NV):
                        env.d.qvel.data[i] = Scalar[DT](v0[i])
                    _ = env.step(act(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    for i in range(NQ):
                        env.d.qpos.data[i] = Scalar[DT](q0[i])
                    for i in range(NV):
                        env.d.qvel.data[i] = Scalar[DT](v0[i])
                    var hold = hold_actions(env, a_lo, a_hi, a_qa)

                    # open the jaw, holding the arm where it is
                    for _ in range(30):
                        step_hold(env, hold, -close)
                    var sx = Float64(env.d.site_xpos.data[GS * 3])
                    var sy = Float64(env.d.site_xpos.data[GS * 3 + 1])
                    var sz = Float64(env.d.site_xpos.data[GS * 3 + 2])
                    var ox = Float64(ix) * STEP_M
                    var oy = Float64(iy) * STEP_M
                    var oz = Float64(iz) * STEP_M
                    place_brick(env, qadr, dadr, sx + ox, sy + oy, sz + oz)
                    # close, and keep holding the arm still
                    for _ in range(120):
                        step_hold(env, hold, close)
                    var z_mid = Float64(env.d.qpos.data[qadr + 2])
                    # ⚠ AND THEN KEEP HOLDING. A flicked cube passes through
                    # a good height on its way back down; a held one is still
                    # there 0.5 s later.
                    for _ in range(250):
                        step_hold(env, hold, close)
                    var z_end = Float64(env.d.qpos.data[qadr + 2])
                    var gc = gripper_contacts(env, brick, grip_body, jaw)
                    var spd = brick_speed(env, dadr)
                    trials += 1
                    # ⚠⚠ ALL FOUR, NOT THE HEIGHT ALONE: above the table, in
                    # contact with the gripper, at rest, and still up at both
                    # samples.
                    var held = (
                        z_end > 0.06 and z_mid > 0.06 and gc > 0
                        and spd < 0.05
                    )
                    if held and z_end > best_hold:
                        best_hold = z_end
                        best_dir = close
                        best_contacts = gc
                        best_off[0] = ox
                        best_off[1] = oy
                        best_off[2] = oz
                    if z_end > best_any:
                        best_any = z_end
                        best_any_gc = gc
                        best_any_spd = spd
    print("  swept", trials, "placements over a +-3 cm grid, both closing"
          " directions")
    print("  best HELD brick z", best_hold, "at offset", best_off[0],
          best_off[1], best_off[2], " gripper action", best_dir,
          " gripper contacts", best_contacts)
    print("  highest brick z of ANY trial", best_any, " gripper contacts",
          best_any_gc, " speed", best_any_spd,
          "  <- height alone, which is NOT a grasp")

    # ⚠ THE CRITERION IS HEIGHT ABOVE THE TABLE TOP (0.02), not above the
    # floor, and not "did it move" — a brick resting on the table top has its
    # centre at 0.04, so anything at or below that was not held.
    print("  (a brick resting on the table top has its centre at 0.04)")
    if best_hold < 0.06:
        print("  ⚠⚠ THE GRIPPER DID NOT HOLD THE BRICK at ANY of the swept"
              " placements, handed a closed grasp for free with the arm"
              " commanded to stand still. Every trial ends with the brick at"
              " the table resting height.")
        print("     LEG 1 rules out collision (13 contacts, condim 3,"
              " friction 1.0) and LEG 4 rules out a dead actuator (1.92 rad"
              " of jaw travel), so what is left is the GEOMETRY of the"
              " pinch: the jaw hull against a 4 cm cube. This probe does NOT"
              " establish which, only that a grasp is not available to be"
              " found — so reward work on this task buys nothing until it"
              " is.")
        fails += 1
    else:
        print("  OK — the closed gripper holds the brick at z", best_hold,
              ". A grasp is physically available, so the 0.0 rate is an"
              " EXPLORATION result: the reward does not lead a policy to a"
              " discrete contact event.")
    print()

    # ── LEG 4: does the gripper joint move at all? ────────────────────────
    # ⚠⚠ MEASURE THE JOINT ANGLE, NOT THE JAW BODY'S ORIGIN. The origin sits
    # ON the hinge axis, so it is INVARIANT under the rotation being measured
    # — the first version of this leg reported a travel of 2.8e-17 m and read
    # as "the gripper does not open", which is a statement about where I put
    # the probe point and not about the gripper.
    print("LEG 4 — the gripper joint's travel under its own actuator")
    var gj = -1
    for i in range(len(fmd.joint_names)):
        if "gripper" in fmd.joint_names[i]:
            gj = i
    if gj < 0:
        raise Error("grasp feasibility: no gripper joint in the scene.")
    var ang = List[Float64]()
    for k in range(2):
        var e = -1.0 if k == 0 else 1.0
        for i in range(NQ):
            env.d.qpos.data[i] = Scalar[DT](q0[i])
        for i in range(NV):
            env.d.qvel.data[i] = Scalar[DT](v0[i])
        _ = env.step(act(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        for i in range(NQ):
            env.d.qpos.data[i] = Scalar[DT](q0[i])
        for i in range(NV):
            env.d.qvel.data[i] = Scalar[DT](v0[i])
        var hold = hold_actions(env, a_lo, a_hi, a_qa)
        for _ in range(120):
            step_hold(env, hold, e)
        var a = Float64(env.d.qpos.data[jadr[gj]])
        ang.append(a)
        print("  gripper action", e, "-> joint", fmd.joint_names[gj],
              "at", a, "rad")
    var travel = ang[1] - ang[0]
    if travel < 0.0:
        travel = -travel
    print("  joint travel between the action extremes:", travel, "rad")
    if travel < 0.05:
        print("  ⚠⚠ THE GRIPPER JOINT DOES NOT MOVE. Its actuator is"
              " commanded to both ends of its range and the angle does not"
              " follow, so nothing the policy does with action 5 can open or"
              " close the jaw — and LEG 3 was never testing a grasp. This is"
              " an ACTUATOR wiring fault, not a reward or an exploration"
              " problem.")
        fails += 1
    else:
        print("  OK — the jaw travels", travel, "rad under its actuator, so"
              " LEG 3's failure is about the GRASP and not about a dead"
              " gripper.")
    print()

    # ── LEG 5: is the PROP too wide for this jaw? ─────────────────────────
    # ⚠ THE DISCRIMINATOR, AND THE CHEAPEST ONE AVAILABLE. If a smaller cube
    # holds where the 4 cm one does not, `so101_lift_brick` is a one-line
    # asset change (`cube.xml`'s `size`) and the jaw is fine. If nothing
    # holds at any width, the pinch itself is the problem.
    #
    # ⚠ THE HALF-EXTENTS ARE RUNTIME FIELDS, so the sweep needs no new asset
    # and no recompile — `mf.geoms` carries `HALF_X/Y/Z` per geom. `RBOUND`
    # is left ALONE deliberately: it is the broadphase radius, and leaving it
    # at the larger value over-reports candidate pairs, which narrowphase
    # then rejects. Shrinking it could drop real pairs and would make a
    # smaller cube look unliftable for a reason that is not the jaw.
    print("LEG 5 — does a SMALLER prop hold?")
    var bgeom = -1
    for gi in range(So101TabletopModel.NGEOM):
        var o = gi * MODEL_GEOM_SIZE
        if Int(Float64(env.mf.geoms.data[o + GEOM_IDX_BODY])) == brick:
            bgeom = gi
    if bgeom < 0:
        raise Error("grasp feasibility: no geom on the brick body.")
    var o_b = bgeom * MODEL_GEOM_SIZE
    print("  brick geom", bgeom, " half-extents",
          Float64(env.mf.geoms.data[o_b + GEOM_IDX_HALF_X]),
          Float64(env.mf.geoms.data[o_b + GEOM_IDX_HALF_Y]),
          Float64(env.mf.geoms.data[o_b + GEOM_IDX_HALF_Z]),
          " rbound", Float64(env.mf.geoms.data[o_b + GEOM_IDX_RBOUND]))
    # ⚠ THE SWEEP IS 3D. A first version swept only x and z; adding y did NOT
    # remove the 0.020 m gap below, so that is not grid coarseness. Most
    # likely the jaw snapping shut EJECTS the smaller cube before it settles,
    # which the strict criterion then rejects — correctly, since an ejected
    # cube is not held. It is unexplained and left visible rather than
    # smoothed over; the decision rests on the TOP of the range, which is
    # monotone: 0.04 and 0.03 never hold, 0.024 does.
    var widths = [0.020, 0.015, 0.012, 0.010]
    var any_held = False
    for wi in range(len(widths)):
        var hw = widths[wi]
        env.mf.geoms.data[o_b + GEOM_IDX_HALF_X] = Scalar[DT](hw)
        env.mf.geoms.data[o_b + GEOM_IDX_HALF_Y] = Scalar[DT](hw)
        env.mf.geoms.data[o_b + GEOM_IDX_HALF_Z] = Scalar[DT](hw)
        var bh = -1.0
        var bgc = 0
        for k in range(2):
            var close = -1.0 if k == 0 else 1.0
            for ix in range(-2, 3):
              for iy in range(-1, 2):
                for iz in range(-2, 3):
                    for i in range(NQ):
                        env.d.qpos.data[i] = Scalar[DT](q0[i])
                    for i in range(NV):
                        env.d.qvel.data[i] = Scalar[DT](v0[i])
                    _ = env.step(act(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    for i in range(NQ):
                        env.d.qpos.data[i] = Scalar[DT](q0[i])
                    for i in range(NV):
                        env.d.qvel.data[i] = Scalar[DT](v0[i])
                    var hold = hold_actions(env, a_lo, a_hi, a_qa)
                    for _ in range(30):
                        step_hold(env, hold, -close)
                    var sx = Float64(env.d.site_xpos.data[GS * 3])
                    var sy = Float64(env.d.site_xpos.data[GS * 3 + 1])
                    var sz = Float64(env.d.site_xpos.data[GS * 3 + 2])
                    place_brick(env, qadr, dadr, sx + Float64(ix) * 0.015,
                                sy + Float64(iy) * 0.015,
                                sz + Float64(iz) * 0.015)
                    for _ in range(120):
                        step_hold(env, hold, close)
                    var zm = Float64(env.d.qpos.data[qadr + 2])
                    for _ in range(250):
                        step_hold(env, hold, close)
                    var ze = Float64(env.d.qpos.data[qadr + 2])
                    var gc = gripper_contacts(env, brick, grip_body, jaw)
                    var sp = brick_speed(env, dadr)
                    if ze > 0.06 and zm > 0.06 and gc > 0 and sp < 0.05:
                        if ze > bh:
                            bh = ze
                            bgc = gc
        print("   half-extent", hw, "(", 2.0 * hw, "m cube ) -> best held z",
              bh, " gripper contacts", bgc,
              "   HELD" if bh > 0.0 else "   dropped")
        if bh > 0.0:
            any_held = True
    # restore, so nothing downstream inherits a shrunken prop
    env.mf.geoms.data[o_b + GEOM_IDX_HALF_X] = Scalar[DT](0.02)
    env.mf.geoms.data[o_b + GEOM_IDX_HALF_Y] = Scalar[DT](0.02)
    env.mf.geoms.data[o_b + GEOM_IDX_HALF_Z] = Scalar[DT](0.02)
    if any_held:
        print("  -> THE PROP IS TOO WIDE FOR THIS JAW. A narrower cube is"
              " held, so the jaw and the solver are fine and the fix is"
              " `cube.xml`'s `size` (or a task with a smaller prop).")
    else:
        print("  -> NO WIDTH HOLDS, down to a 1.2 cm cube. The prop is not"
              " the problem; the pinch is. Look at the jaw mesh's CONVEX"
              " HULL — `load_mesh_hull` is what the collision path gets, and"
              " the hull of a gripper finger is not a gripper finger.")
    print()

    print("=" * 72)
    if fails == 0:
        print("=== FEASIBLE ===")
    else:
        print("=== NOT FEASIBLE —", fails, "leg(s) failed ===")
