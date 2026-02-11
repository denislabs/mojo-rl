"""HalfCheetah Physics Diagnostic — per-step logging of the actual environment.

Runs the real HalfCheetah model with zero actions (free fall),
logging per-body Z positions, contacts, penetration, and velocities
at every physics step.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_cheetah_diagnostics.mojo
"""

from math import sqrt, pi
from builtin.math import abs

from envs.half_cheetah_gc import HalfCheetahGC
from envs.half_cheetah_gc.half_cheetah_def import (
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    BODY_HEAD,
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    CAPSULE_RADIUS,
)

from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver.pgs_solver import PGSSolver
from physics3d.solver.newton_solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import forward_kinematics


fn body_name(id: Int) -> String:
    if id == 0:
        return "torso "
    elif id == 1:
        return "bthigh"
    elif id == 2:
        return "bshin "
    elif id == 3:
        return "bfoot "
    elif id == 4:
        return "fthigh"
    elif id == 5:
        return "fshin "
    elif id == 6:
        return "ffoot "
    elif id == 7:
        return "head  "
    return "???"


fn main():
    print("")
    print("=" * 80)
    print("HalfCheetah Physics Diagnostic — Free Fall (zero actions)")
    print("=" * 80)
    print("")

    # Create the real HalfCheetah environment
    var env = HalfCheetahGC[DType.float64, False]()
    _ = env.reset()

    var dt = Float64(env.model.timestep)
    var num_steps = 500  # 1 second at 500Hz

    print("Physics dt =", dt, "s, running", num_steps, "steps (", Float64(num_steps) * dt, "s)")
    print("Initial torso z (qpos[ROOTZ]) =", env.data.qpos[JOINT_ROOTZ])
    print("Capsule radius =", CAPSULE_RADIUS)
    print("")

    # =========================================================================
    # Part 1: Log body Z positions every N steps
    # =========================================================================
    print("--- PART 1: Body Z positions (world frame) ---")
    print("")

    # Print initial FK positions
    forward_kinematics(env.model, env.data)
    print("  Initial body Z positions (from FK):")
    for b in range(NBODY):
        var bz = env.data.xpos[b * 3 + 2]
        print("    ", body_name(b), ": z =", bz)
    print("")

    print("  step | time   | rootz    | vz       | torso_z  | bfoot_z  | ffoot_z  | contacts | max_pen  | max_imp_n")
    print("  " + "-" * 105)

    var max_pen_ever = Float64(0.0)
    var max_bounce_vel = Float64(0.0)
    var first_contact_step = -1

    for i in range(num_steps):
        # Zero forces (free fall — no actuation)
        env.data.clear_forces()

        # Single physics step
        EulerIntegrator[PGSSolver].step(env.model, env.data)

        var time = Float64(i + 1) * dt
        var rootz = Float64(env.data.qpos[JOINT_ROOTZ])
        var vz = Float64(env.data.qvel[JOINT_ROOTZ])
        var nc = Int(env.data.num_contacts)

        # Get body z positions from xpos (already updated by step)
        var torso_z = Float64(env.data.xpos[BODY_TORSO * 3 + 2])
        var bfoot_z = Float64(env.data.xpos[BODY_BFOOT * 3 + 2])
        var ffoot_z = Float64(env.data.xpos[BODY_FFOOT * 3 + 2])

        # Find max penetration and impulse across all contacts
        var max_pen = Float64(0.0)
        var max_imp = Float64(0.0)
        for c in range(nc):
            var pen = -Float64(env.data.contacts[c].dist)
            var imp = Float64(env.data.contacts[c].impulse_n)
            if pen > max_pen:
                max_pen = pen
            if imp > max_imp:
                max_imp = imp

        if max_pen > max_pen_ever:
            max_pen_ever = max_pen

        if nc > 0 and vz > Float64(0.0) and vz > max_bounce_vel:
            max_bounce_vel = vz

        if nc > 0 and first_contact_step < 0:
            first_contact_step = i + 1

        # Print: first 10 steps, around first contact, then every 25 steps
        var print_step = False
        if i < 10:
            print_step = True
        elif first_contact_step > 0 and i + 1 >= first_contact_step - 2 and i + 1 <= first_contact_step + 50:
            print_step = True
        elif (i + 1) % 25 == 0:
            print_step = True

        if print_step:
            print(
                "  ",
                i + 1,
                " | ",
                time,
                " | ",
                rootz,
                " | ",
                vz,
                " | ",
                torso_z,
                " | ",
                bfoot_z,
                " | ",
                ffoot_z,
                " | ",
                nc,
                " | ",
                max_pen,
                " | ",
                max_imp,
            )

    print("")
    print("  SUMMARY Part 1:")
    print("    First contact at step:", first_contact_step, "(t =", Float64(first_contact_step) * dt, "s)")
    print("    Max penetration ever:", max_pen_ever, "m")
    print("    Max bounce velocity:", max_bounce_vel, "m/s")
    print("    Final rootz:", env.data.qpos[JOINT_ROOTZ])
    print("    Final vz:", env.data.qvel[JOINT_ROOTZ])
    print("")

    # =========================================================================
    # Part 2: Detailed contact log around first contact
    # =========================================================================
    print("--- PART 2: Detailed contact info — fresh run, first 50 steps after contact ---")
    print("")

    _ = env.reset()

    var contact_started = False
    var contact_steps_logged = 0

    for i in range(num_steps):
        env.data.clear_forces()
        EulerIntegrator[PGSSolver].step(env.model, env.data)

        var nc = Int(env.data.num_contacts)

        if nc > 0 and not contact_started:
            contact_started = True
            print("  First contact at step", i + 1, "(t =", Float64(i + 1) * dt, "s)")
            print("  rootz =", env.data.qpos[JOINT_ROOTZ], "vz =", env.data.qvel[JOINT_ROOTZ])
            print("")

        if contact_started and contact_steps_logged < 30:
            contact_steps_logged += 1
            print("  Step", i + 1, ": contacts =", nc, "rootz =", env.data.qpos[JOINT_ROOTZ], "vz =", env.data.qvel[JOINT_ROOTZ])
            for c in range(nc):
                var ct = env.data.contacts[c]
                print(
                    "    c",
                    c,
                    ": body_a =",
                    body_name(Int(ct.body_a)),
                    "body_b =",
                    ct.body_b,
                    "pen =",
                    -Float64(ct.dist),
                    "pos_z =",
                    ct.pos_z,
                    "normal = (",
                    ct.normal_x,
                    ct.normal_y,
                    ct.normal_z,
                    ") imp_n =",
                    ct.impulse_n,
                )
            print("")

    # =========================================================================
    # Part 3: All body Z positions at key moments
    # =========================================================================
    print("--- PART 3: All body Z positions at key moments ---")
    print("")

    _ = env.reset()
    forward_kinematics(env.model, env.data)

    # Log at specific step numbers
    var log_steps = List[Int]()
    log_steps.append(0)
    log_steps.append(50)
    log_steps.append(100)
    log_steps.append(150)
    log_steps.append(200)
    log_steps.append(250)
    log_steps.append(300)
    log_steps.append(400)
    log_steps.append(500)

    var log_idx = 0
    if log_idx < len(log_steps) and log_steps[log_idx] == 0:
        print("  Step 0 (t=0):")
        for b in range(NBODY):
            print("    ", body_name(b), ": z =", env.data.xpos[b * 3 + 2])
        print("    rootz =", env.data.qpos[JOINT_ROOTZ], "rooty =", env.data.qpos[JOINT_ROOTY])
        print("    contacts:", env.data.num_contacts)
        print("")
        log_idx += 1

    for i in range(num_steps):
        env.data.clear_forces()
        EulerIntegrator[PGSSolver].step(env.model, env.data)

        if log_idx < len(log_steps) and (i + 1) == log_steps[log_idx]:
            print("  Step", i + 1, "(t =", Float64(i + 1) * dt, "s):")
            for b in range(NBODY):
                var bz = Float64(env.data.xpos[b * 3 + 2])
                var marker = String("")
                if bz < Float64(CAPSULE_RADIUS):
                    marker = " *** BELOW GROUND ***"
                elif bz < Float64(CAPSULE_RADIUS) + 0.01:
                    marker = " (near ground)"
                print("    ", body_name(b), ": z =", bz, marker)

            print("    rootz =", env.data.qpos[JOINT_ROOTZ], "vz =", env.data.qvel[JOINT_ROOTZ])
            print("    rooty =", env.data.qpos[JOINT_ROOTY], "vy =", env.data.qvel[JOINT_ROOTY])
            print("    contacts:", env.data.num_contacts)

            # Log joint angles
            print("    joint angles: bthigh=", env.data.qpos[3], "bshin=", env.data.qpos[4], "bfoot=", env.data.qpos[5])
            print("                  fthigh=", env.data.qpos[6], "fshin=", env.data.qpos[7], "ffoot=", env.data.qpos[8])
            print("")
            log_idx += 1

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
