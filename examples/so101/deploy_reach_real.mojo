"""Run a sim-trained reach policy on the PHYSICAL SO-101 follower.

The last rung of `ROADMAP_2026_08.md` §5.4: *"import -> GPU-batched training ->
policy binary -> the arm moves."* Same checkpoint as
`sac_so_arm101_reach_eval_cpu.mojo`, driving hardware instead of a simulator.

    pixi run build-serial      # ONCE
    pixi run mojo run -I . examples/so101/deploy_reach_real.mojo

## ⚠⚠ THIS MOVES A REAL ARM. Read the safety notes before the first run.

## The interesting part: the simulator is the observation function

The policy wants 21 numbers — `qpos(6) + qvel(6) + ee(3) + target(3) +
ee_to_target(3)`. The arm can only supply the first twelve:

| term | where it comes from on hardware |
|---|---|
| qpos | servo ticks -> `SimJointMap.to_sim` (gated vs so101-nexus) |
| qvel | `Present_Velocity`, register 58, one more sync-read |
| ee | ⚠ NO SENSOR GIVES THIS — forward kinematics on the measured qpos |
| target | ours to choose; there is no target in the real world |
| ee_to_target | derived |

So this program keeps a `Phyics3dEnv` alive purely as a **kinematics oracle**:
measured `qpos` goes in through `set_state` (which runs
`_sync_mocap_to_fields` then `_fields_fk`), and the end-effector's world
position comes out. It reuses the FK gated against MuJoCo in
`tests/robots/test_so_arm101_vs_mujoco.mojo` — the same model the policy
trained in, used as a sensor.

⚠ Being honest about what this demonstrates: the target is three numbers WE
pick, so the arm is not finding anything. This is RL-learned inverse
kinematics, and plain IK does that job better. Its value is that it exercises
every piece a later task needs — obs construction, action mapping, control
rate, safety — and yields a transfer NUMBER: commanded point vs. where the
end-effector actually ends up.

## Safety

Four guards, on top of the two `SO101Arm` already enforces (clamp to the
calibrated range; clamp the step to `present +/- max_step_ticks`):

1. the follower's goal is parked on its OWN present pose before torque is
   armed, so arming holds instead of snapping to a stale `Goal_Position`;
2. a partial `sync_read` skips the tick rather than commanding a half-updated
   pose;
3. the commanded angle is clamped to the MODEL's `ctrlrange` before it is ever
   mapped back to ticks — the policy trained inside those limits and has no
   reason to be trusted outside them;
4. torque is released in a `finally`.

⚠⚠ A `finally` does NOT run on an abort or a signal. If this dies hard, the
follower is left holding its pose — run `pixi run soarm-torque-off`. That is
the recovery, not the `finally`.
"""

from std.ffi import external_call
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.robot.so101.sim_map import SimJointMap
from mojo_rl.utils.fmt import col, fixed, pad_left, pad_right

comptime FOLLOWER_PORT = "/dev/cu.usbmodem5B8E1139971"
comptime CHECKPOINT_PATH = "sac_so_arm101_reach.ckpt"

comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM  # 21
comptime ACT_DIM = EnvT.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 256
comptime REPLAY_CAPACITY = 100_000
comptime ACTION_SCALE = Scalar[DT](2.0)  # radians; MUST match the trainer

comptime HZ = 50
"""⚠ 50, NOT 30 — IT MUST MATCH THE RATE THE POLICY TRAINED AT.
`SoArmReachConfig` is `FRAME_SKIP = 10` over a 0.002 s timestep, i.e. a 50 Hz
control rate, and the action is a POSITION TARGET: at 30 Hz every commanded
pose is held 67% longer than the policy ever saw, which changes the closed
loop, not just the latency. This was 30 because the loop could not go faster —
it spent six serial round trips per tick reading velocity one joint at a time.
`SO101Arm.read_velocities` makes that one `sync_read`, so the tick is now
three transactions (positions, velocities, goals) instead of eight."""
comptime SECONDS = 5
"""Short on purpose. This program had never been executed against hardware, so
the first live run is a BRING-UP of the guards, not a demonstration of the
policy — five seconds is enough to read the achieved rate and see the arm
settle, and short enough that a guard that does not work costs little."""
comptime MAX_STEP_TICKS = 25
"""Per-tick slew bound, in servo ticks. At 50 Hz this is 25 * 50 = 1250
ticks/s ~= 110 deg/s.

⚠ WAS 60 (~264 deg/s), AND THE DRY RUN IS WHY IT IS NOT. It measured the
policy asking for a mean move of **1252 ticks** and a max of 2197 — the
distance from the arm's parked pose to the pose the policy wants, roughly 110
degrees. The clamp is what turns that into a slew rather than a snap, so its
value IS the opening motion's speed: at 60 the arm crosses that gap in 0.4 s,
at 25 in about 1 s. Raise it once the first live run has shown the direction
is right."""

comptime SMOOTH = 0.15
"""Low-pass on the COMMAND: `cmd <- (1-a)*cmd + a*policy`. 1.0 disables it.

⚠⚠ THE RAW COMMAND STREAM IS NOT SOMETHING TO SEND A SERVO. Measured in sim
over 24 episodes (greedy, after the arm has arrived): the commanded position
moves **~500 ticks per joint per control step and reverses direction on 83% of
consecutive steps**. That is a 44-degree-per-step bang-bang command at 50 Hz.
`max_step_ticks` bounds each move but does not stop the reversals, so the arm
would sit at the target buzzing — continuous current reversal through the gear
train, which is the "will this damage the motors" question answered YES-ish.

The reward cannot see it: `tolerance` has a 0.25 m margin against a 0.02 m
success radius, so hovering 40 mm away scores 0.985 per step. Shake is free.

⚠ AND SMOOTHING IS A DEVIATION FROM TRAINING, so it was measured rather than
assumed — same 24 targets, greedy, in sim:

    alpha | reached <=20mm | mean closest | chatter ticks/step | return
     1.00 |        15/24   |     20.4 mm  |            2520    |  480.5
     0.50 |        15/24   |     18.6 mm  |            1029    |  480.2
     0.30 |        15/24   |     17.9 mm  |             665    |  479.1
     0.20 |        16/24   |     16.9 mm  |             470    |  476.9
     0.10 |        17/24   |     18.7 mm  |             223    |  474.7
     0.05 |        12/24   |     26.5 mm  |              72    |  475.5

Anything in 0.1-0.3 costs NOTHING in task terms — the reach rate is flat
inside the noise of 24 episodes — and buys a 4x to 11x calmer command. 0.05
breaks the policy (the filter outruns the task). 0.15 sits in the middle of
the free band. THE REAL FIX IS AN ACTION-RATE PENALTY IN THE REWARD AND A
RETRAIN; this is the guard for a policy that does not have one."""

comptime SUCCESS_MM = 20.0
"""`SoArmReachConfig.TARGET_RADIUS` in millimetres — the radius inside which
the task calls it reached."""

# The target, in the model's base frame. Inside the shell the policy trained
# on (0.15-0.30 m, elevation 0.17-1.22 rad), or it is being asked for a pose
# it never saw.
comptime TARGET_X = 0.22
comptime TARGET_Y = 0.0
comptime TARGET_Z = 0.18


def col_name(i: Int) -> String:
    """Joint name, padded, so the self-check reads as a table."""
    return pad_right(String(joint_name(i)), 12)


def _sleep_until(deadline_ns: Int):
    """Spin. Measured better than `usleep` on this box — see
    `examples/so101/teleop.mojo` for the table."""
    while perf_counter_ns() < deadline_ns:
        pass


def main() raises:
    # ⚠⚠ THE ARM MOVES ONLY WITH AN EXPLICIT `--live`. Everything else — the
    # bus, the observation, the forward kinematics, the policy, the joint
    # mapping, the clamps, the filter and the loop rate — runs identically in
    # DRY RUN, which never arms torque and never writes a goal. A first
    # contact with hardware that exercises every path except the dangerous one
    # is worth more than a cautious live run, because a fault in ANY of those
    # paths shows up as a printed number instead of as a motion.
    # ⚠ RUNTIME, NOT COMPTIME, for `seconds` and `step`. Hardware bring-up is
    # a sweep — the first live run showed the arm rate-limited by the clamp for
    # its whole duration, and answering "how long does it need" or "how much
    # clamp is enough" should not cost a two-minute rebuild each time. `--live`
    # stays a flag rather than a default for the reason it always was.
    var live = False
    var seconds = SECONDS
    var step_ticks = MAX_STEP_TICKS
    var smooth = SMOOTH
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--live":
            live = True
        elif a == "--seconds" and i + 1 < len(args):
            seconds = Int(String(args[i + 1]))
        elif a == "--step" and i + 1 < len(args):
            step_ticks = Int(String(args[i + 1]))
        elif a == "--smooth" and i + 1 < len(args):
            smooth = Float64(String(args[i + 1]))
    print("=" * 70)
    if live:
        print("SO-ARM101 reach — SIM-TRAINED POLICY ON THE REAL ARM  [LIVE]")
    else:
        print("SO-ARM101 reach — DRY RUN (no torque, no goals written)")
        print("  pass --live to actually move the arm")
    print("=" * 70)

    # ── the policy ────────────────────────────────────────────────────────
    var agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN](
        action_scale=ACTION_SCALE,
    )
    try:
        agent.load(CHECKPOINT_PATH)
    except e:
        print("ERROR loading", CHECKPOINT_PATH, "-", e)
        print("Train first: examples/so101/sac_so_arm101_reach_training_gpu.mojo")
        return
    print("  policy          =", CHECKPOINT_PATH)

    # ── the kinematics oracle ─────────────────────────────────────────────
    # A full env, used for FK only. It is never stepped: `set_state` runs the
    # mocap sync and forward kinematics, which is all the observation needs.
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    _ = env.reset()

    var sf = SoArm101Model.make_spec_fields[DType.float64]()
    var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, SO101_N)
    var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, SO101_N)
    var lo = InlineArray[Float64, SO101_N](fill=0.0)
    var hi = InlineArray[Float64, SO101_N](fill=0.0)
    for i in range(SO101_N):
        lo[i] = Float64(lo_col[i])
        hi[i] = Float64(hi_col[i])
    var jmap = SimJointMap.identity(lo^, hi^)

    # ── the arm ───────────────────────────────────────────────────────────
    print("  opening         =", FOLLOWER_PORT)
    var arm = SO101Arm(String(FOLLOWER_PORT), max_step_ticks=step_ticks)
    arm.bus.timeout_ms = 20
    print("  target          = (", TARGET_X, TARGET_Y, TARGET_Z, ")")
    print("=" * 70)

    var raw = InlineArray[Int32, SO101_N](fill=0)
    if arm.read_positions(Span(raw)) != SO101_N:
        raise Error("deploy: follower did not report 6 positions — not arming")

    # ── the mapping self-check ────────────────────────────────────────────
    #
    # ⚠⚠ `from_sim` IS THE ONLY LINK IN THIS CHAIN THAT NOTHING HAS EVER
    # EXERCISED. `to_sim` is gated against so101-nexus's reference function and
    # is driven every frame by `teleop_sim.mojo`; its inverse is used HERE and
    # nowhere else. A sign error in it does not produce an error — it produces
    # a mirrored pose, at full slew, on a real arm.
    #
    # So round-trip each joint's MEASURED position through both directions
    # before anything is armed. It is not proof the convention is right (a
    # convention wrong in both directions round-trips perfectly — `to_sim` is
    # what pins that, and it is separately gated), but it is proof the two are
    # consistent, which is the failure this file could introduce on its own.
    var worst = 0
    var n_outside = 0
    for i in range(SO101_N):
        # ⚠⚠ A CLAMPED JOINT CANNOT ROUND-TRIP, AND THAT IS NOT A DEFECT.
        # `to_sim` clamps to the MODEL's `ctrlrange`, and three of these
        # joints have calibrated travel that EXCEEDS it (the `gap` column in
        # `SimJointMap.range_report`, and a faithful port of the upstream
        # ranges — see `teleop_sim.mojo`). Sitting outside the model's range,
        # `from_sim(to_sim(raw))` returns the LIMIT, not `raw`, by
        # construction. Asserting on it would block a live run for a condition
        # the mapping is documented to have — the first version of this check
        # did exactly that, on `shoulder_lift`, 68 ticks out.
        var over = jmap.clamped_by(arm.cal, i, raw[i])
        var rad = jmap.to_sim(arm.cal, i, raw[i])
        var back = jmap.from_sim(arm.cal, i, rad)
        var err = Int(back) - Int(raw[i])
        if err < 0:
            err = -err
        var note = String("")
        if over > 0.0:
            n_outside += 1
            note = " ⚠ OUTSIDE the model range by " + fixed(over, 3) + " rad"
        elif err > worst:
            worst = err
        print(
            "  " + col_name(i), "raw", pad_left(String(Int(raw[i])), 6),
            "-> rad", col(rad, 7, 3),
            "-> raw", pad_left(String(Int(back)), 6),
            "  (err " + String(err) + ")" + note,
        )
    if worst > 2:
        raise Error(
            "deploy: to_sim/from_sim do not round-trip inside the model range"
            " (worst " + String(worst) + " ticks) — NOT arming. A sign or"
            " offset in the joint mapping is inconsistent, and the failure it"
            " produces on hardware is a MIRRORED pose at full slew."
        )
    print(
        "  mapping round-trips inside the model range, worst", worst, "ticks"
    )
    if n_outside > 0:
        # Not fatal, and worth saying out loud: until the arm moves back
        # inside, the policy's observation carries a CLAMPED angle rather than
        # the arm's true one, so its first action is taken on a pose that is
        # off by that much. Every command it issues is clamped INTO the range,
        # so the first motion fixes it.
        print(
            "  ⚠", n_outside, "joint(s) are parked outside the model's range."
            " The policy's first\n     observation is clamped there; its"
            " first command moves them back inside."
        )
    print()

    # Guard 1: park the goal on the present pose BEFORE arming torque.
    # ⚠ SKIPPED ENTIRELY IN DRY RUN — writing `Goal_Position` on an unarmed
    # servo is harmless, but not writing it at all is the only way to be sure
    # a dry run cannot move anything, and "sure" is the point of the mode.
    if live:
        arm.set_position_mode()
        var hold = arm.max_step_ticks
        arm.max_step_ticks = 0
        arm.write_goals(Span(raw))
        arm.max_step_ticks = hold
        arm.set_torque(True)
        print("follower torque ON\n")
    else:
        print("dry run — torque left OFF, the arm is backdrivable\n")

    var obs = List[Scalar[DT]]()
    for _ in range(OBS_DIM):
        obs.append(Scalar[DT](0))
    var action = List[Scalar[DT]]()
    for _ in range(ACT_DIM):
        action.append(Scalar[DT](0))

    var qp = List[Float64]()
    var qv = List[Float64]()
    for _ in range(SO101_N):
        qp.append(0.0)
        qv.append(0.0)

    var vraw = InlineArray[Int32, SO101_N](fill=0)
    var goals = InlineArray[Int32, SO101_N](fill=0)
    # ⚠ SEEDED FROM THE ARM'S OWN POSE, not from zero. A filter starting at
    # zero would spend its first ticks sweeping the arm from wherever it is
    # toward the folded pose — a large unwanted motion, produced by the very
    # thing that is there to make motion gentler.
    var cmd = List[Float64]()
    for i in range(SO101_N):
        cmd.append(jmap.to_sim(arm.cal, i, raw[i]))
    var period_ns = 1_000_000_000 // HZ
    var ticks = HZ * seconds
    var dropped = 0
    var best_err = 1.0e9
    var last_err = 0.0
    var inside = 0
    var max_step = 0.0
    var sum_step = 0.0
    var n_step = 0.0
    var reversals = 0.0
    var last_delta = List[Float64](length=SO101_N, fill=0.0)
    var last_goal = List[Float64](length=SO101_N, fill=0.0)
    for i in range(SO101_N):
        last_goal[i] = Float64(raw[i])

    var loop_t0 = perf_counter_ns()
    try:
        for t in range(ticks):
            var t0 = perf_counter_ns()

            # ── measure ───────────────────────────────────────────────────
            if arm.read_positions(Span(raw)) != SO101_N:
                dropped += 1
                _sleep_until(t0 + period_ns)
                continue
            # ⚠ ONE ROUND TRIP, NOT SIX. `Present_Velocity` is sign-magnitude
            # at bit 15, which `sync_read` decodes through `sign_bit_for` —
            # the same path the positions take. Ticks per second -> rad/s uses
            # the same 4095 the position map does, not 4096.
            var n_vel = arm.read_velocities(Span(vraw))
            if n_vel != SO101_N:
                dropped += 1
                _sleep_until(t0 + period_ns)
                continue
            for i in range(SO101_N):
                qp[i] = jmap.to_sim(arm.cal, i, raw[i])
                qv[i] = Float64(vraw[i]) * 2.0 * 3.141592653589793 / 4095.0

            # ── the simulator AS A SENSOR: qpos -> FK -> end-effector ─────
            env.d.mocap_pos.data[SoArm101ReachConfig.TARGET_BODY * 3 + 0] = (
                Scalar[DT](TARGET_X)
            )
            env.d.mocap_pos.data[SoArm101ReachConfig.TARGET_BODY * 3 + 1] = (
                Scalar[DT](TARGET_Y)
            )
            env.d.mocap_pos.data[SoArm101ReachConfig.TARGET_BODY * 3 + 2] = (
                Scalar[DT](TARGET_Z)
            )
            env.set_state(qp, qv)

            var st = env.get_state()
            for i in range(OBS_DIM):
                obs[i] = Scalar[DT](st.data[i])

            # ── act ──────────────────────────────────────────────────────
            agent.select_greedy_action(obs, action)
            for i in range(SO101_N):
                # Guard 3: the policy trained INSIDE the model's ctrlrange and
                # is not trusted outside it.
                # Read the limits back off the MAP, not off `lo`/`hi` — those
                # were transferred into it, and a second copy is a second
                # thing to drift.
                var a = min(
                    jmap.sim_hi[i], max(jmap.sim_lo[i], Float64(action[i]))
                )
                # Guard 5: low-pass the command — see `SMOOTH`. AFTER the
                # clamp, so the filter can never carry a state the limits
                # already rejected, and BEFORE `from_sim`, so what is filtered
                # is an ANGLE and not a tick count whose sign convention
                # differs per joint.
                cmd[i] = (1.0 - smooth) * cmd[i] + smooth * a
                goals[i] = jmap.from_sim(arm.cal, i, cmd[i])
                # The move the servo is being asked for THIS tick, in ticks —
                # the quantity `max_step_ticks` bounds and the one that decides
                # whether this is gentle. Tracked in dry run too, where it is
                # the whole output.
                var delta = Float64(goals[i]) - last_goal[i]
                # ⚠ THE REVERSAL RATE IS THE SHAKE, and it is a different
                # quantity from the step SIZE. A goal that advances 25 ticks
                # every tick in one direction is a smooth slew; one that
                # alternates +25/-25 at the same size is a 25 Hz buzz through
                # the gear train. Only this counter tells them apart, and it
                # is the number to watch when tuning `--smooth`.
                if delta * last_delta[i] < 0.0:
                    reversals += 1.0
                last_delta[i] = delta
                last_goal[i] = Float64(goals[i])
                var step = Float64(goals[i] - raw[i])
                if step < 0.0:
                    step = -step
                if step > max_step:
                    max_step = step
                sum_step += step
                n_step += 1.0
            if live:
                arm.write_goals(Span(goals))

            # ── report ───────────────────────────────────────────────────
            var b = 12  # qpos(6) + qvel(6)
            var dx = Float64(obs[b + 6])
            var dy = Float64(obs[b + 7])
            var dz = Float64(obs[b + 8])
            last_err = (dx * dx + dy * dy + dz * dz) ** 0.5
            if last_err < best_err:
                best_err = last_err
            if last_err * 1000.0 <= SUCCESS_MM:
                inside += 1
            if t % HZ == 0:
                print(
                    "t=" + String(t // HZ) + "s  ee->target = "
                    + fixed(last_err * 1000.0, 1) + " mm"
                )
            _sleep_until(t0 + period_ns)
    finally:
        # Unconditional, dry run included: it costs one packet and it is the
        # net under every path that could have armed something.
        arm.set_torque(False)
        print("\nfollower torque OFF")

    print("=" * 70)
    print("TRANSFER RESULT — sim-trained reach on hardware")
    print("  target            = (", TARGET_X, TARGET_Y, TARGET_Z, ")")
    print("  closest approach  =", fixed(best_err * 1000.0, 1), "mm")
    print("  final error       =", fixed(last_err * 1000.0, 1), "mm")
    # ⚠ CLOSEST APPROACH IS THE FLATTERING NUMBER, and on its own it is the
    # wrong one: in sim this policy's mean CLOSEST approach is ~20 mm while
    # its mean FINAL distance is ~38 mm — it passes through the target and
    # drifts off rather than arriving. "Held" is the question a reach task is
    # actually asking, so it gets its own row.
    print(
        "  steps inside", fixed(SUCCESS_MM, 0), "mm =", inside, "/", ticks,
        "(" + fixed(100.0 * Float64(inside) / Float64(ticks), 0) + "%)",
    )
    print("  dropped ticks     =", dropped, "/", ticks)
    # ⚠ THE RATE THE LOOP ACTUALLY HELD, not the one it asked for. The bus is
    # the budget: three round trips a tick at ~1.3 ms each leaves ~16 ms of
    # the 20 ms period, and a serial hiccup eats it silently. A policy trained
    # at 50 Hz running at 35 would look like a worse policy.
    var achieved = Float64(ticks) * 1e9 / Float64(perf_counter_ns() - loop_t0)
    _ = seconds
    print(
        "  control rate      =", fixed(achieved, 1), "Hz achieved of",
        Int(HZ), "asked (policy trained at 50 Hz)",
    )
    # ⚠ THE ATTENUATION, NOT JUST THE ALPHA. An EMA's steady-state response to
    # a signal that alternates every step is `a / (2 - a)`, so 0.15 leaves 8%
    # of the chatter standing and 0.05 leaves 2.6%. Printing only the alpha
    # invites reading 0.15 as "smoothed" when it is barely filtered — and the
    # policy's command reverses direction on 83-97% of steps, which is exactly
    # the signal this formula is about.
    print(
        "  smoothing         = alpha", fixed(smooth, 3),
        "-> leaves", fixed(100.0 * smooth / (2.0 - smooth), 1),
        "% of an every-step alternation (1.0 = raw policy)",
    )
    print(
        "  commanded step    = mean",
        fixed(sum_step / n_step, 1) if n_step > 0 else String("n/a"),
        "ticks, max", fixed(max_step, 1),
        "(clamp is " + String(step_ticks) + ")",
    )
    # ⚠⚠ THE CLAMP DECIDES THE MOTION WHENEVER THIS RATIO IS LARGE. Measured on
    # the first live run: mean demand 612 ticks against a clamp of 25, i.e. the
    # arm ran at 1/24 of what the policy asked for, for the whole episode, and
    # was still approaching when time ran out. `max_step_ticks` HAS NO
    # COUNTERPART IN SIM — there the commanded pose goes straight to the
    # `<position>` actuator and only its gain limits the response — so a tight
    # clamp is a sim2real gap, not just a safety margin. Read this line before
    # concluding anything about the policy.
    print(
        "  goal reversals    =",
        fixed(100.0 * reversals / n_step, 0) if n_step > 0 else String("n/a"),
        "% of writes changed DIRECTION  <- this is the shake",
    )
    var throttle = (sum_step / n_step) / Float64(step_ticks) if n_step > 0 else 0.0
    if throttle > 2.0:
        print(
            "  ⚠ RATE-LIMITED: the policy asked for",
            fixed(throttle, 1) + "x the clamp on average.",
            "\n     The CLAMP shaped this run, not the policy. Raise --step"
            " or lengthen --seconds",
            "\n     before reading the distance as the policy's fault.",
        )
    if not live:
        print("  ⚠ DRY RUN — nothing was written to the arm. Add --live.")
    # ⚠ The error is FK-derived, not measured with a ruler: it is where the
    # model says the jaw is given the servo angles. A systematic kinematic
    # error is invisible to it. The sim task's own success radius is 20 mm.
    print("  (FK-derived, not externally measured; task radius is 20 mm)")
    print("=" * 70)
