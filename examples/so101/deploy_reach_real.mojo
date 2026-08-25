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
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from mojo_rl.robot.feetech.control_table import SIZE_2, STS_PRESENT_VELOCITY
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.robot.so101.sim_map import SimJointMap
from mojo_rl.utils.fmt import col, fixed

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

comptime HZ = 30
comptime SECONDS = 20
comptime MAX_STEP_TICKS = 60

# The target, in the model's base frame. Inside the shell the policy trained
# on (0.15-0.30 m, elevation 0.17-1.22 rad), or it is being asked for a pose
# it never saw.
comptime TARGET_X = 0.22
comptime TARGET_Y = 0.0
comptime TARGET_Z = 0.18


def _sleep_until(deadline_ns: Int):
    """Spin. Measured better than `usleep` on this box — see
    `examples/so101/teleop.mojo` for the table."""
    while perf_counter_ns() < deadline_ns:
        pass


def main() raises:
    print("=" * 70)
    print("SO-ARM101 reach — SIM-TRAINED POLICY ON THE REAL ARM")
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
    var arm = SO101Arm(String(FOLLOWER_PORT), max_step_ticks=MAX_STEP_TICKS)
    arm.bus.timeout_ms = 20
    print("  target          = (", TARGET_X, TARGET_Y, TARGET_Z, ")")
    print("=" * 70)

    var raw = InlineArray[Int32, SO101_N](fill=0)
    if arm.read_positions(Span(raw)) != SO101_N:
        raise Error("deploy: follower did not report 6 positions — not arming")

    # Guard 1: park the goal on the present pose BEFORE arming torque.
    arm.set_position_mode()
    var hold = arm.max_step_ticks
    arm.max_step_ticks = 0
    arm.write_goals(Span(raw))
    arm.max_step_ticks = hold
    arm.set_torque(True)
    print("follower torque ON\n")

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

    var goals = InlineArray[Int32, SO101_N](fill=0)
    var period_ns = 1_000_000_000 // HZ
    var ticks = HZ * SECONDS
    var dropped = 0
    var best_err = 1.0e9
    var last_err = 0.0

    try:
        for t in range(ticks):
            var t0 = perf_counter_ns()

            # ── measure ───────────────────────────────────────────────────
            if arm.read_positions(Span(raw)) != SO101_N:
                dropped += 1
                _sleep_until(t0 + period_ns)
                continue
            for i in range(SO101_N):
                qp[i] = jmap.to_sim(arm.cal, i, raw[i])
                # ⚠ `Present_Velocity` is sign-magnitude at bit 15, which
                # `read_register` decodes. Ticks per second -> rad/s uses the
                # same 4095, not 4096.
                var v = arm.bus.read_register(
                    arm.ids[i], STS_PRESENT_VELOCITY, SIZE_2
                )
                qv[i] = Float64(v) * 2.0 * 3.141592653589793 / 4095.0

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
                goals[i] = jmap.from_sim(arm.cal, i, a)
            arm.write_goals(Span(goals))

            # ── report ───────────────────────────────────────────────────
            var b = 12  # qpos(6) + qvel(6)
            var dx = Float64(obs[b + 6])
            var dy = Float64(obs[b + 7])
            var dz = Float64(obs[b + 8])
            last_err = (dx * dx + dy * dy + dz * dz) ** 0.5
            if last_err < best_err:
                best_err = last_err
            if t % HZ == 0:
                print(
                    "t=" + String(t // HZ) + "s  ee->target = "
                    + fixed(last_err * 1000.0, 1) + " mm"
                )
            _sleep_until(t0 + period_ns)
    finally:
        arm.set_torque(False)
        print("\nfollower torque OFF")

    print("=" * 70)
    print("TRANSFER RESULT — sim-trained reach on hardware")
    print("  target            = (", TARGET_X, TARGET_Y, TARGET_Z, ")")
    print("  closest approach  =", fixed(best_err * 1000.0, 1), "mm")
    print("  final error       =", fixed(last_err * 1000.0, 1), "mm")
    print("  dropped ticks     =", dropped, "/", ticks)
    # ⚠ The error is FK-derived, not measured with a ruler: it is where the
    # model says the jaw is given the servo angles. A systematic kinematic
    # error is invisible to it. The sim task's own success radius is 20 mm.
    print("  (FK-derived, not externally measured; task radius is 20 mm)")
    print("=" * 70)
