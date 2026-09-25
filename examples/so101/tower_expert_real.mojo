"""The scripted cube-in-bowl expert on the REAL arm: the overhead camera reads
the brick and the bowl, the sim planner plans the grasp, the servos run it.

    pixi run -e jetson mojo run -I . examples/so101/tower_expert_real.mojo \\
        --camera /dev/soarm_cam_overhead                # DRY RUN: reads, plans, prints
    pixi run -e jetson mojo run -I . examples/so101/tower_expert_real.mojo \\
        --camera /dev/soarm_cam_overhead --arm --episodes 10
    pixi run mojo run -I . examples/so101/tower_expert_real.mojo \\
        --plan-only projects/so101-tower/poses/cube-in-bowl-printed_start_poses.csv

## WHAT IT MEASURES

The expert reads nothing but the arm's joints and the two object poses. If it
cannot put the cube in the bowl on the rig with good poses, the gap is the
physics and the servos, and no perception or learning fixes it; if it can, a
sim-derived controller works on the real arm. Vision is out of the loop
except for the two poses read before the episode and the outcome read after.

## ⚠⚠ THIS MOVES A REAL ARM (with `--arm`)

Without `--arm` nothing is energised and no goal is written: the camera, the
estimates, the joint map, the planner and the plan all run, and the plan is
printed. With `--arm`:
- the goal is parked on the present pose before torque goes on;
- every goal is clamped to the calibrated range and to present ±80 ticks
  until the arm has caught up, ±512 after (`SO101Arm.write_goals`, the
  recorders' two-phase clamp);
- Enter during an episode ABORTS it: the arm ramps back to where the episode
  started;
- the run ends through `deploy_shutdown.return_and_release` from a `finally`
  (back to the pose the run started from, release on the operator's word).
  ⚠ A `finally` does not run on a signal: after Ctrl-C the follower holds its
  pose — `pixi run soarm-torque-off`.

## THE PLAN IS THE SIM'S (`tasks/so101_tower_expert_plan.mojo`)

`TowerGraspPlanner` in the human posture with `clear_plan`, as
`tower_expert_record.mojo --posture human --clear-plan` runs it: the tilt
drawn from 20..65 deg, the pinch snapped to the brick's faces, the grasp height
set against the desk and every pose checked against the base, the stand, the
bowl and the brick. The sim env it plans on is put in the RIG's state first:
the arm's six joints as measured (through the follower zero,
`SimJointMap.tower_follower`) and the brick and bowl at the estimates (their
heights are the sim's own resting heights). The place is planned once, from
the start estimate of the bowl.

⚠ `--desk-clear-mm` defaults to -3 (the fingers planned 3 mm INTO the desk),
not the sim's -8: in sim the press is what lets the moving jaw squeeze the
brick against the fixed finger; on the rig the servo's force limit decides
how hard it presses. Start here and measure.

## THE EXECUTION: the sim executor's ramps, with the servos' own waits

Per leg the joint command ramps linearly from the last command to the target
over the leg's step budget at 30 Hz (the recordings' rate), as the sim
executor's default demonstrator does. The real servos lag their goals where
the sim's stiff actuators did not, so three things differ, all printed:
1. after each moving leg the arm is given up to `--leg-settle` steps to settle
   (every joint slower than `--settled-vel`, the servo's Present_Velocity);
2. the close is armed on the last descent leg: it fires once the FK
   fingertip is within `--tip-close-mm` of the plan's `tip_goal` and the arm
   has settled, and past the ramp the arm is given `--settle-steps` more steps
   to get there — then it closes anyway, and the report says which;
3. the close holds the arm and waits for the jaw to stop (it stalls on the
   brick), at least `CLOSE_MIN` steps: the sim's close is one step because
   its jaw is stiff.
The sim executor's sag integral is OFF unless `--sag-ki` is given (measure
the servo's own sag first, from the trace). No contact is read anywhere.

After the retreat the arm ramps back to where the episode started (the
recorder's `--return-rest`), out of the camera's way, and the outcome is read.

## THE OUTCOME: the camera's brick-in-bowl

Over ~1 s of frames with the arm home: the brick found (any coverage — the
bowl's wall hides part of it) within `--in-bowl-mm` of the bowl on at least
half the frames (60 mm horizontal, not the task's 45: see `IN_BOWL_MM`). Otherwise the bucket: MISSED (the
brick within 15 mm of where it started), MOVED (elsewhere on the desk), or
LOST (not seen).

## `--dataset NAME`: the episodes as a LeRobot dataset (with `--arm`)

The expert as a DATA ENGINE: every episode is recorded as `record_ui.mojo`
records a teleoperated one — `projects/<--project, so101-tower>/datasets/NAME`,
checkpointed after each episode (`--resume` continues one), both cameras
(`--camera` -> `observation.images.overhead`, `--wrist-camera` ->
`observation.images.wrist`, 640x480 at 30 Hz, the overhead camera pacing the
tick), `observation.state` the follower's measured joints and `action` the
commanded ones, in LeRobot units (degrees, the gripper 0..100, through the
calibration — no follower zero), and `cube-in-bowl-printed`'s task string
(`TASK_LANGUAGE`, `--task`). An episode runs from the first leg to the arm folded
and still at home (`N_REST_HOLD`). The camera's verdict keeps a SUCCESS and
rejects anything else (`meta/rejected_episodes.json`, skipped on import);
the operator can flip it at the prompt. ⚠ The gripper's action when closed
is 0 (the expert commands fully closed; an operator's leader rarely is).

    pixi run -e jetson mojo run -I . examples/so101/tower_expert_real.mojo --arm \\
        --camera /dev/soarm_cam_overhead --wrist-camera /dev/soarm_cam_wrist \\
        --dataset cube-in-bowl-expert --episodes 50

## `--plan-only CSV`: the planning half alone

No camera and no arm: every layout of a start-poses file
(`tower_pose_real_check.mojo` writes one per recorded dataset) is planned from
the sim's rest pose, and the plans that would need `force` are counted.

## Output (`--out`, default `projects/so101-tower/rig_runs/<stamp>/`)

`episodes.tsv` (one row per episode: estimates, plan, close, outcome),
`epN.tsv` (every control tick: leg, commanded and measured joints in model
rad, joint velocities, Present_Load in % of max torque, the FK fingertip and
its distance to `tip_goal`), and
`epN_start.png` / `epN_end.png` (the overhead frames the poses and the
outcome were read from).
"""

from std.builtin.sort import sort
from std.math import atan2, cos, sin, sqrt, pi
from std.os import makedirs
from std.random import seed as seed_rng
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.core.concurrent.thread import sleep_us
from noeira.core.cont_action import ContAction
from noeira.core.project import project_dataset_dir
from noeira.core.run import epoch_seconds, iso8601_utc
from noeira.data.lerobot_rejected import reject_episode
from noeira.data.lerobot_write import LeRobotWriter, open_recording
from noeira.io.fileio import StdinReader, stdin_is_tty
from noeira.io.png import save_png
from noeira.physics3d.gpu.constants import MODEL_CURRICULUM_SIZE
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.robot.so101 import SO101Arm, SO101_N, joint_name
from noeira.robot.so101.deploy_shutdown import return_and_release
from noeira.robot.so101.ports import follower_port, port_refusal
from noeira.robot.so101.sim_map import SimJointMap
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_expert_plan import (
    TowerExpertEnv, TowerGraspPlanner, TowerGraspPlan, PlanLeg, fingertip_point,
    ACT, N_ARM, NQ, HUMAN_JAW_OPEN, HUMAN_Z_GRASP, CLEAR_PLAN_TILT,
    PLAN_PEN_OK_MM,
)
from noeira.tasks.so101_tower_overhead import (
    OVERHEAD_CALIB, tower_overhead_pose, tower_desk_roi, printed_brick_hsv,
    printed_bowl_hsv, pose_confident,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import fixed
from noeira.vision.calib_file import read_calib
from noeira.vision.camera_thread import CameraReader
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.opencv import opencv_shim_available
from noeira.vision.tabletop_pose import (
    RigCamera, PrismModel, PoseEstimate, estimate_prism_pose,
)

comptime E = TowerExpertEnv
comptime CFG = So101TowerTeleopConfig
comptime NV = So101TowerModel.NV
comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime TASK = "so101_tower_cube_in_bowl"
comptime OUT_ROOT = "projects/so101-tower/rig_runs"

comptime HZ = 30
comptime PERIOD_NS = 33_333_333
"""30 Hz, the recorded datasets' rate (`record_ui.mojo`); recording, the
overhead camera IS the clock. The plans' step budgets are at the sim's 31.25
Hz, so each leg runs 4 % slower than in sim (the first rig run, 10/10, was
at 31.25 before recording existed)."""
comptime N_REST_HOLD = 15
"""Recording: frames held at rest after the ramp home (the sim recorder's
`--return-rest` hold) — the episode ends folded and still."""
comptime TASK_LANGUAGE = "Grab the blue cube and put it in the yellow octogonal bowl"
"""`cube-in-bowl-printed`'s task string, byte for byte, misspelling included
(its `meta/tasks.parquet`; SmolVLA's token table for it, f61c6e2b1, is 15
tokens): episodes merged with that set, or fine-tuned next to it, must carry
the same instruction. NOT the sim task's `language=` ("Grab the cube and put
it in the bowl"), which is the older `cube-in-bowl` set's. `--task` overrides."""
comptime DEFAULT_PROJECT = "so101-tower"
comptime MAX_STEP_TICKS = 80
comptime TRACK_STEP_TICKS = 512
"""The two-phase clamp on the goal's lead over the present position
(`SO101Arm.track_step_ticks`): 80 ticks while the follower catches up after
torque on (`--step`), 512 once it tracks — the recorders' and the SmolVLA
deployment's, so the arm moves like the arm that recorded the demos. ⚠ The
clamp also bounds the GRIPPER's squeeze: stalled on the brick, its goal sits
at most this far past its position, and a single small clamp (60 ticks, ~5
deg) would have squeezed with a twentieth of the teleop's force."""
comptime TIP_CLOSE_MM: Float64 = 8.0
"""The sim executor's `TIP_CLOSE_MM_DEFAULT` (`--tip-close-mm`)."""
comptime SETTLED_VEL: Float64 = 0.15
"""Rad/s, every joint incl. the jaw (the sim executor's; `--settled-vel`).
The servo's Present_Velocity is coarse: retune from the trace."""
comptime SETTLE_STEPS = 45
"""Past the last descent ramp, steps to wait for the close trigger before
closing anyway (`--settle-steps`; 1.4 s)."""
comptime LEG_SETTLE = 20
"""Past every other moving leg's ramp, steps to wait for the arm to settle
(`--leg-settle`; 0.64 s)."""
comptime CLOSE_MIN = 10
comptime CLOSE_MAX = 40
"""The close: at least / at most this many steps, done once the jaw has been
slower than `settled_vel` for 3 steps."""
comptime N_HOME = 60
"""The ramp back to the episode's start pose (the recorder's
`N_RETURN_REST`)."""
comptime SAG_MAX: Float64 = 0.35
"""The sag integral's clamp (rad), the sim executor's."""
comptime DESK_CLEAR_MM: Float64 = -3.0
comptime IN_BOWL_MM: Float64 = 60.0
"""Brick centre to bowl centre, horizontal, for SUCCESS. Not the task's 45 mm
(3D, body origins): in the bowl the camera sees only the brick's upper half
over the wall, and the fit reads it 29-48 mm from the bowl's centre (first rig
run, 2026-09-25, 10 bricks in the bowl, single frames) — at 45 two of ten
real successes were scored MOVED. A brick BESIDE the bowl is at least 68 mm
away (the octagon's apothem 56 + half the brick); 60 is between."""
comptime MISSED_MM: Float64 = 15.0
comptime IK_OK_MM: Float64 = 10.0
"""A plan whose IK misses a leg's target by more needs `force` to run."""
comptime STILL_MM: Float64 = 1.5
comptime SCENE_TIMEOUT_S: Float64 = 120.0
comptime MIN_SEP_M: Float64 = 0.088
"""The brick's centre at least this far from the bowl's (the sim check's
redraw rule): nearer, it is in or against the bowl."""
comptime DROP_ABORT = 8
"""Consecutive ticks with a partial bus read before the episode aborts."""


def _deg(r: Float64) -> String:
    return fixed(r * 180.0 / pi, 1)


def _circ_mean(ys: List[Float64], period: Float64) -> Float64:
    """The mean of angles defined modulo `period`, in [0, period)."""
    var sc = 0.0
    var ss = 0.0
    var w = 2.0 * pi / period
    for y in ys:
        sc += cos(w * y)
        ss += sin(w * y)
    var m = atan2(ss, sc) / w
    if m < 0.0:
        m += period
    return m


@fieldwise_init
struct Scene(Copyable, Movable):
    """The brick and the bowl, still, from the overhead camera (world m)."""

    var brick_x: Float64
    var brick_y: Float64
    var brick_yaw: Float64
    var bowl_x: Float64
    var bowl_y: Float64
    var bowl_yaw: Float64


struct Rig(Movable):
    """The follower, its map to the model's joints, and the control tick."""

    var arm: SO101Arm
    var jmap: SimJointMap
    var armed: Bool
    var raw: List[Int32]
    """The last full position read, ticks."""
    var q_cmd: List[Float64]
    """The command, model rad, 6 (the jaw last)."""
    var q_ref: List[Float64]
    var sag_bias: List[Float64]
    var sag_ki: Float64
    var q: List[Float64]
    """Measured, model rad (unclamped), 6."""
    var v: List[Float64]
    """Measured, rad/s, 6."""
    var load: List[Float64]
    """Present_Load, % of the servo's maximum torque, signed, 6."""
    var load_peak: List[Float64]
    """This leg's largest |load| per joint."""
    var tip: List[Float64]
    var tip_goal: List[Float64]
    var jaw_open: Float64
    var jaw_lo: Float64
    var settled_vel: Float64
    var tip_close_mm: Float64
    var t_next: Int
    var tick_n: Int
    var drops: Int
    var drops_run: Int
    var late: Int
    var leg: String
    var trace: String
    var cams: List[CameraReader]
    """The overhead camera, then (recording) the wrist camera."""
    var writers: List[LeRobotWriter]
    """Recording: the dataset writer (one), else empty."""
    var frames: List[List[UInt8]]
    var recording_now: Bool
    """Inside a recorded episode: the cameras pace the tick and every tick is
    a dataset row."""
    var rec_rows: Int

    def __init__(out self, var arm: SO101Arm, var jmap: SimJointMap):
        self.arm = arm^
        self.jmap = jmap^
        self.armed = False
        self.raw = List[Int32](length=SO101_N, fill=Int32(0))
        self.q_cmd = List[Float64](length=ACT, fill=0.0)
        self.q_ref = List[Float64](length=N_ARM, fill=0.0)
        self.sag_bias = List[Float64](length=N_ARM, fill=0.0)
        self.sag_ki = 0.0
        self.q = List[Float64](length=ACT, fill=0.0)
        self.v = List[Float64](length=ACT, fill=0.0)
        self.load = List[Float64](length=ACT, fill=0.0)
        self.load_peak = List[Float64](length=ACT, fill=0.0)
        self.tip = List[Float64](length=3, fill=0.0)
        self.tip_goal = List[Float64](length=3, fill=0.0)
        self.jaw_open = HUMAN_JAW_OPEN
        self.jaw_lo = 0.0
        self.settled_vel = SETTLED_VEL
        self.tip_close_mm = TIP_CLOSE_MM
        self.t_next = 0
        self.tick_n = 0
        self.drops = 0
        self.drops_run = 0
        self.late = 0
        self.leg = String("")
        self.trace = String("")
        self.cams = List[CameraReader]()
        self.writers = List[LeRobotWriter]()
        self.frames = List[List[UInt8]]()
        self.recording_now = False
        self.rec_rows = 0

    def read_joints(mut self) raises -> Bool:
        """Positions only (torque off or on); False on a partial read."""
        var raw = List[Int32](length=SO101_N, fill=Int32(0))
        if self.arm.read_positions(Span(raw)) != SO101_N:
            return False
        for i in range(SO101_N):
            self.raw[i] = raw[i]
            self.q[i] = self.jmap.to_sim_unclamped(self.arm.cal, i, raw[i])
        return True

    def arm_torque(mut self, step_ticks: Int) raises:
        """Position mode, the goal parked on the present pose, torque on."""
        if not self.read_joints():
            raise Error("the follower did not report 6 positions — not arming")
        self.arm.set_position_mode()
        var park = self.raw.copy()
        self.arm.max_step_ticks = 0
        self.arm.write_goals(Span(park))
        self.arm.max_step_ticks = step_ticks
        self.arm.set_torque(True)
        self.armed = True

    def begin(mut self, mut env: E) raises:
        """Start a command stream from the measured pose."""
        var ok = False
        for _ in range(5):
            if self.sense(env):
                ok = True
                break
        if not ok:
            raise Error("the follower did not report its pose")
        for i in range(ACT):
            self.q_cmd[i] = self.q[i]
        for i in range(N_ARM):
            self.q_ref[i] = self.q[i]
            self.sag_bias[i] = 0.0
        self.t_next = Int(perf_counter_ns()) + PERIOD_NS
        self.drops_run = 0

    def sense(mut self, mut env: E) raises -> Bool:
        """Positions, velocities, and the FK fingertip (on the planner's env)."""
        var raw = List[Int32](length=SO101_N, fill=Int32(0))
        var vraw = List[Int32](length=SO101_N, fill=Int32(0))
        var lraw = List[Int32](length=SO101_N, fill=Int32(0))
        if self.arm.read_positions(Span(raw)) != SO101_N:
            return False
        if self.arm.read_velocities(Span(vraw)) != SO101_N:
            return False
        if self.arm.read_loads(Span(lraw)) != SO101_N:
            return False
        for i in range(SO101_N):
            self.raw[i] = raw[i]
            self.q[i] = self.jmap.to_sim_unclamped(self.arm.cal, i, raw[i])
            self.v[i] = Float64(vraw[i]) * 2.0 * pi / 4096.0
            self.load[i] = Float64(lraw[i]) / 10.0
            self.load_peak[i] = max(self.load_peak[i], abs(self.load[i]))
        for i in range(ACT):
            env.d.qpos.data[i] = self.q[i]
        env._fields_fk()
        var t = fingertip_point(env)
        for k in range(3):
            self.tip[k] = t[k]
        return True

    def tick(mut self, mut env: E, mut stdin: StdinReader) raises:
        """Wait for the period (recording: for the cameras' next frames),
        measure, write the command; one trace row and, recording, one dataset
        row (the frames, the state just measured, the command just written —
        `record.mojo`'s order)."""
        if stdin.has_input():
            raise Error("ABORT: stopped by the operator")
        if self.recording_now:
            for c in range(len(self.cams)):
                var got = self.cams[c].take_blocking(self.frames[c])
                if not got:
                    raise Error("a camera stopped delivering frames")
        else:
            var now = Int(perf_counter_ns())
            if now > self.t_next:
                self.late += 1
                self.t_next = now
            while Int(perf_counter_ns()) < self.t_next:
                pass
            self.t_next += PERIOD_NS
        self.tick_n += 1
        if self.sense(env):
            self.drops_run = 0
        else:
            self.drops += 1
            self.drops_run += 1
            if self.drops_run >= DROP_ABORT:
                raise Error("ABORT: " + String(DROP_ABORT) + " partial bus reads in a row")
        var goals = List[Int32](length=SO101_N, fill=Int32(0))
        for i in range(SO101_N):
            goals[i] = self.jmap.from_sim(self.arm.cal, i, self.q_cmd[i])
        self.arm.write_goals(Span(goals))
        if self.recording_now:
            # LeRobot units, as `record.mojo` writes them: the follower's
            # measured ticks and the commanded ticks, through the calibration
            # (degrees; the gripper 0..100) — no follower zero
            var state = List[Float64]()
            var action = List[Float64]()
            for i in range(SO101_N):
                state.append(self.arm.cal.degrees(i, self.raw[i]))
                action.append(self.arm.cal.degrees(i, goals[i]))
            self.writers[0].add_frame(state, action, self.frames)
            self.rec_rows += 1
        var row = String(self.tick_n) + "\t" + self.leg
        for i in range(ACT):
            row += "\t" + fixed(self.q_cmd[i], 4)
        for i in range(ACT):
            row += "\t" + fixed(self.q[i], 4)
        for i in range(ACT):
            row += "\t" + fixed(self.v[i], 3)
        for i in range(ACT):
            row += "\t" + fixed(self.load[i], 1)
        for k in range(3):
            row += "\t" + fixed(self.tip[k], 4)
        self.trace += row + "\t" + fixed(self.tip_dist_mm(), 1) + "\n"

    def sag_update(mut self):
        if self.sag_ki <= 0.0:
            return
        for i in range(N_ARM):
            var b = self.sag_bias[i] + self.sag_ki * (self.q_ref[i] - self.q[i])
            self.sag_bias[i] = max(-SAG_MAX, min(SAG_MAX, b))

    def set_arm(mut self, ref target: List[Float64]):
        for i in range(N_ARM):
            self.q_ref[i] = target[i]
            self.q_cmd[i] = target[i] + self.sag_bias[i]

    def settled(self) -> Bool:
        for i in range(ACT):
            if abs(self.v[i]) > self.settled_vel:
                return False
        return True

    def tip_dist_mm(self) -> Float64:
        var d = 0.0
        for k in range(3):
            d += (self.tip[k] - self.tip_goal[k]) ** 2
        return sqrt(d) * 1000.0

    def arm_err_deg(self, ref target: List[Float64]) -> Float64:
        var w = 0.0
        for i in range(N_ARM):
            w = max(w, abs(self.q[i] - target[i]))
        return w * 180.0 / pi

    def run_leg(
        mut self, mut env: E, leg: PlanLeg, settle_steps: Int, leg_settle: Int,
        mut stdin: StdinReader,
    ) raises -> String:
        """One plan leg; returns its report line."""
        self.leg = leg.name
        for i in range(ACT):
            self.load_peak[i] = 0.0
        var g_target = self.jaw_open if leg.grip_open else self.jaw_lo
        var target: List[Float64]
        if len(leg.q) > 0:
            target = leg.q.copy()
        else:
            # a HOLD keeps the arm's reference (the sim executor's `hold`)
            target = self.q_ref.copy()
        var q0 = self.q_cmd.copy()
        var r0 = self.q_ref.copy()
        if leg.name == "close":
            self.q_cmd[5] = g_target
            var slow = 0
            var n = 0
            for k in range(CLOSE_MAX):
                self.tick(env, stdin)
                self.sag_update()
                self.set_arm(target)
                n = k + 1
                slow = slow + 1 if abs(self.v[5]) < self.settled_vel else 0
                if n >= CLOSE_MIN and slow >= 3:
                    break
            return (
                "close   " + String(n) + " steps | jaw " + fixed(self.q[5], 3)
                + " rad (commanded " + fixed(g_target, 3) + ") | tip "
                + fixed(self.tip_dist_mm(), 1) + " mm from tip_goal" + self.load_report()
            )
        for k in range(leg.steps):
            var a = Float64(k + 1) / Float64(leg.steps)
            for i in range(N_ARM):
                self.q_ref[i] = r0[i] + (target[i] - r0[i]) * a
                self.q_cmd[i] = self.q_ref[i] + self.sag_bias[i]
            if leg.close_on_tip:
                self.q_cmd[5] = g_target
            else:
                self.q_cmd[5] = q0[5] + (g_target - q0[5]) * a
            self.tick(env, stdin)
            self.sag_update()
            if (
                leg.close_on_tip and k + 1 >= 3
                and self.tip_dist_mm() < self.tip_close_mm and self.settled()
            ):
                return (
                    pad8(leg.name) + String(k + 1) + " steps | CLOSE TRIGGERED on the ramp: tip "
                    + fixed(self.tip_dist_mm(), 1) + " mm | arm err " + fixed(self.arm_err_deg(target), 1) + " deg"
                    + self.load_report()
                )
        var extra = settle_steps if leg.close_on_tip else leg_settle
        var waited = 0
        var ok = False
        for _ in range(extra):
            if leg.close_on_tip:
                if self.tip_dist_mm() < self.tip_close_mm and self.settled():
                    ok = True
                    break
            elif self.settled():
                ok = True
                break
            self.set_arm(target)
            self.tick(env, stdin)
            self.sag_update()
            waited += 1
        var s = pad8(leg.name) + String(leg.steps) + "+" + String(waited) + " steps | "
        if leg.close_on_tip:
            s += (
                "CLOSE TRIGGERED after the ramp" if ok else "⚠ NO TRIGGER, closing anyway"
            ) + ": tip " + fixed(self.tip_dist_mm(), 1) + " mm"
        else:
            s += ("settled" if ok else "⚠ not settled")
        s += " | arm err " + fixed(self.arm_err_deg(target), 1) + " deg | jaw " + fixed(self.q[5], 3)
        return s + self.load_report()

    def load_report(self) -> String:
        """This leg's peak |load| per joint and the load now, % of max torque."""
        var s = String(" | load % peak/now")
        for i in range(ACT):
            s += " " + fixed(self.load_peak[i], 0) + "/" + fixed(self.load[i], 0)
        return s

    def ramp_to(
        mut self, mut env: E, ref target6: List[Float64], steps: Int, var name: String,
        mut stdin: StdinReader,
    ) raises:
        """All six joints to `target6` (the jaw included), sag bias dropped."""
        self.leg = name^
        var q0 = self.q_cmd.copy()
        for i in range(N_ARM):
            self.sag_bias[i] = 0.0
        for k in range(steps):
            var a = Float64(k + 1) / Float64(steps)
            for i in range(ACT):
                self.q_cmd[i] = q0[i] + (target6[i] - q0[i]) * a
            for i in range(N_ARM):
                self.q_ref[i] = self.q_cmd[i]
            self.tick(env, stdin)
        for _ in range(LEG_SETTLE):
            if self.settled():
                break
            self.tick(env, stdin)


def pad8(s: String) -> String:
    var out = s
    while out.byte_length() < 8:
        out += " "
    return out


def read_scene(
    mut reader: CameraReader, cam: RigCamera, mut frame: List[UInt8],
    mut stdin: StdinReader,
) raises -> Optional[Scene]:
    """Brick and bowl both confident and still for a second. None on 'q' or
    the timeout."""
    var roi = tower_desk_roi()
    var cb = printed_brick_hsv()
    var co = printed_bowl_hsv()
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()
    var wt = List[Float64]()
    var bx = List[Float64]()
    var by = List[Float64]()
    var byaw = List[Float64]()
    var ox = List[Float64]()
    var oy = List[Float64]()
    var oyaw = List[Float64]()
    var t0 = perf_counter_ns()
    var last_print = -10.0
    while True:
        var now = Float64(perf_counter_ns() - t0) * 1e-9
        if now > SCENE_TIMEOUT_S:
            print("  no still, confident brick + bowl in", Int(SCENE_TIMEOUT_S), "s")
            return None
        if stdin.has_input():
            var l = stdin.line()
            if l == "q":
                return None
        if reader.take_latest(frame) == 0:
            _ = sleep_us(2000)
            continue
        var eb = estimate_prism_pose(frame, cam, cb, brick, roi)
        var eo = estimate_prism_pose(frame, cam, co, bowl, roi)
        if not (pose_confident(eb) and pose_confident(eo)):
            if now - last_print > 3.0:
                last_print = now
                print(
                    "  waiting: brick", "ok" if pose_confident(eb) else (
                        "seen cov " + fixed(eb.coverage, 2) + " res " + fixed(eb.residual, 2)
                        if eb.found else "not seen"
                    ), "| bowl", "ok" if pose_confident(eo) else (
                        "seen cov " + fixed(eo.coverage, 2) + " res " + fixed(eo.residual, 2)
                        if eo.found else "not seen"
                    ),
                )
            wt.clear()
            bx.clear()
            by.clear()
            byaw.clear()
            ox.clear()
            oy.clear()
            oyaw.clear()
            continue
        wt.append(now)
        bx.append(eb.x)
        by.append(eb.y)
        byaw.append(eb.yaw)
        ox.append(eo.x)
        oy.append(eo.y)
        oyaw.append(eo.yaw)
        while len(wt) > 0 and wt[0] < now - 1.0:
            _ = wt.pop(0)
            _ = bx.pop(0)
            _ = by.pop(0)
            _ = byaw.pop(0)
            _ = ox.pop(0)
            _ = oy.pop(0)
            _ = oyaw.pop(0)
        var n = len(wt)
        if n < 10 or now - wt[0] < 0.8:
            continue
        var mbx = 0.0
        var mby = 0.0
        var mox = 0.0
        var moy = 0.0
        for k in range(n):
            mbx += bx[k] / Float64(n)
            mby += by[k] / Float64(n)
            mox += ox[k] / Float64(n)
            moy += oy[k] / Float64(n)
        var spread = 0.0
        for k in range(n):
            spread = max(spread, sqrt((bx[k] - mbx) ** 2 + (by[k] - mby) ** 2))
            spread = max(spread, sqrt((ox[k] - mox) ** 2 + (oy[k] - moy) ** 2))
        if spread * 1000.0 > STILL_MM:
            continue
        return Scene(
            mbx, mby, _circ_mean(byaw, brick.period), mox, moy,
            _circ_mean(oyaw, bowl.period),
        )


def read_outcome(
    mut reader: CameraReader, cam: RigCamera, mut frame: List[UInt8],
    sc: Scene, in_bowl_mm: Float64,
) raises -> Tuple[String, Float64, Float64, Float64, Float64]:
    """(bucket, brick x, brick y, median brick-to-bowl mm, bowl moved mm)
    over ~1 s of frames; SUCCESS when at least half the frames see the brick
    within `in_bowl_mm`. The brick is taken at ANY coverage: in the bowl its wall hides
    part of it."""
    var roi = tower_desk_roi()
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()
    var n_seen = 0
    var n_in = 0
    var sx = 0.0
    var sy = 0.0
    var ds = List[Float64]()
    var sbm = 0.0
    var n_frames = 0
    var t0 = perf_counter_ns()
    while n_frames < 25 and Float64(perf_counter_ns() - t0) * 1e-9 < 3.0:
        if reader.take_latest(frame) == 0:
            _ = sleep_us(2000)
            continue
        n_frames += 1
        var eb = estimate_prism_pose(frame, cam, printed_brick_hsv(), brick, roi)
        var eo = estimate_prism_pose(frame, cam, printed_bowl_hsv(), bowl, roi)
        var wx = sc.bowl_x
        var wy = sc.bowl_y
        if pose_confident(eo):
            wx = eo.x
            wy = eo.y
            sbm += sqrt((eo.x - sc.bowl_x) ** 2 + (eo.y - sc.bowl_y) ** 2) * 1000.0
        if not eb.found:
            continue
        n_seen += 1
        var d = sqrt((eb.x - wx) ** 2 + (eb.y - wy) ** 2) * 1000.0
        sx += eb.x
        sy += eb.y
        ds.append(d)
        if d < in_bowl_mm:
            n_in += 1
    if n_seen * 2 < max(n_frames, 1):
        return (String("LOST"), 0.0, 0.0, -1.0, sbm / Float64(max(n_frames, 1)))
    var ex = sx / Float64(n_seen)
    var ey = sy / Float64(n_seen)
    # the MEDIAN distance: a half-hidden brick's fit jumps on some frames
    # (the first run's means were 12 mm above its last frames)
    sort(ds)
    var d = ds[len(ds) // 2]
    var bm = sbm / Float64(max(n_frames, 1))
    if n_in * 2 >= n_seen:
        return (String("SUCCESS"), ex, ey, d, bm)
    var moved = sqrt((ex - sc.brick_x) ** 2 + (ey - sc.brick_y) ** 2) * 1000.0
    if moved < MISSED_MM:
        return (String("MISSED"), ex, ey, d, bm)
    return (String("MOVED"), ex, ey, d, bm)


def scene_qpos(
    ref q_scene: List[Float64], ref arm_q: List[Float64], sc: Scene,
    brick_adr: Int, bowl_adr: Int, ref lo: List[Float64], ref hi: List[Float64],
) -> List[Float64]:
    """The sim scene in the rig's state: the arm's joints (clamped to the
    model's limits) and the props at the estimates, at their resting heights."""
    var qs = q_scene.copy()
    for k in range(ACT):
        qs[k] = max(lo[k], min(hi[k], arm_q[k]))
    qs[brick_adr] = sc.brick_x
    qs[brick_adr + 1] = sc.brick_y
    qs[brick_adr + 3] = cos(sc.brick_yaw / 2.0)
    qs[brick_adr + 4] = 0.0
    qs[brick_adr + 5] = 0.0
    qs[brick_adr + 6] = sin(sc.brick_yaw / 2.0)
    qs[bowl_adr] = sc.bowl_x
    qs[bowl_adr + 1] = sc.bowl_y
    qs[bowl_adr + 3] = cos(sc.bowl_yaw / 2.0)
    qs[bowl_adr + 4] = 0.0
    qs[bowl_adr + 5] = 0.0
    qs[bowl_adr + 6] = sin(sc.bowl_yaw / 2.0)
    return qs^


def plan_warnings(plan: TowerGraspPlan) -> String:
    """Why a plan should not run without `force` (empty: none)."""
    var w = String("")
    var worst_ik = max(max(plan.e_grasp, plan.e_lift), max(plan.e_carry, plan.e_place)) * 1000.0
    if worst_ik > IK_OK_MM:
        w += "  ⚠ the IK misses a leg by " + fixed(worst_ik, 1) + " mm\n"
    if plan.pen_mm > PLAN_PEN_OK_MM:
        w += "  ⚠ the plan passes " + fixed(plan.pen_mm, 1) + " mm into an obstacle\n"
    if not plan.close_on_tip:
        w += "  ⚠ not a fingertip-triggered plan\n"
    return w^


def print_plan(plan: TowerGraspPlan, legs: List[PlanLeg]):
    print(
        "  PLAN tilt", _deg(plan.tilt), "deg | pinch", "tangential" if plan.tangential else "radial",
        fixed((plan.yaw - plan.bearing) * 180.0 / pi, 1), "deg from radial | tip_goal",
        fixed(plan.tip_goal[0], 3), fixed(plan.tip_goal[1], 3), fixed(plan.tip_goal[2], 3),
        "| raised", fixed(plan.raise_mm, 1), "mm | penetration", fixed(plan.pen_mm, 1),
        "mm |", plan.tries, "tries",
    )
    print(
        "       IK err mm: grasp", fixed(plan.e_grasp * 1000.0, 1), "lift",
        fixed(plan.e_lift * 1000.0, 1), "carry", fixed(plan.e_carry * 1000.0, 1),
        "place", fixed(plan.e_place * 1000.0, 1),
    )
    for leg in legs:
        var qs = String("")
        for i in range(len(leg.q)):
            qs += " " + _deg(leg.q[i])
        print(
            "    " + pad8(leg.name), leg.steps, "open  " if leg.grip_open else "closed",
            "close_on_tip" if leg.close_on_tip else "", "q deg" + (qs if len(leg.q) > 0 else " (hold)"),
        )


def plan_layouts(
    path: String, mut env: E, mut planner: TowerGraspPlanner,
    ref q_scene: List[Float64], brick_adr: Int, bowl_adr: Int, seed0: Int,
    jaw_open: Float64,
) raises:
    """`--plan-only CSV`: plan every layout of a start-poses file
    (`tower_pose_real_check.mojo`'s: episode, brick x/y, bowl x/y in world mm)
    from the sim's rest pose, brick yaw 0 (the file has none). No camera, no
    arm: the planning half on the real layouts, before a desk session."""
    var lines = open(path, "r").read().split("\n")
    var v0 = List[Float64](length=NV, fill=0.0)
    var arm_q = List[Float64]()
    for k in range(ACT):
        arm_q.append(q_scene[k])
    var n = 0
    var n_clean = 0
    var n_near = 0
    var n_near_clean = 0
    var n_skip = 0
    print("  ep  brick mm        bowl mm         r_mm  tilt tries pen_mm  ik_mm  ok")
    for li in range(1, len(lines)):
        var cols = lines[li].split(",")
        if len(cols) < 5:
            continue
        if String(cols[1]) == "" or String(cols[3]) == "":
            n_skip += 1
            continue
        var sc = Scene(
            Float64(String(cols[1])) / 1000.0, Float64(String(cols[2])) / 1000.0, 0.0,
            Float64(String(cols[3])) / 1000.0, Float64(String(cols[4])) / 1000.0, 0.0,
        )
        if sqrt((sc.brick_x - sc.bowl_x) ** 2 + (sc.brick_y - sc.bowl_y) ** 2) < MIN_SEP_M:
            n_skip += 1
            continue
        var qs = scene_qpos(q_scene, arm_q, sc, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
        var pb: List[Float64] = [sc.brick_x, sc.brick_y, q_scene[brick_adr + 2]]
        var pw: List[Float64] = [sc.bowl_x, sc.bowl_y, q_scene[bowl_adr + 2]]
        var q5 = List[Float64]()
        for k in range(N_ARM):
            q5.append(qs[k])
        env.set_state(qs, v0)
        seed_rng(seed0 + n)
        var plan = planner.plan_grasp(env, pb, sc.brick_yaw, q5, jaw_open)
        planner.plan_place(env, plan, pw)
        var ok = plan_warnings(plan).byte_length() == 0
        var r = sqrt(sc.brick_x ** 2 + sc.brick_y ** 2)
        n += 1
        if ok:
            n_clean += 1
        if r < 0.25:
            n_near += 1
            if ok:
                n_near_clean += 1
        var worst_ik = max(max(plan.e_grasp, plan.e_lift), max(plan.e_carry, plan.e_place)) * 1000.0
        print(
            "  " + pad8(String(cols[0])), fixed(sc.brick_x * 1000.0, 0), fixed(sc.brick_y * 1000.0, 0),
            "  ", fixed(sc.bowl_x * 1000.0, 0), fixed(sc.bowl_y * 1000.0, 0), "  ",
            fixed(r * 1000.0, 0), " ", _deg(plan.tilt), " ", plan.tries, " ",
            fixed(plan.pen_mm, 1), " ", fixed(worst_ik, 1), " ", "ok" if ok else "FORCE",
        )
    print(
        "  layouts planned", n, "| clean (no force needed)", n_clean, "| bricks nearer than 0.25 m:",
        n_near, "(clean", String(n_near_clean) + ")", "| skipped", n_skip,
    )


def _usage():
    print(
        "usage: tower_expert_real.mojo --camera DEV [--arm] [--episodes N] [--seed S]\n"
        "       [--desk-clear-mm MM] [--tilt-range LO,HI] [--jaw-open RAD] [--tip-close-mm MM]\n"
        "       [--settled-vel RAD_S] [--settle-steps N] [--leg-settle N] [--sag-ki K]\n"
        "       [--step TICKS] [--in-bowl-mm MM] [--calib FILE] [--port DEV] [--out DIR]\n"
        "       [--dataset NAME --wrist-camera DEV [--project P] [--task STR] [--resume]]\n"
        "       tower_expert_real.mojo --plan-only START_POSES.csv [--seed S] [--tilt-range ..]"
    )


def main() raises:
    var args = argv()
    var camera = String("")
    var calib_path = String(OVERHEAD_CALIB)
    var port = String("")
    var live = False
    var n_episodes = 10
    var seed0 = 0
    var desk_clear_mm = DESK_CLEAR_MM
    var tilt_range = String(CLEAR_PLAN_TILT)
    var jaw_open = HUMAN_JAW_OPEN
    var tip_close_mm = TIP_CLOSE_MM
    var settled_vel = SETTLED_VEL
    var settle_steps = SETTLE_STEPS
    var leg_settle = LEG_SETTLE
    var sag_ki = 0.0
    var step_ticks = MAX_STEP_TICKS
    var in_bowl_mm = IN_BOWL_MM
    var out_dir = String("")
    var plan_only = String("")
    var project = String(DEFAULT_PROJECT)
    var dataset = String("")
    var wrist = String("")
    var task = String(TASK_LANGUAGE)
    var resume = False
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--arm":
            live = True
            i += 1
            continue
        if a == "--resume":
            resume = True
            i += 1
            continue
        if a == "--help" or a == "-h":
            _usage()
            return
        if i + 1 >= len(args):
            _usage()
            raise Error("flag " + a + " needs a value")
        var v = String(args[i + 1])
        if a == "--camera":
            camera = v
        elif a == "--calib":
            calib_path = v
        elif a == "--port":
            port = v
        elif a == "--episodes":
            n_episodes = Int(v)
        elif a == "--seed":
            seed0 = Int(v)
        elif a == "--desk-clear-mm":
            desk_clear_mm = Float64(v)
        elif a == "--tilt-range":
            tilt_range = v
        elif a == "--jaw-open":
            jaw_open = Float64(v)
        elif a == "--tip-close-mm":
            tip_close_mm = Float64(v)
        elif a == "--settled-vel":
            settled_vel = Float64(v)
        elif a == "--settle-steps":
            settle_steps = Int(v)
        elif a == "--leg-settle":
            leg_settle = Int(v)
        elif a == "--sag-ki":
            sag_ki = Float64(v)
        elif a == "--step":
            step_ticks = Int(v)
        elif a == "--in-bowl-mm":
            in_bowl_mm = Float64(v)
        elif a == "--out":
            out_dir = v
        elif a == "--plan-only":
            plan_only = v
        elif a == "--project":
            project = v
        elif a == "--dataset":
            dataset = v
        elif a == "--wrist-camera":
            wrist = v
        elif a == "--task":
            task = v
        else:
            _usage()
            raise Error("unknown flag " + a)
        i += 2
    if camera == "" and plan_only == "":
        _usage()
        raise Error("--camera <index | /dev/... path> is required")
    var recording = dataset != "" and plan_only == ""
    var dataset_dir = String("")
    if recording:
        if not live:
            raise Error("--dataset records the episodes the ARM runs: it needs --arm")
        if wrist == "":
            raise Error(
                "--dataset needs --wrist-camera (the recordings have both views:"
                " observation.images.overhead, observation.images.wrist)"
            )
        dataset_dir = project_dataset_dir(project, dataset)
    if plan_only == "" and not opencv_shim_available():
        raise Error("the OpenCV shim is not built: pixi run build-opencv")
    var tr = tilt_range.split(",")
    if len(tr) != 2:
        raise Error("--tilt-range needs lo,hi in degrees, got " + tilt_range)
    if out_dir == "":
        out_dir = String(OUT_ROOT) + "/" + iso8601_utc(epoch_seconds()).replace(":", "-")
    if plan_only == "":
        makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    if live:
        print("so101_tower cube-in-bowl — the SIM EXPERT ON THE REAL ARM  [LIVE]")
    else:
        print("so101_tower cube-in-bowl — DRY RUN (reads, plans, prints; nothing moves)")
        print("  pass --arm to run the plans on the arm")
    print("=" * 70)

    # ── the planner's env, in the sim's own resting scene ────────────────
    var ctx = DeviceContext()
    var env = E(ctx)
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3], rheights[0],
    )
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DType.float64](cw[k])
    _ = env.reset()
    var mw = task_meta_words(
        String(TASK), String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    for k in range(len(mw[0])):
        env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
    var body_names = List[String]()
    var brick_body = -1
    var bowl_body = -1
    for b in range(len(fmd.body_names)):
        body_names.append(String(fmd.body_names[b]))
        if body_names[b] == "brick_brick":
            brick_body = b
        if body_names[b] == "bowl_bowl":
            bowl_body = b
    var brick_adr = -1
    var bowl_adr = -1
    var adr = 0
    for j in range(len(fmd.joints)):
        if fmd.joints[j].nq == 7 and fmd.joints[j].body_id == brick_body:
            brick_adr = adr
        if fmd.joints[j].nq == 7 and fmd.joints[j].body_id == bowl_body:
            bowl_adr = adr
        adr += fmd.joints[j].nq
    if brick_adr < 0 or bowl_adr < 0:
        raise Error("the brick's and the bowl's free joints were not found")
    var planner = TowerGraspPlanner(env, body_names)
    planner.posture.human = True
    planner.clear_plan = True
    planner.z_grasp = HUMAN_Z_GRASP
    planner.desk_clear_m = desk_clear_mm / 1000.0
    planner.posture.tilt_lo = Float64(String(tr[0])) * pi / 180.0
    planner.posture.tilt_hi = Float64(String(tr[1])) * pi / 180.0
    # the props' resting heights: the task's reset, settled as the recorder
    # settles it (5 steps, the arm holding)
    var q_scene = posed_qpos[So101TowerPlacement](
        String(TASK), String(FAMILY), So101TowerConfig.SLOT_RADIUS, UInt64(0),
    )
    var v0 = List[Float64](length=NV, fill=0.0)
    env.set_state(q_scene, v0)
    var hold = ContAction[ACT]()
    for k in range(ACT):
        hold.data[k] = 2.0 * (q_scene[k] - planner.arm.lo[k]) / (planner.arm.hi[k] - planner.arm.lo[k]) - 1.0
    for _ in range(5):
        _ = env.step(hold)
    for k in range(NQ):
        q_scene[k] = Float64(env.d.qpos.data[k])
    print(
        "  sim resting heights: brick z", fixed(q_scene[brick_adr + 2], 4), "| bowl z",
        fixed(q_scene[bowl_adr + 2], 4), "(world m)",
    )

    if plan_only != "":
        plan_layouts(
            plan_only, env, planner, q_scene, brick_adr, bowl_adr, seed0, jaw_open,
        )
        return

    # ── the camera ───────────────────────────────────────────────────────
    var lens_cal = read_calib(calib_path)
    lens_cal.require_size(640, 480)
    var lens = FisheyeLens.from_calib(lens_cal)
    var pose = tower_overhead_pose()
    var cam = RigCamera(lens, pose.pos, pose.rot_mj)
    print("  camera pose:", pose.source)
    var cams = List[CameraReader]()
    var specs: List[String] = [camera]
    if recording:
        specs.append(wrist)
    for c in range(len(specs)):
        var reader = CameraReader.from_spec(specs[c], 640, 480, Float64(HZ), rgb=True)
        reader.start()
        if reader.frame_bytes() != 640 * 480 * 3:
            raise Error("camera " + specs[c] + " delivers " + String(reader.frame_bytes()) + " bytes, not 640x480x3")
        var fps = reader.negotiated_fps()
        print("  camera", specs[c], "->", reader.resolved_node(), fixed(fps, 1), "fps")
        if recording and fps > 0.0 and fps < Float64(HZ) - 1.0:
            raise Error("camera " + specs[c] + " negotiated " + fixed(fps, 1) + " fps, below the " + String(HZ) + " the dataset claims")
        cams.append(reader^)
    var frame = List[UInt8](length=640 * 480 * 3, fill=UInt8(0))

    # ── the arm ──────────────────────────────────────────────────────────
    var the_port = follower_port(port)
    var why = port_refusal(the_port, String("follower"))
    if why.byte_length() > 0:
        raise Error("tower_expert_real: " + why)
    print("  opening", the_port)
    var arm = SO101Arm(the_port, max_step_ticks=step_ticks, track_step_ticks=TRACK_STEP_TICKS)
    arm.bus.timeout_ms = 20
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    for k in range(SO101_N):
        lo[k] = planner.arm.lo[k]
        hi[k] = planner.arm.hi[k]
    var jmap = SimJointMap.tower_follower(arm.cal, lo^, hi^)
    print("  " + jmap.describe())
    # the map's inverse, at interior points (`deploy_reach_real.mojo`: a sign
    # or offset error in `from_sim` is a mirrored pose at full slew)
    var worst = 0.0
    for k in range(SO101_N):
        for s in range(3):
            var val = jmap.sim_lo[k] + (0.25 + 0.25 * Float64(s)) * (jmap.sim_hi[k] - jmap.sim_lo[k])
            worst = max(worst, abs(jmap.to_sim(arm.cal, k, jmap.from_sim(arm.cal, k, val)) - val))
    if worst > 0.02:
        raise Error("to_sim/from_sim do not round-trip (worst " + fixed(worst, 4) + " rad) — not arming")
    var rig = Rig(arm^, jmap^)
    for _ in range(len(cams)):
        rig.frames.append(List[UInt8](length=640 * 480 * 3, fill=UInt8(0)))
    rig.cams = cams^
    if recording:
        var names = List[String]()
        for k in range(SO101_N):
            names.append(joint_name(k) + ".pos")
        var cam_names: List[String] = ["observation.images.overhead", "observation.images.wrist"]
        rig.writers.append(open_recording(
            dataset_dir.copy(), HZ, names.copy(), names.copy(), cam_names^, 480, 640, resume,
        ))
        print(
            "  RECORDING ->", dataset_dir, "(" + String(rig.writers[0].n_episodes()),
            "episodes already there) | task:", task,
        )
    rig.jaw_open = jaw_open
    rig.jaw_lo = planner.arm.lo[5]
    rig.settled_vel = settled_vel
    rig.tip_close_mm = tip_close_mm
    rig.sag_ki = sag_ki
    if not rig.read_joints():
        raise Error("the follower did not report 6 positions")
    var start_pose = List[Int32](length=SO101_N, fill=Int32(0))
    for k in range(SO101_N):
        start_pose[k] = rig.raw[k]
    var stdin = StdinReader()
    var interactive = stdin_is_tty()
    print(
        "  plan: human posture, clear_plan, tilt", tilt_range, "deg, desk clear",
        fixed(desk_clear_mm, 1), "mm, jaw open", fixed(jaw_open, 2), "rad | exec: tip close",
        fixed(tip_close_mm, 1), "mm, settled <", fixed(settled_vel, 2), "rad/s, sag ki",
        fixed(sag_ki, 3), "| out", out_dir,
    )

    var summary = String(
        "ep\tseed\tbrick_x\tbrick_y\tbrick_yaw_deg\tbowl_x\tbowl_y\ttilt_deg\ttries\tpen_mm"
        "\tclose\ttip_at_close_mm\tjaw_after_close\toutcome\tend_x\tend_y\tbrick_bowl_mm"
        "\tbowl_moved_mm\tlate_ticks\tdropped\tdataset\n"
    )
    var n_run = 0
    var n_ok = 0
    var n_kept = 0
    var n_rejected = 0
    var buckets = String("")
    var draw = 0
    try:
        var ep = 0
        while ep < n_episodes:
            print("\n[episode", ep + 1, "/", n_episodes, "] brick and bowl on the desk, the arm at rest, hands off")
            var sc_opt = read_scene(rig.cams[0], cam, frame, stdin)
            if not sc_opt:
                print("  quit")
                break
            var sc = sc_opt.value().copy()
            save_png(out_dir + "/ep" + String(ep + 1) + "_start.png", frame, 640, 480, 3)
            var sep = sqrt((sc.brick_x - sc.bowl_x) ** 2 + (sc.brick_y - sc.bowl_y) ** 2)
            print(
                "  brick (", fixed(sc.brick_x * 1000.0, 1), ",", fixed(sc.brick_y * 1000.0, 1),
                ") mm yaw", _deg(sc.brick_yaw), "| bowl (", fixed(sc.bowl_x * 1000.0, 1), ",",
                fixed(sc.bowl_y * 1000.0, 1), ") mm | apart", fixed(sep * 1000.0, 1), "mm",
            )
            if sep < MIN_SEP_M:
                print("  ⚠ the brick is in or against the bowl — move it and press Enter")
                _ = stdin.line()
                continue
            if not (sc.brick_x >= 0.10 and sc.brick_x <= 0.37 and sc.brick_y >= -0.24 and sc.brick_y <= 0.21):
                print("  ⚠ the brick is outside the sim's desk_brick region (x 0.10..0.37, y -0.24..0.21)")
            if not (sc.bowl_x >= 0.13 and sc.bowl_x <= 0.38 and sc.bowl_y >= -0.23 and sc.bowl_y <= 0.19):
                print("  ⚠ the bowl is outside the sim's desk_bowl region (x 0.13..0.38, y -0.23..0.19)")
            if not rig.read_joints():
                print("  the follower did not report its pose; retrying")
                continue
            var q_home = rig.q.copy()
            # the planner's env in the rig's state
            var qs = scene_qpos(q_scene, rig.q, sc, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
            var pb: List[Float64] = [sc.brick_x, sc.brick_y, q_scene[brick_adr + 2]]
            var pw: List[Float64] = [sc.bowl_x, sc.bowl_y, q_scene[bowl_adr + 2]]
            var q5 = List[Float64]()
            for k in range(N_ARM):
                q5.append(qs[k])
            # plan, and let the operator look at it
            var choice = String("r")
            var plan = TowerGraspPlan()
            var legs = List[PlanLeg]()
            var seed_used = 0
            while choice == "r":
                env.set_state(qs, v0)
                seed_used = seed0 + draw
                draw += 1
                seed_rng(seed_used)
                plan = planner.plan_grasp(env, pb, sc.brick_yaw, q5, jaw_open)
                planner.plan_place(env, plan, pw)
                env.set_state(qs, v0)
                legs = plan.legs(place=True, close_steps=1)
                print("  seed", seed_used)
                print_plan(plan, legs)
                var warn = plan_warnings(plan)
                var needs_force = warn.byte_length() > 0
                if needs_force:
                    print(warn, end="")
                if not live:
                    print("  (dry run) Enter = next episode | r = redraw the posture | q = quit")
                else:
                    print(
                        "  Enter = RUN IT" + (" (type 'force': see ⚠ above)" if needs_force else "")
                        + " | r = redraw the posture | s = re-read the scene | q = quit"
                    )
                stdin.discard_pending()
                choice = stdin.line()
                if live and needs_force and choice == "":
                    print("  refused: type 'force' to run this plan")
                    choice = String("r")
            if choice == "q":
                break
            if choice == "s":
                continue
            if not live:
                ep += 1
                continue

            # ── the episode, on the arm ──────────────────────────────────
            if not rig.armed:
                rig.arm_torque(step_ticks)
                print("  follower torque ON")
            n_run += 1
            rig.trace = String(
                "tick\tleg\tcmd0\tcmd1\tcmd2\tcmd3\tcmd4\tcmd5\tq0\tq1\tq2\tq3\tq4\tq5"
                "\tv0\tv1\tv2\tv3\tv4\tv5\tload0\tload1\tload2\tload3\tload4\tload5\ttip_x\ttip_y\ttip_z\ttip_dist_mm\n"
            )
            for k in range(3):
                rig.tip_goal[k] = plan.tip_goal[k]
            rig.tick_n = 0
            rig.late = 0
            rig.drops = 0
            var close_how = String("-")
            var tip_close = -1.0
            var jaw_after = -1.0
            var aborted = String("")
            print("  running — press Enter to ABORT")
            stdin.discard_pending()
            var rec_index = -1
            if recording:
                for c in range(len(rig.cams)):
                    _ = rig.cams[c].drain()
                rec_index = rig.writers[0].n_episodes()
                rig.writers[0].begin_episode(task.copy())
                rig.rec_rows = 0
                rig.recording_now = True
            try:
                rig.begin(env)
                for leg in legs:
                    if leg.name == "hold":
                        continue
                    var rep = rig.run_leg(env, leg, settle_steps, leg_settle, stdin)
                    print("   ", rep)
                    if leg.close_on_tip:
                        tip_close = rig.tip_dist_mm()
                        close_how = String("trigger") if rep.find("TRIGGERED") >= 0 else String("timeout")
                    if leg.name == "close":
                        jaw_after = rig.q[5]
            except e:
                aborted = String(e)
                print("  ⚠", aborted)
            # home, out of the camera's way (also after an abort)
            try:
                if stdin.has_input():
                    _ = stdin.line()
                rig.ramp_to(env, q_home, N_HOME, String("home"), stdin)
                if recording:
                    rig.leg = String("rest")
                    for _ in range(N_REST_HOLD):
                        rig.tick(env, stdin)
            except e:
                print("  ⚠ the ramp home failed:", e, "— ending the run")
                raise Error(String(e))
            rig.recording_now = False
            var oc = read_outcome(rig.cams[0], cam, frame, sc, in_bowl_mm)
            save_png(out_dir + "/ep" + String(ep + 1) + "_end.png", frame, 640, 480, 3)
            var outcome = String("ABORTED") if aborted != "" else oc[0]
            if outcome == "SUCCESS":
                n_ok += 1
            buckets += " " + outcome
            print(
                "  ->", outcome, "| brick", fixed(oc[1] * 1000.0, 1), fixed(oc[2] * 1000.0, 1),
                "mm, from the bowl", fixed(oc[3], 1), "mm | bowl moved", fixed(oc[4], 1),
                "mm | late ticks", rig.late, "| dropped", rig.drops,
            )
            with open(out_dir + "/ep" + String(ep + 1) + ".tsv", "w") as fh:
                fh.write(rig.trace)
            var kept = String("-")
            if recording:
                # the camera's verdict decides, the operator may flip it: a
                # rejected episode stays in the files and the importer skips
                # it (`record.mojo`'s discard)
                var keep = outcome == "SUCCESS"
                stdin.discard_pending()
                print(
                    "  dataset episode", rec_index, "(" + String(rig.rec_rows), "frames):",
                    "KEEP" if keep else "REJECT", "— Enter = agree | k = keep | r = reject",
                )
                var v = stdin.line()
                if v == "k":
                    keep = True
                elif v == "r":
                    keep = False
                rig.writers[0].end_episode()
                if not keep:
                    _ = reject_episode(dataset_dir, rec_index)
                    n_rejected += 1
                else:
                    n_kept += 1
                kept = String("kept") if keep else String("rejected")
                print("  ", kept, "->", dataset_dir)
            summary += (
                String(ep + 1) + "\t" + String(seed_used) + "\t" + fixed(sc.brick_x, 4) + "\t"
                + fixed(sc.brick_y, 4) + "\t" + _deg(sc.brick_yaw) + "\t" + fixed(sc.bowl_x, 4)
                + "\t" + fixed(sc.bowl_y, 4) + "\t" + _deg(plan.tilt) + "\t" + String(plan.tries)
                + "\t" + fixed(plan.pen_mm, 1) + "\t" + close_how + "\t" + fixed(tip_close, 1)
                + "\t" + fixed(jaw_after, 3) + "\t" + outcome + "\t" + fixed(oc[1], 4) + "\t"
                + fixed(oc[2], 4) + "\t" + fixed(oc[3], 1) + "\t" + fixed(oc[4], 1) + "\t"
                + String(rig.late) + "\t" + String(rig.drops) + "\t" + kept + "\n"
            )
            with open(out_dir + "/episodes.tsv", "w") as fh:
                fh.write(summary)
            ep += 1
    finally:
        var released = return_and_release(rig.arm, start_pose, rig.armed, True, stdin, interactive)
        if not released:
            print("⚠ the follower is STILL ENERGISED — run `pixi run soarm-torque-off` once it is safe")
        if recording:
            try:
                if rig.recording_now:
                    # the run died inside an episode: keep the files whole,
                    # and the episode out of the dataset
                    var idx = rig.writers[0].n_episodes()
                    rig.recording_now = False
                    if rig.rec_rows > 0:
                        rig.writers[0].end_episode()
                        _ = reject_episode(dataset_dir, idx)
                        n_rejected += 1
                if n_kept + n_rejected > 0:
                    print("\nwriting the dataset ...")
                    rig.writers[0].close()
            except e:
                print("⚠ closing the dataset failed:", e)
        for c in range(len(rig.cams)):
            try:
                rig.cams[c].stop()
            except:
                pass

    print("=" * 70)
    if live:
        print("  episodes run", n_run, "| SUCCESS", n_ok, "| outcomes:" + buckets)
    if recording:
        print("  dataset", dataset_dir, ": kept", n_kept, "| rejected", n_rejected)
    print("  out:", out_dir)
    print("=" * 70)
