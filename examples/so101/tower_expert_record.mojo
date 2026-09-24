"""A scripted EXPERT for the tower tasks — waypoints + inverse kinematics on the
CPU env, headless, writing `.demo` files the SAC driver loads with `--demos`.

    pixi run mojo run -I . examples/so101/tower_expert_record.mojo                       # lift_brick, 20 episodes
    pixi run mojo run -I . examples/so101/tower_expert_record.mojo so101_tower_cube_in_bowl --episodes 50
    pixi run mojo run -I . examples/so101/tower_expert_record.mojo --episodes 30 --noise 0.05 --seed 7
    pixi run python tools/demo/demo_stats.py projects/so101-tower/demos/*.demo --episodes

## WHY AN EXPERT

Teleoperating the sim from a screen is hard for a reason no fix removes: a
human reads depth from parallax and touch, and gets neither. After the jaw
and contact fixes (see the bake and `gen_tower_props.py`) Denis could lift the
cube "a few times" and it still "did not feel 100% reality". HIL-SERL needs
10-20 clean successes to start from; a script that reaches, pinches, lifts
and places gives hundreds at physics speed, on randomised placements, in the
trainer's own observation and reward. Every success in the file is a real
pinch under the family's contact model — the rung (reward > 1.5) proves it.

## THE PLAN, PER EPISODE

Placements are the task's own (`posed_reset.posed_qpos` at `--seed + k`, the
host sampler over the family's regions). Site targets for `grasp_center`,
the fixed finger pointing DOWN and the pinch axis aligned RADIALLY (the cube
is axis-aligned at reset and the jaw closes across it either way):

    pre-grasp   brick + 8 cm            open
    grasp       brick + 3 cm            open, then CLOSE over 1 s
    lift        brick + 15 cm           closed      -> lift_brick holds here
    carry       bowl  + 16 cm           closed      (cube_in_bowl only)
    place       bowl  + 10 cm           closed, then OPEN
    retreat     bowl  + 16 cm           open        -> Near holds

Each leg interpolates joint targets linearly over its step budget, so the
actions are the normalised joint targets the batched env takes. `--noise`
adds per-step Gaussian noise to those targets WHILE THE JAW IS OPEN — flat
on the approach, tapered to zero along the descent (the last noisy target is
where the jaw closes: 11 mm off at 0.02 flat, 10/20); the lift and the hold
are noise-free, because noise on the lift jiggles the pinch.
Leave the noise ON for a training file: a policy fitted to noiseless ramps
reads its phase off the velocity words, and a 0.004 action error puts the
stiff servo 0.6 rad/s off the ramp, where the fit is unconstrained — the
closed loop then diverges at step 2 (`tower_policy_probe.mojo`, 2026-09-20).
The recorded action is the EXECUTED (noisy) one; its conditional mean is the
clean ramp, which is what the L1 fit converges to.

Success is the family's own predicate held `HOLD_STEPS` consecutive steps
(the recorder's rule); the episode ends there. A failed episode (the script
ran out) is dropped unless `--keep-failures`.

## THE IK

Damped least squares on the env's OWN forward kinematics (`set_state` +
`site_xpos` / `xquat`), finite-difference Jacobian over the five arm joints,
rows = site position (3) + the finger direction's horizontal components (2,
weighted) + the pinch-axis yaw error (1, weighted). Restarts from five seed
poses with the pan pre-aimed, over a relaxing tilt weight, keeping the best
position error. Measured in MuJoCo on the same scene before the port: a
strict vertical finger is unreachable beyond ~33 cm radius and the pinch
works tilted up to ~20 deg, so relaxing the weight is what took the
prototype from 3 of 10 to 10 of 10 placements in the bowl.

DAGGER (`--policy CKPT`): the checkpoint drives, greedy, until the jaw is
within `--handover-mm` (36, `CLOSE_REACH_MM`) of the brick with the arm settled, or for
`--policy-steps` (140); the expert then closes and lifts FROM THE POLICY'S
OWN STATE — a short descent to the grasp pose from wherever the jaw is, the
close, the lift, the hold — and those rows carry the INTERVENED flag. Five
BC runs and a pure-BC run parked at the arrival because no recorded row
says "close from here" for the states the policy reaches on its own; these
are those rows, HIL-SERL's human intervention with the expert as the human.
First smoke (checkpoint 69b67456, stiff jaws): handover in 7 of 12, lifts
in 11 of 12, 1113 of 2113 rows intervened.

DAGGER FROM A RECORDED STUDENT (`--handover-from FILE.demo`): the vision
student cannot run in this CPU recorder (it needs the device tracer), so its
episodes are recorded by `tower_act_eval.mojo --record-demo` and handed over
HERE, offline. Per recorded episode the handover row is the first row at or
after `HANDOVER_MIN_STEPS` with the jaw within `--handover-mm` of the brick,
MOVING OR NOT, else the student's closest row (`_handover_row` says why this
is not the `--policy` rule), the sim is put in that row's full state
(`obs_at` with its qpos + qvel words — brick and bowl included, so the
placement comes with it), and the expert runs from there exactly as after a
live handover. Only the expert's rows are written, all INTERVENED: the
student's own actions are not labels. Each kept episode starts at the
student's arrival — the "close from HERE" rows the student lacks.

HUMAN POSTURE (`--posture human`): per episode the IK targets a drawn TILT
(the finger leaning outward, `--tilt-range`, default 10..55 deg) and a drawn
PINCH from radial (`--pinch-range`, 35..85), snapped to the brick's nearest
face normal; wrist_roll is seeded positive — the operator's measured grasp
(`tools/soarm/grasp_posture.py`: tilt median 35, pinch 72, roll 77), where
the default expert grasps vertical and radial. A tilted grasp needs its own
mechanics, all active only in this mode:
  - the IK aims the FINGERTIPS (`TIP_REACH` past `grasp_center`);
  - the approach runs ALONG THE FINGER AXIS (`APPROACH_D`, through
    `APPROACH_WAYPOINTS` IK solutions), not vertically;
  - the close fires on the tips' distance to their grasp point
    (`--tip-close-mm`), not on `grasp_center`'s height;
  - integral action on the arm's tracking error (`SAG_KI`) takes out the
    shoulder's sag in the loaded poses;
  - the jaw opens to `HUMAN_JAW_OPEN` (0.6 rad) and the grasp is
    `HUMAN_Z_GRASP` (15 mm) — both overridable.
Measured, cube_in_bowl, folded start, 60 episodes at seed 21000: 30/60
(tilt < 20 deg 9/11, 20-30 11/13, 30-40 4/10, > 40 6/26); the realised tilt
matches the operator's (p5 / median / p95 13 / 34 / 59 against 9 / 35 / 60).
The history, 20 episodes each: vertical approach 2/20, tips aimed 4/20,
along the finger 14/20 at jaw 0.6 (5/20 at 0.9). ⚠ The pinch is MIRRORED
(-65 median against +72) while the cube is axis-aligned: the operator's
Duplo lies at arbitrary yaw; the task's opt-in brick yaw draw (DR session)
is what lets the snap reproduce it. The default (`expert`) draws nothing
and is byte-identical to before (20/20 at seed 11000, same bytes).

⚠ THE IK SETS THE ARM'S qpos TO EVALUATE FK AND RESTORES THE STATE AFTER.
It never steps physics. The props' qpos are untouched.
"""

from std.math import sqrt, atan2, cos, sin, log, pi, floor
from std.random import seed as seed_rng, random_float64
from std.sys import argv
from std.pathlib import Path

from noeira.nn.constants import DT
from noeira.core.cont_action import ContAction
from noeira.core.run import epoch_seconds, iso8601_utc
from noeira.io.proc import quote_arg, run_capture
from noeira.deep_agents.demos.ctrl_range import CtrlRange
from noeira.deep_agents.demos.file import DemoSet, read_demo_file
from noeira.deep_agents.demos.recorder import EpisodeRecorder, Handover
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from noeira.tasks.sac_family_policy import (
    SacFamilyPolicy, HIDDEN as FAMILY_HIDDEN, POLICY_BATCH, POLICY_CAP,
)
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.math3d import Quat, Vec3
from max.gpu.host import DeviceContext
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.host_reward import family_reward_host
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import fixed

comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime DEFAULT_TASK = "so101_tower_lift_brick"
comptime DEMO_DIR = "projects/so101-tower/demos"
comptime CFG = So101TowerTeleopConfig
comptime E = Phyics3dEnv[So101TowerModel, CFG, DType.float64, False]
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime NB = So101TowerModel.NBODY
comptime ACT = 6
comptime N_ARM = 5
comptime TIP_REACH: Float64 = 0.019
"""`grasp_center` (gripper z -0.068) to the fingertip boxes' middle (z
-0.087), along the finger. A vertical grasp moves that offset straight up and
`Z_GRASP` absorbs it; a grasp tilted by 35 deg moves the tips 11 mm sideways
off the brick, which then meets one jaw and tips over (2/20, wrist view of
the human-posture run, 2026-09-23). So a drawn tilt aims the tips."""
comptime IK_ROWS = 7
"""Position (3), the finger's horizontal components (2), the pinch axis's
yaw (sin, 1) and its direction (1 - cos, 1; weighted only in `--posture
human`, where the wrist roll's SIGN matters: the wrist camera turns with it,
and an unweighted fold left a quarter of the grasps at roll -79 against the
operator's +77)."""
comptime HUMAN_JAW_OPEN: Float64 = 0.6
comptime HUMAN_Z_GRASP: Float64 = 0.015
comptime APPROACH_D: Float64 = 0.07
"""A tilted grasp's approach length (m): the pre-grasp pose puts the tips this
far back ALONG THE FINGER AXIS from their grasp point (the vertical expert's
`Z_PRE - Z_GRASP`), and the descent follows that line."""
comptime APPROACH_WAYPOINTS = 3
"""IK solutions along the approach line; the joints interpolate between them,
so the tips stay near the line instead of cutting a chord."""
comptime TIP_CLOSE_MM_DEFAULT: Float64 = 8.0
"""A tilted grasp closes when the tips are this close to their grasp point and
the arm has settled (`--tip-close-mm`)."""
comptime SAG_KI: Float64 = 0.15
"""Integral gain on the arm's tracking error, per step (`Expert.integral`)."""
comptime SAG_MAX: Float64 = 0.35
"""The bias's clamp (rad, 20 deg) — above the worst sag measured (16 deg)."""
comptime ROLL_SEED_HUMAN: Float64 = 1.34
"""wrist_roll's seed on the tangential pinch (rad, 77 deg): the operator's
median grasp roll through the follower zero. The pinch axis is folded mod
180 deg, so without it the IK picks either sign."""
comptime GB = So101TowerConfig.OBS_GOAL_BASE
"""The goal words: obs[GB+3..GB+5] is the jaw-to-brick vector (the reach)."""

# ── DAgger: the policy drives to its own arrival, the expert takes over ──
comptime HIDDEN = FAMILY_HIDDEN
"""The family SAC widths, from `noeira/tasks/sac_family_policy.mojo` — the
driver trains with the same constants, so a `--policy` checkpoint loads."""
comptime BATCH = POLICY_BATCH
comptime CAP = POLICY_CAP
comptime Agent = SacFamilyPolicy[E.OBS_DIM, ACT]
comptime N_DESCEND_HANDOVER = 20
"""The expert's descent after a handover: the policy already brought the
jaw near the brick, so a short ramp to the grasp pose from wherever it is."""
comptime JAW_OPEN: Float64 = 0.9
"""The jaw's OPEN target (rad) on the approach — `--jaw-open`; the range is
-0.17..1.75. The moving tip swings DOWN in an arc as the jaw closes (its
lowest corner 80 mm above the desk at 1.75, ~40 at 0.9, 20 at 0.6, the
brick's top at 27): from wide open the close is a long swing whose tip
lands on the brick's top edge and whose success depends on the exact
arrival (a close at the first settled row: 7/40; at the ramp's end 30/40).
Half open, the close is a short swing and grasps 30/40 from the first
settled row — so the close can be state-triggered (`Z_CLOSE_ABOVE_MM`)."""
comptime HANDOVER_MIN_STEPS = 20
comptime CLOSE_REACH_MM: Float64 = 36.0
"""The DAgger handover's reach (obs words GB+3..5): measured on 300 z-0.01
demos, the reach at the grasp pose was 24.7-35.5 mm (median 24.9; the
spread is the lateral error over placements)."""
comptime Z_CLOSE_ABOVE_MM: Float64 = 27.0
"""The close is STATE-TRIGGERED: the descent ends and the jaw closes at the
first row where the gripper site is within this HEIGHT above the brick
centre (the reach word GB+5, which is brick − site, so it reads −25 at the
grasp) and the arm has settled (`SETTLED_VEL`). Measured on 300 z-0.01
demos: the ramp descent overshoots to −23.1 mm and settles at −23.6..−24.8
over its last 7 rows (p10 of placements: −31.7, the far ones the IK cannot
reach lower — those fall back to the ramp's end). Those 7 rows were the
defect: "settled at the grasp pose, jaw open, label OPEN" next to ONE row
of the same state labelled CLOSED, and the fitted policy's jaw command at
that state was open (the probe's teacher-forced error at the close row:
0.34, the full jump). ⚠ NOT THE COMMANDED HEIGHT: a test against the IK
target within 4 mm never fired (IK error + gravity sag), and a reach-norm
test (< 36 mm) fired 10 mm too high (1/40 clean, 0/12 DAgger)."""
comptime SETTLED_VEL: Float64 = 0.15
"""rad/s, every arm joint: settled (the median over the 7 rows before the
recorded close was 0.04-0.10; the descent runs at 0.39)."""
comptime GS = CFG.GRIPPER_SITE
comptime GRIPPER_BODY = CFG.GRIPPER_BODY
comptime HOLD_STEPS: Int = 31
"""The goal held this many consecutive steps = success (the recorder's rule)."""
comptime FB_MIN_STEP: Float64 = 0.02
"""Feedback demonstrator: the slowest a joint is driven, rad per control step
(0.6 rad/s) — the pull-back speed of a joint that noise displaced."""
comptime FB_VEL: Float64 = 0.05
"""Feedback demonstrator: the arm has settled when every joint is slower than
this, rad/s."""

# The legs' step budgets at 31.25 Hz.
comptime N_PRE = 40
comptime N_DESCEND = 40
comptime N_CLOSE = 1
"""The close is a STEP: the jaw command goes to closed at once. It was a
30-step ramp, then 15; a step labels the arrival state with the full-range
jump the fitted policy needs (a ramp's first row is 1/30 of the range below
open, next to the approach's "open" at the same state, and the policy's jaw
never moved). A step only works with the jaw DEEP enough around the brick —
see `Z_GRASP`: at the old 0.03 the tip brushed the brick's top edge (0.14 mm
of overlap, in MuJoCo too) and a fast close missed it every time (0/20)."""
comptime N_LIFT = 40
comptime N_CARRY = 50
comptime N_PLACE = 25
comptime N_OPEN = 20
comptime N_RETREAT = 20
comptime N_RETURN_REST = 60
"""`--return-rest`: steps to fold from the retreat pose back to the start
(~1.9 s; the real fold takes 1-3 s)."""
comptime N_REST_HOLD = 15
"""Steps held at rest after the fold (0.5 s of the real ~3 s): enough to
show "stay folded", short of teaching the idle."""
comptime N_HOLD_MAX = 120
"""Steps to wait for the predicate to hold after the last leg."""

comptime Z_PRE = 0.08
comptime Z_GRASP = 0.01
"""The gripper site's height above the brick CENTRE at the grasp. It was
0.03 — measured against the jaws' phantom hull, before the box jaws — and
the tips only brushed the brick's top edge, so the grasp worked by the slow
ramp sweeping the brick into the fixed jaw. Sweep on 20 placements, stiff
jaws (close steps 15 / 1): 0.03 16/0, 0.025 16/0, 0.02 16/0, 0.015 16/13,
0.01 16/16, 0.005 16/16. At 0.01 a one-step close grasps as well as the
ramp (30/40 on other seeds; 25/40 with flat noise 0.02)."""
comptime Z_LIFT = 0.15
comptime Z_CARRY = 0.16
comptime Z_PLACE = 0.10


# ── kinematics on the env ────────────────────────────────────────────────


struct Arm(Movable):
    """The IK's view of the arm: FK through the env, joint limits, seeds."""

    var lo: List[Float64]
    var hi: List[Float64]
    var seeds: List[List[Float64]]

    def __init__(out self, mut env: E) raises:
        var sf = So101TowerModel.make_spec_fields[DType.float64]()
        var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, ACT)
        var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, ACT)
        self.lo = List[Float64]()
        self.hi = List[Float64]()
        for i in range(ACT):
            self.lo.append(Float64(lo_col[i]))
            self.hi.append(Float64(hi_col[i]))
        self.seeds = List[List[Float64]]()
        self.seeds.append([0.0, -0.5, 0.5, 0.5, 0.0])
        self.seeds.append([0.0, 0.3, 0.8, 0.4, 0.0])
        self.seeds.append([0.0, 0.8, 0.0, 0.0, 0.0])
        self.seeds.append([0.0, -1.0, 1.2, 1.0, 0.0])
        _ = env

    def fk(
        self, mut env: E, ref q: List[Float64],
        mut p: List[Float64], mut fz: List[Float64], mut gx: List[Float64],
    ):
        """Site position, the finger axis (gripper -z) and the pinch axis
        (gripper +x) in the world, at arm joints `q` — the env's FK."""
        for i in range(N_ARM):
            env.d.qpos.data[i] = q[i]
        env._fields_fk()
        for k in range(3):
            p[k] = Float64(env.d.site_xpos.data[GS * 3 + k])
        var o = GRIPPER_BODY * 4
        # xquat is stored (x, y, z, w); Quat is (w, x, y, z)
        var qw = Quat(
            Float64(env.d.xquat.data[o + 3]), Float64(env.d.xquat.data[o]),
            Float64(env.d.xquat.data[o + 1]), Float64(env.d.xquat.data[o + 2]),
        )
        var f = qw.rotate_vec(Vec3(0.0, 0.0, -1.0))
        var g = qw.rotate_vec(Vec3(1.0, 0.0, 0.0))
        fz[0] = f.x
        fz[1] = f.y
        fz[2] = f.z
        gx[0] = g.x
        gx[1] = g.y
        gx[2] = g.z

    def _feats(
        self, mut env: E, ref q: List[Float64], yaw: Float64,
        mut out: List[Float64], tip: Bool = False,
    ):
        """`tip`: the position feature is the FINGERTIP point, `TIP_REACH`
        further along the finger than `grasp_center` (see `TIP_REACH`)."""
        var p = List[Float64](length=3, fill=0.0)
        var fz = List[Float64](length=3, fill=0.0)
        var gx = List[Float64](length=3, fill=0.0)
        self.fk(env, q, p, fz, gx)
        var d = TIP_REACH if tip else 0.0
        out[0] = p[0] + d * fz[0]
        out[1] = p[1] + d * fz[1]
        out[2] = p[2] + d * fz[2]
        out[3] = fz[0]
        out[4] = fz[1]
        # sin of the yaw error between the pinch axis and the wanted axis
        out[5] = gx[0] * sin(yaw) - gx[1] * cos(yaw)
        # 1 - cos of it: zero only when the axis points the WANTED WAY, so
        # it picks the wrist roll's sign (weighted only when `directed`)
        out[6] = 1.0 - (gx[0] * cos(yaw) + gx[1] * sin(yaw))

    def _ik_once(
        self, mut env: E, ref target: List[Float64], ref q0: List[Float64],
        yaw: Float64, w_tilt: Float64, w_yaw: Float64, iters: Int,
        mut q_out: List[Float64], tilt: Float64 = 0.0, directed: Bool = False,
    ) -> Float64:
        """Damped least squares from `q0`; returns the position error.
        `tilt` (rad): the finger's wanted angle from straight down, leaning
        OUTWARD along the target's radial direction (0 = vertical)."""
        var rad = atan2(target[1], target[0])
        var fgoal_x = sin(tilt) * cos(rad)
        var fgoal_y = sin(tilt) * sin(rad)
        # a drawn tilt aims the FINGERTIPS, not `grasp_center`; the target
        # moves down by the same `TIP_REACH`, so at tilt 0 it is the old one
        var tip = tilt > 0.0
        var tgt = List[Float64]()
        tgt.append(target[0])
        tgt.append(target[1])
        tgt.append(target[2] - (TIP_REACH if tip else 0.0))
        var q = List[Float64]()
        for i in range(N_ARM):
            q.append(q0[i])
        var W = List[Float64]()
        W.append(1.0)
        W.append(1.0)
        W.append(1.0)
        W.append(w_tilt)
        W.append(w_tilt)
        W.append(w_yaw)
        # the directed-pinch row: weight 0 unless `directed`, so the default
        # expert's normal system is unchanged, bit for bit
        W.append(w_yaw if directed else 0.0)
        var f = List[Float64](length=IK_ROWS, fill=0.0)
        var f2 = List[Float64](length=IK_ROWS, fill=0.0)
        var e = List[Float64](length=IK_ROWS, fill=0.0)
        var J = List[Float64](length=IK_ROWS * N_ARM, fill=0.0)
        var H = List[Float64](length=N_ARM * N_ARM, fill=0.0)
        var g = List[Float64](length=N_ARM, fill=0.0)
        var perr = 1.0
        for _ in range(iters):
            self._feats(env, q, yaw, f, tip)
            for r in range(IK_ROWS):
                var goal = tgt[r] if r < 3 else (
                    fgoal_x if r == 3 else (fgoal_y if r == 4 else 0.0)
                )
                e[r] = (goal - f[r]) * W[r]
            perr = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
            var dfx = f[3] - fgoal_x
            var dfy = f[4] - fgoal_y
            if perr < 5e-4 and sqrt(dfx * dfx + dfy * dfy) < 0.05:
                break
            var eps = 1e-6
            for j in range(N_ARM):
                var dq = List[Float64]()
                for i in range(N_ARM):
                    dq.append(q[i])
                dq[j] += eps
                self._feats(env, dq, yaw, f2, tip)
                for r in range(IK_ROWS):
                    J[r * N_ARM + j] = (f2[r] - f[r]) * W[r] / eps
            # H = J^T J + lambda I,  g = J^T e
            for a in range(N_ARM):
                g[a] = 0.0
                for b in range(N_ARM):
                    var s = 0.0
                    for r in range(IK_ROWS):
                        s += J[r * N_ARM + a] * J[r * N_ARM + b]
                    H[a * N_ARM + b] = s + (1e-5 if a == b else 0.0)
                for r in range(IK_ROWS):
                    g[a] += J[r * N_ARM + a] * e[r]
            var step = _solve5(H, g)
            for i in range(N_ARM):
                var s = step[i]
                if s > 0.25:
                    s = 0.25
                if s < -0.25:
                    s = -0.25
                q[i] += s
                if q[i] < self.lo[i]:
                    q[i] = self.lo[i]
                if q[i] > self.hi[i]:
                    q[i] = self.hi[i]
        for i in range(N_ARM):
            q_out[i] = q[i]
        return perr

    def ik(
        self, mut env: E, ref target: List[Float64], ref q0: List[Float64],
        yaw: Float64, mut q_out: List[Float64], tilt: Float64 = 0.0,
        roll_seed: Float64 = 0.0, use_roll_seed: Bool = False,
    ) -> Float64:
        """Restarts x a relaxing tilt weight; keeps the best position error.
        Restores the env's arm qpos to what it was.

        `tilt` > 0 targets a finger leaning outward by that angle (the human
        posture, `--posture human`); its weights start 6x stronger, since a
        drawn tilt is a TARGET and the vertical one was only a preference.
        `use_roll_seed` starts every restart's wrist_roll at `roll_seed`:
        the pinch axis is folded mod 180 deg, so the roll's SIGN is chosen by
        the seed (the human's is always positive)."""
        var saved = List[Float64]()
        for i in range(NQ):
            saved.append(Float64(env.d.qpos.data[i]))
        var best = 1e9
        var q_try = List[Float64](length=N_ARM, fill=0.0)
        var weights: List[Float64] = [0.05, 0.02, 0.008, 0.003]
        if tilt > 0.0:
            weights = [0.3, 0.12, 0.05, 0.02]
        var pan = atan2(target[1], target[0])
        for wi in range(len(weights)):
            for s in range(len(self.seeds) + 1):
                var s0 = List[Float64]()
                for i in range(N_ARM):
                    s0.append(q0[i] if s == 0 else self.seeds[s - 1][i])
                s0[0] = pan
                if use_roll_seed and s > 0:
                    s0[4] = roll_seed
                var err = self._ik_once(
                    env, target, s0, yaw, weights[wi], 0.02, 100, q_try, tilt,
                    use_roll_seed,
                )
                if err < best - 1e-4:
                    best = err
                    for i in range(N_ARM):
                        q_out[i] = q_try[i]
            if best < 1e-3:
                break
        for i in range(NQ):
            env.d.qpos.data[i] = saved[i]
        env._fields_fk()
        return best


def _solve5(ref H: List[Float64], ref g: List[Float64]) -> List[Float64]:
    """Gaussian elimination with partial pivoting on the 5x5 normal system."""
    var n = N_ARM
    var A = List[Float64]()
    for i in range(n * n):
        A.append(H[i])
    var b = List[Float64]()
    for i in range(n):
        b.append(g[i])
    for c in range(n):
        var piv = c
        for r in range(c + 1, n):
            if abs(A[r * n + c]) > abs(A[piv * n + c]):
                piv = r
        if piv != c:
            for k in range(n):
                var t = A[c * n + k]
                A[c * n + k] = A[piv * n + k]
                A[piv * n + k] = t
            var tb = b[c]
            b[c] = b[piv]
            b[piv] = tb
        var d = A[c * n + c]
        if abs(d) < 1e-14:
            continue
        for r in range(c + 1, n):
            var f = A[r * n + c] / d
            for k in range(c, n):
                A[r * n + k] -= f * A[c * n + k]
            b[r] -= f * b[c]
    var x = List[Float64](length=n, fill=0.0)
    for c in range(n - 1, -1, -1):
        var s = b[c]
        for k in range(c + 1, n):
            s -= A[c * n + k] * x[k]
        var d = A[c * n + c]
        x[c] = s / d if abs(d) > 1e-14 else 0.0
    return x^


# ── the episode ──────────────────────────────────────────────────────────


struct Expert(Movable):
    var arm: Arm
    var ctrl: CtrlRange
    var rec: EpisodeRecorder
    """The episode bookkeeping (`deep_agents/demos`): rows, the HOLD_STEPS
    success rule, keep / drop, the file rewritten per kept episode."""
    var noise: Float64
    var feedback: Bool
    var flat_noise: Bool
    var intervening: Bool
    """Rows recorded while True carry the INTERVENED flag (the expert
    driving after a `--policy` handover — HIL-SERL's human, scripted)."""
    var human_posture: Bool
    """`--posture human`: per episode, a drawn tilt and a face-snapped pinch
    that cover the operator's measured grasps (see `draw_posture`)."""
    var tilt_lo: Float64
    var tilt_hi: Float64
    var tilt: Float64       # this episode's drawn tilt (rad), 0 = vertical
    var tangential: Bool    # kept for the log: the snapped face is closer to tangential than radial
    var pinch_lo: Float64
    var pinch_hi: Float64
    var pinch_target: Float64  # this episode's drawn pinch from radial (rad), before the snap
    var integral: Bool
    """Integral action on the arm joints' tracking error (on with `--posture
    human`): the tilted poses load the shoulder, and the position servo rests
    8-16 deg of lift short of its target, the gripper 10-40 mm above the
    grasp height, so the close fired at the ramp's end over the brick (2/20).
    Each step `sag_bias += SAG_KI * (reference - q)`, and the command is the
    ramp's reference plus the bias — what a human on the leader arm does by
    eye."""
    var sag_bias: List[Float64]
    var return_rest: Bool
    """`--return-rest`: after a success, fold the arm back to the episode's
    own start pose and hold it — every real episode ends that way (50/50,
    then ~3 s at rest), and without it a student meets the fold back and the
    rest on real frames with no row that shows them (the DR session's phase
    split of H seed 1's real error: rest + return = 54 % of the rows, roll
    error 46-48 deg there). No idle START is recorded: it could freeze a
    policy at rest on the arm."""
    var q_home: List[Float64]
    var tip_trigger: Bool
    """This episode's close fires on the TIP distance (a tilted grasp), not on
    `grasp_center`'s height, which assumes a vertical finger."""
    var tip_goal: List[Float64]
    var tip_close_mm: Float64
    var q_ref: List[Float64]
    """The ramp's reference WITHOUT the bias — what `hold` holds, so the bias
    is not counted twice."""
    var close_steps: Int
    var z_grasp: Float64      # `--z-grasp`: the site's height above the brick centre at the grasp
    var close_above_mm: Float64  # `--close-above-mm`: the close's height trigger; 0 = off (close at the ramp's end)
    var jaw_open: Float64     # `--jaw-open`: the jaw's OPEN target (rad) on the approach; the moving tip
                              # swings down in an arc as it closes (80 mm up at 1.75, ~40 at 0.9, 20 at 0.6),
                              # so a half-open approach makes the close a short swing
    var frame_skip: Int
    var timestep: Float64
    var q_cmd: List[Float64]
    var obs: List[Scalar[DT]]
    var prev_obs: List[Scalar[DT]]
    var act_l: List[Float64]
    var steps: Int
    var rung_rows: Int

    def __init__(
        out self, mut env: E, noise: Float64, feedback: Bool = False,
        out_path: String = "", keep_failures: Bool = False,
    ) raises:
        self.arm = Arm(env)
        self.ctrl = CtrlRange(self.arm.lo.copy(), self.arm.hi.copy())
        self.rec = EpisodeRecorder(
            E.OBS_DIM, ACT, out_path, keep_failures, HOLD_STEPS
        )
        self.noise = noise
        self.feedback = feedback
        self.flat_noise = False
        self.intervening = False
        self.human_posture = False
        self.tilt_lo = 10.0 * pi / 180.0
        self.tilt_hi = 55.0 * pi / 180.0
        self.tilt = 0.0
        self.tangential = False
        self.pinch_lo = 35.0 * pi / 180.0
        self.pinch_hi = 85.0 * pi / 180.0
        self.pinch_target = 0.0
        self.integral = False
        self.sag_bias = List[Float64](length=N_ARM, fill=0.0)
        self.return_rest = False
        self.q_home = List[Float64](length=N_ARM, fill=0.0)
        self.tip_trigger = False
        self.tip_goal = List[Float64](length=3, fill=0.0)
        self.tip_close_mm = TIP_CLOSE_MM_DEFAULT
        self.q_ref = List[Float64](length=N_ARM, fill=0.0)
        self.close_steps = N_CLOSE
        self.z_grasp = Z_GRASP
        self.close_above_mm = Z_CLOSE_ABOVE_MM
        self.jaw_open = JAW_OPEN
        self.frame_skip = CFG.FRAME_SKIP
        self.timestep = So101TowerModel.TIMESTEP
        self.q_cmd = List[Float64](length=ACT, fill=0.0)
        self.obs = List[Scalar[DT]](length=E.OBS_DIM, fill=Scalar[DT](0))
        self.prev_obs = List[Scalar[DT]](length=E.OBS_DIM, fill=Scalar[DT](0))
        self.act_l = List[Float64](length=ACT, fill=0.0)
        self.steps = 0
        self.rung_rows = 0

    def draw_posture(mut self):
        """The operator's grasp, drawn — `--posture human`.

        Measured on the 50 kept real cube-in-bowl episodes at the grasp frame
        (`tools/soarm/grasp_posture.py`, 2026-09-23), in the expert's own IK
        features: TILT, the finger's angle from straight down leaning
        outward, p5..p95 = 9..60 deg, median 35; PINCH, the pinch axis from
        radial folded to (-90, 90], 34..86, median 72 — always on the
        positive side (wrist_roll 35..103, median 77). The vertical expert
        was tilt 0.2 / pinch 0, and students trained on it ignored real
        frames (40-47 deg mean action error on the real import).

        Tilt ~ U(`tilt_lo`, `tilt_hi`); pinch target ~ U(`pinch_lo`,
        `pinch_hi`), then SNAPPED to the brick's nearest face normal
        (`pinch_yaw`): a pinch between two face normals closes on corners.
        With the cube axis-aligned (today's tasks) the snap moves a 45-deg
        world line to 0 or 90, so the realised pinch is mirrored (-65 median,
        measured) — the operator's real Duplo lies at arbitrary yaw; the
        task's opt-in brick yaw draw makes the snap reproduce it."""
        self.tilt = random_float64(self.tilt_lo, self.tilt_hi)
        self.pinch_target = random_float64(self.pinch_lo, self.pinch_hi)

    def pinch_yaw(mut self, bearing: Float64, brick_yaw: Float64) -> Float64:
        """The pinch axis's world yaw for a brick at `bearing` whose own yaw
        is `brick_yaw` (rad): the radial direction, or — in `--posture
        human` — the brick face normal nearest `bearing + pinch_target`."""
        if not self.human_posture:
            return bearing
        var want = bearing + self.pinch_target
        var k = Float64(Int(floor((want - brick_yaw) / (pi / 2.0) + 0.5)))
        var y = brick_yaw + k * (pi / 2.0)
        var rel = y - bearing
        # the log's "tangential": the snapped line is closer to 90 than to 0
        var c = cos(rel)
        self.tangential = c * c < 0.5
        return y

    def _normalized(self, i: Int, q: Float64) -> Float64:
        return self.ctrl.normalize(i, q)

    def _apply(mut self, mut env: E) raises -> Bool:
        """Step the env with `self.act_l`, pay the family's reward, record the
        transition (flagged INTERVENED while `self.intervening`). Returns
        True when the goal has held `HOLD_STEPS` steps."""
        var action = ContAction[ACT]()
        for i in range(ACT):
            action.data[i] = self.act_l[i]
        for i in range(E.OBS_DIM):
            self.prev_obs[i] = self.obs[i]
        var out = env.step(action)
        for i in range(E.OBS_DIM):
            self.obs[i] = Scalar[DT](out[0].data[i])
        self.steps += 1
        var rd = family_reward_host[CFG, DType.float64, E.MD, ACT](
            env.d, env.mf, self.act_l, self.steps, self.frame_skip,
            self.timestep,
        )
        var r = Float64(rd[0])
        if r > 1.5:
            self.rung_rows += 1
        return self.rec.record(
            self.prev_obs, self.act_l, r, self.obs, rd[1], self.intervening
        )

    def tip_dist_mm(self, mut env: E) -> Float64:
        """The fingertip point (`grasp_center` + `TIP_REACH` along the finger)
        to `tip_goal`, from the env's current FK."""
        var o = GRIPPER_BODY * 4
        var qw = Quat(
            Float64(env.d.xquat.data[o + 3]), Float64(env.d.xquat.data[o]),
            Float64(env.d.xquat.data[o + 1]), Float64(env.d.xquat.data[o + 2]),
        )
        var f = qw.rotate_vec(Vec3(0.0, 0.0, -1.0))
        var dx = Float64(env.d.site_xpos.data[GS * 3]) + TIP_REACH * f.x - self.tip_goal[0]
        var dy = Float64(env.d.site_xpos.data[GS * 3 + 1]) + TIP_REACH * f.y - self.tip_goal[1]
        var dz = Float64(env.d.site_xpos.data[GS * 3 + 2]) + TIP_REACH * f.z - self.tip_goal[2]
        return sqrt(dx * dx + dy * dy + dz * dz) * 1000.0

    def reach_mm(self) -> Float64:
        var x = Float64(self.obs[GB + 3])
        var y = Float64(self.obs[GB + 4])
        var z = Float64(self.obs[GB + 5])
        return sqrt(x * x + y * y + z * z) * 1000.0

    def policy_approach(
        mut self, mut env: E, mut agent: Agent, handover_mm: Float64,
        max_steps: Int,
    ) raises -> Tuple[Bool, Bool]:
        """DAgger's first half: the CHECKPOINT drives (greedy) until the jaw
        is within `handover_mm` of the brick with the arm settled, or
        `max_steps` have passed. Rows are recorded unflagged. Returns
        (handed over at the arrival, episode already done)."""
        var a32 = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
        var h = Handover(HANDOVER_MIN_STEPS, max_steps)
        for k in range(h.max_steps):
            agent.select_greedy_action(self.obs, a32)
            for i in range(ACT):
                self.act_l[i] = Float64(a32[i])
            if self._apply(env):
                return (False, True)
            if h.arrived_at(k, self.reach_mm() < handover_mm and self.settled()):
                return (True, False)
        return (False, False)

    def settled(self) -> Bool:
        """Every joint — the jaw included — slower than `SETTLED_VEL`."""
        for i in range(ACT):
            if abs(Float64(self.obs[NQ + i])) > SETTLED_VEL:
                return False
        return True

    def step_to(
        mut self, mut env: E, ref q_target: List[Float64], grip_open: Bool,
        n_steps: Int, taper_noise: Bool = False, until_held: Bool = False,
        close_on_height: Bool = False, past_success: Bool = False,
    ) raises -> Bool:
        """Drive the joints to `q_target` and the gripper open/closed;
        record each transition. Returns True when the goal held
        `HOLD_STEPS` steps (the episode is over).

        ⚠ TWO DEMONSTRATORS. The default interpolates the COMMAND from where
        it was to the target over exactly `n_steps` (ramps) — a function of
        time, not of the state: a policy fitted to it to L1 0.004 still
        parked, because its own rollout arrives at the grasp pose a few mm
        off the recorded arrival and no recorded state says "close from
        here" (`tower_policy_probe.mojo`, run dd8d4a64: the gripper word sat
        1.8 rad from the recording from step 90 on, reward 1.233 for ever).
        `--feedback` commands `q_now + clamp(q_target - q_now, ±d)` every
        step — a STATE-FEEDBACK controller whose action is a function of the
        observation and that moves on when the command has reached the
        target and the arm has settled. ⚠ NOT YET THE BETTER DEMONSTRATOR:
        17/20 clean against the ramps' 18/20 on the same seeds, and 9/20 at
        noise 0.02 against 16/20 — the arm jitters through the arrival test
        and the phases run to their cap. Opt-in until that is understood.
        `--flat-noise` keeps the descent's noise flat instead of tapered:
        the arrivals it records are PERTURBED, and the successes among them
        are the "close from here" states the ramps never wrote.
        `d` is the ramp's own speed (the phase's distance over `n_steps`),
        floored at `FB_MIN_STEP` so a joint that should not move is still
        pulled back when noise moves it."""
        var q_start = List[Float64]()
        for i in range(ACT):
            q_start.append(self.q_cmd[i])
        var g_target = self.jaw_open if grip_open else self.arm.lo[5]
        var dmax = List[Float64]()
        for i in range(N_ARM):
            var qi = Float64(env.d.qpos.data[i])
            var d = abs(q_target[i] - qi) / Float64(n_steps)
            dmax.append(d if d > FB_MIN_STEP else FB_MIN_STEP)
        var g0 = Float64(env.d.qpos.data[5])
        var dg = abs(g_target - g0) / Float64(n_steps)
        if dg < FB_MIN_STEP:
            dg = FB_MIN_STEP
        var max_steps = 2 * n_steps if self.feedback and not until_held else n_steps
        var settled = 0
        for k in range(max_steps):
            var a = Float64(k + 1) / Float64(n_steps)
            if a > 1.0:
                a = 1.0
            if self.feedback:
                for i in range(N_ARM):
                    var qi = Float64(env.d.qpos.data[i])
                    var e = q_target[i] - qi
                    if e > dmax[i]:
                        e = dmax[i]
                    if e < -dmax[i]:
                        e = -dmax[i]
                    self.q_cmd[i] = qi + e
                # ⚠ THE GRIPPER IS RATE-LIMITED ON ITS COMMAND, NOT ON ITS
                # POSITION: closed on the cube it stalls above `lo`, and a
                # command of `g_now - d` squeezes with kp·d only — the brick
                # slipped out on the lift (6/20). The command walks to `lo`
                # and stays there, as the ramps did; the squeeze is kp·(g - lo).
                var eg = g_target - self.q_cmd[5]
                if eg > dg:
                    eg = dg
                if eg < -dg:
                    eg = -dg
                self.q_cmd[5] = self.q_cmd[5] + eg
            else:
                for i in range(N_ARM):
                    var ref_i = q_start[i] + (q_target[i] - q_start[i]) * a
                    self.q_ref[i] = ref_i
                    self.q_cmd[i] = ref_i + self.sag_bias[i] if self.integral else ref_i
                self.q_cmd[5] = q_start[5] + (g_target - q_start[5]) * a
            if close_on_height:
                # the descent's jaw target is a STEP to `jaw_open`: after a
                # DAgger handover the policy's jaw is wide open, and a ramp
                # from there is 20 rows of a time-indexed label at a settled
                # arm — the defect this trigger removes. A step makes the
                # half-close one state-indexed event (jaw at `jaw_open`,
                # stopped: `settled` includes the jaw) and the close the next.
                self.q_cmd[5] = g_target
            for i in range(ACT):
                var v = self._normalized(i, self.q_cmd[i])
                # ⚠ WHILE THE JAW IS OPEN ONLY — the approach and the descent.
                # Noise on the lift jiggles the pinch (10/20 at 0.01); noise on
                # the approach is the coverage the closed loop NEEDS: a policy
                # fitted to noiseless ramps reads its phase off the velocity
                # words, and one 0.004 action error puts the stiff servo 0.6
                # rad/s off the ramp, where the fit says nothing (probe, 20 Sep).
                # ⚠ AND TAPERED TO ZERO ALONG THE DESCENT: the last noisy
                # target is where the jaw sits when it closes, and 0.02 of the
                # range is 11 mm at the cube — 10/20 with the noise flat.
                if self.noise > 0.0 and grip_open:
                    var sigma = self.noise * (1.0 - a) if (taper_noise and not self.flat_noise) else self.noise
                    var u1 = random_float64(1e-12, 1.0)
                    var u2 = random_float64(0.0, 1.0)
                    v += sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
                    if v > 1.0:
                        v = 1.0
                    if v < -1.0:
                        v = -1.0
                self.act_l[i] = v
            if self._apply(env) and not past_success:
                return True
            if self.integral and not self.feedback:
                for i in range(N_ARM):
                    var b = self.sag_bias[i] + SAG_KI * (
                        self.q_ref[i] - Float64(env.d.qpos.data[i])
                    )
                    if b > SAG_MAX:
                        b = SAG_MAX
                    if b < -SAG_MAX:
                        b = -SAG_MAX
                    self.sag_bias[i] = b
            # the descent: hand over to the close at the first settled row
            # at the grasp height — see `Z_CLOSE_ABOVE_MM`
            if (
                close_on_height and self.tip_trigger and k + 1 >= 3
                and self.tip_dist_mm(env) < self.tip_close_mm
                and self.settled()
                and abs(self.q_cmd[5] - g_target) < 1e-9
            ):
                return False
            if (
                close_on_height and not self.tip_trigger
                and self.close_above_mm > 0.0 and k + 1 >= 3
                and Float64(self.obs[GB + 5]) * 1000.0 > -self.close_above_mm
                and self.settled()
                # and the jaw's command has reached its open target: after a
                # DAgger handover the policy's jaw is FULLY open and the
                # descent ramps it to `jaw_open`; a close from wide open is
                # the long swing that misses (2/12 without this)
                and abs(self.q_cmd[5] - g_target) < 1e-9
            ):
                return False
            if self.feedback and not until_held:
                # arrived: the COMMAND of every arm joint is at its target
                # (the clamp is inactive — under gravity the joint itself
                # rests a sag below, and a position test never passes), the
                # arm has stopped moving, and the gripper's command is at its
                # target.
                var arrived = True
                for i in range(N_ARM):
                    if abs(self.q_cmd[i] - q_target[i]) > 1e-9:
                        arrived = False
                    if abs(Float64(env.d.qvel.data[i])) > FB_VEL:
                        arrived = False
                if abs(self.q_cmd[5] - g_target) > 1e-9:
                    arrived = False
                if arrived:
                    settled += 1
                else:
                    settled = 0
                if settled >= 2 and k + 1 >= n_steps // 2:
                    break
        return False

    def hold(
        mut self, mut env: E, grip_open: Bool, n_steps: Int,
        until_held: Bool = False, past_success: Bool = False,
    ) raises -> Bool:
        var q = List[Float64]()
        for i in range(N_ARM):
            q.append(self.q_ref[i] if self.integral else self.q_cmd[i])
        return self.step_to(
            env, q, grip_open, n_steps, until_held=until_held,
            past_success=past_success,
        )


def _body_pos(mut env: E, b: Int) -> List[Float64]:
    var p = List[Float64]()
    for k in range(3):
        p.append(Float64(env.d.xpos.data[b * 3 + k]))
    return p^


def _body_yaw(mut env: E, b: Int) -> Float64:
    """Body `b`'s yaw about world z (rad), from its world quaternion."""
    var o = b * 4
    # xquat is stored (x, y, z, w)
    var x = Float64(env.d.xquat.data[o])
    var y = Float64(env.d.xquat.data[o + 1])
    var z = Float64(env.d.xquat.data[o + 2])
    var w = Float64(env.d.xquat.data[o + 3])
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _above(ref p: List[Float64], dz: Float64) -> List[Float64]:
    var t = List[Float64]()
    t.append(p[0])
    t.append(p[1])
    t.append(p[2] + dz)
    return t^


def _row_reach_mm(ref d: DemoSet, r: Int) -> Float64:
    """`Expert.reach_mm` read off a recorded row's observation."""
    var b = r * d.obs_dim + GB
    var x = Float64(d.obs[b + 3])
    var y = Float64(d.obs[b + 4])
    var z = Float64(d.obs[b + 5])
    return sqrt(x * x + y * y + z * z) * 1000.0


def _row_settled(ref d: DemoSet, r: Int) -> Bool:
    """`Expert.settled` read off a recorded row's observation."""
    for i in range(ACT):
        if abs(Float64(d.obs[r * d.obs_dim + NQ + i])) > SETTLED_VEL:
            return False
    return True


def _handover_row(
    ref d: DemoSet, e: Int, handover_mm: Float64, settled: Bool = False
) -> Tuple[Int, Bool]:
    """The row of episode `e` the expert takes over at: the FIRST row at or
    after `HANDOVER_MIN_STEPS` with the jaw within `handover_mm` of the brick
    (an arrival), else the row where the student came CLOSEST.

    ⚠ NOT THE `--policy` RULE, AND ON PURPOSE. That rule also waits for the
    arm to settle and falls back to a step cap. The first vision student
    (ACT f4ef105f, 128 recorded episodes) arrives MOVING and closes at once:
    its failed episodes reach a median 32 mm at step ~62 with the jaw still
    open (0.91 rad), close, miss, lift away and hover 130-170 mm off — only
    29 of 92 ever had a settled row under 36 mm, so the settled rule fell
    back to the cap at row 140, AFTER the miss, and the expert labelled a
    fresh approach from far away instead of the grasp the student gets
    wrong. The first row under the radius is before the student's close.

    `settled` (`--handover-settled`): the first row under the radius WITH
    every joint slower than `SETTLED_VEL`, and NO fallback — an episode with
    no such row returns row -1 and is skipped. The arm, not the rule, is the
    experiment here: the moving-arrival rows are the leading suspect for the
    first DAgger round's result (ACT sees joint positions and images, no
    velocities, so a moving and a settled arrival can look alike while their
    labels differ), and these rows are the unambiguous half."""
    var start = d.ep_start[e]
    var n = d.ep_len[e]
    if settled:
        for k in range(HANDOVER_MIN_STEPS, n):
            if _row_reach_mm(d, start + k) < handover_mm and _row_settled(d, start + k):
                return (start + k, True)
        return (-1, False)
    for k in range(HANDOVER_MIN_STEPS, n):
        if _row_reach_mm(d, start + k) < handover_mm:
            return (start + k, True)
    var best = start
    var best_r = _row_reach_mm(d, start)
    for k in range(1, n):
        var rk = _row_reach_mm(d, start + k)
        if rk < best_r:
            best_r = rk
            best = start + k
    return (best, False)


def run_episode(
    mut env: E, mut ex: Expert, brick: Int, bowl: Int, place: Bool,
    ep: Int, verbose: Bool,
    agent: Optional[Pointer[Agent, MutAnyOrigin]] = None,
    handover_mm: Float64 = CLOSE_REACH_MM, policy_steps: Int = 140,
    handed_in: Int = -1,
) raises -> Bool:
    """One scripted pick (and place). Returns success.

    With `agent` (DAgger, `--policy`): the checkpoint drives first —
    `policy_approach` — and the expert takes over FROM THE POLICY'S OWN
    STATE, its rows flagged INTERVENED: the descent from wherever the jaw
    is (a short ramp if the policy arrived, the full pre + descent if it
    did not), then the close, the lift and the hold. Those rows are the
    ones no expert-only file holds — "close from HERE", where here is a
    state the policy reaches on its own (five runs parked on exactly that)."""
    ex.rec.begin()
    ex.steps = 0
    ex.rung_rows = 0
    ex.intervening = False
    var handed = False
    var done = False
    var policy_steps_used = 0
    if agent:
        var res = ex.policy_approach(env, agent.value()[], handover_mm, policy_steps)
        handed = res[0]
        done = res[1]
        policy_steps_used = ex.steps
        ex.intervening = True
    elif handed_in >= 0:
        # `--handover-from`: the env is already in the student's state
        handed = handed_in == 1
        ex.intervening = True
    for i in range(ACT):
        ex.q_cmd[i] = Float64(env.d.qpos.data[i])
    for i in range(N_ARM):
        ex.q_ref[i] = Float64(env.d.qpos.data[i])
        ex.sag_bias[i] = 0.0
    var from_reset = not agent and handed_in < 0
    if from_reset:
        for i in range(N_ARM):
            ex.q_home[i] = Float64(env.d.qpos.data[i])
    var pb = _body_pos(env, brick)
    var bearing = atan2(pb[1], pb[0])
    if ex.human_posture:
        ex.draw_posture()
    var yaw = ex.pinch_yaw(bearing, _body_yaw(env, brick))
    var tl = ex.tilt if ex.human_posture else 0.0
    var rs = ROLL_SEED_HUMAN if ex.human_posture else 0.0
    var use_rs = ex.human_posture
    var q = List[Float64]()
    for i in range(N_ARM):
        q.append(Float64(env.d.qpos.data[i]))
    var q1 = List[Float64](length=N_ARM, fill=0.0)
    var q2 = List[Float64](length=N_ARM, fill=0.0)
    var e1: Float64
    var e2: Float64
    # a tilted grasp approaches ALONG THE FINGER: IK targets on the line
    # through the tips' grasp point, `APPROACH_D` back along the finger axis
    var waypoints = List[List[Float64]]()
    ex.tip_trigger = tl > 0.0
    if tl > 0.0:
        var fx = sin(tl) * cos(bearing)
        var fy = sin(tl) * sin(bearing)
        var fzv = -cos(tl)
        ex.tip_goal[0] = pb[0]
        ex.tip_goal[1] = pb[1]
        ex.tip_goal[2] = pb[2] + ex.z_grasp - TIP_REACH
        # the IK's tip mode aims `target - (0, 0, TIP_REACH)`: hand it the
        # tip point lifted by TIP_REACH
        var n_wp = APPROACH_WAYPOINTS
        var prev = q.copy()
        e1 = 0.0
        e2 = 0.0
        for j in range(n_wp + 1):
            var sback = APPROACH_D * Float64(n_wp - j) / Float64(n_wp)
            var t = List[Float64]()
            t.append(ex.tip_goal[0] - sback * fx)
            t.append(ex.tip_goal[1] - sback * fy)
            t.append(ex.tip_goal[2] - sback * fzv + TIP_REACH)
            var qj = List[Float64](length=N_ARM, fill=0.0)
            var ej = ex.arm.ik(env, t, prev, yaw, qj, tl, rs, use_rs)
            if j == 0:
                e1 = ej
            e2 = ej      # the last waypoint IS the grasp target
            prev = qj.copy()
            waypoints.append(qj^)
        for i in range(N_ARM):
            q1[i] = waypoints[0][i]
            q2[i] = waypoints[n_wp][i]
    else:
        e1 = ex.arm.ik(env, _above(pb, Z_PRE), q, yaw, q1, tl, rs, use_rs)
        e2 = ex.arm.ik(env, _above(pb, ex.z_grasp), q1, yaw, q2, tl, rs, use_rs)
    var q3 = List[Float64](length=N_ARM, fill=0.0)
    var e3 = ex.arm.ik(env, _above(pb, Z_LIFT), q2, yaw, q3, tl, rs, use_rs)
    if verbose and ex.human_posture:
        print("  ep", ep, "posture: tilt", fixed(ex.tilt * 180.0 / pi, 1),
              "deg | pinch pair", "tangential" if ex.tangential else "radial",
              "| pinch from radial", fixed((yaw - bearing) * 180.0 / pi, 1), "deg")
    if verbose:
        print("  ep", ep, "brick", fixed(pb[0], 3), fixed(pb[1], 3),
              " ik err mm: pre", fixed(e1 * 1000.0, 1), "grasp",
              fixed(e2 * 1000.0, 1), "lift", fixed(e3 * 1000.0, 1))
    if verbose and agent:
        print("  ep", ep, " policy drove", policy_steps_used, "steps ->",
              "handover at reach " + fixed(ex.reach_mm(), 1) + " mm" if handed
              else "no arrival (cap), the expert does the full approach")
    if not done and not handed:
        done = ex.step_to(env, q1, True, N_PRE)
    if not done and len(waypoints) > 0 and not handed:
        # the tilted descent: along the finger, one leg per waypoint, the
        # close armed on the last
        var n_wp = len(waypoints) - 1
        var leg = N_DESCEND // n_wp
        for j in range(1, n_wp + 1):
            if done:
                break
            done = ex.step_to(
                env, waypoints[j], True, leg, taper_noise=True,
                close_on_height=j == n_wp,
            )
    elif not done:
        done = ex.step_to(
            env, q2, True, N_DESCEND_HANDOVER if handed else N_DESCEND,
            taper_noise=True, close_on_height=True,
        )
    if not done:
        # ⚠ THE CLOSE IS A STEP (`--close-steps`, default 1; it was 30,
        # then 15) — see `N_CLOSE` and `Z_GRASP`. A 30-step ramp labels the arrival state — arm settled at the
        # grasp pose, jaw open — with a command 1/30 of the range below
        # open, next to the approach's "open" at the same state: the fitted
        # policy's jaw command there is a hair below open, the jaw does not
        # move, the state does not change, and it waits for ever (probe,
        # runs dd8d4a64 and c40b4a8b: gripper word 1.8 rad off the
        # recording from step 90 on). Measured on 20 placements: 30 steps
        # 18/20, 15 steps 18/20, 8 steps 14/20, 4 steps 0/20, 1 step 13/20 —
        # the fast closes knock the cube. Fifteen doubles the label.
        done = ex.hold(env, False, ex.close_steps)
    if not done:
        done = ex.step_to(env, q3, False, N_LIFT)
    if place and not done:
        var pw = _body_pos(env, bowl)
        var q4 = List[Float64](length=N_ARM, fill=0.0)
        var e4 = ex.arm.ik(env, _above(pw, Z_CARRY), q3, yaw, q4, tl, rs, use_rs)
        var q5 = List[Float64](length=N_ARM, fill=0.0)
        var e5 = ex.arm.ik(env, _above(pw, Z_PLACE), q4, yaw, q5, tl, rs, use_rs)
        if verbose:
            print("  ep", ep, "bowl", fixed(pw[0], 3), fixed(pw[1], 3),
                  " ik err mm: carry", fixed(e4 * 1000.0, 1), "place",
                  fixed(e5 * 1000.0, 1))
        done = ex.step_to(env, q4, False, N_CARRY)
        if not done:
            done = ex.step_to(env, q5, False, N_PLACE)
        if not done:
            done = ex.hold(env, True, N_OPEN)
        if not done:
            done = ex.step_to(env, q4, True, N_RETREAT)
    if not done:
        done = ex.hold(env, not place, N_HOLD_MAX, until_held=True)
    if done and ex.return_rest and from_reset:
        # the fold back, then a short rest: recorded PAST the success test,
        # which already holds here (the brick is in the bowl)
        var home = ex.q_home.copy()
        _ = ex.step_to(env, home, False, N_RETURN_REST, past_success=True)
        _ = ex.hold(env, False, N_REST_HOLD, past_success=True)
    ex.intervening = False
    var brick_z = Float64(env.d.xpos.data[brick * 3 + 2])
    print(
        "  ep", ep, "->", "SUCCESS" if done else "failed", " steps", ex.steps,
        " return", fixed(ex.rec.ret, 1), " rows>1.5 (rung)", ex.rung_rows,
        " brick z", fixed(brick_z, 3),
    )
    return done


def _usage():
    print("usage: tower_expert_record.mojo [task] [--episodes N] [--seed S]"
          " [--noise SIGMA] [--flat-noise] [--close-steps N] [--z-grasp M] [--jaw-open RAD]\n"
          "       [--close-above-mm MM] [--feedback] [--out FILE]\n"
          "       [--policy CKPT [--handover-mm MM] [--policy-steps N]]   # DAgger\n"
          "       [--handover-from STUDENT.demo [--handover-mm MM] [--handover-settled]]"
          "   # DAgger from a recorded (vision) student\n"
          "       [--posture expert|human [--tilt-range LO,HI] [--pinch-range LO,HI]"
          " [--tip-close-mm MM]] [--return-rest]\n"
          "       [--keep-failures] [--quiet]")


def main() raises:
    var args = argv()
    var task = String(DEFAULT_TASK)
    var n_episodes = 20
    var seed0 = 0
    var noise = 0.0
    var feedback = False
    var flat_noise = False
    var close_steps = N_CLOSE
    var z_grasp = Z_GRASP
    var z_grasp_set = False
    var close_above_mm = Z_CLOSE_ABOVE_MM
    var jaw_open = -1.0
    var policy_ckpt = String("")
    var handover_mm = CLOSE_REACH_MM
    var policy_steps = 140
    var handover_from = String("")
    var posture = String("expert")
    var tilt_range = String("10,55")
    var pinch_range = String("35,85")
    var tip_close_mm = TIP_CLOSE_MM_DEFAULT
    var return_rest = False
    var handover_settled = False
    var out_path = String("")
    var keep_failures = False
    var verbose = True
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--episodes" and i + 1 < len(args):
            n_episodes = Int(String(args[i + 1]))
            i += 2
        elif a == "--seed" and i + 1 < len(args):
            seed0 = Int(String(args[i + 1]))
            i += 2
        elif a == "--noise" and i + 1 < len(args):
            noise = Float64(String(args[i + 1]))
            i += 2
        elif a == "--out" and i + 1 < len(args):
            out_path = String(args[i + 1])
            i += 2
        elif a == "--keep-failures":
            keep_failures = True
            i += 1
        elif a == "--quiet":
            verbose = False
            i += 1
        elif a == "--feedback":
            feedback = True
            i += 1
        elif a == "--flat-noise":
            flat_noise = True
            i += 1
        elif a == "--close-steps" and i + 1 < len(args):
            close_steps = Int(String(args[i + 1]))
            i += 2
        elif a == "--z-grasp" and i + 1 < len(args):
            z_grasp = Float64(String(args[i + 1]))
            z_grasp_set = True
            i += 2
        elif a == "--close-above-mm" and i + 1 < len(args):
            close_above_mm = Float64(String(args[i + 1]))
            i += 2
        elif a == "--jaw-open" and i + 1 < len(args):
            jaw_open = Float64(String(args[i + 1]))
            i += 2
        elif a == "--policy" and i + 1 < len(args):
            policy_ckpt = String(args[i + 1])
            i += 2
        elif a == "--handover-mm" and i + 1 < len(args):
            handover_mm = Float64(String(args[i + 1]))
            i += 2
        elif a == "--posture" and i + 1 < len(args):
            posture = String(args[i + 1])
            i += 2
        elif a == "--tilt-range" and i + 1 < len(args):
            tilt_range = String(args[i + 1])
            i += 2
        elif a == "--return-rest":
            return_rest = True
            i += 1
        elif a == "--tip-close-mm" and i + 1 < len(args):
            tip_close_mm = Float64(String(args[i + 1]))
            i += 2
        elif a == "--pinch-range" and i + 1 < len(args):
            pinch_range = String(args[i + 1])
            i += 2
        elif a == "--handover-settled":
            handover_settled = True
            i += 1
        elif a == "--handover-from" and i + 1 < len(args):
            handover_from = String(args[i + 1])
            i += 2
        elif a == "--policy-steps" and i + 1 < len(args):
            policy_steps = Int(String(args[i + 1]))
            i += 2
        elif a == "--help" or a == "-h":
            _usage()
            return
        elif a.startswith("--"):
            _usage()
            raise Error("unrecognised argument: " + a)
        else:
            task = a
            i += 1
    # cube_in_bowl and its variants (`_wide`: the real layouts' regions)
    var place = task.startswith("so101_tower_cube_in_bowl")
    if task != "so101_tower_lift_brick" and not place:
        raise Error("the expert knows so101_tower_lift_brick and so101_tower_cube_in_bowl*, not " + task)
    seed_rng(seed0)
    if out_path.byte_length() == 0:
        var stamp = iso8601_utc(epoch_seconds()).replace(":", "-")
        out_path = String(DEMO_DIR) + "/" + stamp + "_" + task + (
            "_dagger.demo" if policy_ckpt.byte_length() > 0
            or handover_from.byte_length() > 0 else "_expert.demo"
        )
    if not Path(DEMO_DIR).exists():
        _ = run_capture(String("mkdir -p ") + quote_arg(String(DEMO_DIR)), 4096)

    print("=" * 66)
    print("so101_tower —", task, "— SCRIPTED EXPERT (waypoints + IK)")
    print("=" * 66)
    print("  episodes", n_episodes, " seed", seed0, " noise", noise,
          " demonstrator", "feedback" if feedback else "ramps", " out", out_path)

    var ctx = DeviceContext()
    var env = E(ctx)
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DType.float64](cw[k])
    var mw = task_meta_words(
        task, String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    var brick = -1
    var bowl = -1
    for b in range(len(fmd.body_names)):
        if String(fmd.body_names[b]) == "brick_brick":
            brick = b
        if String(fmd.body_names[b]) == "bowl_bowl":
            bowl = b
    if brick < 0 or bowl < 0:
        raise Error("brick_brick / bowl_bowl not found in the composed scene")

    var ex = Expert(env, noise, feedback, out_path, keep_failures)
    ex.flat_noise = flat_noise
    ex.close_steps = close_steps
    ex.z_grasp = z_grasp
    ex.close_above_mm = close_above_mm
    if jaw_open > 0.0:
        ex.jaw_open = jaw_open
    if posture != "expert" and posture != "human":
        raise Error("--posture is expert or human, not " + posture)
    ex.human_posture = posture == "human"
    ex.integral = ex.human_posture
    if ex.human_posture:
        # the tilted grasp's own defaults (60-episode runs, seed 21000):
        # the jaw half as open as the vertical expert's (the operator's
        # opens 25-41 %; 0.9 rad hits the desk tilted, 5/20 against 14/20)
        # and the tips 5 mm higher (the operator grasps ~1 cm higher)
        if jaw_open <= 0.0:
            ex.jaw_open = HUMAN_JAW_OPEN
        if not z_grasp_set:
            ex.z_grasp = HUMAN_Z_GRASP
    var tr = tilt_range.split(",")
    if len(tr) != 2:
        raise Error("--tilt-range needs lo,hi in degrees, got " + tilt_range)
    ex.tilt_lo = Float64(String(tr[0])) * pi / 180.0
    ex.tilt_hi = Float64(String(tr[1])) * pi / 180.0
    var prr = pinch_range.split(",")
    if len(prr) != 2:
        raise Error("--pinch-range needs lo,hi in degrees, got " + pinch_range)
    ex.pinch_lo = Float64(String(prr[0])) * pi / 180.0
    ex.pinch_hi = Float64(String(prr[1])) * pi / 180.0
    ex.tip_close_mm = tip_close_mm
    ex.return_rest = return_rest
    if ex.human_posture:
        print("  posture  : HUMAN — tilt ~ U(", tilt_range, ") deg outward, pinch",
              "~ U(", pinch_range, ") deg from radial, snapped to the brick's faces")
    var agent: Agent = SAC["cpu", E.OBS_DIM, ACT, BATCH, CAP, HIDDEN](
        action_scale=1.0, learning_starts=0,
    )
    var agent_ptr = Optional[Pointer[Agent, MutAnyOrigin]](None)
    if policy_ckpt.byte_length() > 0:
        if not Path(policy_ckpt).exists():
            raise Error("--policy: no such checkpoint: " + policy_ckpt)
        agent.load(policy_ckpt)
        agent_ptr = Pointer(to=agent).as_unsafe_any_origin()
        print("  DAgger   : the checkpoint drives to its own arrival"
              " (handover under", handover_mm, "mm settled, cap",
              policy_steps, "steps); the expert closes and lifts from there,"
              " those rows flagged INTERVENED")
    if handover_from.byte_length() > 0:
        if policy_ckpt.byte_length() > 0:
            raise Error("--handover-from and --policy are two DAgger sources; pick one")
        if not Path(handover_from).exists():
            raise Error("--handover-from: no such file " + handover_from)
        var sd = read_demo_file(handover_from)
        if sd.obs_dim != E.OBS_DIM or sd.act_dim != ACT:
            raise Error(handover_from + ": obs " + String(sd.obs_dim) + " act "
                        + String(sd.act_dim) + ", this env is obs "
                        + String(E.OBS_DIM) + " act " + String(ACT))
        var n_src = len(sd.ep_len)
        if handover_settled:
            print("  DAgger   : handing over from", n_src, "recorded student episodes (",
                  handover_from, ") — the first SETTLED row under", handover_mm,
                  "mm, episodes without one SKIPPED; only the expert's rows are"
                  " written, INTERVENED")
        else:
            print("  DAgger   : handing over from", n_src, "recorded student episodes (",
                  handover_from, ") — the first row under", handover_mm, "mm (moving"
                  " or not), else the closest row; only the expert's rows are"
                  " written, INTERVENED")
        var n_skipped = 0
        var n_ok_h = 0
        var n_arrived = 0
        var n_src_ok = 0
        for ep in range(n_src):
            if n_episodes > 0 and ep >= n_episodes:
                break
            if sd.ep_success[ep]:
                n_src_ok += 1
            var hr = _handover_row(sd, ep, handover_mm, handover_settled)
            if hr[0] < 0:
                n_skipped += 1
                continue
            if hr[1]:
                n_arrived += 1
            _ = env.reset()
            for k in range(len(mw[0])):
                env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
            var qh = List[Float64](length=NQ, fill=0.0)
            var vh = List[Float64](length=NV, fill=0.0)
            for k in range(NQ):
                qh[k] = Float64(sd.obs[hr[0] * sd.obs_dim + k])
            for k in range(NV):
                vh[k] = Float64(sd.obs[hr[0] * sd.obs_dim + NQ + k])
            var sh = env.obs_at(qh, vh)
            for k in range(E.OBS_DIM):
                ex.obs[k] = Scalar[DT](sh.data[k])
            if verbose:
                var src_ok = sd.ep_success[ep]
                var h_row = hr[0] - sd.ep_start[ep]
                var src_len = sd.ep_len[ep]
                var h_reach = _row_reach_mm(sd, hr[0])
                var how = String("(arrival)") if hr[1] else String("(closest)")
                var st = String("succeeded") if src_ok else String("failed")
                print("  ep", ep, " student", st, "| handover at row", h_row,
                      "of", src_len, how, "reach", fixed(h_reach, 1), "mm")
            var ok_h = run_episode(
                env, ex, brick, bowl, place, ep, verbose, None, handover_mm,
                policy_steps, handed_in=1 if hr[1] else 0,
            )
            _ = ex.rec.end(success=ok_h)
            if ok_h:
                n_ok_h += 1
        print("-" * 66)
        print("  student episodes", n_src, "| the student succeeded in", n_src_ok,
              "| arrivals", n_arrived, "| skipped (no settled row)", n_skipped,
              "| the expert completed", n_ok_h, "->", out_path)
        print("  ", ex.rec.demos.summary())
        return

    var n_ok = 0
    for ep in range(n_episodes):
        _ = env.reset()
        for k in range(len(mw[0])):
            env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
        var q0 = posed_qpos[So101TowerPlacement](
            task, String(FAMILY), So101TowerConfig.SLOT_RADIUS,
            UInt64(seed0 + ep),
        )
        var v0 = List[Float64](length=NV, fill=0.0)
        var s0 = env.obs_at(q0, v0)
        for k in range(E.OBS_DIM):
            ex.obs[k] = Scalar[DT](s0.data[k])
        # let the props settle on the desk before reading their poses
        var zero = ContAction[ACT]()
        for k in range(ACT):
            zero.data[k] = ex._normalized(k, Float64(env.d.qpos.data[k]))
        for _ in range(5):
            var o = env.step(zero)
            for k in range(E.OBS_DIM):
                ex.obs[k] = Scalar[DT](o[0].data[k])
        var ok = run_episode(
            env, ex, brick, bowl, place, ep, verbose, agent_ptr, handover_mm,
            policy_steps,
        )
        _ = ex.rec.end(success=ok)
        if ok:
            n_ok += 1
    print("-" * 66)
    print("  ", n_ok, "of", n_episodes, "episodes succeeded ->", out_path)
    print("  ", ex.rec.demos.summary())
