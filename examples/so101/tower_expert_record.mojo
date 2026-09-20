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
within `--handover-mm` (25) of the brick with the arm settled, or for
`--policy-steps` (140); the expert then closes and lifts FROM THE POLICY'S
OWN STATE — a short descent to the grasp pose from wherever the jaw is, the
close, the lift, the hold — and those rows carry the INTERVENED flag. Five
BC runs and a pure-BC run parked at the arrival because no recorded row
says "close from here" for the states the policy reaches on its own; these
are those rows, HIL-SERL's human intervention with the expert as the human.
First smoke (checkpoint 69b67456, stiff jaws): handover in 7 of 12, lifts
in 11 of 12, 1113 of 2113 rows intervened.

⚠ THE IK SETS THE ARM'S qpos TO EVALUATE FK AND RESTORES THE STATE AFTER.
It never steps physics. The props' qpos are untouched.
"""

from std.math import sqrt, atan2, cos, sin, log, pi
from std.random import seed as seed_rng, random_float64
from std.sys import argv
from std.pathlib import Path

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.core.run import epoch_seconds, iso8601_utc
from mojo_rl.io.proc import quote_arg, run_capture
from mojo_rl.deep_agents.data.demo_file import DemoSet, write_demo_file
from mojo_rl.deep_agents.data.any_replay import AnyReplay
from mojo_rl.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from mojo_rl.deep_agents.training.blocks import ReplaySampleStep
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.math3d import Quat, Vec3
from max.gpu.host import DeviceContext
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, MODEL_CURRICULUM_SIZE,
)
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime
from mojo_rl.tasks.eval import region_sites, region_rects, region_half_heights
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from mojo_rl.tasks.gpu_eval import region_table_words
from mojo_rl.tasks.host_reward import family_reward_host
from mojo_rl.tasks.placement.so101_tower import So101TowerPlacement
from mojo_rl.tasks.posed_reset import posed_qpos, task_meta_words
from mojo_rl.tasks.so101_tower_xml import So101TowerModel
from mojo_rl.tasks.spec import load_family
from mojo_rl.utils.fmt import fixed

comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "mojo_rl/tasks/families/so101_tower.family"
comptime DEFAULT_TASK = "so101_tower_lift_brick"
comptime DEMO_DIR = "projects/so101-tower/demos"
comptime CFG = So101TowerTeleopConfig
comptime E = Phyics3dEnv[So101TowerModel, CFG, DType.float64, False]
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime NB = So101TowerModel.NBODY
comptime ACT = 6
comptime N_ARM = 5
comptime GB = So101TowerConfig.OBS_GOAL_BASE
"""The goal words: obs[GB+3..GB+5] is the jaw-to-brick vector (the reach)."""

# ── DAgger: the policy drives to its own arrival, the expert takes over ──
comptime HIDDEN = 256
"""⚠ MUST MATCH `sac_family_driver.HIDDEN` for `--policy` to load."""
comptime BATCH = 256
comptime CAP = 1000
comptime Agent = SACAgent[
    "cpu",
    ReplaySampleStep[AnyReplay["cpu", E.OBS_DIM, ACT, CAP], BATCH],
    SACActorNet[E.OBS_DIM, ACT, HIDDEN],
    SACCriticNet[E.OBS_DIM, ACT, HIDDEN],
]
comptime N_DESCEND_HANDOVER = 20
"""The expert's descent after a handover: the policy already brought the
jaw near the brick, so a short ramp to the grasp pose from wherever it is."""
comptime HANDOVER_SETTLED: Float64 = 0.3
"""rad/s: every arm joint slower than this counts as settled at the handover."""
comptime HANDOVER_MIN_STEPS = 20
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
        mut out: List[Float64],
    ):
        var p = List[Float64](length=3, fill=0.0)
        var fz = List[Float64](length=3, fill=0.0)
        var gx = List[Float64](length=3, fill=0.0)
        self.fk(env, q, p, fz, gx)
        out[0] = p[0]
        out[1] = p[1]
        out[2] = p[2]
        out[3] = fz[0]
        out[4] = fz[1]
        # sin of the yaw error between the pinch axis and the wanted axis
        out[5] = gx[0] * sin(yaw) - gx[1] * cos(yaw)

    def _ik_once(
        self, mut env: E, ref target: List[Float64], ref q0: List[Float64],
        yaw: Float64, w_tilt: Float64, w_yaw: Float64, iters: Int,
        mut q_out: List[Float64],
    ) -> Float64:
        """Damped least squares from `q0`; returns the position error."""
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
        var f = List[Float64](length=6, fill=0.0)
        var f2 = List[Float64](length=6, fill=0.0)
        var e = List[Float64](length=6, fill=0.0)
        var J = List[Float64](length=6 * N_ARM, fill=0.0)
        var H = List[Float64](length=N_ARM * N_ARM, fill=0.0)
        var g = List[Float64](length=N_ARM, fill=0.0)
        var perr = 1.0
        for _ in range(iters):
            self._feats(env, q, yaw, f)
            for r in range(6):
                var goal = target[r] if r < 3 else 0.0
                e[r] = (goal - f[r]) * W[r]
            perr = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
            var tilt = sqrt(f[3] * f[3] + f[4] * f[4])
            if perr < 5e-4 and tilt < 0.05:
                break
            var eps = 1e-6
            for j in range(N_ARM):
                var dq = List[Float64]()
                for i in range(N_ARM):
                    dq.append(q[i])
                dq[j] += eps
                self._feats(env, dq, yaw, f2)
                for r in range(6):
                    J[r * N_ARM + j] = (f2[r] - f[r]) * W[r] / eps
            # H = J^T J + lambda I,  g = J^T e
            for a in range(N_ARM):
                g[a] = 0.0
                for b in range(N_ARM):
                    var s = 0.0
                    for r in range(6):
                        s += J[r * N_ARM + a] * J[r * N_ARM + b]
                    H[a * N_ARM + b] = s + (1e-5 if a == b else 0.0)
                for r in range(6):
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
        yaw: Float64, mut q_out: List[Float64],
    ) -> Float64:
        """Restarts x a relaxing tilt weight; keeps the best position error.
        Restores the env's arm qpos to what it was."""
        var saved = List[Float64]()
        for i in range(NQ):
            saved.append(Float64(env.d.qpos.data[i]))
        var best = 1e9
        var q_try = List[Float64](length=N_ARM, fill=0.0)
        var weights: List[Float64] = [0.05, 0.02, 0.008, 0.003]
        var pan = atan2(target[1], target[0])
        for wi in range(len(weights)):
            for s in range(len(self.seeds) + 1):
                var s0 = List[Float64]()
                for i in range(N_ARM):
                    s0.append(q0[i] if s == 0 else self.seeds[s - 1][i])
                s0[0] = pan
                var err = self._ik_once(
                    env, target, s0, yaw, weights[wi], 0.02, 100, q_try
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
    var demos: DemoSet
    var noise: Float64
    var feedback: Bool
    var flat_noise: Bool
    var intervening: Bool
    """Rows recorded while True carry the INTERVENED flag (the expert
    driving after a `--policy` handover — HIL-SERL's human, scripted)."""
    var close_steps: Int
    var z_grasp: Float64
    """The gripper site's height above the brick centre at the grasp (`--z-grasp`, default `Z_GRASP`)."""
    var frame_skip: Int
    var timestep: Float64
    var q_cmd: List[Float64]
    var obs: List[Scalar[DT]]
    var prev_obs: List[Scalar[DT]]
    var act_l: List[Float64]
    var held: Int
    var steps: Int
    var ep_return: Float64
    var rung_rows: Int

    def __init__(
        out self, mut env: E, noise: Float64, feedback: Bool = False,
    ) raises:
        self.arm = Arm(env)
        self.demos = DemoSet(E.OBS_DIM, ACT)
        self.noise = noise
        self.feedback = feedback
        self.flat_noise = False
        self.intervening = False
        self.close_steps = N_CLOSE
        self.z_grasp = Z_GRASP
        self.frame_skip = CFG.FRAME_SKIP
        self.timestep = So101TowerModel.TIMESTEP
        self.q_cmd = List[Float64](length=ACT, fill=0.0)
        self.obs = List[Scalar[DT]](length=E.OBS_DIM, fill=Scalar[DT](0))
        self.prev_obs = List[Scalar[DT]](length=E.OBS_DIM, fill=Scalar[DT](0))
        self.act_l = List[Float64](length=ACT, fill=0.0)
        self.held = 0
        self.steps = 0
        self.ep_return = 0.0
        self.rung_rows = 0

    def _normalized(self, i: Int, q: Float64) -> Float64:
        var span = self.arm.hi[i] - self.arm.lo[i]
        var a = 2.0 * (q - self.arm.lo[i]) / span - 1.0 if span != 0.0 else 0.0
        if a > 1.0:
            a = 1.0
        if a < -1.0:
            a = -1.0
        return a

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
        self.demos.add(
            self.prev_obs, self.act_l, r, self.obs, 0.0, self.intervening
        )
        self.ep_return += r
        if r > 1.5:
            self.rung_rows += 1
        if rd[1]:
            self.held += 1
        else:
            self.held = 0
        return self.held >= HOLD_STEPS

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
        for k in range(max_steps):
            agent.select_greedy_action(self.obs, a32)
            for i in range(ACT):
                self.act_l[i] = Float64(a32[i])
            if self._apply(env):
                return (False, True)
            if k + 1 >= HANDOVER_MIN_STEPS and self.reach_mm() < handover_mm:
                var settled = True
                for i in range(N_ARM):
                    if abs(Float64(self.obs[NQ + i])) > HANDOVER_SETTLED:
                        settled = False
                if settled:
                    return (True, False)
        return (False, False)

    def step_to(
        mut self, mut env: E, ref q_target: List[Float64], grip_open: Bool,
        n_steps: Int, taper_noise: Bool = False, until_held: Bool = False,
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
        var g_target = self.arm.hi[5] if grip_open else self.arm.lo[5]
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
                    self.q_cmd[i] = q_start[i] + (q_target[i] - q_start[i]) * a
                self.q_cmd[5] = q_start[5] + (g_target - q_start[5]) * a
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
            if self._apply(env):
                return True
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
        until_held: Bool = False,
    ) raises -> Bool:
        var q = List[Float64]()
        for i in range(N_ARM):
            q.append(self.q_cmd[i])
        return self.step_to(env, q, grip_open, n_steps, until_held=until_held)


def _body_pos(mut env: E, b: Int) -> List[Float64]:
    var p = List[Float64]()
    for k in range(3):
        p.append(Float64(env.d.xpos.data[b * 3 + k]))
    return p^


def _above(ref p: List[Float64], dz: Float64) -> List[Float64]:
    var t = List[Float64]()
    t.append(p[0])
    t.append(p[1])
    t.append(p[2] + dz)
    return t^


def run_episode(
    mut env: E, mut ex: Expert, brick: Int, bowl: Int, place: Bool,
    ep: Int, verbose: Bool,
    agent: Optional[Pointer[Agent, MutAnyOrigin]] = None,
    handover_mm: Float64 = 25.0, policy_steps: Int = 140,
) raises -> Bool:
    """One scripted pick (and place). Returns success.

    With `agent` (DAgger, `--policy`): the checkpoint drives first —
    `policy_approach` — and the expert takes over FROM THE POLICY'S OWN
    STATE, its rows flagged INTERVENED: the descent from wherever the jaw
    is (a short ramp if the policy arrived, the full pre + descent if it
    did not), then the close, the lift and the hold. Those rows are the
    ones no expert-only file holds — "close from HERE", where here is a
    state the policy reaches on its own (five runs parked on exactly that)."""
    ex.demos.begin_episode()
    ex.held = 0
    ex.steps = 0
    ex.ep_return = 0.0
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
    for i in range(ACT):
        ex.q_cmd[i] = Float64(env.d.qpos.data[i])
    var pb = _body_pos(env, brick)
    var yaw = atan2(pb[1], pb[0])
    var q = List[Float64]()
    for i in range(N_ARM):
        q.append(Float64(env.d.qpos.data[i]))
    var q1 = List[Float64](length=N_ARM, fill=0.0)
    var e1 = ex.arm.ik(env, _above(pb, Z_PRE), q, yaw, q1)
    var q2 = List[Float64](length=N_ARM, fill=0.0)
    var e2 = ex.arm.ik(env, _above(pb, ex.z_grasp), q1, yaw, q2)
    var q3 = List[Float64](length=N_ARM, fill=0.0)
    var e3 = ex.arm.ik(env, _above(pb, Z_LIFT), q2, yaw, q3)
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
    if not done:
        done = ex.step_to(
            env, q2, True, N_DESCEND_HANDOVER if handed else N_DESCEND,
            taper_noise=True,
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
        _ = ex.arm.ik(env, _above(pw, Z_CARRY), q3, yaw, q4)
        var q5 = List[Float64](length=N_ARM, fill=0.0)
        _ = ex.arm.ik(env, _above(pw, Z_PLACE), q4, yaw, q5)
        done = ex.step_to(env, q4, False, N_CARRY)
        if not done:
            done = ex.step_to(env, q5, False, N_PLACE)
        if not done:
            done = ex.hold(env, True, N_OPEN)
        if not done:
            done = ex.step_to(env, q4, True, N_RETREAT)
    if not done:
        done = ex.hold(env, not place, N_HOLD_MAX, until_held=True)
    ex.intervening = False
    var brick_z = Float64(env.d.xpos.data[brick * 3 + 2])
    print(
        "  ep", ep, "->", "SUCCESS" if done else "failed", " steps", ex.steps,
        " return", fixed(ex.ep_return, 1), " rows>1.5 (rung)", ex.rung_rows,
        " brick z", fixed(brick_z, 3),
    )
    return done


def _usage():
    print("usage: tower_expert_record.mojo [task] [--episodes N] [--seed S]"
          " [--noise SIGMA] [--flat-noise] [--close-steps N] [--z-grasp M] [--feedback] [--out FILE]\n"
          "       [--policy CKPT [--handover-mm MM] [--policy-steps N]]   # DAgger\n"
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
    var policy_ckpt = String("")
    var handover_mm = 25.0
    var policy_steps = 140
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
            i += 2
        elif a == "--policy" and i + 1 < len(args):
            policy_ckpt = String(args[i + 1])
            i += 2
        elif a == "--handover-mm" and i + 1 < len(args):
            handover_mm = Float64(String(args[i + 1]))
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
    var place = task == "so101_tower_cube_in_bowl"
    if task != "so101_tower_lift_brick" and not place:
        raise Error("the expert knows so101_tower_lift_brick and so101_tower_cube_in_bowl, not " + task)
    seed_rng(seed0)
    if out_path.byte_length() == 0:
        var stamp = iso8601_utc(epoch_seconds()).replace(":", "-")
        out_path = String(DEMO_DIR) + "/" + stamp + "_" + task + (
            "_dagger.demo" if policy_ckpt.byte_length() > 0 else "_expert.demo"
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

    var ex = Expert(env, noise, feedback)
    ex.flat_noise = flat_noise
    ex.close_steps = close_steps
    ex.z_grasp = z_grasp
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
        if ok or keep_failures:
            ex.demos.end_episode(success=ok)
            write_demo_file(out_path, ex.demos)
        else:
            ex.demos.discard_episode()
        if ok:
            n_ok += 1
    print("-" * 66)
    print("  ", n_ok, "of", n_episodes, "episodes succeeded ->", out_path)
    print("  ", ex.demos.summary())
