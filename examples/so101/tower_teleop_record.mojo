"""Record SO-101 tower DEMONSTRATIONS in the sim — the leader arm drives, the
family's own reward is paid, every transition lands in a `.demo` file the SAC
driver loads with `--demos` (HIL-SERL / RLPD).

    pixi run build-imgui && pixi run build-serial                         # ONCE
    pixi run soarm-tower-record                                           # lift_brick, leader drives
    pixi run soarm-tower-record -- so101_tower_cube_in_bowl
    pixi run soarm-tower-record -- so101_tower_lift_brick \\
        --policy projects/so101-tower/runs/<run>/checkpoints/last.ckpt      # HIL: the policy drives, `i` intervenes
    pixi run soarm-tower-record -- --no-arm --policy <ckpt>                # no leader: policy only (a pipeline check)

then train on what was recorded:

    pixi run -e apple mojo run -I . examples/tasks/sac_tower_gpu.mojo so101_tower_lift_brick \\
        --steps 200000 --warmup 1000 --demos projects/so101-tower/demos/<file>.demo

## THE PROTOCOL (what the keys do)

  * The episode starts with the arm folded at HOME and the brick placed by
    the task's own sampler (`posed_reset`, seed 0 — the same placement the
    policy viewer shows). Move the leader: the sim arm follows.
  * **Success ends the episode by itself**: when the goal predicate has held
    for `HOLD_STEPS_SUCCESS` consecutive steps (1 s) the episode is saved as
    a SUCCESS and the scene resets. For `lift_brick` that is "brick above the
    desk by the margin, held for a second".
  * **`n`** ends the episode NOW and keeps it (a success only if the goal was
    holding at that moment). **`x`** discards the attempt and resets.
  * The task's own horizon (300 steps, 10 s) also ends an episode — a
    timeout is a FAILURE and is dropped unless `--keep-failures`.
  * With `--policy CKPT` the CHECKPOINT drives (greedy) and **`i`** toggles
    the human in: while intervening, the leader's pose is the action and the
    rows are flagged INTERVENED — HIL-SERL's `intervene_action`. `--demo-filter
    intervened` on the driver trains on just those rows.
  * The file is rewritten after every kept episode, so a crash loses at most
    the episode in flight. Default path:
    `projects/so101-tower/demos/<utc-time>_<task>.demo`; `--out` overrides.

⚠ THE FIRST STEP AFTER A RESET SWINGS THE ARM from HOME to wherever the leader
is — the reset hook cannot see the leader (`teleop_sim.mojo` says why). Park
the leader near the folded pose before pressing `n`, and the swing is small.

## WHAT IS RECORDED, AND WHY IT MATCHES TRAINING

Per step: `(obs_t, a_t, r_t, obs_{t+1}, done=0)`, the replay's own tuple.

  * `obs` is the CPU env's observation — `custom_extract_obs_cpu`, gated
    word-for-word against the GPU hook by `tests/tasks/test_active_mask.mojo`.
  * `a_t` is the NORMALIZED action the batched env takes: the leader's joint
    angle mapped onto each actuator's `ctrlrange` as `[-1, 1]`
    (`NORMALIZED_ACTIONS`), the inverse of what `apply_actions` does. A
    policy trained on these rows commands the same thing the human did.
  * `r_t` is `So101TowerConfig.compute_reward_and_done_gpu` evaluated ON THE
    HOST over the env's `Data` (`tasks/host_reward.mojo`) — the kernel's own
    function, not a transcription: the shaped goal term, the reach term, the
    grasp rung (contacts) and the closing bonus, at the run's default weights.
    ⚠ A driver launched with non-default `--shape-*` / `--*-margin` pays a
    different reward from the file's; record with the same flags
    (`--shape-goal` etc. here) or accept the mismatch knowingly.
  * `done` is 0 throughout, as online rows are (`TERMINATE_ON_UNHEALTHY=False`).

The viewer is paced at the env's control period (32 ms = 16 substeps x 2 ms)
so the sim runs at REAL TIME under the human's hand; a demo's tempo is the
tempo the policy will be asked to reproduce.

⚠ RUN FROM THE REPO ROOT (mesh paths), ON THE LAPTOP (it opens a window).
`1` wrist camera, `2` overhead, free camera by default.
"""

from std.random import seed
from std.sys import argv
from std.pathlib import Path

from mojo_rl.nn.constants import DT
from mojo_rl.core.run import epoch_seconds, iso8601_utc
from mojo_rl.io.proc import quote_arg, run_capture
from mojo_rl.deep_agents.data.any_replay import AnyReplay
from mojo_rl.deep_agents.data.demo_file import DemoSet, write_demo_file
from mojo_rl.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from mojo_rl.deep_agents.training.blocks import ReplaySampleStep
from mojo_rl.envs.dm_control.viewer_core import (
    ActionSource, StepObserver, DRIVE_POLICY, ViewerState, run_view,
    task_index,
)
from mojo_rl.physics3d.fields import Data, Model, DimsLike, actuator_column
from mojo_rl.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, MODEL_CURRICULUM_SIZE,
)
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.robot.so101.ports import leader_port, port_refusal
from mojo_rl.robot.so101.sim_map import SimJointMap
from mojo_rl.tasks.eval import region_sites, region_rects, region_half_heights
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.family_config import So101TowerConfig
from mojo_rl.tasks.gpu_eval import region_table_words
from mojo_rl.tasks.host_reward import family_reward_host
from mojo_rl.tasks.placement.so101_tower import So101TowerPlacement
from mojo_rl.tasks.posed_reset import posed_qpos, task_meta_words
from mojo_rl.tasks.so101_tower_xml import So101TowerModel
from mojo_rl.tasks.spec import load_family
from mojo_rl.utils.fmt import fixed

comptime SEED: Int = 0
comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "mojo_rl/tasks/families/so101_tower.family"
comptime DEFAULT_TASK = "so101_tower_lift_brick"
comptime DEMO_DIR = "projects/so101-tower/demos"

comptime OBS_DIM = So101TowerModel.OBS_DIM
comptime ACT_DIM = 6
comptime HIDDEN = 256
"""⚠ MUST MATCH `sac_family_driver.HIDDEN` for `--policy` to load."""
comptime BATCH = 256
comptime CAP = 1000
comptime ACTION_SCALE = 1.0
comptime GOAL_BASE = So101TowerConfig.OBS_GOAL_BASE
comptime REACH_RADIUS_MM = So101TowerConfig.REACH_RADIUS * 1000.0

comptime KEY_I: Int = 105
"""`i`: toggle the human in over a `--policy` (SDL keycode = ASCII)."""
comptime KEY_X: Int = 120
"""`x`: discard the episode in flight and reset."""
comptime HOLD_STEPS_SUCCESS: Int = 31
"""The goal must HOLD this many consecutive steps (1 s at 31.25 Hz) before an
episode ends as a success — a brick that flashes through the band while
being knocked over is not a lift."""
comptime CONTROL_PERIOD_MS: Int = 32
"""16 substeps x 2 ms: the env's control period, and the viewer's frame
target here so the sim runs at real time under the leader."""


def task_names() -> List[String]:
    var t = List[String]()
    t.append(String("so101_tower_lift_brick"))
    t.append(String("so101_tower_cube_in_bowl"))
    t.append(String("so101_tower_reach_clear"))
    return t^


struct TowerTeleop(ActionSource, StepObserver, Movable):
    """The leader arm as the action (or a checkpoint with the leader as the
    override), AND the recorder of what happened — one object on both of
    `run_view`'s seams, so the intervention toggle and the rows it flags
    share one state.
    """

    # ── the human ──
    var arm: Optional[SO101Arm]
    var map: SimJointMap
    var _raw: Array[Int32, SO101_N]
    var _last_ok: Int
    var lo: Array[Float64, SO101_N]
    var hi: Array[Float64, SO101_N]
    var last_action: List[Scalar[DT]]

    # ── the policy (HIL mode) ──
    var agent: SACAgent[
        "cpu",
        ReplaySampleStep[AnyReplay["cpu", OBS_DIM, ACT_DIM, CAP], BATCH],
        SACActorNet[OBS_DIM, ACT_DIM, HIDDEN],
        SACCriticNet[OBS_DIM, ACT_DIM, HIDDEN],
    ]
    var have_policy: Bool
    var policy_label: String
    var intervening: Bool

    # ── the recording ──
    var demos: DemoSet
    var out_path: String
    var keep_failures: Bool
    var frame_skip: Int
    var timestep: Float64
    var held_steps: Int
    var ep_success: Bool
    var discard: Bool
    var ep_return: Float64
    var ep_rows: Int
    var ep_intervened: Int
    var last_reward: Float64
    var reach_mm: Float64
    var goal_mm: Float64
    var n_saved: Int
    var n_dropped: Int

    def __init__(
        out self,
        port: String,
        policy_ckpt: String,
        out_path: String,
        keep_failures: Bool,
        frame_skip: Int,
        timestep: Float64,
    ) raises:
        # The TOWER scene's actuator ranges — the action space the batched
        # env normalises onto. Same joint order as the bare arm (the SO-101
        # is the first attached model), so `SimJointMap` indexes line up.
        var sf = So101TowerModel.make_spec_fields[DType.float64]()
        var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, SO101_N)
        var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, SO101_N)
        self.lo = Array[Float64, SO101_N](fill=0.0)
        self.hi = Array[Float64, SO101_N](fill=0.0)
        var lo2 = Array[Float64, SO101_N](fill=0.0)
        var hi2 = Array[Float64, SO101_N](fill=0.0)
        for i in range(SO101_N):
            self.lo[i] = Float64(lo_col[i])
            self.hi[i] = Float64(hi_col[i])
            lo2[i] = Float64(lo_col[i])
            hi2[i] = Float64(hi_col[i])
        self.map = SimJointMap.identity(lo2^, hi2^)
        self._raw = Array[Int32, SO101_N](fill=0)
        self._last_ok = -1
        self.last_action = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0))
        self.arm = None
        if port.byte_length() > 0:
            # Read-only: torque off so the leader is backdriven by hand;
            # nothing here ever writes a goal to a servo.
            var a = SO101Arm(port, max_step_ticks=0)
            a.bus.timeout_ms = 20
            a.set_torque(False)
            self.arm = a^

        self.agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, CAP, HIDDEN](
            action_scale=ACTION_SCALE, learning_starts=0,
        )
        self.have_policy = False
        self.policy_label = String("")
        self.intervening = False
        if policy_ckpt.byte_length() > 0:
            if not Path(policy_ckpt).exists():
                raise Error("--policy: no such checkpoint: " + policy_ckpt)
            self.agent.load(policy_ckpt)
            self.have_policy = True
            self.policy_label = policy_ckpt
            print("  policy loaded:", policy_ckpt, "— press `i` to intervene")

        self.demos = DemoSet(OBS_DIM, ACT_DIM)
        self.out_path = out_path
        self.keep_failures = keep_failures
        self.frame_skip = frame_skip
        self.timestep = timestep
        self.held_steps = 0
        self.ep_success = False
        self.discard = False
        self.ep_return = 0.0
        self.ep_rows = 0
        self.ep_intervened = 0
        self.last_reward = 0.0
        self.reach_mm = -1.0
        self.goal_mm = -1.0
        self.n_saved = 0
        self.n_dropped = 0
        self.demos.begin_episode()

    # ── ActionSource ─────────────────────────────────────────────────────

    def obs_dim(self) -> Int:
        return OBS_DIM

    def act_dim(self) -> Int:
        return ACT_DIM

    def variant_labels(self) -> List[String]:
        var v = List[String]()
        if self.have_policy:
            v.append(String("policy drives; `i` = human intervenes"))
        elif self.arm:
            v.append(String("leader arm (live)"))
        else:
            v.append(String("no arm, no policy — use the drive modes"))
        return v^

    def choose(mut self, i: Int) raises:
        pass

    def status(self) -> String:
        var out = String("")
        if self.have_policy:
            out += String("HUMAN ") if self.intervening else String("policy ")
        if self.arm:
            if self._last_ok < 0:
                out += "leader: waiting for the first read  "
            elif self._last_ok != SO101_N:
                out += "leader: " + String(self._last_ok) + "/6 answered  "
        out += (
            "r " + fixed(self.last_reward, 3) + "  reach "
            + fixed(self.reach_mm, 0) + "/" + fixed(REACH_RADIUS_MM, 0)
            + "mm  goal " + fixed(self.goal_mm, 0) + "mm  held "
            + String(self.held_steps) + "/" + String(HOLD_STEPS_SUCCESS)
            + "  ep " + String(self.ep_rows) + " rows R=" + fixed(self.ep_return, 1)
            + "  saved " + String(self.n_saved) + " dropped " + String(self.n_dropped)
        )
        return out^

    def _leader_action(mut self, mut action_out: List[Scalar[DT]]) raises:
        """Leader ticks -> tower-model radians -> the normalised action."""
        if not self.arm:
            for j in range(ACT_DIM):
                action_out[j] = self.last_action[j]
            return
        var n = self.arm.value().read_positions(Span(self._raw))
        self._last_ok = n
        if n != SO101_N:
            # Hold the last command rather than a half-updated pose.
            for j in range(ACT_DIM):
                action_out[j] = self.last_action[j]
            return
        for i in range(SO101_N):
            var q = self.map.to_sim(self.arm.value().cal, i, self._raw[i])
            var span = self.hi[i] - self.lo[i]
            var a = 2.0 * (q - self.lo[i]) / span - 1.0 if span != 0.0 else 0.0
            if a > 1.0:
                a = 1.0
            if a < -1.0:
                a = -1.0
            action_out[i] = Scalar[DT](a)
            self.last_action[i] = Scalar[DT](a)

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        if self.have_policy and not self.intervening:
            self.agent.select_greedy_action(obs, action_out)
            # Keep the leader read alive so the handover is instant and the
            # status line keeps reporting the bus.
            if self.arm:
                var tmp = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0))
                self._leader_action(tmp)
            return
        self._leader_action(action_out)

    # ── StepObserver ─────────────────────────────────────────────────────

    def on_step[DTYPE: DType, D: DimsLike](
        mut self,
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
        ref prev_obs: List[Scalar[DT]],
        ref action: List[Float64],
        ref obs: List[Scalar[DT]],
        done: Bool,
        step_i: Int,
        key: Int,
    ) raises -> Bool:
        if key == KEY_X:
            self.discard = True
            print("  `x`: discarding this episode")
            return True
        if key == KEY_I:
            if self.have_policy:
                self.intervening = not self.intervening
                print("  `i`:", "HUMAN intervening" if self.intervening
                      else "policy driving")
            else:
                print("  `i` does nothing without --policy")

        # THE reward — the kernel's function on the host (see the header).
        var rd = family_reward_host[So101TowerConfig, DTYPE, D, ACT_DIM](
            d, mf, action, step_i, self.frame_skip, self.timestep
        )
        var reward = Float64(rd[0])
        var holds = rd[1]
        var intervened = self.have_policy and self.intervening
        # `done` stays 0: the driver runs TERMINATE_ON_UNHEALTHY=False, so an
        # online row never carries a terminal either. `done` from the CPU env
        # here is only the horizon, which is a truncation.
        self.demos.add(prev_obs, action, reward, obs, 0.0, intervened=intervened)
        self.ep_rows += 1
        if intervened:
            self.ep_intervened += 1
        self.ep_return += reward
        self.last_reward = reward
        _ = done

        # the status distances, from the observation's goal words
        var rx = Float64(obs[GOAL_BASE + 3])
        var ry = Float64(obs[GOAL_BASE + 4])
        var rz = Float64(obs[GOAL_BASE + 5])
        self.reach_mm = ((rx * rx + ry * ry + rz * rz) ** 0.5) * 1000.0
        var gx = Float64(obs[GOAL_BASE + 6])
        var gy = Float64(obs[GOAL_BASE + 7])
        var gz = Float64(obs[GOAL_BASE + 8])
        self.goal_mm = ((gx * gx + gy * gy + gz * gz) ** 0.5) * 1000.0

        if holds:
            self.held_steps += 1
        else:
            self.held_steps = 0
        if self.held_steps >= HOLD_STEPS_SUCCESS:
            self.ep_success = True
            return True     # success: end the episode now
        return False

    def on_episode_end(mut self, step_i: Int, manual: Bool) raises:
        # `n` while the goal is holding counts; a timeout with it holding
        # (the horizon fell inside the hold window) counts too.
        if self.held_steps > 0:
            self.ep_success = True
        var keep = (not self.discard) and (self.ep_success or self.keep_failures)
        if keep and self.ep_rows > 0:
            self.demos.end_episode(success=self.ep_success)
            write_demo_file(self.out_path, self.demos)
            self.n_saved += 1
            print(
                "  SAVED episode", self.n_saved, "—", self.ep_rows, "rows,",
                "SUCCESS" if self.ep_success else "failure (kept)",
                ", return", fixed(self.ep_return, 2),
                ", intervened rows", self.ep_intervened,
                "->", self.out_path, "(" + self.demos.summary() + ")",
            )
        else:
            self.demos.discard_episode()
            self.n_dropped += 1
            print(
                "  dropped episode (", self.ep_rows, "rows,",
                "discarded" if self.discard else "no success", ")",
            )
        self.demos.begin_episode()
        self.held_steps = 0
        self.ep_success = False
        self.discard = False
        self.ep_return = 0.0
        self.ep_rows = 0
        self.ep_intervened = 0
        _ = step_i
        _ = manual


def _usage():
    print("usage: tower_teleop_record.mojo [task] [--policy CKPT] [--out FILE]")
    print("           [--keep-failures] [--no-arm] [--port /dev/...]")
    print("           [--shape-goal W] [--shape-reach W] [--goal-margin M] [--reach-margin M]")
    print("  tasks:")
    for n in task_names():
        print("    ", n)


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return
    var args = argv()
    var task = String(DEFAULT_TASK)
    var policy_ckpt = String("")
    var out_path = String("")
    var keep_failures = False
    var no_arm = False
    var port_cli = String("")
    var w_goal = So101TowerConfig.SHAPE_W_GOAL
    var w_reach = So101TowerConfig.SHAPE_W_REACH
    var m_goal = So101TowerConfig.GOAL_MARGIN
    var m_reach = So101TowerConfig.REACH_MARGIN
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--policy" and i + 1 < len(args):
            policy_ckpt = String(args[i + 1])
            i += 2
        elif a == "--out" and i + 1 < len(args):
            out_path = String(args[i + 1])
            i += 2
        elif a == "--port" and i + 1 < len(args):
            port_cli = String(args[i + 1])
            i += 2
        elif a == "--shape-goal" and i + 1 < len(args):
            w_goal = Float64(String(args[i + 1]))
            i += 2
        elif a == "--shape-reach" and i + 1 < len(args):
            w_reach = Float64(String(args[i + 1]))
            i += 2
        elif a == "--goal-margin" and i + 1 < len(args):
            m_goal = Float64(String(args[i + 1]))
            i += 2
        elif a == "--reach-margin" and i + 1 < len(args):
            m_reach = Float64(String(args[i + 1]))
            i += 2
        elif a == "--keep-failures":
            keep_failures = True
            i += 1
        elif a == "--no-arm":
            no_arm = True
            i += 1
        elif a == "--help" or a == "-h":
            _usage()
            return
        elif a.startswith("--"):
            _usage()
            raise Error("unrecognised argument: " + a)
        else:
            task = a
            i += 1
    var ti = task_index(task, task_names())
    if ti < 0:
        _usage()
        raise Error("unknown task: " + task)

    var port = String("")
    if not no_arm:
        port = leader_port(port_cli)
        var why = port_refusal(port, String("leader"))
        if why.byte_length() > 0:
            raise Error(
                "tower_teleop_record: " + why
                + "  (pass --no-arm to run without the leader)"
            )
        print("opening leader:", port)
    if out_path.byte_length() == 0:
        var stamp = iso8601_utc(epoch_seconds()).replace(":", "-")
        out_path = String(DEMO_DIR) + "/" + stamp + "_" + task + ".demo"
    if not Path(DEMO_DIR).exists():
        _ = run_capture(String("mkdir -p ") + quote_arg(String(DEMO_DIR)), 4096)

    print("=" * 66)
    print("so101_tower —", task, "— DEMONSTRATION RECORDER")
    print("=" * 66)
    print("  out          :", out_path)
    print("  keep failures:", keep_failures)
    print("  reward       : the family's GPU hook on the host, weights",
          w_goal, "/", w_reach, " margins", m_goal, "/", m_reach)
    print("  keys         : `n` end+keep   `x` discard   `i` intervene (with --policy)")
    print("                 success = goal held", HOLD_STEPS_SUCCESS,
          "steps -> saved automatically")
    print("=" * 66)

    var src = TowerTeleop(
        port, policy_ckpt, out_path, keep_failures,
        So101TowerConfig.FRAME_SKIP, So101TowerModel.TIMESTEP,
    )
    if src.arm:
        print("\n" + src.map.range_report(src.arm.value().cal))

    # The region table — what the SAC driver uploads to `curriculum`. The
    # tower's goals do not read it, but the reward hook is generic over the
    # language and a viewer that omitted it would evaluate a different table.
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )

    var domains = List[String]()
    domains.append(String(FAMILY))
    var td = List[Int]()
    for _ in range(len(task_names())):
        td.append(0)
    var st = ViewerState(ti, DRIVE_POLICY, 1.0, task_names(), domains^, td^)
    st.policy_variant = 0
    st.free_camera = True
    st.episode_steps = 0        # the task's own horizon is the only limit
    st.frame_ms = CONTROL_PERIOD_MS
    st.reset_curriculum = List[Float64]()
    for k in range(MODEL_CURRICULUM_SIZE):
        st.reset_curriculum.append(cw[k])
    var pol = Pointer(to=src).as_unsafe_any_origin()
    var obs = Pointer(to=src).as_unsafe_any_origin()
    while not st.quit:
        var name = task_names()[st.task]
        st.reset_qpos = posed_qpos[So101TowerPlacement](
            name, String(FAMILY), So101TowerConfig.SLOT_RADIUS
        )
        var mw = task_meta_words(
            name, String(FAMILY), w_goal, w_reach, m_goal, m_reach,
        )
        st.reset_meta_idx = mw[0].copy()
        st.reset_meta_val = mw[1].copy()
        run_view[So101TowerModel, So101TowerConfig, TowerTeleop, TowerTeleop](
            name, st, pol, obs
        )
    _ = src

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
    print("recorder closed:", src.demos.summary(), "->", out_path)
