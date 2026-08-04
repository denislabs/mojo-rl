"""Interactive dm_control viewer — the whole tool, minus argv parsing.

    from mojo_rl.envs.dm_control.viewer import run_viewer, task_index
    run_viewer(task, drive, scale)

`examples/dm_control/dm_viewer_imgui.mojo` is a ~40-line wrapper over this.
The logic lives here so it can be precompiled with the package
(`pixi run build`) instead of being re-elaborated by every build of the
example, and so a second front end (a training monitor, a headless recorder)
can reuse it without copying 600 lines.

⚠ REQUIRES THE ImGui SHIM: `pixi run build-imgui`. `imgui_shim_available()`
answers that without touching FFI; `run_viewer` checks it up front, because
the loader ABORTS the process rather than raising.

COMPILE COST LIVES IN `dispatch`. Each task is a distinct compile-time
`Phyics3dEnv[MODEL, CONFIG]`, and `dispatch` is the one place all 43 are
named — so build time is roughly proportional to its arm count and to how
much code `run_view` carries PER ARM. That is why the sidebar is built by
`build_sidebar`, which is NOT generic: it takes plain data in and returns
requests out, so 43 instantiations share one copy of the widget code instead
of stamping out 43. Keep it that way when adding features — anything that can
be phrased over plain values belongs outside `run_view`.

⚠ DOG IS DELIBERATELY ABSENT. It is still being ported in parallel; adding it
here would couple this file to a moving target.

WHAT THIS IS FOR. It answers "is the model built and posed the way I think" —
geometry, joint axes, ranges, the reset pose, whether anything falls through
the floor. It does NOT check parity with MuJoCo; `tests/dm_control/` does
that, and a model can look perfect while its dynamics differ.

⚠ ACTIONS ARE NOT A POLICY. The drive modes move every joint so you can see it
articulate. A tumbling humanoid under random torque is the expected picture.
"""

from std.random import random_float64
from std.math import sin, pi, min, max
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_env import Phyics3dEnv, Phyics3dEnvConfig
from mojo_rl.physics3d.model import ModelDefLike
from mojo_rl.render.imgui import (
    imgui_shim_available,
    ig_begin_panel, ig_end, ig_begin_child, ig_end_child,
    ig_text, ig_text_colored, ig_text_disabled,
    ig_separator_text, ig_same_line, ig_spacing,
    ig_button, ig_toggle_button, ig_selectable,
    ig_slider_float, ig_combo, ig_input_text, ig_tree_node, ig_tree_pop,
    ig_set_next_item_width, ig_plot_lines,
    ig_push_id_int, ig_pop_id, ig_style_dark, ig_content_width,
    TextBuffer,
)

from mojo_rl.envs.dm_control.acrobot.acrobot_xml import DMAcrobotModel
from mojo_rl.envs.dm_control.acrobot.acrobot_config import DMAcrobotConfig
from mojo_rl.envs.dm_control.ball_in_cup.ball_in_cup_xml import DMBallInCupModel
from mojo_rl.envs.dm_control.ball_in_cup.ball_in_cup_config import (
    DMBallInCupConfig,
)
from mojo_rl.envs.dm_control.cartpole.cartpole_xml import (
    DMCartpole1Model, DMCartpole2Model, DMCartpole3Model,
)
from mojo_rl.envs.dm_control.cartpole.cartpole_config import DMCartpoleConfig
from mojo_rl.envs.dm_control.cheetah.cheetah_xml import DMCheetahModel
from mojo_rl.envs.dm_control.cheetah.cheetah_config import DMCheetahConfig
from mojo_rl.envs.dm_control.finger.finger_xml import (
    DMFingerSpinModel, DMFingerTurnModel,
)
from mojo_rl.envs.dm_control.finger.finger_config import (
    DMFingerSpinConfig, DMFingerTurnConfig,
)
from mojo_rl.envs.dm_control.fish.fish_xml import (
    DMFishSwimModel, DMFishUprightModel,
)
from mojo_rl.envs.dm_control.fish.fish_config import (
    DMFishSwimConfig, DMFishUprightConfig,
)
from mojo_rl.envs.dm_control.hopper.hopper_xml import DMHopperModel
from mojo_rl.envs.dm_control.hopper.hopper_config import DMHopperConfig
from mojo_rl.envs.dm_control.humanoid.humanoid_xml import (
    DMHumanoidModel, DMHumanoidPureModel,
)
from mojo_rl.envs.dm_control.humanoid.humanoid_config import (
    DMHumanoidConfig, WALK_SPEED, RUN_SPEED,
)
from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_xml import (
    DMHumanoidCMUModel,
)
# ⚠ ALIASED, NOT IMPORTED BARE. `humanoid_cmu_config` defines its OWN
# WALK_SPEED/RUN_SPEED, which would collide with `humanoid_config`'s. They
# happen to hold the same values today (1.0 / 10.0) — which is exactly why a
# bare import would be a silent trap rather than a compile error if one domain
# later retunes its speeds.
from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_config import (
    DMHumanoidCMUConfig,
    WALK_SPEED as CMU_WALK_SPEED,
    RUN_SPEED as CMU_RUN_SPEED,
)
from mojo_rl.envs.dm_control.manipulator.manipulator_xml import (
    DMManipulatorBringBallModel, DMManipulatorBringPegModel,
    DMManipulatorInsertBallModel, DMManipulatorInsertPegModel,
)
from mojo_rl.envs.dm_control.manipulator.manipulator_config import (
    DMManipulatorBringBallConfig, DMManipulatorBringPegConfig,
    DMManipulatorInsertBallConfig, DMManipulatorInsertPegConfig,
)
from mojo_rl.envs.dm_control.pendulum.pendulum_xml import DMPendulumModel
from mojo_rl.envs.dm_control.pendulum.pendulum_config import DMPendulumConfig
from mojo_rl.envs.dm_control.point_mass.point_mass_xml import DMPointMassModel
from mojo_rl.envs.dm_control.point_mass.point_mass_config import (
    DMPointMassConfig,
)
from mojo_rl.envs.dm_control.point_mass.point_mass_hard_config import (
    DMPointMassHardConfig,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import (
    DMQuadrupedWalkModel, DMQuadrupedRunModel, DMQuadrupedFetchModel,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_config import (
    DMQuadrupedWalkConfig, DMQuadrupedRunConfig,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_fetch_config import (
    DMQuadrupedFetchConfig,
)
from mojo_rl.envs.dm_control.reacher.reacher_xml import DMReacherModel
from mojo_rl.envs.dm_control.reacher.reacher_config import DMReacherConfig
from mojo_rl.envs.dm_control.stacker.stacker_xml import (
    DMStacker2Model, DMStacker4Model,
)
from mojo_rl.envs.dm_control.stacker.stacker_config import (
    DMStacker2Config, DMStacker4Config,
)
from mojo_rl.envs.dm_control.swimmer.swimmer_xml import (
    DMSwimmer6Model, DMSwimmer15Model,
)
from mojo_rl.envs.dm_control.swimmer.swimmer_config import DMSwimmerConfig
from mojo_rl.envs.dm_control.walker.walker_xml import DMWalkerModel
from mojo_rl.envs.dm_control.walker.walker_config import DMWalkerConfig


# ── drive modes ─────────────────────────────────────────────────────────────
#
#   zero    no torque. The honest reset pose plus gravity: where the model
#           actually settles, and whether it spawns intersecting the floor.
#   random  uniform in [-1, 1] per actuator, resampled every HOLD_STEPS.
#           Shakes every joint; spots one that cannot move or has no limit.
#   sweep   a slow out-of-phase sine per actuator (DEFAULT). The clearest way
#           to read joint AXES and RANGES — each joint traces its full arc
#           smoothly instead of jittering.
comptime DRIVE_ZERO: Int = 0
comptime DRIVE_RANDOM: Int = 1
comptime DRIVE_SWEEP: Int = 2

comptime HOLD_STEPS: Int = 25
comptime N_TASKS: Int = 43
comptime SWEEP_PERIOD: Float64 = 120.0
comptime EPISODE_STEPS: Int = 1000
comptime FRAME_DELAY_MS: Int = 16
comptime SIDEBAR_W: Float32 = 320.0
comptime PLOT_N: Int = 120
"""Samples in the reward sparkline. A ring buffer, so `ig_plot_lines` gets the
write cursor as its offset and plots oldest-first without a rotation."""


def task_names() -> List[String]:
    """The 43 tasks, in the order `dispatch` indexes them.

    ⚠ THIS LIST AND `dispatch` ARE POSITIONALLY COUPLED. Index i here must be
    the arm `st.task == i` there; a mismatch shows up as clicking one robot and
    getting another, which is confusing precisely because everything still
    works. `task_index` is the only lookup, so argv names come from here too
    and cannot drift separately.
    """
    var t = List[String]()
    t.append(String("acrobot_swingup"))
    t.append(String("acrobot_swingup_sparse"))
    t.append(String("ball_in_cup_catch"))
    t.append(String("cartpole_balance"))
    t.append(String("cartpole_balance_sparse"))
    t.append(String("cartpole_swingup"))
    t.append(String("cartpole_swingup_sparse"))
    t.append(String("cartpole_two_poles"))
    t.append(String("cartpole_three_poles"))
    t.append(String("cheetah_run"))
    t.append(String("finger_spin"))
    t.append(String("finger_turn_easy"))
    t.append(String("finger_turn_hard"))
    t.append(String("fish_upright"))
    t.append(String("fish_swim"))
    t.append(String("hopper_stand"))
    t.append(String("hopper_hop"))
    t.append(String("humanoid_stand"))
    t.append(String("humanoid_walk"))
    t.append(String("humanoid_run"))
    t.append(String("humanoid_run_pure_state"))
    t.append(String("humanoid_cmu_stand"))
    t.append(String("humanoid_cmu_walk"))
    t.append(String("humanoid_cmu_run"))
    t.append(String("manipulator_bring_ball"))
    t.append(String("manipulator_bring_peg"))
    t.append(String("manipulator_insert_ball"))
    t.append(String("manipulator_insert_peg"))
    t.append(String("pendulum_swingup"))
    t.append(String("point_mass_easy"))
    t.append(String("point_mass_hard"))
    t.append(String("quadruped_walk"))
    t.append(String("quadruped_run"))
    t.append(String("quadruped_fetch"))
    t.append(String("reacher_easy"))
    t.append(String("reacher_hard"))
    t.append(String("stacker_stack_2"))
    t.append(String("stacker_stack_4"))
    t.append(String("swimmer_swimmer6"))
    t.append(String("swimmer_swimmer15"))
    t.append(String("walker_stand"))
    t.append(String("walker_walk"))
    t.append(String("walker_run"))
    return t^


def domain_names() -> List[String]:
    """The 17 domains, in the order `task_domain` indexes them."""
    var d = List[String]()
    d.append(String("acrobot"))
    d.append(String("ball_in_cup"))
    d.append(String("cartpole"))
    d.append(String("cheetah"))
    d.append(String("finger"))
    d.append(String("fish"))
    d.append(String("hopper"))
    d.append(String("humanoid"))
    d.append(String("humanoid_cmu"))
    d.append(String("manipulator"))
    d.append(String("pendulum"))
    d.append(String("point_mass"))
    d.append(String("quadruped"))
    d.append(String("reacher"))
    d.append(String("stacker"))
    d.append(String("swimmer"))
    d.append(String("walker"))
    return d^


def task_domain() -> List[Int]:
    """Domain index per task id.

    ⚠ EXPLICIT, NOT DERIVED FROM THE NAME. Prefix-splitting looks tempting and
    is wrong here: `ball_in_cup_catch`, `point_mass_easy` and
    `humanoid_run_pure_state` all break a split-on-first-underscore rule, in
    three different ways.
    """
    var t = List[Int]()
    for _ in range(2):
        t.append(0)   # acrobot
    t.append(1)       # ball_in_cup
    for _ in range(6):
        t.append(2)   # cartpole
    t.append(3)       # cheetah
    for _ in range(3):
        t.append(4)   # finger
    for _ in range(2):
        t.append(5)   # fish
    for _ in range(2):
        t.append(6)   # hopper
    for _ in range(4):
        t.append(7)   # humanoid
    for _ in range(3):
        t.append(8)   # humanoid_cmu
    for _ in range(4):
        t.append(9)   # manipulator
    t.append(10)      # pendulum
    for _ in range(2):
        t.append(11)  # point_mass
    for _ in range(3):
        t.append(12)  # quadruped  (walk, run, fetch)
    for _ in range(2):
        t.append(13)  # reacher
    for _ in range(2):
        t.append(14)  # stacker
    for _ in range(2):
        t.append(15)  # swimmer
    for _ in range(3):
        t.append(16)  # walker
    return t^


def drive_names() -> List[String]:
    var d = List[String]()
    d.append(String("zero"))
    d.append(String("random"))
    d.append(String("sweep"))
    return d^


def parse_drive(name: String) -> Int:
    if name == "zero":
        return DRIVE_ZERO
    if name == "random":
        return DRIVE_RANDOM
    return DRIVE_SWEEP


def task_index(name: String) -> Int:
    """Task id for a name, or -1."""
    var names = task_names()
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


def _contains(haystack: String, needle: String) -> Bool:
    """Substring test — `find` returning -1 is the whole implementation."""
    if needle.byte_length() == 0:
        return True
    return haystack.find(needle) != -1


def _fmt2(v: Float64) -> String:
    """Two decimals without a formatting library."""
    var scaled = Int(v * 100.0 + (0.5 if v >= 0 else -0.5))
    var whole = scaled // 100
    var frac = scaled % 100
    if frac < 0:
        frac = -frac
    var f = String(frac) if frac >= 10 else "0" + String(frac)
    return String(whole) + "." + f


# ═══════════════════════════════════════════════════════════════════════════
# state
# ═══════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct ViewerState(Copyable, Movable):
    """Everything that must SURVIVE a task switch.

    A switch tears the env down, so anything held in `run_view`'s locals dies
    with it. Drive mode, scale and the filter live here so picking a new robot
    does not silently reset how it is being driven or re-show the list the user
    just narrowed — the two most annoying ways a picker can behave.
    """

    var task: Int
    var drive: Int32
    """Int32, not Int: `ig_combo` writes it through a C `int*`."""
    var scale: Float32
    """Float32 for the same reason — `ig_slider_float` owns a `float*`."""
    var quit: Bool
    """Set when the window closed for real, as opposed to for a switch."""
    var filter: TextBuffer
    var show_plot: Bool

    def __init__(out self, task: Int, drive: Int, scale: Float64):
        self.task = task
        self.drive = Int32(drive)
        self.scale = Float32(scale)
        self.quit = False
        self.filter = TextBuffer()
        self.show_plot = True


@fieldwise_init
struct ViewControls(Copyable, Movable):
    """Renderer state IN, requests OUT — the panel never touches an env.

    Keeping it plain data is what lets the sidebar be ordinary non-generic
    code instead of something parameterised on the env's compile-time type.
    That is a compile-time lever, not only a tidiness one: see this module's
    header.
    """

    var n_cameras: Int
    var current_camera: Int
    var paused: Bool
    var recording: Bool
    var rec_frames: Int

    var want_camera: Int
    """-1 when no numbered camera button was pressed this frame."""
    var want_free_camera: Bool
    var want_screenshot: Bool
    var want_toggle_pause: Bool
    var want_toggle_record: Bool

    def __init__(out self, n_cameras: Int, current_camera: Int, paused: Bool,
                 recording: Bool, rec_frames: Int):
        self.n_cameras = n_cameras
        self.current_camera = current_camera
        self.paused = paused
        self.recording = recording
        self.rec_frames = rec_frames
        self.want_camera = -1
        self.want_free_camera = False
        self.want_screenshot = False
        self.want_toggle_pause = False
        self.want_toggle_record = False


@fieldwise_init
struct SidebarOut(Copyable, Movable):
    """What the user asked for this frame."""

    var picked: Int
    """Task id clicked in the tree, or -1."""
    var reset_episode: Bool
    var zero_now: Bool


# ═══════════════════════════════════════════════════════════════════════════
# UI components — plain functions over plain data, composed by `build_sidebar`
# ═══════════════════════════════════════════════════════════════════════════


def ui_status(name: String, episode: Int, step_i: Int, ep_return: Float64,
              obs_dim: Int, act_dim: Int) raises:
    """Header: which task, and how its episode is going."""
    ig_text_colored(name, 0.47, 0.90, 1.0)
    ig_text_disabled(
        String("obs ") + String(obs_dim) + String("   act ") + String(act_dim)
    )
    ig_text(
        String("ep ") + String(episode) + String("   step ") + String(step_i)
    )
    ig_text(String("return ") + _fmt2(ep_return))


def ui_reward_plot(history: List[Float32], cursor: Int) raises:
    """A sparkline of recent per-step reward.

    ⚠ THE CURSOR IS PASSED, NOT ROTATED AWAY. `ig_plot_lines` takes a ring
    buffer's write index as `offset` and reads from there, so the oldest sample
    is drawn leftmost with no copying.
    """
    ig_text_disabled(String("reward"))
    # ⚠ LABEL HIDDEN ("##" PREFIX), CAPTION DRAWN ABOVE. ImGui puts a plot's
    # label to its RIGHT, inside the same item — so a full-width plot pushes
    # the label off the panel and leaves a clipped letter at the edge.
    ig_plot_lines(
        String("##reward"), history, offset=cursor, lo=0.0, hi=1.0,
        w=-1.0, h=44.0,
    )


def ui_task_tree(
    mut st: ViewerState,
    tasks: List[String],
    domains: List[String],
    domain_of: List[Int],
    height: Float32,
) raises -> Int:
    """Filter box + domain tree. Returns the picked task id, or -1.

    A non-empty filter FLATTENS the tree: matching tasks are listed directly,
    because making the user expand a domain to reach a result they just named
    would defeat the search.
    """
    var picked = -1

    ig_set_next_item_width(-1.0)
    _ = ig_input_text(String("##filter"), st.filter, String("filter tasks..."))
    var query = st.filter.value()

    # The child is what makes 43 tasks a non-problem: it SCROLLS. The
    # hand-rolled sidebar had to cap the row count and print
    # "+N more — narrow the filter" instead.
    if ig_begin_child(String("tasks"), 0.0, height, True):
        if query.byte_length() > 0:
            var n_shown = 0
            for i in range(len(tasks)):
                if not _contains(tasks[i], query):
                    continue
                n_shown += 1
                if ig_selectable(tasks[i], i == st.task):
                    picked = i
            if n_shown == 0:
                ig_text_disabled(String("no match"))
        else:
            for d in range(len(domains)):
                # ⚠ ID PUSHED PER DOMAIN. ImGui identifies widgets by label; a
                # task whose name matched its domain's would otherwise collide
                # with the header and one of the two would stop responding.
                ig_push_id_int(d)
                if ig_tree_node(domains[d]):
                    for i in range(len(tasks)):
                        if domain_of[i] != d:
                            continue
                        if ig_selectable(tasks[i], i == st.task):
                            picked = i
                    ig_tree_pop()
                ig_pop_id()
    ig_end_child()
    return picked


def ui_drive_controls(mut st: ViewerState) raises -> SidebarOut:
    """Drive mode, action scale, and the two episode buttons."""
    ig_set_next_item_width(-60.0)
    _ = ig_combo(String("mode"), st.drive, drive_names())

    ig_set_next_item_width(-60.0)
    # A real DRAG: the value streams for the whole gesture. This is the
    # capability the hand-rolled widget layer could not provide at all, which
    # is why scale used to be two nudge buttons.
    _ = ig_slider_float(String("scale"), st.scale, 0.0, 8.0, String("%.2f"))

    var half = (ig_content_width() - 8.0) * 0.5
    var out = SidebarOut(-1, False, False)
    if ig_button(String("reset episode"), half, 0.0):
        out.reset_episode = True
    ig_same_line()
    if ig_button(String("zero torque"), half, 0.0):
        out.zero_now = True
    return out^


def ui_view_controls(mut vc: ViewControls) raises:
    """Camera selection and capture. Writes its answers back into `vc`."""
    vc.want_camera = -1
    vc.want_free_camera = False
    vc.want_screenshot = False
    vc.want_toggle_pause = False
    vc.want_toggle_record = False

    ig_text_disabled(String("camera"))
    # "free" is NOT one of the model's cameras — it is dm_control's -1, the
    # absence of one, and the only camera the mouse fully controls. It leads
    # the row because it is where dm_control's own viewer starts.
    #
    # Sized from the model, not fixed: dm_control models carry between one and
    # four cameras, plus this button.
    var n_slots = vc.n_cameras + 1
    var w = (ig_content_width() - Float32(n_slots - 1) * 8.0) / Float32(n_slots)
    if ig_toggle_button(String("free"), vc.current_camera < 0, w, 0.0):
        vc.want_free_camera = True
    for c in range(vc.n_cameras):
        ig_same_line()
        if ig_toggle_button(String(c + 1), c == vc.current_camera, w, 0.0):
            vc.want_camera = c

    # ⚠ "###" PINS THE ID WHILE THE TEXT VARIES. ImGui derives a widget's
    # identity from its LABEL, so a button whose caption changes is a DIFFERENT
    # widget each frame — the press and the release land on two different ids
    # and the click never completes.
    #
    # That is not theoretical: the record button showed a live frame count
    # ("rec 1", "rec 2", ...), so once recording started its id changed every
    # single frame and it could no longer be clicked to STOP. It would start a
    # recording and then refuse to end it; the only way out was switching task.
    #
    # Everything after "###" is the id and is not drawn; everything before is
    # drawn and does NOT contribute to the id (unlike "##", where the visible
    # part still does — which would not have helped here).
    var third = (ig_content_width() - 16.0) / 3.0
    if ig_toggle_button(
        String("resume###pause") if vc.paused else String("pause###pause"),
        vc.paused, third, 0.0,
    ):
        vc.want_toggle_pause = True
    ig_same_line()
    if ig_button(String("shot"), third, 0.0):
        vc.want_screenshot = True
    ig_same_line()
    if ig_toggle_button(
        (String("rec ") + String(vc.rec_frames) + String("###rec"))
        if vc.recording else String("rec###rec"),
        vc.recording, third, 0.0,
    ):
        vc.want_toggle_record = True


def build_sidebar(
    name: String,
    episode: Int,
    step_i: Int,
    ep_return: Float64,
    obs_dim: Int,
    act_dim: Int,
    history: List[Float32],
    cursor: Int,
    panel_h: Float32,
    mut st: ViewerState,
    tasks: List[String],
    domains: List[String],
    domain_of: List[Int],
    mut vc: ViewControls,
) raises -> SidebarOut:
    """The whole panel, in one NON-GENERIC function.

    ⚠ KEEP IT NON-GENERIC. `run_view` is instantiated once per task, so every
    line inside it is compiled 43 times; every line here is compiled once.
    Taking the env's state as plain arguments and returning requests is what
    buys that, and it is the difference between the sidebar being free and it
    dominating the build.
    """
    ig_begin_panel(String("dm_control"), 0.0, 0.0, SIDEBAR_W, panel_h)

    ui_status(name, episode, step_i, ep_return, obs_dim, act_dim)
    if st.show_plot:
        ui_reward_plot(history, cursor)

    ig_separator_text(String("task"))
    # The tree takes what is left after the fixed-height sections below it, so
    # the list grows with the window instead of being clipped by a hardcoded
    # row budget.
    var picked = ui_task_tree(st, tasks, domains, domain_of, panel_h - 400.0)

    ig_separator_text(String("drive"))
    var out = ui_drive_controls(st)
    out.picked = picked

    ig_separator_text(String("view"))
    ui_view_controls(vc)
    ig_spacing()
    ig_text_disabled(String("drag: orbit   shift+drag: pan"))

    ig_end()
    return out^


# ═══════════════════════════════════════════════════════════════════════════
# the viewer loop
# ═══════════════════════════════════════════════════════════════════════════


def run_view[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig
](name: String, mut st: ViewerState) raises:
    """The viewer loop for one task, running until it quits or switches.

    Returns through `st`: either `st.quit`, or `st.task` now naming a DIFFERENT
    task, which `run_viewer` then launches. It cannot call itself with the new
    task — each task is a separate compile-time instantiation, so recursion
    would try to instantiate all 43 inside all 43.

    ⚠ PARAMETERISED ON MODEL+CONFIG, NOT ON THE ENV TYPE. The obvious
    factoring — `def run_view[E: BoxContinuousActionEnv & RenderableEnv](mut
    env: E)` — does not compile: `E.ActionType()` cannot be constructed through
    the trait bound ("no matching function in initialization"). Building the
    concrete `Phyics3dEnv[...]` inside makes `E.ActionType` a real type again.

    ⚠ SWITCHING TASKS DESTROYS AND REBUILDS THE WINDOW, so expect a blink and
    the camera to return to its default. `Phyics3dEnv[MODEL, CONFIG]` is a
    distinct type per task and OWNS its renderer, which owns the window.
    Keeping the window across a switch means splitting window and device out of
    `Renderer3D` — worth doing if this becomes a studio, pointless before.
    """
    comptime E = Phyics3dEnv[MODEL, CONFIG, DT, False]
    comptime ACT_DIM = E.ACTION_DIM

    print("=" * 66)
    print("dm_control viewer (ImGui) —", name)
    print("=" * 66)
    print("  obs dim    =", E.OBS_DIM)
    print("  action dim =", ACT_DIM)
    print("  pick another task in the window, or close it to quit")

    var ctx = DeviceContext()
    var env = E(ctx)
    _ = env.reset()

    if not env.init_renderer():
        print("No renderer available — is SDL3 present?")
        st.quit = True
        return

    var have_ui = env.imgui_init()
    if not have_ui:
        # The scene is perfectly usable without a sidebar, so this degrades
        # rather than quitting. Note the strip is NOT reserved and the HUD is
        # NOT hidden on this path: reserving space for a panel that will never
        # be drawn would waste a fifth of the window, and the HUD is then the
        # only remaining UI.
        print("  (continuing without the ImGui sidebar)")
    else:
        ig_style_dark()
        # The built-in HUD lists the same keybinds and reports the same camera,
        # step and pause state the sidebar shows — and it paints them OVER the
        # robot. With a panel up it is pure occlusion.
        env.renderer_set_show_hud(False)
        # Reserve the strip. This INSETS the 3D viewport and corrects the
        # camera aspect, so the scene sits beside the panel rather than behind
        # it — a panel over an uninset scene would hide a fifth of the robot.
        env.set_ui_sidebar_width(Int(SIDEBAR_W))

    var tasks = task_names()
    var domains = domain_names()
    var domain_of = task_domain()
    var held = E.ActionType()
    var step_i = 0
    var episode = 0
    var ep_return = Float64(0)
    var switching = False

    var history = List[Float32]()
    for _ in range(PLOT_N):
        history.append(0.0)
    var cursor = 0

    while env.is_renderer_open():
        # Pump events FIRST: ImGui drains the queue in its NewFrame, so events
        # polled here are the ones this frame's widgets react to.
        if env.check_renderer_quit():
            break

        env.imgui_new_frame()

        var vc = ViewControls(
            env.renderer_n_cameras(), env.renderer_current_camera(),
            env.renderer_paused(), env.renderer_is_recording(),
            env.renderer_recording_frames(),
        )
        var ui = SidebarOut(-1, False, False)
        if have_ui:
            ui = build_sidebar(
                name, episode, step_i, ep_return, E.OBS_DIM, ACT_DIM,
                history, cursor, Float32(E.RENDER_HEIGHT), st,
                tasks, domains, domain_of, vc,
            )

        # ── apply what the UI asked for ──────────────────────────────────
        if ui.reset_episode:
            _ = env.reset()
            step_i = 0
            ep_return = 0.0
        elif ui.zero_now:
            st.drive = Int32(DRIVE_ZERO)
            st.scale = 1.0
        if vc.want_free_camera:
            env.renderer_request_free_camera()
        if vc.want_camera >= 0:
            env.renderer_request_camera(vc.want_camera)
        if vc.want_screenshot:
            env.renderer_request_screenshot()
        if vc.want_toggle_pause:
            env.renderer_toggle_pause()
        if vc.want_toggle_record:
            # ⚠ CATCH: video encoding goes through Python `imageio`, and a
            # missing import raised straight out of the frame loop and killed
            # the viewer. Screenshots already degrade to a printed message;
            # recording must not be the one button that can end the session.
            try:
                env.renderer_toggle_recording()
            except e:
                print("recording unavailable:", e)
                print(
                    "  imageio lives in the pixi env — launch through"
                    " `pixi run`, not the bare binary"
                )

        # Leave BEFORE stepping: the new task's env is built by `run_viewer`,
        # and there is nothing to gain from one more frame of the old one.
        #
        # ⚠ `render_frame` STILL RUNS below on the switching frame, because
        # `imgui_new_frame` already opened an ImGui frame and only `end_frame`
        # closes it. Breaking here instead would leave that frame open and the
        # NEXT task's first `NewFrame` would assert.
        if ui.picked >= 0 and ui.picked != st.task:
            st.task = ui.picked
            switching = True

        # ⚠ PAUSE MUST GATE THE STEP, not merely label the button. Both this
        # viewer and the ui.mojo one read `renderer_paused()` only to choose
        # between "pause" and "resume" and then stepped anyway, so the button
        # toggled a flag that nothing downstream honoured — the renderer's own
        # `is_paused` only ever froze its HUD step counter.
        #
        # `renderer_step_once()` is the RIGHT-arrow single-step: true for
        # exactly the one frame after the key, so a paused sim advances one
        # step and stops. It is reset at the top of every `check_quit`, which
        # this loop calls once per iteration, so reading it here is correct.
        var stepping = not switching
        if stepping and env.renderer_paused() and not env.renderer_step_once():
            stepping = False

        var action = E.ActionType()
        if stepping:
            if Int(st.drive) == DRIVE_RANDOM:
                if step_i % HOLD_STEPS == 0:
                    for a in range(ACT_DIM):
                        held.data[a] = (
                            random_float64(-1.0, 1.0) * Float64(st.scale)
                        )
                for a in range(ACT_DIM):
                    action.data[a] = held.data[a]
            elif Int(st.drive) == DRIVE_SWEEP:
                var t = Float64(step_i) / SWEEP_PERIOD
                for a in range(ACT_DIM):
                    var phase = Float64(a) / Float64(
                        ACT_DIM if ACT_DIM > 0 else 1
                    )
                    action.data[a] = (
                        sin(2.0 * pi * (t + phase)) * Float64(st.scale)
                    )

            var out = env.step(action)
            ep_return += Float64(out[1])
            history[cursor] = Float32(out[1])
            cursor = (cursor + 1) % PLOT_N
            step_i += 1

            if out[2] or step_i >= EPISODE_STEPS:
                episode += 1
                print("  episode", episode, "ended after", step_i,
                      "steps, return =", ep_return)
                _ = env.reset()
                step_i = 0
                ep_return = 0.0

        env.render_frame()
        if switching:
            break
        env.renderer_delay(FRAME_DELAY_MS)

    st.quit = not switching
    env.close_renderer()
    if switching:
        print("  switching to", tasks[st.task])
    else:
        print("viewer closed")


def dispatch(mut st: ViewerState) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    ⚠ INDEX ORDER MUST MATCH `task_names`. This is the one place all 43
    compile-time instantiations are named, and what the build time is
    proportional to.
    """
    var name = task_names()[st.task]
    if st.task == 0:
        run_view[DMAcrobotModel, DMAcrobotConfig[False]](name, st)
    elif st.task == 1:
        run_view[DMAcrobotModel, DMAcrobotConfig[True]](name, st)
    elif st.task == 2:
        run_view[DMBallInCupModel, DMBallInCupConfig](name, st)
    elif st.task == 3:
        run_view[DMCartpole1Model, DMCartpoleConfig[1, False, False]](name, st)
    elif st.task == 4:
        run_view[DMCartpole1Model, DMCartpoleConfig[1, False, True]](name, st)
    elif st.task == 5:
        run_view[DMCartpole1Model, DMCartpoleConfig[1, True, False]](name, st)
    elif st.task == 6:
        run_view[DMCartpole1Model, DMCartpoleConfig[1, True, True]](name, st)
    elif st.task == 7:
        run_view[DMCartpole2Model, DMCartpoleConfig[2, True, False]](name, st)
    elif st.task == 8:
        run_view[DMCartpole3Model, DMCartpoleConfig[3, True, False]](name, st)
    elif st.task == 9:
        run_view[DMCheetahModel, DMCheetahConfig](name, st)
    elif st.task == 10:
        run_view[DMFingerSpinModel, DMFingerSpinConfig](name, st)
    elif st.task == 11:
        run_view[DMFingerTurnModel, DMFingerTurnConfig[0.07]](name, st)
    elif st.task == 12:
        run_view[DMFingerTurnModel, DMFingerTurnConfig[0.03]](name, st)
    elif st.task == 13:
        run_view[DMFishUprightModel, DMFishUprightConfig](name, st)
    elif st.task == 14:
        run_view[DMFishSwimModel, DMFishSwimConfig](name, st)
    elif st.task == 15:
        run_view[DMHopperModel, DMHopperConfig[False]](name, st)
    elif st.task == 16:
        run_view[DMHopperModel, DMHopperConfig[True]](name, st)
    elif st.task == 17:
        run_view[DMHumanoidModel, DMHumanoidConfig[0.0, False]](name, st)
    elif st.task == 18:
        run_view[DMHumanoidModel, DMHumanoidConfig[WALK_SPEED, False]](name, st)
    elif st.task == 19:
        run_view[DMHumanoidModel, DMHumanoidConfig[RUN_SPEED, False]](name, st)
    elif st.task == 20:
        run_view[
            DMHumanoidPureModel, DMHumanoidConfig[RUN_SPEED, True]
        ](name, st)
    elif st.task == 21:
        run_view[DMHumanoidCMUModel, DMHumanoidCMUConfig[0.0]](name, st)
    elif st.task == 22:
        run_view[
            DMHumanoidCMUModel, DMHumanoidCMUConfig[CMU_WALK_SPEED]
        ](name, st)
    elif st.task == 23:
        run_view[
            DMHumanoidCMUModel, DMHumanoidCMUConfig[CMU_RUN_SPEED]
        ](name, st)
    elif st.task == 24:
        run_view[
            DMManipulatorBringBallModel, DMManipulatorBringBallConfig
        ](name, st)
    elif st.task == 25:
        run_view[
            DMManipulatorBringPegModel, DMManipulatorBringPegConfig
        ](name, st)
    elif st.task == 26:
        run_view[
            DMManipulatorInsertBallModel, DMManipulatorInsertBallConfig
        ](name, st)
    elif st.task == 27:
        run_view[
            DMManipulatorInsertPegModel, DMManipulatorInsertPegConfig
        ](name, st)
    elif st.task == 28:
        run_view[DMPendulumModel, DMPendulumConfig](name, st)
    elif st.task == 29:
        run_view[DMPointMassModel, DMPointMassConfig](name, st)
    elif st.task == 30:
        run_view[DMPointMassModel, DMPointMassHardConfig](name, st)
    elif st.task == 31:
        run_view[DMQuadrupedWalkModel, DMQuadrupedWalkConfig](name, st)
    elif st.task == 32:
        run_view[DMQuadrupedRunModel, DMQuadrupedRunConfig](name, st)
    elif st.task == 33:
        run_view[DMQuadrupedFetchModel, DMQuadrupedFetchConfig](name, st)
    elif st.task == 34:
        run_view[DMReacherModel, DMReacherConfig[0.05]](name, st)
    elif st.task == 35:
        run_view[DMReacherModel, DMReacherConfig[0.015]](name, st)
    elif st.task == 36:
        run_view[DMStacker2Model, DMStacker2Config](name, st)
    elif st.task == 37:
        run_view[DMStacker4Model, DMStacker4Config](name, st)
    elif st.task == 38:
        run_view[DMSwimmer6Model, DMSwimmerConfig](name, st)
    elif st.task == 39:
        run_view[DMSwimmer15Model, DMSwimmerConfig](name, st)
    elif st.task == 40:
        run_view[DMWalkerModel, DMWalkerConfig[0.0]](name, st)
    elif st.task == 41:
        run_view[DMWalkerModel, DMWalkerConfig[1.0]](name, st)
    elif st.task == 42:
        run_view[DMWalkerModel, DMWalkerConfig[8.0]](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def run_viewer(start_task: Int, drive: Int, scale: Float64) raises:
    """Open the viewer on `start_task` and run until the window is closed.

    One task runs at a time; picking another in the window ends that task's
    loop and comes back here to build the next one. `st` is what crosses the
    gap, since the env and its window do not.
    """
    if not imgui_shim_available():
        print("Dear ImGui shim not built.")
        print("  Run:  pixi run build-imgui")
        print("  (or use examples/dm_control/dm_viewer.mojo, which needs no")
        print("   native dependency)")
        return

    var st = ViewerState(start_task, drive, scale)
    while not st.quit:
        dispatch(st)


def print_task_list() raises:
    """The registered names, for an argv error path."""
    var names = task_names()
    for i in range(len(names)):
        print("   ", names[i])
