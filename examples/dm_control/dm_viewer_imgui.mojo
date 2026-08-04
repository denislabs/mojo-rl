"""Interactive dm_control viewer with a Dear ImGui sidebar.

    pixi run build-imgui                                   # ONCE
    pixi run mojo run -I . examples/dm_control/dm_viewer_imgui.mojo
    pixi run mojo run -I . examples/dm_control/dm_viewer_imgui.mojo cheetah_run

⚠ `pixi run build-imgui` IS A PREREQUISITE, and its absence is a RUNTIME
failure, not a compile error — the shim is loaded by dlopen. This file checks
for it up front and says so rather than aborting mid-frame.

RELATIONSHIP TO `dm_viewer.mojo`. That one stays; this is a parallel port, not
a replacement. Same physics, same renderer, same task-switch machinery — the
whole difference is that the sidebar is Dear ImGui instead of the hand-rolled
`mojo_rl/render/ui.mojo` widgets. Keeping both means the ImGui dependency
stays optional and the two can be compared side by side.

⚠ TEN TASKS, NOT THIRTY-NINE. `dm_viewer.mojo` instantiates all 39, which
costs ~8 minutes to build because every task is a distinct COMPILE-TIME type.
The subset below is one task per domain across ten of the sixteen domains —
enough to exercise every renderer path (free joints, tendons, planar and 3D
robots, multi-camera models) at roughly a quarter of the build. ADDING ONE is
two lines: a name in `_task_names` and an arm in `_dispatch`, at the same
index. See `dm_viewer.mojo`'s header for the full 39.

WHAT THIS IS FOR. It answers "is the model built and posed the way I think" —
geometry, joint axes, ranges, the reset pose, whether anything falls through
the floor. It does NOT check parity with MuJoCo; `tests/dm_control/` does that,
and a model can look perfect while its dynamics differ.

⚠ ACTIONS ARE NOT A POLICY. The drive modes move every joint so you can see it
articulate. A tumbling humanoid under random torque is the expected picture.

CONTROLS
  sidebar             task tree, filter, drive, scale, cameras, capture
  mouse drag / scroll orbit, zoom — suppressed while the pointer is over the UI
  1-9 / SPACE / R     camera, pause, reset view (renderer bindings)
  close window        quit

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it. CPU physics on purpose: one env at 60 Hz needs no GPU.
"""

from std.random import seed, random_float64
from std.math import sin, pi, min, max
from std.sys import argv
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
from mojo_rl.envs.dm_control.cartpole.cartpole_xml import DMCartpole1Model
from mojo_rl.envs.dm_control.cartpole.cartpole_config import DMCartpoleConfig
from mojo_rl.envs.dm_control.cheetah.cheetah_xml import DMCheetahModel
from mojo_rl.envs.dm_control.cheetah.cheetah_config import DMCheetahConfig
from mojo_rl.envs.dm_control.finger.finger_xml import DMFingerSpinModel
from mojo_rl.envs.dm_control.finger.finger_config import DMFingerSpinConfig
from mojo_rl.envs.dm_control.hopper.hopper_xml import DMHopperModel
from mojo_rl.envs.dm_control.hopper.hopper_config import DMHopperConfig
from mojo_rl.envs.dm_control.humanoid.humanoid_xml import DMHumanoidModel
from mojo_rl.envs.dm_control.humanoid.humanoid_config import DMHumanoidConfig
from mojo_rl.envs.dm_control.point_mass.point_mass_xml import DMPointMassModel
from mojo_rl.envs.dm_control.point_mass.point_mass_config import (
    DMPointMassConfig,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import DMQuadrupedWalkModel
from mojo_rl.envs.dm_control.quadruped.quadruped_config import (
    DMQuadrupedWalkConfig,
)
from mojo_rl.envs.dm_control.walker.walker_xml import DMWalkerModel
from mojo_rl.envs.dm_control.walker.walker_config import DMWalkerConfig


comptime DRIVE_ZERO: Int = 0
comptime DRIVE_RANDOM: Int = 1
comptime DRIVE_SWEEP: Int = 2

comptime HOLD_STEPS: Int = 25        # RANDOM: steps between resamples
comptime N_TASKS: Int = 10
comptime SWEEP_PERIOD: Float64 = 120.0
comptime EPISODE_STEPS: Int = 1000
comptime FRAME_DELAY_MS: Int = 16    # ~60 FPS
comptime SEED: Int = 0

comptime SIDEBAR_W: Float32 = 320.0
comptime PLOT_N: Int = 120
"""Samples in the reward sparkline. A ring buffer, so `ig_plot_lines` is given
the write cursor as its offset and plots oldest-first without a rotation."""


def _task_names() -> List[String]:
    """⚠ POSITIONALLY COUPLED TO `_dispatch`. Index i here must be the arm
    `st.task == i` there; a mismatch shows up as clicking one robot and getting
    another, which is confusing precisely because everything still works."""
    var t = List[String]()
    t.append(String("acrobot_swingup"))
    t.append(String("ball_in_cup_catch"))
    t.append(String("cartpole_swingup"))
    t.append(String("cheetah_run"))
    t.append(String("finger_spin"))
    t.append(String("hopper_stand"))
    t.append(String("humanoid_stand"))
    t.append(String("point_mass_easy"))
    t.append(String("quadruped_walk"))
    t.append(String("walker_walk"))
    return t^


def _domain_names() -> List[String]:
    var d = List[String]()
    d.append(String("acrobot"))
    d.append(String("ball_in_cup"))
    d.append(String("cartpole"))
    d.append(String("cheetah"))
    d.append(String("finger"))
    d.append(String("hopper"))
    d.append(String("humanoid"))
    d.append(String("point_mass"))
    d.append(String("quadruped"))
    d.append(String("walker"))
    return d^


def _task_domain() -> List[Int]:
    """Domain index per task id.

    ⚠ EXPLICIT, NOT DERIVED FROM THE NAME. Splitting on the first underscore
    looks tempting and is wrong: `ball_in_cup_catch` and `point_mass_easy`
    break it in two different ways. One task per domain today, but the mapping
    is kept general so adding a second task to a domain needs no rewrite.
    """
    var t = List[Int]()
    for i in range(N_TASKS):
        t.append(i)
    return t^


def _drive_names() -> List[String]:
    var d = List[String]()
    d.append(String("zero"))
    d.append(String("random"))
    d.append(String("sweep"))
    return d^


def _contains(haystack: String, needle: String) -> Bool:
    if needle.byte_length() == 0:
        return True
    return haystack.find(needle) != -1


def _task_index(name: String) -> Int:
    var names = _task_names()
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


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

    A switch tears the env down, so anything held in `_view`'s locals dies with
    it. Drive mode, scale and the filter live here so picking a new robot does
    not silently reset how it is being driven or re-show the list the user just
    narrowed — the two most annoying ways a picker can behave.
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


@fieldwise_init
struct ViewControls(Copyable, Movable):
    """Renderer state IN, requests OUT — the panel below never touches an env.

    Keeping it plain data is what lets `_ui_view_controls` be an ordinary
    function instead of one parameterised on the env's compile-time type. The
    UI stays testable and the env stays out of the widget code.
    """

    var n_cameras: Int
    var current_camera: Int
    var paused: Bool
    var recording: Bool
    var rec_frames: Int

    var want_camera: Int
    """-1 when no camera button was pressed this frame."""
    var want_screenshot: Bool
    var want_toggle_pause: Bool
    var want_toggle_record: Bool


# ═══════════════════════════════════════════════════════════════════════════
# UI components — each one panel section, composed by `_ui_sidebar`
# ═══════════════════════════════════════════════════════════════════════════


def _ui_status(name: String, episode: Int, step_i: Int, ep_return: Float64,
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


def _ui_reward_plot(history: List[Float32], cursor: Int) raises:
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


def _ui_task_tree(
    mut st: ViewerState,
    tasks: List[String],
    domains: List[String],
    task_domain: List[Int],
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

    # The child is what makes an arbitrarily long list a non-problem: it
    # scrolls. The old hand-rolled sidebar had to cap the row count and print
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
                        if task_domain[i] != d:
                            continue
                        if ig_selectable(tasks[i], i == st.task):
                            picked = i
                    ig_tree_pop()
                ig_pop_id()
    ig_end_child()
    return picked


def _ui_drive_controls(mut st: ViewerState) raises -> Int:
    """Drive mode, action scale, and the two episode buttons.

    Returns 1 to reset the episode, 2 to drop to zero torque, else 0.
    """
    ig_set_next_item_width(-60.0)
    _ = ig_combo(String("mode"), st.drive, _drive_names())

    ig_set_next_item_width(-60.0)
    # A real DRAG: the value streams for the whole gesture. This is the
    # capability the hand-rolled widget layer could not provide at all, which
    # is why scale used to be two nudge buttons.
    _ = ig_slider_float(String("scale"), st.scale, 0.0, 8.0, String("%.2f"))

    var half = (ig_content_width() - 8.0) * 0.5
    var action = 0
    if ig_button(String("reset episode"), half, 0.0):
        action = 1
    ig_same_line()
    if ig_button(String("zero torque"), half, 0.0):
        action = 2
    return action


def _ui_view_controls(mut vc: ViewControls) raises:
    """Camera selection and capture. Writes its answers back into `vc`."""
    vc.want_camera = -1
    vc.want_screenshot = False
    vc.want_toggle_pause = False
    vc.want_toggle_record = False

    if vc.n_cameras > 0:
        ig_text_disabled(String("camera"))
        # Sized from the model, not fixed: dm_control models carry between one
        # and four cameras.
        var w = (ig_content_width() - Float32(vc.n_cameras - 1) * 8.0) / Float32(
            vc.n_cameras
        )
        for c in range(vc.n_cameras):
            if c > 0:
                ig_same_line()
            if ig_toggle_button(
                String(c + 1), c == vc.current_camera, w, 0.0
            ):
                vc.want_camera = c

    var third = (ig_content_width() - 16.0) / 3.0
    if ig_toggle_button(
        String("resume") if vc.paused else String("pause"), vc.paused, third,
        0.0,
    ):
        vc.want_toggle_pause = True
    ig_same_line()
    if ig_button(String("shot"), third, 0.0):
        vc.want_screenshot = True
    ig_same_line()
    if ig_toggle_button(
        (String("rec ") + String(vc.rec_frames)) if vc.recording
        else String("rec"),
        vc.recording, third, 0.0,
    ):
        vc.want_toggle_record = True


# ═══════════════════════════════════════════════════════════════════════════
# the viewer loop
# ═══════════════════════════════════════════════════════════════════════════


def _view[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig
](name: String, mut st: ViewerState) raises:
    """The viewer loop for one task, running until it quits or switches.

    Returns through `st`: either `st.quit`, or `st.task` now naming a DIFFERENT
    task, which `main` then launches. It cannot call itself with the new task —
    each task is a separate compile-time instantiation, so recursion would try
    to instantiate all ten inside all ten.

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

    if not env.imgui_init():
        # The scene is perfectly usable without a sidebar, so this degrades
        # rather than quitting; `main` already said what to run. Note that the
        # strip is NOT reserved and the HUD is NOT hidden on this path —
        # reserving space for a panel that will never be drawn would waste a
        # fifth of the window, and the HUD is the only remaining UI.
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

    var tasks = _task_names()
    var domains = _domain_names()
    var task_domain = _task_domain()
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

        var picked = -1
        var drive_action = 0
        var vc = ViewControls(
            env.renderer_n_cameras(), env.renderer_current_camera(),
            env.renderer_paused(), env.renderer_is_recording(),
            env.renderer_recording_frames(),
            -1, False, False, False,
        )

        if env.imgui_active():
            ig_begin_panel(
                String("dm_control"), 0.0, 0.0, SIDEBAR_W,
                Float32(E.RENDER_HEIGHT),
            )
            _ui_status(name, episode, step_i, ep_return, E.OBS_DIM, ACT_DIM)
            if st.show_plot:
                _ui_reward_plot(history, cursor)

            ig_separator_text(String("task"))
            # The tree gets whatever is left after the fixed-height sections
            # below it, so the list grows with the window instead of being
            # clipped by a hardcoded row budget.
            picked = _ui_task_tree(
                st, tasks, domains, task_domain, Float32(E.RENDER_HEIGHT) - 400.0
            )

            ig_separator_text(String("drive"))
            drive_action = _ui_drive_controls(st)

            ig_separator_text(String("view"))
            _ui_view_controls(vc)
            ig_spacing()
            ig_text_disabled(String("drag: orbit   shift+drag: pan"))
            ig_end()

        # ── apply what the UI asked for ──────────────────────────────────
        if drive_action == 1:
            _ = env.reset()
            step_i = 0
            ep_return = 0.0
        elif drive_action == 2:
            st.drive = 0
            st.scale = 1.0
        if vc.want_camera >= 0:
            env.renderer_request_camera(vc.want_camera)
        if vc.want_screenshot:
            env.renderer_request_screenshot()
        if vc.want_toggle_pause:
            env.renderer_toggle_pause()
        if vc.want_toggle_record:
            env.renderer_toggle_recording()

        # Leave BEFORE stepping: the new task's env is built by `main`, and
        # there is nothing to gain from one more frame of the old one.
        #
        # ⚠ `render_frame` STILL RUNS below on the switching frame, because
        # `imgui_new_frame` already opened an ImGui frame and only `end_frame`
        # closes it. Breaking here instead would leave that frame open and the
        # NEXT task's first `NewFrame` would assert.
        if picked >= 0 and picked != st.task:
            st.task = picked
            switching = True

        var action = E.ActionType()
        if not switching:
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


def _dispatch(mut st: ViewerState) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    ⚠ INDEX ORDER MUST MATCH `_task_names`. This is the one place every
    compile-time instantiation is named, and what the build time is
    proportional to.
    """
    var name = _task_names()[st.task]
    if st.task == 0:
        _view[DMAcrobotModel, DMAcrobotConfig[False]](name, st)
    elif st.task == 1:
        _view[DMBallInCupModel, DMBallInCupConfig](name, st)
    elif st.task == 2:
        _view[DMCartpole1Model, DMCartpoleConfig[1, True, False]](name, st)
    elif st.task == 3:
        _view[DMCheetahModel, DMCheetahConfig](name, st)
    elif st.task == 4:
        _view[DMFingerSpinModel, DMFingerSpinConfig](name, st)
    elif st.task == 5:
        _view[DMHopperModel, DMHopperConfig[False]](name, st)
    elif st.task == 6:
        _view[DMHumanoidModel, DMHumanoidConfig[0.0, False]](name, st)
    elif st.task == 7:
        _view[DMPointMassModel, DMPointMassConfig](name, st)
    elif st.task == 8:
        _view[DMQuadrupedWalkModel, DMQuadrupedWalkConfig](name, st)
    elif st.task == 9:
        _view[DMWalkerModel, DMWalkerConfig[1.0]](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def main() raises:
    seed(SEED)

    # Checked BEFORE any ImGui symbol is touched. The loader aborts the process
    # on a missing dylib — correct for a hard dependency, a terrible first
    # impression for an optional one.
    if not imgui_shim_available():
        print("Dear ImGui shim not built.")
        print("  Run:  pixi run build-imgui")
        print("  (or use examples/dm_control/dm_viewer.mojo, which needs no")
        print("   native dependency)")
        return

    var args = argv()
    var start = String(args[1]) if len(args) > 1 else String("quadruped_walk")
    var task = _task_index(start)
    if task < 0:
        print("unknown task:", start, "— this viewer registers:")
        var names = _task_names()
        for i in range(len(names)):
            print("   ", names[i])
        print("(dm_viewer.mojo has all 39)")
        return

    var drive = Int32(DRIVE_SWEEP)
    if len(args) > 2:
        var d = String(args[2])
        if d == "zero":
            drive = Int32(DRIVE_ZERO)
        elif d == "random":
            drive = Int32(DRIVE_RANDOM)

    var st = ViewerState(task, drive, Float32(1.0), False, TextBuffer(), True)
    while not st.quit:
        _dispatch(st)
