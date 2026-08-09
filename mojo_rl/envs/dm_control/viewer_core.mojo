"""Task-agnostic half of the dm_control viewer — state, sidebar, run loop.

`viewer.mojo` is the other half: the 47-arm `dispatch` that names every task's
compile-time `Phyics3dEnv[MODEL, CONFIG]`, plus `run_viewer` around it.

⚠ THE SPLIT IS A BUILD-TIME LEVER, not tidiness. `dispatch` instantiates
`run_view` 47 times, and importing the module that holds it pays for all 47
even if the importer only wants two. Keeping everything task-agnostic HERE lets
a front end with its own short task list — `examples/dm_control/
dm_viewer_imgui_two.mojo`, which exists so this code can be exercised in
seconds rather than minutes — import `run_view` without dragging the full
dispatch in behind it.

⚠ SO NOTHING IN THIS FILE MAY NAME A CONCRETE TASK. The task table lives on
`ViewerState`, supplied by the caller, precisely so that stays true.

Read `viewer.mojo`'s header for the controls, the drive modes, and what the
tool can and cannot tell you.
"""

from std.random import random_float64
from std.math import sin, pi, min, max
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_env import Phyics3dEnv, Phyics3dEnvConfig
from mojo_rl.physics3d.model import ModelDefLike
from mojo_rl.render.renderer3d import Renderer3D, RendererHandoff
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
comptime SWEEP_PERIOD: Float64 = 120.0
comptime EPISODE_STEPS: Int = 1000
comptime FRAME_DELAY_MS: Int = 16
comptime SIDEBAR_W: Float32 = 320.0
comptime PLOT_N: Int = 120
"""Samples in the reward sparkline. A ring buffer, so `ig_plot_lines` gets the
write cursor as its offset and plots oldest-first without a rotation."""


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


def task_index(name: String, names: List[String]) -> Int:
    """Position of `name` in a task table, or -1.

    Takes the table rather than reaching for a global one: this module must not
    know which tasks exist (see the header).
    """
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

    THE WINDOW CROSSES THE GAP TOO, via `handoff`. That is the difference
    between "the viewer switched model" and "the viewer closed and a new one
    opened somewhere": see the field's own note.
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

    var tasks: List[String]
    var domains: List[String]
    var domain_of: List[Int]
    """The task table the sidebar lists and `st.task` indexes — SUPPLIED BY THE
    CALLER, never built here.

    ⚠ THIS IS WHAT KEEPS THIS MODULE TASK-AGNOSTIC, and the agnosticism is a
    build-time lever (see the header). It also means the table and the caller's
    dispatch are positionally coupled: index i in `tasks` must be the arm the
    caller runs for `st.task == i`, or the user clicks one robot and gets
    another — confusing precisely because everything still works.

    `domain_of[i]` is the index into `domains` of task i's group; the tree
    shows a task under that heading and nowhere else."""

    var handoff: Optional[RendererHandoff]
    """The live SDL window + GPU device BETWEEN two tasks — set only while no
    env owns them.

    ⚠ WITHOUT THIS THE WINDOW MOVES ITSELF. Destroying and re-creating it per
    switch means the OS re-places the replacement, which on a multi-monitor
    desktop puts it back on the PRIMARY display: drag the viewer to an external
    screen, pick another robot, and it jumps home. Carrying the window also
    keeps its size, its ImGui state (the sidebar's scroll position and expanded
    domains) and skips a re-blink plus shader/font/mesh rebuild.

    ⚠ SET MEANS UNOWNED, AND UNOWNED MEANS LEAKABLE. Whoever holds this must
    hand it to the next `init_renderer` or end it with
    `Renderer3D.close_handoff`; `run_viewer` is the one place that guarantees
    both."""

    def __init__(
        out self,
        task: Int,
        drive: Int,
        scale: Float64,
        var tasks: List[String],
        var domains: List[String],
        var domain_of: List[Int],
    ):
        self.task = task
        self.drive = Int32(drive)
        self.scale = Float32(scale)
        self.quit = False
        self.filter = TextBuffer()
        self.show_plot = True
        self.tasks = tasks^
        self.domains = domains^
        self.domain_of = domain_of^
        self.handoff = None


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

    # The child is what makes 47 tasks a non-problem: it SCROLLS. The
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
    line inside it is compiled 47 times; every line here is compiled once.
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
    would try to instantiate all 47 inside all 47.

    ⚠ PARAMETERISED ON MODEL+CONFIG, NOT ON THE ENV TYPE. The obvious
    factoring — `def run_view[E: BoxContinuousActionEnv & RenderableEnv](mut
    env: E)` — does not compile: `E.ActionType()` cannot be constructed through
    the trait bound ("no matching function in initialization"). Building the
    concrete `Phyics3dEnv[...]` inside makes `E.ActionType` a real type again.

    ⚠ SWITCHING TASKS REBUILDS THE ENV, NOT THE WINDOW. `Phyics3dEnv[MODEL,
    CONFIG]` is a distinct type per task and its renderer dies with it, but the
    SDL window and GPU device are handed across the gap in `st.handoff` and
    ADOPTED here — so a switch keeps the window's monitor, position, size and
    ImGui state, and re-does only what is model-specific (cameras, lights,
    skybox, the geom mesh caches). Expect the CAMERA to reframe, since it is
    the model's; expect nothing else to move.

    ⚠ ADOPTING TAKES OWNERSHIP. `st.handoff` is cleared the moment the env has
    it, and re-filled only by `detach_renderer` on the way out — so exactly one
    party can free the window at any time.
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

    var adopt = st.handoff.copy()
    # Cleared BEFORE the call, not after: if `init_renderer` raised with `st`
    # still naming the window the env now owns, the error path below would
    # close it a second time.
    st.handoff = None
    if not env.init_renderer(True, adopt):
        print("No renderer available — is SDL3 present?")
        if adopt:
            Renderer3D.close_handoff(adopt.value().copy())
        st.quit = True
        return

    # Idempotent, and on the adopt path it is a no-op that returns True: the
    # ImGui context is attached to the window and device, both of which just
    # came across intact.
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

    # Copied out ONCE, not read from `st` per frame: `build_sidebar` takes `st`
    # mutably, so handing it `st.tasks` in the same call would be a borrow of a
    # field of the thing being mutated.
    var tasks = st.tasks.copy()
    var domains = st.domains.copy()
    var domain_of = st.domain_of.copy()
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
    if switching:
        # KEEP the window: only this env's model-specific GPU state goes.
        st.handoff = env.detach_renderer()
        print("  switching to", tasks[st.task])
    else:
        # Real close — the renderer still owns the window, so the ordinary
        # teardown is the correct one and `st.handoff` stays empty.
        env.close_renderer()
        print("viewer closed")
