"""Replay a `.demo` file in the 3D viewer — the expert's (or the leader's) episodes, rendered.

    pixi run soarm-tower-replay -- projects/so101-tower/demos/expert_lift_brick_cs15_clean_500.demo
    pixi run soarm-tower-replay -- FILE.demo --episode 12 --speed 0.5

Every recorded row carries the FULL state — the observation is `qpos`, then
`qvel`, then the task words — so playback poses the sim at each row and
renders; no physics runs, nothing can drift from what was recorded, and any
`.demo` (expert, leader, intervention) plays the same way. The sidebar shows
the episode, the step, the recorded action (the six normalised joint targets)
and reward, and whether the row paid the rung (reward > 1.5 = the brick held
above the desk).

Keys (the renderer's own: SPACE pause, RIGHT ARROW one step while paused,
ESC quit, mouse orbit; `pixi run soarm-tower-record` lists the rest):

    n / p     next / previous episode
    b         restart the episode
    [ / ]     slower / faster (x0.5 / x2)
    l         loop the episode

Three eyes at once, as in the recorder: the free camera, with the overhead
and the wrist camera as insets on the right.
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.pathlib import Path

from max.gpu.host import DeviceContext

from mojo_rl.deep_agents.data.demo_file import DemoSet, read_demo_file
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.render.imgui import (
    ig_begin_panel, ig_end, ig_text, ig_text_colored, ig_text_disabled,
    ig_separator_text, ig_same_line, ig_spacing, ig_button, ig_checkbox,
    ig_slider_int, ig_slider_float, ig_progress_bar, ig_style_dark,
)
from mojo_rl.tasks.family_config import So101TowerTeleopConfig
from mojo_rl.tasks.so101_tower_xml import So101TowerModel
from mojo_rl.utils.fmt import fixed

comptime CFG = So101TowerTeleopConfig
comptime E = Phyics3dEnv[So101TowerModel, CFG, DType.float64, False]
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime OBS_DIM = So101TowerModel.OBS_DIM
comptime ACT = 6

comptime CONTROL_PERIOD_MS: Float64 = 32.0
"""The recorder's control period (16 substeps of 2 ms): one row per 32 ms at
speed 1."""
comptime SIDEBAR_W: Float32 = 320.0
comptime RUNG: Float64 = 1.5
"""A row paying more than this held the brick above the desk."""
comptime CLOSING: Float64 = 1.24
"""A row paying more than this had the jaw on the brick."""

# SDL keycodes = ASCII for letters and brackets
comptime KEY_N: Int = 110
comptime KEY_P: Int = 112
comptime KEY_B: Int = 98
comptime KEY_L: Int = 108
comptime KEY_LBRACKET: Int = 91
comptime KEY_RBRACKET: Int = 93

def _joint_name(j: Int) -> String:
    if j == 0:
        return String("pan")
    if j == 1:
        return String("lift")
    if j == 2:
        return String("elbow")
    if j == 3:
        return String("wrist flex")
    if j == 4:
        return String("wrist roll")
    return String("gripper")


def _usage():
    print("usage: tower_demo_viewer.mojo FILE.demo [--episode K] [--speed X]"
          " [--loop] [--frames N]")


struct Playback(Movable):
    """Where the playback is: the episode, the frame, and how it advances."""

    var episode: Int
    var frame: Int        # 0..len: frame k poses row start+k's obs; frame len the last row's next obs
    var speed: Float64
    var loop: Bool
    var ep_return: Float64  # return of the rows played so far
    var rung_rows: Int

    def __init__(out self, episode: Int, speed: Float64, loop: Bool):
        self.episode = episode
        self.frame = 0
        self.speed = speed
        self.loop = loop
        self.ep_return = 0.0
        self.rung_rows = 0

    def restart(mut self):
        self.frame = 0
        self.ep_return = 0.0
        self.rung_rows = 0


def _pose(mut env: E, ref d: DemoSet, row: Int, next_obs: Bool):
    """Pose the sim at one recorded observation (its qpos + qvel words)."""
    var q = List[Float64](length=NQ, fill=0.0)
    var v = List[Float64](length=NV, fill=0.0)
    var base = row * d.obs_dim
    if next_obs:
        for i in range(NQ):
            q[i] = Float64(d.nobs[base + i])
        for i in range(NV):
            v[i] = Float64(d.nobs[base + NQ + i])
    else:
        for i in range(NQ):
            q[i] = Float64(d.obs[base + i])
        for i in range(NV):
            v[i] = Float64(d.obs[base + NQ + i])
    env.set_state(q, v)


def _sidebar(
    ref d: DemoSet, path: String, mut pb: Playback, paused: Bool,
    panel_h: Float32,
) raises -> Int:
    """The panel. Returns an episode the user picked on the slider (or -1)."""
    ig_begin_panel(String("replay"), 0.0, 0.0, SIDEBAR_W, panel_h)
    ig_text(String("demo replay"))
    ig_text_disabled(path)
    ig_text(String("episodes ") + String(d.n_episodes()) + "  rows "
            + String(d.count()) + "  successes " + String(d.n_successes()))
    ig_separator_text(String("episode"))
    var ep32 = Int32(pb.episode)
    var picked = -1
    if ig_slider_int(String("##ep"), ep32, 0, Int32(d.n_episodes() - 1)):
        picked = Int(ep32)
    ig_same_line()
    if ig_button(String("<")):
        picked = pb.episode - 1 if pb.episode > 0 else d.n_episodes() - 1
    ig_same_line()
    if ig_button(String(">")):
        picked = (pb.episode + 1) % d.n_episodes()
    var start = d.ep_start[pb.episode]
    var ln = d.ep_len[pb.episode]
    var success = d.ep_success[pb.episode]
    var intervened = 0
    for k in range(ln):
        if d.row_intervened(start + k):
            intervened += 1
    if success:
        ig_text_colored(String("SUCCESS"), 0.4, 0.9, 0.4, 1.0)
    else:
        ig_text_colored(String("failed"), 0.9, 0.5, 0.4, 1.0)
    ig_same_line()
    ig_text(String("  ") + String(ln) + " steps ("
            + fixed(Float64(ln) * CONTROL_PERIOD_MS / 1000.0, 1) + " s)"
            + ("  intervened " + String(intervened) if intervened > 0 else String("")))

    ig_separator_text(String("playback"))
    var fr32 = Int32(pb.frame)
    if ig_slider_int(String("##frame"), fr32, 0, Int32(ln)):
        pb.frame = Int(fr32)
        # the return up to here, recomputed: scrubbing is not playing
        pb.ep_return = 0.0
        pb.rung_rows = 0
        for k in range(pb.frame):
            var r = Float64(d.rew[start + k])
            pb.ep_return += r
            if r > RUNG:
                pb.rung_rows += 1
    ig_text(String("step ") + String(pb.frame) + " / " + String(ln)
            + ("   PAUSED (SPACE)" if paused else String("")))
    if ig_button(String("restart (b)")):
        pb.restart()
    ig_same_line()
    if ig_checkbox(String("loop (l)"), pb.loop):
        pass
    var sp32 = Float32(pb.speed)
    if ig_slider_float(String("speed"), sp32, 0.1, 4.0, String("x%.2f")):
        pb.speed = Float64(sp32)

    ig_separator_text(String("this row"))
    var row = start + (pb.frame if pb.frame < ln else ln - 1)
    var r = Float64(d.rew[row])
    ig_text(String("reward ") + fixed(r, 3) + "   return so far "
            + fixed(pb.ep_return, 1))
    if r > RUNG:
        ig_text_colored(String("rung paid: brick above the desk"), 0.4, 0.9, 0.4, 1.0)
    elif r > CLOSING:
        ig_text_colored(String("jaw on the brick"), 0.9, 0.8, 0.3, 1.0)
    else:
        ig_text_disabled(String("approach"))
    ig_text(String("rung rows so far ") + String(pb.rung_rows))
    if d.row_intervened(row):
        ig_text_colored(String("HUMAN (intervened row)"), 0.9, 0.5, 0.9, 1.0)
    ig_spacing()
    ig_text(String("action (normalised joint targets)"))
    for j in range(ACT):
        var a = Float64(d.act[row * d.act_dim + j])
        ig_progress_bar(Float32((a + 1.0) * 0.5), -1.0, 0.0,
                        _joint_name(j) + " " + fixed(a, 2))
    ig_spacing()
    ig_text_disabled(String("n/p episode  b restart  [ ] speed  l loop"))
    ig_text_disabled(String("SPACE pause  RIGHT step  ESC quit"))
    ig_end()
    return picked


def main() raises:
    var args = argv()
    var path = String("")
    var episode = 0
    var speed = 1.0
    var loop = False
    var max_frames = 0
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--episode" and i + 1 < len(args):
            episode = Int(String(args[i + 1]))
            i += 2
        elif a == "--speed" and i + 1 < len(args):
            speed = Float64(String(args[i + 1]))
            i += 2
        elif a == "--loop":
            loop = True
            i += 1
        elif a == "--frames" and i + 1 < len(args):
            # a smoke run: close after this many frames (0 = until closed)
            max_frames = Int(String(args[i + 1]))
            i += 2
        elif a == "--help" or a == "-h":
            _usage()
            return
        elif a.startswith("--"):
            _usage()
            raise Error("unrecognised argument: " + a)
        else:
            path = a
            i += 1
    if path.byte_length() == 0 or not Path(path).exists():
        _usage()
        raise Error("no such demo file: '" + path + "'")

    var d = read_demo_file(path)
    print("=" * 66)
    print("so101_tower — demo replay:", path)
    print("=" * 66)
    print(" ", d.summary())
    if d.n_episodes() == 0:
        raise Error("the file holds no closed episode")
    if d.obs_dim != OBS_DIM or d.act_dim != ACT:
        raise Error("the file is obs " + String(d.obs_dim) + " / act "
                    + String(d.act_dim) + " but the tower family is "
                    + String(OBS_DIM) + " / " + String(ACT))
    if episode < 0 or episode >= d.n_episodes():
        raise Error("--episode out of range 0.." + String(d.n_episodes() - 1))

    var ctx = DeviceContext()
    var env = E(ctx)
    _ = env.reset()
    if not env.init_renderer(False):
        raise Error("No renderer available — is SDL3 present?")
    env.renderer_request_free_camera()
    var pip = List[Int]()
    pip.append(1)   # overhead
    pip.append(0)   # wrist
    env.renderer_set_pip_cameras(pip^)
    var have_ui = env.imgui_init()
    if have_ui:
        ig_style_dark()
        env.renderer_set_show_hud(False)
        env.set_ui_sidebar_width(Int(SIDEBAR_W))
    else:
        print("  (no ImGui sidebar — `pixi run build-imgui`; keys still work)")

    var pb = Playback(episode, speed, loop)
    var frame_t0 = perf_counter_ns()
    var n_frames = 0
    while env.is_renderer_open():
        if env.check_renderer_quit():
            break
        if max_frames > 0 and n_frames >= max_frames:
            break
        n_frames += 1
        env.imgui_new_frame()

        var paused = env.renderer_paused()
        var picked = -1
        if have_ui:
            picked = _sidebar(d, path, pb, paused, Float32(E.RENDER_HEIGHT))

        var key = env.renderer_take_key()
        if key == KEY_N:
            picked = (pb.episode + 1) % d.n_episodes()
        elif key == KEY_P:
            picked = pb.episode - 1 if pb.episode > 0 else d.n_episodes() - 1
        elif key == KEY_B:
            pb.restart()
        elif key == KEY_L:
            pb.loop = not pb.loop
        elif key == KEY_LBRACKET:
            pb.speed = pb.speed * 0.5 if pb.speed > 0.1 else pb.speed
        elif key == KEY_RBRACKET:
            pb.speed = pb.speed * 2.0 if pb.speed < 4.0 else pb.speed
        if picked >= 0 and picked != pb.episode:
            pb.episode = picked
            pb.restart()
            print("  episode", pb.episode, "—",
                  "SUCCESS" if d.ep_success[pb.episode] else "failed",
                  d.ep_len[pb.episode], "steps")

        var start = d.ep_start[pb.episode]
        var ln = d.ep_len[pb.episode]
        # pose this frame
        if pb.frame < ln:
            _pose(env, d, start + pb.frame, False)
        else:
            _pose(env, d, start + ln - 1, True)
        env.render_frame()

        # advance, unless paused (RIGHT ARROW steps one frame while paused)
        var advance = not paused or env.renderer_step_once()
        if advance:
            if pb.frame < ln:
                var r = Float64(d.rew[start + pb.frame])
                pb.ep_return += r
                if r > RUNG:
                    pb.rung_rows += 1
                pb.frame += 1
            elif pb.loop:
                pb.restart()

        var period_ms = Int(CONTROL_PERIOD_MS / pb.speed)
        var spent_ms = Int((perf_counter_ns() - frame_t0) // 1_000_000)
        if spent_ms < period_ms:
            env.renderer_delay(period_ms - spent_ms)
        frame_t0 = perf_counter_ns()

    env.close_renderer()
    print("replay closed after", n_frames, "frames — episode", pb.episode,
          "frame", pb.frame, "return so far", fixed(pb.ep_return, 1),
          "rung rows", pb.rung_rows)
