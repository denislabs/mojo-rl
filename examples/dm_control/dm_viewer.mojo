"""Interactive 3D viewer for any ported dm_control task — the "does this model
look right?" tool, in the spirit of dm_control's own `suite.explore` viewer.

    pixi run mojo run -I . examples/dm_control/dm_viewer.mojo
    pixi run mojo run -I . examples/dm_control/dm_viewer.mojo cheetah_run random

PICK THE TASK IN THE WINDOW, from the list down the right-hand side; argv only
sets which one opens first. All 39 live in one binary, which costs about eight
minutes to build (measured: 90 s for one task, 474 s for all 39 — near enough
linear, no compiler blowup, the failure mode this tree has a long history of).

⚠ SWITCHING TASKS DESTROYS AND REBUILDS THE WINDOW, so expect it to blink and
the camera to return to its default. That is not a bug that can be polished
out from here: `Phyics3dEnv[MODEL, CONFIG]` is a distinct COMPILE-TIME type per
task and the env OWNS its `ModelRenderer`, which owns the `Renderer3D`, which
owns the window. Switching therefore destroys one env and builds another, and
the window goes with it. Keeping the window alive across a switch means
splitting the window and GPU device out of `Renderer3D` — worth doing if this
grows into a "physics3d studio", and pointless before then. (That SDL3 tolerates
close→create at all inside one process was verified by probe first, since the
whole picker rests on it.)

ALL 39 PORTED TASKS — argv accepts any of these names:

  acrobot      DMAcrobotSwingup  DMAcrobotSwingupSparse
  ball_in_cup  DMBallInCupCatch
  cartpole     DMCartpoleBalance  DMCartpoleBalanceSparse  DMCartpoleSwingup
               DMCartpoleSwingupSparse  DMCartpoleTwoPoles  DMCartpoleThreePoles
  cheetah      DMCheetahRun
  finger       DMFingerSpin  DMFingerTurnEasy  DMFingerTurnHard
  fish         DMFishSwim  DMFishUpright
  hopper       DMHopperHop  DMHopperStand
  humanoid     DMHumanoidStand  DMHumanoidWalk  DMHumanoidRun
               DMHumanoidRunPureState
  manipulator  DMManipulatorBringBall  DMManipulatorBringPeg
               DMManipulatorInsertBall  DMManipulatorInsertPeg
  pendulum     DMPendulum
  point_mass   DMPointMassEasy  DMPointMassHard
  quadruped    DMQuadrupedWalk  DMQuadrupedRun
  reacher      DMReacherEasy  DMReacherHard
  stacker      DMStacker2  DMStacker4
  swimmer      DMSwimmer6  DMSwimmer15
  walker       DMWalkerStand  DMWalkerWalk  DMWalkerRun

WHAT THIS IS FOR, and what it cannot tell you. It answers "is the model built
and posed the way I think" — geometry, joint axes, ranges, the reset pose, and
whether anything falls through the floor. It does NOT check parity with
MuJoCo; that is what `tests/dm_control/` is for, and a model can look perfect
while its dynamics differ. Use it to catch the errors that are obvious to an
eye and invisible to a tolerance: a limb pointing the wrong way, a geom at the
origin, a robot spawned underground.

⚠ ACTIONS ARE NOT A POLICY. The drive modes below exist to move every joint so
you can see it articulate, not to accomplish the task. A tumbling humanoid
under RANDOM torque is the expected picture, not a bug.

CONTROLS (renderer window)
  task list (right)     click any of the 39 tasks to switch to it
  control panel (left)  drive mode, action scale, reset
  mouse drag / scroll   orbit, zoom
  D / [ / ] / N / 0     drive, scale down/up, reset episode, zero torque
  1-9                   switch camera (if the model declares several)
  close window          quit

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it. It is CPU physics on purpose: one env at 60 Hz needs no GPU, and
the fields facade still wants a `DeviceContext` for host staging.
"""

from std.random import seed, random_float64
from std.math import sin, pi, min, max
from std.sys import argv
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_env import Phyics3dEnv, Phyics3dEnvConfig
from mojo_rl.physics3d.model import ModelDefLike
from mojo_rl.render.types import Color
from mojo_rl.render.ui import (
    UI, UI_ROW_H_SMALL, ui_apply_key, UI_KEY_ESCAPE, UI_KEY_RETURN,
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
    DMQuadrupedWalkModel, DMQuadrupedRunModel,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_config import (
    DMQuadrupedWalkConfig, DMQuadrupedRunConfig,
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


# ── drive mode — RUNTIME, so all three live in one binary ────────────────────
#
#   zero    no torque. The honest reset pose plus gravity: use this to see
#           where the model actually settles, and to spot a robot that spawns
#           intersecting the floor.
#   random  uniform in [-1, 1] per actuator, resampled every `HOLD_STEPS`.
#           Shakes every joint; good for spotting a joint that cannot move or
#           one with no range limit.
#   sweep   a slow out-of-phase sine per actuator (DEFAULT). The clearest way
#           to read joint AXES and RANGES, because each joint traces its full
#           arc smoothly instead of jittering.
#
# Selected by argv so switching costs nothing — only the ENV alias above needs
# a rebuild, and making DRIVE comptime would additionally warn on every
# unreachable branch:
#
#   pixi run mojo run -I . examples/dm_control/dm_viewer.mojo sweep
#   pixi run mojo run -I . examples/dm_control/dm_viewer.mojo random 0.4
#
# The optional second argument is the action scale.
comptime DRIVE_ZERO: Int = 0
comptime DRIVE_RANDOM: Int = 1
comptime DRIVE_SWEEP: Int = 2

comptime HOLD_STEPS: Int = 25       # RANDOM: steps between resamples
comptime N_TASKS: Int = 39
comptime SWEEP_PERIOD: Float64 = 120.0  # SWEEP: steps per full cycle
comptime EPISODE_STEPS: Int = 1000  # auto-reset cadence
comptime FRAME_DELAY_MS: Int = 16   # ~60 FPS
comptime SEED: Int = 0


def _task_names() -> List[String]:
    """The 39 tasks, in the order `_dispatch` indexes them.

    ⚠ THIS LIST AND `_dispatch` ARE POSITIONALLY COUPLED. Index i here must be
    the arm `st.task == i` there; a mismatch shows up as clicking one robot and
    getting another, which is confusing precisely because everything still
    works. `_task_index` is the only lookup, so argv names come from here too
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
    t.append(String("manipulator_bring_ball"))
    t.append(String("manipulator_bring_peg"))
    t.append(String("manipulator_insert_ball"))
    t.append(String("manipulator_insert_peg"))
    t.append(String("pendulum_swingup"))
    t.append(String("point_mass_easy"))
    t.append(String("point_mass_hard"))
    t.append(String("quadruped_walk"))
    t.append(String("quadruped_run"))
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


def _domain_names() -> List[String]:
    """The 16 domains, in the order `_task_domain` indexes them."""
    var d = List[String]()
    d.append(String("acrobot"))
    d.append(String("ball_in_cup"))
    d.append(String("cartpole"))
    d.append(String("cheetah"))
    d.append(String("finger"))
    d.append(String("fish"))
    d.append(String("hopper"))
    d.append(String("humanoid"))
    d.append(String("manipulator"))
    d.append(String("pendulum"))
    d.append(String("point_mass"))
    d.append(String("quadruped"))
    d.append(String("reacher"))
    d.append(String("stacker"))
    d.append(String("swimmer"))
    d.append(String("walker"))
    return d^


def _task_domain() -> List[Int]:
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
    for _ in range(4):
        t.append(8)   # manipulator
    t.append(9)       # pendulum
    for _ in range(2):
        t.append(10)  # point_mass
    for _ in range(2):
        t.append(11)  # quadruped
    for _ in range(2):
        t.append(12)  # reacher
    for _ in range(2):
        t.append(13)  # stacker
    for _ in range(2):
        t.append(14)  # swimmer
    for _ in range(3):
        t.append(15)  # walker
    return t^


def _contains(haystack: String, needle: String) -> Bool:
    """Substring test — `find` returning -1 is the whole implementation."""
    if needle.byte_length() == 0:
        return True
    return haystack.find(needle) != -1


def _task_index(name: String) -> Int:
    """Task id for an argv name, or -1."""
    var names = _task_names()
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


@fieldwise_init
struct ViewerState(Copyable, Movable):
    """Everything that must SURVIVE a task switch.

    A switch tears the env down, so anything held in `_view`'s locals is lost
    with it. Drive mode and action scale live here so that picking a new robot
    does not silently reset how it is being driven — the single most annoying
    way a picker can behave.
    """

    var task: Int
    """Task to run; `_view` overwrites it when a list row is clicked."""
    var drive: Int
    var scale: Float64
    var quit: Bool
    """Set when the window closed for real, as opposed to for a switch."""
    var filter: String
    """Task-name filter. Survives a switch so the list does not reset itself
    under the user the moment they pick something from it."""
    var open_domain: Int
    """Expanded domain, or -1. One at a time: the sidebar has room for every
    header plus one group, and an accordion needs no scrolling."""


def _parse_drive(name: String) -> Int:
    if name == "zero":
        return DRIVE_ZERO
    if name == "random":
        return DRIVE_RANDOM
    return DRIVE_SWEEP


def _drive_name(d: Int) -> String:
    if d == DRIVE_ZERO:
        return String("zero")
    if d == DRIVE_RANDOM:
        return String("random")
    return String("sweep")


def _fmt2(v: Float64) -> String:
    """Two decimals without a formatting library."""
    var scaled = Int(v * 100.0 + (0.5 if v >= 0 else -0.5))
    var whole = scaled // 100
    var frac = scaled % 100
    if frac < 0:
        frac = -frac
    var f = String(frac) if frac >= 10 else "0" + String(frac)
    return String(whole) + "." + f


def _view[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig
](name: String, mut st: ViewerState) raises:
    """The viewer loop for one task, running until it quits or switches.

    Returns through `st`: either `st.quit` (the window was closed) or
    `st.task` now naming a DIFFERENT task, which `main` then launches. It
    cannot simply call itself with the new task — each task is a separate
    compile-time instantiation, so recursion would try to instantiate all 39
    inside all 39.

    ⚠ PARAMETERISED ON MODEL+CONFIG, NOT ON THE ENV TYPE. The obvious
    factoring — `def _view[E: BoxContinuousActionEnv & RenderableEnv](mut env: E)`
    — does not compile: `E.ActionType()` cannot be constructed through the
    trait bound ("no matching function in initialization"). Building the
    concrete `Phyics3dEnv[...]` inside the function makes `E.ActionType` a real
    type again, and costs nothing.
    """
    comptime E = Phyics3dEnv[MODEL, CONFIG, DT, False]
    comptime ACT_DIM = E.ACTION_DIM

    print("=" * 66)
    print("dm_control viewer —", name)
    print("=" * 66)
    print("  obs dim      =", E.OBS_DIM)
    print("  action dim   =", ACT_DIM)
    if st.drive == DRIVE_ZERO:
        print("  drive        = zero (reset pose + gravity)")
    elif st.drive == DRIVE_RANDOM:
        print("  drive        = random, scale", st.scale)
    else:
        print("  drive        = sweep, scale", st.scale)
    print("  pick another task in the window, or close it to quit")
    print("=" * 66)

    var ctx = DeviceContext()
    var env = E(ctx)
    _ = env.reset()

    if not env.init_renderer():
        print("No renderer available — is SDL3 present?")
        st.quit = True
        return

    # ── sidebar layout ───────────────────────────────────────────────────
    # 300 px is set by the longest task name: "manipulator_insert_ball" is 23
    # characters, 8 px each at text scale 1, plus padding. Scale 2 would need
    # 39 * 22 = 858 px of list height in a 720 px window, so the list would
    # have to scroll; at scale 1 all 39 fit with room for the controls under
    # them, and nothing has to scroll.
    comptime SIDEBAR_W = Float32(300.0)
    comptime PAD = Float32(10.0)
    comptime INNER_W = SIDEBAR_W - 2.0 * PAD
    comptime TREE_TOP = Float32(80.0)
    comptime CTRL_TOP = Float32(556.0)
    comptime TREE_ROWS = 32
    """Rows the tree region fits: (CTRL_TOP - TREE_TOP) / 14, minus a margin.
    16 headers plus the largest group (cartpole, 6) is 22, so the accordion
    never needs to scroll; only a very loose filter can overflow, and that
    case says so rather than silently truncating."""

    env.set_ui_sidebar_width(Int(SIDEBAR_W))

    var tasks = _task_names()
    var domains = _domain_names()
    var task_domain = _task_domain()
    var filter_focused = False
    var held = E.ActionType()
    var step_i = 0
    var episode = 0
    var ep_return = Float64(0)
    var live_drive = st.drive
    var live_scale = st.scale
    var switching = False

    while env.is_renderer_open():
        # ── live keys ────────────────────────────────────────────────────
        # Only keys the renderer does not already bind. Taken: ESC, 1-9,
        # SPACE, RIGHT, R, S, V. `renderer_take_key` clears on read, so each
        # press fires once.
        var k = env.renderer_take_key()
        if filter_focused:
            # While the field has focus the renderer forwards EVERY keycode
            # (see `set_text_input_mode`), so the shortcuts below must not run
            # — otherwise typing "reacher" would also cycle the drive mode.
            if k == UI_KEY_ESCAPE or k == UI_KEY_RETURN:
                filter_focused = False
                env.renderer_set_text_input_mode(False)
            elif k != 0:
                _ = ui_apply_key(st.filter, k)
        elif k == 0x44 or k == 0x64:  # D — cycle drive mode
            live_drive = (live_drive + 1) % 3
        elif k == 0x4E or k == 0x6E:  # N — restart the episode
            _ = env.reset()
            step_i = 0
            ep_return = 0.0
        elif k == 0x5D:  # ] — more torque
            live_scale = min(live_scale * 1.25, 8.0)
        elif k == 0x5B:  # [ — less torque
            live_scale = max(live_scale * 0.8, 0.01)
        elif k == 0x30:  # 0 — zero torque, the fastest way back to rest
            live_drive = DRIVE_ZERO

        # ── sidebar ──────────────────────────────────────────────────────
        # Everything lives in the RESERVED strip: `set_ui_sidebar_width` insets
        # the 3D viewport and corrects the camera aspect, so these widgets sit
        # beside the scene rather than on top of it.
        #
        # Immediate mode: each widget both draws and answers, so the sidebar is
        # rebuilt from scratch every frame and holds no state of its own — the
        # filter text and the expanded domain live in `ViewerState`, which is
        # also what carries them across a task switch.
        var ui = UI(
            env.renderer_mouse_x(),
            env.renderer_mouse_y(),
            env.renderer_take_click(),
        )
        # Slab first, so its own contents hit-test above it: a click is
        # consumed by the first widget that wants it.
        ui.panel(0.0, 0.0, SIDEBAR_W, Float32(E.RENDER_HEIGHT),
                 Color(16, 18, 26, 255))
        ui.label(PAD, 8.0, String("dm_control"))
        ui.label(PAD, 30.0, name, Color(120, 230, 255, 255), 1)
        ui.label(
            PAD, 42.0,
            String("ep ") + String(episode) + String("  step ")
            + String(step_i) + String("  return ") + _fmt2(ep_return),
            Color(150, 165, 190, 255), 1,
        )

        # ── filter ───────────────────────────────────────────────────────
        if ui.text_input(PAD, 56.0, INNER_W, 18.0, st.filter, filter_focused,
                         String("filter tasks...")):
            filter_focused = not filter_focused
            env.renderer_set_text_input_mode(filter_focused)

        # ── task tree ────────────────────────────────────────────────────
        # A filter flattens the tree: with a query, matching tasks are listed
        # directly, because making the user expand a domain to reach a result
        # they already named would defeat the search.
        var picked = -1
        var ty = TREE_TOP
        if st.filter.byte_length() > 0:
            var shown = List[String]()
            var shown_ids = List[Int]()
            for i in range(N_TASKS):
                if _contains(tasks[i], st.filter):
                    shown.append(tasks[i])
                    shown_ids.append(i)
            if len(shown) == 0:
                ui.label(PAD, ty, String("no match"),
                         Color(150, 165, 190, 255), 1)
            else:
                var hit = ui.list_select(
                    PAD, ty, INNER_W, TREE_ROWS, shown, 0, -1,
                    text_scale=1, row_h=UI_ROW_H_SMALL,
                )
                if hit >= 0:
                    picked = shown_ids[hit]
                if len(shown) > TREE_ROWS:
                    ui.label(
                        PAD, ty + Float32(TREE_ROWS) * UI_ROW_H_SMALL,
                        String("+") + String(len(shown) - TREE_ROWS)
                        + String(" more — narrow the filter"),
                        Color(150, 165, 190, 255), 1,
                    )
        else:
            for d in range(len(domains)):
                var n_in = 0
                for i in range(N_TASKS):
                    if task_domain[i] == d:
                        n_in += 1
                if ui.tree_header(PAD, ty, INNER_W, UI_ROW_H_SMALL - 1.0,
                                  domains[d], st.open_domain == d, n_in):
                    st.open_domain = -1 if st.open_domain == d else d
                ty += UI_ROW_H_SMALL
                if st.open_domain == d:
                    for i in range(N_TASKS):
                        if task_domain[i] != d:
                            continue
                        if ui.button(PAD + 12.0, ty, INNER_W - 12.0,
                                     UI_ROW_H_SMALL - 2.0, tasks[i],
                                     i == st.task, 1):
                            picked = i
                        ty += UI_ROW_H_SMALL

        # ── controls, pinned to the bottom ───────────────────────────────
        var cy = CTRL_TOP
        ui.label(PAD, cy, String("drive   [D]"), Color(150, 165, 190, 255), 1)
        cy += 14.0
        if ui.button(PAD, cy, 90.0, 22.0, String("zero"),
                     live_drive == DRIVE_ZERO, 1):
            live_drive = DRIVE_ZERO
        if ui.button(PAD + 95.0, cy, 90.0, 22.0, String("random"),
                     live_drive == DRIVE_RANDOM, 1):
            live_drive = DRIVE_RANDOM
        if ui.button(PAD + 190.0, cy, 90.0, 22.0, String("sweep"),
                     live_drive == DRIVE_SWEEP, 1):
            live_drive = DRIVE_SWEEP

        cy += 28.0
        ui.label(PAD, cy + 6.0, String("scale ") + _fmt2(live_scale)
                 + String("   [ ] ]"), Color(150, 165, 190, 255), 1)
        if ui.button(PAD + 210.0, cy, 34.0, 22.0, String("-"), False, 1):
            live_scale = max(live_scale * 0.8, 0.01)
        if ui.button(PAD + 248.0, cy, 32.0, 22.0, String("+"), False, 1):
            live_scale = min(live_scale * 1.25, 8.0)

        cy += 28.0
        if ui.button(PAD, cy, 137.0, 22.0, String("reset ep  [N]"), False, 1):
            _ = env.reset()
            step_i = 0
            ep_return = 0.0
        if ui.button(PAD + 143.0, cy, 137.0, 22.0, String("zero now  [0]"),
                     False, 1):
            live_drive = DRIVE_ZERO
            live_scale = 1.0

        # Camera row — one button per camera the model declares. Models carry
        # between one and four, so the row is sized from the model, not fixed.
        cy += 28.0
        var n_cam = env.renderer_n_cameras()
        ui.label(PAD, cy + 6.0, String("cam"), Color(150, 165, 190, 255), 1)
        var cam_w = (INNER_W - 40.0) / Float32(n_cam if n_cam > 0 else 1)
        for c in range(n_cam):
            if ui.button(PAD + 36.0 + Float32(c) * cam_w, cy, cam_w - 3.0,
                         22.0, String(c + 1),
                         c == env.renderer_current_camera(), 1):
                env.renderer_request_camera(c)

        cy += 28.0
        if ui.button(PAD, cy, 90.0, 22.0,
                     String("pause") if not env.renderer_paused()
                     else String("resume"), env.renderer_paused(), 1):
            env.renderer_toggle_pause()
        if ui.button(PAD + 95.0, cy, 90.0, 22.0, String("shot  [S]"),
                     False, 1):
            env.renderer_request_screenshot()
        var rec = env.renderer_is_recording()
        if ui.button(
            PAD + 190.0, cy, 90.0, 22.0,
            # Labels are sized to the button: 90 px holds 11 characters at
            # 8 px each, and "record  [V]" is exactly one over, so it spilled
            # past its own edge.
            (String("rec ") + String(env.renderer_recording_frames()))
            if rec else String("rec  [V]"),
            rec, 1,
        ):
            env.renderer_toggle_recording()

        env.set_ui(ui.rects, ui.texts)

        # Leave BEFORE stepping: the new task's env has to be built by `main`,
        # and there is nothing to gain from one more frame of the old one.
        if picked >= 0 and picked != st.task:
            st.task = picked
            switching = True
            break

        var action = E.ActionType()
        if live_drive == DRIVE_RANDOM:
            if step_i % HOLD_STEPS == 0:
                for a in range(ACT_DIM):
                    held.data[a] = random_float64(-1.0, 1.0) * live_scale
            for a in range(ACT_DIM):
                action.data[a] = held.data[a]
        elif live_drive == DRIVE_SWEEP:
            var t = Float64(step_i) / SWEEP_PERIOD
            for a in range(ACT_DIM):
                var phase = Float64(a) / Float64(ACT_DIM if ACT_DIM > 0 else 1)
                action.data[a] = sin(2.0 * pi * (t + phase)) * live_scale

        var out = env.step(action)
        ep_return += Float64(out[1])
        env.render_frame()
        env.renderer_delay(FRAME_DELAY_MS)
        if env.check_renderer_quit():
            break
        step_i += 1
        if out[2] or step_i >= EPISODE_STEPS:
            episode += 1
            print("  episode", episode, "ended after", step_i,
                  "steps, return =", ep_return)
            _ = env.reset()
            step_i = 0
            ep_return = 0.0

    # Drive settings outlive the env; the window does not.
    st.drive = live_drive
    st.scale = live_scale
    st.quit = not switching
    env.close_renderer()
    if switching:
        print("  switching to", tasks[st.task])
    else:
        print("viewer closed")


def _dispatch(mut st: ViewerState) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    ⚠ INDEX ORDER MUST MATCH `_task_names`. This is the one place all 39
    compile-time instantiations are named, which is what makes the binary
    take ~8 minutes to build and what lets the picker exist at all.
    """
    var name = _task_names()[st.task]
    if st.task == 0:
        _view[DMAcrobotModel, DMAcrobotConfig[False]](name, st)
    elif st.task == 1:
        _view[DMAcrobotModel, DMAcrobotConfig[True]](name, st)
    elif st.task == 2:
        _view[DMBallInCupModel, DMBallInCupConfig](name, st)
    elif st.task == 3:
        _view[DMCartpole1Model, DMCartpoleConfig[1, False, False]](name, st)
    elif st.task == 4:
        _view[DMCartpole1Model, DMCartpoleConfig[1, False, True]](name, st)
    elif st.task == 5:
        _view[DMCartpole1Model, DMCartpoleConfig[1, True, False]](name, st)
    elif st.task == 6:
        _view[DMCartpole1Model, DMCartpoleConfig[1, True, True]](name, st)
    elif st.task == 7:
        _view[DMCartpole2Model, DMCartpoleConfig[2, True, False]](name, st)
    elif st.task == 8:
        _view[DMCartpole3Model, DMCartpoleConfig[3, True, False]](name, st)
    elif st.task == 9:
        _view[DMCheetahModel, DMCheetahConfig](name, st)
    elif st.task == 10:
        _view[DMFingerSpinModel, DMFingerSpinConfig](name, st)
    elif st.task == 11:
        _view[DMFingerTurnModel, DMFingerTurnConfig[0.07]](name, st)
    elif st.task == 12:
        _view[DMFingerTurnModel, DMFingerTurnConfig[0.03]](name, st)
    elif st.task == 13:
        _view[DMFishUprightModel, DMFishUprightConfig](name, st)
    elif st.task == 14:
        _view[DMFishSwimModel, DMFishSwimConfig](name, st)
    elif st.task == 15:
        _view[DMHopperModel, DMHopperConfig[False]](name, st)
    elif st.task == 16:
        _view[DMHopperModel, DMHopperConfig[True]](name, st)
    elif st.task == 17:
        _view[DMHumanoidModel, DMHumanoidConfig[0.0, False]](name, st)
    elif st.task == 18:
        _view[DMHumanoidModel, DMHumanoidConfig[WALK_SPEED, False]](name, st)
    elif st.task == 19:
        _view[DMHumanoidModel, DMHumanoidConfig[RUN_SPEED, False]](name, st)
    elif st.task == 20:
        _view[DMHumanoidPureModel, DMHumanoidConfig[RUN_SPEED, True]](name, st)
    elif st.task == 21:
        _view[
            DMManipulatorBringBallModel, DMManipulatorBringBallConfig
        ](name, st)
    elif st.task == 22:
        _view[
            DMManipulatorBringPegModel, DMManipulatorBringPegConfig
        ](name, st)
    elif st.task == 23:
        _view[
            DMManipulatorInsertBallModel, DMManipulatorInsertBallConfig
        ](name, st)
    elif st.task == 24:
        _view[
            DMManipulatorInsertPegModel, DMManipulatorInsertPegConfig
        ](name, st)
    elif st.task == 25:
        _view[DMPendulumModel, DMPendulumConfig](name, st)
    elif st.task == 26:
        _view[DMPointMassModel, DMPointMassConfig](name, st)
    elif st.task == 27:
        _view[DMPointMassModel, DMPointMassHardConfig](name, st)
    elif st.task == 28:
        _view[DMQuadrupedWalkModel, DMQuadrupedWalkConfig](name, st)
    elif st.task == 29:
        _view[DMQuadrupedRunModel, DMQuadrupedRunConfig](name, st)
    elif st.task == 30:
        _view[DMReacherModel, DMReacherConfig[0.05]](name, st)
    elif st.task == 31:
        _view[DMReacherModel, DMReacherConfig[0.015]](name, st)
    elif st.task == 32:
        _view[DMStacker2Model, DMStacker2Config](name, st)
    elif st.task == 33:
        _view[DMStacker4Model, DMStacker4Config](name, st)
    elif st.task == 34:
        _view[DMSwimmer6Model, DMSwimmerConfig](name, st)
    elif st.task == 35:
        _view[DMSwimmer15Model, DMSwimmerConfig](name, st)
    elif st.task == 36:
        _view[DMWalkerModel, DMWalkerConfig[0.0]](name, st)
    elif st.task == 37:
        _view[DMWalkerModel, DMWalkerConfig[1.0]](name, st)
    elif st.task == 38:
        _view[DMWalkerModel, DMWalkerConfig[8.0]](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def main() raises:
    seed(SEED)
    var args = argv()
    var start = String(args[1]) if len(args) > 1 else String("quadruped_walk")
    var task = _task_index(start)
    if task < 0:
        print("unknown task:", start, "— the 39 registered tasks are:")
        var names = _task_names()
        for i in range(len(names)):
            print("   ", names[i])
        return

    var drive = _parse_drive(String(args[2])) if len(args) > 2 else DRIVE_SWEEP
    var act_scale = Float64(1.0)
    if len(args) > 3:
        try:
            act_scale = Float64(String(args[3]))
        except:
            print("bad scale, using 1.0")

    # One task runs at a time; picking another in the window ends that task's
    # loop and comes back here to build the next one. `st` is what crosses the
    # gap, since the env and its window do not.
    var st = ViewerState(task, drive, act_scale, False, String(""), -1)
    while not st.quit:
        _dispatch(st)
