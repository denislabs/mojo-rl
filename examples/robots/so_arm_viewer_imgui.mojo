"""Interactive SO-ARM viewer with a Dear ImGui sidebar — SO-100 and SO-101.

    pixi run build-imgui                                        # ONCE
    pixi run mojo run -I . examples/robots/so_arm_viewer_imgui.mojo
    pixi run mojo run -I . examples/robots/so_arm_viewer_imgui.mojo so_arm101_reach
    pixi run mojo run -I . examples/robots/so_arm_viewer_imgui.mojo so_arm100_reach sweep 1.5

argv only picks which arm opens FIRST; both are selectable in the window. The
optional second argument is the drive mode (zero | random | sweep) and the
third is the action scale.

⚠ `pixi run build-imgui` IS A PREREQUISITE, and its absence is a RUNTIME
failure, not a compile error — the shim is loaded by dlopen, and the loader
ABORTS the process rather than raising, so `imgui_shim_available()` is checked
up front.

⚠ RUN FROM THE REPO ROOT. Both models reference their meshes by repo-root
relative path (our parser does not implement `<compiler meshdir>`), so a stray
cwd gives you an arm with no collision geometry and a printed warning that is
easy to miss.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it. CPU physics on purpose: one arm at 60 Hz needs no GPU.

THE VIEWER ITSELF IS `mojo_rl.envs.dm_control.viewer_core`, shared verbatim
with the dm_control front ends — same sidebar, same run loop, same window
handoff. This file is a two-entry task table and a `dispatch`, which is the
`dm_viewer_imgui_two.mojo` shape rather than the 47-arm `viewer.mojo` one:
two arms build in a couple of minutes instead of ten-plus.

⚠⚠ THE DRIVE MODES COMMAND JOINT ANGLES IN RADIANS HERE, NOT NORMALISED
TORQUES. Every dm_control task in the other front ends takes a [-1, 1] torque;
these two arms drive `<position>` servos whose `ctrlrange` IS the joint range,
so `sweep` at scale 1.0 sweeps each joint over +/-1 rad and the slider's upper
end (8.0) is clamped to the joint limits by `apply_actions`. That is the
intended picture, not a bug — but it means "scale 1.0" means something
completely different here than it does for the cheetah.

WHAT THIS IS FOR: confirming the model is built and posed the way you think —
geometry, joint axes, ranges, the reset pose, whether the mesh geometry
loaded. It does NOT check parity with MuJoCo; `tests/robots/` does that, and
an arm can look perfect while its dynamics differ. Specifically, the servo
gains were wrong by 50x on SO-100 and absent entirely on SO-101 while this
viewer showed a perfectly plausible arm.

⚠ SO-101 IS THE SLOW ONE, at both compile and run time — 33 280 hull vertices
against SO-100's 2 560. If you are iterating on viewer behaviour, start with
SO-100.
"""

from std.random import seed
from std.sys import argv

from mojo_rl.envs.dm_control.viewer_core import (
    ViewerState, run_view, task_index, parse_drive, DRIVE_SWEEP,
)
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D

from mojo_rl.envs.robots.so_arm100_xml import SoArm100Model
from mojo_rl.envs.robots.so_arm100 import SoArm100ReachConfig
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig

comptime SEED: Int = 0


def task_names() -> List[String]:
    var t = List[String]()
    t.append(String("so_arm100_reach"))
    t.append(String("so_arm101_reach"))
    return t^


def domain_names() -> List[String]:
    var d = List[String]()
    d.append(String("so_arm100"))
    d.append(String("so_arm101"))
    return d^


def task_domain() -> List[Int]:
    var t = List[Int]()
    t.append(0)
    t.append(1)
    return t^


def dispatch(mut st: ViewerState) raises:
    """Run whichever arm `st.task` names, and return when it wants another.

    ⚠ THE TABLE AND THIS FUNCTION ARE POSITIONALLY COUPLED: index i in
    `task_names` must be the arm `st.task == i` selects here. Same coupling as
    the dm_control front ends, and the same failure if it drifts — you get the
    other robot, silently.
    """
    var name = task_names()[st.task]
    if st.task == 0:
        run_view[SoArm100Model, SoArm100ReachConfig](name, st)
    elif st.task == 1:
        run_view[SoArm101Model, SoArm101ReachConfig](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    var args = argv()
    var start = String(args[1]) if len(args) > 1 else String("so_arm100_reach")
    var task = task_index(start, task_names())
    if task < 0:
        print("unknown task:", start, "— this front end registers:")
        var names = task_names()
        for i in range(len(names)):
            print("   ", names[i])
        return

    var drive = parse_drive(String(args[2])) if len(args) > 2 else DRIVE_SWEEP
    var scale = Float64(1.0)
    if len(args) > 3:
        try:
            scale = Float64(String(args[3]))
        except:
            print("bad scale, using 1.0")

    var st = ViewerState(
        task, drive, scale, task_names(), domain_names(), task_domain()
    )
    # ⚠ SO-101's ONLY camera is `<camera name="wrist_cam">`, bolted to the
    # wrist, and `active_camera` starts at 0 — so without this the viewer opens
    # looking down the gripper, re-aimed every frame and immune to the mouse.
    # SO-100 declares no camera at all, where this costs nothing but a 3/4
    # reframe and an honest "free" toggle. Press `1` for the wrist view.
    st.free_camera = True
    while not st.quit:
        dispatch(st)

    # The last task's window is handed OUT, not closed, whenever the loop ends
    # on a switch — and `dispatch`'s unknown-index arm ends it without one.
    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
