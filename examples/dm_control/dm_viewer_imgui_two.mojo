"""The ImGui dm_control viewer with TWO tasks — the fast build to iterate on.

    pixi run build-imgui                                          # ONCE
    pixi run mojo run -I . examples/dm_control/dm_viewer_imgui_two.mojo
    pixi run mojo run -I . examples/dm_control/dm_viewer_imgui_two.mojo walker_run

WHY THIS EXISTS. `dm_viewer_imgui.mojo` instantiates `run_view` 43 times and
takes ten-plus minutes to build, which is not a loop anyone iterates in. This
is the SAME code — same `viewer_core`, same sidebar, same run loop, same
window handoff — over a two-entry task table, so it builds in a fraction of
the time. Work on the viewer here; compile the 43-task one to confirm.

⚠ IT IS ALSO THE GATE FOR THE WINDOW HANDOFF, which is what the two tasks are
for. Switching between them exercises `RendererHandoff` end to end: drag the
window to a second monitor, resize it, switch task, and the window must NOT
move, resize or blink — only the robot changes. Two DIFFERENT domains on
purpose (a 6-DOF cheetah and a 6-DOF walker), so the geom caches really are
rebuilt rather than trivially reused.

⚠ THE TABLE AND `dispatch` ARE POSITIONALLY COUPLED, exactly as in the real
viewer: index i in `task_names` must be the arm `st.task == i` below.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it.
"""

from std.random import seed
from std.sys import argv

from mojo_rl.envs.dm_control.viewer_core import (
    ViewerState, run_view, task_index, parse_drive, DRIVE_SWEEP,
)
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D

from mojo_rl.envs.dm_control.cheetah.cheetah_xml import DMCheetahModel
from mojo_rl.envs.dm_control.cheetah.cheetah_config import DMCheetahConfig
from mojo_rl.envs.dm_control.walker.walker_xml import DMWalkerModel
from mojo_rl.envs.dm_control.walker.walker_config import DMWalkerConfig

comptime SEED: Int = 0


def task_names() -> List[String]:
    var t = List[String]()
    t.append(String("cheetah_run"))
    t.append(String("walker_run"))
    return t^


def domain_names() -> List[String]:
    var d = List[String]()
    d.append(String("cheetah"))
    d.append(String("walker"))
    return d^


def task_domain() -> List[Int]:
    var t = List[Int]()
    t.append(0)
    t.append(1)
    return t^


def dispatch(mut st: ViewerState) raises:
    """Run whichever task `st.task` names, and return when it wants another."""
    var name = task_names()[st.task]
    if st.task == 0:
        run_view[DMCheetahModel, DMCheetahConfig](name, st)
    elif st.task == 1:
        run_view[DMWalkerModel, DMWalkerConfig[8.0]](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    var args = argv()
    var start = String(args[1]) if len(args) > 1 else String("cheetah_run")
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
    while not st.quit:
        dispatch(st)

    # The last task's window is handed OUT, not closed, whenever the loop ends
    # on a switch — and `dispatch`'s unknown-index arm ends it without one.
    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
