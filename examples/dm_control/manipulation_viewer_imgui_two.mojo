"""The ImGui `manipulation` viewer with TWO tasks — the fast build to iterate on.

    pixi run build-imgui                                                  # ONCE
    pixi run mojo run -I . examples/dm_control/manipulation_viewer_imgui_two.mojo
    pixi run mojo run -I . examples/dm_control/manipulation_viewer_imgui_two.mojo stack_2_bricks

WHY THIS EXISTS. `manipulation_viewer_imgui.mojo` instantiates `run_view` 13
times over the largest models in the port and takes tens of minutes to build,
which is not a loop anyone iterates in. This is the SAME code — same
`viewer_core`, same sidebar, same run loop, same window handoff — over a
two-entry table, so it builds in a fraction of the time. Work on the viewer
here; compile the 13-task one to confirm.

Exactly the role `dm_viewer_imgui_two.mojo` plays for the 47 suite tasks.

⚠ THE TWO TASKS ARE CHOSEN TO EXERCISE THE WINDOW HANDOFF, which is what a
two-task front end is really for. `reach_site` has NO props at all (nq 9, one
target site) and `stack_2_bricks` has two Duplos and a welded base (nq 16, 185
geoms), so switching between them genuinely rebuilds the geom caches rather
than trivially reusing them. Drag the window to a second monitor, resize it,
switch task: the window must NOT move, resize or blink — only the scene
changes.

⚠ THE TABLE AND `dispatch` ARE POSITIONALLY COUPLED, exactly as in the real
registry: index i in `task_names` must be the arm `st.task == i` below.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it.
"""

from std.random import seed
from std.sys import argv

from mojo_rl.envs.dm_control.viewer_core import (
    ViewerState, run_view, task_index, parse_drive, DRIVE_ZERO,
)
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D

from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)
from mojo_rl.envs.dm_control.manipulation_reach_config import (
    ReachSiteFeaturesConfig,
)
from mojo_rl.envs.dm_control.manipulation_stack2_def import Stack2BricksModel
from mojo_rl.envs.dm_control.manipulation_stack2_config import (
    Stack2BricksConfig,
)

comptime SEED: Int = 0


def task_names() -> List[String]:
    var t = List[String]()
    t.append(String("reach_site"))
    t.append(String("stack_2_bricks"))
    return t^


def domain_names() -> List[String]:
    var d = List[String]()
    d.append(String("reach"))
    d.append(String("stack"))
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
        run_view[ReachSiteFeaturesModel, ReachSiteFeaturesConfig](name, st)
    elif st.task == 1:
        run_view[Stack2BricksModel, Stack2BricksConfig](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    var args = argv()
    var start = String(args[1]) if len(args) > 1 else String("reach_site")
    var task = task_index(start, task_names())
    if task < 0:
        print("unknown task:", start, "— this front end registers:")
        var names = task_names()
        for i in range(len(names)):
            print("   ", names[i])
        return

    # ⚠ `zero` by default — the reset pose is the subject in manipulation.
    var drive = parse_drive(String(args[2])) if len(args) > 2 else DRIVE_ZERO
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

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
