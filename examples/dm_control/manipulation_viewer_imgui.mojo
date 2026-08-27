"""Interactive `manipulation` viewer with a Dear ImGui sidebar — all 13 tasks.

    pixi run build-imgui                                              # ONCE
    pixi run mojo run -I . examples/dm_control/manipulation_viewer_imgui.mojo
    pixi run mojo run -I . examples/dm_control/manipulation_viewer_imgui.mojo stack_3_bricks
    pixi run mojo run -I . examples/dm_control/manipulation_viewer_imgui.mojo reassemble_5_bricks_random_order zero

argv only picks which task opens FIRST; every task is selectable in the window.
The optional second argument is the drive mode (zero | random | sweep) and the
third is the action scale.

⚠ THE DEFAULT DRIVE IS `zero` HERE, NOT `sweep` AS IN THE SUITE VIEWER, and
that is the point of these tasks rather than a nicety. What is worth looking at
in `manipulation` is the RESET: a rejection-sampled IK pose for the arm, props
placed and settled under gravity, and for `reassemble` a stack assembled by
lining corner holes up with corner studs. `sweep` drives the arm through that
scene and knocks it over within a second. Pick `sweep` deliberately, to watch
the arm articulate.

⚠ `pixi run build-imgui` IS A PREREQUISITE, and its absence is a RUNTIME
failure, not a compile error — the shim is loaded by dlopen.
`run_manipulation_viewer` checks for it up front and says so.

THE VIEWER ITSELF LIVES IN `mojo_rl.envs.dm_control.viewer_manipulation`. This
file is argv parsing and nothing else. Read that module's header for the build
cost, why the reset visibly pauses, and what the tool can and cannot tell you.

⚠⚠ THIS BUILDS SLOWLY — tens of minutes. All 13 arms are the same Jaco plus
props, so each carries a 267-to-431-geom model with nine STL meshes;
`reassemble_5` is the largest model anywhere in this port. It is NOT a loop to
iterate on viewer behaviour in. For that, copy this file, keep two entries in
the table and two arms in `dispatch`, exactly as
`dm_viewer_imgui_two.mojo` does for the suite.

RELATIONSHIP TO `dm_viewer_imgui.mojo`. Parallel, not a replacement: that one
registers the 47 SUITE tasks and this one the 13 MANIPULATION tasks. They share
`viewer_core` (state, sidebar, run loop, window handoff) and nothing else, so
neither pays the other's compile cost.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it. CPU physics on purpose: one env at 60 Hz needs no GPU.
"""

from std.random import seed
from std.sys import argv

from mojo_rl.envs.dm_control.viewer_manipulation import (
    run_manipulation_viewer,
    manipulation_task_index,
    print_manipulation_task_list,
)
from mojo_rl.envs.dm_control.viewer_core import parse_drive, DRIVE_ZERO

comptime SEED: Int = 0


def main() raises:
    seed(SEED)
    var args = argv()
    var start = String(args[1]) if len(args) > 1 else String("stack_3_bricks")
    var task = manipulation_task_index(start)
    if task < 0:
        print("unknown task:", start, "— the 13 registered tasks are:")
        print_manipulation_task_list()
        print()
        print("(the `_features` suffix is optional — both spellings resolve)")
        return

    # ⚠ `zero` by default — see the module docstring. The reset is the subject.
    var drive = parse_drive(String(args[2])) if len(args) > 2 else DRIVE_ZERO
    var scale = Float64(1.0)
    if len(args) > 3:
        try:
            scale = Float64(String(args[3]))
        except:
            print("bad scale, using 1.0")

    run_manipulation_viewer(task, drive, scale)
