"""Interactive dm_control viewer with a Dear ImGui sidebar — all 48 tasks.

    pixi run build-imgui                                   # ONCE
    pixi run mojo run -I . examples/dm_control/dm_viewer_imgui.mojo
    pixi run mojo run -I . examples/dm_control/dm_viewer_imgui.mojo cheetah_run
    pixi run mojo run -I . examples/dm_control/dm_viewer_imgui.mojo walker_run random 0.4

argv only picks which task opens FIRST; every task is selectable in the window.
The optional second argument is the drive mode (zero | random | sweep) and the
third is the action scale.

⚠ `pixi run build-imgui` IS A PREREQUISITE, and its absence is a RUNTIME
failure, not a compile error — the shim is loaded by dlopen. `run_viewer`
checks for it up front and says so rather than aborting mid-frame.

THE VIEWER ITSELF LIVES IN `mojo_rl.envs.dm_control.viewer`. This file is
argv parsing and nothing else, so the logic can be precompiled with the
package and reused by other front ends. Read that module's header for the
controls, the drive modes, what the tool can and cannot tell you, and why
`build_sidebar` is deliberately not generic.

RELATIONSHIP TO `dm_viewer.mojo`. That one stays; this is a parallel port, not
a replacement. Same physics, same renderer, same task-switch machinery — the
difference is a Dear ImGui sidebar instead of the hand-rolled
`mojo_rl/render/ui.mojo` widgets. Keeping both means the ImGui dependency
stays optional.

⚠ TWO TASKS ARE ABSENT, AND FOR THE SAME REASON: `dog_fetch` and the dog-sized
sibling of `quadruped_fetch` both keep a ball and a target, which adds a free
joint and a second free-jointed object to collide. The other four dog tasks ARE
registered — an older note here said dog was not, which stopped being true.

⚠ `quadruped_escape` DRAWS ITS TERRAIN, and it is the only task here that does.
Its heightfield is rewritten on every reset, so what you see is that episode's
bowl and not a fixed asset — see `physics3d/model/hfield_surface.mojo` for why
the drawn surface and the one the rangefinders hit come from one description.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it. CPU physics on purpose: one env at 60 Hz needs no GPU.
"""

from std.random import seed
from std.sys import argv

from mojo_rl.envs.dm_control.viewer import (
    run_viewer, task_index, parse_drive, print_task_list, DRIVE_SWEEP,
)

comptime SEED: Int = 0


def main() raises:
    seed(SEED)
    var args = argv()
    var start = String(args[1]) if len(args) > 1 else String("quadruped_walk")
    var task = task_index(start)
    if task < 0:
        print("unknown task:", start, "— the 48 registered tasks are:")
        print_task_list()
        return

    var drive = parse_drive(String(args[2])) if len(args) > 2 else DRIVE_SWEEP
    var scale = Float64(1.0)
    if len(args) > 3:
        try:
            scale = Float64(String(args[3]))
        except:
            print("bad scale, using 1.0")

    run_viewer(task, drive, scale)
