"""Interactive viewer for the SO-101 task families, with a Dear ImGui sidebar.

    pixi run build-imgui                                        # ONCE
    pixi run mojo run -I . examples/so101/tower_viewer_imgui.mojo
    pixi run mojo run -I . examples/so101/tower_viewer_imgui.mojo so101_tower_lift_brick
    pixi run mojo run -I . examples/so101/tower_viewer_imgui.mojo so101_gather_bricks sweep 0.5

argv only picks which task opens FIRST; every task is selectable in the
window. The optional second argument is the drive mode (zero | random | sweep)
and the third is the action scale. `examples/robots/so_arm_viewer_imgui.mojo`
is the same shape for the bare arms; this one is for the FAMILIES — the
`so101-tower` rig (`docs/camera-rig.md`, both cameras) and the tabletop.

⚠ `pixi run build-imgui` IS A PREREQUISITE, and its absence is a RUNTIME
failure: the shim is dlopen'ed and the loader aborts rather than raising, so
`imgui_shim_available()` is checked up front. RUN FROM THE REPO ROOT (mesh
paths), ON THE LAPTOP (it opens an SDL3 window), on the CPU (one arm at 60 Hz).

## WHAT IS DIFFERENT FROM THE ARM VIEWER

A family's free slots are PARKED 50 m up in the composed scene's `qpos0`, and
`Phyics3dEnv.reset` restores `qpos0`. On the GPU path `init_qpos_gpu` places
them; on the host the eval and the task viewer call `tasks/sampler` +
`tasks/reset.reset_slots`. `viewer_core.run_view` did neither, so a family
viewed through it showed props falling from the sky. `ViewerState.reset_qpos`
is the hook: `posed_qpos` below draws the task's `init=` placements ONCE, from
the family's GENERATED (or hand-written) placement table — region sites, park
poses, slot addresses — and the viewer writes that `qpos` through `obs_at`
after every reset. The same layout every episode, on purpose: a viewer is for
looking at one scene.

The task's tape, mask, init and shaping words go into `meta` after every
reset too (`ViewerState.reset_meta_*`, from `tasks/posed_reset`), so the
observation plot shows what a policy would see. Nothing here trains.

⚠⚠ THE DRIVE MODES COMMAND [-1, 1] PER JOINT, mapped onto each actuator's own
`ctrlrange` (`NORMALIZED_ACTIONS` is True on these configs, unlike the bare
arms' — see `So101FamilyConfig`). `zero` is therefore the CENTRE of every
range, gripper half open, not "no torque". `sweep` at scale 1.0 sweeps every
joint end to end.

CAMERAS. The tower family has two `<camera>`s: `1` is `robot_wrist_cam`, `2`
is `tower_overhead_cam`, `0`/`free` is the mouse-driven one it opens with.
The tabletop family has the stock arm's `wrist_cam` only.
"""

from std.random import seed
from std.sys import argv

from mojo_rl.envs.dm_control.viewer_core import (
    ViewerState, run_view, task_index, parse_drive, DRIVE_SWEEP,
)
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D

from mojo_rl.tasks.posed_reset import posed_qpos, task_meta_words
from mojo_rl.tasks.placement.so101_tower import So101TowerPlacement
from mojo_rl.tasks.family_config import (
    So101TabletopConfig, So101TabletopPlacement, So101TowerConfig,
)
from mojo_rl.tasks.so101_tabletop_xml import So101TabletopModel
from mojo_rl.tasks.so101_tower_xml import So101TowerModel

comptime SEED: Int = 0
comptime N_TOWER_TASKS = 3


def task_names() -> List[String]:
    var t = List[String]()
    # ⚠ THE FIRST `N_TOWER_TASKS` ARE THE TOWER FAMILY'S; `dispatch` counts.
    t.append(String("so101_tower_cube_in_bowl"))
    t.append(String("so101_tower_lift_brick"))
    t.append(String("so101_tower_reach_clear"))
    t.append(String("so101_lift_brick"))
    t.append(String("so101_gather_bricks"))
    t.append(String("so101_reach_clear"))
    return t^


def domain_names() -> List[String]:
    var d = List[String]()
    d.append(String("so101_tower"))
    d.append(String("so101_tabletop"))
    return d^


def task_domain() -> List[Int]:
    var t = List[Int]()
    for i in range(len(task_names())):
        t.append(0 if i < N_TOWER_TASKS else 1)
    return t^


def dispatch(mut st: ViewerState) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    ⚠ THE TABLE AND THIS FUNCTION ARE POSITIONALLY COUPLED, as in every front
    end: index i in `task_names` must be the env `st.task == i` selects.
    """
    var name = task_names()[st.task]
    if st.task < N_TOWER_TASKS:
        st.reset_qpos = posed_qpos[So101TowerPlacement](
            name, String("so101_tower"), So101TowerConfig.SLOT_RADIUS
        )
        var mw = task_meta_words(
            name, String("so101_tower"), So101TowerConfig.SHAPE_W_GOAL,
            So101TowerConfig.SHAPE_W_REACH, So101TowerConfig.GOAL_MARGIN,
            So101TowerConfig.REACH_MARGIN,
        )
        st.reset_meta_idx = mw[0].copy()
        st.reset_meta_val = mw[1].copy()
        run_view[So101TowerModel, So101TowerConfig](name, st)
    elif st.task < len(task_names()):
        st.reset_qpos = posed_qpos[So101TabletopPlacement](
            name, String("so101_tabletop"), So101TabletopConfig.SLOT_RADIUS
        )
        var mw = task_meta_words(
            name, String("so101_tabletop"), So101TabletopConfig.SHAPE_W_GOAL,
            So101TabletopConfig.SHAPE_W_REACH, So101TabletopConfig.GOAL_MARGIN,
            So101TabletopConfig.REACH_MARGIN,
        )
        st.reset_meta_idx = mw[0].copy()
        st.reset_meta_val = mw[1].copy()
        run_view[So101TabletopModel, So101TabletopConfig](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    var args = argv()
    var start = (
        String(args[1]) if len(args) > 1 else String("so101_tower_cube_in_bowl")
    )
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
    # Camera 0 is the wrist camera on both families — see the arm viewer for
    # why a body-attached camera 0 must not be the one the viewer opens with.
    st.free_camera = True
    while not st.quit:
        dispatch(st)

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
