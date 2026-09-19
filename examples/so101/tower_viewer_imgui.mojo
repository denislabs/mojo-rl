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

⚠ THE ACTIVE MASK IS NOT SET. `write_task_obs_host` zeroes an inactive slot's
words in the observation plot; with `meta` untouched every slot reads as
inactive there. The physics and the picture are right; only the plotted
observation's slot words are zero. Nothing here trains.

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

from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
)
from mojo_rl.tasks.sampler import sample_placements, RegionFrame, SampleReport
from mojo_rl.tasks.reset import reset_slots, SlotAddress
from mojo_rl.tasks.placement.table import PlacementTable
from mojo_rl.tasks.placement.so101_tower import So101TowerPlacement
from mojo_rl.tasks.family_config import (
    So101TabletopConfig, So101TabletopPlacement, So101TowerConfig,
)
from mojo_rl.tasks.so101_tabletop_xml import So101TabletopModel
from mojo_rl.tasks.so101_tower_xml import So101TowerModel

comptime SEED: Int = 0
comptime DT = DType.float64
comptime FAMILY_DIR = "mojo_rl/tasks/families/"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
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


def posed_qpos[P: PlacementTable](
    task: String, family: String, fallback_radius: Float64
) raises -> List[Float64]:
    """The composed scene's `qpos0` with the task's `init=` placements drawn.

    Everything comes from the placement table `P` — the park poses, the
    region sites' world frames (FK at rest, baked by the generator), the
    slot addresses — so no second model is built and no FK is run here. The
    host sampler is the one the eval uses (`sampler.sample_placements`), at
    seed 0, lane 0. `fallback_radius` is what it uses for a free slot WITHOUT
    `slot_geom=` (every tabletop slot; no tower slot).
    """
    var f = load_family(String(FAMILY_DIR) + family + ".family")
    var t = load_task(String(TASK_DIR) + task + ".task")
    validate_task_against_family(t, f)
    var q0 = List[Float64](length=P.NQ, fill=0.0)
    for j in range(P.N_FREE):
        var adr = P.free_qadr(j)
        q0[adr] = Float64(P.free_park_x[DT](j))
        q0[adr + 1] = Float64(P.free_park_y[DT](j))
        q0[adr + 2] = Float64(P.free_park_z[DT](j))
        q0[adr + 3] = 1.0
    var frames = List[RegionFrame]()
    for r in range(P.N_REGIONS):
        frames.append(
            RegionFrame(
                Float64(P.region_site_x[DT](r)),
                Float64(P.region_site_y[DT](r)),
                Float64(P.region_site_z[DT](r)),
            )
        )
    var radii = List[Float64](length=len(f.slots), fill=fallback_radius)
    var addrs = List[SlotAddress]()
    var j = 0
    for si in range(len(f.slots)):
        if f.slots[si].kind == SLOT_FREE:
            addrs.append(SlotAddress(P.free_qadr(j), P.free_dadr(j)))
            j += 1
        else:
            addrs.append(SlotAddress(-1, -1))
    var rep = SampleReport()
    var placed = sample_placements(t, f, frames, radii, UInt64(SEED), 0, rep)
    var v0 = List[Float64](length=P.NV, fill=0.0)
    reset_slots(t, f, placed, addrs, q0, v0)
    print("  placed", len(placed), "slot(s) for", t.name, "| goal:", t.goal)
    return q0^


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
        run_view[So101TowerModel, So101TowerConfig](name, st)
    elif st.task < len(task_names()):
        st.reset_qpos = posed_qpos[So101TabletopPlacement](
            name, String("so101_tabletop"), So101TabletopConfig.SLOT_RADIUS
        )
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
