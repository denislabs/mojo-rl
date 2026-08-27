"""Interactive `manipulation` viewer — the TASK REGISTRY and the 13-arm dispatch.

    from mojo_rl.envs.dm_control.viewer_manipulation import (
        run_manipulation_viewer, manipulation_task_index,
    )
    run_manipulation_viewer(task, drive, scale)

`examples/dm_control/manipulation_viewer_imgui.mojo` is a ~40-line wrapper over
this. Same split, same reasons, as `viewer.mojo` for the 47 suite tasks — read
that module's header first; everything it says about `viewer_core`,
`build_sidebar` staying non-generic, and compile cost living in `dispatch`
applies here unchanged.

THIS IS A SECOND REGISTRY, NOT AN EXTENSION OF THE FIRST. The suite viewer and
this one share `viewer_core` and nothing else, deliberately: importing
`viewer.mojo` costs all 47 suite arms and importing this costs all 13
manipulation ones, so merging them would make every front end pay for both.

⚠⚠ THE BUILD IS SLOW EVEN BY THIS FAMILY'S STANDARDS, AND THE REASON IS NOT
ARM COUNT. Every one of the 13 is the SAME Jaco arm plus props, so each arm
carries a 267-to-431-geom model with nine STL meshes. `reassemble_5` alone is
431 geoms / 181 sites — the largest model anywhere in this port. Expect tens of
minutes, and use `examples/dm_control/manipulation_viewer_imgui_two.mojo` while
iterating on viewer behaviour.

⚠⚠ AND THAT COST IS NOW PART OF `pixi run build`, WHICH IS SHARED. The package
task is `mojo precompile mojo_rl`, so being in the package is enough to be
compiled — there is no manifest to opt into. `viewer.mojo`'s 47 suite arms
already set that precedent, but these 13 are individually much larger models,
so this module measurably lengthens a build that everyone pays. If that becomes
the binding cost, the fix is to move the registry OUT of the package and into
the example (losing precompilation for it alone), not to trim the task list —
a viewer that silently omits tasks is worse than a slow build.

⚠ WHAT THIS IS FOR, same as the suite viewer: "is the model built and posed the
way I think" — geometry, the reset pose, whether the gripper is where the IK
claims. It does NOT check parity with MuJoCo; `tests/dm_control/` does that,
and all 13 are gated there.

⚠⚠ THE RESET IS THE INTERESTING PART OF THESE TASKS, AND IT IS EXPENSIVE.
Every manipulation reset runs a rejection-sampling TCP inverse-kinematics solve
(up to 10 samples x 10 attempts), and the brick tasks additionally place props
and settle them under gravity for up to 2 s of simulated time. A reset here is
not the instant snap the suite viewer's tasks give you — pressing R visibly
pauses. That is the task, not a hang.

⚠ AND `reset()` CANNOT FAIL LOUDLY. `Phyics3dEnv._reset_state` PRINTS a
`custom_reset_full_cpu FAILED` line and carries on with whatever the hook left,
because the `Env` trait's `reset` does not raise. If a pose looks wrong right
after a reset, check stdout before suspecting the model.

⚠ ACTIONS ARE NOT A POLICY. The drive modes move every joint so you can see the
arm articulate; a Jaco flailing under random velocity commands and knocking the
bricks over is the expected picture. `zero` is the useful one here — it holds
the reset pose so you can inspect the prop placement the IK and the settle
produced.
"""

from mojo_rl.envs.dm_control.viewer_core import (
    ViewerState, run_view, drive_names, parse_drive,
    task_index as _index_in,
    DRIVE_ZERO, DRIVE_RANDOM, DRIVE_SWEEP,
)
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D

from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)
from mojo_rl.envs.dm_control.manipulation_reach_config import (
    ReachSiteFeaturesConfig,
)
from mojo_rl.envs.dm_control.manipulation_reach_duplo_def import ReachDuploModel
from mojo_rl.envs.dm_control.manipulation_reach_duplo_config import (
    ReachDuploConfig,
)
from mojo_rl.envs.dm_control.manipulation_lift_box_def import LiftLargeBoxModel
from mojo_rl.envs.dm_control.manipulation_lift_box_config import (
    LiftLargeBoxConfig,
)
from mojo_rl.envs.dm_control.manipulation_lift_brick_def import LiftBrickModel
from mojo_rl.envs.dm_control.manipulation_lift_brick_config import (
    LiftBrickConfig,
)
from mojo_rl.envs.dm_control.manipulation_place_cradle_def import (
    PlaceCradleModel,
)
from mojo_rl.envs.dm_control.manipulation_place_cradle_config import (
    PlaceCradleConfig,
)
from mojo_rl.envs.dm_control.manipulation_place_brick_def import PlaceBrickModel
from mojo_rl.envs.dm_control.manipulation_place_brick_config import (
    PlaceBrickConfig,
)
from mojo_rl.envs.dm_control.manipulation_stack2_def import Stack2BricksModel
from mojo_rl.envs.dm_control.manipulation_stack2_config import (
    Stack2BricksConfig,
)
from mojo_rl.envs.dm_control.manipulation_stack_2_bricks_moveable_base_def import (
    Stack2MoveableModel,
)
from mojo_rl.envs.dm_control.manipulation_stack_2_bricks_moveable_base_config import (
    Stack2MoveableConfig,
)
from mojo_rl.envs.dm_control.manipulation_stack_3_bricks_def import (
    Stack3BricksModel,
)
from mojo_rl.envs.dm_control.manipulation_stack_3_bricks_config import (
    Stack3BricksConfig,
)
from mojo_rl.envs.dm_control.manipulation_stack3r_def import Stack3RandomModel
from mojo_rl.envs.dm_control.manipulation_stack3r_config import (
    Stack3RandomConfig,
)
from mojo_rl.envs.dm_control.manipulation_stack2of3_def import Stack2of3Model
from mojo_rl.envs.dm_control.manipulation_stack2of3_config import (
    Stack2of3Config,
)
from mojo_rl.envs.dm_control.manipulation_reassemble3_def import (
    Reassemble3Model,
)
from mojo_rl.envs.dm_control.manipulation_reassemble3_config import (
    Reassemble3Config,
)
from mojo_rl.envs.dm_control.manipulation_reassemble5_def import (
    Reassemble5Model,
)
from mojo_rl.envs.dm_control.manipulation_reassemble5_config import (
    Reassemble5Config,
)


def manipulation_task_names() -> List[String]:
    """The 13 `_features` tasks, in `dm_control.manipulation.ALL` order.

    ⚠ THESE ARE THE REGISTERED dm_control NAMES MINUS THE `_features` SUFFIX,
    so `manipulation.load(name + "_features")` is the reference for any of
    them. The suffix is dropped because every task here is `_features` — the
    12 `_vision` variants are NOT ported (they need render-to-observation) and
    listing a suffix that never varies is noise in a picker.
    """
    var t = List[String]()
    t.append(String("reach_site"))
    t.append(String("reach_duplo"))
    t.append(String("lift_large_box"))
    t.append(String("lift_brick"))
    t.append(String("place_cradle"))
    t.append(String("place_brick"))
    t.append(String("stack_2_bricks"))
    t.append(String("stack_2_bricks_moveable_base"))
    t.append(String("stack_3_bricks"))
    t.append(String("stack_3_bricks_random_order"))
    t.append(String("stack_2_of_3_bricks_random_order"))
    t.append(String("reassemble_3_bricks_fixed_order"))
    t.append(String("reassemble_5_bricks_random_order"))
    return t^


def manipulation_domain_names() -> List[String]:
    """Grouping for the sidebar's filter.

    ⚠ `manipulation` HAS NO DOMAINS the way the suite does — every task is the
    same Jaco arm on the same table, and dm_control groups them only by the
    module they live in. These five are the PROP FAMILIES, which is the axis a
    person actually browses by: what is on the table and what has to happen
    to it.
    """
    var d = List[String]()
    d.append(String("reach"))
    d.append(String("lift"))
    d.append(String("place"))
    d.append(String("stack"))
    d.append(String("reassemble"))
    return d^


def manipulation_task_domain() -> List[Int]:
    """Index into `manipulation_domain_names` per task. ⚠ Positionally coupled
    to `manipulation_task_names`."""
    var t = List[Int]()
    t.append(0)  # reach_site
    t.append(0)  # reach_duplo
    t.append(1)  # lift_large_box
    t.append(1)  # lift_brick
    t.append(2)  # place_cradle
    t.append(2)  # place_brick
    t.append(3)  # stack_2_bricks
    t.append(3)  # stack_2_bricks_moveable_base
    t.append(3)  # stack_3_bricks
    t.append(3)  # stack_3_bricks_random_order
    t.append(3)  # stack_2_of_3_bricks_random_order
    t.append(4)  # reassemble_3_bricks_fixed_order
    t.append(4)  # reassemble_5_bricks_random_order
    return t^


def dispatch(mut st: ViewerState) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    ⚠ INDEX ORDER MUST MATCH `manipulation_task_names`. This is the one place
    all 13 compile-time instantiations are named, and what the build time is
    proportional to — see the module header on why each arm is expensive here.
    """
    var name = manipulation_task_names()[st.task]
    if st.task == 0:
        run_view[ReachSiteFeaturesModel, ReachSiteFeaturesConfig](name, st)
    elif st.task == 1:
        run_view[ReachDuploModel, ReachDuploConfig](name, st)
    elif st.task == 2:
        run_view[LiftLargeBoxModel, LiftLargeBoxConfig](name, st)
    elif st.task == 3:
        run_view[LiftBrickModel, LiftBrickConfig](name, st)
    elif st.task == 4:
        run_view[PlaceCradleModel, PlaceCradleConfig](name, st)
    elif st.task == 5:
        run_view[PlaceBrickModel, PlaceBrickConfig](name, st)
    elif st.task == 6:
        run_view[Stack2BricksModel, Stack2BricksConfig](name, st)
    elif st.task == 7:
        run_view[Stack2MoveableModel, Stack2MoveableConfig](name, st)
    elif st.task == 8:
        run_view[Stack3BricksModel, Stack3BricksConfig](name, st)
    elif st.task == 9:
        run_view[Stack3RandomModel, Stack3RandomConfig](name, st)
    elif st.task == 10:
        run_view[Stack2of3Model, Stack2of3Config](name, st)
    elif st.task == 11:
        run_view[Reassemble3Model, Reassemble3Config](name, st)
    elif st.task == 12:
        run_view[Reassemble5Model, Reassemble5Config](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def run_manipulation_viewer(
    start_task: Int, drive: Int, scale: Float64
) raises:
    """Open the viewer on `start_task` and run until the window is closed.

    One task at a time; picking another ends that task's loop and comes back
    here to build the next. `st` crosses the gap, INCLUDING THE WINDOW — see
    `viewer_core.run_view` on the handoff.

    ⚠ THE `st.handoff` CHECK IS NOT DEFENSIVE PADDING — `dispatch`'s
    unknown-index arm returns with a live handoff and `quit` set, and there is
    no renderer left to close it.
    """
    if not imgui_shim_available():
        print("Dear ImGui shim not built.")
        print("  Run:  pixi run build-imgui")
        return

    var st = ViewerState(
        start_task,
        drive,
        scale,
        manipulation_task_names(),
        manipulation_domain_names(),
        manipulation_task_domain(),
    )
    while not st.quit:
        dispatch(st)

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None


def manipulation_task_index(name: String) -> Int:
    """Task id for one of the 13 registered names, or -1.

    ⚠ ACCEPTS THE `_features` SUFFIX TOO, because that is what dm_control calls
    them and what anyone copying a name from `tests/dm_control/` will paste.

    ⚠ MATCHED BY APPENDING THE SUFFIX TO EACH CANDIDATE, not by slicing it off
    the input. `len(String)` and `s[a:b]` are both rejected by the compiler now
    ("Mojo strings are UTF-8 encoded, so a single length is ambiguous") and
    would need `byte_length()` / `s[byte=a:b]`. Going the other direction needs
    neither and cannot get the units wrong.
    """
    var names = manipulation_task_names()
    var direct = _index_in(name, names)
    if direct >= 0:
        return direct
    comptime SUFFIX = String("_features")
    for i in range(len(names)):
        if name == names[i] + SUFFIX:
            return i
    return -1


def print_manipulation_task_list():
    """The registry, grouped by prop family, for an unknown-name message."""
    var names = manipulation_task_names()
    var domains = manipulation_domain_names()
    var of = manipulation_task_domain()
    for d in range(len(domains)):
        print("  " + domains[d] + ":")
        for i in range(len(names)):
            if of[i] == d:
                print("     ", names[i])
