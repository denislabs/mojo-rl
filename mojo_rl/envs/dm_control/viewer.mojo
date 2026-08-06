"""Interactive dm_control viewer — the TASK REGISTRY and the 43-arm dispatch.

    from mojo_rl.envs.dm_control.viewer import run_viewer, task_index
    run_viewer(task, drive, scale)

`examples/dm_control/dm_viewer_imgui.mojo` is a ~40-line wrapper over this.
The logic lives in the package so it can be precompiled (`pixi run build`)
instead of being re-elaborated by every build of the example, and so a second
front end (a training monitor, a headless recorder) can reuse it without
copying 600 lines.

⚠ REQUIRES THE ImGui SHIM: `pixi run build-imgui`. `imgui_shim_available()`
answers that without touching FFI; `run_viewer` checks it up front, because
the loader ABORTS the process rather than raising.

THE OTHER HALF IS `viewer_core.mojo` — state, sidebar and the per-task run
loop, none of which names a task. Everything task-agnostic belongs there.

COMPILE COST LIVES IN `dispatch`. Each task is a distinct compile-time
`Phyics3dEnv[MODEL, CONFIG]`, and `dispatch` is the one place all 43 are
named — so build time is roughly proportional to its arm count and to how
much code `run_view` carries PER ARM. Two consequences, both load-bearing:

  · `build_sidebar` is NOT generic. It takes plain data in and returns
    requests out, so 43 instantiations share one copy of the widget code
    instead of stamping out 43. Keep it that way when adding features —
    anything that can be phrased over plain values belongs in `viewer_core`.

  · IMPORTING THIS MODULE COSTS ALL 43, whether or not you call `dispatch`.
    That is why the split exists: a front end that wants two tasks imports
    `viewer_core` and writes its own two-arm dispatch, and compiles in
    seconds. `examples/dm_control/dm_viewer_imgui_two.mojo` is that front
    end, and is the one to iterate against while working on the viewer.

⚠ DOG IS DELIBERATELY ABSENT. It is still being ported in parallel; adding it
here would couple this file to a moving target.

WHAT THIS IS FOR. It answers "is the model built and posed the way I think" —
geometry, joint axes, ranges, the reset pose, whether anything falls through
the floor. It does NOT check parity with MuJoCo; `tests/dm_control/` does
that, and a model can look perfect while its dynamics differ.

⚠ ACTIONS ARE NOT A POLICY. The drive modes move every joint so you can see it
articulate. A tumbling humanoid under random torque is the expected picture.
"""

from mojo_rl.envs.dm_control.viewer_core import (
    ViewerState, run_view, drive_names, parse_drive,
    task_index as _index_in,
    DRIVE_ZERO, DRIVE_RANDOM, DRIVE_SWEEP,
)
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D

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
from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_xml import (
    DMHumanoidCMUModel,
)
# ⚠ ALIASED, NOT IMPORTED BARE. `humanoid_cmu_config` defines its OWN
# WALK_SPEED/RUN_SPEED, which would collide with `humanoid_config`'s. They
# happen to hold the same values today (1.0 / 10.0) — which is exactly why a
# bare import would be a silent trap rather than a compile error if one domain
# later retunes its speeds.
from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_config import (
    DMHumanoidCMUConfig,
    WALK_SPEED as CMU_WALK_SPEED,
    RUN_SPEED as CMU_RUN_SPEED,
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
    DMQuadrupedWalkModel, DMQuadrupedRunModel, DMQuadrupedFetchModel,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_config import (
    DMQuadrupedWalkConfig, DMQuadrupedRunConfig,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_fetch_config import (
    DMQuadrupedFetchConfig,
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


def task_names() -> List[String]:
    """The 43 tasks, in the order `dispatch` indexes them.

    ⚠ THIS LIST AND `dispatch` ARE POSITIONALLY COUPLED. Index i here must be
    the arm `st.task == i` there; a mismatch shows up as clicking one robot and
    getting another, which is confusing precisely because everything still
    works. `task_index` is the only lookup, so argv names come from here too
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
    t.append(String("humanoid_cmu_stand"))
    t.append(String("humanoid_cmu_walk"))
    t.append(String("humanoid_cmu_run"))
    t.append(String("manipulator_bring_ball"))
    t.append(String("manipulator_bring_peg"))
    t.append(String("manipulator_insert_ball"))
    t.append(String("manipulator_insert_peg"))
    t.append(String("pendulum_swingup"))
    t.append(String("point_mass_easy"))
    t.append(String("point_mass_hard"))
    t.append(String("quadruped_walk"))
    t.append(String("quadruped_run"))
    t.append(String("quadruped_fetch"))
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


def domain_names() -> List[String]:
    """The 17 domains, in the order `task_domain` indexes them."""
    var d = List[String]()
    d.append(String("acrobot"))
    d.append(String("ball_in_cup"))
    d.append(String("cartpole"))
    d.append(String("cheetah"))
    d.append(String("finger"))
    d.append(String("fish"))
    d.append(String("hopper"))
    d.append(String("humanoid"))
    d.append(String("humanoid_cmu"))
    d.append(String("manipulator"))
    d.append(String("pendulum"))
    d.append(String("point_mass"))
    d.append(String("quadruped"))
    d.append(String("reacher"))
    d.append(String("stacker"))
    d.append(String("swimmer"))
    d.append(String("walker"))
    return d^


def task_domain() -> List[Int]:
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
    for _ in range(3):
        t.append(8)   # humanoid_cmu
    for _ in range(4):
        t.append(9)   # manipulator
    t.append(10)      # pendulum
    for _ in range(2):
        t.append(11)  # point_mass
    for _ in range(3):
        t.append(12)  # quadruped  (walk, run, fetch)
    for _ in range(2):
        t.append(13)  # reacher
    for _ in range(2):
        t.append(14)  # stacker
    for _ in range(2):
        t.append(15)  # swimmer
    for _ in range(3):
        t.append(16)  # walker
    return t^


def dispatch(mut st: ViewerState) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    ⚠ INDEX ORDER MUST MATCH `task_names`. This is the one place all 43
    compile-time instantiations are named, and what the build time is
    proportional to.
    """
    var name = task_names()[st.task]
    if st.task == 0:
        run_view[DMAcrobotModel, DMAcrobotConfig[False]](name, st)
    elif st.task == 1:
        run_view[DMAcrobotModel, DMAcrobotConfig[True]](name, st)
    elif st.task == 2:
        run_view[DMBallInCupModel, DMBallInCupConfig](name, st)
    elif st.task == 3:
        run_view[DMCartpole1Model, DMCartpoleConfig[1, False, False]](name, st)
    elif st.task == 4:
        run_view[DMCartpole1Model, DMCartpoleConfig[1, False, True]](name, st)
    elif st.task == 5:
        run_view[DMCartpole1Model, DMCartpoleConfig[1, True, False]](name, st)
    elif st.task == 6:
        run_view[DMCartpole1Model, DMCartpoleConfig[1, True, True]](name, st)
    elif st.task == 7:
        run_view[DMCartpole2Model, DMCartpoleConfig[2, True, False]](name, st)
    elif st.task == 8:
        run_view[DMCartpole3Model, DMCartpoleConfig[3, True, False]](name, st)
    elif st.task == 9:
        run_view[DMCheetahModel, DMCheetahConfig](name, st)
    elif st.task == 10:
        run_view[DMFingerSpinModel, DMFingerSpinConfig](name, st)
    elif st.task == 11:
        run_view[DMFingerTurnModel, DMFingerTurnConfig[0.07]](name, st)
    elif st.task == 12:
        run_view[DMFingerTurnModel, DMFingerTurnConfig[0.03]](name, st)
    elif st.task == 13:
        run_view[DMFishUprightModel, DMFishUprightConfig](name, st)
    elif st.task == 14:
        run_view[DMFishSwimModel, DMFishSwimConfig](name, st)
    elif st.task == 15:
        run_view[DMHopperModel, DMHopperConfig[False]](name, st)
    elif st.task == 16:
        run_view[DMHopperModel, DMHopperConfig[True]](name, st)
    elif st.task == 17:
        run_view[DMHumanoidModel, DMHumanoidConfig[0.0, False]](name, st)
    elif st.task == 18:
        run_view[DMHumanoidModel, DMHumanoidConfig[WALK_SPEED, False]](name, st)
    elif st.task == 19:
        run_view[DMHumanoidModel, DMHumanoidConfig[RUN_SPEED, False]](name, st)
    elif st.task == 20:
        run_view[
            DMHumanoidPureModel, DMHumanoidConfig[RUN_SPEED, True]
        ](name, st)
    elif st.task == 21:
        run_view[DMHumanoidCMUModel, DMHumanoidCMUConfig[0.0]](name, st)
    elif st.task == 22:
        run_view[
            DMHumanoidCMUModel, DMHumanoidCMUConfig[CMU_WALK_SPEED]
        ](name, st)
    elif st.task == 23:
        run_view[
            DMHumanoidCMUModel, DMHumanoidCMUConfig[CMU_RUN_SPEED]
        ](name, st)
    elif st.task == 24:
        run_view[
            DMManipulatorBringBallModel, DMManipulatorBringBallConfig
        ](name, st)
    elif st.task == 25:
        run_view[
            DMManipulatorBringPegModel, DMManipulatorBringPegConfig
        ](name, st)
    elif st.task == 26:
        run_view[
            DMManipulatorInsertBallModel, DMManipulatorInsertBallConfig
        ](name, st)
    elif st.task == 27:
        run_view[
            DMManipulatorInsertPegModel, DMManipulatorInsertPegConfig
        ](name, st)
    elif st.task == 28:
        run_view[DMPendulumModel, DMPendulumConfig](name, st)
    elif st.task == 29:
        run_view[DMPointMassModel, DMPointMassConfig](name, st)
    elif st.task == 30:
        run_view[DMPointMassModel, DMPointMassHardConfig](name, st)
    elif st.task == 31:
        run_view[DMQuadrupedWalkModel, DMQuadrupedWalkConfig](name, st)
    elif st.task == 32:
        run_view[DMQuadrupedRunModel, DMQuadrupedRunConfig](name, st)
    elif st.task == 33:
        run_view[DMQuadrupedFetchModel, DMQuadrupedFetchConfig](name, st)
    elif st.task == 34:
        run_view[DMReacherModel, DMReacherConfig[0.05]](name, st)
    elif st.task == 35:
        run_view[DMReacherModel, DMReacherConfig[0.015]](name, st)
    elif st.task == 36:
        run_view[DMStacker2Model, DMStacker2Config](name, st)
    elif st.task == 37:
        run_view[DMStacker4Model, DMStacker4Config](name, st)
    elif st.task == 38:
        run_view[DMSwimmer6Model, DMSwimmerConfig](name, st)
    elif st.task == 39:
        run_view[DMSwimmer15Model, DMSwimmerConfig](name, st)
    elif st.task == 40:
        run_view[DMWalkerModel, DMWalkerConfig[0.0]](name, st)
    elif st.task == 41:
        run_view[DMWalkerModel, DMWalkerConfig[1.0]](name, st)
    elif st.task == 42:
        run_view[DMWalkerModel, DMWalkerConfig[8.0]](name, st)
    else:
        print("unknown task index:", st.task)
        st.quit = True


def run_viewer(start_task: Int, drive: Int, scale: Float64) raises:
    """Open the viewer on `start_task` and run until the window is closed.

    One task runs at a time; picking another in the window ends that task's
    loop and comes back here to build the next one. `st` is what crosses the
    gap — including the WINDOW, which outlives every env that draws into it
    (`ViewerState.handoff`).

    ⚠ THE `st.handoff` CHECK BELOW IS NOT DEFENSIVE PADDING. `dispatch` can
    return with a live handoff and `quit` set — an out-of-range task id does
    exactly that — and there is no renderer left to close it. Without this the
    window would stay mapped after the process stopped drawing to it.
    """
    if not imgui_shim_available():
        print("Dear ImGui shim not built.")
        print("  Run:  pixi run build-imgui")
        print("  (or use examples/dm_control/dm_viewer.mojo, which needs no")
        print("   native dependency)")
        return

    var st = ViewerState(
        start_task, drive, scale, task_names(), domain_names(), task_domain()
    )
    while not st.quit:
        dispatch(st)

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None


def task_index(name: String) -> Int:
    """Task id for one of the 43 registered names, or -1.

    The registry's own lookup; `viewer_core.task_index` is the same search over
    an arbitrary table, since that module is not allowed to know this one.
    """
    return _index_in(name, task_names())


def print_task_list() raises:
    """The registered names, for an argv error path."""
    var names = task_names()
    for i in range(len(names)):
        print("   ", names[i])
