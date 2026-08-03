"""Interactive 3D viewer for any ported dm_control task — the "does this model
look right?" tool, in the spirit of dm_control's own `suite.explore` viewer.

    pixi run mojo run -I . examples/dm_control/dm_viewer.mojo

PICK THE TASK BY EDITING `ENV` BELOW. There is no runtime `--env` flag and
that is a deliberate limitation, not an oversight: `Phyics3dEnv[MODEL, CONFIG]`
is a distinct COMPILE-TIME type per task, so a runtime selector would have to
instantiate all 39 in one binary. Each one costs minutes of compile on its own
and this tree has a documented history of compiler blowups from far less (see
`feedback_mojo_*_compile_explosion` in the notes). One task per build keeps a
rebuild at "normal test file" cost. Editing one line is the cheap version of a
menu; if you want a real menu later, the honest way is a small set of separate
binaries plus a shell picker, not one 39-way executable.

ALL 39 PORTED TASKS — copy one into `ENV` and the matching `from` line:

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
  mouse drag / scroll   orbit, zoom
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
comptime SWEEP_PERIOD: Float64 = 120.0  # SWEEP: steps per full cycle
comptime EPISODE_STEPS: Int = 1000  # auto-reset cadence
comptime FRAME_DELAY_MS: Int = 16   # ~60 FPS
comptime SEED: Int = 0


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


def _hud(
    name: String, drive: Int, scale: Float64, step: Int, episode: Int,
    ep_return: Float64,
) -> List[String]:
    var out = List[String]()
    out.append(String("task  ") + name)
    out.append(
        String("[D] drive ") + _drive_name(drive)
        + String("   [ ] scale ") + _fmt2(scale)
    )
    out.append(String("[N] reset   [0] zero"))
    out.append(
        String("ep ") + String(episode) + String("  step ") + String(step)
        + String("  return ") + _fmt2(ep_return)
    )
    return out^


def _view[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig
](name: String, drive: Int, act_scale: Float64) raises:
    """The viewer loop, generic over one task's (model, config) pair.

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
    if drive == DRIVE_ZERO:
        print("  drive        = zero (reset pose + gravity)")
    elif drive == DRIVE_RANDOM:
        print("  drive        = random, scale", act_scale)
    else:
        print("  drive        = sweep, scale", act_scale)
    print("  close the window to quit")
    print("=" * 66)

    var ctx = DeviceContext()
    var env = E(ctx)
    _ = env.reset()

    if not env.init_renderer():
        print("No renderer available — is SDL3 present?")
        return

    var held = E.ActionType()
    var step_i = 0
    var episode = 0
    var ep_return = Float64(0)
    var live_drive = drive
    var live_scale = act_scale

    while env.is_renderer_open():
        # ── live keys ────────────────────────────────────────────────────
        # Only keys the renderer does not already bind. Taken: ESC, 1-9,
        # SPACE, RIGHT, R, S, V. `renderer_take_key` clears on read, so each
        # press fires once.
        var k = env.renderer_take_key()
        if k == 0x44 or k == 0x64:  # D — cycle drive mode
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

        env.set_hud_extra(_hud(name, live_drive, live_scale, step_i,
                               episode, ep_return))

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

    env.close_renderer()
    print("viewer closed")


def main() raises:
    seed(SEED)
    var args = argv()
    var task = String(args[1]) if len(args) > 1 else String("quadruped_walk")
    var drive = _parse_drive(String(args[2])) if len(args) > 2 else DRIVE_SWEEP
    var act_scale = Float64(1.0)
    if len(args) > 3:
        try:
            act_scale = Float64(String(args[3]))
        except:
            print("bad scale, using 1.0")

    if task == "acrobot_swingup":
        _view[DMAcrobotModel, DMAcrobotConfig[False]](task, drive, act_scale)
    elif task == "acrobot_swingup_sparse":
        _view[DMAcrobotModel, DMAcrobotConfig[True]](task, drive, act_scale)
    elif task == "ball_in_cup_catch":
        _view[DMBallInCupModel, DMBallInCupConfig](task, drive, act_scale)
    elif task == "cartpole_balance":
        _view[DMCartpole1Model, DMCartpoleConfig[1, False, False]](
            task, drive, act_scale)
    elif task == "cartpole_balance_sparse":
        _view[DMCartpole1Model, DMCartpoleConfig[1, False, True]](
            task, drive, act_scale)
    elif task == "cartpole_swingup":
        _view[DMCartpole1Model, DMCartpoleConfig[1, True, False]](
            task, drive, act_scale)
    elif task == "cartpole_swingup_sparse":
        _view[DMCartpole1Model, DMCartpoleConfig[1, True, True]](
            task, drive, act_scale)
    elif task == "cartpole_two_poles":
        _view[DMCartpole2Model, DMCartpoleConfig[2, True, False]](
            task, drive, act_scale)
    elif task == "cartpole_three_poles":
        _view[DMCartpole3Model, DMCartpoleConfig[3, True, False]](
            task, drive, act_scale)
    elif task == "cheetah_run":
        _view[DMCheetahModel, DMCheetahConfig](task, drive, act_scale)
    elif task == "finger_spin":
        _view[DMFingerSpinModel, DMFingerSpinConfig](task, drive, act_scale)
    elif task == "finger_turn_easy":
        _view[DMFingerTurnModel, DMFingerTurnConfig[0.07]](
            task, drive, act_scale)
    elif task == "finger_turn_hard":
        _view[DMFingerTurnModel, DMFingerTurnConfig[0.03]](
            task, drive, act_scale)
    elif task == "fish_upright":
        _view[DMFishUprightModel, DMFishUprightConfig](task, drive, act_scale)
    elif task == "fish_swim":
        _view[DMFishSwimModel, DMFishSwimConfig](task, drive, act_scale)
    elif task == "hopper_stand":
        _view[DMHopperModel, DMHopperConfig[False]](task, drive, act_scale)
    elif task == "hopper_hop":
        _view[DMHopperModel, DMHopperConfig[True]](task, drive, act_scale)
    elif task == "humanoid_stand":
        _view[DMHumanoidModel, DMHumanoidConfig[0.0, False]](
            task, drive, act_scale)
    elif task == "humanoid_walk":
        _view[DMHumanoidModel, DMHumanoidConfig[WALK_SPEED, False]](
            task, drive, act_scale)
    elif task == "humanoid_run":
        _view[DMHumanoidModel, DMHumanoidConfig[RUN_SPEED, False]](
            task, drive, act_scale)
    elif task == "humanoid_run_pure_state":
        _view[DMHumanoidPureModel, DMHumanoidConfig[RUN_SPEED, True]](
            task, drive, act_scale)
    elif task == "manipulator_bring_ball":
        _view[DMManipulatorBringBallModel, DMManipulatorBringBallConfig](
            task, drive, act_scale)
    elif task == "manipulator_bring_peg":
        _view[DMManipulatorBringPegModel, DMManipulatorBringPegConfig](
            task, drive, act_scale)
    elif task == "manipulator_insert_ball":
        _view[DMManipulatorInsertBallModel, DMManipulatorInsertBallConfig](
            task, drive, act_scale)
    elif task == "manipulator_insert_peg":
        _view[DMManipulatorInsertPegModel, DMManipulatorInsertPegConfig](
            task, drive, act_scale)
    elif task == "pendulum_swingup":
        _view[DMPendulumModel, DMPendulumConfig](task, drive, act_scale)
    elif task == "point_mass_easy":
        _view[DMPointMassModel, DMPointMassConfig](task, drive, act_scale)
    elif task == "point_mass_hard":
        _view[DMPointMassModel, DMPointMassHardConfig](task, drive, act_scale)
    elif task == "quadruped_walk":
        _view[DMQuadrupedWalkModel, DMQuadrupedWalkConfig](
            task, drive, act_scale)
    elif task == "quadruped_run":
        _view[DMQuadrupedRunModel, DMQuadrupedRunConfig](
            task, drive, act_scale)
    elif task == "reacher_easy":
        _view[DMReacherModel, DMReacherConfig[0.05]](task, drive, act_scale)
    elif task == "reacher_hard":
        _view[DMReacherModel, DMReacherConfig[0.015]](task, drive, act_scale)
    elif task == "stacker_stack_2":
        _view[DMStacker2Model, DMStacker2Config](task, drive, act_scale)
    elif task == "stacker_stack_4":
        _view[DMStacker4Model, DMStacker4Config](task, drive, act_scale)
    elif task == "swimmer_swimmer6":
        _view[DMSwimmer6Model, DMSwimmerConfig](task, drive, act_scale)
    elif task == "swimmer_swimmer15":
        _view[DMSwimmer15Model, DMSwimmerConfig](task, drive, act_scale)
    elif task == "walker_stand":
        _view[DMWalkerModel, DMWalkerConfig[0.0]](task, drive, act_scale)
    elif task == "walker_walk":
        _view[DMWalkerModel, DMWalkerConfig[1.0]](task, drive, act_scale)
    elif task == "walker_run":
        _view[DMWalkerModel, DMWalkerConfig[8.0]](task, drive, act_scale)
    else:
        print("unknown task:", task, "— the 39 registered tasks are:")
        print("  acrobot_swingup acrobot_swingup_sparse ball_in_cup_catch")
        print("  cartpole_balance cartpole_balance_sparse cartpole_swingup")
        print("  cartpole_swingup_sparse cartpole_two_poles cartpole_three_poles")
        print("  cheetah_run finger_spin finger_turn_easy finger_turn_hard")
        print("  fish_upright fish_swim hopper_stand hopper_hop")
        print("  humanoid_stand humanoid_walk humanoid_run")
        print("  humanoid_run_pure_state")
        print("  manipulator_bring_ball manipulator_bring_peg")
        print("  manipulator_insert_ball manipulator_insert_peg")
        print("  pendulum_swingup point_mass_easy point_mass_hard")
        print("  quadruped_walk quadruped_run reacher_easy reacher_hard")
        print("  stacker_stack_2 stacker_stack_4")
        print("  swimmer_swimmer6 swimmer_swimmer15")
        print("  walker_stand walker_walk walker_run")
