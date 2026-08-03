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
from std.math import sin, pi
from std.sys import argv
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.envs.dm_control.quadruped import DMQuadrupedWalk


# ── the task under the microscope ────────────────────────────────────────────
# Change BOTH this alias and its `from ... import` line above.
comptime ENV = DMQuadrupedWalk

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


def main() raises:
    seed(SEED)

    comptime OBS_DIM = ENV.OBS_DIM
    comptime ACT_DIM = ENV.ACTION_DIM

    var args = argv()
    var drive = _parse_drive(String(args[1])) if len(args) > 1 else DRIVE_SWEEP
    var act_scale = Float64(1.0)
    if len(args) > 2:
        try:
            act_scale = Float64(String(args[2]))
        except:
            print("could not parse action scale '", String(args[2]),
                  "' — using 1.0")

    print("=" * 66)
    print("dm_control viewer")
    print("=" * 66)
    print("  obs dim      =", OBS_DIM)
    print("  action dim   =", ACT_DIM)
    if drive == DRIVE_ZERO:
        print("  drive        = zero (reset pose + gravity)")
    elif drive == DRIVE_RANDOM:
        print("  drive        = random, scale", act_scale)
    else:
        print("  drive        = sweep, period", SWEEP_PERIOD, "steps, scale",
              act_scale)
    print("  (drive mode is argv[1]: zero | random | sweep; argv[2] = scale)")
    print("  reset every  =", EPISODE_STEPS, "steps")
    print("  close the window to quit")
    print("=" * 66)

    var ctx = DeviceContext()
    var env = ENV(ctx)
    _ = env.reset()

    if not env.init_renderer():
        print("No renderer available — is SDL3 present? Running headless is")
        print("pointless for this script, so stopping here.")
        return

    # Held action for RANDOM, so the model is driven rather than dithered.
    var held = ENV.ActionType()
    var step_i = 0
    var episode = 0
    var ep_return = Float64(0)

    while env.is_renderer_open():
        var action = ENV.ActionType()

        if drive == DRIVE_RANDOM:
            if step_i % HOLD_STEPS == 0:
                for a in range(ACT_DIM):
                    held.data[a] = random_float64(-1.0, 1.0) * act_scale
            for a in range(ACT_DIM):
                action.data[a] = held.data[a]
        elif drive == DRIVE_SWEEP:
            # Out-of-phase so neighbouring joints do not move together — a
            # synchronised sweep hides a mirrored or duplicated axis.
            var t = Float64(step_i) / SWEEP_PERIOD
            for a in range(ACT_DIM):
                var phase = Float64(a) / Float64(ACT_DIM if ACT_DIM > 0 else 1)
                action.data[a] = sin(2.0 * pi * (t + phase)) * act_scale

        var out = env.step(action)
        ep_return += Float64(out[1])

        env.render_frame()
        env.renderer_delay(FRAME_DELAY_MS)
        if env.check_renderer_quit():
            break

        step_i += 1
        # `out[2]` is the env's own terminated flag; dm_control tasks are
        # mostly non-terminating, so the step budget is the usual reset.
        if out[2] or step_i >= EPISODE_STEPS:
            episode += 1
            print(
                "  episode", episode, "ended after", step_i,
                "steps, return =", ep_return,
            )
            _ = env.reset()
            step_i = 0
            ep_return = 0.0

    env.close_renderer()
    print("viewer closed")
