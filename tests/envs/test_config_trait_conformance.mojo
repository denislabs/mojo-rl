"""Every `Phyics3dEnvConfig` implementor still typechecks — the 2a gate.

⚠⚠ THIS EXISTS BECAUSE THE PHYSICS GATES ARE VACUOUS FOR THIS CLASS OF CHANGE.
Phase 2a rewrites the trait's CPU hooks from a loose list of dimension `Int`s
to one `D: DimsLike` provider, and every implementor must be rewritten in the
SAME COMMIT or its signature stops matching. But a config is only compiled when
something instantiates `Phyics3dEnv` with it, and the parity suite instantiates
roughly a dozen of the 51. The first `pre_step_cpu` sweep passed six green
physics gates while most of the files it edited were never handed to the
compiler at all — those gates were measuring the suite's coverage, not the
edit.

⚠⚠ AND NAMING A CONFIG IS NOT ENOUGH — THE HOOKS MUST BE CALLED. The first
version of this file only read `C.FRAME_SKIP` through a trait-bound parameter,
and its negative control PASSED: an invented extra parameter on a config
(`pre_step_cpu <DTYPE, D, BOGUS: Int>` — angle brackets so that the sweep's own
signature-drift checker does not read this sentence as a declaration, which it
did) compiled clean and the gate still reported all 52 conforming. `_conform`
now calls all six CPU hooks, which is what makes the check real.

⚠⚠ EVEN SO, THIS GATE CANNOT CATCH A HALF-FINISHED SWEEP. On this Mojo a hook
whose signature drifts from the trait is not an error — the compiler stops
seeing it as an override and silently substitutes the TRAIT DEFAULT, so the env
loses its observation or reward and runs `pass` forever. Completeness is
checked statically instead, by `scratchpad/check_hook_uniform.py`, which
asserts every implementor's parameter list is textually the trait's.

⚠ THE PARAMETER BINDINGS BELOW ARE REAL ONES. Fourteen configs are themselves
parameterized (`DMWalkerConfig[MOVE_SPEED]`, `DMStackerConfig[N_BOXES]`); each
value here was lifted from an instantiation already in the tree, so the gate
compiles shipped configurations rather than invented ones.

⚠ REGENERATE, DON'T HAND-EDIT: `scratchpad/gen_conformance.py`. It bracket-
matches the base-class list because 14 of the 51 declare their bases over more
than one line, and a `^struct \\w+\\(...\\)$` regex misses every one of them.
"""

from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.envs.ant.ant_config import AntConfig
from mojo_rl.envs.dm_control.acrobot.acrobot_config import DMAcrobotConfig
from mojo_rl.envs.dm_control.ball_in_cup.ball_in_cup_config import DMBallInCupConfig
from mojo_rl.envs.dm_control.cartpole.cartpole_config import DMCartpoleConfig
from mojo_rl.envs.dm_control.cheetah.cheetah_config import DMCheetahConfig
from mojo_rl.envs.dm_control.dog.dog_config import DMDogMoveConfig, DMDogStandConfig
from mojo_rl.envs.dm_control.dog.dog_fetch_config import DMDogFetchConfig
from mojo_rl.envs.dm_control.finger.finger_config import DMFingerSpinConfig, DMFingerTurnConfig
from mojo_rl.envs.dm_control.fish.fish_config import DMFishSwimConfig, DMFishUprightConfig
from mojo_rl.envs.dm_control.hopper.hopper_config import DMHopperConfig
from mojo_rl.envs.dm_control.humanoid.humanoid_config import DMHumanoidConfig
from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_config import DMHumanoidCMUConfig
from mojo_rl.envs.dm_control.manipulation_lift_box_config import LiftLargeBoxConfig
from mojo_rl.envs.dm_control.manipulation_lift_brick_config import LiftBrickConfig
from mojo_rl.envs.dm_control.manipulation_place_brick_config import PlaceBrickConfig
from mojo_rl.envs.dm_control.manipulation_place_cradle_config import PlaceCradleConfig
from mojo_rl.envs.dm_control.manipulation_reach_config import ReachSiteFeaturesConfig
from mojo_rl.envs.dm_control.manipulation_reach_duplo_config import ReachDuploConfig
from mojo_rl.envs.dm_control.manipulation_reassemble3_config import Reassemble3Config
from mojo_rl.envs.dm_control.manipulation_reassemble5_config import Reassemble5Config
from mojo_rl.envs.dm_control.manipulation_stack2_config import Stack2BricksConfig
from mojo_rl.envs.dm_control.manipulation_stack2of3_config import Stack2of3Config
from mojo_rl.envs.dm_control.manipulation_stack3r_config import Stack3RandomConfig
from mojo_rl.envs.dm_control.manipulation_stack_2_bricks_moveable_base_config import Stack2MoveableConfig
from mojo_rl.envs.dm_control.manipulation_stack_3_bricks_config import Stack3BricksConfig
from mojo_rl.envs.dm_control.manipulator.manipulator_config import DMManipulatorConfig
from mojo_rl.envs.dm_control.pendulum.pendulum_config import DMPendulumConfig
from mojo_rl.envs.dm_control.point_mass.point_mass_config import DMPointMassConfig
from mojo_rl.envs.dm_control.point_mass.point_mass_hard_config import DMPointMassHardConfig
from mojo_rl.envs.dm_control.quadruped.quadruped_config import DMQuadrupedConfig
from mojo_rl.envs.dm_control.quadruped.quadruped_fetch_config import DMQuadrupedFetchConfig
from mojo_rl.envs.dm_control.reacher.reacher_config import DMReacherConfig
from mojo_rl.envs.dm_control.stacker.stacker_config import DMStackerConfig
from mojo_rl.envs.dm_control.swimmer.swimmer_config import DMSwimmerConfig
from mojo_rl.envs.dm_control.walker.walker_config import DMWalkerConfig
from mojo_rl.envs.dm_control.wide_reset import WALKER_ROOTZ_ADR, WideResetConfig
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from mojo_rl.envs.hopper.hopper_config import HopperConfig
from mojo_rl.envs.humanoid.humanoid_config import HumanoidConfig
from mojo_rl.envs.humanoid_standup.humanoid_standup_config import HumanoidStandupConfig
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_config import InvertedDoublePendulumConfig
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_config import InvertedPendulumConfig
from mojo_rl.envs.metaworld.sawyer_reach_config import SawyerReachConfig
from mojo_rl.envs.pusher.pusher_config import PusherConfig
from mojo_rl.envs.reacher.reacher_config import ReacherConfig
from mojo_rl.envs.robots.so_arm100 import SoArm100ReachConfig
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig
from mojo_rl.envs.robots.so_arm_reach_config import SoArmReachConfig
from mojo_rl.envs.swimmer.swimmer_config import SwimmerConfig
from mojo_rl.envs.walker2d.walker2d_config import Walker2dConfig


comptime N_CONFIGS: Int = 52
comptime DT = DType.float64
comptime PD = Dims[
    nq=1,
    nv=1,
    nbody=1,
    njoint=1,
    ngeom=1,
    nsite=1,
    max_contacts=1,
    nequality=1,
    ntendon=1,
    nexclude=1,
    nmesh_verts=1,
    npair=1,
    nact=1,
    nten=1,
    nkey=1,
]
"""⚠ ONE PROVIDER NOW — `custom_reset_full_cpu` LANDED (2a.6b). It used to take
twelve loose dims and spell its `Data` and its `Model` from overlapping SUBSETS
of them, so this file needed two providers and the hook could be handed a
`Data` and a `Model` describing DIFFERENT MODELS without complaint. It takes
one `D` now, and a single `Dims` with every field set is what type-checks —
that collapse IS the phase working."""


@always_inline
def _conform[C: Phyics3dEnvConfig](never: Bool) raises -> Int:
    """Typecheck `C`'s six CPU hooks against the trait, without running them.

    ⚠ `never` IS RUNTIME-OPAQUE ON PURPOSE. The calls have to be COMPILED but
    must not EXECUTE: the hooks index real body and site addresses (dog's reset
    reaches into the forties) and the 1-DOF probe `Data` below has none of
    them, so running one is an out-of-bounds read. `main` passes a value the
    folder cannot see through, so the branch is compiled — which is the entire
    point — and never entered.
    """
    var d = Data[DT, PD, 1]()
    var m = Model[DT, PD]()
    var empty = List[Scalar[DT]]()
    var acts = List[Float64]()
    var obs = List[Scalar[DT]]()
    var px = Scalar[DT](0)
    if never:
        C.pre_step_cpu(d, px)
        _ = C.custom_extract_obs_cpu(d, empty, empty, empty, empty, empty, obs)
        _ = C.compute_reward_and_done_cpu(
            d, empty, empty, empty, empty, px, acts, 0, 1
        )
        C.custom_reset_cpu(d, empty, empty, empty, empty)
        C.custom_reset_full_cpu(d, m)
        _ = C.custom_apply_actions_cpu(
            d, empty, empty, empty, empty, empty, empty, empty, acts
        )
    return C.FRAME_SKIP + C.MAX_STEPS + C.INTEGRATOR_WS_EXTRA


def main() raises:
    var seen = 0
    var acc = 0
    # Opaque to the folder: `seen` is 0 here, so `never` is False, but the
    # compiler still has to compile every guarded call.
    var never = seen < 0
    acc += _conform[AntConfig](never)
    seen += 1
    acc += _conform[DMAcrobotConfig[False]](never)
    seen += 1
    acc += _conform[DMBallInCupConfig](never)
    seen += 1
    acc += _conform[DMCartpoleConfig[1, False, False]](never)
    seen += 1
    acc += _conform[DMCheetahConfig](never)
    seen += 1
    acc += _conform[DMDogMoveConfig[1.0]](never)
    seen += 1
    acc += _conform[DMDogStandConfig](never)
    seen += 1
    acc += _conform[DMDogFetchConfig](never)
    seen += 1
    acc += _conform[DMFingerSpinConfig](never)
    seen += 1
    acc += _conform[DMFingerTurnConfig[0.03]](never)
    seen += 1
    acc += _conform[DMFishSwimConfig](never)
    seen += 1
    acc += _conform[DMFishUprightConfig](never)
    seen += 1
    acc += _conform[DMHopperConfig[False]](never)
    seen += 1
    acc += _conform[DMHumanoidConfig[0.0, False]](never)
    seen += 1
    acc += _conform[DMHumanoidCMUConfig[0.0]](never)
    seen += 1
    acc += _conform[LiftLargeBoxConfig](never)
    seen += 1
    acc += _conform[LiftBrickConfig](never)
    seen += 1
    acc += _conform[PlaceBrickConfig](never)
    seen += 1
    acc += _conform[PlaceCradleConfig](never)
    seen += 1
    acc += _conform[ReachSiteFeaturesConfig](never)
    seen += 1
    acc += _conform[ReachDuploConfig](never)
    seen += 1
    acc += _conform[Reassemble3Config](never)
    seen += 1
    acc += _conform[Reassemble5Config](never)
    seen += 1
    acc += _conform[Stack2BricksConfig](never)
    seen += 1
    acc += _conform[Stack2of3Config](never)
    seen += 1
    acc += _conform[Stack3RandomConfig](never)
    seen += 1
    acc += _conform[Stack2MoveableConfig](never)
    seen += 1
    acc += _conform[Stack3BricksConfig](never)
    seen += 1
    acc += _conform[DMManipulatorConfig[False, False]](never)
    seen += 1
    acc += _conform[DMPendulumConfig](never)
    seen += 1
    acc += _conform[DMPointMassConfig](never)
    seen += 1
    acc += _conform[DMPointMassHardConfig](never)
    seen += 1
    acc += _conform[DMQuadrupedConfig[0.5]](never)
    seen += 1
    acc += _conform[DMQuadrupedFetchConfig](never)
    seen += 1
    acc += _conform[DMReacherConfig[0.05]](never)
    seen += 1
    acc += _conform[DMStackerConfig[2]](never)
    seen += 1
    acc += _conform[DMSwimmerConfig](never)
    seen += 1
    acc += _conform[DMWalkerConfig[1.0]](never)
    seen += 1
    acc += _conform[WideResetConfig[DMWalkerConfig[0.0], WALKER_ROOTZ_ADR]](never)
    seen += 1
    acc += _conform[HalfCheetahConfig](never)
    seen += 1
    acc += _conform[HopperConfig](never)
    seen += 1
    acc += _conform[HumanoidConfig](never)
    seen += 1
    acc += _conform[HumanoidStandupConfig](never)
    seen += 1
    acc += _conform[InvertedDoublePendulumConfig](never)
    seen += 1
    acc += _conform[InvertedPendulumConfig](never)
    seen += 1
    acc += _conform[SawyerReachConfig](never)
    seen += 1
    acc += _conform[PusherConfig](never)
    seen += 1
    acc += _conform[ReacherConfig](never)
    seen += 1
    acc += _conform[SwimmerConfig](never)
    seen += 1
    acc += _conform[Walker2dConfig](never)
    seen += 1
    acc += _conform[SoArm100ReachConfig](never)
    seen += 1
    acc += _conform[SoArm101ReachConfig](never)
    seen += 1

    print("configs checked:", seen, "(checksum", acc, ")")
    if seen != N_CONFIGS:
        print("FAIL: expected", N_CONFIGS, "conformance checks, ran", seen)
        raise Error("config roster drifted — the gate lost coverage")
    print("PASS: all", seen, "Phyics3dEnvConfig implementors typecheck")
