"""dm_control `dog` — the four in-scope tasks as env aliases.

    from mojo_rl.envs.dm_control.dog import DMDogStand
    var env = DMDogStand()

stand, walk, trot and run share ONE task class hierarchy (`Move` subclasses
`Stand`) but not one model: the floor plane's half-extent is
`move_speed * _DEFAULT_TIME_LIMIT`, so stand and walk get 15, trot 45 and run
135. stand therefore reuses walk's model def rather than owning a fourth.

CPU ONLY, for the same reason quadruped is: the GPU-batched facade does not
carry `act`, and every one of dog's 38 actuators is `<general dyntype="filter">`
whose force IS `gainprm[0] * act`. A batched dog would run with a permanently
zero activation — completely limp — so the alias is deliberately not offered.

`fetch` is Phase 5. It keeps the ball and the target, which adds a free joint
(njnt 75 / nq 87), a second free-jointed object to collide, and the domain's
only use of `sigmoid='reciprocal'`.
"""

from .dog_xml import DMDogStandWalkModel, DMDogTrotModel, DMDogRunModel
from .dog_config import DMDogStandConfig, DMDogMoveConfig
from .dog_xml import DOG_WALK_SPEED, DOG_TROT_SPEED, DOG_RUN_SPEED
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — the driver only ever sees
# truncation at the 1000-step limit.
comptime DMDogStand[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMDogStandWalkModel, DMDogStandConfig, DTYPE, False
]

comptime DMDogWalk[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMDogStandWalkModel, DMDogMoveConfig[DOG_WALK_SPEED], DTYPE, False
]

comptime DMDogTrot[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMDogTrotModel, DMDogMoveConfig[DOG_TROT_SPEED], DTYPE, False
]

comptime DMDogRun[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMDogRunModel, DMDogMoveConfig[DOG_RUN_SPEED], DTYPE, False
]
