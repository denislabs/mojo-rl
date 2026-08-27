"""`dm_control` `dog` — the four in-scope tasks as env aliases.

    from mojo_rl.envs.dm_control.dog import DMDogStand
    var env = DMDogStand()

stand, walk, trot and run share ONE task class hierarchy (`Move` subclasses
`Stand`) but not one model: the floor plane's half-extent is
`move_speed * _DEFAULT_TIME_LIMIT`, so stand and walk get 15, trot 45 and run
135. stand therefore reuses walk's model def rather than owning a fourth.

GPU-BATCHED as of 2026-08-10 (`*Batched` below). The blocker was never dog's:
`Phyics3dBatchedEnv` carried no `act` slab, and every one of dog's 38
actuators is `<general dyntype="filter">` whose force IS `gainprm[0] * act`, so
a batched dog would have run permanently limp. quadruped's blocker E added the
slab and moved the actuator call inside the substep loop, which is what a
`dyntype` activation needs; dog then needed only its own GPU hooks and a `Je`
that fits (`solver/je_budget` — dog's is 151 KB and spills to global).

⚠ THE GPU RESET IS NOT THE CPU RESET. `initialize_episode` also draws
`act[i] = uniform(*ctrlrange[i])` for all 38 actuators and the GPU hook has no
actuator table, so batched episodes start at zero activation. See
`_dog_init_qpos_gpu`. This makes the batched task slightly easier and is
invisible to the parity gate, which injects a shared qpos/qvel.

`fetch` is Phase 5. It keeps the ball and the target, which adds a free joint
(njnt 75 / nq 87), a second free-jointed object to collide, and the domain's
only use of `sigmoid='reciprocal'`.
"""

from .dog_xml import DMDogStandWalkModel, DMDogTrotModel, DMDogRunModel
from .dog_config import DMDogStandConfig, DMDogMoveConfig
from .dog_fetch_xml import DMDogFetchModel
from .dog_fetch_config import DMDogFetchConfig
from .dog_xml import DOG_WALK_SPEED, DOG_TROT_SPEED, DOG_RUN_SPEED
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


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

# ⚠ `fetch` HAS ITS OWN MODEL, not one of the three above: it is the only dog
# task that keeps the ball, the target and the walls, and it takes dog.xml's
# DEFAULT floor half-extent of 10 rather than `move_speed * 15`. CPU only, for
# the same reason as the others — every actuator is `dyntype="filter"` and the
# batched facade carries no `act`.
comptime DMDogFetch[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMDogFetchModel, DMDogFetchConfig, DTYPE, False
]


# ── GPU-batched aliases ────────────────────────────────────────────────
#
# N_ENVS is the caller's; the driver instantiates one per training run.
# `TERMINATE_ON_UNHEALTHY=False` because no suite task terminates early — the
# driver only ever sees truncation at MAX_STEPS.
#
# ⚠⚠ APPLE BUILDS THESE AND CANNOT RUN THEM. There were TWO barriers, and
# only the first is fixed. Both measured 2026-08-10.
#
# BARRIER 1 — `noslip` in Float64. FIXED (`4ca15f77`). The build failed with
#
#     Function 'air.convert.f.f64.f.f32' has Metal-unsupported instructions
#     LLVM ERROR: Failed to verify LLVM IR for Metal
#
# because `solver/noslip.mojo` widened to Float64 in 25 places and Metal
# rejects `double`. dog and dog_fetch are the only models declaring `<option
# noslip_iterations="4">`, so no already-gated model was affected and this was
# NOT a regression from `c9ae9a33` — before it the batched env hardcoded
# NOSLIP_ITER=0 and the pass never ran on GPU. That port was needed for NVIDIA
# too, where Float64 is equally banned.
#
# BARRIER 2 — the per-thread stack at NV=79. NOT FIXED, and probably not
# fixable on Apple. `mojo build` now SUCCEEDS and emits a binary; the failure
# has moved to run time, when Metal actually builds the pipeline state:
#
#     Failed to create compute pipeline state (GPU machine code generation):
#     Compute function exceeds available stack space
#
# This is the same ceiling that skips humanoid_CMU (NV=62). dog is NV=79.
#
# ⚠⚠ A GREEN `mojo build` IS NOT EVIDENCE THAT METAL CAN RUN THE KERNEL.
# Metal compiles lazily — machine-code generation happens at pipeline-state
# creation, i.e. on the first launch — so a build that exits 0 proves only
# that valid Metal IR was emitted. This cost a wrong "Apple builds dog"
# conclusion; the only way to know is to RUN it.
#
# CONSEQUENCE: an Apple run cannot gate dog's GPU path. It still gates
# BUILD-level correctness of everything here (type-checking and Metal IR for
# the config's own hooks), which is real but is not numerics. Every NUMERIC
# claim about batched dog has to come from an NVIDIA run.
#
# `fetch` is NOT offered batched: it is Phase 5, and its second free joint
# (njnt 75 / nq 87) has never been through a parity gate at all.
comptime DMDogStandBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMDogStandWalkModel, DMDogStandConfig, N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMDogWalkBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMDogStandWalkModel, DMDogMoveConfig[DOG_WALK_SPEED], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMDogTrotBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMDogTrotModel, DMDogMoveConfig[DOG_TROT_SPEED], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMDogRunBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMDogRunModel, DMDogMoveConfig[DOG_RUN_SPEED], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
