"""`dm_control` `quadruped` — the two in-scope tasks as env aliases.

    from mojo_rl.envs.dm_control.quadruped import DMQuadrupedWalk
    var env = DMQuadrupedWalk()

walk and run differ in the target speed AND in the floor geom's half-extent
(`_DEFAULT_TIME_LIMIT * speed`, so 10 vs 100), which is a different XML and
therefore a different model-def alias rather than one config parameter. The
floor size has no effect on the dynamics at these episode lengths; it is
carried because the model is otherwise a verbatim copy of what
`make_model(floor_size=...)` emits, and a gate compares the two.

walk and run are GPU-BATCHED as of 2026-08-07 (`*Batched` below). Getting
there took the whole of blocker E: `Phyics3dBatchedEnv` had no `act` slab, so
quadruped's twelve `<general dyntype="filter">` servos would have run with a
permanently zero activation (E3); `RNE_POST` was never wired into the batched
integrator, so `cacc`/`cfrc_int` — and with them 30 of the 78 observation
dims — would have been zero (E1); the actuator kernel ran once per CONTROL
step where these servos need it once per SUBSTEP; and the reset's
`_find_non_contacting_height` had no batched form, so every lane would have
spawned embedded in the floor.

`fetch` stays CPU-only for now — it is under active development.

`escape` LANDED 2026-08-25, and it is the 49th and last suite task. It needed
three engine features that did not exist when the rest of quadruped was ported
— a heightfield geom, a heightfield whose grid is per-episode STATE, and the
`<rangefinder>` sensor — which is why it sat descoped through the whole port.
⚠ It is CPU-ONLY: `ray_model` is a host-side linear scan and the terrain
rewrite is a CPU write plus an upload, so `HAS_GPU_HOOKS` is False and there is
no `*Batched` alias below.

`fetch` brought two engine prerequisites with it: oriented planes (its four walls are tilted planes) and
condim>=4 friction rows (its ball is the only condim-6 geom in the tree, and
the pyramidal edge builder silently dropped those rows until 004fe439).
"""

from .quadruped_xml import (
    DMQuadrupedWalkModel, DMQuadrupedRunModel, DMQuadrupedFetchModel,
    DMQuadrupedEscapeModel,
)
from .quadruped_config import DMQuadrupedWalkConfig, DMQuadrupedRunConfig
from .quadruped_fetch_config import DMQuadrupedFetchConfig
from .quadruped_escape_config import DMQuadrupedEscapeConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — the driver only ever sees
# truncation at the 1000-step limit.
comptime DMQuadrupedWalk[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMQuadrupedWalkModel, DMQuadrupedWalkConfig, DTYPE, False
]

comptime DMQuadrupedRun[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMQuadrupedRunModel, DMQuadrupedRunConfig, DTYPE, False
]

comptime DMQuadrupedFetch[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMQuadrupedFetchModel, DMQuadrupedFetchConfig, DTYPE, False
]

comptime DMQuadrupedEscape[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMQuadrupedEscapeModel, DMQuadrupedEscapeConfig, DTYPE, False
]


# ── GPU-batched aliases ────────────────────────────────────────────────
#
# N_ENVS is the caller's; the driver instantiates one per training run.
# `TERMINATE_ON_UNHEALTHY=False` because no suite task terminates early —
# the driver only ever sees truncation at MAX_STEPS.
comptime DMQuadrupedWalkBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMQuadrupedWalkModel, DMQuadrupedWalkConfig, N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMQuadrupedRunBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMQuadrupedRunModel, DMQuadrupedRunConfig, N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMQuadrupedEscapeBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMQuadrupedEscapeModel, DMQuadrupedEscapeConfig, N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
"""⚠ THE ONLY BATCHED SUITE MODEL WITH A HEIGHTFIELD, and the first to use
all three of `init_hfield_gpu`, `custom_extract_obs_ray_gpu` and
`compute_reward_and_done_gpu`. Each lane draws its OWN bowl at reset — that is
the point of the per-lane hook, and it is why the terrain lives in `Data`.

⚠ COST. Twenty rangefinders over eighteen geoms is 360 geom queries per lane
per step on top of the physics, and `ray_model` is a linear scan. It is worth
having because those 360 are embarrassingly parallel across the batch, not
because any one of them is cheap."""
