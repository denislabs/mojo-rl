"""dm_control `swimmer` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.swimmer import DMSwimmer6, DMSwimmer15
    var env = DMSwimmer6()

Both run the same `Swimmer` task; they differ only in how many links
`_make_model` generated, which changes NQ/NV and therefore the observation
width (25 vs 61). The reference registers exactly these two under
`@SUITE.add('benchmarking')`; its `swimmer(n_links=3)` helper is not a
registered task and is not ported.

GPU-BATCHED as of 2026-08-08 (`*Batched` below). It needed no new engine
work beyond blocker E's: `site_frame_velocity_gpu` (E2) covers
`body_velocities`, the mocap target rides the batched sync (blocker H), and
the FLUID path — the only thing here that turns joint torque into locomotion,
since contacts are disabled and gravity does nothing to a body sliding in the
x-y plane — already runs through the batched integrators' passive seam.
"""

from .swimmer_xml import DMSwimmer6Model, DMSwimmer15Model
from .swimmer_config import DMSwimmerConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMSwimmer6[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMSwimmer6Model, DMSwimmerConfig, DTYPE, False
]

comptime DMSwimmer15[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMSwimmer15Model, DMSwimmerConfig, DTYPE, False
]


# ── GPU-batched aliases ────────────────────────────────────────────────
comptime DMSwimmer6Batched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMSwimmer6Model, DMSwimmerConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime DMSwimmer15Batched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMSwimmer15Model, DMSwimmerConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
