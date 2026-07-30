"""dm_control `swimmer` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.swimmer import DMSwimmer6, DMSwimmer15
    var env = DMSwimmer6()

Both run the same `Swimmer` task; they differ only in how many links
`_make_model` generated, which changes NQ/NV and therefore the observation
width (25 vs 61). The reference registers exactly these two under
`@SUITE.add('benchmarking')`; its `swimmer(n_links=3)` helper is not a
registered task and is not ported.

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI does not carry the mocap fields yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .swimmer_xml import DMSwimmer6Model, DMSwimmer15Model
from .swimmer_config import DMSwimmerConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMSwimmer6[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMSwimmer6Model, DMSwimmerConfig, DTYPE, False
]

comptime DMSwimmer15[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMSwimmer15Model, DMSwimmerConfig, DTYPE, False
]
