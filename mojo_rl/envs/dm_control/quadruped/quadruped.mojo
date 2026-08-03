"""dm_control `quadruped` — the two in-scope tasks as env aliases.

    from mojo_rl.envs.dm_control.quadruped import DMQuadrupedWalk
    var env = DMQuadrupedWalk()

walk and run differ in the target speed AND in the floor geom's half-extent
(`_DEFAULT_TIME_LIMIT * speed`, so 10 vs 100), which is a different XML and
therefore a different model-def alias rather than one config parameter. The
floor size has no effect on the dynamics at these episode lengths; it is
carried because the model is otherwise a verbatim copy of what
`make_model(floor_size=...)` emits, and a gate compares the two.

CPU ONLY, and unlike the other domains not merely because of a hook ABI: the
GPU-batched facade does not carry `act`, and quadruped's twelve `<general
dyntype="filter">` servos are integrated through it. A batched quadruped
would run with a permanently zero activation — every actuator dead — so the
alias is deliberately not offered.

`escape` stays descoped — it needs heightfield terrain and rangefinders,
neither of which the engine has. `fetch` is IN, and brought two engine
prerequisites with it: oriented planes (its four walls are tilted planes) and
condim>=4 friction rows (its ball is the only condim-6 geom in the tree, and
the pyramidal edge builder silently dropped those rows until 004fe439).
"""

from .quadruped_xml import (
    DMQuadrupedWalkModel, DMQuadrupedRunModel, DMQuadrupedFetchModel,
)
from .quadruped_config import DMQuadrupedWalkConfig, DMQuadrupedRunConfig
from .quadruped_fetch_config import DMQuadrupedFetchConfig
from ...phyics3d_env import Phyics3dEnv


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
