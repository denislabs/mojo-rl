"""`dm_control` `ball_in_cup` model — port of `dm_control/suite/ball_in_cup.xml`.

VERBATIM apart from the three `<include>` lines, which `merge_mjcf` splices
in. Unlike `point_mass`, nothing here is substituted: this is the first ported
domain whose `<tendon>` is load-bearing and expressed directly.

WHAT THIS MODEL NEEDED THAT DID NOT EXIST
-----------------------------------------
`<spatial>` tendons. The string is a two-site polyline from the ball's site to
the cup's, with `limited="true" range="0 0.3"` — so it is inextensible past
30 cm and does nothing at all below that. Before this port the engine had only
FIXED (joint-coefficient) tendons, in two disjoint representations neither of
which could express a site-routed length:

  - `dynamics/tendon.mojo`      — length + dense moment arm (mj_tendon)
  - `constraints/tendon_limit.mojo` — the `mjCNSTR_LIMIT_TENDON` row
  - `full_parser` `_fill_tendons` — `<tendon>` had NO runtime parsing at all;
    `fields_build` hardcoded `ntendon = 0`, so every tendon record was dead.

The limit is built as a ROW OF THE SAME SYSTEM as the contacts, not as a
post-pass. ball_in_cup is precisely the shape that made the sequential split
visible on finger (commit 04a7c508): a caught ball rests on the cup capsules
while the string is taut, on shared dofs.

ORDERING. Our geom/site/body/joint numbering coincides with MuJoCo's here
(geoms: ground, cup_part_0..4, ball; sites: cup, target, ball; bodies: world,
cup, ball) because every world geom precedes the first body — the interleaving
that bit `point_mass` does not occur. The parity test pins all four orders
explicitly rather than trusting that.

CONE. `<option>` is absent, so MuJoCo's defaults apply: timestep 0.002,
Newton solver, Euler integrator, PYRAMIDAL cone. The pyramidal cone matters —
tendon limit rows are built on the pyramidal edge list only, and
`ModelDefFromXML` raises if a model asks for elliptic with a limited tendon.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.ball_in_cup.ball_in_cup_dims import (
    DM_BALL_IN_CUP_DIMS,
)





comptime bicp = DM_BALL_IN_CUP_DIMS

# obs = position (qpos, 4) + velocity (qvel, 4) = 8
comptime DMBallInCupModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/ball_in_cup.xml",
    nbody=bicp.NBODY, njoint=bicp.NJOINT, nq=bicp.NQ, nv=bicp.NV,
    ngeom=bicp.NGEOM, nact=bicp.NACT, ntex=bicp.NTEX, nmat=bicp.NMAT,
    nlight=bicp.NLIGHT, ncam=bicp.NCAM, nsite=bicp.NSITE,
    max_tendon=bicp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    # The ball can touch several cup capsules at once while it settles; 8 is
    # comfortably above the 3-4 MuJoCo reports at rest in the cup.
    max_contacts=8,
    obs_dim_override=8,
    obs_qpos_skip=0,
    timestep=bicp.TIMESTEP,
]

# --- Indices, in OUR ordering (== MuJoCo's here; the parity test pins both).
comptime BALL_BODY_IDX: Int = 2
comptime CUP_SITE_IDX: Int = 0
comptime TARGET_SITE_IDX: Int = 1
comptime BALL_SITE_IDX: Int = 2
comptime BALL_GEOM_IDX: Int = 6
comptime CUP_GEOM_FIRST: Int = 1  # cup_part_0
comptime CUP_GEOM_LAST: Int = 5  # cup_part_4

# `site_size['target', [0, 2]]` and `geom_size['ball', 0]`, which
# `Physics.in_target` differences. Asserted against the model tensors in the
# parity test rather than trusted.
comptime TARGET_HALF_X: Float64 = 0.05
comptime TARGET_HALF_Z: Float64 = 0.05
comptime BALL_RADIUS: Float64 = 0.025
