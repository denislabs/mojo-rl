"""`dm_control` `cartpole` models — port of `dm_control/suite/cartpole.xml`.

The reference builds the multi-pole variants procedurally with lxml
(`cartpole._make_model(n_poles)`): it clones the first pole body, moves the
floor down to `1 - n_poles - .05`, and pushes the cameras back. Our models are
comptime strings, so the 1/2/3-pole variants are written out here, applying
exactly those transformations.

`<default class="pole">` supplies the ENTIRE definition of every pole joint
and geom — the elements themselves are bare `<joint name="hinge_1"/>` /
`<geom name="pole_1"/>` and the bodies pick the class up via
`childclass="pole"`. That needs MJCF default-class inheritance for structural
attributes plus `childclass` propagation, both added to the parser on
2026-07-29; see docs/DM_CONTROL_PORT.md.

Note `<option ... integrator="RK4">` here, unlike pendulum (which omits the
attribute and therefore gets MuJoCo's Euler default).
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.cartpole.cartpole_dims import (
    DM_CARTPOLE1_DIMS,
    DM_CARTPOLE2_DIMS,
    DM_CARTPOLE3_DIMS,
)



# ⚠ THE THREE POLE VARIANTS WERE BUILT HERE by comptime concatenation —
# `_CARTPOLE_HEAD + <the cameras/floor/rails that differ> + _CARTPOLE_TAIL +
# <the pole chain> + _CARTPOLE_END`. They are three FILES now
# (`assets/cartpole1.xml`, `2`, `3`), extracted verbatim from what that
# concatenation produced, so the per-variant differences — the floor z
# (-.05 / -1.05 / -2.05), the camera distances, and the nesting depth of the
# pole chain — are visible in the assets instead of being assembled.



comptime pm1 = DM_CARTPOLE1_DIMS

comptime pm2 = DM_CARTPOLE2_DIMS

comptime pm3 = DM_CARTPOLE3_DIMS

# obs = cart_position(1) + per-pole (zz, xz) + qvel(nv)
#     = 1 + 2*n_poles + (1 + n_poles)
comptime DMCartpole1Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/cartpole1.xml",
    nbody=pm1.NBODY, njoint=pm1.NJOINT, nq=pm1.NQ, nv=pm1.NV,
    ngeom=pm1.NGEOM, nact=pm1.NACT, ntex=pm1.NTEX, nmat=pm1.NMAT,
    nlight=pm1.NLIGHT, ncam=pm1.NCAM, nsite=pm1.NSITE,
    max_contacts=4,
    obs_dim_override=5,
    timestep=pm1.TIMESTEP,
]

comptime DMCartpole2Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/cartpole2.xml",
    nbody=pm2.NBODY, njoint=pm2.NJOINT, nq=pm2.NQ, nv=pm2.NV,
    ngeom=pm2.NGEOM, nact=pm2.NACT, ntex=pm2.NTEX, nmat=pm2.NMAT,
    nlight=pm2.NLIGHT, ncam=pm2.NCAM, nsite=pm2.NSITE,
    max_contacts=4,
    obs_dim_override=8,
    timestep=pm2.TIMESTEP,
]

comptime DMCartpole3Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/cartpole3.xml",
    nbody=pm3.NBODY, njoint=pm3.NJOINT, nq=pm3.NQ, nv=pm3.NV,
    ngeom=pm3.NGEOM, nact=pm3.NACT, ntex=pm3.NTEX, nmat=pm3.NMAT,
    nlight=pm3.NLIGHT, ncam=pm3.NCAM, nsite=pm3.NSITE,
    max_contacts=4,
    obs_dim_override=11,
    timestep=pm3.TIMESTEP,
]

# Body indices: 0 = world, 1 = cart, 2.. = pole_1, pole_2, ...
comptime CART_BODY_IDX: Int = 1
comptime FIRST_POLE_BODY_IDX: Int = 2
