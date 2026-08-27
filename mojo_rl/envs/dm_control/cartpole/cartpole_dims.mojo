"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/cartpole1.xml
comptime DM_CARTPOLE1_DIMS = ParsedModel(
    nbody=3,
    njoint=2,
    nq=2,
    nv=2,
    ngeom=5,
    nact=1,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=0,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=0,
    timestep=0.01,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/cartpole2.xml
comptime DM_CARTPOLE2_DIMS = ParsedModel(
    nbody=4,
    njoint=3,
    nq=3,
    nv=3,
    ngeom=6,
    nact=1,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=0,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=0,
    timestep=0.01,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/cartpole3.xml
comptime DM_CARTPOLE3_DIMS = ParsedModel(
    nbody=5,
    njoint=4,
    nq=4,
    nv=4,
    ngeom=7,
    nact=1,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=0,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=0,
    timestep=0.01,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
