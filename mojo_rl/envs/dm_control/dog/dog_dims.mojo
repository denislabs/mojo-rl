"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/dog_stand_walk.xml
comptime DM_DOG_STAND_WALK_DIMS = ParsedModel(
    nbody=62,
    njoint=74,
    nq=80,
    nv=79,
    ngeom=128,
    nact=38,
    ntex=3,
    nmat=14,
    nlight=1,
    ncam=2,
    nsite=12,
    neq=0,
    nexclude=30,
    npair=0,
    ntendon=8,
    timestep=0.005,
    max_condim=6,
    noslip_iter=4,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/dog_run.xml
comptime DM_DOG_RUN_DIMS = ParsedModel(
    nbody=62,
    njoint=74,
    nq=80,
    nv=79,
    ngeom=128,
    nact=38,
    ntex=3,
    nmat=14,
    nlight=1,
    ncam=2,
    nsite=12,
    neq=0,
    nexclude=30,
    npair=0,
    ntendon=8,
    timestep=0.005,
    max_condim=6,
    noslip_iter=4,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/dog_trot.xml
comptime DM_DOG_TROT_DIMS = ParsedModel(
    nbody=62,
    njoint=74,
    nq=80,
    nv=79,
    ngeom=128,
    nact=38,
    ntex=3,
    nmat=14,
    nlight=1,
    ncam=2,
    nsite=12,
    neq=0,
    nexclude=30,
    npair=0,
    ntendon=8,
    timestep=0.005,
    max_condim=6,
    noslip_iter=4,
    ccd_tol=1e-06,
    ccd_iter=35,
)
