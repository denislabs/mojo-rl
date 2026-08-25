"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/quadruped_walk.xml
comptime DM_QUADRUPED_WALK_DIMS = ParsedModel(
    nbody=18,
    njoint=17,
    nq=23,
    nv=22,
    ngeom=20,
    nact=12,
    ntex=3,
    nmat=14,
    nlight=1,
    ncam=4,
    nsite=29,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=12,
    timestep=0.005,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/quadruped_run.xml
comptime DM_QUADRUPED_RUN_DIMS = ParsedModel(
    nbody=18,
    njoint=17,
    nq=23,
    nv=22,
    ngeom=20,
    nact=12,
    ntex=3,
    nmat=14,
    nlight=1,
    ncam=4,
    nsite=29,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=12,
    timestep=0.005,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/quadruped_fetch.xml
comptime DM_QUADRUPED_FETCH_DIMS = ParsedModel(
    nbody=19,
    njoint=18,
    nq=30,
    nv=28,
    ngeom=25,
    nact=12,
    ntex=3,
    nmat=14,
    nlight=2,
    ncam=4,
    nsite=30,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=12,
    timestep=0.005,
    max_condim=6,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/quadruped_escape.xml
comptime DM_QUADRUPED_ESCAPE_DIMS = ParsedModel(
    nbody=18,
    njoint=17,
    nq=23,
    nv=22,
    ngeom=21,
    nact=12,
    ntex=3,
    nmat=14,
    nlight=1,
    ncam=4,
    nsite=29,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=12,
    timestep=0.005,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
