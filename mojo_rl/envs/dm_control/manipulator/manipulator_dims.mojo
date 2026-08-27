"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/manipulator_bring_ball.xml
comptime DM_MANIPULATOR_BRING_BALL_DIMS = ParsedModel(
    nbody=12,
    njoint=11,
    nq=11,
    nv=11,
    ngeom=21,
    nact=5,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=9,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=2,
    timestep=0.001,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/manipulator_bring_peg.xml
comptime DM_MANIPULATOR_BRING_PEG_DIMS = ParsedModel(
    nbody=13,
    njoint=11,
    nq=11,
    nv=11,
    ngeom=25,
    nact=5,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=15,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=2,
    timestep=0.001,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/manipulator_insert_ball.xml
comptime DM_MANIPULATOR_INSERT_BALL_DIMS = ParsedModel(
    nbody=13,
    njoint=11,
    nq=11,
    nv=11,
    ngeom=25,
    nact=5,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=10,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=2,
    timestep=0.001,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/manipulator_insert_peg.xml
comptime DM_MANIPULATOR_INSERT_PEG_DIMS = ParsedModel(
    nbody=14,
    njoint=11,
    nq=11,
    nv=11,
    ngeom=28,
    nact=5,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=17,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=2,
    timestep=0.001,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
