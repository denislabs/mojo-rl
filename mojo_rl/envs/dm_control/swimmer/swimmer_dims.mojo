"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/swimmer6.xml
comptime DM_SWIMMER6_DIMS = ParsedModel(
    nbody=8,
    njoint=8,
    nq=8,
    nv=8,
    ngeom=17,
    nact=5,
    ntex=2,
    nmat=13,
    nlight=2,
    ncam=3,
    nsite=6,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=0,
    timestep=0.002,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)


# mojo_rl/envs/dm_control/assets/swimmer15.xml
comptime DM_SWIMMER15_DIMS = ParsedModel(
    nbody=17,
    njoint=17,
    nq=17,
    nv=17,
    ngeom=35,
    nact=14,
    ntex=2,
    nmat=13,
    nlight=2,
    ncam=3,
    nsite=15,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=0,
    timestep=0.002,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
