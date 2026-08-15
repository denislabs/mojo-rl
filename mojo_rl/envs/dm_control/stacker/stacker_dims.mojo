"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/stacker_2.xml
comptime DM_STACKER_2_DIMS = ParsedModel(
    nbody=13,
    njoint=14,
    nq=14,
    nv=14,
    ngeom=22,
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


# mojo_rl/envs/dm_control/assets/stacker_4.xml
comptime DM_STACKER_4_DIMS = ParsedModel(
    nbody=15,
    njoint=20,
    nq=20,
    nv=20,
    ngeom=24,
    nact=5,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=12,
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
