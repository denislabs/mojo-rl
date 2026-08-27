"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/dog_fetch.xml
comptime DM_DOG_FETCH_DIMS = ParsedModel(
    nbody=63,
    njoint=75,
    nq=87,
    nv=85,
    ngeom=134,
    nact=38,
    ntex=3,
    nmat=15,
    nlight=1,
    ncam=4,
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
