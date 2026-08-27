"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/metaworld/assets/sawyer_reach.xml
comptime SAWYER_REACH_DIMS = ParsedModel(
    nbody=34,
    njoint=10,
    nq=16,
    nv=15,
    ngeom=37,
    nact=2,
    ntex=5,
    nmat=8,
    nlight=3,
    ncam=7,
    nsite=8,
    neq=1,
    nexclude=0,
    npair=0,
    ntendon=0,
    timestep=0.0025,
    max_condim=4,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
