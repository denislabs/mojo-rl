"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/robots/assets/so_arm101.xml
comptime SO_ARM101_DIMS = ParsedModel(
    nbody=9,
    njoint=6,
    nq=6,
    nv=6,
    ngeom=32,
    nact=6,
    ntex=1,
    nmat=14,
    nlight=1,
    ncam=0,
    nsite=2,
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
