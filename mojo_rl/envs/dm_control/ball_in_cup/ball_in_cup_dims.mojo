"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/ball_in_cup.xml
comptime DM_BALL_IN_CUP_DIMS = ParsedModel(
    nbody=3,
    njoint=4,
    nq=4,
    nv=4,
    ngeom=7,
    nact=2,
    ntex=2,
    nmat=13,
    nlight=1,
    ncam=2,
    nsite=3,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=1,
    timestep=0.002,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
