"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/dm_control/assets/manipulation/lift_large_box.xml
comptime LIFT_LARGE_BOX_DIMS = ParsedModel(
    nbody=18,
    njoint=10,
    nq=16,
    nv=15,
    ngeom=22,
    nact=9,
    ntex=2,
    nmat=7,
    nlight=1,
    ncam=1,
    nsite=20,
    neq=0,
    nexclude=4,
    npair=0,
    ntendon=0,
    timestep=0.002,
    max_condim=4,
    noslip_iter=5,
    ccd_tol=1e-06,
    ccd_iter=35,
)
