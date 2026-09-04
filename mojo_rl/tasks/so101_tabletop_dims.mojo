"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/tasks/scenes/so101_tabletop.xml
comptime SO101_TABLETOP_DIMS = ParsedModel(
    nbody=13,
    njoint=9,
    nq=27,
    nv=24,
    ngeom=37,
    nact=6,
    ntex=1,
    nmat=14,
    nlight=2,
    ncam=1,
    nsite=3,
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
