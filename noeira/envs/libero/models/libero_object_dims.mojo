"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from noeira.physics3d.parser.xml_parser import ParsedModel


# noeira/tasks/scenes/libero_object.xml
comptime LIBERO_OBJECT_DIMS = ParsedModel(
    nbody=29,
    njoint=20,
    nq=86,
    nv=75,
    ngeom=175,
    nact=9,
    ntex=18,
    nmat=67,
    nlight=2,
    ncam=6,
    nsite=10,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=0,
    nsensor=0,
    nsensordata=0,
    timestep=0.002,
    max_condim=4,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
