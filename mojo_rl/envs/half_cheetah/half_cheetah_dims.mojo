"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/half_cheetah/assets/half_cheetah.xml
comptime HALF_CHEETAH_DIMS = ParsedModel(
    nbody=8,
    njoint=9,
    nq=9,
    nv=9,
    ngeom=9,
    nact=6,
    ntex=3,
    nmat=2,
    nlight=1,
    ncam=1,
    nsite=0,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=0,
    timestep=0.01,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
