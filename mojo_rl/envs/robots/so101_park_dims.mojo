"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from mojo_rl.physics3d.parser.xml_parser import ParsedModel


# mojo_rl/envs/robots/assets/so101_park_k0.xml
comptime SO101_PARK_K0_DIMS = ParsedModel(
    nbody=9,
    njoint=6,
    nq=6,
    nv=6,
    ngeom=32,
    nact=6,
    ntex=1,
    nmat=14,
    nlight=1,
    ncam=1,
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


# mojo_rl/envs/robots/assets/so101_park_k3.xml
comptime SO101_PARK_K3_DIMS = ParsedModel(
    nbody=12,
    njoint=9,
    nq=27,
    nv=24,
    ngeom=35,
    nact=6,
    ntex=1,
    nmat=14,
    nlight=1,
    ncam=1,
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


# mojo_rl/envs/robots/assets/so101_park_k6.xml
comptime SO101_PARK_K6_DIMS = ParsedModel(
    nbody=15,
    njoint=12,
    nq=48,
    nv=42,
    ngeom=38,
    nact=6,
    ntex=1,
    nmat=14,
    nlight=1,
    ncam=1,
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


# mojo_rl/envs/robots/assets/so101_park_k9.xml
comptime SO101_PARK_K9_DIMS = ParsedModel(
    nbody=18,
    njoint=15,
    nq=69,
    nv=60,
    ngeom=41,
    nact=6,
    ntex=1,
    nmat=14,
    nlight=1,
    ncam=1,
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
