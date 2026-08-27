"""`dm_control` `pendulum` model — port of `dm_control/suite/pendulum.xml`.

Body/option/actuator content is byte-identical to the reference; the three
`<include file="./common/*.xml"/>` lines are resolved by `merge_mjcf` over the
shared fragments in `..common_xml` (same order as the reference).

Notes for anyone diffing against the reference:
  - `<option timestep="0.02">` carries NO `integrator` attribute, so MuJoCo's
    default (Euler) applies — hence `INTEGRATOR = "euler"` in the config.
    cartpole/acrobot DO say `integrator="RK4"`; do not copy this blindly.
  - `<flag contact="disable"/>` is not modelled by our parser (gap G5), but
    the pendulum can never touch the floor anyway: the hinge sits at z=0.6
    and the tip sphere (r=0.05) bottoms out at z=0.05. `max_contacts=4` is
    slack, not a contact budget.
  - nq = nv = 1 (single hinge), nu = 1, nbody = 2 (world + pole).
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.pendulum.pendulum_dims import DM_PENDULUM_DIMS





comptime pm = DM_PENDULUM_DIMS

comptime DMPendulumModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/pendulum.xml",
    nbody=pm.NBODY,
    njoint=pm.NJOINT,
    nq=pm.NQ,
    nv=pm.NV,
    ngeom=pm.NGEOM,
    nact=pm.NACT,
    ntex=pm.NTEX,
    nmat=pm.NMAT,
    nlight=pm.NLIGHT,
    ncam=pm.NCAM,
    nsite=pm.NSITE,
    max_contacts=4,
    # obs = [xmat_zz, xmat_xz, qvel] — orientation is a rotation-matrix
    # column pair, not qpos, so the default nq-skip+nv formula does not apply.
    obs_dim_override=3,
    timestep=pm.TIMESTEP,
]

# Body index of "pole" in the parsed model (0 = worldbody).
comptime POLE_BODY_IDX: Int = 1
