from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_dims import (
    INVERTED_PENDULUM_DIMS,
)


comptime pm = INVERTED_PENDULUM_DIMS

comptime InvertedPendulumModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/inverted_pendulum/assets/inverted_pendulum.xml",
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
    obs_qpos_skip=0,  # full qpos in obs: [cart_x, pole_angle] + qvel → OBS_DIM=4
    max_contacts=5,
    timestep=pm.TIMESTEP,
]
