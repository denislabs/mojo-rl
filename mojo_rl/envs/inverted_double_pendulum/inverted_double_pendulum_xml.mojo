"""InvertedDoublePendulum model definition from embedded MJCF XML."""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_dims import (
    INVERTED_DOUBLE_PENDULUM_DIMS,
)


comptime pm = INVERTED_DOUBLE_PENDULUM_DIMS

# OBS_DIM=9: [cart_x, sin(q1), sin(q2), cos(q1), cos(q2), clip(qvel[0:3],-10,10), 0.0]
# Formula nq-skip+nv = 3-0+3 = 6 but we need 9 for the sin/cos encoding.
# obs_dim_override=9 enables custom_extract_obs_gpu to write the correct 9D obs.
comptime InvertedDoublePendulumModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/inverted_double_pendulum/assets/inverted_double_pendulum.xml",
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
    obs_qpos_skip=0,
    obs_dim_override=9,  # custom obs: [cart_x, sin(q1), sin(q2), cos(q1), cos(q2), qvel*3, 0]
    max_contacts=5,  # contype=0 on geoms → minimal contacts
    timestep=pm.TIMESTEP,
]
