from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.envs.walker2d.walker2d_dims import WALKER2D_DIMS


comptime pm = WALKER2D_DIMS

comptime Walker2dModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/walker2d/assets/walker2d.xml",
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
    obs_qpos_skip=1,  # skip rootx=qpos[0]; obs = qpos[1:9] + qvel[0:9] → OBS_DIM=17
    max_contacts=20,
    timestep=pm.TIMESTEP,
]
