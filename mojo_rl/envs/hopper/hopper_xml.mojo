from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.envs.hopper.hopper_dims import HOPPER_DIMS


comptime pm = HOPPER_DIMS

comptime HopperModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/hopper/assets/hopper.xml",
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
    obs_qpos_skip=1,
    max_contacts=20,
    timestep=pm.TIMESTEP,
]
