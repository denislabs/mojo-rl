from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.envs.half_cheetah.half_cheetah_dims import HALF_CHEETAH_DIMS



comptime pm = HALF_CHEETAH_DIMS

comptime HalfCheetahModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/half_cheetah/assets/half_cheetah.xml",
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
    max_contacts=20,
    obs_qpos_skip=1,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.PYRAMIDAL,  # MuJoCo default (XML has no <option cone=...>)
]
