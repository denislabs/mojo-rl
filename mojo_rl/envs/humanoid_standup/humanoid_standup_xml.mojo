"""HumanoidStandup model definition from embedded MJCF XML."""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.envs.humanoid_standup.humanoid_standup_dims import (
    HUMANOID_STANDUP_DIMS,
)


comptime pm = HUMANOID_STANDUP_DIMS

# OBS_DIM = (NQ - obs_qpos_skip) + NV = (24 - 2) + 23 = 45
# Simplified obs: qpos[2:] + qvel (excludes free joint x/y translation).
comptime HumanoidStandupModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/humanoid_standup/assets/humanoid_standup.xml",
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
    nexclude=pm.NEXCLUDE,
    obs_qpos_skip=2,  # skip free joint x/y translation from obs
    max_contacts=50,  # body contacts with ground while lying/standing
    max_tendon=2,  # left_hipknee + right_hipknee
    timestep=pm.TIMESTEP,
]
