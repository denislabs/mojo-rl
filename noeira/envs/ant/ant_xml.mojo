"""Ant model definition via XML parsing.

Uses the MuJoCo ant.xml directly via ModelDefFromXML — same approach as
walker2d_xml.mojo. All geometry, joint, and actuator data is read at
compile time from the embedded XML string.

Dimensions: NQ=15, NV=14, OBS_DIM=27 (qpos[2:15] + qvel[0:14]), ACTION_DIM=8.
obs_qpos_skip=2 skips the free-joint x and y translations from the observation.
"""

from noeira.physics3d.parser import ModelDefFromXML
from noeira.envs.ant.ant_dims import ANT_DIMS


comptime pm = ANT_DIMS

comptime AntModel = ModelDefFromXML[
    xml_path="noeira/envs/ant/assets/ant.xml",
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
    nsensor=pm.NSENSOR, nsensordata=pm.NSENSORDATA,
    obs_qpos_skip=2,  # skip x, y from free-joint qpos; obs = qpos[2:15] + qvel[0:14] → OBS_DIM=27
    max_contacts=40,
    timestep=pm.TIMESTEP,
]
