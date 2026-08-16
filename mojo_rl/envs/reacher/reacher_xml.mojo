"""Reacher model definition from embedded MJCF XML.

Two-link planar arm with fingertip end-effector and a movable target.
Bodies: worldbody(0), body0(1), body1(2), fingertip(3), target(4)
Joints: joint0(hinge), joint1(hinge), target_x(slide), target_y(slide)
NQ=4, NV=4, NACT=2
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.envs.reacher.reacher_dims import REACHER_DIMS


comptime pm = REACHER_DIMS

# OBS_DIM=10: [cos(q0), cos(q1), sin(q0), sin(q1), qpos[2:4], qvel[0:2], delta_xy]
# Formula nq-skip+nv = 4-0+4 = 8 but we need 10 for cos/sin encoding + fingertip delta.
comptime ReacherModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/reacher/assets/reacher.xml",
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
    obs_dim_override=10,  # custom obs: cos/sin encoding + target pos + vel + delta
    max_contacts=5,
    timestep=pm.TIMESTEP,
]
