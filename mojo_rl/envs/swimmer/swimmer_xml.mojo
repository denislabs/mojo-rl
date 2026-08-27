from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.envs.swimmer.swimmer_dims import SWIMMER_DIMS

# The swimmer.xml uses density=4000 and viscosity=0.1 for fluid dynamics.
# These are parsed from <option> and applied as inertia-box fluid forces
# (viscous + pressure drag) in the integrator.

comptime pm = SWIMMER_DIMS

comptime SwimmerModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/swimmer/assets/swimmer.xml",
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
    # Skip qpos[0]=slider1(x) and qpos[1]=slider2(y) from obs.
    # Obs = [free_body_rot, motor1_rot, motor2_rot] + qvel[0:5] → OBS_DIM=8
    obs_qpos_skip=2,
    max_contacts=5,  # geoms have contype=0, minimal contacts
    timestep=pm.TIMESTEP,
]
