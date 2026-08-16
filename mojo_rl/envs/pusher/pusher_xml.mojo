"""Pusher model definition from embedded MJCF XML.

7-DOF robotic arm (PR2-style) pushing a cylinder to a goal position.
Zero gravity table-top manipulation.

Bodies (depth-first):
  0: worldbody
  1: r_shoulder_pan_link     (joint: r_shoulder_pan_joint, hinge Z)
  2: r_shoulder_lift_link    (joint: r_shoulder_lift_joint, hinge Y)
  3: r_upper_arm_roll_link   (joint: r_upper_arm_roll_joint, hinge X)
  4: r_upper_arm_link        (no joint, welded to parent)
  5: r_elbow_flex_link       (joint: r_elbow_flex_joint, hinge Y)
  6: r_forearm_roll_link     (joint: r_forearm_roll_joint, hinge X)
  7: r_forearm_link          (no joint, welded to parent)
  8: r_wrist_flex_link       (joint: r_wrist_flex_joint, hinge Y)
  9: r_wrist_roll_link       (joint: r_wrist_roll_joint, hinge X)
  10: tips_arm               (no joint, end effector)
  11: object                 (joints: obj_slidey, obj_slidex)
  12: goal                   (joints: goal_slidey, goal_slidex)

Joints: 7 arm hinges + 2 object slides + 2 goal slides = 11
NQ=11, NV=11, NACT=7
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.envs.pusher.pusher_dims import PUSHER_DIMS


comptime pm = PUSHER_DIMS

# OBS_DIM=23: [qpos[:7], qvel[:7], tips_arm_xpos(3), object_xpos(3), goal_xpos(3)]
# Formula nq-skip+nv = 11-0+11 = 22 but we need 23 with body positions.
comptime PusherModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/pusher/assets/pusher.xml",
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
    obs_dim_override=23,  # custom obs: qpos[:7] + qvel[:7] + 3 body positions
    max_contacts=20,
    timestep=pm.TIMESTEP,
]
