"""Sawyer Reach-v3 model definition via merge_mjcf.

Merges modular XML fragments following MuJoCo's <include> semantics:
  - sawyer_scene_xml (table, walls, floor, lights, visual settings, solver options)
  - sawyer_block_deps_xml (block textures, materials, mesh)
  - sawyer_deps_xml (compiler flags, named default classes, robot meshes)
  - sawyer_robot_xml (Sawyer 7-DOF arm + gripper + mocap body)
  - reach task XML (object, goal, actuators, weld constraint)

Reference: references/Metaworld-master/metaworld/assets/sawyer_xyz/sawyer_reach_v3.xml
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.metaworld.sawyer_reach_dims import SAWYER_REACH_DIMS

# Block dependencies (textures, materials, mesh)
# From: references/Metaworld-master/metaworld/assets/objects/assets/block_dependencies.xml

# Task-specific XML (object + goal + actuators + equality)

# Merge all fragments (same order as MetaWorld's includes)

comptime pm = SAWYER_REACH_DIMS

comptime SawyerReachModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/metaworld/assets/sawyer_reach.xml",
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
    neq=pm.NEQ,
    max_contacts=30,
    max_equality=6,  # 1 weld = 6 rows
    obs_dim_override=10,  # hand_xyz(3) + gripper(1) + obj_xyz(3) + goal(3)
    action_dim_override=4,  # delta_xyz(3) + gripper(1)
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    # The two kp=400 gripper <position> servos are never actuated through
    # MODEL_DEF.apply_actions: SawyerReachConfig.custom_apply_actions_cpu
    # returns True and writes the mirrored gripper force into qfrc[7]/qfrc[8]
    # itself (mocap control drives the arm). Opt out of the servo guard.
    allow_unsupported_actuators=True,
]
