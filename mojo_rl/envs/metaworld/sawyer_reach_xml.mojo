"""Sawyer Reach-v3 model definition via merge_mjcf.

Merges modular XML fragments following MuJoCo's <include> semantics:
  - sawyer_scene_xml (table, walls, floor, lights, solver options)
  - sawyer_deps_xml (compiler flags, named default classes)
  - sawyer_robot_xml (Sawyer 7-DOF arm + gripper + mocap body)
  - reach task XML (object, goal, actuators, weld constraint)

Reference: references/Metaworld-master/metaworld/assets/sawyer_xyz/sawyer_reach_v3.xml
"""

from mojo_rl.physics3d.parser import parse_xml, merge_mjcf, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .sawyer_scene_xml import sawyer_scene_xml
from .sawyer_deps_xml import sawyer_deps_xml
from .sawyer_robot_xml import sawyer_robot_xml

# Task-specific XML (object + goal + actuators + equality)
comptime sawyer_reach_task_xml = """
<mujoco>
    <worldbody>
        <body name="obj" pos="0 0.6 0.02">
            <joint name="objjoint" type="free" limited="false" damping="0" armature="0"/>
            <inertial pos="0 0 0" mass="0.75"
                      diaginertia="8.80012e-04 8.80012e-04 8.80012e-04"/>
            <geom name="objGeom" type="cylinder" pos="0 0 0"
                  solimp="0.99 0.99 0.01" size="0.02 0.02" rgba="1 0 0 1"
                  solref="0.01 1" contype="1" conaffinity="1"
                  friction="1 0.1 0.002" condim="4"/>
        </body>

        <site name="goal" pos="-0.1 0.8 0.2" size="0.02" rgba="0.8 0 0 1"/>
    </worldbody>

    <actuator>
        <position ctrllimited="true" ctrlrange="-1 1" joint="r_close" kp="400"/>
        <position ctrllimited="true" ctrlrange="-1 1" joint="l_close" kp="400"/>
    </actuator>

    <equality>
        <weld body1="mocap" body2="hand" solref="0.02 1"/>
    </equality>
</mujoco>
"""

# Merge all fragments (same order as MetaWorld's includes)
comptime sawyer_reach_xml = merge_mjcf(
    sawyer_scene_xml,
    sawyer_deps_xml,
    sawyer_robot_xml,
    sawyer_reach_task_xml,
)

comptime pm = parse_xml(sawyer_reach_xml)

comptime SawyerReachModel = ModelDefFromXML[
    xml=sawyer_reach_xml,
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
]
