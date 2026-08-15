"""Sawyer Reach-v3 model definition via merge_mjcf.

Merges modular XML fragments following MuJoCo's <include> semantics:
  - sawyer_scene_xml (table, walls, floor, lights, visual settings, solver options)
  - sawyer_block_deps_xml (block textures, materials, mesh)
  - sawyer_deps_xml (compiler flags, named default classes, robot meshes)
  - sawyer_robot_xml (Sawyer 7-DOF arm + gripper + mocap body)
  - reach task XML (object, goal, actuators, weld constraint)

Reference: references/Metaworld-master/metaworld/assets/sawyer_xyz/sawyer_reach_v3.xml
"""

from mojo_rl.physics3d.parser import merge_mjcf, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .sawyer_scene_xml import sawyer_scene_xml
from .sawyer_deps_xml import sawyer_deps_xml
from .sawyer_robot_xml import sawyer_robot_xml
from mojo_rl.envs.metaworld.sawyer_reach_dims import SAWYER_REACH_DIMS

# Block dependencies (textures, materials, mesh)
# From: references/Metaworld-master/metaworld/assets/objects/assets/block_dependencies.xml
comptime sawyer_block_deps_xml = """
<mujocoinclude>
    <asset>
        <texture name="T_block_wood" type="cube"
                 file="mojo_rl/envs/metaworld/assets/textures/wood4.png"/>
        <material name="block_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0"/>
        <material name="block_wood" texture="T_block_wood" shininess="1"
                  reflectance="0.7" specular="0.5"/>
        <material name="block_red" rgba="0.8 0 0 1" shininess="0.2"
                  reflectance="0.2" specular="0.5"/>
        <mesh file="mojo_rl/envs/metaworld/assets/meshes/block/block.stl" name="block"/>
    </asset>

    <default>
        <default class="block_base">
            <joint armature="0.001" damping="2" limited="true"/>
            <geom conaffinity="0" contype="0" group="1" type="mesh"/>
            <position ctrllimited="true" ctrlrange="0 1.57"/>
            <default class="block_col">
                <geom conaffinity="1" condim="4" contype="1" group="4"
                      material="block_col" solimp="0.99 0.99 0.01" solref="0.01 1"/>
            </default>
        </default>
    </default>
</mujocoinclude>
"""

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
                  friction="1 0.1 0.002" condim="4" material="block_wood"/>
        </body>

        <site name="goal" pos="-0.1 0.8 0.2" size="0.02" rgba="0.8 0 0 1"/>
    </worldbody>

    <actuator>
        <position ctrllimited="true" ctrlrange="-1 1" joint="r_close" kp="400" user="1"/>
        <position ctrllimited="true" ctrlrange="-1 1" joint="l_close" kp="400" user="1"/>
    </actuator>

    <equality>
        <!-- relpose is MetaWorld's `reset_mocap_welds`, baked in.
             `SawyerMocapBase.reset_mocap_welds` (metaworld/sawyer_xyz_env.py)
             OVERWRITES eq_data at reset with
             `[0,0,0, 0,0,0, -1,0,0,0, 5]` — relpose position zero, an
             identity quaternion AND TORQUESCALE 5 — so the hand tracks the
             mocap body itself, with the orientation rows scaled by 5.
             MuJoCo's COMPILER derives something else entirely from qpos0 here:
             pos (1.1355, 0.1603, 0.317), quat (0.64279, -0.76604, 0, 0), which
             makes the hand track `mocap (x) relpose`, over a metre away.
             ⚠ WRITING IT OUT IS NOT A DEVIATION FROM METAWORLD, IT IS THE PORT
             OF A RUNTIME STEP METAWORLD PERFORMS. We used to get it by
             accident: `relpose` defaulted to identity in our parser, so the
             derived value never existed. Once that default was fixed to follow
             MuJoCo (2026-08-12) the arm was flung across the workspace and the
             obj fell off the gripper — see
             tests/physics3d/test_sawyer_settle_vs_mujoco, whose reference side
             performs exactly this zeroing. -->
        <weld body1="mocap" body2="hand" relpose="0 0 0 1 0 0 0" torquescale="5" solref="0.02 1"/>
    </equality>
</mujoco>
"""

# Merge all fragments (same order as MetaWorld's includes)
comptime sawyer_reach_xml = merge_mjcf(
    sawyer_scene_xml,
    sawyer_block_deps_xml,
    sawyer_deps_xml,
    sawyer_robot_xml,
    sawyer_reach_task_xml,
)

comptime pm = SAWYER_REACH_DIMS

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
    # The two kp=400 gripper <position> servos are never actuated through
    # MODEL_DEF.apply_actions: SawyerReachConfig.custom_apply_actions_cpu
    # returns True and writes the mirrored gripper force into qfrc[7]/qfrc[8]
    # itself (mocap control drives the arm). Opt out of the servo guard.
    allow_unsupported_actuators=True,
]
