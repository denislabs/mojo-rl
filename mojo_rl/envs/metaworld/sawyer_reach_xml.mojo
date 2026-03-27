"""Sawyer Reach-v3 model definition via XML parsing.

Flattened from MetaWorld's modular XML files:
  - basic_scene.xml (table, walls, floor, solver options)
  - xyz_base_dependencies.xml (compiler flags, named defaults)
  - xyz_base.xml (Sawyer 7-DOF arm + gripper + mocap body)
  - sawyer_reach_v3.xml (reach object + goal site + actuators + weld constraint)

All <include> tags resolved into a single XML string.
Mesh geoms replaced with primitive collision approximations (contype=0 on visual meshes).

Reference: references/Metaworld-master/metaworld/envs/assets_v2/
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

comptime sawyer_reach_xml = """
<mujoco model="sawyer_reach_v3">
  <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
  <option timestep="0.0025" iterations="50" tolerance="1e-10" solver="Newton"
          jacobian="dense" cone="elliptic"/>

  <default>
    <default class="xyz_base">
      <joint armature="0.001" damping="2" limited="true"/>
      <geom conaffinity="0" contype="0" group="1"/>
      <default class="base_col">
        <geom conaffinity="1" condim="4" contype="1" group="4" margin="0.001"
              solimp="0.8 0.9 0.01" solref="0.02 1"/>
      </default>
    </default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.495 0.48"
             rgb2="0.5 0.495 0.48" width="32" height="32"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0 0 0"
             rgb2="0.8 0.8 0.8" width="100" height="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1"
              texrepeat="60 60" texture="texplane"/>
  </asset>

  <worldbody>
    <!-- Table -->
    <body name="tablelink" pos="0 0.6 0">
      <geom type="box" size="0.7 0.4 0.027" pos="0 0 -0.027"
            conaffinity="0" contype="0" rgba="0.6 0.4 0.2 1"/>
      <geom group="4" pos="0.0 0.0 -0.46" size="0.7 0.4 0.46" type="box"
            conaffinity="1" contype="0"/>
    </body>

    <!-- Retaining walls -->
    <body name="RetainingWall" pos="0.0 0.6 0.06">
      <geom type="box" size="0.7 0.01 0.06" pos="0 -0.39 0"
            conaffinity="1" condim="3" contype="0" rgba="0.5 0.5 0.5 1"/>
      <geom type="box" size="0.7 0.01 0.06" pos="0 0.39 0"
            conaffinity="1" condim="3" contype="0" rgba="0.5 0.5 0.5 1"/>
      <geom type="box" size="0.01 0.38 0.06" pos="-0.69 0 0"
            conaffinity="1" condim="3" contype="0" rgba="0.5 0.5 0.5 1"/>
      <geom type="box" size="0.01 0.38 0.06" pos="0.69 0 0"
            conaffinity="1" condim="3" contype="0" rgba="0.5 0.5 0.5 1"/>
    </body>

    <!-- Floor -->
    <geom name="floor" size="4 4 0.1" pos="0 0 -0.913" conaffinity="1"
          contype="1" type="plane" condim="3" material="MatPlane"/>

    <!-- ================================================================== -->
    <!-- Sawyer Robot (from xyz_base.xml, flattened) -->
    <!-- ================================================================== -->
    <body name="base" pos="0 0 0">
      <inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/>

      <body name="controller_box" pos="0 0 0">
        <inertial pos="-0.325 0 -0.38" mass="46.64"
                  diaginertia="1.71363 1.27988 0.809981"/>
        <geom size="0.11 0.2 0.265" pos="-0.325 0 -0.38" type="box"
              rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
      </body>

      <body name="pedestal_feet" pos="0 0 0">
        <inertial pos="-0.1225 0 -0.758" mass="167.09"
                  diaginertia="8.16095 9.59375 15.0785"/>
        <geom size="0.385 0.35 0.155" pos="-0.1225 0 -0.758" type="box"
              rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
      </body>

      <body name="torso" pos="0 0 0">
        <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08"/>
        <geom size="0.05 0.05 0.05" type="box" contype="0" conaffinity="0"
              rgba="0.2 0.2 0.2 1"/>
      </body>

      <body name="pedestal" pos="0 0 0">
        <inertial pos="0 0 0" quat="0.659267 -0.259505 -0.260945 0.655692"
                  mass="60.864" diaginertia="6.0869 5.81635 4.20915"/>
        <geom size="0.18 0.31" pos="-0.02 0 -0.29" type="cylinder"
              rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
      </body>

      <body name="right_arm_base_link" pos="0 0 0">
        <inertial pos="-0.0006241 -2.8025e-05 0.065404"
                  quat="-0.209285 0.674441 0.227335 0.670558"
                  mass="2.0687" diaginertia="0.00740351 0.00681776 0.00672942"/>
        <geom size="0.08 0.12" pos="0 0 0.12" type="cylinder"
              rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>

        <body name="right_l0" pos="0 0 0.08">
          <inertial pos="0.024366 0.010969 0.14363"
                    quat="0.894823 0.00899958 -0.170275 0.412573"
                    mass="5.3213" diaginertia="0.0651588 0.0510944 0.0186218"/>
          <joint name="right_j0" pos="0 0 0" axis="0 0 1"
                 limited="true" range="-3.0503 3.0503" damping="10"/>
          <geom size="0.07 0.1" pos="0 0 0.14" type="cylinder"
                rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>

          <body name="head" pos="0 0 0.2965">
            <inertial pos="0.0053207 -2.6549e-05 0.1021"
                      quat="0.999993 7.08405e-05 -0.00359857 -0.000626247"
                      mass="1.5795" diaginertia="0.0118334 0.00827089 0.00496574"/>
            <geom size="0.08 0.08 0.06" pos="0 0 0.08" type="box"
                  rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>
          </body>

          <body name="right_l1" pos="0.081 0.05 0.237" quat="0.5 -0.5 0.5 0.5">
            <inertial pos="-0.0030849 -0.026811 0.092521"
                      quat="0.424888 0.891987 0.132364 -0.0794296"
                      mass="4.505" diaginertia="0.0224339 0.0221624 0.0097097"/>
            <joint name="right_j1" pos="0 0 0" axis="0 0 1"
                   limited="true" range="-3.8 -0.5" damping="10"/>
            <geom size="0.06 0.12" pos="0 0 0.07" type="cylinder"
                  rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>

            <body name="right_l2" pos="0 -0.14 0.1425" quat="0.707107 0.707107 0 0">
              <inertial pos="-0.00016044 -0.014967 0.13582"
                        quat="0.707831 -0.0524761 0.0516007 0.702537"
                        mass="1.745" diaginertia="0.0257928 0.025506 0.00292515"/>
              <joint name="right_j2" pos="0 0 0" axis="0 0 1"
                     limited="true" range="-3.0426 3.0426" damping="10"/>
              <geom size="0.06 0.17" pos="0 0 0.08" type="cylinder"
                    rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>

              <body name="right_l3" pos="0 -0.042 0.26" quat="0.707107 -0.707107 0 0">
                <inertial pos="-0.0048135 -0.0281 -0.084154"
                          quat="0.902999 0.385391 -0.0880901 0.168247"
                          mass="2.5097" diaginertia="0.0102404 0.0096997 0.00369622"/>
                <joint name="right_j3" pos="0 0 0" axis="0 0 1"
                       limited="true" range="-3.0439 3.0439" damping="10"/>
                <geom size="0.05 0.06" pos="0 -0.06 -0.07" type="cylinder"
                      rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>

                <body name="right_l4" pos="0 -0.125 -0.1265" quat="0.707107 0.707107 0 0">
                  <inertial pos="-0.0018844 0.0069001 0.1341"
                            quat="0.803612 0.031257 -0.0298334 0.593582"
                            mass="1.1136" diaginertia="0.0136549 0.0135493 0.00127353"/>
                  <joint name="right_j4" pos="0 0 0" axis="0 0 1"
                         limited="true" range="-2.9761 2.9761" damping="10"/>
                  <geom size="0.045 0.15" pos="0 0 0.11" type="cylinder"
                        rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>

                  <body name="right_l5" pos="0 0.031 0.275" quat="0.707107 -0.707107 0 0">
                    <inertial pos="0.0061133 -0.023697 0.076416"
                              quat="0.404076 0.9135 0.0473125 0.00158335"
                              mass="1.5625" diaginertia="0.00474131 0.00422857 0.00190672"/>
                    <joint name="right_j5" pos="0 0 0" axis="0 0 1"
                           limited="true" range="-2.9761 2.9761" damping="10"/>
                    <geom size="0.04 0.06" pos="0 0 0.06" type="cylinder"
                          rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>

                    <body name="right_l6" pos="0 -0.11 0.1053"
                          quat="0.0616248 0.06163 -0.704416 0.704416">
                      <inertial pos="-8.0726e-06 0.0085838 -0.0049566"
                                quat="0.479044 0.515636 -0.513069 0.491322"
                                mass="0.3292" diaginertia="0.000360258 0.000311068 0.000214974"/>
                      <joint name="right_j6" pos="0 0 0" axis="0 0 1"
                             limited="true" range="-4.7124 4.7124" damping="10"/>
                      <geom size="0.055 0.025" pos="0 0.015 -0.01" type="cylinder"
                            rgba="0.5 0.1 0.1 1" contype="0" conaffinity="0"/>

                      <body name="right_hand" pos="0 0 0.0245" quat="0.707107 0 0 0.707107">
                        <inertial pos="0 0 0" mass="1e-08" diaginertia="1e-08 1e-08 1e-08"/>
                        <geom size="0.035 0.014" pos="0 0 0.015" type="cylinder"
                              rgba="0 0 0 1" contype="0" conaffinity="0"/>

                        <body name="hand" pos="0 0 0.12" quat="-1 0 1 0">
                          <site name="endEffector" pos="0.04 0 0" size="0.01"/>
                          <geom name="rail" type="box" pos="-0.05 0 0" size="0.005 0.055 0.005"
                                rgba="0.5 0.5 0.5 1" condim="3" friction="2 0.1 0.002"
                                contype="1" conaffinity="1"/>

                          <!-- Right gripper claw -->
                          <body name="rightclaw" pos="0 -0.05 0">
                            <geom class="base_col" name="rightclaw_it" condim="4" margin="0.001"
                                  type="box" pos="0 0 0" size="0.045 0.003 0.015"
                                  rgba="1 1 1 1" contype="1" conaffinity="1"/>
                            <joint name="r_close" pos="0 0 0" axis="0 1 0"
                                   range="0 0.04" armature="100" damping="1000"
                                   limited="true" type="slide"/>
                            <site name="rightEndEffector" pos="0.045 0 0" size="0.01"/>
                            <body name="rightpad" pos="0 0.003 0">
                              <geom name="rightpad_geom" condim="4" margin="0.001" type="box"
                                    pos="0 0 0" size="0.045 0.003 0.015" rgba="1 1 1 1"
                                    solimp="0.95 0.99 0.01" solref="0.01 1"
                                    friction="2 0.1 0.002" contype="1" conaffinity="1" mass="1"/>
                            </body>
                          </body>

                          <!-- Left gripper claw -->
                          <body name="leftclaw" pos="0 0.05 0">
                            <geom class="base_col" name="leftclaw_it" condim="4" margin="0.001"
                                  type="box" pos="0 0 0" size="0.045 0.003 0.015"
                                  rgba="0 1 1 1" contype="1" conaffinity="1"/>
                            <joint name="l_close" pos="0 0 0" axis="0 1 0"
                                   range="-0.03 0" armature="100" damping="1000"
                                   limited="true" type="slide"/>
                            <site name="leftEndEffector" pos="0.045 0 0" size="0.01"/>
                            <body name="leftpad" pos="0 -0.003 0">
                              <geom name="leftpad_geom" condim="4" margin="0.001" type="box"
                                    pos="0 0 0" size="0.045 0.003 0.015" rgba="0 1 1 1"
                                    solimp="0.95 0.99 0.01" solref="0.01 1"
                                    friction="2 0.1 0.002" contype="1" conaffinity="1"/>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Mocap body (externally controlled position target for weld constraint) -->
    <body mocap="true" name="mocap" pos="0 0.6 0.2">
      <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0.5 0.5 0.3"
            size="0.01" type="sphere"/>
    </body>

    <!-- Reach target object -->
    <body name="obj" pos="0 0.6 0.02">
      <joint name="objjoint" type="free" limited="false" damping="0" armature="0"/>
      <inertial pos="0 0 0" mass="0.75"
                diaginertia="8.80012e-04 8.80012e-04 8.80012e-04"/>
      <geom name="objGeom" type="cylinder" pos="0 0 0"
            solimp="0.99 0.99 0.01" size="0.02 0.02" rgba="1 0 0 1"
            solref="0.01 1" contype="1" conaffinity="1"
            friction="1 0.1 0.002" condim="4"/>
    </body>

    <!-- Reach goal (site only, no physics) -->
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
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
]
