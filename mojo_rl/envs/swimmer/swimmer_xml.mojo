from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML

# The swimmer.xml uses density=4000 and viscosity=0.1 for fluid dynamics.
# These are parsed from <option> and applied as inertia-box fluid forces
# (viscous + pressure drag) in the integrator.
comptime swimmer_xml = """
<mujoco model="swimmer">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option density="4000" integrator="RK4" timestep="0.01" viscosity="0.1"/>
  <default>
    <geom conaffinity="0" condim="1" contype="0" material="geom" rgba="0.8 0.6 .4 1"/>
    <joint armature='0.1'  />
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="30 30" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom condim="3" material="MatPlane" name="floor" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1" size="40 40 0.1" type="plane"/>
    <body name="torso" pos="0 0 0">
      <camera name="track" mode="trackcom" pos="0 -3 3" xyaxes="1 0 0 0 1 1"/>
      <geom density="1000" fromto="1.5 0 0 0.5 0 0" size="0.1" type="capsule"/>
      <joint axis="1 0 0" name="slider1" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" name="slider2" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" name="free_body_rot" pos="0 0 0" type="hinge"/>
      <body name="mid" pos="0.5 0 0">
        <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="motor1_rot" pos="0 0 0" range="-100 100" type="hinge"/>
        <body name="back" pos="-1 0 0">
          <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
          <joint axis="0 0 1" limited="true" name="motor2_rot" pos="0 0 0" range="-100 100" type="hinge"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="motor1_rot"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="motor2_rot"/>
  </actuator>
</mujoco>
"""

comptime pm = parse_xml(swimmer_xml)

comptime SwimmerModel = ModelDefFromXML[
    xml=swimmer_xml,
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
