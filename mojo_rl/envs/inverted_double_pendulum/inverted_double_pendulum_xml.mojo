"""InvertedDoublePendulum model definition from embedded MJCF XML."""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML

comptime inverted_double_pendulum_xml = """
<mujoco model="cartpole">
  <compiler coordinate="local" inertiafromgeom="true"/>
  <custom>
    <numeric data="2" name="frame_skip"/>
  </custom>
  <default>
    <joint damping="0.05"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="1e-5 0 -9.81" integrator="RK4" timestep="0.01"/>
  <size nstack="3000"/>
  <worldbody>
    <geom name="floor" pos="0 0 -3.0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
    <body name="cart" pos="0 0 0">
      <joint axis="1 0 0" limited="true" margin="0.01" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
      <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
      <body name="pole" pos="0 0 0">
        <joint axis="0 1 0" name="hinge" pos="0 0 0" type="hinge"/>
        <geom fromto="0 0 0 0 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
        <body name="pole2" pos="0 0 0.6">
          <joint axis="0 1 0" name="hinge2" pos="0 0 0" type="hinge"/>
          <geom fromto="0 0 0 0 0 0.6" name="cpole2" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
          <site name="tip" pos="0 0 .6" size="0.01 0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="500" joint="slider" name="slide"/>
  </actuator>
</mujoco>
"""

comptime pm = parse_xml(inverted_double_pendulum_xml)

# OBS_DIM=9: [cart_x, sin(q1), sin(q2), cos(q1), cos(q2), clip(qvel[0:3],-10,10), 0.0]
# Formula nq-skip+nv = 3-0+3 = 6 but we need 9 for the sin/cos encoding.
# obs_dim_override=9 enables custom_extract_obs_gpu to write the correct 9D obs.
comptime InvertedDoublePendulumModel = ModelDefFromXML[
    xml=inverted_double_pendulum_xml,
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
    obs_dim_override=9,  # custom obs: [cart_x, sin(q1), sin(q2), cos(q1), cos(q2), qvel*3, 0]
    max_contacts=5,  # contype=0 on geoms → minimal contacts
    timestep=pm.TIMESTEP,
]
