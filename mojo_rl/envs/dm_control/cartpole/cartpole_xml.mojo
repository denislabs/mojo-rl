"""`dm_control` `cartpole` models — port of `dm_control/suite/cartpole.xml`.

The reference builds the multi-pole variants procedurally with lxml
(`cartpole._make_model(n_poles)`): it clones the first pole body, moves the
floor down to `1 - n_poles - .05`, and pushes the cameras back. Our models are
comptime strings, so the 1/2/3-pole variants are written out here, applying
exactly those transformations.

`<default class="pole">` supplies the ENTIRE definition of every pole joint
and geom — the elements themselves are bare `<joint name="hinge_1"/>` /
`<geom name="pole_1"/>` and the bodies pick the class up via
`childclass="pole"`. That needs MJCF default-class inheritance for structural
attributes plus `childclass` propagation, both added to the parser on
2026-07-29; see docs/DM_CONTROL_PORT.md.

Note `<option ... integrator="RK4">` here, unlike pendulum (which omits the
attribute and therefore gets MuJoCo's Euler default).
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml
from mojo_rl.envs.dm_control.cartpole.cartpole_dims import (
    DM_CARTPOLE1_DIMS,
    DM_CARTPOLE2_DIMS,
    DM_CARTPOLE3_DIMS,
)


# --- shared prefix: everything up to the pole chain -------------------------
comptime _CARTPOLE_HEAD = """
<mujoco model="cart-pole">
  <option timestep="0.01" integrator="RK4">
    <flag contact="disable" energy="enable"/>
  </option>

  <default>
    <default class="pole">
      <joint type="hinge" axis="0 1 0"  damping="2e-6"/>
      <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" material="self" mass=".1"/>
    </default>
  </default>

  <worldbody>
    <light name="light" pos="0 0 6"/>
"""

comptime _CARTPOLE_TAIL = """
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-1.8 1.8" solreflimit=".08 1" damping="5e-4"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" material="self"  mass="1"/>
"""

comptime _CARTPOLE_END = """
    </body>
  </worldbody>

  <actuator>
    <motor name="slide" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1" />
  </actuator>
</mujoco>
"""


# --- 1 pole (the reference file verbatim) -----------------------------------
comptime _cartpole1_body = (
    _CARTPOLE_HEAD
    + """    <camera name="fixed" pos="0 -4 1" zaxis="0 -1 0"/>
    <camera name="lookatcart" mode="targetbody" target="cart" pos="0 -2 2"/>
    <geom name="floor" pos="0 0 -.05" size="4 4 .2" type="plane" material="grid"/>
    <geom name="rail1" type="capsule" pos="0  .07 1" zaxis="1 0 0" size="0.02 2" material="decoration" />
    <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2" material="decoration" />
"""
    + _CARTPOLE_TAIL
    + """      <body name="pole_1" childclass="pole">
        <joint name="hinge_1"/>
        <geom name="pole_1"/>
      </body>
"""
    + _CARTPOLE_END
)

# --- 2 poles: floor to 1-2-.05 = -1.05, cameras back -----------------------
comptime _cartpole2_body = (
    _CARTPOLE_HEAD
    + """    <camera name="fixed" pos="0 -5 1" zaxis="0 -1 0"/>
    <camera name="lookatcart" mode="targetbody" target="cart" pos="0 -4 2"/>
    <geom name="floor" pos="0 0 -1.05" size="4 4 .2" type="plane" material="grid"/>
    <geom name="rail1" type="capsule" pos="0  .07 1" zaxis="1 0 0" size="0.02 2" material="decoration" />
    <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2" material="decoration" />
"""
    + _CARTPOLE_TAIL
    + """      <body name="pole_1" childclass="pole">
        <joint name="hinge_1"/>
        <geom name="pole_1"/>
        <body name="pole_2" pos="0 0 1" childclass="pole">
          <joint name="hinge_2"/>
          <geom name="pole_2"/>
        </body>
      </body>
"""
    + _CARTPOLE_END
)

# --- 3 poles: floor to 1-3-.05 = -2.05, cameras back -----------------------
comptime _cartpole3_body = (
    _CARTPOLE_HEAD
    + """    <camera name="fixed" pos="0 -7 1" zaxis="0 -1 0"/>
    <camera name="lookatcart" mode="targetbody" target="cart" pos="0 -6 2"/>
    <geom name="floor" pos="0 0 -2.05" size="4 4 .2" type="plane" material="grid"/>
    <geom name="rail1" type="capsule" pos="0  .07 1" zaxis="1 0 0" size="0.02 2" material="decoration" />
    <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2" material="decoration" />
"""
    + _CARTPOLE_TAIL
    + """      <body name="pole_1" childclass="pole">
        <joint name="hinge_1"/>
        <geom name="pole_1"/>
        <body name="pole_2" pos="0 0 1" childclass="pole">
          <joint name="hinge_2"/>
          <geom name="pole_2"/>
          <body name="pole_3" pos="0 0 1" childclass="pole">
            <joint name="hinge_3"/>
            <geom name="pole_3"/>
          </body>
        </body>
      </body>
"""
    + _CARTPOLE_END
)


comptime dm_cartpole1_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _cartpole1_body
)
comptime dm_cartpole2_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _cartpole2_body
)
comptime dm_cartpole3_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _cartpole3_body
)

comptime pm1 = DM_CARTPOLE1_DIMS

comptime pm2 = DM_CARTPOLE2_DIMS

comptime pm3 = DM_CARTPOLE3_DIMS

# obs = cart_position(1) + per-pole (zz, xz) + qvel(nv)
#     = 1 + 2*n_poles + (1 + n_poles)
comptime DMCartpole1Model = ModelDefFromXML[
    xml=dm_cartpole1_xml,
    xml_path="mojo_rl/envs/dm_control/assets/cartpole1.xml",
    nbody=pm1.NBODY, njoint=pm1.NJOINT, nq=pm1.NQ, nv=pm1.NV,
    ngeom=pm1.NGEOM, nact=pm1.NACT, ntex=pm1.NTEX, nmat=pm1.NMAT,
    nlight=pm1.NLIGHT, ncam=pm1.NCAM, nsite=pm1.NSITE,
    max_contacts=4,
    obs_dim_override=5,
    timestep=pm1.TIMESTEP,
]

comptime DMCartpole2Model = ModelDefFromXML[
    xml=dm_cartpole2_xml,
    xml_path="mojo_rl/envs/dm_control/assets/cartpole2.xml",
    nbody=pm2.NBODY, njoint=pm2.NJOINT, nq=pm2.NQ, nv=pm2.NV,
    ngeom=pm2.NGEOM, nact=pm2.NACT, ntex=pm2.NTEX, nmat=pm2.NMAT,
    nlight=pm2.NLIGHT, ncam=pm2.NCAM, nsite=pm2.NSITE,
    max_contacts=4,
    obs_dim_override=8,
    timestep=pm2.TIMESTEP,
]

comptime DMCartpole3Model = ModelDefFromXML[
    xml=dm_cartpole3_xml,
    xml_path="mojo_rl/envs/dm_control/assets/cartpole3.xml",
    nbody=pm3.NBODY, njoint=pm3.NJOINT, nq=pm3.NQ, nv=pm3.NV,
    ngeom=pm3.NGEOM, nact=pm3.NACT, ntex=pm3.NTEX, nmat=pm3.NMAT,
    nlight=pm3.NLIGHT, ncam=pm3.NCAM, nsite=pm3.NSITE,
    max_contacts=4,
    obs_dim_override=11,
    timestep=pm3.TIMESTEP,
]

# Body indices: 0 = world, 1 = cart, 2.. = pole_1, pole_2, ...
comptime CART_BODY_IDX: Int = 1
comptime FIRST_POLE_BODY_IDX: Int = 2
