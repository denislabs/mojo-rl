"""Shared MJCF fragments — port of `dm_control/suite/common/*.xml`.

Every suite domain starts with the same three `<include>` lines:

    <include file="./common/visual.xml"/>
    <include file="./common/skybox.xml"/>
    <include file="./common/materials.xml"/>

They are purely cosmetic (headlight/shadow settings, a skybox texture, and
the named material palette that geoms reference by name), but the materials
must exist for `material="grid"` / `material="self"` lookups to resolve.

Domains compose them exactly as MuJoCo's include semantics do, via
`merge_mjcf`, keeping the include order from the reference `.xml`:

    comptime pendulum_xml = merge_mjcf(
        dm_visual_xml, dm_skybox_xml, dm_materials_xml, _pendulum_body_xml
    )

Content is byte-for-byte the reference apart from XML comments.
"""


comptime dm_visual_xml = """
<mujoco>
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
  </visual>
</mujoco>
"""


comptime dm_skybox_xml = """
<mujoco>
  <asset>
      <texture name="skybox" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0"
               width="800" height="800" mark="random" markrgb="1 1 1"/>
  </asset>
</mujoco>
"""


comptime dm_materials_xml = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <material name="self" rgba=".7 .5 .3 1"/>
    <material name="self_default" rgba=".7 .5 .3 1"/>
    <material name="self_highlight" rgba="0 .5 .3 1"/>
    <material name="effector" rgba=".7 .4 .2 1"/>
    <material name="effector_default" rgba=".7 .4 .2 1"/>
    <material name="effector_highlight" rgba="0 .5 .3 1"/>
    <material name="decoration" rgba=".3 .5 .7 1"/>
    <material name="eye" rgba="0 .2 1 1"/>
    <material name="target" rgba=".6 .3 .3 1"/>
    <material name="target_default" rgba=".6 .3 .3 1"/>
    <material name="target_highlight" rgba=".6 .3 .3 .4"/>
    <material name="site" rgba=".5 .5 .5 .3"/>
  </asset>
</mujoco>
"""
