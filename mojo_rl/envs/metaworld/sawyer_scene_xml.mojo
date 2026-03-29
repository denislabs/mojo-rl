"""MetaWorld basic scene — table, walls, floor, lights, visual settings, solver options.

From: references/Metaworld-master/metaworld/assets/scene/basic_scene.xml
"""

comptime sawyer_scene_xml = """
<mujocoinclude>
    <option timestep="0.0025" iterations="50" tolerance="1e-10" solver="Newton"
            jacobian="dense" cone="elliptic"/>

    <visual>
        <map fogstart="1.5" fogend="5" force="0.1" znear="0.01"/>
        <quality shadowsize="4096" offsamples="4"/>
        <headlight ambient="0.4 0.4 0.4"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.50 0.495 0.48"
                 rgb2="0.50 0.495 0.48" width="32" height="32"/>
        <texture name="T_table" type="cube"
                 file="mojo_rl/envs/metaworld/assets/textures/wood2.png"/>
        <texture name="T_floor" type="2d"
                 file="mojo_rl/envs/metaworld/assets/textures/floor2.png"/>
        <texture name="T_wallmetal" type="cube"
                 file="mojo_rl/envs/metaworld/assets/textures/metal.png"/>

        <material name="basic_floor" texture="T_floor" texrepeat="12 12"
                  shininess="0.3" specular="0.5" reflectance="0.2"/>
        <material name="table_wood" texture="T_table" shininess="0.3"
                  specular="0.5"/>
        <material name="table_col" rgba="0.3 0.3 1.0 0.5" shininess="0"
                  specular="0"/>
        <material name="wall_metal" texture="T_wallmetal" shininess="1"
                  reflectance="1" specular="0.5"/>

        <mesh file="mojo_rl/envs/metaworld/assets/meshes/table/tablebody.stl"
              name="tablebody" scale="1 1 1"/>
        <mesh file="mojo_rl/envs/metaworld/assets/meshes/table/tabletop.stl"
              name="tabletop" scale="1 1 1"/>
    </asset>

    <worldbody>
        <light castshadow="false" directional="true" diffuse="0.3 0.3 0.3"
               specular="0.3 0.3 0.3" pos="-1 -1 1" dir="1 1 -1"/>
        <light directional="true" diffuse="0.3 0.3 0.3"
               specular="0.3 0.3 0.3" pos="1 -1 1" dir="-1 1 -1"/>
        <light castshadow="false" directional="true" diffuse="0.3 0.3 0.3"
               specular="0.3 0.3 0.3" pos="0 1 1" dir="0 -1 -1"/>

        <body name="tablelink" pos="0 0.6 0">
            <geom material="table_wood" group="1" type="box" size="0.7 0.4 0.027"
                  pos="0 0 -0.027" conaffinity="0" contype="0"/>
            <geom material="table_wood" group="1" type="mesh" mesh="tablebody"
                  pos="0 0 -0.65" conaffinity="0" contype="0"/>
            <geom material="table_col" group="4" pos="0 0 -0.46"
                  size="0.7 0.4 0.46" type="box" conaffinity="1" contype="0"/>
        </body>

        <body name="RetainingWall" pos="0.0 0.6 0.06">
            <geom material="wall_metal" type="box" size="0.7 0.01 0.06"
                  pos="0 -0.39 0" conaffinity="1" condim="3" contype="0"/>
            <geom material="wall_metal" type="box" size="0.7 0.01 0.06"
                  pos="0 0.39 0" conaffinity="1" condim="3" contype="0"/>
            <geom material="wall_metal" type="box" size="0.01 0.38 0.06"
                  pos="-0.69 0 0" conaffinity="1" condim="3" contype="0"/>
            <geom material="wall_metal" type="box" size="0.01 0.38 0.06"
                  pos="0.69 0 0" conaffinity="1" condim="3" contype="0"/>
        </body>

        <geom name="floor" size="4 4 0.1" pos="0 0 -0.913" conaffinity="1"
              contype="1" type="plane" material="basic_floor" condim="3"/>
    </worldbody>
</mujocoinclude>
"""
