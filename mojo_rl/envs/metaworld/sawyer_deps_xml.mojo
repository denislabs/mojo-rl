"""Sawyer robot dependencies — compiler flags, named default classes.

From: references/Metaworld-master/metaworld/assets/objects/assets/xyz_base_dependencies.xml
Mesh references removed (we use primitive approximations).
"""

comptime sawyer_deps_xml = """
<mujocoinclude>
    <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>

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
</mujocoinclude>
"""
