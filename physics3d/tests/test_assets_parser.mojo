"""Smoke test for material, texture, light, camera, site parsing."""

from physics3d.parser import (
    parse_xml, parse_xml_full,
    TEX_SKYBOX, TEX_2D, TEX_BUILTIN_GRADIENT, TEX_BUILTIN_CHECKER,
    LIGHT_MODE_FIXED,
    CAM_MODE_TRACKCOM,
)

comptime xml = """
<mujoco model="test">
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512"/>
    <material name="MatPlane" texture="texplane" reflectance="0.5" shininess="0.3" specular="1"/>
    <material name="MatGeom" rgba="0.8 0.6 0.4 1" shininess="0.3" specular="1"/>
  </asset>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" diffuse="1 1 1" specular=".1 .1 .1"/>
    <light pos="0 -1 1" dir="0 1 -1" castshadow="false" diffuse="0.5 0.5 0.5"/>
    <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
    <geom name="floor" type="plane" size="40 40 40"/>
    <body name="torso" pos="0 0 1">
      <joint name="root" type="slide" axis="0 0 1"/>
      <geom name="body" type="sphere" size="0.2"/>
      <site name="center" type="sphere" pos="0 0 0" size="0.01"/>
      <site name="tip" type="sphere" pos="0 0 0.2" size="0.005"/>
    </body>
  </worldbody>
</mujoco>
"""

fn main():
    comptime pm = parse_xml(xml)
    print("NBODY =", pm.NBODY, " (expected 2)")
    print("NTEX  =", pm.NTEX,  " (expected 2)")
    print("NMAT  =", pm.NMAT,  " (expected 2)")
    print("NLIGHT=", pm.NLIGHT," (expected 2)")
    print("NCAM  =", pm.NCAM,  " (expected 1)")
    print("NSITE =", pm.NSITE, " (expected 2)")
    print()

    @parameter
    if pm.NTEX != 2 or pm.NMAT != 2 or pm.NLIGHT != 2 or pm.NCAM != 1 or pm.NSITE != 2:
        print("ERROR: dimension mismatch — aborting")
        return

    var fmd = parse_xml_full[
        pm.NBODY, pm.NJOINT, pm.NQ, pm.NV, pm.NGEOM, pm.NACT,
        pm.NTEX, pm.NMAT, pm.NLIGHT, pm.NCAM, pm.NSITE,
    ](xml)

    print("=== Textures ===")
    print("tex[0] type    =", fmd.textures[0].tex_type, "(expected", TEX_SKYBOX, "=skybox)")
    print("tex[0] builtin =", fmd.textures[0].builtin, "(expected", TEX_BUILTIN_GRADIENT, "=gradient)")
    print("tex[0] rgb1_r  =", fmd.textures[0].rgb1_r, "(expected 0.3)")
    print("tex[0] width   =", fmd.textures[0].width, "(expected 512)")
    print("tex[1] type    =", fmd.textures[1].tex_type, "(expected", TEX_2D, "=2d)")
    print("tex[1] builtin =", fmd.textures[1].builtin, "(expected", TEX_BUILTIN_CHECKER, "=checker)")
    print()

    print("=== Materials ===")
    print("mat[0] tex_id      =", fmd.materials[0].tex_id, "(expected 1 = texplane)")
    print("mat[0] reflectance =", fmd.materials[0].reflectance, "(expected 0.5)")
    print("mat[0] shininess   =", fmd.materials[0].shininess, "(expected 0.3)")
    print("mat[1] tex_id      =", fmd.materials[1].tex_id, "(expected -1)")
    print("mat[1] rgba_r      =", fmd.materials[1].rgba_r, "(expected 0.8)")
    print("mat[1] rgba_g      =", fmd.materials[1].rgba_g, "(expected 0.6)")
    print()

    print("=== Lights ===")
    print("light[0] pos_z       =", fmd.lights[0].pos_z, "(expected 1.5)")
    print("light[0] directional =", fmd.lights[0].directional, "(expected True)")
    print("light[0] diffuse_r   =", fmd.lights[0].diffuse_r, "(expected 1.0)")
    print("light[1] pos_y       =", fmd.lights[1].pos_y, "(expected -1.0)")
    print("light[1] castshadow  =", fmd.lights[1].castshadow, "(expected False)")
    print("light[1] diffuse_r   =", fmd.lights[1].diffuse_r, "(expected 0.5)")
    print()

    print("=== Cameras ===")
    print("cam[0] pos_y =", fmd.cameras[0].pos_y, "(expected -3.0)")
    print("cam[0] pos_z =", fmd.cameras[0].pos_z, "(expected 0.3)")
    print("cam[0] mode  =", fmd.cameras[0].mode, "(expected", CAM_MODE_TRACKCOM, "=trackcom)")
    # xyaxes="1 0 0 0 0 1": X=(1,0,0), Y=(0,0,1) → Z=cross(X,Y)=(0,-1,0) → looking along -Y
    # The camera frame has Y pointing up (+Z world) and X pointing right (+X world)
    print("cam[0] quat_w (non-zero):", fmd.cameras[0].quat_w)
    print()

    print("=== Sites ===")
    print("site[0] body_id =", fmd.sites[0].body_id, "(expected 1)")
    print("site[0] pos_z   =", fmd.sites[0].pos_z, "(expected 0.0)")
    print("site[0] size_0  =", fmd.sites[0].size_0, "(expected 0.01)")
    print("site[1] pos_z   =", fmd.sites[1].pos_z, "(expected 0.2)")
    print("site[1] size_0  =", fmd.sites[1].size_0, "(expected 0.005)")
    print()
    print("=== All asset/visual parsing tests done ===")
