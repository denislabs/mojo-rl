"""`<mesh inertia="shell">` is a hollow part, not a rounding option.

    pixi run mojo run -I . tests/physics3d/test_mesh_inertia_shell_vs_mujoco.mojo

WHAT WAS MISSING. `mjCMesh::ComputeInertia` (user_mesh.cc:1590) has two modes
this tree meets, and we only implemented one. LEGACY spreads the mass through
the enclosed VOLUME; SHELL spreads it over the SURFACE, as if the part were a
thin skin. Three lines differ:

    weight   area                     instead of  |dot(center, n)| * area / 3
    divisor  12                       instead of  20
    abs()    not taken (areas are >=0) instead of taken per face

and the centre of mass is area-weighted rather than volume-weighted
(`ComputeSurfaceArea`, :1230) — with the SAME `center*3/4 + facecen/4` pyramid
formula, which is the part that is easy to drop when transcribing.

⚠ IT IS NOT A SMALL CORRECTION ON A SOLID PART. On shadow_dexee's knuckle STL
the two modes give moments 1.489x, 1.509x and 1.640x apart and different
centres of mass. Keeping the divisor at 20 while switching the weight would
scale every shell moment by 0.6 — a plausible-looking number.

⚠ AND MUJOCO ITSELF SENDS MODELS HERE. Its error for a degenerate mesh reads
"mesh volume is too small . Try setting inertia to shell", so a mesh declaring
it may have no usable volume at all. Three Menagerie models do:
hello_robot_stretch_3 (11 meshes), hello_robot_stretch (8), pndbotics_adam_lite
(4).

MEASURED. stretch_3's two aruco marker bodies had their inertial ORIENTATION
wrong by 2.935e-01 (as `1 - |dot|` of the quaternions); the model-wide figure
is now 1.05e-07 and their `ipos` matches to 1e-11.

⚠ THE STEP-1 SWEEP DOES NOT SEE IT — three scenes move only in their last
digits — because every shell mesh in this tree is a sticker or a marker
weighing 3.6e-06 kg. That is the point of measuring the compiled MODEL rather
than the trajectory: a body too light to move the robot is still a body whose
inertia the engine got wrong, and the next model to use the attribute may not
be a sticker.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_IPOS_X, BODY_IDX_IPOS_Y, BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_W, BODY_IDX_IQUAT_X, BODY_IDX_IQUAT_Y, BODY_IDX_IQUAT_Z,
    BODY_IDX_IXX, BODY_IDX_IYY, BODY_IDX_IZZ,
)

comptime DT = DType.float64

comptime DEXEE_DIR = String(
    "references/mujoco_menagerie-main/shadow_dexee/"
)
comptime STRETCH3 = String(
    "references/mujoco_menagerie-main/hello_robot_stretch_3/scene.xml"
)

# ⚠ TWO BODIES, ONE MESH FILE, ONE ATTRIBUTE APART. Body `a` is the control:
# it shares the loader, the hull and every line of the inertia routine with
# `b`, so a mismatch on `a` is not the shell mode.
comptime XML = String(
    """<mujoco>
  <compiler angle="radian" meshdir="assets"/>
  <asset>
    <mesh name="solid" file="MRH-F-J0Link-Visual,00.stl"/>
    <mesh name="hollow" file="MRH-F-J0Link-Visual,00.stl" inertia="shell"/>
  </asset>
  <worldbody>
    <body name="a"><joint type="hinge" axis="0 0 1"/>
      <geom type="mesh" mesh="solid" mass="0.13077995"/></body>
    <body name="b" pos="1 0 0"><joint type="hinge" axis="0 0 1"/>
      <geom type="mesh" mesh="hollow" mass="0.13077995"/></body>
  </worldbody>
</mujoco>"""
)

# MuJoCo 3.10.0 `body_inertia` / `body_ipos` for the two bodies above.
comptime A_IXX = 6.233881188373815e-05
comptime A_IYY = 5.631648265865958e-05
comptime A_IZZ = 3.5160911517250895e-05
comptime A_PY = -0.015295115474376413
comptime A_PZ = 0.007837674710718487
comptime B_IXX = 9.282365616743105e-05
comptime B_IYY = 8.49855899372866e-05
comptime B_IZZ = 5.766185523852096e-05
comptime B_PY = -0.014307231389314926
comptime B_PZ = 0.009951528427090846

# hello_robot_stretch_3's two aruco marker bodies, whose meshes are shells.
comptime S3_LEFT = 28
comptime S3_LEFT_QW = 0.5196796411760785
comptime S3_LEFT_QX = 0.48403132628106993
comptime S3_LEFT_PZ = -7.54763141549725e-05


def _bodies(xml: String, base: String) raises -> List[Float64]:
    """`[ixx, iyy, izz, ipx, ipy, ipz, qw, qx, qy, qz]` per body."""
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var out = List[Float64]()
    for b in range(dims.get_nbody()):
        var o = b * MODEL_BODY_SIZE
        out.append(Float64(m.bodies.data[o + BODY_IDX_IXX]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IYY]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IZZ]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IPOS_X]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IPOS_Y]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IPOS_Z]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IQUAT_W]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IQUAT_X]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IQUAT_Y]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IQUAT_Z]))
    return out^


def test_shell_and_legacy_on_one_mesh() raises:
    """Same file, same mass, one attribute apart, MuJoCo's two answers."""
    print("=== <mesh inertia=\"shell\"> vs legacy, one file ===")
    var v = _bodies(XML, DEXEE_DIR)
    assert_true(
        len(v) == 3 * 10,
        "the fixture must build a world plus two bodies; got "
        + String(len(v) // 10),
    )
    var a = 10  # body 1
    var b = 20  # body 2
    print("  solid  I", v[a], v[a + 1], v[a + 2], " ipos y/z", v[a + 4],
          v[a + 5])
    print("  MuJoCo  ", A_IXX, A_IYY, A_IZZ, "           ", A_PY, A_PZ)
    print("  shell  I", v[b], v[b + 1], v[b + 2], " ipos y/z", v[b + 4],
          v[b + 5])
    print("  MuJoCo  ", B_IXX, B_IYY, B_IZZ, "           ", B_PY, B_PZ)
    # ⚠ THE CONTROL. It must not move: the shell branch is one `if` away from
    # every line the legacy path uses.
    assert_true(
        abs(v[a] - A_IXX) < 1e-15 and abs(v[a + 1] - A_IYY) < 1e-15
        and abs(v[a + 2] - A_IZZ) < 1e-15 and abs(v[a + 4] - A_PY) < 1e-14
        and abs(v[a + 5] - A_PZ) < 1e-14,
        "the LEGACY body moved; the shell branch has disturbed the default"
        " path. ixx " + String(v[a]) + " against " + String(A_IXX),
    )
    # ⚠ THE MOMENTS ARE 1.489x, 1.509x AND 1.640x THE SOLID ONES, so a file
    # that silently fell back to legacy fails by 33-39%, not by rounding.
    assert_true(
        abs(v[b] - B_IXX) < 1e-15 and abs(v[b + 1] - B_IYY) < 1e-15
        and abs(v[b + 2] - B_IZZ) < 1e-15,
        "the SHELL body has moments (" + String(v[b]) + ", " + String(v[b + 1])
        + ", " + String(v[b + 2]) + ") against MuJoCo's (" + String(B_IXX)
        + ", " + String(B_IYY) + ", " + String(B_IZZ) + "). Matching the"
        " LEGACY row instead means the attribute was ignored; being 0.6x off"
        " means the divisor is still 20 rather than 12.",
    )
    # ⚠ AND THE CENTRE OF MASS MOVES TOO. Area weighting is not volume
    # weighting, so a shell's CoM is its own — checking the moments alone
    # would pass an implementation that switched the divisor and nothing else.
    assert_true(
        abs(v[b + 4] - B_PY) < 1e-14 and abs(v[b + 5] - B_PZ) < 1e-14,
        "the SHELL body's centre of mass is (" + String(v[b + 4]) + ", "
        + String(v[b + 5]) + ") against MuJoCo's (" + String(B_PY) + ", "
        + String(B_PZ) + ") — `ComputeSurfaceArea` weighs each face by AREA.",
    )
    print("  PASS")


def test_stretch3_aruco_markers() raises:
    """The real model: two marker bodies whose meshes declare the attribute."""
    print("=== hello_robot_stretch_3 aruco markers ===")
    var src = read_model_source(STRETCH3)
    var v = _bodies(src[0], src[1])
    var o = S3_LEFT * 10
    assert_true(
        len(v) > o + 9,
        "stretch_3 must have at least " + String(S3_LEFT + 1) + " bodies",
    )
    print("  ipos z ours", v[o + 5], " MuJoCo", S3_LEFT_PZ)
    print("  iquat  ours", v[o + 6], v[o + 7], " MuJoCo", S3_LEFT_QW,
          S3_LEFT_QX)
    assert_true(
        abs(v[o + 5] - S3_LEFT_PZ) < 1e-11,
        "link_SG3_gripper_left_finger_aruco's ipos z is " + String(v[o + 5])
        + " against MuJoCo's " + String(S3_LEFT_PZ),
    )
    # ⚠ THE ORIENTATION IS WHAT MOVED MOST — 2.935e-01 on `1 - |dot|` before.
    var dot = (
        v[o + 6] * S3_LEFT_QW + v[o + 7] * S3_LEFT_QX
        + v[o + 8] * (-0.47983529029276584)
        + v[o + 9] * 0.5151745722717209
    )
    print("  1 - |dot| ", abs(abs(dot) - 1.0))
    assert_true(
        abs(abs(dot) - 1.0) < 1e-6,
        "the marker's inertial frame is " + String(abs(abs(dot) - 1.0))
        + " away from MuJoCo's (as 1 - |dot| of the quaternions); a value"
        " near 0.2935 is the legacy tensor standing in for the shell one.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
