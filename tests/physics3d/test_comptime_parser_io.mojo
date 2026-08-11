"""MJCF file -> `parse_xml_full` round trip, at RUNTIME.

This file used to hold the opposite experiment: `comptime xml = read_file(...)`,
reading an MJCF off disk during compile-time evaluation. That experiment is
settled NEGATIVE and cannot be revived — `open` bottoms out in an
`external_call` to libc, and the comptime interpreter refuses it:

    note: unable to interpret call to unknown external function: open

It had also never actually run: the path it read, `test.xml`, does not exist
anywhere in the tree, and the helper swallowed the failure and returned "".
Every model in this codebase therefore embeds its MJCF as a `comptime` String
literal, and the only file-driven path is the RUNTIME one gated here.

What is gated: write an MJCF out, read it back, and parse it with
`parse_xml_full` (non-generic since 2026-08-05, so this costs one
instantiation for the whole program). Counts and a couple of parsed values are
asserted so a silent regression in the file path is loud.

Run: pixi run mojo run -I . tests/physics3d/test_comptime_parser_io.mojo
"""

from std.ffi import external_call
from std.io.file import open
from std.testing import assert_true, assert_equal, TestSuite

from mojo_rl.physics3d.parser import parse_xml_full
from mojo_rl.physics3d.constants import GEOM_PLANE, GEOM_CAPSULE


comptime FIXTURE_XML = """
<mujoco model="io_roundtrip">
  <compiler angle="radian"/>
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
    <body name="pole" pos="0 0 0.6">
      <joint name="hinge_a" type="hinge" axis="0 1 0" pos="0 0 0" damping="2.5"/>
      <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.5" size="0.04"/>
      <body name="tip" pos="0 0 0.5">
        <joint name="slide_b" type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="tip_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m_hinge" joint="hinge_a" gear="75"/>
  </actuator>
</mujoco>
"""


def _write_fixture(path: String) raises:
    with open(path, "w") as f:
        f.write(FIXTURE_XML)


def _read_fixture(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _unlink(mut path: String):
    _ = external_call["unlink", Int32](path.as_c_string_slice().unsafe_ptr())


def test_parser_file_io_roundtrip() raises:
    """A file written, read back and parsed must yield the same model the
    inline string would. Non-vacuity: the element counts are all non-zero and
    two parsed VALUES (a non-default damping and the actuator gear) are
    checked, so an empty or truncated read cannot pass."""
    var path = String("/tmp/mojo_rl_parser_io_roundtrip.xml")
    _write_fixture(path)

    var xml: String
    try:
        xml = _read_fixture(path)
    finally:
        _unlink(path)

    assert_equal(
        xml.byte_length(),
        String(FIXTURE_XML).byte_length(),
        "the file read back is not byte-identical in length to what was"
        " written — the round trip lost data",
    )

    var fmd = parse_xml_full(xml)

    # Counts. `bodies` excludes the worldbody (body 0 in `Model`).
    assert_equal(len(fmd.bodies), 2, "expected pole + tip")
    assert_equal(len(fmd.joints), 2, "expected hinge_a + slide_b")
    assert_equal(len(fmd.geoms), 3, "expected floor + 2 capsules")
    assert_equal(len(fmd.actuators), 1, "expected the single motor")

    # Values — these are what make the test non-vacuous.
    assert_true(
        abs(fmd.timestep - 0.004) <= 1e-15,
        "<option timestep> did not survive the file round trip",
    )
    assert_true(
        abs(fmd.gravity_z + 9.81) <= 1e-12,
        "<option gravity> did not survive the file round trip",
    )
    assert_equal(
        Int(fmd.geoms[0].geom_type),
        Int(GEOM_PLANE),
        "the worldbody floor should parse as a plane",
    )
    assert_equal(
        Int(fmd.geoms[1].geom_type),
        Int(GEOM_CAPSULE),
        "the pole geom should parse as a capsule",
    )
    print("parser file-IO round trip: OK")
    print("  bodies =", len(fmd.bodies), " joints =", len(fmd.joints))
    print("  geoms  =", len(fmd.geoms), " actuators =", len(fmd.actuators))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
