"""`<hfield elevation>` — the grid written in the XML (AUD-14), against 3.12.

    pixi run mojo run -I . tests/physics3d/test_hfield_elevation_vs_mujoco.mojo

A heightfield can carry its grid inline instead of in a PNG or a binary file.
We parsed the `nrow`/`ncol` and then wrote ZEROS — a flat plate where the model
declares terrain, silently. Not a numeric drift: the terrain is simply gone,
and a robot walks across a floor that the model says is a hill.

Two rules, and both are easy to get wrong in a way that still looks like a
heightfield:

1. ⚠⚠ THE ROWS ARE REVERSED ON THE WAY IN. `mjXReader` flips them —
   "copy in reverse row order, so XML string is top-to-bottom"
   (xml_native_reader.cc:2285-2292) — because `hfield_data` row 0 is the
   field's minus-y edge while a human writing a grid puts the far edge on the
   first line. Reading it straight through MIRRORS the terrain about y, which
   is a perfectly plausible heightfield and the wrong one.

2. The data is then rescaled min-max to [0, 1] exactly as a decoded file is
   (`mjCHField::Compile`, user_objects.cc:4881-4895), so the numbers in the
   XML are shape, not metres; `size[2]` supplies the metres.

⚠ THE FIXTURE IS DELIBERATELY NOT SYMMETRIC IN EITHER AXIS. A grid that reads
the same forwards and backwards cannot see rule 1 at all, and an
already-normalised one cannot see rule 2. `0 1 2 3 / 4 5 6 7 / 8 9 10 11` has
a distinct value in every cell, so a row flip, a column flip, a transpose and a
missing rescale each give a different answer.

⚠ AND THE ZERO CASE IS NOT A BUG. With neither a `file` nor an `elevation`,
MuJoCo fills zeros (xml_native_reader.cc:2296-2300) — the one case where the
old behaviour was right, and the last test holds it.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.physics3d.parser.full_parser import parse_xml_full

comptime TOL = 1e-7  # MuJoCo stores hfield_data as float32

comptime HF_XML = String(
    """<mujoco model="inline_hfield">
  <compiler angle="radian"/>
  <asset>
    <hfield name="terrain" nrow="3" ncol="4" size="2 2 0.5 0.1"
            elevation="ELEVDATA"/>
  </asset>
  <worldbody>
    <geom name="hf" type="hfield" hfield="terrain" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""
)

comptime BARE_XML = String(
    """<mujoco model="bare_hfield">
  <compiler angle="radian"/>
  <asset>
    <hfield name="terrain" nrow="3" ncol="4" size="2 2 0.5 0.1"/>
  </asset>
  <worldbody>
    <geom name="hf" type="hfield" hfield="terrain" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""
)


def _xml(elev: String) -> String:
    return HF_XML.replace(String("ELEVDATA"), elev)


def _compare(elev: String, label: String, mut n: Int) raises:
    var xml = _xml(elev)
    var mujoco = Python.import_module("mujoco")
    var mm = mujoco.MjModel.from_xml_string(xml)
    var fmd = parse_xml_full(xml, String("."))
    assert_true(
        len(fmd.hfield_data) == 12,
        label + ": we produced " + String(len(fmd.hfield_data))
        + " grid values, expected 12",
    )
    print("  ", label)
    for k in range(12):
        var ours = fmd.hfield_data[k]
        var refv = Float64(py=mm.hfield_data[k])
        n += 1
        assert_true(
            abs(ours - refv) <= TOL,
            label + " cell " + String(k) + ": ours " + String(ours)
            + ", MuJoCo " + String(refv),
        )
    print("      ours  ", fmd.hfield_data[0], fmd.hfield_data[1],
          fmd.hfield_data[2], fmd.hfield_data[3], "...")
    print("      MuJoCo", Float64(py=mm.hfield_data[0]),
          Float64(py=mm.hfield_data[1]), Float64(py=mm.hfield_data[2]),
          Float64(py=mm.hfield_data[3]), "...")


def test_a_monotone_grid_matches_cell_for_cell() raises:
    """Every cell distinct, so a flip or a transpose cannot pass."""
    print("=== AUD-14: elevation='0 1 2 3  4 5 6 7  8 9 10 11' ===")
    var n = 0
    _compare(String("0 1 2 3  4 5 6 7  8 9 10 11"), String("monotone"), n)

    # ⚠ THE ROW FLIP, NAMED. MuJoCo's row 0 is the XML's LAST row, so cell 0
    # holds the value 8 scaled by 1/11 and cell 8 holds 0. Stated as its own
    # assertion because "12 cells agree" would also pass if BOTH sides were
    # mirrored, and this one says which way round is right.
    var fmd = parse_xml_full(
        _xml(String("0 1 2 3  4 5 6 7  8 9 10 11")), String(".")
    )
    print("  cell 0 =", fmd.hfield_data[0], " (8/11 = 0.7272…)")
    print("  cell 8 =", fmd.hfield_data[8], " (0/11 = 0)")
    assert_true(
        abs(fmd.hfield_data[0] - 8.0 / 11.0) <= TOL
        and abs(fmd.hfield_data[8]) <= TOL,
        "the rows were NOT reversed: cell 0 is " + String(fmd.hfield_data[0])
        + " and cell 8 is " + String(fmd.hfield_data[8]) + ". The terrain is"
        " mirrored about y.",
    )
    print("  values compared against MuJoCo:", n)
    assert_true(n == 12, "expected 12 comparisons, made " + String(n))


def test_a_non_monotone_grid_normalises_the_same_way() raises:
    """Negative values and a repeated maximum — the rescale, on its own.

    ⚠ THE MONOTONE FIXTURE ABOVE CANNOT SEPARATE `v/max` FROM
    `(v - min)/(max - min)`, because its minimum is 0. This one's minimum is
    -3, so the two rules differ in every cell.
    """
    print("=== elevation='-3 5 -3 5  0 0 0 0  2 2 2 2' ===")
    var n = 0
    _compare(
        String("-3 5 -3 5  0 0 0 0  2 2 2 2"), String("non-monotone"), n
    )
    print("  values compared against MuJoCo:", n)
    assert_true(n == 12, "expected 12 comparisons, made " + String(n))


def test_a_constant_grid_does_not_divide_by_zero() raises:
    """`emax - emin` is 0 here; MuJoCo subtracts the min and skips the divide."""
    print("=== elevation='7 7 7 7  7 7 7 7  7 7 7 7' ===")
    var n = 0
    _compare(
        String("7 7 7 7  7 7 7 7  7 7 7 7"), String("constant"), n
    )
    var fmd = parse_xml_full(
        _xml(String("7 7 7 7  7 7 7 7  7 7 7 7")), String(".")
    )
    assert_true(
        abs(fmd.hfield_data[0]) <= TOL,
        "a constant grid should normalise to all zeros, got "
        + String(fmd.hfield_data[0]),
    )
    print("  values compared against MuJoCo:", n)
    assert_true(n == 12, "expected 12 comparisons, made " + String(n))


def test_a_wrong_length_refuses() raises:
    """MuJoCo: 'elevation data length must match nrow*ncol'."""
    print("=== eleven values for a 3x4 grid ===")
    var raised = False
    var msg = String("")
    try:
        _ = parse_xml_full(_xml(String("0 1 2 3 4 5 6 7 8 9 10")), String("."))
    except e:
        raised = True
        msg = String(e)
    print("  raised:", raised)
    assert_true(
        raised,
        "an 11-value grid for a 3x4 heightfield loaded. A short grid read as"
        " terrain is worse than no terrain: the tail is whatever the list"
        " parser returned.",
    )
    assert_true(
        msg.find("nrow*ncol") >= 0,
        "the error does not name the rule: " + msg,
    )


def test_no_elevation_and_no_file_is_still_flat() raises:
    """⚠ THE CASE THE OLD BEHAVIOUR GOT RIGHT, kept as a control.

    A fix that started raising here would refuse a model MuJoCo accepts:
    `mjXReader` fills zeros when neither a `file` nor an `elevation` is given
    (xml_native_reader.cc:2296-2300).
    """
    print("=== <hfield nrow ncol size/> with neither file nor elevation ===")
    var mujoco = Python.import_module("mujoco")
    var mm = mujoco.MjModel.from_xml_string(BARE_XML)
    var fmd = parse_xml_full(BARE_XML, String("."))
    assert_true(
        len(fmd.hfield_data) == 12,
        "expected a 12-cell flat grid, got " + String(len(fmd.hfield_data)),
    )
    var worst = Float64(0)
    for k in range(12):
        worst = max(worst, abs(fmd.hfield_data[k] - Float64(py=mm.hfield_data[k])))
    print("  worst |d| against MuJoCo:", worst, " (both flat)")
    assert_true(worst <= TOL, "a bare heightfield is not flat: " + String(worst))


def main() raises:
    var suite = TestSuite()
    suite.test[test_a_monotone_grid_matches_cell_for_cell]()
    suite.test[test_a_non_monotone_grid_normalises_the_same_way]()
    suite.test[test_a_constant_grid_does_not_divide_by_zero]()
    suite.test[test_a_wrong_length_refuses]()
    suite.test[test_no_elevation_and_no_file_is_still_flat]()
    suite^.run()
