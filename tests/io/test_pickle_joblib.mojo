"""`mojo_rl/io/pickle.mojo` against a joblib fixture and the LAFAN pickle.

    pixi run mojo run -I . tests/io/test_pickle_joblib.mojo

The fixture (`tests/fixtures/joblib_arrays.pkl`, `tools/io/make_joblib_fixture.py`)
holds every dtype and rank the reader claims — f4 rank 2 and 3, f8, i8, i4,
a 2000 x 10 array — plus an int, a bool, a str, a float and None, in nested
dicts. The values pinned below are the ramp the fixture script prints. The
second test reads the reference's `lafan_29dof.pkl` when it is present
(gitignored, 200 MB) and pins what joblib reports for it: 40 clips, the
first and last keys, shapes, fps, three values and the float64 sum of one
clip's `pose_aa` (6945.71934827885).
"""

from std.math import abs
from std.os.path import exists
from std.testing import assert_true, TestSuite

from mojo_rl.io.pickle import JoblibPickle, PV_DICT, PV_ARRAY, PV_INT, PV_BOOL, PV_STR, PV_FLOAT, PV_NONE


comptime FIXTURE = "tests/fixtures/joblib_arrays.pkl"
comptime LAFAN = "references/BFM-Zero-main/humanoidverse/data/lafan_29dof.pkl"


def _sum32(xs: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(xs)):
        s += Float64(xs[i])
    return s


def test_fixture_values() raises:
    var p = JoblibPickle.load(String(FIXTURE))
    assert_true(p.kind(p.root) == PV_DICT, "root is a dict")
    var keys = p.dict_keys(p.root)
    assert_true(len(keys) == 2 and keys[0] == "clipA" and keys[1] == "clipB", "top keys")
    var a = p.get(p.root, String("clipA"))
    var f4 = p.get(a, String("f4_2d"))
    var sh = p.shape_of(f4)
    assert_true(len(sh) == 2 and sh[0] == 5 and sh[1] == 3, "f4_2d shape")
    var v = p.array_f32(f4)
    assert_true(abs(Float64(v[0]) - 1.3783783912658691) < 1e-7 and abs(Float64(v[2]) - 0.4054054021835327) < 1e-7, "f4_2d[0]")
    assert_true(abs(_sum32(v) - (-0.3753753751516342)) < 1e-6, "f4_2d sum")
    var f43 = p.get(a, String("f4_3d"))
    var sh3 = p.shape_of(f43)
    assert_true(len(sh3) == 3 and sh3[0] == 2 and sh3[1] == 4 and sh3[2] == 3, "f4_3d shape")
    var v3 = p.array_f32(f43)
    assert_true(abs(Float64(v3[1 * 12 + 3 * 3 + 2]) - 0.5735735893249512) < 1e-7, "f4_3d[1,3,2]")
    assert_true(abs(_sum32(v3) - (-2.1141141206026077)) < 1e-6, "f4_3d sum")
    var f8 = p.array_f64(p.get(a, String("f8_2d")))
    assert_true(abs(f8[0] - (-1.8768768768768769)) < 1e-15 and abs(f8[3] - 2.66966966966967) < 1e-15, "f8_2d")
    var i8 = p.array_i64(p.get(a, String("i8_1d")))
    assert_true(len(i8) == 3 and i8[0] == 3 and i8[1] == -7 and i8[2] == 11, "i8_1d")
    var i4 = p.array_i32(p.get(a, String("i4_1d")))
    assert_true(len(i4) == 4 and i4[3] == 4, "i4_1d")
    var big = p.get(a, String("big"))
    var bsh = p.shape_of(big)
    assert_true(bsh[0] == 2000 and bsh[1] == 10, "big shape")
    var bv = p.array_f32(big)
    assert_true(abs(Float64(bv[1999 * 10 + 9]) - 2.987987995147705) < 1e-7, "big[1999,9]")
    assert_true(abs(_sum32(bv)) < 1e-3, "big sum")
    assert_true(p.int_of(p.get(a, String("fps"))) == 30, "fps")
    assert_true(p.kind(p.get(a, String("flag"))) == PV_BOOL and p.int_of(p.get(a, String("flag"))) == 1, "flag")
    assert_true(p.kind(p.get(a, String("name"))) == PV_STR and p.nodes[p.get(a, String("name"))].s == "clip-A", "name")
    assert_true(p.kind(p.get(a, String("scale"))) == PV_FLOAT and abs(p.nodes[p.get(a, String("scale"))].f - 0.25) < 1e-15, "scale")
    assert_true(p.kind(p.get(a, String("nothing"))) == PV_NONE, "none")
    var b = p.get(p.root, String("clipB"))
    assert_true(abs(_sum32(p.array_f32(p.get(b, String("f4_2d")))) - (-2.054054021835327)) < 1e-6, "clipB sum")
    assert_true(p.int_of(p.get(b, String("fps"))) == 60, "clipB fps")
    print("  fixture: 2 clips, every dtype/rank, scalars and None read back")


def test_lafan_pickle() raises:
    if not exists(String(LAFAN)):
        print("  SKIP: " + String(LAFAN) + " not present")
        return
    var p = JoblibPickle.load(String(LAFAN))
    var keys = p.dict_keys(p.root)
    assert_true(len(keys) == 40, "40 clips")
    assert_true(keys[0] == "fallAndGetUp1_subject4" and keys[39] == "fightAndSports1_subject4", "clip names")
    var c = p.get(p.root, String(keys[0]))
    var pa = p.get(c, String("pose_aa"))
    var sh = p.shape_of(pa)
    assert_true(sh[0] == 5047 and sh[1] == 30 and sh[2] == 3, "pose_aa shape")
    var rt = p.array_f32(p.get(c, String("root_trans_offset")))
    assert_true(abs(Float64(rt[2]) - 0.7860220074653625) < 1e-7, "root_trans[0].z")
    var pv = p.array_f32(pa)
    assert_true(abs(Float64(pv[1]) - (-0.023538783192634583)) < 1e-7, "pose_aa[0,0,1]")
    assert_true(abs(_sum32(pv) - 6945.71934827885) < 1e-4, "pose_aa float64 sum")
    var dof = p.array_f32(p.get(c, String("dof")))
    assert_true(abs(Float64(dof[2]) - 0.13536499440670013) < 1e-7, "dof[0,2]")
    assert_true(p.int_of(p.get(c, String("fps"))) == 30, "fps")
    print("  lafan_29dof.pkl: 40 clips, shapes, fps and pinned values as joblib reports them")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
