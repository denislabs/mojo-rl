"""`qpos0` is MuJoCo's `qpos0` — `<custom><numeric name="init_qpos">` is NOT applied.

WHY THIS EXISTS. `_fill_qpos0` used to end by copying a `<custom><numeric
name="init_qpos">` over the pose it had just built from joint `ref`s and
free-joint body poses, mirroring the legacy parser, which mirrored
mujoco-py's `MjSim`. `mj_resetData` never reads that numeric and neither
does Gymnasium on the current bindings (`init_qpos = data.qpos`). Gymnasium's
ant is the one model in the tree that ships one (z 0.55, ankles ±1 rad), so
our ant and MuJoCo's started from two different poses: 2.9e-01 on the
three-tree board at 50 steps (PERFORMANCE.md §13.30), 4e-16 once the
override was dropped. MuJoCo's ant starts at z 0.75 with every ankle at 0,
OUTSIDE its `range="30 70"`, and its first step is a limit shove — that IS
the reference, and the Ant env now resets from it like Gymnasium does.

WHAT IT GATES. `spec_fields_runtime(...).qpos0` against `MjModel.qpos0`,
elementwise, on the three Gymnasium/dm_control assets whose pose has a free
root and joint `ref`s or a stale numeric: ant (the numeric), humanoid
(a free root at z 1.4) and dm_control's quadruped (a `<body quat>` root).
Plus an inline fixture whose numeric disagrees with everything, so the gate
cannot pass by the numeric happening to equal the pose.

Run: pixi run mojo run -I . tests/physics3d/test_qpos0_vs_mujoco.mojo
"""
from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime, read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf

comptime DT = DType.float64
comptime TOL: Float64 = 1e-12

comptime NUMERIC = """
<mujoco model="numeric_is_not_qpos0">
  <custom><numeric name="init_qpos" data="9 9 9 9 9 9 9 9 9"/></custom>
  <worldbody>
    <body pos="0.1 0.2 0.3" quat="0 0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.05"/>
      <body pos="0.1 0 0">
        <joint name="h1" type="hinge" axis="0 1 0" ref="0.25"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <body pos="0.2 0 0">
          <joint name="s1" type="slide" axis="1 0 0" ref="-0.05"/>
          <geom type="sphere" size="0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _ours_from_text(xml: String, base: String) raises -> List[Float64]:
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var out = List[Float64]()
    for i in range(dims.get_nq()):
        out.append(Float64(sf.qpos0.data[i]))
    return out^


def _ours_from_path(path: String) raises -> List[Float64]:
    var src = read_model_source(path)
    return _ours_from_text(src[0], src[1])


def _mj(m: PythonObject) raises -> List[Float64]:
    var q = m.qpos0.flatten().tolist()
    var out = List[Float64]()
    for i in range(Int(py=m.nq)):
        out.append(Float64(py=q[i]))
    return out^


def _check(name: String, ours: List[Float64], want: List[Float64]) raises:
    assert_true(
        len(ours) == len(want),
        name + ": nq " + String(len(ours)) + " vs MuJoCo " + String(len(want)),
    )
    var worst = 0.0
    var at = -1
    for i in range(len(want)):
        var e = abs(ours[i] - want[i])
        if e > worst:
            worst = e
            at = i
    print("  ", name, ": nq", len(want), " |d(qpos0)|max =", worst, "at", at)
    assert_true(
        worst <= TOL,
        name + ": qpos0 differs from MuJoCo's by " + String(worst)
        + " at index " + String(at) + " — a <custom> numeric or a body pose"
        " is leaking into the reference pose",
    )


def test_gym_and_dm_assets() raises:
    print("=== qpos0 vs MuJoCo: Gymnasium ant / humanoid, dm_control quadruped ===")
    var mujoco = Python.import_module("mujoco")
    var paths = List[String]()
    paths.append(String("references/Gymnasium-main/gymnasium/envs/mujoco/assets/ant.xml"))
    paths.append(String("references/Gymnasium-main/gymnasium/envs/mujoco/assets/humanoid.xml"))
    paths.append(String("references/dm_control-main/dm_control/suite/quadruped.xml"))
    for i in range(len(paths)):
        var m = mujoco.MjModel.from_xml_path(paths[i])
        _check(paths[i], _ours_from_path(paths[i]), _mj(m))
    print("  PASS")


def test_numeric_is_ignored() raises:
    print("=== qpos0 vs MuJoCo: a <custom> init_qpos that disagrees with the pose ===")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[NUMERIC]())
    var want = _mj(m)
    # The numeric says 9 everywhere; MuJoCo says body pose + refs. If the
    # gate ever sees a 9, the override is back.
    var ours = _ours_from_text(materialize[NUMERIC](), String("."))
    for i in range(len(ours)):
        assert_true(abs(ours[i] - 9.0) > 1e-9, "qpos0 carries the numeric's 9 at index " + String(i))
    _check(String("numeric fixture"), ours, want)
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
