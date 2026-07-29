"""`compute_body_velocities` (xvel / xangvel) vs MuJoCo.

`Data.xvel[b]` is the world-frame linear velocity of body b's CoM and
`Data.xangvel[b]` its world-frame angular velocity. MuJoCo exposes exactly
those via `mj_objectVelocity(..., mjOBJ_BODY, b, res, 0)`, which returns
`[angular(3); linear(3)]` in world coordinates about the body's INERTIAL
frame (i.e. the CoM) — the same reference point our kernel uses.

There was no gate on this path before 2026-07-29, which is how a real bug
survived: `_vel_body` propagated `v = v_parent + w_parent x r` and then, for a
HINGE, added the joint's contribution to the ANGULAR velocity only, never the
matching linear coupling `w_joint x (body_CoM - joint_anchor)`. The SLIDE
branch did add its linear term, so only bodies below a hinge or ball were
wrong — by ~7% on walker2d. Nothing caught it because the main dynamics read
`cvel`/`cdof`, not `xvel`; the sole consumer is `dynamics/fluid_forces.mojo`
(swimmer + fish drag), whose own test only checks CPU==GPU self-consistency.

Covers a planar hinge/slide chain (walker2d, branching), a simple chain
(hopper), and a FREE-joint root (ant) so the free branch is exercised too.

Run with:
    pixi run mojo run -I . tests/physics3d/test_body_velocities_vs_mujoco.mojo
"""

from std.math import abs, sin, cos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)

from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel, walker2d_xml
from mojo_rl.envs.hopper.hopper_xml import HopperModel, hopper_xml
from mojo_rl.envs.ant.ant_xml import AntModel, ant_xml


comptime TOL: Float64 = 1e-9
comptime N_STATES: Int = 5


def _report(label: String, worst_v: Float64, worst_w: Float64) raises:
    print(
        "  ", label, ": max |d(xvel)| =", worst_v,
        "  max |d(xangvel)| =", worst_w, " (bound ", TOL, ")",
    )
    assert_true(worst_v <= TOL, "xvel deviated from MuJoCo: " + label)
    assert_true(worst_w <= TOL, "xangvel deviated from MuJoCo: " + label)


def test_walker2d_body_velocities() raises:
    comptime M = Walker2dModel
    var mj = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var model = mj.MjModel.from_xml_string(String(walker2d_xml))
    var data = mj.MjData(model)
    var ctx = DeviceContext()
    var mf = Model[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DType.float64, 0](ctx, mf)
    var d = Data[
        DType.float64, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1
    ]()
    var wv = Float64(0)
    var ww = Float64(0)
    var res = np.zeros(6)
    for s in range(N_STATES):
        mj.mj_resetData(model, data)
        for i in range(M.NQ):
            var q = 0.35 * sin(1.7 * Float64(i) + 0.9 * Float64(s))
            d.qpos.data[i] = q
            data.qpos[i] = q
        for i in range(M.NV):
            var v = 0.8 * cos(2.1 * Float64(i) - 0.6 * Float64(s))
            d.qvel.data[i] = v
            data.qvel[i] = v
        mj.mj_forward(model, data)
        forward_kinematics[
            "cpu", DType.float64, M.NQ, M.NV, M.NBODY, M.NJOINT,
            M.MAX_CONTACTS, M.NGEOM, M.MAX_EQUALITY, M.MAX_TENDON,
            M.NSITE, M.NEXCLUDE, 0, 1,
        ](d, mf, None)
        compute_body_velocities[
            "cpu", DType.float64, M.NQ, M.NV, M.NBODY, M.NJOINT,
            M.MAX_CONTACTS, M.NGEOM, M.MAX_EQUALITY, M.MAX_TENDON,
            M.NSITE, M.NEXCLUDE, 0, 1,
        ](d, mf, None)
        for b in range(M.NBODY):
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, b, res, 0)
            for k in range(3):
                var dv = abs(
                    Float64(py=res[3 + k]) - Float64(d.xvel.data[b * 3 + k])
                )
                var dw = abs(
                    Float64(py=res[k]) - Float64(d.xangvel.data[b * 3 + k])
                )
                if dv > wv:
                    wv = dv
                if dw > ww:
                    ww = dw
    _report("walker2d", wv, ww)


def test_hopper_body_velocities() raises:
    comptime M = HopperModel
    var mj = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var model = mj.MjModel.from_xml_string(String(hopper_xml))
    var data = mj.MjData(model)
    var ctx = DeviceContext()
    var mf = Model[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DType.float64, 0](ctx, mf)
    var d = Data[
        DType.float64, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1
    ]()
    var wv = Float64(0)
    var ww = Float64(0)
    var res = np.zeros(6)
    for s in range(N_STATES):
        mj.mj_resetData(model, data)
        for i in range(M.NQ):
            var q = 0.3 * sin(1.3 * Float64(i) + 0.7 * Float64(s))
            d.qpos.data[i] = q
            data.qpos[i] = q
        for i in range(M.NV):
            var v = 0.9 * cos(1.9 * Float64(i) - 0.5 * Float64(s))
            d.qvel.data[i] = v
            data.qvel[i] = v
        mj.mj_forward(model, data)
        forward_kinematics[
            "cpu", DType.float64, M.NQ, M.NV, M.NBODY, M.NJOINT,
            M.MAX_CONTACTS, M.NGEOM, M.MAX_EQUALITY, M.MAX_TENDON,
            M.NSITE, M.NEXCLUDE, 0, 1,
        ](d, mf, None)
        compute_body_velocities[
            "cpu", DType.float64, M.NQ, M.NV, M.NBODY, M.NJOINT,
            M.MAX_CONTACTS, M.NGEOM, M.MAX_EQUALITY, M.MAX_TENDON,
            M.NSITE, M.NEXCLUDE, 0, 1,
        ](d, mf, None)
        for b in range(M.NBODY):
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, b, res, 0)
            for k in range(3):
                var dv = abs(
                    Float64(py=res[3 + k]) - Float64(d.xvel.data[b * 3 + k])
                )
                var dw = abs(
                    Float64(py=res[k]) - Float64(d.xangvel.data[b * 3 + k])
                )
                if dv > wv:
                    wv = dv
                if dw > ww:
                    ww = dw
    _report("hopper", wv, ww)


def test_ant_body_velocities() raises:
    """Ant has a FREE-joint root, so this covers the free branch (which
    overwrites v/w outright) as well as the hinge chain below it."""
    comptime M = AntModel
    var mj = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var model = mj.MjModel.from_xml_string(String(ant_xml))
    var data = mj.MjData(model)
    var ctx = DeviceContext()
    var mf = Model[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DType.float64, 0](ctx, mf)
    var d = Data[
        DType.float64, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1
    ]()
    var wv = Float64(0)
    var ww = Float64(0)
    var res = np.zeros(6)
    for s in range(N_STATES):
        mj.mj_resetData(model, data)
        # Free-joint root: qpos[0:3] position, qpos[3:7] quaternion.
        for i in range(M.NQ):
            d.qpos.data[i] = 0.0
            data.qpos[i] = 0.0
        d.qpos.data[2] = 0.75
        data.qpos[2] = 0.75
        var ang = 0.3 * Float64(s)
        d.qpos.data[3] = cos(ang)
        data.qpos[3] = cos(ang)
        d.qpos.data[4] = 0.0
        data.qpos[4] = 0.0
        d.qpos.data[5] = sin(ang)
        data.qpos[5] = sin(ang)
        d.qpos.data[6] = 0.0
        data.qpos[6] = 0.0
        for i in range(7, M.NQ):
            var q = 0.25 * sin(1.1 * Float64(i) + 0.8 * Float64(s))
            d.qpos.data[i] = q
            data.qpos[i] = q
        for i in range(M.NV):
            var v = 0.6 * cos(1.7 * Float64(i) - 0.4 * Float64(s))
            d.qvel.data[i] = v
            data.qvel[i] = v
        mj.mj_forward(model, data)
        forward_kinematics[
            "cpu", DType.float64, M.NQ, M.NV, M.NBODY, M.NJOINT,
            M.MAX_CONTACTS, M.NGEOM, M.MAX_EQUALITY, M.MAX_TENDON,
            M.NSITE, M.NEXCLUDE, 0, 1,
        ](d, mf, None)
        compute_body_velocities[
            "cpu", DType.float64, M.NQ, M.NV, M.NBODY, M.NJOINT,
            M.MAX_CONTACTS, M.NGEOM, M.MAX_EQUALITY, M.MAX_TENDON,
            M.NSITE, M.NEXCLUDE, 0, 1,
        ](d, mf, None)
        for b in range(M.NBODY):
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, b, res, 0)
            for k in range(3):
                var dv = abs(
                    Float64(py=res[3 + k]) - Float64(d.xvel.data[b * 3 + k])
                )
                var dw = abs(
                    Float64(py=res[k]) - Float64(d.xangvel.data[b * 3 + k])
                )
                if dv > wv:
                    wv = dv
                if dw > ww:
                    ww = dw
    _report("ant (free root)", wv, ww)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
