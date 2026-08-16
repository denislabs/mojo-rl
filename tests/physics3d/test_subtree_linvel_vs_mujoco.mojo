"""`subtree_linvel` vs MuJoCo's `mj_subtreeVel`.

Validates the sensor (`physics3d/sensors/subtree.mojo`) against MuJoCo's own
`data.subtree_linvel`, on Walker2d — a model that already has FK parity and
whose body tree BRANCHES (two legs), so the subtree walk has to handle
siblings rather than a single chain. Every body is checked, over several
states.

MuJoCo only fills `data.subtree_linvel` when something asks for it, so the
test calls `mj_subtreeVel` explicitly rather than relying on a sensor being
declared in the XML.

This runs end to end through our own `compute_body_velocities`. It was
temporarily driven from MuJoCo's velocities while that kernel was missing the
hinge angular->linear coupling term; that bug is fixed and separately gated by
test_body_velocities_vs_mujoco.mojo.

Deliberately NOT generic over `ModelDefLike`: monomorphizing the whole
init_fields + FK + velocity chain more than once in a single function pushes
compile time into the minutes, for no extra coverage.

First entry in the sensor package; see gap G1 in docs/DM_CONTROL_PORT.md.

Run with:
    pixi run mojo run -I . tests/physics3d/test_subtree_linvel_vs_mujoco.mojo
"""

from std.math import abs, sin, cos
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.sensors.subtree import subtree_linvel

from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel


comptime M = Walker2dModel
# Inherits the engine's own xvel agreement with MuJoCo (~1.2e-10 on
# walker2d, see test_body_velocities_vs_mujoco), so this bound tracks that
# one rather than the sensor's arithmetic — which is exact to 4.4e-16 when
# fed identical velocities.
comptime TOL: Float64 = 1e-9
comptime N_STATES: Int = 6


def test_subtree_linvel_matches_mujoco() raises:
    var mj = Python.import_module("mujoco")
    var model = mj.MjModel.from_xml_path("mojo_rl/envs/walker2d/assets/walker2d.xml")
    var data = mj.MjData(model)

    var ctx = DeviceContext()
    var mf = Model[DType.float64, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0]]()
    M.init_fields[DType.float64, 0](ctx, mf)
    var d = Data[
        DType.float64, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1
    ]()

    var worst = Float64(0)

    for s in range(N_STATES):
        # Deterministic, non-trivial state; both engines get identical values.
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
        mj.mj_subtreeVel(model, data)

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
            var vx = Float64(0)
            var vy = Float64(0)
            var vz = Float64(0)
            subtree_linvel(
                d.xvel.data, mf.bodies.data, M.NBODY, b, vx, vy, vz
            )
            var dx = abs(Float64(py=data.subtree_linvel[b][0]) - vx)
            var dy = abs(Float64(py=data.subtree_linvel[b][1]) - vy)
            var dz = abs(Float64(py=data.subtree_linvel[b][2]) - vz)
            if dx > worst:
                worst = dx
            if dy > worst:
                worst = dy
            if dz > worst:
                worst = dz
            if dx > TOL or dy > TOL or dz > TOL:
                print(
                    "  MISMATCH state", s, " body", b,
                    " ours=(", vx, ",", vy, ",", vz, ")",
                    " mj=(", Float64(py=data.subtree_linvel[b][0]), ",",
                    Float64(py=data.subtree_linvel[b][1]), ",",
                    Float64(py=data.subtree_linvel[b][2]), ")",
                )

    print(
        "subtree_linvel vs MuJoCo (walker2d,", N_STATES,
        "states x", M.NBODY, "bodies): max |diff| =", worst,
        " (bound ", TOL, ")",
    )
    assert_true(worst <= TOL, "subtree_linvel deviated from MuJoCo")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
