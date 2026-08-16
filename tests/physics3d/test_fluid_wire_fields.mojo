"""Stage-A wire smoke: the fields integrators now STEP Swimmer (fluid active,
density=4000/viscosity=0.1) instead of raising "fluid not ported".

Runs a few contact-free passive steps of Euler / Implicit / RK4 on Swimmer and
checks the state stays finite (fluid drag is dissipative, so velocities must
not blow up). Validates all three fluid-call wirings compile + run.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_fluid_wire_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.integrator.implicit import (
    ImplicitIntegrator,
)
from mojo_rl.physics3d.integrator.rk4 import RK4Integrator
from mojo_rl.envs.swimmer.swimmer_xml import SwimmerModel

comptime DT = DType.float32
comptime NQ = SwimmerModel.NQ
comptime NV = SwimmerModel.NV
comptime NBODY = SwimmerModel.NBODY
comptime NJOINT = SwimmerModel.NJOINT
comptime NGEOM = SwimmerModel.NGEOM
comptime MC = SwimmerModel.MAX_CONTACTS
comptime NEQ = SwimmerModel.MAX_EQUALITY
comptime NTD = SwimmerModel.MAX_TENDON
comptime NSITE = SwimmerModel.NSITE
comptime NEXCL = SwimmerModel.NEXCLUDE
comptime CONE = SwimmerModel.CONE_TYPE
comptime BATCH = 1
comptime N_STEPS = 3


def _load_model(ctx: DeviceContext) raises -> Model[DT, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTD, nsite=NSITE, nexclude=NEXCL, nmesh_verts=0]]:
    var mf = Model[DT, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTD, nsite=NSITE, nexclude=NEXCL, nmesh_verts=0]]()
    SwimmerModel.init_fields[DT, 0](ctx, mf)
    return mf^


def _fresh_data() raises -> Data[DT, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MC, nsite=NSITE], BATCH]:
    var d = Data[DT, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MC, nsite=NSITE], BATCH]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DT]((i * 3) % 5 - 2) / 20.0
    for i in range(NV):
        d.qvel.data[i] = Scalar[DT]((i * 7 + 3) % 11 - 5) * Scalar[DT](0.2)
    return d^


def _check_finite(
    mut d: Data[DT, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MC, nsite=NSITE], BATCH], ctx: DeviceContext, name: String
) raises:
    d.qpos.download(ctx)
    d.qvel.download(ctx)
    for i in range(NQ):
        var v = Float64(d.qpos.data[i])
        if v != v or abs(v) > 1e6:
            raise Error(name + ": non-finite qpos")
    for i in range(NV):
        var v = Float64(d.qvel.data[i])
        if v != v or abs(v) > 1e6:
            raise Error(name + ": non-finite qvel")


def main() raises:
    print("=== Stage-A fluid wire smoke: Swimmer through fields integrators ===")
    var ctx = DeviceContext()
    var mf = _load_model(ctx)

    # Euler
    var dE = _fresh_data()
    dE.upload_all(ctx)
    var integE = EulerIntegrator[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, CONE, BATCH,
    ]()
    integE.prepare_gpu(ctx)
    for _s in range(N_STEPS):
        integE.step["gpu", False](dE, mf, ctx)
    _check_finite(dE, ctx, "Euler")
    print("  Euler[Swimmer] fluid step PASS")

    # Implicit
    var dI = _fresh_data()
    dI.upload_all(ctx)
    var integI = ImplicitIntegrator[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, CONE, BATCH,
    ]()
    integI.prepare_gpu(ctx)
    for _s in range(N_STEPS):
        integI.step["gpu", False](dI, mf, ctx)
    _check_finite(dI, ctx, "Implicit")
    print("  Implicit[Swimmer] fluid step PASS")

    # RK4 (Swimmer's native integrator)
    var dR = _fresh_data()
    dR.upload_all(ctx)
    var integR = RK4Integrator[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, CONE, BATCH,
    ]()
    integR.prepare_gpu(ctx)
    for _s in range(N_STEPS):
        integR.step["gpu", False](dR, mf, ctx)
    _check_finite(dR, ctx, "RK4")
    print("  RK4[Swimmer] fluid step PASS")

    print("test_fluid_wire_fields: ALL PASS")
