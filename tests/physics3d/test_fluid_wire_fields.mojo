"""Stage-A wire smoke: the fields integrators now STEP Swimmer (fluid active,
density=4000/viscosity=0.1) instead of raising "fluid not ported".

Runs a few contact-free passive steps of Euler / Implicit / RK4 on Swimmer and
checks the state stays finite (fluid drag is dissipative, so velocities must
not blow up). Validates all three fluid-call wirings compile + run.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_fluid_wire_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler_fields import EulerIntegratorFields
from mojo_rl.physics3d.integrator.implicit_fields import (
    ImplicitIntegratorFields,
)
from mojo_rl.physics3d.integrator.rk4_fields import RK4IntegratorFields
from mojo_rl.physics3d.gpu.constants import model_size_with_invweight
from mojo_rl.envs.swimmer.swimmer_xml import SwimmerModel

comptime DT = DType.float32
comptime NQ = SwimmerModel.NQ
comptime NV = SwimmerModel.NV
comptime NBODY = SwimmerModel.NBODY
comptime NJOINT = SwimmerModel.NJOINT
comptime NGEOM = SwimmerModel.NGEOM
comptime MC = SwimmerModel.MAX_CONTACTS
comptime CONE = SwimmerModel.CONE_TYPE
comptime BATCH = 1
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime N_STEPS = 3


def _load_model(ctx: DeviceContext) raises -> ModelFields[
    DT, NV, NBODY, NJOINT, NGEOM, 0, 0, 0, 0, 0
]:
    var model_t = TensorImpl[DT].alloc(MS)
    model_t.upload(ctx)
    SwimmerModel.init_model_gpu(ctx, model_t.dev.value())
    model_t.download(ctx)
    var mf = ModelFields[DT, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)
    return mf^


def _fresh_data() raises -> DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH]:
    var d = DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DT]((i * 3) % 5 - 2) / 20.0
    for i in range(NV):
        d.qvel.data[i] = Scalar[DT]((i * 7 + 3) % 11 - 5) * Scalar[DT](0.2)
    return d^


def _check_finite(
    mut d: DataFields[DT, NQ, NV, NBODY, MC, 0, BATCH], ctx: DeviceContext, name: String
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
    var integE = EulerIntegratorFields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
    ]()
    integE.prepare_gpu(ctx)
    for _s in range(N_STEPS):
        integE.step["gpu", False](dE, mf, ctx)
    _check_finite(dE, ctx, "Euler")
    print("  Euler[Swimmer] fluid step PASS")

    # Implicit
    var dI = _fresh_data()
    dI.upload_all(ctx)
    var integI = ImplicitIntegratorFields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
    ]()
    integI.prepare_gpu(ctx)
    for _s in range(N_STEPS):
        integI.step["gpu", False](dI, mf, ctx)
    _check_finite(dI, ctx, "Implicit")
    print("  Implicit[Swimmer] fluid step PASS")

    # RK4 (Swimmer's native integrator)
    var dR = _fresh_data()
    dR.upload_all(ctx)
    var integR = RK4IntegratorFields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
    ]()
    integR.prepare_gpu(ctx)
    for _s in range(N_STEPS):
        integR.step["gpu", False](dR, mf, ctx)
    _check_finite(dR, ctx, "RK4")
    print("  RK4[Swimmer] fluid step PASS")

    print("test_fluid_wire_fields: ALL PASS")
