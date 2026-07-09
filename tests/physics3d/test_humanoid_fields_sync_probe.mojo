"""NVIDIA localization probe for the humanoid GPU-training crash.

Runs the humanoid (FREE joint, NV=23, NGEOM=18 -> SAP, MAX_TENDON=2) FIELDS
kernels in the PRODUCTION config (RK4 + Newton, PARALLEL_GPU, treewalk, auto
broadphase = SAP), with a ctx.synchronize() + print after each stage, so a
CUDA illegal-address is attributed to the exact stage (the last "[n] ... ok"
NOT printed). No legacy kernels.

Run: MODULAR_DEBUG=device-sync-mode pixi run -e nvidia mojo run -I . \
        tests/physics3d/test_humanoid_fields_sync_probe.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
)
from mojo_rl.physics3d.collision.broadphase_sap_fields import (
    detect_contacts_auto_fields,
)
from mojo_rl.physics3d.integrator.rk4_fields import RK4IntegratorFields
from mojo_rl.physics3d.gpu.constants import (
    model_size_with_invweight,
    META_IDX_NUM_CONTACTS,
)
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel

comptime DTYPE = DType.float32  # match the gate (Metal is fragile on float64)
comptime NQ = HumanoidModel.NQ  # 24
comptime NV = HumanoidModel.NV  # 23 (free joint: NQ-NV = 1)
comptime NBODY = HumanoidModel.NBODY  # 14
comptime NJOINT = HumanoidModel.NJOINT  # 18
comptime NGEOM = HumanoidModel.NGEOM  # 18 (>= 16 -> SAP)
comptime NTEN = HumanoidModel.MAX_TENDON  # 2
comptime NEQ = HumanoidModel.MAX_EQUALITY  # 0
comptime NSITE = HumanoidModel.NSITE  # 0
comptime NEXCL = HumanoidModel.nexclude  # 0
comptime CONE = HumanoidModel.CONE_TYPE
comptime MC = HumanoidModel.MAX_CONTACTS  # 50
comptime BATCH = 2
comptime MS = model_size_with_invweight[
    NBODY, NJOINT, NV, NGEOM, NEQUALITY=NEQ, NTENDON=NTEN, NSITE=NSITE
]()


def main() raises:
    print("=== Humanoid FIELDS production-config sync probe ===")
    print(
        "NQ=", NQ, " NV=", NV, " NBODY=", NBODY, " NGEOM=", NGEOM,
        " (SAP:", NGEOM >= 16, ") NTENDON=", NTEN,
    )
    var ctx = DeviceContext()

    var mf = ModelFields[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL, 0
    ]()
    HumanoidModel.init_fields[DTYPE, 0](ctx, mf)

    var d = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    # LOW pose so the humanoid is in ground contact — exercises SAP + the
    # Newton contact solve (the path a falling humanoid hits during training).
    for e in range(BATCH):
        d.qpos.data[e * NQ + 2] = Scalar[DTYPE](0.2)  # torso z in contact
        d.qpos.data[e * NQ + 3] = Scalar[DTYPE](1.0)  # quat w
    d.upload_all(ctx)

    print("[1] forward_kinematics_fields PARALLEL=True (_mt) ...")
    forward_kinematics_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, BATCH, PARALLEL=True,
    ](d, mf, ctx)
    ctx.synchronize()
    print("    [1] FK _mt ok")

    print("[2] detect_contacts_auto_fields (SAP, NGEOM>=16) ...")
    detect_contacts_auto_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, BATCH,
    ](d, mf, ctx)
    ctx.synchronize()
    d.meta.download(ctx)
    ctx.synchronize()
    print("    [2] detection (SAP) ok — ncon env0 =",
          d.meta.data[META_IDX_NUM_CONTACTS])

    print("[3] RK4+Newton+treewalk full step (production config) ...")
    var integ = RK4IntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE, NEXCL, 0,
        CONE, BATCH, SOLVER="newton", PARALLEL_GPU=True, CRBA_TREEWALK=True,
    ]()
    integ.prepare_gpu(ctx)
    for s in range(5):
        integ.step["gpu"](d, mf, ctx)
        ctx.synchronize()
        print("    [3] production step", s, "ok")

    print("=== ALL HUMANOID FIELDS KERNELS OK (with ground contact) ===")
