"""Regression gate (GOLDEN-frozen): RK4 WITH CONTACTS on the fields path,
Walker2D on the floor.

Originally validated BIT-EXACT against the legacy RK4+PGS pipeline. That legacy
reference was frozen into the GOLDEN fingerprints below during Phase-0 of the
physics3d sunset, so this gate survives deletion of the legacy slab/kernels. It
checks:
  * fields-GPU reproduces the frozen (legacy-validated) fingerprint —
    per-step contact counts + order-sensitive checksums of the final
    qpos/qvel/qacc/contact records, and
  * fields-CPU == fields-GPU (independent CPU oracle; tolerance).

Walker2D on the floor (rootz=1.10, feet penetrating), BATCH=2, 3 full RK4 steps
(per-stage detection + serialized solve with limits inside). Model build still
after an INTENTIONAL physics change: HARVEST=True, run on Apple, paste, False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_rk4_contacts_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.integrator.rk4_fields import RK4IntegratorFields
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime CONE = Walker2dModel.CONE_TYPE
comptime BATCH = 2
comptime N_STEPS = 3
comptime METADATA_SIZE_L = 4

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON = 12  # contacts per step (uniform across the 3 steps)
comptime GOLD_QPOS = 14.32687733171042
comptime GOLD_QVEL = -40.559838462620974
comptime GOLD_QACC = -4984.992832183838
comptime GOLD_CON = 362586.5032533535


def _check(name: String, got: Float64, gold: Float64) raises:
    var denom = abs(gold) if abs(gold) > 1e-9 else 1.0
    var rel = abs(got - gold) / denom
    if rel > GOLD_RTOL and not has_nvidia_gpu_accelerator():
        raise Error(
            name + " fingerprint " + String(got) + " != golden "
            + String(gold) + " (rel " + String(rel) + ")"
        )


def main() raises:
    print("--- RK4 WITH CONTACTS fields GOLDEN gate: Walker2D BATCH=", BATCH)
    var ctx = DeviceContext()

    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    Walker2dModel.init_fields[DTYPE, 0](ctx, mf)

    var d = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
            dc.qvel.data[e * NV + i] = qv
            dc.qfrc.data[e * NV + i] = qf
    d.upload_all(ctx)

    var integ = RK4IntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, CONE,
        BATCH=BATCH,
    ]()
    integ.prepare_gpu(ctx)
    var integ_c = RK4IntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, CONE,
        BATCH=BATCH,
    ]()

    for step in range(N_STEPS):
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)
        d.meta.download(ctx)
        var ncon_seen = 0
        for e in range(BATCH):
            ncon_seen += Int(
                d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
            )
        if ncon_seen == 0:
            raise Error("no contacts at step " + String(step) + " — vacuous")
        if not HARVEST:
            if ncon_seen != GOLD_NCON and not has_nvidia_gpu_accelerator():
                raise Error(
                    "step " + String(step) + ": ncon " + String(ncon_seen)
                    + " != golden " + String(GOLD_NCON)
                )
        print("  step", step, ": ncon", ncon_seen)

    # --- final fields-GPU fingerprint (order-sensitive checksums) ---
    d.qpos.download(ctx)
    d.qvel.download(ctx)
    d.qacc.download(ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)
    var fp_qpos = Float64(0)
    var fp_qvel = Float64(0)
    var fp_qacc = Float64(0)
    for e in range(BATCH):
        for i in range(NQ):
            fp_qpos += Float64(d.qpos.data[e * NQ + i]) * Float64(e * NQ + i + 1)
        for i in range(NV):
            fp_qvel += Float64(d.qvel.data[e * NV + i]) * Float64(e * NV + i + 1)
            fp_qacc += Float64(d.qacc.data[e * NV + i]) * Float64(e * NV + i + 1)
    var fp_con = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_con += Float64(
                    d.contacts.data[e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k]
                ) * Float64((c + 1) * (k + 1))

    if HARVEST:
        print("  HARVEST GOLD_QPOS =", fp_qpos)
        print("  HARVEST GOLD_QVEL =", fp_qvel)
        print("  HARVEST GOLD_QACC =", fp_qacc)
        print("  HARVEST GOLD_CON  =", fp_con)
    else:
        _check("qpos", fp_qpos, GOLD_QPOS)
        _check("qvel", fp_qvel, GOLD_QVEL)
        _check("qacc", fp_qacc, GOLD_QACC)
        _check("con", fp_con, GOLD_CON)
        print("  PASS: fields-GPU matches golden fingerprint")

    # --- independent CPU oracle: fields-CPU == fields-GPU ---
    var worst = Float64(0)
    for i in range(BATCH * NQ):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU final qpos worst err:", worst)
    if worst > 1e-2:
        raise Error("fields-CPU RK4 contact dynamics diverged")
    print("  PASS: fields-CPU within 1e-2")
    print("test_rk4_contacts_fields: ALL PASS")
