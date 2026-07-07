"""P1 gates for the per-field tensor containers (fields/DataFields,
fields/ModelFields — docs/PHYSICS3D_TENSOR_MIGRATION_SCOPE.md).

1. DataFields slab round-trip: load_from_slab -> store_to_slab is
   bit-identical over every region/offset (host only).
2. DataFields device round-trip: upload_all -> scramble host ->
   download_all restores exactly.
3. ModelFields slab round-trip: all 13 record families bit-identical.
4. Zero-width params compile/run (NSITE=0, NGEOM=0, ... variant).
5. cvel through DataFields on GPU: per-field kernel consuming
   DataFields views is BIT-EXACT vs the legacy flat-slab compute_cvel_gpu
   on identical inputs (also the first dedicated cvel test).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_tensor_fields.mojo
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.gpu.cvel_gpu import compute_cvel_gpu
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    cvel_offset,
)

comptime DTYPE = DType.float32
comptime NQ = 9
comptime NV = 9
comptime NBODY = 8
comptime NJOINT = 6
comptime MAX_CONTACTS = 8
comptime NSITE = 2
comptime NGEOM = 5
comptime NEQUALITY = 1
comptime NTENDON = 1
comptime NEXCLUDE = 2
comptime NMESH_VERTS = 4
comptime BATCH = 4
comptime SS = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MS = model_size_with_invweight[
    NBODY, NJOINT, NV, NGEOM, NEQUALITY, NTENDON, NSITE, NEXCLUDE, NMESH_VERTS
]()
comptime TPB = 64

comptime DF = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH]
comptime MF = ModelFields[
    DTYPE, NV, NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE, NEXCLUDE,
    NMESH_VERTS,
]


def _pseudo(e: Int, off: Int) -> Scalar[DTYPE]:
    var h = (e * 131071 + off * 524287) % 1999
    return Scalar[DTYPE](h - 999) / 333.0


def _fill_slab(mut slab: List[Scalar[DTYPE]], per_env: Int, envs: Int):
    for e in range(envs):
        for off in range(per_env):
            slab[e * per_env + off] = _pseudo(e, off)


def _compare(
    a: List[Scalar[DTYPE]], b: List[Scalar[DTYPE]], n: Int, label: String
) raises:
    var bad = 0
    for i in range(n):
        if a[i] != b[i]:
            if bad < 5:
                print("  MISMATCH", label, "i=", i, a[i], "vs", b[i])
            bad += 1
    if bad != 0:
        raise Error(label + ": mismatches")
    print("  PASS:", label, "(", n, "values bit-identical )")


# The ported per-field cvel kernel (arithmetic verbatim from cvel_gpu.mojo).
def _cvel_fields_kernel(
    xpos: LayoutTensor[DTYPE, DF.L_B3, MutAnyOrigin],
    xvel: LayoutTensor[DTYPE, DF.L_B3, MutAnyOrigin],
    xangvel: LayoutTensor[DTYPE, DF.L_B3, MutAnyOrigin],
    xipos: LayoutTensor[DTYPE, DF.L_B3, MutAnyOrigin],
    cvel: LayoutTensor[DTYPE, DF.L_B6, MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    for b in range(NBODY):
        var ox = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 0])
        var oy = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 1])
        var oz = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 2])
        var vx = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 0])
        var vy = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 1])
        var vz = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 2])
        var px = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 0])
        var py = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 1])
        var pz = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2])
        var cx = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 0])
        var cy = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 1])
        var cz = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 2])
        var dx = cx - px
        var dy = cy - py
        var dz = cz - pz
        var vcx = vx + (oy * dz - oz * dy)
        var vcy = vy + (oz * dx - ox * dz)
        var vcz = vz + (ox * dy - oy * dx)
        var base = b * 6
        cvel[env, base + 0] = ox
        cvel[env, base + 1] = oy
        cvel[env, base + 2] = oz
        cvel[env, base + 3] = vcx
        cvel[env, base + 4] = vcy
        cvel[env, base + 5] = vcz


def main() raises:
    print("test_tensor_fields: SS=", SS, " MS=", MS, " BATCH=", BATCH)

    # ── 1. DataFields slab round-trip (host) ─────────────────────────────
    var slab = List[Scalar[DTYPE]](
        length=BATCH * SS, fill=Scalar[DTYPE](0)
    )
    _fill_slab(slab, SS, BATCH)
    var d = DF()
    d.load_from_slab(slab)
    var slab2 = List[Scalar[DTYPE]](length=BATCH * SS, fill=Scalar[DTYPE](0))
    d.store_to_slab(slab2)
    _compare(slab, slab2, BATCH * SS, "DataFields slab round-trip")

    # ── 2. DataFields device round-trip ──────────────────────────────────
    var ctx = DeviceContext()
    d.upload_all(ctx)
    # Scramble host copies, then restore from device.
    for i in range(BATCH * NQ):
        d.qpos.data[i] = 12345.0
    for i in range(BATCH * NBODY * 3):
        d.xpos.data[i] = 12345.0
    for i in range(BATCH * MAX_CONTACTS * 23):
        d.contacts.data[i] = 12345.0
    d.download_all(ctx)
    var slab3 = List[Scalar[DTYPE]](length=BATCH * SS, fill=Scalar[DTYPE](0))
    d.store_to_slab(slab3)
    _compare(slab, slab3, BATCH * SS, "DataFields device round-trip")

    # ── 3. ModelFields slab round-trip (host) ────────────────────────────
    var flat = List[Scalar[DTYPE]](length=MS, fill=Scalar[DTYPE](0))
    _fill_slab(flat, MS, 1)
    var m = MF()
    m.load_from_slab(flat)
    var flat2 = List[Scalar[DTYPE]](length=MS, fill=Scalar[DTYPE](0))
    m.store_to_slab(flat2)
    _compare(flat, flat2, MS, "ModelFields slab round-trip")
    m.upload_all(ctx)  # smoke: device upload of all record tensors

    # ── 4. Zero-width params variant ─────────────────────────────────────
    comptime SS0 = state_size[3, 3, 2, 4, 0]()
    var d0 = DataFields[DTYPE, 3, 3, 2, 4, 0, 2]()
    var s0 = List[Scalar[DTYPE]](length=2 * SS0, fill=Scalar[DTYPE](0))
    _fill_slab(s0, SS0, 2)
    d0.load_from_slab(s0)
    var s0b = List[Scalar[DTYPE]](length=2 * SS0, fill=Scalar[DTYPE](0))
    d0.store_to_slab(s0b)
    _compare(s0, s0b, 2 * SS0, "zero-width (NSITE=0) round-trip")
    var m0 = ModelFields[DTYPE, 3, 2, 2]()  # NGEOM=NEQ=NTENDON=NSITE=0
    var f0 = List[Scalar[DTYPE]](
        length=ModelFields[DTYPE, 3, 2, 2].MS, fill=Scalar[DTYPE](0)
    )
    _fill_slab(f0, ModelFields[DTYPE, 3, 2, 2].MS, 1)
    m0.load_from_slab(f0)
    var f0b = List[Scalar[DTYPE]](
        length=ModelFields[DTYPE, 3, 2, 2].MS, fill=Scalar[DTYPE](0)
    )
    m0.store_to_slab(f0b)
    _compare(
        f0, f0b, ModelFields[DTYPE, 3, 2, 2].MS, "zero-width model round-trip"
    )

    # ── 5. cvel through DataFields vs legacy slab kernel (GPU, bit-exact) ─
    # A: legacy flat-slab kernel.
    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS)
    for i in range(BATCH * SS):
        slab_t.data[i] = slab[i]
    slab_t.upload(ctx)
    var sbuf = slab_t.dev.value()
    compute_cvel_gpu[DTYPE, BATCH, SS, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
        ctx, sbuf
    )
    slab_t.download(ctx)
    # B: per-field kernel over DataFields (d already holds the same inputs).
    comptime BLOCKS = (BATCH + TPB - 1) // TPB
    ctx.enqueue_function[_cvel_fields_kernel](
        d.xpos.lt["gpu", DF.L_B3](),
        d.xvel.lt["gpu", DF.L_B3](),
        d.xangvel.lt["gpu", DF.L_B3](),
        d.xipos.lt["gpu", DF.L_B3](),
        d.cvel.lt["gpu", DF.L_B6](),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )
    d.cvel.download(ctx)
    comptime O_CVEL = cvel_offset[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
    var bad = 0
    for e in range(BATCH):
        for j in range(NBODY * 6):
            var a = slab_t.data[e * SS + O_CVEL + j]
            var b = d.cvel.data[e * NBODY * 6 + j]
            if a != b:
                if bad < 5:
                    print("  MISMATCH cvel e=", e, " j=", j, a, "vs", b)
                bad += 1
    if bad != 0:
        raise Error("cvel per-field vs legacy: mismatches")
    print(
        "  PASS: cvel per-field vs legacy slab kernel (",
        BATCH * NBODY * 6,
        "values bit-exact )",
    )

    print("test_tensor_fields: ALL PASS")
