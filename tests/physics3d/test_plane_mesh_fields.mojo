"""P4 gate: PLANE-vs-MESH contact record emission — fields vs legacy,
BIT-EXACT, on BOTH detection paths (O(N^2) and SAP).

The fields mesh port (collision/contact_detection_fields.mojo,
`_plane_mesh_contacts_fields`) is verbatim-verified but its plane-mesh
record emission was never OUTPUT-verified: no reachable sawyer pose puts a
collision-enabled mesh below the floor plane (the robot meshes never reach
z=-0.913, and pedestal/table hulls are contype=0 / body-0). This gate
closes the hole with a SYNTHETIC hull:

  SawyerReach (it already has NMESH_VERTS capacity + mesh plumbing),
  BATCH=2. The obj cylinder geom (free joint -> world pose comes straight
  from qpos) is overridden IN THE SLAB to GEOM_MESH pointing at an injected
  4-vertex tetrahedron appended after the STL hulls (mesh id =
  model.num_meshes, vertadr = existing hull vert count). The injection is a
  single write point — the host slab — BEFORE upload and BEFORE
  ModelFields.load_from_slab, so the legacy kernel and the fields tensors
  read IDENTICAL records by construction (setup mirrors
  tests/physics3d/test_mesh_detection_fields.mojo, which documents building
  the mesh-sized slab via setup_model_and_data + copy_*_to_buffer because
  init_model_gpu under-sizes). The obj is teleported to (2, 2, z) — far
  from every other geom — with z chosen so exactly ONE tetra vertex falls
  below the floor plane (env0 dist -0.017, env1 dist -0.029; the other
  three vertices stay >= +0.02 above, decisively outside any margin).

Leg 1 (O(N^2)): legacy FK+detect_contacts_gpu on the flat slab vs fields
FK+detect_contacts_fields, both GPU: contact count + every populated
contact record column BIT-EXACT (`!=`). Non-vacuity: each env must contain
a PLANE-MESH record with the O(N^2) conventions (BODY_A = obj body,
BODY_B = 0, normal +z, DIST < 0).

Leg 2 (SAP): legacy detect_contacts_sap_gpu vs detect_contacts_sap_fields
on the same state, BIT-EXACT. The SAP plane-mesh branch writes DIFFERENT
conventions (BODY_B = -1, DIST offset by the combined margin, no
INCLUDEMARGIN slot — documented in broadphase_sap_fields.mojo's module
docstring), so its non-vacuity check asserts BODY_A = obj body,
BODY_B = -1, normal +z. With both legs green, BOTH plane-mesh
implementations are output-verified.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_plane_mesh_fields.mojo
"""

from std.math import abs
from std.gpu import block_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.constants import GEOM_MESH, GEOM_CYLINDER
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics_gpu,
)
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
)
from mojo_rl.physics3d.collision.contact_detection import detect_contacts_gpu
from mojo_rl.physics3d.collision.contact_detection_fields import (
    detect_contacts_fields,
)
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap_gpu
from mojo_rl.physics3d.collision.broadphase_sap_fields import (
    detect_contacts_sap_fields,
)
from mojo_rl.physics3d.gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_tendons_to_buffer,
    copy_invweight0_to_buffer,
    copy_mesh_hull_to_buffer,
)
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    model_geom_offset,
    model_mesh_meta_offset,
    model_mesh_vert_offset,
    qpos_offset,
    contacts_offset,
    metadata_offset,
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_MESH_ID,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    MAX_GPU_MESHES,
)
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel

comptime DTYPE = DType.float32
comptime NQ = SawyerReachModel.NQ
comptime NV = SawyerReachModel.NV
comptime NBODY = SawyerReachModel.NBODY
comptime NJOINT = SawyerReachModel.NJOINT
comptime NGEOM = SawyerReachModel.NGEOM
comptime NEQ = SawyerReachModel.MAX_EQUALITY
comptime NTD = SawyerReachModel.MAX_TENDON
comptime NSITE = SawyerReachModel.NSITE
comptime MC = SawyerReachModel.MAX_CONTACTS
comptime BATCH = 2
comptime METADATA_SIZE_L = 4
# Mesh hull vertex CAPACITY (compile-time); guarded below against the
# actual STL total + the 4 injected tetra verts.
comptime NMESHV = MAX_GPU_MESHES * 256
comptime SS = state_size[NQ, NV, NBODY, MC, NSITE]()
comptime MS = model_size_with_invweight[
    NBODY, NJOINT, NV, NGEOM, NEQ, NTD, NSITE, 0, NMESHV
]()

# Floor plane world z (sawyer_scene_xml: pos="0 0 -0.913").
comptime FLOOR_Z: Float64 = -0.913
# Obj z per env: exactly one tetra vertex (local z = -0.03) below the floor.
comptime OBJ_Z_ENV0: Float64 = -0.900  # vertex dist = -0.017
comptime OBJ_Z_ENV1: Float64 = -0.912  # vertex dist = -0.029


def _legacy_fk_detect_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_
    ](env, state, model)
    detect_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, NGEOM, NEQ, NTD, NSITE
    ](env, state, model)


def _legacy_fk_sap_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_
    ](env, state, model)
    detect_contacts_sap_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, NGEOM, NEQ, NTD, NSITE
    ](env, state, model)


def _qpos_for_env(e: Int) -> List[Float64]:
    """Canonical reset arm pose (test_mesh_detection_fields); obj teleported
    to (2, 2, z_e) — far from every non-plane geom, so the only new pair is
    plane-vs-(synthetic mesh)."""
    var q = List[Float64](length=NQ, fill=0.0)
    q[0] = 1.889288  # j0
    q[1] = -0.575769  # j1
    q[2] = -0.976659  # j2
    q[3] = 1.641991  # j3
    q[4] = 0.942860  # j4
    q[5] = 1.043696  # j5
    q[6] = 2.292833  # j6
    q[7] = 0.0  # r_close
    q[8] = 0.0  # l_close
    q[9] = 2.0  # obj x
    q[10] = 2.0  # obj y
    q[11] = OBJ_Z_ENV0 if e == 0 else OBJ_Z_ENV1  # obj z
    q[12] = 1.0  # obj quat w (identity)
    return q^


def _compare_bit_exact(
    label: String,
    slab_t: TensorImpl[DTYPE],
    d: DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH],
) raises:
    """Contact count + every populated record column must satisfy `==`
    bit-exactly between the legacy slab and the fields tensors."""
    comptime O_CON = contacts_offset[NQ, NV, NBODY]()
    comptime O_META = metadata_offset[NQ, NV, NBODY, MC]()
    var bad = 0
    for e in range(BATCH):
        var ncon_legacy = Int(
            slab_t.data[e * SS + O_META + META_IDX_NUM_CONTACTS]
        )
        var ncon_fields = Int(
            d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        print(
            "  [", label, "] env", e, ": ncon legacy=", ncon_legacy,
            " fields=", ncon_fields,
        )
        if ncon_legacy != ncon_fields:
            bad += 1
            continue
        if ncon_legacy == 0:
            raise Error(label + ": zero contacts — gate is vacuous")
        for c in range(ncon_legacy):
            for k in range(CONTACT_SIZE):
                var a = d.contacts.data[
                    e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                var b = slab_t.data[e * SS + O_CON + c * CONTACT_SIZE + k]
                if a != b:
                    if bad < 5:
                        print(
                            "  MISMATCH env", e, "contact", c, "field", k,
                            ": fields=", a, " legacy=", b,
                        )
                    bad += 1
    if bad != 0:
        raise Error(label + ": fields-GPU vs legacy-GPU record mismatch")
    print("  [", label, "] PASS: counts + records BIT-EXACT vs legacy")


def _assert_plane_mesh_contact(
    label: String,
    d: DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH],
    obj_body: Int,
    expected_body_b: Int,
) raises:
    """Every env must contain >= 1 plane-mesh record: BODY_A = obj body,
    BODY_B = 0 (O(N^2) convention) or -1 (SAP convention), normal +z."""
    for e in range(BATCH):
        var ncon = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
        var found = 0
        for c in range(ncon):
            var base = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
            var ba = Int(d.contacts.data[base + CONTACT_IDX_BODY_A])
            var bb = Int(d.contacts.data[base + CONTACT_IDX_BODY_B])
            var nx = Float64(d.contacts.data[base + CONTACT_IDX_NX])
            var ny = Float64(d.contacts.data[base + CONTACT_IDX_NY])
            var nz = Float64(d.contacts.data[base + CONTACT_IDX_NZ])
            if ba == obj_body and bb == expected_body_b and nz == 1.0:
                if nx == 0.0 and ny == 0.0:
                    found += 1
                    print(
                        "  [", label, "] env", e,
                        ": plane-mesh contact bodies (", ba, ",", bb,
                        ") dist=",
                        d.contacts.data[base + CONTACT_IDX_DIST],
                    )
        if found == 0:
            raise Error(
                label
                + ": no PLANE-MESH contact for the synthetic hull in env "
                + String(e)
                + " — gate is vacuous"
            )


def main() raises:
    print("--- plane-mesh contact emission: fields vs legacy, BATCH=", BATCH)
    var ctx = DeviceContext()

    # ── CPU model (loads the STL hulls) + mesh-sized slab, exactly like
    # test_mesh_detection_fields (init_model_gpu under-sizes mesh models).
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ,
        SawyerReachModel.CONE_TYPE, NTD, NSITE,
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MC, NSITE]()
    SawyerReachModel.setup_model_and_data[DTYPE](model, data)
    var n_stl_meshes = model.num_meshes
    var n_stl_verts = len(model.mesh_vert) // 3
    print("  num_meshes:", n_stl_meshes, " hull verts:", n_stl_verts)
    if n_stl_meshes == 0 or n_stl_verts == 0:
        raise Error("expected STL mesh hulls — gate is vacuous")
    if n_stl_meshes + 1 > MAX_GPU_MESHES:
        raise Error("no free mesh_meta slot for the synthetic tetrahedron")
    if n_stl_verts + 4 > NMESHV:
        raise Error("no vertex capacity for the tetrahedron — raise NMESHV")

    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](MS)
    ctx.synchronize()
    for i in range(MS):
        host_buf[i] = Scalar[DTYPE](0)
    copy_model_to_buffer(model, host_buf)
    copy_geoms_to_buffer(model, host_buf)
    copy_tendons_to_buffer(model, host_buf)
    copy_invweight0_to_buffer(model, host_buf)
    copy_mesh_hull_to_buffer(model, host_buf)

    var model_t = TensorImpl[DTYPE].alloc(MS)
    for i in range(MS):
        model_t.data[i] = host_buf[i]

    # ── Locate the obj cylinder geom (contype=1, free-joint body).
    var g_obj = -1
    var obj_body = -1
    for g in range(NGEOM):
        var g_off = model_geom_offset[NBODY, NJOINT](g)
        var gt = Int(model_t.data[g_off + GEOM_IDX_TYPE])
        if gt != GEOM_CYLINDER:
            continue
        var r = Float64(model_t.data[g_off + GEOM_IDX_RADIUS])
        var hl = Float64(model_t.data[g_off + GEOM_IDX_HALF_LENGTH])
        if abs(r - 0.02) < 1e-6 and abs(hl - 0.02) < 1e-6:
            g_obj = g
            obj_body = Int(model_t.data[g_off + GEOM_IDX_BODY])
    print("  obj geom:", g_obj, " obj body:", obj_body)
    if g_obj < 0 or obj_body <= 0:
        raise Error("could not identify the obj cylinder geom")

    # ── SINGLE-POINT INJECTION into the host slab (before upload and before
    # load_from_slab -> both sides read identical records):
    #   1. synthetic tetrahedron appended after the STL hull verts,
    #   2. obj geom overridden to GEOM_MESH pointing at it.
    comptime MESH_META_OFF = model_mesh_meta_offset[
        NBODY, NJOINT, NV, NGEOM, NEQ, NTD, NSITE, 0
    ]()
    comptime MESH_VERT_OFF = model_mesh_vert_offset[
        NBODY, NJOINT, NV, NGEOM, NEQ, NTD, NSITE, 0
    ]()
    var tetra_id = n_stl_meshes
    model_t.data[MESH_META_OFF + tetra_id * 2 + 0] = Scalar[DTYPE](n_stl_verts)
    model_t.data[MESH_META_OFF + tetra_id * 2 + 1] = Scalar[DTYPE](4)
    # Local-frame tetrahedron: only vertex 0 (z=-0.03) can dip below the
    # plane at the test poses; the rest sit >= +0.02 above it.
    var tetra = List[Float64](length=12, fill=0.0)
    tetra[0] = 0.015
    tetra[1] = 0.0
    tetra[2] = -0.03
    tetra[3] = -0.015
    tetra[4] = 0.012
    tetra[5] = 0.02
    tetra[6] = 0.006
    tetra[7] = -0.015
    tetra[8] = 0.025
    tetra[9] = -0.004
    tetra[10] = 0.008
    tetra[11] = 0.03
    for k in range(12):
        model_t.data[MESH_VERT_OFF + n_stl_verts * 3 + k] = Scalar[DTYPE](
            tetra[k]
        )
    var g_off_obj = model_geom_offset[NBODY, NJOINT](g_obj)
    model_t.data[g_off_obj + GEOM_IDX_TYPE] = Scalar[DTYPE](GEOM_MESH)
    model_t.data[g_off_obj + GEOM_IDX_MESH_ID] = Scalar[DTYPE](tetra_id)

    model_t.upload(ctx)
    var mf = ModelFields[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, 0, NMESHV
    ]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)
    # Injection sanity: the fields tensors carry the tetra + override.
    if Int(mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_TYPE]) != GEOM_MESH:
        raise Error("geom override did not reach ModelFields")
    if Int(mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID]) != (
        tetra_id
    ):
        raise Error("mesh id override did not reach ModelFields")
    if Int(mf.mesh_meta.data[tetra_id * 2 + 1]) != 4:
        raise Error("tetra mesh_meta did not reach ModelFields")

    # ── Poses.
    comptime O_QPOS = qpos_offset[NQ, NV]()

    # ================= Leg 1: O(N^2) detection ==========================
    var slab_a = TensorImpl[DTYPE].alloc(BATCH * SS)
    var d_a = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        var q = _qpos_for_env(e)
        for i in range(NQ):
            slab_a.data[e * SS + O_QPOS + i] = Scalar[DTYPE](q[i])
            d_a.qpos.data[e * NQ + i] = Scalar[DTYPE](q[i])
    slab_a.upload(ctx)
    d_a.upload_all(ctx)

    ctx.enqueue_function[_legacy_fk_detect_kernel[BATCH]](
        slab_a.lt["gpu", Layout.row_major(BATCH, SS)](),
        model_t.lt["gpu", Layout.row_major(1, MS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    slab_a.download(ctx)

    forward_kinematics_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_a, mf, ctx)
    detect_contacts_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_a, mf, ctx)
    d_a.contacts.download(ctx)
    d_a.meta.download(ctx)

    _compare_bit_exact("O(N^2)", slab_a, d_a)
    # O(N^2) plane-mesh convention: BODY_B = 0, DIST = dist_v (< 0 here).
    _assert_plane_mesh_contact("O(N^2)", d_a, obj_body, 0)
    for e in range(BATCH):
        var ncon = Int(
            d_a.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        var found_neg = False
        for c in range(ncon):
            var base = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
            if (
                Int(d_a.contacts.data[base + CONTACT_IDX_BODY_A]) == obj_body
                and Int(d_a.contacts.data[base + CONTACT_IDX_BODY_B]) == 0
                and Float64(d_a.contacts.data[base + CONTACT_IDX_DIST]) < 0.0
            ):
                found_neg = True
        if not found_neg:
            raise Error("O(N^2): plane-mesh contact has non-negative DIST")
    print("  [ O(N^2) ] PASS: plane-mesh record present (BODY_B=0, DIST<0)")

    # ================= Leg 2: SAP detection =============================
    var slab_b = TensorImpl[DTYPE].alloc(BATCH * SS)
    var d_b = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        var q = _qpos_for_env(e)
        for i in range(NQ):
            slab_b.data[e * SS + O_QPOS + i] = Scalar[DTYPE](q[i])
            d_b.qpos.data[e * NQ + i] = Scalar[DTYPE](q[i])
    slab_b.upload(ctx)
    d_b.upload_all(ctx)

    ctx.enqueue_function[_legacy_fk_sap_kernel[BATCH]](
        slab_b.lt["gpu", Layout.row_major(BATCH, SS)](),
        model_t.lt["gpu", Layout.row_major(1, MS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    slab_b.download(ctx)

    forward_kinematics_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_b, mf, ctx)
    detect_contacts_sap_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_b, mf, ctx)
    d_b.contacts.download(ctx)
    d_b.meta.download(ctx)

    _compare_bit_exact("SAP", slab_b, d_b)
    # SAP plane-mesh convention: BODY_B = -1, DIST = dist_v - margin.
    _assert_plane_mesh_contact("SAP", d_b, obj_body, -1)
    print("  [ SAP ] PASS: plane-mesh record present (BODY_B=-1)")

    print("test_plane_mesh_fields: ALL PASS")
