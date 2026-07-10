"""Per-record tensor container for static model config (migration P1).

`ModelFields` is the end-state replacement for the flat GPU model slab: each
record FAMILY becomes one owned packed tensor (`bodies [NBODY, 25]`,
`joints [NJOINT, 26]`, `geoms [NGEOM, 30]`, …) rather than one giant buffer.
Column indices are the existing `BODY_IDX_*` / `JOINT_IDX_*` / … constants —
after sunset they move here and become the single, local column definition.

Records stay PACKED (not one tensor per column) by design: the P0 audit +
the measured Metal operand cliff (28 ok / 29 = JIT abort) make per-column
splits of body/joint/geom records blow the kernel operand budget on the wide
kernels. Kernels bind one record tensor per entity kind and read columns —
same addressing as today's record-strided slab, minus the cross-family
offset math.

Build: `load_from_model` fills the record tensors DIRECTLY from the CPU
`Model` (`i * MODEL_<KIND>_SIZE + <KIND>_IDX_*`), no flat slab. The
transitional `load_from_slab` / `store_to_slab` bridges + the cross-family
`model_*_offset` tables were deleted at the P6 sunset.

Dtype note: all-`DTYPE` (int columns like `BODY_IDX_PARENT` are float-encoded).
Honest `int32` record tensors are a later cleanup.
"""

from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl

from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    model_size_with_invweight,
)

# Column-index constants for the offset-free `load_from_model` fill. These are
# per-record COLUMN offsets: a body field is at `body * MODEL_BODY_SIZE +
# BODY_IDX_*` inside the packed `bodies` tensor.
from ..gpu.constants import (
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_ROOTID,
    BODY_IDX_WELDID,
    BODY_IDX_MOCAP,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TAU_LIMIT,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
    JOINT_IDX_QPOS0,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    MODEL_META_IDX_NEXCLUDE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4,
    GEOM_IDX_MARGIN,
    GEOM_IDX_MESH_ID,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_JOINT_1,
    TENDON_IDX_JOINT_2,
    TENDON_IDX_JOINT_3,
    TENDON_IDX_COEF_0,
    TENDON_IDX_COEF_1,
    TENDON_IDX_COEF_2,
    TENDON_IDX_COEF_3,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
)
from ..types import Model, ConeType


@always_inline
def _at_least_one(n: Int) -> Int:
    # Zero-entity record tensors still get a 1-element buffer so `.lt["gpu"]`
    # can always bind them as kernel operands (the layouts are zero-sized and
    # never indexed; an empty Optional[DeviceBuffer] would abort at bind).
    return n if n > 0 else 1


struct ModelFields[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
](Movable):
    """Static model config as one packed tensor per record family (13
    tensors). See module docstring."""

    comptime MS = model_size_with_invweight[
        Self.NBODY,
        Self.NJOINT,
        Self.NV,
        Self.NGEOM,
        Self.NEQUALITY,
        Self.NTENDON,
        Self.NSITE,
        Self.NEXCLUDE,
        Self.NMESH_VERTS,
    ]()

    # Record view layouts ([N_ENTITY, RECORD_SIZE] row-major; tails 1-D).
    comptime L_BODY = Layout.row_major(Self.NBODY, MODEL_BODY_SIZE)
    comptime L_JOINT = Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE)
    comptime L_META = Layout.row_major(MODEL_META_SIZE)
    comptime L_CURRICULUM = Layout.row_major(MODEL_CURRICULUM_SIZE)
    comptime L_GEOM = Layout.row_major(Self.NGEOM, MODEL_GEOM_SIZE)
    comptime L_EQ = Layout.row_major(Self.NEQUALITY, MODEL_EQ_SIZE)
    comptime L_TENDON = Layout.row_major(Self.NTENDON, MODEL_TENDON_SIZE)
    comptime L_SITE = Layout.row_major(Self.NSITE, MODEL_SITE_SIZE)
    comptime L_BODY_INVW = Layout.row_major(Self.NBODY, 2)
    comptime L_DOF_INVW = Layout.row_major(Self.NV)
    comptime L_EXCLUDE = Layout.row_major(Self.NEXCLUDE, 2)
    comptime L_MESH_META = Layout.row_major(MAX_GPU_MESHES, MODEL_MESH_META_SIZE)
    comptime L_MESH_VERT = Layout.row_major(Self.NMESH_VERTS, 3)

    var bodies: TensorImpl[Self.DTYPE]  # [NBODY, MODEL_BODY_SIZE]
    var joints: TensorImpl[Self.DTYPE]  # [NJOINT, MODEL_JOINT_SIZE]
    var meta: TensorImpl[Self.DTYPE]  # [MODEL_META_SIZE]
    var curriculum: TensorImpl[Self.DTYPE]  # [MODEL_CURRICULUM_SIZE]
    var geoms: TensorImpl[Self.DTYPE]  # [NGEOM, MODEL_GEOM_SIZE]
    var equality: TensorImpl[Self.DTYPE]  # [NEQUALITY, MODEL_EQ_SIZE]
    var tendons: TensorImpl[Self.DTYPE]  # [NTENDON, MODEL_TENDON_SIZE]
    var sites: TensorImpl[Self.DTYPE]  # [NSITE, MODEL_SITE_SIZE]
    var body_invweight0: TensorImpl[Self.DTYPE]  # [NBODY, 2]
    var dof_invweight0: TensorImpl[Self.DTYPE]  # [NV]
    var excludes: TensorImpl[Self.DTYPE]  # [NEXCLUDE, 2]
    var mesh_meta: TensorImpl[Self.DTYPE]  # [MAX_GPU_MESHES, 2]
    var mesh_verts: TensorImpl[Self.DTYPE]  # [NMESH_VERTS, 3]

    def __init__(out self) raises:
        self.bodies = TensorImpl[Self.DTYPE].alloc(
            Self.NBODY * MODEL_BODY_SIZE
        )
        self.joints = TensorImpl[Self.DTYPE].alloc(
            Self.NJOINT * MODEL_JOINT_SIZE
        )
        self.meta = TensorImpl[Self.DTYPE].alloc(MODEL_META_SIZE)
        self.curriculum = TensorImpl[Self.DTYPE].alloc(MODEL_CURRICULUM_SIZE)
        self.geoms = TensorImpl[Self.DTYPE].alloc(_at_least_one(Self.NGEOM * MODEL_GEOM_SIZE))
        self.equality = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(Self.NEQUALITY * MODEL_EQ_SIZE)
        )
        self.tendons = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(Self.NTENDON * MODEL_TENDON_SIZE)
        )
        self.sites = TensorImpl[Self.DTYPE].alloc(_at_least_one(Self.NSITE * MODEL_SITE_SIZE))
        self.body_invweight0 = TensorImpl[Self.DTYPE].alloc(Self.NBODY * 2)
        self.dof_invweight0 = TensorImpl[Self.DTYPE].alloc(Self.NV)
        self.excludes = TensorImpl[Self.DTYPE].alloc(_at_least_one(Self.NEXCLUDE * 2))
        self.mesh_meta = TensorImpl[Self.DTYPE].alloc(
            MAX_GPU_MESHES * MODEL_MESH_META_SIZE
        )
        self.mesh_verts = TensorImpl[Self.DTYPE].alloc(_at_least_one(Self.NMESH_VERTS * 3))

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Host -> device for every record tensor (static config: called once
        after setup, like the model slab upload today)."""
        self.bodies.upload(ctx)
        self.joints.upload(ctx)
        self.meta.upload(ctx)
        self.curriculum.upload(ctx)
        self.geoms.upload(ctx)
        self.equality.upload(ctx)
        self.tendons.upload(ctx)
        self.sites.upload(ctx)
        self.body_invweight0.upload(ctx)
        self.dof_invweight0.upload(ctx)
        self.excludes.upload(ctx)
        self.mesh_meta.upload(ctx)
        self.mesh_verts.upload(ctx)

    # ── Offset-free packed fill (B3): the sunset-surviving model build ───
    def load_from_model[
        NQ: Int,
        MAX_CONTACTS: Int,
        CONE_TYPE: Int,
    ](
        mut self,
        model: Model[
            Self.DTYPE,
            NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
            Self.NGEOM,
            Self.NEQUALITY,
            CONE_TYPE,
            Self.NTENDON,
            Self.NSITE,
        ],
    ):
        """Populate the packed record tensors DIRECTLY from a CPU `Model`,
        with no flat slab and no cross-family offset math.

        Bit-exact replacement for the transitional
        `copy_*_to_buffer` + `load_from_slab` round-trip used by
        `init_fields`: each record `i` writes at `i * MODEL_<KIND>_SIZE +
        <KIND>_IDX_*` inside its own packed tensor, so the only constants
        needed are per-record COLUMN indices — the `model_*_offset` slab
        tables (and the whole `gpu/constants` offset layer) die at sunset
        while this fill survives. Records left unwritten by the legacy build
        path (sites, curriculum) stay zero, matching `init_fields` today.

        Caller uploads afterwards via `upload_all`.
        """
        # ── bodies ──────────────────────────────────────────────────────
        for b in range(Self.NBODY):
            var o = b * MODEL_BODY_SIZE
            self.bodies.data[o + BODY_IDX_MASS] = model.body_mass[b]
            self.bodies.data[o + BODY_IDX_INV_MASS] = model.body_inv_mass[b]
            self.bodies.data[o + BODY_IDX_IXX] = model.body_inertia[b * 3 + 0]
            self.bodies.data[o + BODY_IDX_IYY] = model.body_inertia[b * 3 + 1]
            self.bodies.data[o + BODY_IDX_IZZ] = model.body_inertia[b * 3 + 2]
            self.bodies.data[o + BODY_IDX_INV_IXX] = model.body_inv_inertia[
                b * 3 + 0
            ]
            self.bodies.data[o + BODY_IDX_INV_IYY] = model.body_inv_inertia[
                b * 3 + 1
            ]
            self.bodies.data[o + BODY_IDX_INV_IZZ] = model.body_inv_inertia[
                b * 3 + 2
            ]
            self.bodies.data[o + BODY_IDX_POS_X] = model.body_pos[b * 3 + 0]
            self.bodies.data[o + BODY_IDX_POS_Y] = model.body_pos[b * 3 + 1]
            self.bodies.data[o + BODY_IDX_POS_Z] = model.body_pos[b * 3 + 2]
            self.bodies.data[o + BODY_IDX_QUAT_X] = model.body_quat[b * 4 + 0]
            self.bodies.data[o + BODY_IDX_QUAT_Y] = model.body_quat[b * 4 + 1]
            self.bodies.data[o + BODY_IDX_QUAT_Z] = model.body_quat[b * 4 + 2]
            self.bodies.data[o + BODY_IDX_QUAT_W] = model.body_quat[b * 4 + 3]
            self.bodies.data[o + BODY_IDX_PARENT] = Scalar[Self.DTYPE](
                model.body_parent[b]
            )
            self.bodies.data[o + BODY_IDX_IPOS_X] = model.body_ipos[b * 3 + 0]
            self.bodies.data[o + BODY_IDX_IPOS_Y] = model.body_ipos[b * 3 + 1]
            self.bodies.data[o + BODY_IDX_IPOS_Z] = model.body_ipos[b * 3 + 2]
            self.bodies.data[o + BODY_IDX_IQUAT_X] = model.body_iquat[b * 4 + 0]
            self.bodies.data[o + BODY_IDX_IQUAT_Y] = model.body_iquat[b * 4 + 1]
            self.bodies.data[o + BODY_IDX_IQUAT_Z] = model.body_iquat[b * 4 + 2]
            self.bodies.data[o + BODY_IDX_IQUAT_W] = model.body_iquat[b * 4 + 3]
            self.bodies.data[o + BODY_IDX_ROOTID] = Scalar[Self.DTYPE](
                model.body_rootid[b]
            )
            self.bodies.data[o + BODY_IDX_WELDID] = Scalar[Self.DTYPE](
                model.body_weldid[b]
            )
            self.bodies.data[o + BODY_IDX_MOCAP] = Scalar[Self.DTYPE](
                1.0 if model.body_mocap[b] else 0.0
            )

        # ── joints ──────────────────────────────────────────────────────
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var o = j * MODEL_JOINT_SIZE
            self.joints.data[o + JOINT_IDX_TYPE] = Scalar[Self.DTYPE](
                joint.jnt_type
            )
            self.joints.data[o + JOINT_IDX_BODY_ID] = Scalar[Self.DTYPE](
                joint.body_id
            )
            self.joints.data[o + JOINT_IDX_QPOS_ADR] = Scalar[Self.DTYPE](
                joint.qpos_adr
            )
            self.joints.data[o + JOINT_IDX_DOF_ADR] = Scalar[Self.DTYPE](
                joint.dof_adr
            )
            self.joints.data[o + JOINT_IDX_POS_X] = joint.pos_x
            self.joints.data[o + JOINT_IDX_POS_Y] = joint.pos_y
            self.joints.data[o + JOINT_IDX_POS_Z] = joint.pos_z
            self.joints.data[o + JOINT_IDX_AXIS_X] = joint.axis_x
            self.joints.data[o + JOINT_IDX_AXIS_Y] = joint.axis_y
            self.joints.data[o + JOINT_IDX_AXIS_Z] = joint.axis_z
            self.joints.data[o + JOINT_IDX_TAU_LIMIT] = joint.tau_limit
            self.joints.data[o + JOINT_IDX_RANGE_MIN] = joint.range_min
            self.joints.data[o + JOINT_IDX_RANGE_MAX] = joint.range_max
            self.joints.data[o + JOINT_IDX_ARMATURE] = joint.armature
            self.joints.data[o + JOINT_IDX_DAMPING] = joint.damping
            self.joints.data[o + JOINT_IDX_STIFFNESS] = joint.stiffness
            self.joints.data[o + JOINT_IDX_SPRINGREF] = joint.springref
            self.joints.data[o + JOINT_IDX_FRICTIONLOSS] = joint.frictionloss
            self.joints.data[o + JOINT_IDX_SOLREF_LIMIT_0] = (
                model.joint_solref_limit[j * 2 + 0]
            )
            self.joints.data[o + JOINT_IDX_SOLREF_LIMIT_1] = (
                model.joint_solref_limit[j * 2 + 1]
            )
            self.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_0] = (
                model.joint_solimp_limit[j * 5 + 0]
            )
            self.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_1] = (
                model.joint_solimp_limit[j * 5 + 1]
            )
            self.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_2] = (
                model.joint_solimp_limit[j * 5 + 2]
            )
            self.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_3] = (
                model.joint_solimp_limit[j * 5 + 3]
            )
            self.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_4] = (
                model.joint_solimp_limit[j * 5 + 4]
            )
            self.joints.data[o + JOINT_IDX_QPOS0] = model.qpos0[joint.qpos_adr]

        # ── meta ────────────────────────────────────────────────────────
        self.meta.data[MODEL_META_IDX_NBODY] = Scalar[Self.DTYPE](Self.NBODY)
        self.meta.data[MODEL_META_IDX_NJOINT] = Scalar[Self.DTYPE](
            model.num_joints
        )
        self.meta.data[MODEL_META_IDX_GRAVITY_X] = model.gravity[0]
        self.meta.data[MODEL_META_IDX_GRAVITY_Y] = model.gravity[1]
        self.meta.data[MODEL_META_IDX_GRAVITY_Z] = model.gravity[2]
        self.meta.data[MODEL_META_IDX_TIMESTEP] = model.timestep
        self.meta.data[MODEL_META_IDX_DENSITY] = model.opt_density
        self.meta.data[MODEL_META_IDX_VISCOSITY] = model.opt_viscosity
        self.meta.data[MODEL_META_IDX_SOLREF_CONTACT_0] = model.solref_contact[0]
        self.meta.data[MODEL_META_IDX_SOLREF_CONTACT_1] = model.solref_contact[1]
        self.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_0] = model.solimp_contact[0]
        self.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_1] = model.solimp_contact[1]
        self.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_2] = model.solimp_contact[2]
        self.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_3] = model.solimp_contact[3]
        self.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_4] = model.solimp_contact[4]
        self.meta.data[MODEL_META_IDX_SOLREF_LIMIT_0] = model.solref_limit[0]
        self.meta.data[MODEL_META_IDX_SOLREF_LIMIT_1] = model.solref_limit[1]
        self.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_0] = model.solimp_limit[0]
        self.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_1] = model.solimp_limit[1]
        self.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_2] = model.solimp_limit[2]
        self.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_3] = model.solimp_limit[3]
        self.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_4] = model.solimp_limit[4]
        self.meta.data[MODEL_META_IDX_IMPRATIO] = model.impratio
        self.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[Self.DTYPE](
            model.num_equality
        )
        self.meta.data[MODEL_META_IDX_NTENDON] = Scalar[Self.DTYPE](
            model.num_tendons
        )
        self.meta.data[MODEL_META_IDX_NEXCLUDE] = Scalar[Self.DTYPE](
            model.num_excludes
        )

        # ── geoms ───────────────────────────────────────────────────────
        for g in range(Self.NGEOM):
            var o = g * MODEL_GEOM_SIZE
            self.geoms.data[o + GEOM_IDX_TYPE] = Scalar[Self.DTYPE](
                model.geom_type[g]
            )
            self.geoms.data[o + GEOM_IDX_BODY] = Scalar[Self.DTYPE](
                model.geom_body[g]
            )
            self.geoms.data[o + GEOM_IDX_POS_X] = model.geom_pos[g * 3 + 0]
            self.geoms.data[o + GEOM_IDX_POS_Y] = model.geom_pos[g * 3 + 1]
            self.geoms.data[o + GEOM_IDX_POS_Z] = model.geom_pos[g * 3 + 2]
            self.geoms.data[o + GEOM_IDX_QUAT_X] = model.geom_quat[g * 4 + 0]
            self.geoms.data[o + GEOM_IDX_QUAT_Y] = model.geom_quat[g * 4 + 1]
            self.geoms.data[o + GEOM_IDX_QUAT_Z] = model.geom_quat[g * 4 + 2]
            self.geoms.data[o + GEOM_IDX_QUAT_W] = model.geom_quat[g * 4 + 3]
            self.geoms.data[o + GEOM_IDX_RADIUS] = model.geom_radius[g]
            self.geoms.data[o + GEOM_IDX_HALF_LENGTH] = model.geom_half_length[g]
            self.geoms.data[o + GEOM_IDX_HALF_X] = model.geom_half_x[g]
            self.geoms.data[o + GEOM_IDX_HALF_Y] = model.geom_half_y[g]
            self.geoms.data[o + GEOM_IDX_HALF_Z] = model.geom_half_z[g]
            self.geoms.data[o + GEOM_IDX_FRICTION] = model.geom_friction[g]
            self.geoms.data[o + GEOM_IDX_CONTYPE] = Scalar[Self.DTYPE](
                model.geom_contype[g]
            )
            self.geoms.data[o + GEOM_IDX_CONAFFINITY] = Scalar[Self.DTYPE](
                model.geom_conaffinity[g]
            )
            self.geoms.data[o + GEOM_IDX_CONDIM] = Scalar[Self.DTYPE](
                model.geom_condim[g]
            )
            self.geoms.data[o + GEOM_IDX_FRICTION_SPIN] = (
                model.geom_friction_spin[g]
            )
            self.geoms.data[o + GEOM_IDX_FRICTION_ROLL] = (
                model.geom_friction_roll[g]
            )
            self.geoms.data[o + GEOM_IDX_RBOUND] = model.geom_rbound[g]
            self.geoms.data[o + GEOM_IDX_SOLREF_0] = model.geom_solref[g * 2 + 0]
            self.geoms.data[o + GEOM_IDX_SOLREF_1] = model.geom_solref[g * 2 + 1]
            self.geoms.data[o + GEOM_IDX_SOLIMP_0] = model.geom_solimp[g * 5 + 0]
            self.geoms.data[o + GEOM_IDX_SOLIMP_1] = model.geom_solimp[g * 5 + 1]
            self.geoms.data[o + GEOM_IDX_SOLIMP_2] = model.geom_solimp[g * 5 + 2]
            self.geoms.data[o + GEOM_IDX_SOLIMP_3] = model.geom_solimp[g * 5 + 3]
            self.geoms.data[o + GEOM_IDX_SOLIMP_4] = model.geom_solimp[g * 5 + 4]
            self.geoms.data[o + GEOM_IDX_MARGIN] = model.geom_margin[g]
            self.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[Self.DTYPE](
                model.geom_mesh_id[g]
            )

        # ── equality ────────────────────────────────────────────────────
        for e in range(model.num_equality):
            var eq = model.equality_constraints[e]
            var o = e * MODEL_EQ_SIZE
            self.equality.data[o + EQ_IDX_TYPE] = Scalar[Self.DTYPE](eq.eq_type)
            self.equality.data[o + EQ_IDX_BODY_A] = Scalar[Self.DTYPE](eq.body_a)
            self.equality.data[o + EQ_IDX_BODY_B] = Scalar[Self.DTYPE](eq.body_b)
            self.equality.data[o + EQ_IDX_ANCHOR_AX] = eq.anchor_a_x
            self.equality.data[o + EQ_IDX_ANCHOR_AY] = eq.anchor_a_y
            self.equality.data[o + EQ_IDX_ANCHOR_AZ] = eq.anchor_a_z
            self.equality.data[o + EQ_IDX_ANCHOR_BX] = eq.anchor_b_x
            self.equality.data[o + EQ_IDX_ANCHOR_BY] = eq.anchor_b_y
            self.equality.data[o + EQ_IDX_ANCHOR_BZ] = eq.anchor_b_z
            self.equality.data[o + EQ_IDX_RELPOSE_X] = eq.relpose_x
            self.equality.data[o + EQ_IDX_RELPOSE_Y] = eq.relpose_y
            self.equality.data[o + EQ_IDX_RELPOSE_Z] = eq.relpose_z
            self.equality.data[o + EQ_IDX_RELPOSE_W] = eq.relpose_w
            self.equality.data[o + EQ_IDX_SOLREF_0] = eq.solref_0
            self.equality.data[o + EQ_IDX_SOLREF_1] = eq.solref_1
            self.equality.data[o + EQ_IDX_SOLIMP_0] = eq.solimp_0
            self.equality.data[o + EQ_IDX_SOLIMP_1] = eq.solimp_1
            self.equality.data[o + EQ_IDX_SOLIMP_2] = eq.solimp_2
            self.equality.data[o + EQ_IDX_SOLIMP_3] = eq.solimp_3
            self.equality.data[o + EQ_IDX_SOLIMP_4] = eq.solimp_4

        # ── tendons ─────────────────────────────────────────────────────
        for t in range(model.num_tendons):
            var ten = model.tendons[t]
            var o = t * MODEL_TENDON_SIZE
            self.tendons.data[o + TENDON_IDX_NUM_JOINTS] = Scalar[Self.DTYPE](
                ten.num_joints
            )
            self.tendons.data[o + TENDON_IDX_JOINT_0] = Scalar[Self.DTYPE](
                ten.joint_idx_0
            )
            self.tendons.data[o + TENDON_IDX_JOINT_1] = Scalar[Self.DTYPE](
                ten.joint_idx_1
            )
            self.tendons.data[o + TENDON_IDX_JOINT_2] = Scalar[Self.DTYPE](
                ten.joint_idx_2
            )
            self.tendons.data[o + TENDON_IDX_JOINT_3] = Scalar[Self.DTYPE](
                ten.joint_idx_3
            )
            self.tendons.data[o + TENDON_IDX_COEF_0] = ten.coef_0
            self.tendons.data[o + TENDON_IDX_COEF_1] = ten.coef_1
            self.tendons.data[o + TENDON_IDX_COEF_2] = ten.coef_2
            self.tendons.data[o + TENDON_IDX_COEF_3] = ten.coef_3
            self.tendons.data[o + TENDON_IDX_LENGTH_REF] = ten.length_ref
            self.tendons.data[o + TENDON_IDX_SOLREF_0] = ten.solref_0
            self.tendons.data[o + TENDON_IDX_SOLREF_1] = ten.solref_1
            self.tendons.data[o + TENDON_IDX_SOLIMP_0] = ten.solimp_0
            self.tendons.data[o + TENDON_IDX_SOLIMP_1] = ten.solimp_1
            self.tendons.data[o + TENDON_IDX_SOLIMP_2] = ten.solimp_2
            self.tendons.data[o + TENDON_IDX_SOLIMP_3] = ten.solimp_3
            self.tendons.data[o + TENDON_IDX_SOLIMP_4] = ten.solimp_4

        # ── invweight0 (CPU-computed by setup_model_and_data) ────────────
        for i in range(Self.NBODY * 2):
            self.body_invweight0.data[i] = model.body_invweight0[i]
        for i in range(Self.NV):
            self.dof_invweight0.data[i] = model.dof_invweight0[i]

        # ── contact exclusion pairs ──────────────────────────────────────
        for i in range(model.num_excludes):
            self.excludes.data[i * 2 + 0] = Scalar[Self.DTYPE](
                model.exclude_body1[i]
            )
            self.excludes.data[i * 2 + 1] = Scalar[Self.DTYPE](
                model.exclude_body2[i]
            )

        # ── mesh hulls ──────────────────────────────────────────────────
        for m in range(model.num_meshes):
            if m >= MAX_GPU_MESHES:
                break
            self.mesh_meta.data[m * MODEL_MESH_META_SIZE + 0] = Scalar[Self.DTYPE](
                model.mesh_vertadr[m]
            )
            self.mesh_meta.data[m * MODEL_MESH_META_SIZE + 1] = Scalar[Self.DTYPE](
                model.mesh_vertnum[m]
            )
        for i in range(len(model.mesh_vert)):
            if i >= Self.NMESH_VERTS * 3:
                break
            self.mesh_verts.data[i] = model.mesh_vert[i]
