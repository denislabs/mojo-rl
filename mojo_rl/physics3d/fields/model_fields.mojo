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

Coexistence: `load_from_slab` / `store_to_slab` bridge to/from the flat
model buffer produced by the existing `copy_model_to_buffer` hand-flattening
(so the parser/setup path is reused verbatim and the bridge is gated
bit-identically). Transitional; dies at P6.

Dtype note: all-`DTYPE` for bit-compatibility with the existing slab (int
columns like `BODY_IDX_PARENT` are float-encoded there). Honest `int32`
record tensors come after consumers stop round-tripping through slabs.
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
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    model_curriculum_offset,
    model_geom_offset,
    model_equality_offset,
    model_tendon_offset,
    model_site_offset,
    model_body_invweight0_offset,
    model_dof_invweight0_offset,
    model_exclude_offset,
    model_mesh_meta_offset,
    model_mesh_vert_offset,
    model_size_with_invweight,
)


@always_inline
def _at_least_one(n: Int) -> Int:
    # Zero-entity record tensors still get a 1-element buffer so `.lt["gpu"]`
    # can always bind them as kernel operands (the layouts are zero-sized and
    # never indexed; an empty Optional[DeviceBuffer] would abort at bind).
    return n if n > 0 else 1


@always_inline
def _block_in[
    dt: DType
](mut t: TensorImpl[dt], flat: List[Scalar[dt]], off: Int, width: Int):
    for i in range(width):
        t.data[i] = flat[off + i]


@always_inline
def _block_out[
    dt: DType
](t: TensorImpl[dt], mut flat: List[Scalar[dt]], off: Int, width: Int):
    for i in range(width):
        flat[off + i] = t.data[i]


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

    # ── Transitional slab bridges (die at P6 with gpu/constants.mojo) ────
    def load_from_slab(mut self, flat: List[Scalar[Self.DTYPE]]):
        """Fill all record tensors from a flat model buffer (the output of
        the existing `copy_model_to_buffer` flattening)."""
        _block_in(
            self.bodies, flat, model_body_offset(0), Self.NBODY * MODEL_BODY_SIZE
        )
        _block_in(
            self.joints,
            flat,
            model_joint_offset[Self.NBODY](0),
            Self.NJOINT * MODEL_JOINT_SIZE,
        )
        _block_in(
            self.meta,
            flat,
            model_metadata_offset[Self.NBODY, Self.NJOINT](),
            MODEL_META_SIZE,
        )
        _block_in(
            self.curriculum,
            flat,
            model_curriculum_offset[Self.NBODY, Self.NJOINT](),
            MODEL_CURRICULUM_SIZE,
        )
        _block_in(
            self.geoms,
            flat,
            model_geom_offset[Self.NBODY, Self.NJOINT](0),
            Self.NGEOM * MODEL_GEOM_SIZE,
        )
        _block_in(
            self.equality,
            flat,
            model_equality_offset[Self.NBODY, Self.NJOINT, Self.NGEOM](0),
            Self.NEQUALITY * MODEL_EQ_SIZE,
        )
        _block_in(
            self.tendons,
            flat,
            model_tendon_offset[
                Self.NBODY, Self.NJOINT, Self.NGEOM, Self.NEQUALITY
            ](0),
            Self.NTENDON * MODEL_TENDON_SIZE,
        )
        _block_in(
            self.sites,
            flat,
            model_site_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
            ](0),
            Self.NSITE * MODEL_SITE_SIZE,
        )
        _block_in(
            self.body_invweight0,
            flat,
            model_body_invweight0_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
            ](),
            Self.NBODY * 2,
        )
        _block_in(
            self.dof_invweight0,
            flat,
            model_dof_invweight0_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
            ](),
            Self.NV,
        )
        _block_in(
            self.excludes,
            flat,
            model_exclude_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NV,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
            ](),
            Self.NEXCLUDE * 2,
        )
        _block_in(
            self.mesh_meta,
            flat,
            model_mesh_meta_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NV,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
                Self.NEXCLUDE,
            ](),
            MAX_GPU_MESHES * MODEL_MESH_META_SIZE,
        )
        _block_in(
            self.mesh_verts,
            flat,
            model_mesh_vert_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NV,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
                Self.NEXCLUDE,
            ](),
            Self.NMESH_VERTS * 3,
        )

    def store_to_slab(self, mut flat: List[Scalar[Self.DTYPE]]):
        """Write all record tensors back into a flat model buffer."""
        _block_out(
            self.bodies, flat, model_body_offset(0), Self.NBODY * MODEL_BODY_SIZE
        )
        _block_out(
            self.joints,
            flat,
            model_joint_offset[Self.NBODY](0),
            Self.NJOINT * MODEL_JOINT_SIZE,
        )
        _block_out(
            self.meta,
            flat,
            model_metadata_offset[Self.NBODY, Self.NJOINT](),
            MODEL_META_SIZE,
        )
        _block_out(
            self.curriculum,
            flat,
            model_curriculum_offset[Self.NBODY, Self.NJOINT](),
            MODEL_CURRICULUM_SIZE,
        )
        _block_out(
            self.geoms,
            flat,
            model_geom_offset[Self.NBODY, Self.NJOINT](0),
            Self.NGEOM * MODEL_GEOM_SIZE,
        )
        _block_out(
            self.equality,
            flat,
            model_equality_offset[Self.NBODY, Self.NJOINT, Self.NGEOM](0),
            Self.NEQUALITY * MODEL_EQ_SIZE,
        )
        _block_out(
            self.tendons,
            flat,
            model_tendon_offset[
                Self.NBODY, Self.NJOINT, Self.NGEOM, Self.NEQUALITY
            ](0),
            Self.NTENDON * MODEL_TENDON_SIZE,
        )
        _block_out(
            self.sites,
            flat,
            model_site_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
            ](0),
            Self.NSITE * MODEL_SITE_SIZE,
        )
        _block_out(
            self.body_invweight0,
            flat,
            model_body_invweight0_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
            ](),
            Self.NBODY * 2,
        )
        _block_out(
            self.dof_invweight0,
            flat,
            model_dof_invweight0_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
            ](),
            Self.NV,
        )
        _block_out(
            self.excludes,
            flat,
            model_exclude_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NV,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
            ](),
            Self.NEXCLUDE * 2,
        )
        _block_out(
            self.mesh_meta,
            flat,
            model_mesh_meta_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NV,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
                Self.NEXCLUDE,
            ](),
            MAX_GPU_MESHES * MODEL_MESH_META_SIZE,
        )
        _block_out(
            self.mesh_verts,
            flat,
            model_mesh_vert_offset[
                Self.NBODY,
                Self.NJOINT,
                Self.NV,
                Self.NGEOM,
                Self.NEQUALITY,
                Self.NTENDON,
                Self.NSITE,
                Self.NEXCLUDE,
            ](),
            Self.NMESH_VERTS * 3,
        )
