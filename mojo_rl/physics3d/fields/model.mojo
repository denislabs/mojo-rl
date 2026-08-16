"""Per-record tensor container for static model config (migration P1).

`Model` is the end-state replacement for the flat GPU model slab: each
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

from max.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl

from .dims import DimsLike

from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_PAIR_SIZE,
    MODEL_MESH_META_SIZE,
    MODEL_MESH_POLY_SIZE,
    MAX_GPU_MESHES,
    mesh_max_poly,
    mesh_max_polyvert,
    mesh_max_edge,
    MODEL_META_IDX_CCD_TOLERANCE,
    MODEL_META_IDX_CCD_ITERATIONS,
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
)

@always_inline
def _at_least_one(n: Int) -> Int:
    # Zero-entity record tensors still get a 1-element buffer so `.lt["gpu"]`
    # can always bind them as kernel operands (the layouts are zero-sized and
    # never indexed; an empty Optional[DeviceBuffer] would abort at bind).
    return n if n > 0 else 1


struct Model[
    DTYPE: DType,
    D: DimsLike,
](Movable):
    """Static model config as one packed tensor per record family (13
    tensors). See module docstring."""

    # ⚠ THE POSITIONAL HAZARD THIS STRUCT DOCUMENTED IS GONE.
    #
    # The parameter list used to carry a standing warning: every entry was an
    # `Int`, so inserting one mid-list silently shifted every positional
    # instantiation — `NMESH_VERTS` would take `NPAIR`'s value and mesh
    # collision would switch itself off across the tree with nothing to
    # compile-error on, and the comment recorded that "that exact failure has
    # happened here before". New dimensions had to go on the END, forever.
    #
    # With one `D` there are no positions to shift: a dimension is named, and
    # the 272 call sites that used to spell 7, 9 or 10 `Int`s in a fixed order
    # now name none. New dimensions can go anywhere in `DimsLike`.
    comptime NV = Self.D.NV
    comptime NBODY = Self.D.NBODY
    comptime NJOINT = Self.D.NJOINT
    comptime NGEOM = Self.D.NGEOM
    comptime NEQUALITY = Self.D.NEQUALITY
    comptime NTENDON = Self.D.NTENDON
    comptime NSITE = Self.D.NSITE
    comptime NEXCLUDE = Self.D.NEXCLUDE
    comptime NMESH_VERTS = Self.D.NMESH_VERTS
    comptime NPAIR = Self.D.NPAIR


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
    comptime L_PAIR = Layout.row_major(Self.NPAIR, MODEL_PAIR_SIZE)
    comptime L_MESH_META = Layout.row_major(MAX_GPU_MESHES, MODEL_MESH_META_SIZE)
    comptime L_MESH_VERT = Layout.row_major(Self.NMESH_VERTS, 3)
    # Mesh POLYGON topology for the native multi-contact path. The capacities
    # are Euler's formula on NMESH_VERTS, so they need no type parameter of
    # their own — see `mesh_max_poly` / `mesh_max_polyvert`.
    comptime NMESH_POLY = mesh_max_poly(Self.NMESH_VERTS)
    comptime NMESH_POLYVERT = mesh_max_polyvert(Self.NMESH_VERTS)
    comptime L_MESH_POLY = Layout.row_major(
        Self.NMESH_POLY, MODEL_MESH_POLY_SIZE
    )
    comptime L_MESH_POLYVERT = Layout.row_major(Self.NMESH_POLYVERT)
    comptime L_MESH_POLYMAP = Layout.row_major(Self.NMESH_POLYVERT)
    comptime L_MESH_VERT_POLYMAP = Layout.row_major(Self.NMESH_VERTS, 2)
    comptime NMESH_EDGE = mesh_max_edge(Self.NMESH_VERTS)
    comptime L_MESH_VERT_EDGEADR = Layout.row_major(Self.NMESH_VERTS)
    comptime L_MESH_EDGE = Layout.row_major(Self.NMESH_EDGE)

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
    var pairs: TensorImpl[Self.DTYPE]  # [NPAIR, MODEL_PAIR_SIZE]
    var mesh_meta: TensorImpl[Self.DTYPE]  # [MAX_GPU_MESHES, 4]
    var mesh_verts: TensorImpl[Self.DTYPE]  # [NMESH_VERTS, 3]
    var mesh_polys: TensorImpl[Self.DTYPE]  # [NMESH_POLY, 5]
    var mesh_polyvert: TensorImpl[Self.DTYPE]  # [NMESH_POLYVERT]
    var mesh_polymap: TensorImpl[Self.DTYPE]  # [NMESH_POLYVERT]
    var mesh_vert_polymap: TensorImpl[Self.DTYPE]  # [NMESH_VERTS, 2] adr,num
    # Hull vertex adjacency for `mjc_PlaneConvex`'s graph path — see
    # `collision/convex_hull.mojo::build_hull_edge_graph`. `mesh_vert_edgeadr`
    # is indexed by GLOBAL hull vertex (parallel to `mesh_verts`) and points
    # into `mesh_edges`, whose per-vertex runs are -1 terminated.
    var mesh_vert_edgeadr: TensorImpl[Self.DTYPE]  # [NMESH_VERTS]
    var mesh_edges: TensorImpl[Self.DTYPE]  # [NMESH_EDGE_SLOTS(NMESH_VERTS)]

    def __init__(out self) raises:
        self.bodies = TensorImpl[Self.DTYPE].alloc(
            Self.NBODY * MODEL_BODY_SIZE
        )
        self.joints = TensorImpl[Self.DTYPE].alloc(
            Self.NJOINT * MODEL_JOINT_SIZE
        )
        self.meta = TensorImpl[Self.DTYPE].alloc(MODEL_META_SIZE)
        # ⚠ MUJOCO'S CCD DEFAULTS, SEEDED HERE BECAUSE A ZERO IS A LEGAL-LOOKING
        # VALUE FOR BOTH. `alloc` does not promise zeroed memory and a zero
        # tolerance / zero iteration count would silently mean "iterate to the
        # array cap on every pair" and "never iterate" respectively — neither of
        # which resembles the reference. Every hand-built Model (the GPU env
        # specs, test fixtures) gets MuJoCo's behaviour without knowing this
        # slot exists; `fields_build` overwrites both from `<option>`.
        self.meta.data[MODEL_META_IDX_CCD_TOLERANCE] = Scalar[Self.DTYPE](
            MJ_CCD_TOLERANCE
        )
        self.meta.data[MODEL_META_IDX_CCD_ITERATIONS] = Scalar[Self.DTYPE](
            MJ_CCD_ITERATIONS
        )
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
        self.pairs = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(Self.NPAIR * MODEL_PAIR_SIZE)
        )
        self.mesh_meta = TensorImpl[Self.DTYPE].alloc(
            MAX_GPU_MESHES * MODEL_MESH_META_SIZE
        )
        self.mesh_verts = TensorImpl[Self.DTYPE].alloc(_at_least_one(Self.NMESH_VERTS * 3))
        self.mesh_polys = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(Self.NMESH_POLY * MODEL_MESH_POLY_SIZE)
        )
        self.mesh_polyvert = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(Self.NMESH_POLYVERT)
        )
        self.mesh_polymap = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(Self.NMESH_POLYVERT)
        )
        self.mesh_vert_polymap = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(Self.NMESH_VERTS * 2)
        )
        self.mesh_vert_edgeadr = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(Self.NMESH_VERTS)
        )
        # ⚠⚠ -1 MEANS "NO GRAPH", AND ZERO WOULD NOT. Zero is a valid offset
        # into `mesh_edges`, so a vertex whose adjacency was never built would
        # silently borrow the FIRST mesh's neighbour list and collide against
        # unrelated vertices. That is not hypothetical: it is what
        # `test_plane_mesh_fields` does — it appends a synthetic tetrahedron
        # straight into the packed tensors, past everything `fields_build`
        # wrote — and it turned 1 contact into 3. `_plane_mesh_contacts` reads
        # -1 as MuJoCo reads `mesh_graphadr < 0` and takes the exhaustive
        # branch instead.
        for v in range(_at_least_one(Self.NMESH_VERTS)):
            self.mesh_vert_edgeadr.data[v] = Scalar[Self.DTYPE](-1)
        self.mesh_edges = TensorImpl[Self.DTYPE].alloc(Self.NMESH_EDGE)
        # Pre-filled with the terminator, not zero: a neighbour walk reads
        # until it sees -1, and zero is a VALID vertex index. The real guard
        # against an unbuilt graph is the -1 in `mesh_vert_edgeadr` above; this
        # is belt and braces for a run that is entered anyway.
        for k in range(Self.NMESH_EDGE):
            self.mesh_edges.data[k] = Scalar[Self.DTYPE](-1)

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
        self.pairs.upload(ctx)
        self.mesh_meta.upload(ctx)
        self.mesh_verts.upload(ctx)
        self.mesh_polys.upload(ctx)
        self.mesh_polyvert.upload(ctx)
        self.mesh_polymap.upload(ctx)
        self.mesh_vert_polymap.upload(ctx)
        self.mesh_vert_edgeadr.upload(ctx)
        self.mesh_edges.upload(ctx)

    # `load_from_model` (CPU `Model` -> record fill) was deleted at the G4
    # fields sunset — the spec-direct build is
    # `fields_build.build_model_fields_from_flat`.
