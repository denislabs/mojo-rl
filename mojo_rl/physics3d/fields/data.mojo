"""Per-field tensor container for batched simulation state (migration P1).

`Data` is the end-state replacement for the flat GPU state slab
`[BATCH, STATE_SIZE]`: every state region becomes its own `TensorImpl`
(host `List` + optional device buffer), owned by this struct. Kernels take
exactly the field tensors they touch as `LayoutTensor` operands (via
`t.lt["gpu", Self.L_*]()`), instead of one giant slab plus offset math.

Coexistence (P1..P5): the flat-slab path stays untouched and running; newly
ported kernels consume `Data`. The `load_from_slab` / `store_to_slab`
bridges convert between the two at pipeline boundaries and in A/B gates —
they are TRANSITIONAL and die with the slab at sunset (P6), taking the
`gpu/constants.mojo` offset imports below with them.

Field set is closed (this is physics, not an arbitrary module tree), so all
walking (`upload_all`, `download_all`, bridges) is hand-written — no
reflection. Batch layout is `[BATCH, F]` row-major per field (nn convention);
CPU single-env use is `BATCH=1`.

Dtype note: every field is `DTYPE` for bit-compatibility with the existing
slab encoding (which float-encodes the few int-ish state slots, e.g. the
contact `condim` column and the metadata counters). Per-field `int32`
upgrades happen after the code stops round-tripping through slabs.
"""

from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl

from ..gpu.constants import CONTACT_SIZE, METADATA_SIZE


struct Data[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int = 0,
    BATCH: Int = 1,
](Movable):
    """Batched per-field simulation state (the 17 regions of the flat state
    slab, one owned tensor each). See module docstring."""

    # Per-field view layouts ([BATCH, F] row-major).
    comptime L_QPOS = Layout.row_major(Self.BATCH, Self.NQ)
    comptime L_NV = Layout.row_major(Self.BATCH, Self.NV)  # qvel/qacc/qfrc/qfrc_actuator
    comptime L_B3 = Layout.row_major(Self.BATCH, Self.NBODY * 3)
    comptime L_B4 = Layout.row_major(Self.BATCH, Self.NBODY * 4)
    comptime L_B6 = Layout.row_major(Self.BATCH, Self.NBODY * 6)
    comptime L_B10 = Layout.row_major(Self.BATCH, Self.NBODY * 10)
    comptime L_CONTACTS = Layout.row_major(
        Self.BATCH, Self.MAX_CONTACTS * CONTACT_SIZE
    )
    comptime L_META = Layout.row_major(Self.BATCH, METADATA_SIZE)
    comptime L_SITE = Layout.row_major(Self.BATCH, Self.NSITE * 3)

    # Joint space
    var qpos: TensorImpl[Self.DTYPE]  # [BATCH, NQ]
    var qvel: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var qacc: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var qfrc: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    # World space (FK products)
    var xpos: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var xquat: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*4]
    var xipos: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var xvel: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var xangvel: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    # Contacts (packed record columns = CONTACT_IDX_*) + per-env metadata
    var contacts: TensorImpl[Self.DTYPE]  # [BATCH, MAX_CONTACTS*CONTACT_SIZE]
    var meta: TensorImpl[Self.DTYPE]  # [BATCH, METADATA_SIZE]
    # Derived / auxiliary
    var site_xpos: TensorImpl[Self.DTYPE]  # [BATCH, NSITE*3]
    var cfrc_ext: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var cvel: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var cinert: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*10]
    var subtree_com: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var qfrc_actuator: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    # Mocap body targets (world frame; hook-written, FK skips mocap bodies —
    # the facades preset the mocap body pose from these before each step)
    var mocap_pos: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var mocap_quat: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*4]

    def __init__(out self) raises:
        comptime B = Self.BATCH
        self.qpos = TensorImpl[Self.DTYPE].alloc(B * Self.NQ)
        self.qvel = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.qacc = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.qfrc = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.xpos = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 3)
        self.xquat = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 4)
        self.xipos = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 3)
        self.xvel = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 3)
        self.xangvel = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 3)
        self.contacts = TensorImpl[Self.DTYPE].alloc(
            B * Self.MAX_CONTACTS * CONTACT_SIZE
        )
        self.meta = TensorImpl[Self.DTYPE].alloc(B * METADATA_SIZE)
        self.site_xpos = TensorImpl[Self.DTYPE].alloc(B * Self.NSITE * 3)
        self.cfrc_ext = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 6)
        self.cvel = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 6)
        self.cinert = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 10)
        self.subtree_com = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 3)
        self.qfrc_actuator = TensorImpl[Self.DTYPE].alloc(B * Self.NV)
        self.mocap_pos = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 3)
        self.mocap_quat = TensorImpl[Self.DTYPE].alloc(B * Self.NBODY * 4)

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Host -> device for every field (creates/replaces device buffers;
        NOT capture-safe — use per-field `upload_resident` under capture)."""
        self.qpos.upload(ctx)
        self.qvel.upload(ctx)
        self.qacc.upload(ctx)
        self.qfrc.upload(ctx)
        self.xpos.upload(ctx)
        self.xquat.upload(ctx)
        self.xipos.upload(ctx)
        self.xvel.upload(ctx)
        self.xangvel.upload(ctx)
        self.contacts.upload(ctx)
        self.meta.upload(ctx)
        if Self.NSITE > 0:
            self.site_xpos.upload(ctx)
        self.cfrc_ext.upload(ctx)
        self.cvel.upload(ctx)
        self.cinert.upload(ctx)
        self.subtree_com.upload(ctx)
        self.qfrc_actuator.upload(ctx)
        self.mocap_pos.upload(ctx)
        self.mocap_quat.upload(ctx)

    def download_all(mut self, ctx: DeviceContext) raises:
        """Device -> host for every field."""
        self.qpos.download(ctx)
        self.qvel.download(ctx)
        self.qacc.download(ctx)
        self.qfrc.download(ctx)
        self.xpos.download(ctx)
        self.xquat.download(ctx)
        self.xipos.download(ctx)
        self.xvel.download(ctx)
        self.xangvel.download(ctx)
        self.contacts.download(ctx)
        self.meta.download(ctx)
        if Self.NSITE > 0:
            self.site_xpos.download(ctx)
        self.cfrc_ext.download(ctx)
        self.cvel.download(ctx)
        self.cinert.download(ctx)
        self.subtree_com.download(ctx)
        self.qfrc_actuator.download(ctx)
        self.mocap_pos.download(ctx)
        self.mocap_quat.download(ctx)

