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

from max.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl

from .dims import DimsLike

from ..gpu.constants import CONTACT_SIZE, METADATA_SIZE
from ..collision.ccd_workspace import CCD_WS_SIZE


struct Data[
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](Movable):
    """Batched per-field simulation state (the 17 regions of the flat state
    slab, one owned tensor each). See module docstring."""

    # Body unchanged — see `Rk4Scratch`. `Data` is named in 98 files and 311
    # of its spellings are in SIGNATURES, so keeping every `Self.NQ`/`Self.NV`
    # in the allocation and layout code spelled as it was is what holds this
    # to a mechanical change.
    comptime NQ = Self.D.NQ
    comptime NV = Self.D.NV
    comptime NBODY = Self.D.NBODY
    comptime MAX_CONTACTS = Self.D.MAX_CONTACTS
    comptime NSITE = Self.D.NSITE

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
    comptime L_CCD_WS = Layout.row_major(Self.BATCH, CCD_WS_SIZE)
    comptime L_SITE = Layout.row_major(Self.BATCH, Self.NSITE * 3)

    # Joint space
    var qpos: TensorImpl[Self.DTYPE]  # [BATCH, NQ]
    var qvel: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var qacc: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var qfrc: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    # THIS STEP's actuator damping diagonal — `-diag(d qfrc_actuator/d qvel)`.
    # ⚠ IT IS STATE-DEPENDENT AND `Model` CANNOT HOLD IT: MuJoCo's
    # `mjd_actuator_vel` skips any actuator whose force is CLAMPED by its
    # `forcerange`, and whether it is clamped changes every step. Guarded by
    # `META_IDX_ACTDAMP_LIVE`, which says whether this was filled.
    var dof_actdamp: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    # ⚠ THE SAME LIVE/BANKED SPLIT, FOR THE OFF-DIAGONAL. `Model.actdamp_trn`
    # holds each actuator's `(dof, gear*coef)` pairs and its `kv`, which are
    # model-time constants; this says WHICH ACTUATORS CONTRIBUTED THIS STEP —
    # 1.0 normally, 0.0 for one whose force `mjd_actuator_vel` skips because
    # `forcerange` clamped it. Written by `apply_actions_fields` beside
    # `dof_actdamp`, and read only when `META_IDX_ACTDAMP_LIVE` is up; with
    # the flag down every actuator counts, because nothing can be saturated
    # in a step that never actuated.
    var actdamp_act: TensorImpl[Self.DTYPE]  # [BATCH, NACT]
    # World space (FK products)
    var xpos: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var xquat: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*4]
    var xipos: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var xvel: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var xangvel: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    # Contacts (packed record columns = CONTACT_IDX_*) + per-env metadata
    var contacts: TensorImpl[Self.DTYPE]  # [BATCH, MAX_CONTACTS*CONTACT_SIZE]
    var meta: TensorImpl[Self.DTYPE]  # [BATCH, METADATA_SIZE]
    # ⚠ EPA'S POLYTOPE, AND IT IS DATA RATHER THAN A LOCAL BECAUSE MUJOCO'S IS.
    # `mjc_penetration` hands EPA a `config->buffer` carved out of mjData's
    # arena (or the thread-local `ccd_buffer`), never the C stack; ours used
    # the stack and the ~7.5 KB frame is what kept heightfield collision off
    # the GPU. One row per env, so it is thread-local in the collision kernels
    # by the same argument that makes `contacts` thread-local. Nothing reads it
    # across calls — it is pure scratch, never uploaded or downloaded.
    var ccd_ws: TensorImpl[Self.DTYPE]  # [BATCH, CCD_WS_SIZE]
    # Derived / auxiliary
    var hfield_data: TensorImpl[Self.DTYPE]  # [BATCH, NHFIELD_DATA]
    """The heightfield elevation grids, PER ENVIRONMENT.

    ⚠⚠ IN `Data` AND NOT IN `Model`, which is where MuJoCo keeps it. The grid
    is STATE for `quadruped escape`: `initialize_episode` rewrites the terrain
    on every reset, and in a batch the lanes reset at different times, so one
    shared grid would hand every environment whichever terrain reset last —
    silently, and in a way that looks like a correlated policy rather than a
    bug. `hfield_meta` (adr, nrow, ncol, sizes) is genuinely per-asset and
    stays in `Model`.

    ⚠ EVERY LANE IS FILLED AT BUILD TIME from the parsed asset, so a model
    whose terrain is a FILE behaves exactly as before and a model whose
    terrain is generated starts from the same zeros MuJoCo would.

    Indexed `[env * NHFIELD_DATA + hfield_adr + r * ncol + c]`. ⚠ `hfield_adr`
    is an offset within ONE environment's block, not a global one.
    """
    var site_xpos: TensorImpl[Self.DTYPE]  # [BATCH, NSITE*3]
    var cfrc_ext: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var cvel: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var cinert: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*10]
    # `mj_rnePostConstraint` products (torque:force, world-oriented, at the
    # subtree CoM of each body's kinematic root). Written ONLY by the
    # `compute_rne_post` stage, which an integrator runs when its
    # `RNE_POST` parameter is set — every other model leaves them zero.
    # They feed the acceleration-stage sensors (accelerometer, force,
    # torque); see physics3d/sensors/site_acc.mojo.
    var cacc: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var cfrc_int: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    # ⚠ THE FK PRODUCTS AS THEY STOOD WHEN `cacc`/`cfrc_int` WERE WRITTEN.
    #
    # An acceleration-stage sensor transports `cacc`/`cfrc_int` to a site and
    # rotates into the site frame, so it needs the site pose FROM THE SAME
    # INSTANT. MuJoCo gets that for free: it evaluates the stage before
    # integrating and stores the RESULT in `sensordata`. We evaluate lazily in
    # the observation hook, by which point `_fields_fk` has moved
    # `xpos`/`xquat`/`site_xpos` to the POST-integration state — which is
    # required for the position/velocity-stage dims and wrong for these.
    #
    # That mix was defect 19: dog's accelerometer read 1.484 where dm_control
    # reads -6.386, while `cacc` itself was exact to 4.5e-10 on a magnitude of
    # 13463. Rebuilding the sensor from the mixed fields reproduced our output
    # to 5.1e-13, which is how the cause was pinned rather than argued.
    #
    # `cvel` and `subtree_com` need no snapshot: nothing refreshes them after
    # the substep, so they are already the pre-integration values. These two
    # are the whole inconsistency.
    var site_xpos_acc: TensorImpl[Self.DTYPE]  # [BATCH, NSITE*3]
    var xquat_acc: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*4]
    var subtree_com: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var qfrc_actuator: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    # Mocap body targets (world frame; hook-written, FK skips mocap bodies —
    # the facades preset the mocap body pose from these before each step)
    var mocap_pos: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*3]
    var mocap_quat: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*4]

    # ⚠ THE PROVIDER AS A VALUE, not just as the type parameter `D` (3a).
    #
    # A dispatcher holds `D` as a comptime TYPE and has no value to build a
    # `RuntimeLayout` from; `AsStatic[D]()` re-spells one, but only on the
    # static leg — on a dynamic provider it expands to `Dims[nq=-1, …]`. The
    # dimensions of a runtime model are runtime DATA, so they have to be
    # STORED, and the container that owns the buffers is where they belong.
    # Every dispatcher already takes `d`, so `d.dims` reaches essentially all
    # of them without threading 206 arguments (§15.3).
    #
    # Costs nothing on the static leg: `Dims` is stateless, so this field is
    # zero-size and `dm.get_nv()` folds to the same immediate a comptime
    # parameter would (§10.6 read the asm).
    var dims: Self.D

    def __init__(out self) raises:
        """The shipped constructor — dimensions from the comptime provider.

        ⚠ RAISES ON A DYNAMIC PROVIDER, deliberately. See
        `DimsLike.comptime_value`: `DynDims` has no comptime value, so this
        fails at construction naming the fix rather than allocating nothing.
        """
        self = Self(Self.D.comptime_value())

    def __init__(out self, dims: Self.D) raises:
        """Dimensions passed in, and ALLOCATED FROM (3b).

        ⚠ Every size below reads `dims`, never a comptime member. Those
        members still exist and still size the GPU layouts, but they are
        `DIM_POISON` on a dynamic provider, so an `alloc` that read one would
        ask for a NEGATIVE length.

        The nullary constructor delegates here with
        `Self.D.comptime_value()`, whose dimensions are the same integers, so
        the static leg allocates exactly what it always did.
        """
        self.dims = dims
        comptime B = Self.BATCH
        self.qpos = TensorImpl[Self.DTYPE].alloc(B * dims.get_nq())
        self.qvel = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.qacc = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.qfrc = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.dof_actdamp = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        # ⚠ NEVER ZERO-LENGTH. A model with no actuators would otherwise
        # allocate 0 and any later bind of this tensor is a null view — the
        # trap `site_xpos` sprang on every site-less model.
        var _nact_alloc = B * dims.get_nact()
        if _nact_alloc < 1:
            _nact_alloc = 1
        self.actdamp_act = TensorImpl[Self.DTYPE].alloc(_nact_alloc)
        self.xpos = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 3)
        self.xquat = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 4)
        self.xipos = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 3)
        self.xvel = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 3)
        self.xangvel = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 3)
        self.contacts = TensorImpl[Self.DTYPE].alloc(
            B * dims.get_max_contacts() * CONTACT_SIZE
        )
        self.meta = TensorImpl[Self.DTYPE].alloc(B * METADATA_SIZE)
        self.ccd_ws = TensorImpl[Self.DTYPE].alloc(B * CCD_WS_SIZE)
        # ⚠ `_at_least_one`: a model with no heightfield must still allocate,
        # because `alloc(0)` is not a valid buffer and every kernel binds this
        # tensor whether the model uses it or not.
        var _hfn = dims.get_nhfield_data()
        if _hfn < 1:
            _hfn = 1
        self.hfield_data = TensorImpl[Self.DTYPE].alloc(B * _hfn)
        self.site_xpos = TensorImpl[Self.DTYPE].alloc(B * dims.get_nsite() * 3)
        self.cfrc_ext = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 6)
        self.cvel = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 6)
        self.cinert = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 10)
        self.cacc = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 6)
        self.cfrc_int = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 6)
        # Sized exactly like `site_xpos` above, including its lack of a
        # zero-extent guard: a model with NSITE == 0 never reaches a site
        # sensor, and diverging from the field it shadows would be its own bug.
        self.site_xpos_acc = TensorImpl[Self.DTYPE].alloc(B * dims.get_nsite() * 3)
        self.xquat_acc = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 4)
        self.subtree_com = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 3)
        self.qfrc_actuator = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.mocap_pos = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 3)
        self.mocap_quat = TensorImpl[Self.DTYPE].alloc(B * dims.get_nbody() * 4)

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Host -> device for every field (creates/replaces device buffers;
        NOT capture-safe — use per-field `upload_resident` under capture)."""
        self.hfield_data.upload(ctx)
        self.qpos.upload(ctx)
        self.qvel.upload(ctx)
        self.qacc.upload(ctx)
        self.qfrc.upload(ctx)
        self.dof_actdamp.upload(ctx)
        self.actdamp_act.upload(ctx)
        self.xpos.upload(ctx)
        self.xquat.upload(ctx)
        self.xipos.upload(ctx)
        self.xvel.upload(ctx)
        self.xangvel.upload(ctx)
        self.contacts.upload(ctx)
        self.meta.upload(ctx)
        # Scratch: the device buffer has to EXIST (the collision kernel binds
        # it), but its contents are written before they are read on every
        # call, so the host copy is never meaningful in either direction —
        # hence no matching `download`.
        self.ccd_ws.upload(ctx)
        if Self.NSITE > 0:
            self.site_xpos.upload(ctx)
        self.cfrc_ext.upload(ctx)
        self.cvel.upload(ctx)
        self.cinert.upload(ctx)
        self.cacc.upload(ctx)
        self.site_xpos_acc.upload(ctx)
        self.xquat_acc.upload(ctx)
        self.cfrc_int.upload(ctx)
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
        self.cacc.download(ctx)
        self.site_xpos_acc.download(ctx)
        self.xquat_acc.download(ctx)
        self.cfrc_int.download(ctx)
        self.subtree_com.download(ctx)
        self.qfrc_actuator.download(ctx)
        self.mocap_pos.download(ctx)
        self.mocap_quat.download(ctx)


