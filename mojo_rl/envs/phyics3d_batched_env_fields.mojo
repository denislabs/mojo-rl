"""GPU-batched physics3d environment on the PER-FIELD tensor path.

`Phyics3dBatchedEnvFields[MODEL_DEF, CONFIG, N_ENVS]` is the fields-path
counterpart of `BatchedGpuEnv[Phyics3dEnv[...]]`: it implements the
`BatchedEnv` driver ABI directly as a STATEFUL struct (it owns
`DataFields` / `ModelFields` / `RK4IntegratorFields` plus the driver IO
buffers), so the off-policy GPU drivers (`run_offpolicy_train_batched`,
i.e. `SACAgent.train[...]` etc.) train against fields-path physics with
zero driver changes.

Model: fully on the fields path — the model is built offset-free via
`MODEL_DEF.init_fields` (no model slab, no init_model_gpu / load_from_slab),
and the three former model-slab consumers now read `ModelFields` directly:
reset FK -> `forward_kinematics_fields`, `compute_cfrc_ext_fields` (reads
`mf.bodies`), and the reward hook's curriculum params (fed `mf.curriculum`
as a `[1, MODEL_CURRICULUM_SIZE]` view with offset 0 — the CONFIG hook is
generic over MODEL_SIZE + curriculum_offset, so it is UNCHANGED).

State bridge (transitional, dies at P6): ONE device state slab
`[N_ENVS, STATE_SIZE]` is kept as the adapter for the obs/reward GPU hooks
that still speak slab+offsets — `MODEL_DEF.reset_env_gpu` /
`apply_actions_kernel_gpu` / `extract_obs_gpu` and the CONFIG
`pre_step_gpu` / `init_qpos_gpu` / `compute_reward_and_done_gpu` hooks,
plus `compute_cvel_gpu`. The PHYSICS never touches the slab: per step, one
kernel copies qpos/qvel/qfrc slab->fields before the substeps and one
copies the stepped state (qpos/qvel/qacc + FK products + contacts)
fields->slab for the hooks. Zero changes to any MODEL_DEF or CONFIG; the
hook arithmetic is the legacy `Phyics3dEnv` GPU code verbatim, so given
bit-exact physics (gated in tests/physics3d/test_rk4_contacts_fields.mojo)
the whole env is bit-exact vs the legacy slab pipeline.

Scope (mirrors `Phyics3dEnvFields`): contacts + joint limits via the
per-stage RK4 PGS solve — hopper/walker-class locomotion in scope;
equality/tendon-constrained and mesh-collision models are not. Fluid
force models raise at construction.

Metadata split: the slab metadata keeps step_count / prev_x (hook
state); the fields `meta` tensor carries num_contacts (written by
detection) and is the source of truth synced INTO the slab each step.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.nn.core.target_storage import require_ctx

from mojo_rl.deep_agents.training.batched_env import BatchedEnv

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.integrator.rk4_fields import RK4IntegratorFields
from mojo_rl.physics3d.integrator.euler_fields import EulerIntegratorFields
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
)
from mojo_rl.physics3d.gpu import compute_cfrc_ext_fields, compute_cvel_gpu
from mojo_rl.physics3d.gpu.constants import (
    TPB,
    CONTACT_SIZE,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    META_IDX_STEP_COUNT,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    state_size,
    MODEL_CURRICULUM_SIZE,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    xquat_offset,
    xipos_offset,
    xvel_offset,
    xangvel_offset,
    contacts_offset,
    metadata_offset,
    site_xpos_offset,
    cfrc_ext_offset,
    cvel_offset,
)

from .phyics3d_env_config import Phyics3dEnvConfig


# ──────────────────────────────────────────────────────────────────────
# Sync kernels (transitional slab<->fields bridges; die at P6)
# ──────────────────────────────────────────────────────────────────────


def _rng_counter_bump_kernel(
    counter: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Bump the device-resident env-reset RNG counter by 1 (capture-safe
    reset randomness; mirrors the legacy `increment_env_rng_kernel`)."""
    if Int(thread_idx.x) == 0:
        counter[0] = counter[0] + UInt64(1)


def _slab_to_fields_kernel[
    dt: DType, NQ: Int, NV: Int, SS: Int, B: Int
](
    slab: LayoutTensor[dt, Layout.row_major(B, SS), MutAnyOrigin],
    qpos: LayoutTensor[dt, Layout.row_major(B, NQ), MutAnyOrigin],
    qvel: LayoutTensor[dt, Layout.row_major(B, NV), MutAnyOrigin],
    qfrc: LayoutTensor[dt, Layout.row_major(B, NV), MutAnyOrigin],
):
    """Hand the hook-side state (post reset / post apply_actions) to the
    physics fields. Identity copy for lanes the hooks did not touch."""
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= B:
        return
    comptime O_QPOS = qpos_offset[NQ, NV]()
    comptime O_QVEL = qvel_offset[NQ, NV]()
    comptime O_QFRC = qfrc_offset[NQ, NV]()
    for i in range(NQ):
        qpos[env, i] = rebind[Scalar[dt]](slab[env, O_QPOS + i])
    for i in range(NV):
        qvel[env, i] = rebind[Scalar[dt]](slab[env, O_QVEL + i])
        qfrc[env, i] = rebind[Scalar[dt]](slab[env, O_QFRC + i])


def _fields_to_slab_kernel[
    dt: DType, NQ: Int, NV: Int, NBODY: Int, MC: Int, SS: Int, B: Int
](
    slab: LayoutTensor[dt, Layout.row_major(B, SS), MutAnyOrigin],
    qpos: LayoutTensor[dt, Layout.row_major(B, NQ), MutAnyOrigin],
    qvel: LayoutTensor[dt, Layout.row_major(B, NV), MutAnyOrigin],
    qacc: LayoutTensor[dt, Layout.row_major(B, NV), MutAnyOrigin],
    xpos: LayoutTensor[dt, Layout.row_major(B, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[dt, Layout.row_major(B, NBODY * 4), MutAnyOrigin],
    xipos: LayoutTensor[dt, Layout.row_major(B, NBODY * 3), MutAnyOrigin],
    xvel: LayoutTensor[dt, Layout.row_major(B, NBODY * 3), MutAnyOrigin],
    xangvel: LayoutTensor[dt, Layout.row_major(B, NBODY * 3), MutAnyOrigin],
    contacts: LayoutTensor[
        dt, Layout.row_major(B, MC * CONTACT_SIZE), MutAnyOrigin
    ],
    meta: LayoutTensor[dt, Layout.row_major(B, METADATA_SIZE), MutAnyOrigin],
):
    """Publish the stepped physics state to the hook slab: joint state,
    FK products (obs/reward inputs), contact records + num_contacts (the
    cfrc_ext inputs). Slab step_count / prev_x metadata is hook-owned and
    left untouched."""
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= B:
        return
    comptime O_QPOS = qpos_offset[NQ, NV]()
    comptime O_QVEL = qvel_offset[NQ, NV]()
    comptime O_QACC = qacc_offset[NQ, NV]()
    comptime O_XPOS = xpos_offset[NQ, NV, NBODY]()
    comptime O_XQUAT = xquat_offset[NQ, NV, NBODY]()
    comptime O_XIPOS = xipos_offset[NQ, NV, NBODY]()
    comptime O_XVEL = xvel_offset[NQ, NV, NBODY]()
    comptime O_XANG = xangvel_offset[NQ, NV, NBODY]()
    comptime O_CON = contacts_offset[NQ, NV, NBODY]()
    comptime O_META = metadata_offset[NQ, NV, NBODY, MC]()
    for i in range(NQ):
        slab[env, O_QPOS + i] = rebind[Scalar[dt]](qpos[env, i])
    for i in range(NV):
        slab[env, O_QVEL + i] = rebind[Scalar[dt]](qvel[env, i])
        slab[env, O_QACC + i] = rebind[Scalar[dt]](qacc[env, i])
    for i in range(NBODY * 3):
        slab[env, O_XPOS + i] = rebind[Scalar[dt]](xpos[env, i])
        slab[env, O_XIPOS + i] = rebind[Scalar[dt]](xipos[env, i])
        slab[env, O_XVEL + i] = rebind[Scalar[dt]](xvel[env, i])
        slab[env, O_XANG + i] = rebind[Scalar[dt]](xangvel[env, i])
    for i in range(NBODY * 4):
        slab[env, O_XQUAT + i] = rebind[Scalar[dt]](xquat[env, i])
    for i in range(MC * CONTACT_SIZE):
        slab[env, O_CON + i] = rebind[Scalar[dt]](contacts[env, i])
    slab[env, O_META + META_IDX_NUM_CONTACTS] = rebind[Scalar[dt]](
        meta[env, META_IDX_NUM_CONTACTS]
    )


def _sites_to_slab_kernel[
    dt: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MC: Int,
    NSITE: Int,
    SS: Int,
    B: Int,
](
    slab: LayoutTensor[dt, Layout.row_major(B, SS), MutAnyOrigin],
    site_xpos: LayoutTensor[
        dt, Layout.row_major(B, NSITE * 3), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= B:
        return
    comptime O_SITE = site_xpos_offset[NQ, NV, NBODY, MC]()
    for i in range(NSITE * 3):
        slab[env, O_SITE + i] = rebind[Scalar[dt]](site_xpos[env, i])


# ──────────────────────────────────────────────────────────────────────
# The batched fields env
# ──────────────────────────────────────────────────────────────────────


struct Phyics3dBatchedEnvFields[
    MODEL_DEF: ModelDefLike,
    CONFIG: Phyics3dEnvConfig,
    N_ENVS: Int,
    TERMINATE_ON_UNHEALTHY: Bool = False,
    SOLVER: StaticString = "newton",
    PARALLEL_GPU: Bool = True,
    CRBA_TREEWALK: Bool = True,
](BatchedEnv):
    """GPU-batched MuJoCo env, physics on the per-field tensor path,
    driver IO via the `BatchedEnv` ABI. See module docstring.

    The physics defaults are the LEGACY PRODUCTION bundle
    (CONFIG.physics_substep_gpu = RK4 + Newton at STEP_THREADS=NV with
    RK4_PARALLEL_* on): SOLVER="newton", PARALLEL_GPU=True (cooperative
    _mt kernels, bit-exact vs serial), CRBA_TREEWALK=True (the
    production mass-matrix algorithm — tolerance-equal to dense at
    ~1e-8/eval, exactly as in legacy). Gates that need a bit-exact
    baseline against a serial legacy reference pin these params
    explicitly."""

    comptime ENV_TARGET: StaticString = "gpu"
    comptime OBS_DIM: Int = Self.MODEL_DEF.OBS_DIM
    comptime ACT_DIM: Int = Self.MODEL_DEF.ACTION_DIM

    comptime NQ: Int = Self.MODEL_DEF.NQ
    comptime NV: Int = Self.MODEL_DEF.NV
    comptime NBODY: Int = Self.MODEL_DEF.NBODY
    comptime NJOINT: Int = Self.MODEL_DEF.NJOINT
    comptime MC: Int = Self.MODEL_DEF.MAX_CONTACTS
    comptime NGEOM: Int = Self.MODEL_DEF.NGEOM
    comptime NSITE: Int = Self.MODEL_DEF.NSITE
    comptime SS: Int = state_size[
        Self.NQ, Self.NV, Self.NBODY, Self.MC, Self.NSITE
    ]()
    comptime BLOCKS: Int = (Self.N_ENVS + TPB - 1) // TPB

    # Fields path (the actual physics state)
    var d: DataFields[
        DT, Self.NQ, Self.NV, Self.NBODY, Self.MC, Self.NSITE, Self.N_ENVS
    ]
    var mf: ModelFields[
        DT,
        Self.NV,
        Self.NBODY,
        Self.NJOINT,
        Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY,
        Self.MODEL_DEF.MAX_TENDON,
        Self.NSITE,
        Self.MODEL_DEF.NEXCLUDE,
    ]
    # Both integrators are held; the step comptime-dispatches on
    # CONFIG.INTEGRATOR (HalfCheetah/Pusher/MetaWorld = Euler+Newton, the
    # other 9 envs = RK4+Newton). Only the SELECTED one is `prepare_gpu`'d, so
    # the unused one allocates NO device memory.
    comptime IntegRK4 = RK4IntegratorFields[
        DT, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT, Self.MC, Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY, Self.MODEL_DEF.MAX_TENDON, Self.NSITE,
        Self.MODEL_DEF.NEXCLUDE, 0, Self.MODEL_DEF.CONE_TYPE, Self.N_ENVS,
        SOLVER = Self.SOLVER, PARALLEL_GPU = Self.PARALLEL_GPU,
        CRBA_TREEWALK = Self.CRBA_TREEWALK,
    ]
    comptime IntegEuler = EulerIntegratorFields[
        DT, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT, Self.MC, Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY, Self.MODEL_DEF.MAX_TENDON, Self.NSITE,
        Self.MODEL_DEF.NEXCLUDE, 0, Self.MODEL_DEF.CONE_TYPE, Self.N_ENVS,
        SOLVER = Self.SOLVER, PARALLEL_GPU = Self.PARALLEL_GPU,
        CRBA_TREEWALK = Self.CRBA_TREEWALK,
    ]
    var integ_rk4: Self.IntegRK4
    var integ_euler: Self.IntegEuler

    # Hooks adapter (transitional): device state slab for the obs/reward ABI.
    var _slab: TensorImpl[DT]

    # Driver IO (env-owns-buffers per the BatchedEnv ABI)
    var _obs: DeviceBuffer[DT]
    var _action: DeviceBuffer[DT]
    var _reward: DeviceBuffer[DT]
    var _done: DeviceBuffer[DT]
    var _terminated: DeviceBuffer[DT]
    var _env_rng_counter: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext) raises:
        # Offset-free fields-native model build (no model slab, no
        # init_model_gpu / load_from_slab). init_fields runs
        # setup_model_and_data + load_from_model and uploads every record
        # tensor (bodies/joints/meta/curriculum/…) — the reset FK, cfrc_ext,
        # and reward-curriculum hooks now read those directly.
        self.mf = type_of(self.mf)()
        Self.MODEL_DEF.init_fields[DT, 0](ctx, self.mf)

        # Fluid guard once here (the fields integrators don't model it).
        if (
            self.mf.meta.data[MODEL_META_IDX_DENSITY] != 0
            or self.mf.meta.data[MODEL_META_IDX_VISCOSITY] != 0
        ):
            raise Error(
                "Phyics3dBatchedEnvFields: fluid forces not ported to the"
                " fields path yet"
            )

        self._slab = TensorImpl[DT].alloc(Self.N_ENVS * Self.SS)
        self._slab.upload(ctx)

        self.d = type_of(self.d)()
        self.d.upload_all(ctx)
        # Construct both (host scratch); prepare only the selected integrator so
        # the unused one allocates no device buffers.
        self.integ_rk4 = Self.IntegRK4()
        self.integ_euler = Self.IntegEuler()
        comptime if Self.CONFIG.INTEGRATOR == "euler":
            self.integ_euler.prepare_gpu(ctx)
        else:
            self.integ_rk4.prepare_gpu(ctx)

        self._obs = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.OBS_DIM)
        self._action = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.ACT_DIM
        )
        self._reward = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._done = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._terminated = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        ctx.enqueue_memset(self._obs, 0)
        ctx.enqueue_memset(self._action, 0)
        ctx.enqueue_memset(self._reward, 0)
        ctx.enqueue_memset(self._done, 0)
        ctx.enqueue_memset(self._terminated, 0)
        self._env_rng_counter = ctx.enqueue_create_buffer[DType.uint64](1)
        self._env_rng_counter.enqueue_fill(UInt64(42))

    # ── slab<->fields bridges ─────────────────────────────────────────

    def _sync_slab_to_fields(mut self, c: DeviceContext) raises:
        c.enqueue_function[
            _slab_to_fields_kernel[DT, Self.NQ, Self.NV, Self.SS, Self.N_ENVS]
        ](
            self._slab.lt["gpu", Layout.row_major(Self.N_ENVS, Self.SS)](),
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )

    def _sync_fields_to_slab(mut self, c: DeviceContext) raises:
        c.enqueue_function[
            _fields_to_slab_kernel[
                DT, Self.NQ, Self.NV, Self.NBODY, Self.MC, Self.SS,
                Self.N_ENVS,
            ]
        ](
            self._slab.lt["gpu", Layout.row_major(Self.N_ENVS, Self.SS)](),
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.qacc.lt["gpu", type_of(self.d).L_NV](),
            self.d.xpos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xquat.lt["gpu", type_of(self.d).L_B4](),
            self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xvel.lt["gpu", type_of(self.d).L_B3](),
            self.d.xangvel.lt["gpu", type_of(self.d).L_B3](),
            self.d.contacts.lt["gpu", type_of(self.d).L_CONTACTS](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if Self.NSITE > 0:
            c.enqueue_function[
                _sites_to_slab_kernel[
                    DT, Self.NQ, Self.NV, Self.NBODY, Self.MC, Self.NSITE,
                    Self.SS, Self.N_ENVS,
                ]
            ](
                self._slab.lt[
                    "gpu", Layout.row_major(Self.N_ENVS, Self.SS)
                ](),
                self.d.site_xpos.lt["gpu", type_of(self.d).L_SITE](),
                grid_dim=(Self.BLOCKS,),
                block_dim=(TPB,),
            )

    def _run_fields_fk(mut self, c: DeviceContext) raises:
        """Fields FK over the whole batch (mf -> DataFields xpos/xquat/xipos
        [+ site_xpos]). Replaces the legacy slab `forward_kinematics_gpu` in
        the reset paths so reset no longer reads the model slab."""
        forward_kinematics_fields[
            "gpu", DT, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT, Self.MC,
            Self.NGEOM, Self.MODEL_DEF.MAX_EQUALITY,
            Self.MODEL_DEF.MAX_TENDON, Self.NSITE, Self.MODEL_DEF.NEXCLUDE, 0,
            Self.N_ENVS,
        ](self.d, self.mf, c)

    # ── hook kernels (legacy Phyics3dEnv GPU code, verbatim) ──────────

    def _extract_obs_only(mut self, c: DeviceContext) raises:
        """Obs from the slab: CONFIG custom extraction else MODEL_DEF
        default qpos[skip:]+qvel (legacy `extract_obs_kernel_gpu`)."""
        comptime QPOS_OFF = qpos_offset[Self.NQ, Self.NV]()
        comptime QVEL_OFF = qvel_offset[Self.NQ, Self.NV]()
        comptime XPOS_OFF = xpos_offset[Self.NQ, Self.NV, Self.NBODY]()

        @parameter
        @always_inline
        def obs_kernel(
            states: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.SS), MutAnyOrigin
            ],
            obs: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.OBS_DIM),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            if not Self.CONFIG.custom_extract_obs_gpu[
                DT, Self.N_ENVS, Self.SS, Self.OBS_DIM
            ](states, obs, env, QPOS_OFF, QVEL_OFF, XPOS_OFF):
                Self.MODEL_DEF.extract_obs_gpu[
                    DT, Self.N_ENVS, Self.SS, Self.OBS_DIM
                ](states, obs, env)

        var obs_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS_DIM)
        ](self._obs)
        c.enqueue_function[obs_kernel](
            self._slab.lt["gpu", Layout.row_major(Self.N_ENVS, Self.SS)](),
            obs_t,
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )

    def _extract_obs_rewards_dones(mut self, c: DeviceContext) raises:
        """Step-count bump + obs + CONFIG reward/termination from the slab
        (legacy `_extract_obs_rewards_dones_gpu`, verbatim)."""
        comptime QPOS_OFF = qpos_offset[Self.NQ, Self.NV]()
        comptime QVEL_OFF = qvel_offset[Self.NQ, Self.NV]()
        comptime XPOS_OFF = xpos_offset[Self.NQ, Self.NV, Self.NBODY]()
        comptime XIPOS_OFF = xipos_offset[Self.NQ, Self.NV, Self.NBODY]()
        comptime META_OFF = metadata_offset[
            Self.NQ, Self.NV, Self.NBODY, Self.MC
        ]()
        comptime CFRC_EXT_OFF = cfrc_ext_offset[
            Self.NQ, Self.NV, Self.NBODY, Self.MC, Self.NSITE
        ]()
        comptime CVEL_OFF = cvel_offset[
            Self.NQ, Self.NV, Self.NBODY, Self.MC, Self.NSITE
        ]()

        @parameter
        @always_inline
        def extract_kernel(
            states: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.SS), MutAnyOrigin
            ],
            curriculum: LayoutTensor[
                DT, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.ACT_DIM),
                MutAnyOrigin,
            ],
            rewards: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            dones: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            terminated_out: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            obs: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.OBS_DIM),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return

            var step_count = Int(
                rebind[Scalar[DT]](
                    states[env, META_OFF + META_IDX_STEP_COUNT]
                )
            )
            step_count += 1
            states[env, META_OFF + META_IDX_STEP_COUNT] = Scalar[DT](
                step_count
            )

            if not Self.CONFIG.custom_extract_obs_gpu[
                DT, Self.N_ENVS, Self.SS, Self.OBS_DIM
            ](states, obs, env, QPOS_OFF, QVEL_OFF, XPOS_OFF):
                Self.MODEL_DEF.extract_obs_gpu[
                    DT, Self.N_ENVS, Self.SS, Self.OBS_DIM
                ](states, obs, env)

            # The reward hook is generic over MODEL_SIZE + curriculum_offset;
            # feed the packed ModelFields.curriculum tensor as a [1, K] view
            # with offset 0, so `model[0, curriculum_offset + k]` reads
            # curriculum[k] — no model slab, and the CONFIG hook is unchanged.
            var result = Self.CONFIG.compute_reward_and_done_gpu[
                DT, Self.N_ENVS, Self.SS, Self.ACT_DIM, MODEL_CURRICULUM_SIZE
            ](
                states,
                curriculum,
                actions,
                env,
                QPOS_OFF,
                XPOS_OFF,
                XIPOS_OFF,
                CFRC_EXT_OFF,
                CVEL_OFF,
                META_OFF,
                0,
                step_count,
                Self.CONFIG.FRAME_SKIP,
                Scalar[DT](Self.CONFIG.get_timestep()),
            )
            rewards[env] = result[0]

            var is_terminated = result[1]
            comptime if not Self.TERMINATE_ON_UNHEALTHY:
                is_terminated = False
            var truncated = step_count >= Self.CONFIG.MAX_STEPS

            if is_terminated or truncated:
                dones[env] = Scalar[DT](1.0)
            else:
                dones[env] = Scalar[DT](0.0)
            terminated_out[env] = Scalar[DT](
                1.0
            ) if is_terminated else Scalar[DT](0.0)

        var rewards_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._reward
        )
        var dones_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._done
        )
        var term_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._terminated
        )
        var obs_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS_DIM)
        ](self._obs)
        var actions_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.ACT_DIM)
        ](self._action)
        c.enqueue_function[extract_kernel](
            self._slab.lt["gpu", Layout.row_major(Self.N_ENVS, Self.SS)](),
            self.mf.curriculum.lt[
                "gpu", Layout.row_major(1, MODEL_CURRICULUM_SIZE)
            ](),
            actions_t,
            rewards_t,
            dones_t,
            term_t,
            obs_t,
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )

    # ── BatchedEnv ABI ────────────────────────────────────────────────

    def reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "Phyics3dBatchedEnvFields: reset_batch BATCH must match"
            " N_ENVS"
        )
        var c = require_ctx["Phyics3dBatchedEnvFields.reset_batch"](ctx)

        # Reset every lane on the slab (joint noise + CONFIG qpos + metadata).
        @parameter
        @always_inline
        def reset_kernel(
            states: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.SS), MutAnyOrigin
            ],
            seed: Int,
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= Self.N_ENVS:
                return
            Self._reset_env_lane(states, i, seed)

        c.enqueue_function[reset_kernel](
            self._slab.lt["gpu", Layout.row_major(Self.N_ENVS, Self.SS)](),
            Int(rng_seed),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        # Reset qpos/qvel -> fields, FK on the fields, publish the FK products
        # (xpos/xquat/xipos [+ sites]) back to the slab for the obs hooks.
        self._sync_slab_to_fields(c)
        self._run_fields_fk(c)
        self._sync_fields_to_slab(c)
        self._extract_obs_only(c)

    def step_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        # Trait-conforming entry; delegates to the instrumentable impl.
        self._step_impl[BATCH, False](ctx, rng_seed)

    def _step_impl[
        BATCH: Int, DEBUG: Bool = False
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "Phyics3dBatchedEnvFields: step_batch BATCH must match N_ENVS"
        )
        _ = rng_seed
        var c = require_ctx["Phyics3dBatchedEnvFields.step_batch"](ctx)
        comptime META_OFF = metadata_offset[
            Self.NQ, Self.NV, Self.NBODY, Self.MC
        ]()

        # 1) CONFIG pre-step hook (save prev_x etc. into slab metadata).
        @parameter
        @always_inline
        def pre_step_kernel(
            states: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.SS), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            Self.CONFIG.pre_step_gpu[DT, Self.N_ENVS, Self.SS](
                states, env, META_OFF
            )

        c.enqueue_function[pre_step_kernel](
            self._slab.lt["gpu", Layout.row_major(Self.N_ENVS, Self.SS)](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 1 pre_step ok")

        # 2) Actions -> qfrc via the comptime actuator logic (slab).
        var sbuf = self._slab.dev.value()
        Self.MODEL_DEF.apply_actions_kernel_gpu[
            DT, Self.N_ENVS, Self.SS, Self.ACT_DIM
        ](c, sbuf, self._action)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 2 apply_actions ok")

        # 3) Hand qpos/qvel (fresh after any reset) + qfrc to the fields.
        self._sync_slab_to_fields(c)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 3 slab_to_fields ok")

        # 4) Physics: fields integrator (RK4 or Euler per CONFIG.INTEGRATOR)
        #    with per-substep contact/limit solving.
        for _ in range(Self.CONFIG.FRAME_SKIP):
            comptime if Self.CONFIG.INTEGRATOR == "euler":
                self.integ_euler.step["gpu"](self.d, self.mf, ctx)
            else:
                self.integ_rk4.step["gpu"](self.d, self.mf, ctx)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 4 physics step ok")

        # 5) Publish stepped state to the slab for the hooks, then the
        #    derived quantities the reward hooks may read.
        self._sync_fields_to_slab(c)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 5a fields_to_slab ok")
        compute_cfrc_ext_fields[
            DT,
            Self.N_ENVS,
            Self.SS,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MC,
            Self.NSITE,
        ](c, sbuf, self.mf.bodies.dev.value())
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 5b compute_cfrc_ext ok")
        compute_cvel_gpu[
            DT,
            Self.N_ENVS,
            Self.SS,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.MC,
            Self.NSITE,
        ](c, sbuf)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 5c compute_cvel ok")

        # 6) Obs + reward + done/terminated (CONFIG hooks).
        self._extract_obs_rewards_dones(c)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 6 extract_obs_rewards_dones ok")

    def selective_reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "Phyics3dBatchedEnvFields: selective_reset_batch BATCH must"
            " match N_ENVS"
        )
        _ = rng_seed  # device counter drives reset randomness (capture-safe)
        var c = require_ctx[
            "Phyics3dBatchedEnvFields.selective_reset_batch"
        ](ctx)
        var cnt_t = LayoutTensor[DType.uint64, Layout.row_major(1)](
            self._env_rng_counter
        )
        c.enqueue_function[_rng_counter_bump_kernel](
            cnt_t, grid_dim=(1,), block_dim=(1,)
        )

        @parameter
        @always_inline
        def selective_reset_kernel(
            states: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.SS), MutAnyOrigin
            ],
            dones: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            counter: LayoutTensor[
                DType.uint64, Layout.row_major(1), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= Self.N_ENVS:
                return
            if dones[i] > Scalar[DT](0.5):
                Self._reset_env_lane(
                    states, i, Int(rebind[Scalar[DType.uint64]](counter[0]))
                )
                dones[i] = Scalar[DT](0.0)

        var dones_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._done
        )
        c.enqueue_function[selective_reset_kernel](
            self._slab.lt["gpu", Layout.row_major(Self.N_ENVS, Self.SS)](),
            dones_t,
            cnt_t,
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        # FK on the fields for the batch (idempotent for the live lanes, whose
        # fields state is unchanged since the last step), publish FK products
        # to the slab, then refresh obs so reset lanes start their episode from
        # the reset observation (identity for live lanes).
        self._sync_slab_to_fields(c)
        self._run_fields_fk(c)
        self._sync_fields_to_slab(c)
        self._extract_obs_only(c)

    @always_inline
    @staticmethod
    def _reset_env_lane(
        states: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.SS), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """One env lane's reset on the slab (legacy `_reset_env_gpu`,
        verbatim: joint reset noise + CONFIG qpos offsets + metadata)."""
        var RESET_NOISE = Scalar[DT](Self.CONFIG.get_reset_noise())
        Self.MODEL_DEF.reset_env_gpu[DT, Self.N_ENVS, Self.SS](
            states, env, RESET_NOISE, seed
        )
        comptime QPOS_OFF = qpos_offset[Self.NQ, Self.NV]()
        Self.CONFIG.init_qpos_gpu[DT, Self.N_ENVS, Self.SS](
            states, env, QPOS_OFF
        )
        comptime META_OFF = metadata_offset[
            Self.NQ, Self.NV, Self.NBODY, Self.MC
        ]()
        states[env, META_OFF + META_IDX_STEP_COUNT] = Scalar[DT](0.0)
        Self.CONFIG.pre_step_gpu[DT, Self.N_ENVS, Self.SS](
            states, env, META_OFF
        )

    # ── pointer accessors ─────────────────────────────────────────────

    def obs_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._obs.unsafe_ptr())

    def action_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._action.unsafe_ptr())

    def reward_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._reward.unsafe_ptr())

    def done_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._done.unsafe_ptr())

    def terminated_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._terminated.unsafe_ptr())
