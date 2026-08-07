"""GPU-batched physics3d environment on the PER-FIELD tensor path.

`Phyics3dBatchedEnv[MODEL_DEF, CONFIG, N_ENVS]` is the fields-path
counterpart of `BatchedGpuEnv[Phyics3dEnv[...]]`: it implements the
`BatchedEnv` driver ABI directly as a STATEFUL struct (it owns
`Data` / `Model` / `RK4Integrator` plus the driver IO
buffers), so the off-policy GPU drivers (`run_offpolicy_train_batched`,
i.e. `SACAgent.train[...]` etc.) train against fields-path physics with
zero driver changes.

Model: fully on the fields path — the model is built offset-free via
`MODEL_DEF.init_fields` (spec-direct, offset-free),
and the three former model-slab consumers now read `Model` directly:
reset FK -> `forward_kinematics`, `compute_cfrc_ext` (reads
`mf.bodies`), and the reward hook's curriculum params (fed `mf.curriculum`
as a `[1, MODEL_CURRICULUM_SIZE]` view with offset 0 — the CONFIG hook is
generic over MODEL_SIZE + curriculum_offset, so it is UNCHANGED).

G5: the state slab is GONE. Every GPU hook (`MODEL_DEF.reset_env_gpu` /
`apply_actions_kernel_gpu` / `extract_obs_gpu` and the CONFIG
`pre_step_gpu` / `init_qpos_gpu` / `custom_extract_obs_gpu` /
`compute_reward_and_done_gpu`) takes the per-field tensors it needs, and
`compute_cfrc_ext` / `compute_cvel` read/write the Data
cfrc_ext/cvel tensors directly. Hook state (step_count / prev_x) lives in
`d.meta` alongside num_contacts (detection writes ONLY the num_contacts
slot). The hook arithmetic is unchanged — only the addressing moved from
slab+offset to field tensors.

Scope (mirrors `Phyics3dEnv`): contacts + joint limits via the
per-stage RK4 PGS/Newton solve — hopper/walker-class locomotion in scope;
fluid-force models (Swimmer) run via the integrators' passive seam
(Stage A). Mesh-collision models are not in scope.

"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.core.target_storage import require_ctx

from mojo_rl.deep_agents.training.batched_env import BatchedEnv

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.integrator.rk4 import RK4Integrator
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.gpu import compute_cfrc_ext, compute_cvel
from mojo_rl.physics3d.gpu.constants import (
    TPB,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    METADATA_SIZE,
    META_IDX_STEP_COUNT,
    MODEL_CURRICULUM_SIZE,
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


# ──────────────────────────────────────────────────────────────────────
# The batched fields env
# ──────────────────────────────────────────────────────────────────────


struct Phyics3dBatchedEnv[
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
    comptime BLOCKS: Int = (Self.N_ENVS + TPB - 1) // TPB

    # Fields path (the actual physics state)
    var d: Data[
        DT, Self.NQ, Self.NV, Self.NBODY, Self.MC, Self.NSITE, Self.N_ENVS
    ]
    var mf: Model[
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
    comptime IntegRK4 = RK4Integrator[
        DT, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT, Self.MC, Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY, Self.MODEL_DEF.MAX_TENDON, Self.NSITE,
        Self.MODEL_DEF.NEXCLUDE, 0, Self.MODEL_DEF.CONE_TYPE, Self.N_ENVS,
        SOLVER = Self.SOLVER, PARALLEL_GPU = Self.PARALLEL_GPU,
        CRBA_TREEWALK = Self.CRBA_TREEWALK,
    ]
    comptime IntegEuler = EulerIntegrator[
        DT, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT, Self.MC, Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY, Self.MODEL_DEF.MAX_TENDON, Self.NSITE,
        Self.MODEL_DEF.NEXCLUDE, 0, Self.MODEL_DEF.CONE_TYPE, Self.N_ENVS,
        SOLVER = Self.SOLVER, PARALLEL_GPU = Self.PARALLEL_GPU,
        CRBA_TREEWALK = Self.CRBA_TREEWALK,
    ]
    var integ_rk4: Self.IntegRK4
    var integ_euler: Self.IntegEuler

    # Driver IO (env-owns-buffers per the BatchedEnv ABI)
    var _obs: DeviceBuffer[DT]
    var _action: DeviceBuffer[DT]
    var _reward: DeviceBuffer[DT]
    var _done: DeviceBuffer[DT]
    var _terminated: DeviceBuffer[DT]
    var _env_rng_counter: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext) raises:
        # ⚠ A CONFIG without GPU hooks inherits `compute_reward_and_done_gpu`'s
        # INERT DEFAULT and trains against a flat-zero reward curve — it compiles,
        # it runs, it reports episodes, and it learns nothing. Every dm_control
        # task config was in exactly that state for months (gap G10). Refuse the
        # instantiation instead. See Phyics3dEnvConfig.HAS_GPU_HOOKS.
        comptime assert Self.CONFIG.HAS_GPU_HOOKS, (
            "Phyics3dBatchedEnv: this CONFIG does not implement the GPU hooks"
            " (HAS_GPU_HOOKS is False), so its reward would be a constant 0 and"
            " its observation the model default. Implement"
            " compute_reward_and_done_gpu (+ custom_extract_obs_gpu if the model"
            " default is not the task's observation), then set HAS_GPU_HOOKS ="
            " True. See docs/DM_CONTROL_GPU_TRAINING_G10.md."
        )
        # Offset-free fields-native model build: init_fields runs
        # the spec-direct fields build and uploads every record
        # tensor (bodies/joints/meta/curriculum/…) — the reset FK, cfrc_ext,
        # and reward-curriculum hooks now read those directly.
        self.mf = type_of(self.mf)()
        Self.MODEL_DEF.init_fields[DT, 0](ctx, self.mf)
        # Fluid forces (density/viscosity) are handled by the fields
        # integrators' passive seam (Stage A: compute_fluid_forces), so
        # no guard — Swimmer runs on this facade.

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

    # ── kinematics ────────────────────────────────────────────────────

    def _run_fields_fk(mut self, c: DeviceContext) raises:
        """Fields FK over the whole batch (mf -> Data xpos/xquat/xipos
        [+ site_xpos]). Replaces the legacy slab `forward_kinematics_gpu` in
        the reset paths so reset no longer reads the model slab."""
        forward_kinematics[
            "gpu", DT, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT, Self.MC,
            Self.NGEOM, Self.MODEL_DEF.MAX_EQUALITY,
            Self.MODEL_DEF.MAX_TENDON, Self.NSITE, Self.MODEL_DEF.NEXCLUDE, 0,
            Self.N_ENVS,
        ](self.d, self.mf, c)

    def _run_fields_vel(mut self, c: DeviceContext) raises:
        """Body world velocities (xvel/xangvel) over the batch, from the
        current qvel. Companion to `_run_fields_fk` — the integrators compute
        these mid-step, so hooks reading them after integration need a
        refresh. Mirrors `Phyics3dEnv._fields_vel`."""
        compute_body_velocities[
            "gpu", DT, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT, Self.MC,
            Self.NGEOM, Self.MODEL_DEF.MAX_EQUALITY,
            Self.MODEL_DEF.MAX_TENDON, Self.NSITE, Self.MODEL_DEF.NEXCLUDE, 0,
            Self.N_ENVS,
        ](self.d, self.mf, c)

    # ── hook kernels (legacy Phyics3dEnv GPU code, verbatim) ──────────

    def _extract_obs_only(mut self, c: DeviceContext) raises:
        """Obs from the field tensors: CONFIG custom extraction else
        MODEL_DEF default qpos[skip:]+qvel."""

        @parameter
        @always_inline
        def obs_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            xpos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            xquat: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 4),
                MutAnyOrigin,
            ],
            xvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            bodies: LayoutTensor[
                DT,
                Layout.row_major(Self.NBODY, MODEL_BODY_SIZE),
                MutAnyOrigin,
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
                DT, Self.N_ENVS, Self.NQ, Self.NV, Self.NBODY, Self.OBS_DIM
            ](qpos, qvel, xpos, xquat, xvel, bodies, obs, env):
                Self.MODEL_DEF.extract_obs_gpu[
                    DT, Self.N_ENVS, Self.OBS_DIM
                ](qpos, qvel, obs, env)

        var obs_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS_DIM)
        ](self._obs)
        c.enqueue_function[obs_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.xpos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xquat.lt["gpu", type_of(self.d).L_B4](),
            self.d.xvel.lt["gpu", type_of(self.d).L_B3](),
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
            obs_t,
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )

    def _extract_obs_rewards_dones(mut self, c: DeviceContext) raises:
        """Step-count bump + obs + CONFIG reward/termination from the field
        tensors (hook arithmetic unchanged from the slab era)."""

        @parameter
        @always_inline
        def extract_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            xpos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            xipos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            xquat: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 4),
                MutAnyOrigin,
            ],
            xvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            bodies: LayoutTensor[
                DT,
                Layout.row_major(Self.NBODY, MODEL_BODY_SIZE),
                MutAnyOrigin,
            ],
            cfrc_ext: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            cvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
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
                rebind[Scalar[DT]](meta[env, META_IDX_STEP_COUNT])
            )
            step_count += 1
            meta[env, META_IDX_STEP_COUNT] = Scalar[DT](step_count)

            if not Self.CONFIG.custom_extract_obs_gpu[
                DT, Self.N_ENVS, Self.NQ, Self.NV, Self.NBODY, Self.OBS_DIM
            ](qpos, qvel, xpos, xquat, xvel, bodies, obs, env):
                Self.MODEL_DEF.extract_obs_gpu[
                    DT, Self.N_ENVS, Self.OBS_DIM
                ](qpos, qvel, obs, env)

            var result = Self.CONFIG.compute_reward_and_done_gpu[
                DT, Self.N_ENVS, Self.NQ, Self.NV, Self.NBODY, Self.ACT_DIM
            ](
                qpos,
                qvel,
                xpos,
                xipos,
                xquat,
                xvel,
                bodies,
                cfrc_ext,
                cvel,
                meta,
                curriculum,
                actions,
                env,
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
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.xpos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xquat.lt["gpu", type_of(self.d).L_B4](),
            self.d.xvel.lt["gpu", type_of(self.d).L_B3](),
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
            self.d.cfrc_ext.lt["gpu", type_of(self.d).L_B6](),
            self.d.cvel.lt["gpu", type_of(self.d).L_B6](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
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
            "Phyics3dBatchedEnv: reset_batch BATCH must match"
            " N_ENVS"
        )
        var c = require_ctx["Phyics3dBatchedEnv.reset_batch"](ctx)

        # Reset every lane on the field tensors (joint noise + CONFIG qpos +
        # hook metadata), then FK for the reset observation.
        @parameter
        @always_inline
        def reset_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            qacc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            qfrc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
            joints: LayoutTensor[
                DT,
                Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE),
                MutAnyOrigin,
            ],
            seed: Int,
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= Self.N_ENVS:
                return
            Self._reset_env_lane(qpos, qvel, qacc, qfrc, meta, joints, i, seed)

        c.enqueue_function[reset_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.qacc.lt["gpu", type_of(self.d).L_NV](),
            self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            self.mf.joints.lt["gpu", type_of(self.mf).L_JOINT](),
            Int(rng_seed),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        self._run_fields_fk(c)
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
            "Phyics3dBatchedEnv: step_batch BATCH must match N_ENVS"
        )
        _ = rng_seed
        var c = require_ctx["Phyics3dBatchedEnv.step_batch"](ctx)

        # 1) CONFIG pre-step hook (save prev_x etc. into d.meta).
        @parameter
        @always_inline
        def pre_step_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            Self.CONFIG.pre_step_gpu[DT, Self.N_ENVS, Self.NQ](
                qpos, meta, env
            )

        c.enqueue_function[pre_step_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 1 pre_step ok")

        # 2) Actions -> qfrc via the comptime actuator logic (field tensor).
        var actions_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.ACT_DIM)
        ](self._action)
        Self.MODEL_DEF.apply_actions_kernel_gpu[
            DT, Self.N_ENVS, Self.ACT_DIM
        ](
            c,
            self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
            rebind[
                LayoutTensor[
                    DT,
                    Layout.row_major(Self.N_ENVS, Self.ACT_DIM),
                    MutAnyOrigin,
                ]
            ](actions_t),
        )
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 2 apply_actions ok")

        # 3) Physics: fields integrator (RK4 or Euler per CONFIG.INTEGRATOR)
        #    with per-substep contact/limit solving.
        for _ in range(Self.CONFIG.FRAME_SKIP):
            comptime if Self.CONFIG.INTEGRATOR == "euler":
                self.integ_euler.step["gpu"](self.d, self.mf, ctx)
            else:
                self.integ_rk4.step["gpu"](self.d, self.mf, ctx)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 4 physics step ok")

        # 3b) Put the FK products and body velocities in sync with the
        #     INTEGRATED qpos/qvel before anything derived is computed from
        #     them. Both integrators run FK -> body velocities -> subtree_com
        #     at the START of each substep, so after the frame-skip loop
        #     xpos/xquat/xipos/site_xpos/xvel describe the state BEFORE the
        #     last substep. That is raw `mj_step` — right for the Gym-derived
        #     envs and wrong for dm_control, whose tasks read mjData in sync
        #     with qpos/qvel. See Phyics3dEnvConfig.SYNC_FK_AFTER_STEP.
        #
        # ⚠ This was missing on the batched path entirely until 2026-08-06 —
        #     `Phyics3dEnv` (single-env, CPU) honoured the flag and this file
        #     did not, so a suite config ported to the GPU hooks would have
        #     produced rewards one control step stale WITH ITS CPU GATE STILL
        #     PASSING. See docs/DM_CONTROL_GPU_TRAINING_G10.md §4.
        #
        # Comptime-gated, so the Gym envs pay nothing. Both calls are
        # deterministic and RNG-free, so they stay inside a USE_ENV_CUDA_GRAPH
        # capture safely.
        comptime if Self.CONFIG.SYNC_FK_AFTER_STEP:
            self._run_fields_fk(c)
            self._run_fields_vel(c)
            comptime if DEBUG:
                c.synchronize()
                print("[step_batch] 4b sync_fk_after_step ok")

        # 4) Derived quantities the reward hooks may read (cfrc_ext / cvel),
        #    straight on the field tensors.
        compute_cfrc_ext[DT, Self.N_ENVS, Self.NBODY, Self.MC](
            c,
            self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
            self.d.contacts.lt["gpu", type_of(self.d).L_CONTACTS](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            self.d.cfrc_ext.lt["gpu", type_of(self.d).L_B6](),
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
        )
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 5b compute_cfrc_ext ok")
        compute_cvel[DT, Self.N_ENVS, Self.NBODY](
            c,
            self.d.xpos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xvel.lt["gpu", type_of(self.d).L_B3](),
            self.d.xangvel.lt["gpu", type_of(self.d).L_B3](),
            self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
            self.d.cvel.lt["gpu", type_of(self.d).L_B6](),
        )
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
            "Phyics3dBatchedEnv: selective_reset_batch BATCH must"
            " match N_ENVS"
        )
        _ = rng_seed  # device counter drives reset randomness (capture-safe)
        var c = require_ctx[
            "Phyics3dBatchedEnv.selective_reset_batch"
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
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            qacc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            qfrc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            joints: LayoutTensor[
                DT,
                Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE),
                MutAnyOrigin,
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
                    qpos,
                    qvel,
                    qacc,
                    qfrc,
                    meta,
                    joints,
                    i,
                    Int(rebind[Scalar[DType.uint64]](counter[0])),
                )
                dones[i] = Scalar[DT](0.0)

        var dones_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._done
        )
        c.enqueue_function[selective_reset_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.qacc.lt["gpu", type_of(self.d).L_NV](),
            self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            dones_t,
            self.mf.joints.lt["gpu", type_of(self.mf).L_JOINT](),
            cnt_t,
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        # FK for the batch (idempotent for the live lanes, whose state is
        # unchanged since the last step), then refresh obs so reset lanes
        # start their episode from the reset observation.
        self._run_fields_fk(c)
        self._extract_obs_only(c)

    @always_inline
    @staticmethod
    def _reset_env_lane(
        qpos: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
        ],
        qacc: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
        ],
        qfrc: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, METADATA_SIZE), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DT, Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """One env lane's reset on the field tensors (arithmetic verbatim
        from the slab era: joint reset noise + CONFIG qpos + hook metadata)."""
        var RESET_NOISE = Scalar[DT](Self.CONFIG.get_reset_noise())
        Self.MODEL_DEF.reset_env_gpu[DT, Self.N_ENVS](
            qpos, qvel, qacc, qfrc, env, RESET_NOISE, seed
        )
        Self.CONFIG.init_qpos_gpu[
            DT, Self.N_ENVS, Self.NQ, Self.NJOINT, Self.NV
        ](qpos, qvel, joints, env, seed)
        meta[env, META_IDX_STEP_COUNT] = Scalar[DT](0.0)
        Self.CONFIG.pre_step_gpu[DT, Self.N_ENVS, Self.NQ](qpos, meta, env)

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
