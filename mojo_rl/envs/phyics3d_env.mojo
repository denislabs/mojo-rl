"""Generic MuJoCo Environment — shared logic for all MuJoCo-style environments.

Phyics3dEnv[MODEL_DEF, CONFIG] delegates everything to CONFIG:
  - Model setup, integrator choice, reward, termination, GPU model init
  - Obs extraction, reset, enforce limits (MODEL_DEF delegates to Joints)
  - Action application (MODEL_DEF delegates to Actuators)

The CONFIG has full access to physics state for reward/termination —
no hardcoded assumptions about which qpos indices matter.
"""

from std.collections import InlineArray

from std.memory import alloc
from std.random import random_float64
from mojo_rl.core import (
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
    State,
    Action,
    ObsState,
    ContAction,
)
from mojo_rl.render import Renderer2D
from mojo_rl.nn import dtype as gpu_dtype

# GPU imports
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

# Import physics engine
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from mojo_rl.physics3d.gpu.constants import (
    TPB,
    state_size,
    qpos_offset,
    qvel_offset,
    xpos_offset,
    xipos_offset,
    metadata_offset,
    model_size,
    model_size_with_invweight,
    integrator_workspace_size,
    META_IDX_STEP_COUNT,
    META_IDX_PREV_X,
    model_curriculum_offset,
    MODEL_CURRICULUM_SIZE,
    cfrc_ext_offset,
    cvel_offset,
)
from mojo_rl.physics3d.gpu import compute_cfrc_ext_gpu, compute_cvel_gpu

from mojo_rl.physics3d.model.model_renderer import ModelRenderer

from .phyics3d_env_config import Phyics3dEnvConfig

from mojo_rl.physics3d.model.model_def import ModelDefLike

# =============================================================================
# Generic Phyics3d Environment
# =============================================================================


struct Phyics3dEnv[
    MODEL_DEF: ModelDefLike,
    CONFIG: Phyics3dEnvConfig,
    DTYPE: DType where DTYPE.is_floating_point() = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
](
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
):
    """Generic MuJoCo environment parameterized by MODEL_DEF and CONFIG.

    MODEL_DEF provides: model setup, obs extraction, reset, enforce limits,
    action application.
    CONFIG provides: integrator, reward, termination, pre-step hooks.
    """

    # Trait type aliases
    comptime dtype = Self.DTYPE
    comptime StateType = ObsState[Self.MODEL_DEF.OBS_DIM]
    comptime ActionType = ContAction[Self.MODEL_DEF.ACTION_DIM]
    comptime NAME: String = "Physics3dEnv"

    # Layout constants
    comptime OBS_DIM: Int = Self.MODEL_DEF.OBS_DIM
    comptime ACTION_DIM: Int = Self.MODEL_DEF.ACTION_DIM

    # Physics layout constants
    comptime NQ: Int = Self.MODEL_DEF.NQ
    comptime NV: Int = Self.MODEL_DEF.NV
    comptime NUM_BODIES: Int = Self.MODEL_DEF.NBODY
    comptime NUM_JOINTS: Int = Self.MODEL_DEF.NJOINT
    comptime MAX_CONTACTS: Int = Self.MODEL_DEF.MAX_CONTACTS
    comptime NGEOM: Int = Self.MODEL_DEF.NGEOM
    comptime MAX_EQUALITY: Int = Self.MODEL_DEF.MAX_EQUALITY
    comptime CONE_TYPE: Int = Self.MODEL_DEF.CONE_TYPE
    comptime MAX_TENDON: Int = Self.MODEL_DEF.MAX_TENDON
    comptime NSITE: Int = Self.MODEL_DEF.NSITE

    # GPU state size (includes site_xpos when NSITE > 0)
    comptime STATE_SIZE: Int = state_size[
        Self.MODEL_DEF.NQ,
        Self.MODEL_DEF.NV,
        Self.MODEL_DEF.NBODY,
        Self.MODEL_DEF.MAX_CONTACTS,
        Self.MODEL_DEF.NSITE,
    ]()

    # Pre-allocated workspace sizes for step_kernel_gpu
    comptime STEP_WS_SHARED: Int = model_size_with_invweight[
        Self.MODEL_DEF.NBODY,
        Self.MODEL_DEF.NJOINT,
        Self.MODEL_DEF.NV,
        Self.MODEL_DEF.NGEOM,
        NEQUALITY=Self.MODEL_DEF.MAX_EQUALITY,
        NTENDON=Self.MODEL_DEF.MAX_TENDON,
        NSITE=Self.MODEL_DEF.NSITE,
    ]()
    comptime STEP_WS_PER_ENV: Int = integrator_workspace_size[
        Self.MODEL_DEF.NV, Self.MODEL_DEF.NBODY
    ]() + Self.MODEL_DEF.NV * Self.MODEL_DEF.NV + NewtonSolver.solver_workspace_size[
        Self.MODEL_DEF.NV, Self.MODEL_DEF.MAX_CONTACTS
    ]() + Self.CONFIG.INTEGRATOR_WS_EXTRA

    # Physics model and data
    var model: Model[
        Self.DTYPE,
        Self.MODEL_DEF.NQ,
        Self.MODEL_DEF.NV,
        Self.MODEL_DEF.NBODY,
        Self.MODEL_DEF.NJOINT,
        Self.MODEL_DEF.MAX_CONTACTS,
        Self.MODEL_DEF.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY,
        Self.MODEL_DEF.CONE_TYPE,
        Self.MAX_TENDON,
        Self.NSITE,
    ]
    var data: Data[
        Self.DTYPE,
        Self.MODEL_DEF.NQ,
        Self.MODEL_DEF.NV,
        Self.MODEL_DEF.NBODY,
        Self.MODEL_DEF.NJOINT,
        Self.MODEL_DEF.MAX_CONTACTS,
        Self.NSITE,
    ]

    # Environment parameters
    var max_steps: Int
    var current_step: Int
    var frame_skip: Int

    # Per-env persistent state (used by config's pre_step hook)
    var prev_x: Scalar[Self.DTYPE]

    # Renderer (optional)
    var _renderer: UnsafePointer[ModelRenderer[Self.MODEL_DEF], MutAnyOrigin]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        out self,
        max_steps: Int = Self.CONFIG.MAX_STEPS,
        frame_skip: Int = Self.CONFIG.FRAME_SKIP,
    ):
        self.max_steps = max_steps
        self.current_step = 0
        self.frame_skip = frame_skip
        self.prev_x = Scalar[Self.DTYPE](0.0)

        # Initialize model and data
        self.model = Model[
            Self.DTYPE,
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.NJOINT,
            Self.MODEL_DEF.MAX_CONTACTS,
            Self.MODEL_DEF.NGEOM,
            Self.MODEL_DEF.MAX_EQUALITY,
            Self.MODEL_DEF.CONE_TYPE,
            Self.MAX_TENDON,
            Self.NSITE,
        ]()
        self.data = Data[
            Self.DTYPE,
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.NJOINT,
            Self.MODEL_DEF.MAX_CONTACTS,
            Self.NSITE,
        ]()

        # Renderer not initialized
        self._renderer = UnsafePointer[
            ModelRenderer[Self.MODEL_DEF], MutAnyOrigin
        ]()
        self._renderer_initialized = False

        # Delegate full setup to config
        Self.MODEL_DEF.setup_model_and_data(self.model, self.data)

        # Initialize prev_x via config's pre_step hook
        self.current_step = 0
        Self.CONFIG.pre_step_cpu(self.data, self.prev_x)

    # =========================================================================
    # Physics State Management
    # =========================================================================

    def _reset_state(mut self):
        """Reset to initial position with uniform noise (matching Gymnasium)."""
        Self.MODEL_DEF.reset_data(self.data)

        # Add uniform noise to qpos and qvel (matches Gymnasium's reset)
        var noise_scale = Self.CONFIG.get_reset_noise()
        if noise_scale > 0.0:
            for i in range(Self.MODEL_DEF.NQ):
                var noise = Scalar[Self.dtype](
                    (random_float64() * 2.0 - 1.0) * noise_scale
                )
                self.data.qpos[i] = self.data.qpos[i] + noise
            for i in range(Self.MODEL_DEF.NV):
                var noise = Scalar[Self.dtype](
                    (random_float64() * 2.0 - 1.0) * noise_scale
                )
                self.data.qvel[i] = self.data.qvel[i] + noise

        # Custom reset (e.g., set mocap positions, warmup steps)
        Self.CONFIG.custom_reset_cpu(self.model, self.data)

        # Run forward kinematics to compute xpos/xquat
        forward_kinematics(self.model, self.data)

        # Reset step counter and prev_x
        self.current_step = 0
        self.prev_x = Scalar[Self.dtype](0)
        Self.CONFIG.pre_step_cpu(self.data, self.prev_x)

    def _get_obs(self) -> ObsState[Self.MODEL_DEF.OBS_DIM]:
        """Extract observation from current physics data."""
        var obs_list = List[Scalar[Self.DTYPE]](capacity=Self.MODEL_DEF.OBS_DIM)
        var custom = Self.CONFIG.custom_extract_obs_cpu(self.data, obs_list)
        if not custom:
            Self.MODEL_DEF.extract_obs(self.data, obs_list)
        var obs = ObsState[Self.MODEL_DEF.OBS_DIM]()
        for i in range(Self.MODEL_DEF.OBS_DIM):
            obs.data[i] = Float64(obs_list[i])
        return obs^

    # =========================================================================
    # BoxContinuousActionEnv Interface
    # =========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=Self.MODEL_DEF.OBS_DIM)
        var custom = Self.CONFIG.custom_extract_obs_cpu(self.data, obs)
        if not custom:
            Self.MODEL_DEF.extract_obs(self.data, obs)
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        self._reset_state()
        return self.get_obs_list()

    def obs_dim(self) -> Int:
        return Self.MODEL_DEF.OBS_DIM

    def action_dim(self) -> Int:
        return Self.MODEL_DEF.ACTION_DIM

    def action_low(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](Self.MODEL_DEF.CTRL_MIN)

    def action_high(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](Self.MODEL_DEF.CTRL_MAX)

    def step_continuous[
        DTYPE2: DType
    ](mut self, action: Scalar[DTYPE2]) -> Tuple[
        List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool
    ]:
        var actions = List[Scalar[DTYPE2]]()
        for _ in range(Self.MODEL_DEF.ACTION_DIM):
            actions.append(Scalar[DTYPE2](action))
        return self.step_continuous_vec[DTYPE2](actions)

    def step_continuous_vec[
        DTYPE2: DType
    ](
        mut self,
        action: List[Scalar[DTYPE2]],
        verbose: Bool = False,
    ) -> Tuple[
        List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool
    ]:
        # Convert to ContAction
        var act = ContAction[Self.MODEL_DEF.ACTION_DIM]()
        for i in range(min(Self.MODEL_DEF.ACTION_DIM, len(action))):
            act.data[i] = Float64(action[i])

        # Take step
        var result = self.step(act, verbose=verbose)

        # Build observation list (use custom extraction if available)
        var obs_list = List[Scalar[Self.DTYPE]](capacity=Self.MODEL_DEF.OBS_DIM)
        var custom = Self.CONFIG.custom_extract_obs_cpu(self.data, obs_list)
        if not custom:
            Self.MODEL_DEF.extract_obs(self.data, obs_list)
        var obs = List[Scalar[DTYPE2]](capacity=Self.MODEL_DEF.OBS_DIM)
        for i in range(Self.MODEL_DEF.OBS_DIM):
            obs.append(Scalar[DTYPE2](obs_list[i]))

        return (obs^, Scalar[DTYPE2](result[1]), result[2])

    # =========================================================================
    # Env Interface
    # =========================================================================

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        # Pre-step: let config save whatever it needs
        Self.CONFIG.pre_step_cpu(self.data, self.prev_x)

        # Apply actions via actuators (config can override for mocap control etc.)
        # Per-motor ctrlrange clamping now happens inside apply_actions() itself
        # (ModelDefFromXML uses per-motor _acd.motor_ctrl_min/max).
        var clamped_action = action.copy()
        var action_list = clamped_action.to_list()
        var custom_applied = Self.CONFIG.custom_apply_actions_cpu(
            self.data, action_list
        )
        if not custom_applied:
            Self.MODEL_DEF.apply_actions(self.data, action_list)

        # Physics step (with frame skip)
        # Note: joint limits are handled by the soft constraint solver (same as
        # MuJoCo). Hard clamping qpos/qvel would corrupt the B*v_n damping term
        # in the limit constraint bias, causing incorrect contact dynamics.
        for _ in range(self.frame_skip):
            Self.CONFIG.physics_substep(self.model, self.data, verbose)

        # Run FK after integration so data.xpos matches the new qpos.
        # Without this, rendering shows the position from the START of the
        # last substep, not the corrected position after all integrations.
        forward_kinematics(self.model, self.data)

        self.current_step += 1

        # Compute reward and termination via config (full state access)
        var result = Self.CONFIG.compute_reward_and_done_cpu(
            self.data,
            self.prev_x,
            clamped_action.to_list(),
            self.current_step,
            self.frame_skip,
        )
        var reward = result[0]
        var terminated = result[1]

        comptime if not Self.TERMINATE_ON_UNHEALTHY:
            terminated = False

        var truncated = self.current_step >= self.max_steps
        var done = terminated or truncated

        return (self._get_obs(), Scalar[Self.dtype](reward), done)

    def get_state(self) -> Self.StateType:
        return self._get_obs()

    def reset(mut self) -> Self.StateType:
        self._reset_state()
        return self._get_obs()

    def render(mut self, mut renderer: Renderer2D):
        pass

    def close(mut self):
        pass

    # =========================================================================
    # State Accessors
    # =========================================================================

    def get_xpos(self, idx: Int) -> Scalar[Self.DTYPE]:
        """Get xpos element by flat index (for rendering)."""
        return self.data.xpos[idx]

    def get_xquat(self, idx: Int) -> Scalar[Self.DTYPE]:
        """Get xquat element by flat index (for rendering)."""
        return self.data.xquat[idx]

    def get_qpos(self, idx: Int) -> Scalar[Self.DTYPE]:
        """Get qpos element by index."""
        return self.data.qpos[idx]

    def get_qvel(self, idx: Int) -> Scalar[Self.DTYPE]:
        """Get qvel element by index."""
        return self.data.qvel[idx]

    def get_x_position(self) -> Scalar[Self.DTYPE]:
        return self.data.qpos[0]

    def get_x_velocity(self) -> Scalar[Self.DTYPE]:
        return self.data.qvel[0]

    def get_current_step(self) -> Int:
        return self.current_step

    def get_max_steps(self) -> Int:
        return self.max_steps

    def is_done(self) -> Bool:
        var truncated = self.current_step >= self.max_steps

        comptime if Self.TERMINATE_ON_UNHEALTHY:
            # Check termination via config
            var dummy_actions = List[Float64]()
            var result = Self.CONFIG.compute_reward_and_done_cpu(
                self.data,
                self.prev_x,
                dummy_actions,
                self.current_step,
                self.frame_skip,
            )
            return truncated or result[1]
        else:
            return truncated

    # =========================================================================
    # GPUContinuousEnv Interface (Static GPU Kernels)
    # =========================================================================

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
        ACTION_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut terminated_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[gpu_dtype]] = [],
        workspace_ptr: UnsafePointer[
            Scalar[gpu_dtype], MutAnyOrigin
        ] = UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin](),
        rng_counter_ptr: UnsafePointer[
            Scalar[DType.uint64], MutAnyOrigin
        ] = UnsafePointer[Scalar[DType.uint64], MutAnyOrigin](),
    ) raises:
        """Batched GPU step function using physics engine."""
        comptime MODEL_SIZE = model_size_with_invweight[
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.NJOINT,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NGEOM,
            NEQUALITY=Self.MODEL_DEF.MAX_EQUALITY,
            NTENDON=Self.MODEL_DEF.MAX_TENDON,
            NSITE=Self.MODEL_DEF.NSITE,
        ]()
        comptime WS_SIZE = integrator_workspace_size[
            Self.MODEL_DEF.NV, Self.MODEL_DEF.NBODY
        ]() + Self.MODEL_DEF.NV * Self.MODEL_DEF.NV + NewtonSolver.solver_workspace_size[
            Self.MODEL_DEF.NV, Self.MODEL_DEF.MAX_CONTACTS
        ]() + Self.CONFIG.INTEGRATOR_WS_EXTRA

        var model_buf: DeviceBuffer[gpu_dtype]
        var workspace_buf: DeviceBuffer[gpu_dtype]

        if workspace_ptr:
            model_buf = DeviceBuffer[gpu_dtype](
                ctx,
                workspace_ptr,
                MODEL_SIZE,
                owning=False,
            )
            workspace_buf = DeviceBuffer[gpu_dtype](
                ctx,
                workspace_ptr + MODEL_SIZE,
                BATCH_SIZE * WS_SIZE,
                owning=False,
            )
        else:
            model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
            Self.MODEL_DEF.init_model_gpu(ctx, model_buf)
            workspace_buf = ctx.enqueue_create_buffer[gpu_dtype](
                BATCH_SIZE * WS_SIZE
            )

        # Pre-step: let config save per-env state (e.g., prev_x)
        Self._pre_step_gpu[BATCH_SIZE, STATE_SIZE_VAL](ctx, states_buf)

        # Apply actions to qfrc via actuators
        Self.MODEL_DEF.apply_actions_kernel_gpu[
            gpu_dtype, BATCH_SIZE, STATE_SIZE_VAL, ACTION_DIM_VAL
        ](ctx, states_buf, actions_buf)

        # Run FRAME_SKIP physics sub-steps
        for _ in range(Self.CONFIG.FRAME_SKIP):
            Self.CONFIG.physics_substep_gpu[
                gpu_dtype,
                BATCH_SIZE,
                Self.MODEL_DEF.NQ,
                Self.MODEL_DEF.NV,
                Self.MODEL_DEF.NBODY,
                Self.MODEL_DEF.NJOINT,
                Self.MODEL_DEF.MAX_CONTACTS,
                Self.MODEL_DEF.NGEOM,
                Self.MODEL_DEF.MAX_EQUALITY,
                Self.MODEL_DEF.CONE_TYPE,
                Self.MAX_TENDON,
            ](ctx, states_buf, model_buf, workspace_buf)

        # Compute post-substep derived quantities: cfrc_ext, cvel
        compute_cfrc_ext_gpu[
            gpu_dtype,
            BATCH_SIZE,
            STATE_SIZE_VAL,
            MODEL_SIZE,
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.MAX_CONTACTS,
            Self.MODEL_DEF.NSITE,
        ](ctx, states_buf, model_buf)

        compute_cvel_gpu[
            gpu_dtype,
            BATCH_SIZE,
            STATE_SIZE_VAL,
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.MAX_CONTACTS,
            Self.MODEL_DEF.NSITE,
        ](ctx, states_buf)

        # Extract observations, compute rewards, check termination
        Self._extract_obs_rewards_dones_gpu[
            BATCH_SIZE,
            STATE_SIZE_VAL,
            MODEL_SIZE,
            OBS_DIM_VAL,
            Self.CONFIG.MAX_STEPS,
        ](
            ctx,
            states_buf,
            model_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
        )

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments on GPU.

        Combines reset + FK into a single kernel dispatch to avoid a Metal
        shader compilation bug where FK compiled as a separate closure inside
        a generic struct method produces incorrect code.
        """
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        comptime MODEL_SIZE = model_size_with_invweight[
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.NJOINT,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NGEOM,
            NEQUALITY=Self.MODEL_DEF.MAX_EQUALITY,
            NTENDON=Self.MODEL_DEF.MAX_TENDON,
            NSITE=Self.MODEL_DEF.NSITE,
        ]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        Self.MODEL_DEF.init_model_gpu(ctx, model_buf)

        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        def reset_with_fk_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            seed: Int,
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](states, i, seed)
            forward_kinematics_gpu[
                gpu_dtype,
                Self.MODEL_DEF.NQ,
                Self.MODEL_DEF.NV,
                Self.MODEL_DEF.NBODY,
                Self.MODEL_DEF.NJOINT,
                Self.MODEL_DEF.MAX_CONTACTS,
                STATE_SIZE_VAL,
                MODEL_SIZE,
                BATCH_SIZE,
            ](i, states, model)

        ctx.enqueue_function[reset_with_fk_wrapper, reset_with_fk_wrapper](
            states,
            model,
            Int(rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
        workspace_ptr: UnsafePointer[
            Scalar[gpu_dtype], MutAnyOrigin
        ] = UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin](),
        rng_counter_ptr: UnsafePointer[
            Scalar[DType.uint64], MutAnyOrigin
        ] = UnsafePointer[Scalar[DType.uint64], MutAnyOrigin](),
    ) raises:
        """Reset only done environments on GPU."""
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        comptime MODEL_SIZE = model_size_with_invweight[
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.NJOINT,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NGEOM,
            NEQUALITY=Self.MODEL_DEF.MAX_EQUALITY,
            NTENDON=Self.MODEL_DEF.MAX_TENDON,
            NSITE=Self.MODEL_DEF.NSITE,
        ]()

        # Reuse model from pre-allocated workspace if available,
        # otherwise allocate (backward compatible)
        var model_buf: DeviceBuffer[gpu_dtype]
        if workspace_ptr:
            model_buf = DeviceBuffer[gpu_dtype](
                ctx,
                workspace_ptr,
                MODEL_SIZE,
                owning=False,
            )
        else:
            model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
            Self.MODEL_DEF.init_model_gpu(ctx, model_buf)

        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        if rng_counter_ptr:
            var counter_t = LayoutTensor[
                DType.uint64, Layout.row_major(1), MutAnyOrigin
            ](rng_counter_ptr)

            @always_inline
            def selective_reset_with_fk_counter_wrapper(
                states: LayoutTensor[
                    gpu_dtype,
                    Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                    MutAnyOrigin,
                ],
                dones: LayoutTensor[
                    gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
                ],
                model: LayoutTensor[
                    gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
                ],
                counter: LayoutTensor[
                    DType.uint64, Layout.row_major(1), MutAnyOrigin
                ],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i >= BATCH_SIZE:
                    return
                if dones[i] > Scalar[gpu_dtype](0.5):
                    Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](
                        states, i, Int(rebind[Scalar[DType.uint64]](counter[0]))
                    )
                    forward_kinematics_gpu[
                        gpu_dtype,
                        Self.MODEL_DEF.NQ,
                        Self.MODEL_DEF.NV,
                        Self.MODEL_DEF.NBODY,
                        Self.MODEL_DEF.NJOINT,
                        Self.MODEL_DEF.MAX_CONTACTS,
                        STATE_SIZE_VAL,
                        MODEL_SIZE,
                        BATCH_SIZE,
                    ](i, states, model)
                    dones[i] = Scalar[gpu_dtype](0.0)

            ctx.enqueue_function[
                selective_reset_with_fk_counter_wrapper,
                selective_reset_with_fk_counter_wrapper,
            ](
                states,
                dones,
                model,
                counter_t,
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            def selective_reset_with_fk_wrapper(
                states: LayoutTensor[
                    gpu_dtype,
                    Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                    MutAnyOrigin,
                ],
                dones: LayoutTensor[
                    gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
                ],
                model: LayoutTensor[
                    gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
                ],
                seed: Int,
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i >= BATCH_SIZE:
                    return
                if dones[i] > Scalar[gpu_dtype](0.5):
                    Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](
                        states, i, seed
                    )
                    forward_kinematics_gpu[
                        gpu_dtype,
                        Self.MODEL_DEF.NQ,
                        Self.MODEL_DEF.NV,
                        Self.MODEL_DEF.NBODY,
                        Self.MODEL_DEF.NJOINT,
                        Self.MODEL_DEF.MAX_CONTACTS,
                        STATE_SIZE_VAL,
                        MODEL_SIZE,
                        BATCH_SIZE,
                    ](i, states, model)
                    dones[i] = Scalar[gpu_dtype](0.0)

            ctx.enqueue_function[
                selective_reset_with_fk_wrapper,
                selective_reset_with_fk_wrapper,
            ](
                states,
                dones,
                model,
                Int(rng_seed),
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Extract observations from state buffer.

        Uses CONFIG's custom extraction if available (e.g., sin/cos encoding),
        otherwise falls back to MODEL_DEF's default qpos[skip:]+qvel extraction.
        """
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var obs = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, OBS_DIM_VAL),
            MutAnyOrigin,
        ](obs_buf.unsafe_ptr())

        comptime QPOS_OFF = qpos_offset[
            Self.MODEL_DEF.NQ, Self.MODEL_DEF.NV
        ]()
        comptime QVEL_OFF = qvel_offset[
            Self.MODEL_DEF.NQ, Self.MODEL_DEF.NV
        ]()
        comptime XPOS_OFF = xpos_offset[
            Self.MODEL_DEF.NQ, Self.MODEL_DEF.NV, Self.MODEL_DEF.NBODY
        ]()
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def custom_obs_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            obs: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, OBS_DIM_VAL),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            if not Self.CONFIG.custom_extract_obs_gpu[
                gpu_dtype, BATCH_SIZE, STATE_SIZE_VAL, OBS_DIM_VAL
            ](states, obs, env, QPOS_OFF, QVEL_OFF, XPOS_OFF):
                Self.MODEL_DEF.extract_obs_gpu[
                    gpu_dtype, BATCH_SIZE, STATE_SIZE_VAL, OBS_DIM_VAL
                ](states, obs, env)

        ctx.enqueue_function[custom_obs_kernel, custom_obs_kernel](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def is_terminal_obs_gpu[
        BATCH_SIZE: Int,
        OBS_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        obs_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Check termination from observations for model-based rollouts."""
        comptime TPB_VAL = 256
        comptime BLOCKS = (BATCH_SIZE + TPB_VAL - 1) // TPB_VAL

        @always_inline
        def term_obs_wrapper(
            obs: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, OBS_DIM_VAL),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            if Self.CONFIG.is_terminal_from_obs_gpu[
                gpu_dtype, BATCH_SIZE, OBS_DIM_VAL
            ](obs, i):
                dones[i] = Scalar[gpu_dtype](1.0)
            else:
                dones[i] = Scalar[gpu_dtype](0.0)

        var obs_t = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, OBS_DIM_VAL),
            MutAnyOrigin,
        ](obs_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        ctx.enqueue_function[term_obs_wrapper, term_obs_wrapper](
            obs_t, dones_t,
            grid_dim=(BLOCKS,), block_dim=(TPB_VAL,),
        )

    # =========================================================================
    # GPU Helper Functions
    # =========================================================================

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype]) raises:
        """Initialize pre-allocated step workspace buffer."""
        comptime MODEL_SIZE = model_size_with_invweight[
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.NJOINT,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NGEOM,
            NEQUALITY=Self.MODEL_DEF.MAX_EQUALITY,
            NTENDON=Self.MODEL_DEF.MAX_TENDON,
            NSITE=Self.MODEL_DEF.NSITE,
        ]()
        var model_view = DeviceBuffer[gpu_dtype](
            ctx,
            workspace_buf.unsafe_ptr(),
            MODEL_SIZE,
            owning=False,
        )
        Self.MODEL_DEF.init_model_gpu(ctx, model_view)

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        """Update curriculum parameters in a pre-allocated workspace.

        Writes up to MODEL_CURRICULUM_SIZE values from the list into the
        curriculum section of the model buffer via a small GPU kernel.
        Uses a kernel instead of enqueue_copy to avoid Metal sub-pointer
        issues (Metal requires base buffer pointers, not offset pointers).
        """
        var n = len(curriculum_values)
        if n == 0:
            return

        # Extract values (curriculum has at most MODEL_CURRICULUM_SIZE entries)
        var v0 = curriculum_values[0] if n > 0 else Scalar[gpu_dtype](0.0)
        var v1 = curriculum_values[1] if n > 1 else Scalar[gpu_dtype](0.0)

        comptime CURR_OFF = model_curriculum_offset[
            Self.MODEL_DEF.NBODY, Self.MODEL_DEF.NJOINT
        ]()
        comptime MODEL_SIZE = model_size_with_invweight[
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.NJOINT,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NGEOM,
            NEQUALITY=Self.MODEL_DEF.MAX_EQUALITY,
            NTENDON=Self.MODEL_DEF.MAX_TENDON,
            NSITE=Self.MODEL_DEF.NSITE,
        ]()

        # Use the base pointer of workspace_buf (= start of model section).
        # A single-thread kernel writes curriculum values at the correct offset.
        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](workspace_buf.unsafe_ptr())

        @always_inline
        def write_curriculum_kernel(
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            v0: Scalar[gpu_dtype],
            v1: Scalar[gpu_dtype],
        ):
            model[0, CURR_OFF + 0] = v0
            model[0, CURR_OFF + 1] = v1

        ctx.enqueue_function[write_curriculum_kernel, write_curriculum_kernel](
            model,
            v0,
            v1,
            grid_dim=(1,),
            block_dim=(1,),
        )

    @staticmethod
    def _pre_step_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[gpu_dtype]) raises:
        """Run config's pre-step hook for all environments on GPU."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime META_OFF = metadata_offset[
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.MAX_CONTACTS,
        ]()

        @always_inline
        def pre_step_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.CONFIG.pre_step_gpu[gpu_dtype, BATCH_SIZE, STATE_SIZE](
                states, env, META_OFF
            )

        ctx.enqueue_function[pre_step_kernel, pre_step_kernel](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def _extract_obs_rewards_dones_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        OBS_DIM_VAL: Int,
        MAX_STEPS_VAL: Int = 1000,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        model_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut terminated_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Extract observations, compute rewards, check termination."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())
        var actions = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, Self.MODEL_DEF.ACTION_DIM),
            MutAnyOrigin,
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var terminated_out = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](terminated_buf.unsafe_ptr())
        var obs = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM_VAL), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime QPOS_OFF = qpos_offset[Self.MODEL_DEF.NQ, Self.MODEL_DEF.NV]()
        comptime QVEL_OFF = qvel_offset[Self.MODEL_DEF.NQ, Self.MODEL_DEF.NV]()
        comptime XPOS_OFF = xpos_offset[
            Self.MODEL_DEF.NQ, Self.MODEL_DEF.NV, Self.MODEL_DEF.NBODY
        ]()
        comptime XIPOS_OFF = xipos_offset[
            Self.MODEL_DEF.NQ, Self.MODEL_DEF.NV, Self.MODEL_DEF.NBODY
        ]()
        comptime META_OFF = metadata_offset[
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.MAX_CONTACTS,
        ]()
        comptime CURRICULUM_OFF = model_curriculum_offset[
            Self.MODEL_DEF.NBODY, Self.MODEL_DEF.NJOINT
        ]()
        comptime CFRC_EXT_OFF = cfrc_ext_offset[
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.MAX_CONTACTS,
            Self.MODEL_DEF.NSITE,
        ]()
        comptime CVEL_OFF = cvel_offset[
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.MAX_CONTACTS,
            Self.MODEL_DEF.NSITE,
        ]()

        @always_inline
        def extract_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, Self.MODEL_DEF.ACTION_DIM),
                MutAnyOrigin,
            ],
            rewards: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            terminated_out: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, OBS_DIM_VAL),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Increment step counter
            var step_count = Int(
                rebind[Scalar[gpu_dtype]](
                    states[env, META_OFF + META_IDX_STEP_COUNT]
                )
            )
            step_count += 1
            states[env, META_OFF + META_IDX_STEP_COUNT] = Scalar[gpu_dtype](
                step_count
            )

            # Extract observations: try config's custom extraction first,
            # fall back to model's default qpos[skip:]+qvel extraction.
            if not Self.CONFIG.custom_extract_obs_gpu[
                gpu_dtype, BATCH_SIZE, STATE_SIZE, OBS_DIM_VAL
            ](states, obs, env, QPOS_OFF, QVEL_OFF, XPOS_OFF):
                Self.MODEL_DEF.extract_obs_gpu[
                    gpu_dtype, BATCH_SIZE, STATE_SIZE, OBS_DIM_VAL
                ](states, obs, env)

            # Compute reward and termination via config (full state access)
            var result = Self.CONFIG.compute_reward_and_done_gpu[
                gpu_dtype,
                BATCH_SIZE,
                STATE_SIZE,
                Self.MODEL_DEF.ACTION_DIM,
                MODEL_SIZE,
            ](
                states,
                model,
                actions,
                env,
                QPOS_OFF,
                XPOS_OFF,
                XIPOS_OFF,
                CFRC_EXT_OFF,
                CVEL_OFF,
                META_OFF,
                CURRICULUM_OFF,
                step_count,
                Self.CONFIG.FRAME_SKIP,
                Scalar[gpu_dtype](Self.CONFIG.get_timestep()),
            )
            rewards[env] = result[0]

            # Determine termination
            var is_terminated = result[1]

            comptime if not Self.TERMINATE_ON_UNHEALTHY:
                is_terminated = False

            var truncated = step_count >= MAX_STEPS_VAL

            # dones = terminated OR truncated (for episode tracking/resets)
            if is_terminated or truncated:
                dones[env] = Scalar[gpu_dtype](1.0)
            else:
                dones[env] = Scalar[gpu_dtype](0.0)

            # terminated_out = only true termination (for replay buffer TD targets)
            terminated_out[env] = Scalar[gpu_dtype](
                1.0
            ) if is_terminated else Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[extract_kernel, extract_kernel](
            states,
            model,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    def _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int = 0,
    ):
        """Reset a single environment on GPU with random noise."""
        var RESET_NOISE = Scalar[gpu_dtype](Self.CONFIG.get_reset_noise())

        # Use model's Joints reset delegate
        Self.MODEL_DEF.reset_env_gpu[gpu_dtype, BATCH_SIZE, STATE_SIZE](
            states, env, RESET_NOISE, seed
        )

        # Apply non-zero qpos offsets (e.g., Humanoid z=1.4, quat_w=1.0)
        comptime QPOS_OFF = qpos_offset[Self.MODEL_DEF.NQ, Self.MODEL_DEF.NV]()
        Self.CONFIG.init_qpos_gpu[gpu_dtype, BATCH_SIZE, STATE_SIZE](
            states, env, QPOS_OFF
        )

        # Reset step counter and prev_x via config's pre_step hook
        comptime META_OFF = metadata_offset[
            Self.MODEL_DEF.NQ,
            Self.MODEL_DEF.NV,
            Self.MODEL_DEF.NBODY,
            Self.MODEL_DEF.MAX_CONTACTS,
        ]()
        states[env, META_OFF + META_IDX_STEP_COUNT] = Scalar[gpu_dtype](0.0)
        # Let config initialize prev_x
        Self.CONFIG.pre_step_gpu[gpu_dtype, BATCH_SIZE, STATE_SIZE](
            states, env, META_OFF
        )

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    def init_renderer(mut self) raises -> Bool:
        return self._init_renderer(show_velocity=True)

    def init_renderer(mut self, show_velocity: Bool) raises -> Bool:
        return self._init_renderer(show_velocity=show_velocity)

    def _init_renderer(mut self, show_velocity: Bool) raises -> Bool:
        if self._renderer_initialized:
            return True

        self._renderer = alloc[ModelRenderer[Self.MODEL_DEF]](1)

        var renderer = ModelRenderer[Self.MODEL_DEF](
            width=1280,
            height=720,
            visual_radius_scale=1.0,
            axes_offset=1.5,
            vel_arrow_height=0.15,
            vel_arrow_scale=0.1,
            show_velocity=show_velocity,
        )
        renderer.init()

        self._renderer.init_pointee_move(renderer^)
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return

        if not self._renderer[].is_open():
            return

        # Copy via accessor methods to bypass Mojo type-expression identity bug
        var xpos = InlineArray[Scalar[Self.DTYPE], Self.MODEL_DEF.NBODY * 3](
            uninitialized=True
        )
        var xquat = InlineArray[Scalar[Self.DTYPE], Self.MODEL_DEF.NBODY * 4](
            uninitialized=True
        )
        for i in range(Self.MODEL_DEF.NBODY * 3):
            xpos[i] = self.get_xpos(i)
        for i in range(Self.MODEL_DEF.NBODY * 4):
            xquat[i] = self.get_xquat(i)
        self._renderer[].render_from_body_state(
            xpos,
            xquat,
            Self.MODEL_DEF.NBODY,
            vel_x=Float64(self.get_x_velocity()),
        )

    def close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return

        self._renderer[].close()
        self._renderer.free()
        self._renderer_initialized = False

    def is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer[].is_open()

    def check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer[].check_quit()

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer[].delay(ms)

    def renderer_is_paused(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer[].renderer.is_paused

    def renderer_step_once(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer[].renderer.step_once
