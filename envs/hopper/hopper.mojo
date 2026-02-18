"""Hopper Environment - MuJoCo-style Hopper using Generalized Coordinates engine.

This implementation uses the physics3d Generalized Coordinates (GC) engine:
- Model/Data for joint-space physics (MuJoCo-style)
- DefaultIntegrator for constraint-based contact solving
- Joint-space state: qpos (positions), qvel (velocities)
- Forward kinematics computes body positions (xpos, xquat)

Uses generic ObsState[11] and ContAction[3] types instead of per-env structs.
Observation extraction, action application, and joint limit enforcement
are all handled by HopperJoints methods from the model definition.
"""

from collections import InlineArray
from random.philox import Random as PhiloxRandom

from core import (
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
    State,
    Action,
    ObsState,
    ContAction,
)
from render import Renderer2D
from deep_rl import dtype as gpu_dtype

# GPU imports
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

# Import GC physics engine
from physics3d.types import Model, Data
from physics3d.integrator import RK4Integrator
from physics3d.solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from physics3d.gpu.constants import (
    TPB,
    state_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    metadata_offset,
    model_size,
    model_size_with_invweight,
    integrator_workspace_size,
    META_IDX_NUM_CONTACTS,
    META_IDX_STEP_COUNT,
    META_IDX_PREV_X,
    model_curriculum_offset,
    CURRICULUM_IDX_MIN_HEIGHT,
    CURRICULUM_IDX_MAX_PITCH,
)

from .hopper_def import (
    HopperModel,
    HopperBodies,
    HopperJoints,
    HopperGeoms,
    HopperActuators,
    HopperParams,
    HopperDefaults,
    HopperCamera,
    HopperLight,
    BODY_TORSO,
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    NGEOM,
    OBS_DIM,
    ACTION_DIM,
    FRAME_SKIP,
    FORWARD_REWARD_WEIGHT,
    CTRL_COST_WEIGHT,
    HEALTHY_REWARD,
    RESET_NOISE_SCALE,
)
from .renderer import HopperRenderer


# =============================================================================
# Hopper Environment
# =============================================================================


struct Hopper[
    DTYPE: DType where DTYPE.is_floating_point() = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
](
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
):
    """Hopper environment using Generalized Coordinates physics.

    Uses generic ObsState[11] and ContAction[3] types instead of per-env structs.
    Observation extraction, action application, and joint limit enforcement
    are all handled by HopperJoints methods from the model definition.
    """

    # Trait type aliases
    comptime dtype = Self.DTYPE
    comptime StateType = ObsState[OBS_DIM]
    comptime ActionType = ContAction[ACTION_DIM]

    comptime MAX_STEPS_VAL: Int = 1000

    # Layout constants
    comptime OBS_DIM: Int = OBS_DIM
    comptime ACTION_DIM: Int = ACTION_DIM

    # GC physics layout constants
    comptime NQ: Int = NQ
    comptime NV: Int = NV
    comptime NUM_BODIES: Int = NBODY
    comptime NUM_JOINTS: Int = NJOINT
    comptime MAX_CONTACTS: Int = MAX_CONTACTS
    comptime NGEOM: Int = NGEOM

    # GPU state size
    comptime STATE_SIZE: Int = state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()

    # Pre-allocated workspace sizes for step_kernel_gpu
    comptime STEP_WS_SHARED: Int = model_size_with_invweight[
        NBODY, NJOINT, NV, NGEOM
    ]()
    comptime STEP_WS_PER_ENV: Int = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

    # Physics model and data
    var model: Model[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NUM_BODIES,
        Self.NUM_JOINTS,
        Self.MAX_CONTACTS,
        Self.NGEOM,
    ]
    var data: Data[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NUM_BODIES,
        Self.NUM_JOINTS,
        Self.MAX_CONTACTS,
    ]

    # Environment parameters
    var max_steps: Int
    var current_step: Int
    var frame_skip: Int

    # Previous x position for velocity calculation
    var prev_x_position: Scalar[Self.DTYPE]

    # Renderer (optional)
    var _renderer: UnsafePointer[HopperRenderer, MutAnyOrigin]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        max_steps: Int = 1000,
        frame_skip: Int = 4,
    ):
        """Initialize the Hopper environment."""
        self.max_steps = max_steps
        self.current_step = 0
        self.frame_skip = frame_skip
        self.prev_x_position = Scalar[Self.DTYPE](0.0)
        self._renderer = UnsafePointer[HopperRenderer, MutAnyOrigin]()
        self._renderer_initialized = False

        # Initialize GC model with solver parameters
        comptime P = HopperParams[Self.DTYPE]
        self.model = Model[
            Self.DTYPE,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
            Self.NGEOM,
        ]()
        HopperModel.setup_solver_params[Defaults=HopperDefaults](
            self.model,
        )

        # Initialize data
        self.data = Data[
            Self.DTYPE,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
        ]()

        # Configure bodies, joints, and geoms from compile-time model definition
        HopperBodies.setup_model(self.model)
        HopperJoints.setup_model[Defaults=HopperDefaults](self.model)
        HopperGeoms.setup_model[Defaults=HopperDefaults](self.model)

        # Reset qpos to initial values, run FK + compute body inverse weights
        HopperJoints.reset_data(self.data)
        HopperModel.finalize(self.model, self.data)

        # Reset step counter and previous position
        self.current_step = 0
        self.prev_x_position = self.data.qpos[JOINT_ROOTX]

    # =========================================================================
    # Physics State Management
    # =========================================================================

    fn _reset_state(mut self):
        """Reset to initial standing position."""
        HopperJoints.reset_data(self.data)

        # Run forward kinematics to compute xpos/xquat
        forward_kinematics(self.model, self.data)

        # Reset step counter and previous position
        self.current_step = 0
        self.prev_x_position = self.data.qpos[JOINT_ROOTX]

    fn _get_obs(self) -> ObsState[OBS_DIM]:
        """Extract observation from current physics data."""
        var obs_list = List[Scalar[Self.DTYPE]](capacity=OBS_DIM)
        HopperJoints.extract_obs(self.data, obs_list)
        var obs = ObsState[OBS_DIM]()
        for i in range(OBS_DIM):
            obs.data[i] = Float64(obs_list[i])
        return obs^

    fn _is_healthy(self) -> Bool:
        """Check if hopper is in a healthy state."""
        comptime P = HopperParams[Self.DTYPE]
        var z = self.data.qpos[JOINT_ROOTZ]
        var pitch = self.data.qpos[JOINT_ROOTY]

        if z < P.MIN_HEIGHT:
            return False
        if pitch > P.MAX_PITCH or pitch < -P.MAX_PITCH:
            return False
        return True

    fn _compute_reward(
        self,
        x_velocity: Float64,
        action: ContAction[ACTION_DIM],
        is_healthy: Bool,
    ) -> Float64:
        """Compute reward for current state (MuJoCo Hopper-v5 compatible).

        Reward = forward_reward + healthy_reward - ctrl_cost
        - forward_reward: x_velocity (forward_reward_weight = 1.0)
        - healthy_reward: 1.0 if healthy, 0.0 otherwise
        - ctrl_cost: 0.001 * sum(action^2) using NORMALIZED actions [-1, 1]
        """
        var forward_reward = FORWARD_REWARD_WEIGHT * x_velocity

        var healthy_reward: Float64 = 0.0
        if is_healthy:
            healthy_reward = Float64(HEALTHY_REWARD)

        var ctrl_cost = CTRL_COST_WEIGHT * action.squared_sum()

        return forward_reward + healthy_reward - ctrl_cost

    # =========================================================================
    # BoxContinuousActionEnv Interface
    # =========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a list."""
        var obs = List[Scalar[Self.dtype]](capacity=Self.OBS_DIM)
        HopperJoints.extract_obs(self.data, obs)
        return obs^

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial continuous observation."""
        self._reset_state()
        return self.get_obs_list()

    fn obs_dim(self) -> Int:
        return Self.OBS_DIM

    fn action_dim(self) -> Int:
        return Self.ACTION_DIM

    fn action_low(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](-1.0)

    fn action_high(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](1.0)

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take 1D continuous action (broadcasts to all joints)."""
        var actions = List[Scalar[Self.dtype]]()
        for _ in range(Self.ACTION_DIM):
            actions.append(action)
        return self.step_continuous_vec(actions)

    fn step_continuous_vec[
        DTYPE2: DType
    ](mut self, action: List[Scalar[DTYPE2]], verbose: Bool = False) -> Tuple[
        List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool
    ]:
        """Take multi-dimensional continuous action and return (obs, reward, done).
        """
        # Convert to ContAction
        var act = ContAction[ACTION_DIM]()
        for i in range(min(ACTION_DIM, len(action))):
            act.data[i] = Float64(action[i])

        # Take step
        var result = self.step(act)

        # Build observation list
        var obs_list = List[Scalar[Self.DTYPE]](capacity=Self.OBS_DIM)
        HopperJoints.extract_obs(self.data, obs_list)
        var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
        for i in range(Self.OBS_DIM):
            obs.append(Scalar[DTYPE2](obs_list[i]))

        return (obs^, Scalar[DTYPE2](result[1]), result[2])

    # =========================================================================
    # Env Interface
    # =========================================================================

    fn step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        # Store previous x position for velocity calculation
        self.prev_x_position = self.data.qpos[JOINT_ROOTX]

        # Apply actions via actuators (MuJoCo-style)
        var clamped_action = action.clamp()
        HopperActuators.apply_actions(self.data, clamped_action.to_list())

        # Physics step (with frame skip)
        for _ in range(self.frame_skip):
            RK4Integrator[SOLVER=NewtonSolver].step(self.model, self.data)
            # Enforce joint limits after each physics step
            HopperJoints.enforce_limits(self.data)

        self.current_step += 1

        # Compute velocity from position change
        var x_position_after = Float64(self.data.qpos[JOINT_ROOTX])
        var dt = Float64(DT * self.frame_skip)
        var x_velocity = (x_position_after - Float64(self.prev_x_position)) / dt

        # Health check and termination
        var is_healthy = self._is_healthy()
        var terminated = False

        @parameter
        if Self.TERMINATE_ON_UNHEALTHY:
            terminated = not is_healthy
        var truncated = self.current_step >= self.max_steps
        var done = terminated or truncated

        # Compute reward using NORMALIZED actions (not torques!)
        var reward = self._compute_reward(
            x_velocity, clamped_action, is_healthy
        )

        return (self._get_obs(), Scalar[Self.dtype](reward), done)

    fn get_state(self) -> Self.StateType:
        """Get current state."""
        return self._get_obs()

    fn reset(mut self) -> Self.StateType:
        """Reset and return initial state."""
        self._reset_state()
        return self._get_obs()

    fn render(mut self, mut renderer: Renderer2D):
        pass

    fn close(mut self):
        if self._renderer_initialized:
            try:
                self._renderer[].close()
            except:
                pass
            self._renderer.free()
            self._renderer_initialized = False

    # =========================================================================
    # Position/State Accessors
    # =========================================================================

    fn get_qpos(self) -> InlineArray[Scalar[Self.DTYPE], 6]:
        var qpos = InlineArray[Scalar[Self.DTYPE], 6](uninitialized=True)
        for i in range(6):
            qpos[i] = self.data.qpos[i]
        return qpos^

    fn get_qvel(self) -> InlineArray[Scalar[Self.DTYPE], 6]:
        var qvel = InlineArray[Scalar[Self.DTYPE], 6](uninitialized=True)
        for i in range(6):
            qvel[i] = self.data.qvel[i]
        return qvel^

    fn get_body_position(
        self, body_id: Int
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        return (
            self.data.xpos[body_id * 3 + 0],
            self.data.xpos[body_id * 3 + 1],
            self.data.xpos[body_id * 3 + 2],
        )

    fn get_body_quaternion(
        self, body_id: Int
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        return (
            self.data.xquat[body_id * 4 + 0],
            self.data.xquat[body_id * 4 + 1],
            self.data.xquat[body_id * 4 + 2],
            self.data.xquat[body_id * 4 + 3],
        )

    fn get_torso_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        return self.get_body_position(BODY_TORSO)

    fn get_thigh_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        return self.get_body_position(1)

    fn get_leg_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        return self.get_body_position(2)

    fn get_foot_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        return self.get_body_position(3)

    fn get_torso_quaternion(
        self,
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        return self.get_body_quaternion(BODY_TORSO)

    fn get_thigh_quaternion(
        self,
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        return self.get_body_quaternion(1)

    fn get_leg_quaternion(
        self,
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        return self.get_body_quaternion(2)

    fn get_foot_quaternion(
        self,
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        return self.get_body_quaternion(3)

    fn get_x_velocity(self) -> Scalar[Self.DTYPE]:
        return self.data.qvel[JOINT_ROOTX]

    fn get_current_step(self) -> Int:
        return self.current_step

    fn get_max_steps(self) -> Int:
        return self.max_steps

    fn is_done(self) -> Bool:
        var truncated = self.current_step >= self.max_steps

        @parameter
        if Self.TERMINATE_ON_UNHEALTHY:
            return truncated or not self._is_healthy()
        else:
            return truncated

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    fn init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True

        self._renderer = alloc[HopperRenderer](1)

        var renderer = HopperRenderer(
            width=1024,
            height=576,
            visual_radius_scale=1.5,
            cam_eye_y=HopperCamera.POS_Y,
            cam_eye_z=HopperCamera.POS_Z,
            cam_target_z=HopperCamera.TARGET_Z,
            axes_offset=0.8,
            vel_arrow_height=0.25,
            vel_arrow_scale=0.15,
            light_dir_x=HopperLight.DIR_X,
            light_dir_y=HopperLight.DIR_Y,
            light_dir_z=HopperLight.DIR_Z,
            light_color_r=HopperLight.COLOR_R,
            light_color_g=HopperLight.COLOR_G,
            light_color_b=HopperLight.COLOR_B,
            light_ambient=HopperLight.AMBIENT,
        )
        renderer.init()

        self._renderer.init_pointee_move(renderer^)
        self._renderer_initialized = True
        return True

    fn render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return

        if not self._renderer[].is_open():
            return

        self._renderer[].render_from_body_state(
            self.data.xpos, self.data.xquat, Hopper.NUM_BODIES,
            vel_x=Float64(self.get_x_velocity()),
        )

    fn close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return

        self._renderer[].close()
        self._renderer.free()
        self._renderer_initialized = False

    fn is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer[].is_open()

    fn check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer[].check_quit()

    fn renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer[].delay(ms)

    # =========================================================================
    # GPUContinuousEnv Interface (Static GPU Kernels)
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
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
        mut obs_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[gpu_dtype]] = [],
        workspace_ptr: UnsafePointer[
            Scalar[gpu_dtype], MutAnyOrigin
        ] = UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin](),
    ) raises:
        """Batched GPU step function using GC physics engine."""

        comptime MODEL_SIZE = model_size[
            Hopper.NUM_BODIES, Hopper.NUM_JOINTS, Hopper.NGEOM
        ]()
        comptime P = HopperParams[gpu_dtype]
        comptime WS_SIZE = integrator_workspace_size[
            Self.NV, Self.NUM_BODIES
        ]() + Self.NV * Self.NV + NewtonSolver.solver_workspace_size[
            Self.NV, Self.MAX_CONTACTS
        ]()

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
            var min_height = (
                curriculum_values[0] if len(curriculum_values)
                > 0 else P.MIN_HEIGHT
            )
            var max_pitch = (
                curriculum_values[1] if len(curriculum_values)
                > 1 else P.MAX_PITCH
            )
            Self._init_model_gpu(ctx, model_buf, min_height, max_pitch)
            workspace_buf = ctx.enqueue_create_buffer[gpu_dtype](
                BATCH_SIZE * WS_SIZE
            )

        # Store prev_x_position before physics
        Self._store_prev_x_gpu[BATCH_SIZE, STATE_SIZE_VAL](ctx, states_buf)

        # Apply actions to qfrc via actuators (MuJoCo-style)
        HopperActuators.apply_actions_kernel_gpu[
            gpu_dtype,
            BATCH_SIZE,
            STATE_SIZE_VAL,
            ACTION_DIM_VAL,
            Self.NQ,
            Self.NV,
        ](ctx, states_buf, actions_buf)

        # Run FRAME_SKIP physics sub-steps
        for _ in range(P.FRAME_SKIP):
            RK4Integrator[SOLVER=NewtonSolver].step_gpu[
                gpu_dtype,
                Self.NQ,
                Self.NV,
                Self.NUM_BODIES,
                Self.NUM_JOINTS,
                Self.MAX_CONTACTS,
                BATCH_SIZE,
                NGEOM = Self.NGEOM,
            ](
                ctx,
                states_buf,
                model_buf,
                workspace_buf,
            )

        # Extract observations, compute rewards, check termination
        Self._extract_obs_rewards_dones_gpu[
            BATCH_SIZE,
            STATE_SIZE_VAL,
            MODEL_SIZE,
            OBS_DIM_VAL,
        ](
            ctx,
            states_buf,
            model_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments on GPU."""
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            seed: Int,
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](states, i, seed)

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            Int(rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

        # Run forward kinematics
        comptime MODEL_SIZE = model_size[
            Self.NUM_BODIES, Self.NUM_JOINTS, Self.NGEOM
        ]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        Self._init_model_gpu(ctx, model_buf)

        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn fk_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            forward_kinematics_gpu[
                gpu_dtype,
                Self.NQ,
                Self.NV,
                Self.NUM_BODIES,
                Self.NUM_JOINTS,
                Self.MAX_CONTACTS,
                STATE_SIZE_VAL,
                MODEL_SIZE,
                BATCH_SIZE,
            ](i, states, model)

        ctx.enqueue_function[fk_wrapper, fk_wrapper](
            states,
            model,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
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

        comptime MODEL_SIZE = model_size[
            Self.NUM_BODIES, Self.NUM_JOINTS, Self.NGEOM
        ]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        Self._init_model_gpu(ctx, model_buf)

        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn selective_reset_with_fk_wrapper(
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
                Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](states, i, seed)
                forward_kinematics_gpu[
                    gpu_dtype,
                    Self.NQ,
                    Self.NV,
                    Self.NUM_BODIES,
                    Self.NUM_JOINTS,
                    Self.MAX_CONTACTS,
                    STATE_SIZE_VAL,
                    MODEL_SIZE,
                    BATCH_SIZE,
                ](i, states, model)
                dones[i] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[
            selective_reset_with_fk_wrapper, selective_reset_with_fk_wrapper
        ](
            states,
            dones,
            model,
            Int(rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Extract observations from GC state buffer using generic Joints method.
        """
        HopperJoints.extract_obs_kernel_gpu[
            gpu_dtype, BATCH_SIZE, STATE_SIZE_VAL, OBS_DIM_VAL
        ](ctx, states_buf, obs_buf)

    # =========================================================================
    # GPU Helper Functions
    # =========================================================================

    @staticmethod
    fn _init_model_gpu(
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[gpu_dtype],
        min_height: Scalar[gpu_dtype] = HopperParams[gpu_dtype].MIN_HEIGHT,
        max_pitch: Scalar[gpu_dtype] = HopperParams[gpu_dtype].MAX_PITCH,
    ) raises:
        """Initialize model buffer with Hopper parameters."""
        comptime P = HopperParams[gpu_dtype]

        var model = Model[
            gpu_dtype,
            Hopper.NQ,
            Hopper.NV,
            Hopper.NUM_BODIES,
            Hopper.NUM_JOINTS,
            Hopper.MAX_CONTACTS,
            Hopper.NGEOM,
        ]()
        HopperModel.setup_solver_params[Defaults=HopperDefaults](
            model,
        )
        HopperBodies.setup_model(model)
        HopperJoints.setup_model[Defaults=HopperDefaults](model)
        HopperGeoms.setup_model[Defaults=HopperDefaults](model)

        # Compute body_invweight0 and dof_invweight0 at reference pose
        var data_ref = Data[
            gpu_dtype,
            Hopper.NQ,
            Hopper.NV,
            Hopper.NUM_BODIES,
            Hopper.NUM_JOINTS,
            Hopper.MAX_CONTACTS,
        ]()
        HopperModel.finalize(model, data_ref)

        var host_buf = HopperModel.create_gpu_model_buffer[
            gpu_dtype, Self.MAX_CONTACTS
        ](ctx, model)

        var curr = model_curriculum_offset[
            Hopper.NUM_BODIES, Hopper.NUM_JOINTS
        ]()
        host_buf[curr + CURRICULUM_IDX_MIN_HEIGHT] = min_height
        host_buf[curr + CURRICULUM_IDX_MAX_PITCH] = max_pitch

        ctx.enqueue_copy(model_buf, host_buf.unsafe_ptr())

    @staticmethod
    fn init_model_gpu_with_curriculum(
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[gpu_dtype],
        min_height: Scalar[gpu_dtype],
        max_pitch: Scalar[gpu_dtype],
    ) raises:
        Self._init_model_gpu(ctx, model_buf, min_height, max_pitch)

    @staticmethod
    fn init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype],) raises:
        """Initialize pre-allocated step workspace buffer."""
        comptime MODEL_SIZE = model_size_with_invweight[
            Hopper.NUM_BODIES, Hopper.NUM_JOINTS, Hopper.NV, Hopper.NGEOM
        ]()
        var model_view = DeviceBuffer[gpu_dtype](
            ctx,
            workspace_buf.unsafe_ptr(),
            MODEL_SIZE,
            owning=False,
        )
        Self._init_model_gpu(ctx, model_view)

    @staticmethod
    fn update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        """Update only the curriculum parameters in a pre-allocated workspace.
        """
        if len(curriculum_values) < 2:
            return
        var curr_offset = model_curriculum_offset[
            Hopper.NUM_BODIES, Hopper.NUM_JOINTS
        ]()
        var curriculum_host = InlineArray[Scalar[gpu_dtype], 2](
            fill=[
                curriculum_values[0],
                curriculum_values[1],
            ],
        )
        ctx.enqueue_copy(
            workspace_buf.unsafe_ptr() + curr_offset,
            curriculum_host.unsafe_ptr(),
            2,
        )

    @staticmethod
    fn _store_prev_x_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[gpu_dtype],) raises:
        """Store current rootx position into metadata for velocity computation.
        """
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime QPOS_OFF = qpos_offset[Hopper.NQ, Hopper.NV]()
        comptime META_OFF = metadata_offset[
            Hopper.NQ,
            Hopper.NV,
            Hopper.NUM_BODIES,
            Hopper.MAX_CONTACTS,
        ]()

        @always_inline
        fn store_prev_x_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            states[env, META_OFF + META_IDX_PREV_X] = states[env, QPOS_OFF + 0]

        ctx.enqueue_function[store_prev_x_kernel, store_prev_x_kernel](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _extract_obs_rewards_dones_gpu[
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
            gpu_dtype, Layout.row_major(BATCH_SIZE, 3), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var obs = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM_VAL), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime P = HopperParams[gpu_dtype]
        comptime QPOS_OFF = qpos_offset[Hopper.NQ, Hopper.NV]()
        comptime META_OFF = metadata_offset[
            Hopper.NQ,
            Hopper.NV,
            Hopper.NUM_BODIES,
            Hopper.MAX_CONTACTS,
        ]()
        comptime CURRICULUM_OFF = model_curriculum_offset[
            Hopper.NUM_BODIES, Hopper.NUM_JOINTS
        ]()

        @always_inline
        fn extract_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE, 3), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
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

            # Read curriculum parameters from model buffer
            var min_height = model[
                0, CURRICULUM_OFF + CURRICULUM_IDX_MIN_HEIGHT
            ]
            var max_pitch = model[0, CURRICULUM_OFF + CURRICULUM_IDX_MAX_PITCH]

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

            # Extract observations using generic Joints method
            HopperJoints.extract_obs_gpu[
                gpu_dtype, BATCH_SIZE, STATE_SIZE, OBS_DIM_VAL
            ](states, obs, env)

            # Compute position-based velocity for reward
            var x_position_after = states[env, QPOS_OFF + 0]
            var prev_x = states[env, META_OFF + META_IDX_PREV_X]
            var effective_dt = P.DT * Scalar[gpu_dtype](P.FRAME_SKIP)
            var x_velocity_reward = (x_position_after - prev_x) / effective_dt

            # Check health using curriculum bounds
            var z_pos = states[env, QPOS_OFF + 1]  # rootz
            var y_angle = states[env, QPOS_OFF + 2]  # rooty
            var is_healthy = True
            if z_pos < min_height:
                is_healthy = False
            if y_angle > max_pitch or y_angle < -max_pitch:
                is_healthy = False

            # Clamp actions for reward computation
            var ctrl_cost_sum = Scalar[gpu_dtype](0.0)
            for a_idx in range(3):
                var a = rebind[Scalar[gpu_dtype]](actions[env, a_idx])
                if a > Scalar[gpu_dtype](1.0):
                    a = Scalar[gpu_dtype](1.0)
                elif a < Scalar[gpu_dtype](-1.0):
                    a = Scalar[gpu_dtype](-1.0)
                ctrl_cost_sum += a * a

            var ctrl_cost = P.CTRL_COST_WEIGHT * ctrl_cost_sum

            var healthy_reward = P.HEALTHY_REWARD
            if not is_healthy:
                healthy_reward = Scalar[gpu_dtype](0.0)

            # Reward = position-based forward_velocity + healthy_reward - ctrl_cost
            var reward = x_velocity_reward + healthy_reward - ctrl_cost
            rewards[env] = reward

            # Determine termination
            var terminated = False
            var truncated = step_count >= MAX_STEPS_VAL

            @parameter
            if Self.TERMINATE_ON_UNHEALTHY:
                terminated = not is_healthy

            if terminated or truncated:
                dones[env] = Scalar[gpu_dtype](1.0)
            else:
                dones[env] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[extract_kernel, extract_kernel](
            states,
            model,
            actions,
            rewards,
            dones,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    fn _reset_env_gpu[
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
        comptime RESET_NOISE: Scalar[gpu_dtype] = 0.005

        # Use generic Joints reset
        HopperJoints.reset_env_gpu[gpu_dtype, BATCH_SIZE, STATE_SIZE](
            states, env, RESET_NOISE, seed
        )

        # Reset step counter and prev_x
        comptime META_OFF = metadata_offset[
            Hopper.NQ,
            Hopper.NV,
            Hopper.NUM_BODIES,
            Hopper.MAX_CONTACTS,
        ]()
        comptime QPOS_OFF = qpos_offset[Hopper.NQ, Hopper.NV]()
        states[env, META_OFF + META_IDX_STEP_COUNT] = Scalar[gpu_dtype](0.0)
        states[env, META_OFF + META_IDX_PREV_X] = states[env, QPOS_OFF + 0]
