"""HalfCheetah Environment - MuJoCo-style Half Cheetah using Generalized Coordinates engine.

This implementation uses the physics3d Generalized Coordinates (GC) engine:
- Model/Data for joint-space physics (MuJoCo-style)
- DefaultIntegrator for constraint-based contact solving
- Joint-space state: qpos (positions), qvel (velocities)
- Forward kinematics computes body positions (xpos, xquat)

The Half Cheetah is a 2D planar model (movement in XZ plane, rotation around Y axis)
consisting of a torso with two leg chains (front and back) and a head, totaling:
- 8 bodies: torso, bthigh, bshin, bfoot, fthigh, fshin, ffoot, head
- 10 joints: 3 root DOFs (unactuated) + 6 leg joints (actuated) + 1 head (fixed)
- 17D observation: 8 qpos (excluding rootx and head) + 9 qvel (excluding head)
- 6D action: torques for the 6 actuated leg joints
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
from render import RendererBase
from deep_rl import dtype as gpu_dtype

# GPU imports
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

# Import GC physics engine
from physics3d.types import Model, Data
from physics3d.integrator import ImplicitFastIntegrator
from physics3d.solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from physics3d.gpu.buffer_utils import copy_model_to_buffer, create_model_buffer
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
    integrator_workspace_size,
    META_IDX_NUM_CONTACTS,
    META_IDX_STEP_COUNT,
    META_IDX_PREV_X,
    model_curriculum_offset,
    CURRICULUM_IDX_MIN_HEIGHT,
    CURRICULUM_IDX_MAX_PITCH,
)

from .half_cheetah_def import (
    HalfCheetahWorldBody,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahParams,
    BODY_TORSO,
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    OBS_DIM,
    ACTION_DIM,
    DT,
    FRAME_SKIP,
    GRAVITY_Z,
    FORWARD_REWARD_WEIGHT,
    CTRL_COST_WEIGHT,
    RESET_NOISE_SCALE,
)
from .renderer import HalfCheetahRenderer

# Math types for renderer
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# HalfCheetah Environment
# =============================================================================


struct HalfCheetah[
    DTYPE: DType where DTYPE.is_floating_point() = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
](
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
):
    """HalfCheetah environment using Generalized Coordinates physics.

    Uses generic ObsState[17] and ContAction[6] types instead of per-env structs.
    Observation extraction, action application, and joint limit enforcement
    are all handled by HalfCheetahJoints methods from the model definition.
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

    # GPU state size
    comptime STATE_SIZE: Int = state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()

    # Pre-allocated workspace sizes for step_kernel_gpu
    comptime STEP_WS_SHARED: Int = model_size[NBODY, NJOINT]()
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
    var _renderer: UnsafePointer[HalfCheetahRenderer, MutAnyOrigin]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        max_steps: Int = 1000,
        frame_skip: Int = 5,
        timestep: Scalar[Self.DTYPE] = 0.002,
    ):
        """Initialize the HalfCheetah environment."""
        self.max_steps = max_steps
        self.current_step = 0
        self.frame_skip = frame_skip
        self.prev_x_position = Scalar[Self.DTYPE](0.0)
        self._renderer = UnsafePointer[HalfCheetahRenderer, MutAnyOrigin]()
        self._renderer_initialized = False

        # Initialize GC model
        self.model = Model[
            Self.DTYPE,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
        ](
            gravity_z=Scalar[Self.DTYPE](GRAVITY_Z),
            timestep=timestep,
        )

        # Set solref/solimp from MuJoCo half_cheetah.xml
        comptime P = HalfCheetahParams[Self.DTYPE]
        self.model.solref_contact[0] = P.SOLREF_CONTACT_0
        self.model.solref_contact[1] = P.SOLREF_CONTACT_1
        self.model.solimp_contact[0] = P.SOLIMP_CONTACT_0
        self.model.solimp_contact[1] = P.SOLIMP_CONTACT_1
        self.model.solimp_contact[2] = P.SOLIMP_CONTACT_2
        self.model.solref_limit[0] = P.SOLREF_LIMIT_0
        self.model.solref_limit[1] = P.SOLREF_LIMIT_1
        self.model.solimp_limit[0] = P.SOLIMP_LIMIT_0
        self.model.solimp_limit[1] = P.SOLIMP_LIMIT_1
        self.model.solimp_limit[2] = P.SOLIMP_LIMIT_2

        # Initialize data
        self.data = Data[
            Self.DTYPE,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
        ]()

        # Configure worldbody, bodies, and joints from compile-time model definition
        HalfCheetahWorldBody.setup_model(self.model)
        HalfCheetahBodies.setup_model(self.model)
        HalfCheetahJoints.setup_model(self.model)

        # Reset to initial state
        self._reset_state()

    # =========================================================================
    # Physics State Management
    # =========================================================================

    fn _reset_state(mut self):
        """Reset to initial standing position."""
        HalfCheetahJoints.reset_data(self.data)

        # Run forward kinematics to compute xpos/xquat
        forward_kinematics(self.model, self.data)

        # Reset step counter and previous position
        self.current_step = 0
        self.prev_x_position = self.data.qpos[JOINT_ROOTX]

    fn _get_obs(self) -> ObsState[OBS_DIM]:
        """Extract observation from current physics data."""
        var obs_list = List[Scalar[Self.DTYPE]](capacity=OBS_DIM)
        HalfCheetahJoints.extract_obs(self.data, obs_list)
        var obs = ObsState[OBS_DIM]()
        for i in range(OBS_DIM):
            obs.data[i] = Float64(obs_list[i])
        return obs^

    fn _compute_reward(
        self,
        x_velocity: Float64,
        action: ContAction[ACTION_DIM],
        y_angle: Float64,
    ) -> Float64:
        """Compute reward for current state."""
        var forward_reward = FORWARD_REWARD_WEIGHT * x_velocity
        var ctrl_cost = CTRL_COST_WEIGHT * action.squared_sum()
        comptime P = HalfCheetahParams[DType.float64]
        var abs_angle = y_angle if y_angle >= 0.0 else -y_angle
        var angle_penalty = Float64(P.ANGLE_PENALTY_WEIGHT) * abs_angle
        return forward_reward - ctrl_cost - angle_penalty

    # =========================================================================
    # BoxContinuousActionEnv Interface
    # =========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a list."""
        var obs = List[Scalar[Self.dtype]](capacity=Self.OBS_DIM)
        HalfCheetahJoints.extract_obs(self.data, obs)
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
    ](mut self, action: List[Scalar[DTYPE2]]) -> Tuple[
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
        HalfCheetahJoints.extract_obs(self.data, obs_list)
        var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
        for i in range(Self.OBS_DIM):
            obs.append(Scalar[DTYPE2](obs_list[i]))

        return (obs^, Scalar[DTYPE2](result[1]), result[2])

    # =========================================================================
    # Env Interface
    # =========================================================================

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        # Store previous x position for velocity calculation
        self.prev_x_position = self.data.qpos[JOINT_ROOTX]

        # Apply actions via generic Joints method
        HalfCheetahJoints.apply_actions(self.data, action.to_list())

        # Physics step (with frame skip)
        for _ in range(self.frame_skip):
            ImplicitFastIntegrator[SOLVER=NewtonSolver].step(
                self.model, self.data
            )
            # Enforce joint limits after each physics step
            HalfCheetahJoints.enforce_limits(self.data)
            # Ground safety clamp
            comptime MIN_ROOTZ: Scalar[Self.DTYPE] = -0.3
            if self.data.qpos[JOINT_ROOTZ] < MIN_ROOTZ:
                self.data.qpos[JOINT_ROOTZ] = MIN_ROOTZ
                if self.data.qvel[JOINT_ROOTZ] < Scalar[Self.DTYPE](0):
                    self.data.qvel[JOINT_ROOTZ] = Scalar[Self.DTYPE](0)

        self.current_step += 1

        # Compute velocity from position change
        var x_position_after = Float64(self.data.qpos[JOINT_ROOTX])
        var dt = Float64(DT * self.frame_skip)
        var x_velocity = (x_position_after - Float64(self.prev_x_position)) / dt

        # Compute reward
        var clamped_action = action.clamp()
        var y_angle = Float64(self.data.qpos[JOINT_ROOTY])
        var reward = self._compute_reward(x_velocity, clamped_action, y_angle)

        # Health check and termination
        var terminated = False

        @parameter
        if Self.TERMINATE_ON_UNHEALTHY:
            comptime P = HalfCheetahParams[DType.float64]
            var abs_angle = y_angle if y_angle >= 0.0 else -y_angle
            terminated = abs_angle > Float64(P.MAX_PITCH)
        var truncated = self.current_step >= self.max_steps
        var done = terminated or truncated

        return (self._get_obs(), Scalar[Self.dtype](reward), done)

    fn get_state(self) -> Self.StateType:
        """Get current state."""
        return self._get_obs()

    fn reset(mut self) -> Self.StateType:
        """Reset and return initial state."""
        self._reset_state()
        return self._get_obs()

    fn render(mut self, mut renderer: RendererBase):
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

    fn get_qpos(self) -> InlineArray[Scalar[Self.DTYPE], 10]:
        var qpos = InlineArray[Scalar[Self.DTYPE], 10](uninitialized=True)
        for i in range(10):
            qpos[i] = self.data.qpos[i]
        return qpos^

    fn get_qvel(self) -> InlineArray[Scalar[Self.DTYPE], 10]:
        var qvel = InlineArray[Scalar[Self.DTYPE], 10](uninitialized=True)
        for i in range(10):
            qvel[i] = self.data.qvel[i]
        return qvel^

    fn get_x_position(self) -> Scalar[Self.DTYPE]:
        return self.data.qpos[JOINT_ROOTX]

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
            comptime P = HalfCheetahParams[DType.float64]
            var y_angle = Float64(self.data.qpos[JOINT_ROOTY])
            var abs_angle = y_angle if y_angle >= 0.0 else -y_angle
            return truncated or abs_angle > Float64(P.MAX_PITCH)
        else:
            return truncated

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    fn init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True

        from memory import alloc

        self._renderer = alloc[HalfCheetahRenderer](1)

        var renderer = HalfCheetahRenderer(
            width=1280,
            height=720,
            follow_cheetah=True,
            show_velocity=True,
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

        var positions = List[Vec3](capacity=Self.NUM_BODIES)
        var quaternions = List[Quat](capacity=Self.NUM_BODIES)

        for i in range(Self.NUM_BODIES):
            var pos = self.data.get_body_position(i)
            positions.append(
                Vec3(Float64(pos[0]), Float64(pos[1]), Float64(pos[2]))
            )

            var quat = self.data.get_body_quaternion(i)
            quaternions.append(
                Quat(
                    Float64(quat[3]),
                    Float64(quat[0]),
                    Float64(quat[1]),
                    Float64(quat[2]),
                )
            )

        var vel_x = Float64(self.get_x_velocity())
        self._renderer[].render(positions, quaternions, vel_x)

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
            HalfCheetah.NUM_BODIES, HalfCheetah.NUM_JOINTS
        ]()
        comptime P = HalfCheetahParams[gpu_dtype]
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
            var max_pitch = (
                curriculum_values[1] if len(curriculum_values)
                > 1 else P.MAX_PITCH
            )
            Self._init_model_gpu(ctx, model_buf, max_pitch)
            workspace_buf = ctx.enqueue_create_buffer[gpu_dtype](
                BATCH_SIZE * WS_SIZE
            )

        # Store prev_x_position before physics
        Self._store_prev_x_gpu[BATCH_SIZE, STATE_SIZE_VAL](ctx, states_buf)

        # Apply actions to qfrc via generic Joints method
        HalfCheetahJoints.apply_actions_kernel_gpu[
            gpu_dtype, BATCH_SIZE, STATE_SIZE_VAL, ACTION_DIM_VAL
        ](ctx, states_buf, actions_buf)

        # Run FRAME_SKIP physics sub-steps with joint limit enforcement
        for _ in range(P.FRAME_SKIP):
            ImplicitFastIntegrator[SOLVER=NewtonSolver].step_gpu[
                gpu_dtype,
                Self.NQ,
                Self.NV,
                Self.NUM_BODIES,
                Self.NUM_JOINTS,
                Self.MAX_CONTACTS,
                BATCH_SIZE,
            ](
                ctx,
                states_buf,
                model_buf,
                workspace_buf,
                dt=Scalar[gpu_dtype](P.DT),
                gravity_z=Scalar[gpu_dtype](-9.81),
                ground_z=Scalar[gpu_dtype](0.0),
            )
            # Enforce joint limits via generic Joints method
            HalfCheetahJoints.enforce_limits_kernel_gpu[
                gpu_dtype, BATCH_SIZE, STATE_SIZE_VAL
            ](ctx, states_buf)
            # Ground safety clamp
            Self._ground_clamp_gpu[BATCH_SIZE, STATE_SIZE_VAL](ctx, states_buf)

        # Extract observations, compute rewards, check termination
        Self._extract_obs_rewards_dones_gpu[
            BATCH_SIZE,
            STATE_SIZE_VAL,
            MODEL_SIZE,
            OBS_DIM_VAL,
            Self.MAX_STEPS_VAL,
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
        comptime MODEL_SIZE = model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()
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

        comptime MODEL_SIZE = model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()
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
        HalfCheetahJoints.extract_obs_kernel_gpu[
            gpu_dtype, BATCH_SIZE, STATE_SIZE_VAL, OBS_DIM_VAL
        ](ctx, states_buf, obs_buf)

    # =========================================================================
    # GPU Helper Functions
    # =========================================================================

    @staticmethod
    fn _init_model_gpu(
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[gpu_dtype],
        max_pitch: Scalar[gpu_dtype] = HalfCheetahParams[gpu_dtype].MAX_PITCH,
    ) raises:
        """Initialize model buffer with HalfCheetah parameters."""
        comptime P = HalfCheetahParams[gpu_dtype]

        var model = Model[
            gpu_dtype,
            HalfCheetah.NQ,
            HalfCheetah.NV,
            HalfCheetah.NUM_BODIES,
            HalfCheetah.NUM_JOINTS,
            HalfCheetah.MAX_CONTACTS,
        ](
            gravity_z=P.GRAVITY_Z,
            timestep=P.DT,
        )

        model.solref_contact[0] = P.SOLREF_CONTACT_0
        model.solref_contact[1] = P.SOLREF_CONTACT_1
        model.solimp_contact[0] = P.SOLIMP_CONTACT_0
        model.solimp_contact[1] = P.SOLIMP_CONTACT_1
        model.solimp_contact[2] = P.SOLIMP_CONTACT_2
        model.solref_limit[0] = P.SOLREF_LIMIT_0
        model.solref_limit[1] = P.SOLREF_LIMIT_1
        model.solimp_limit[0] = P.SOLIMP_LIMIT_0
        model.solimp_limit[1] = P.SOLIMP_LIMIT_1
        model.solimp_limit[2] = P.SOLIMP_LIMIT_2

        HalfCheetahWorldBody.setup_model(model)
        HalfCheetahBodies.setup_model(model)
        HalfCheetahJoints.setup_model(model)

        var host_buf = create_model_buffer[
            gpu_dtype, HalfCheetah.NUM_BODIES, HalfCheetah.NUM_JOINTS
        ](ctx)
        copy_model_to_buffer(model, host_buf)

        var curr = model_curriculum_offset[
            HalfCheetah.NUM_BODIES, HalfCheetah.NUM_JOINTS
        ]()
        host_buf[curr + CURRICULUM_IDX_MIN_HEIGHT] = Scalar[gpu_dtype](0.0)
        host_buf[curr + CURRICULUM_IDX_MAX_PITCH] = max_pitch

        ctx.enqueue_copy(model_buf, host_buf.unsafe_ptr())

    @staticmethod
    fn init_model_gpu_with_curriculum(
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[gpu_dtype],
        max_pitch: Scalar[gpu_dtype],
    ) raises:
        Self._init_model_gpu(ctx, model_buf, max_pitch)

    @staticmethod
    fn init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype],) raises:
        """Initialize pre-allocated step workspace buffer."""
        comptime MODEL_SIZE = model_size[
            HalfCheetah.NUM_BODIES, HalfCheetah.NUM_JOINTS
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
            HalfCheetah.NUM_BODIES, HalfCheetah.NUM_JOINTS
        ]()
        var curriculum_host = InlineArray[Scalar[gpu_dtype], 2](
            fill=[
                curriculum_values[0],
                curriculum_values[1],
            ]
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
        comptime QPOS_OFF = qpos_offset[HalfCheetah.NQ, HalfCheetah.NV]()
        comptime META_OFF = metadata_offset[
            HalfCheetah.NQ,
            HalfCheetah.NV,
            HalfCheetah.NUM_BODIES,
            HalfCheetah.MAX_CONTACTS,
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
    fn _ground_clamp_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[gpu_dtype]) raises:
        """Clamp rootz to prevent catastrophic ground penetration on GPU."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime QPOS_OFF = qpos_offset[HalfCheetah.NQ, HalfCheetah.NV]()
        comptime QVEL_OFF = qvel_offset[HalfCheetah.NQ, HalfCheetah.NV]()
        comptime MIN_ROOTZ: Scalar[gpu_dtype] = -0.3

        @always_inline
        fn ground_clamp_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            var rootz = states[env, QPOS_OFF + 1]
            if rootz < MIN_ROOTZ:
                states[env, QPOS_OFF + 1] = MIN_ROOTZ
                var vz = states[env, QVEL_OFF + 1]
                if vz < Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 1] = Scalar[gpu_dtype](0)

        ctx.enqueue_function[ground_clamp_kernel, ground_clamp_kernel](
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
            gpu_dtype, Layout.row_major(BATCH_SIZE, 6), MutAnyOrigin
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
        comptime P = HalfCheetahParams[gpu_dtype]
        comptime QPOS_OFF = qpos_offset[HalfCheetah.NQ, HalfCheetah.NV]()
        comptime META_OFF = metadata_offset[
            HalfCheetah.NQ,
            HalfCheetah.NV,
            HalfCheetah.NUM_BODIES,
            HalfCheetah.MAX_CONTACTS,
        ]()
        comptime CURRICULUM_OFF = model_curriculum_offset[
            HalfCheetah.NUM_BODIES, HalfCheetah.NUM_JOINTS
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
                gpu_dtype, Layout.row_major(BATCH_SIZE, 6), MutAnyOrigin
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
            HalfCheetahJoints.extract_obs_gpu[
                gpu_dtype, BATCH_SIZE, STATE_SIZE, OBS_DIM_VAL
            ](states, obs, env)

            # Clamp actions for reward computation
            var ctrl_cost_sum = Scalar[gpu_dtype](0.0)
            for a_idx in range(6):
                var a = rebind[Scalar[gpu_dtype]](actions[env, a_idx])
                if a > Scalar[gpu_dtype](1.0):
                    a = Scalar[gpu_dtype](1.0)
                elif a < Scalar[gpu_dtype](-1.0):
                    a = Scalar[gpu_dtype](-1.0)
                ctrl_cost_sum += a * a

            # Read curriculum parameters
            var max_pitch = model[0, CURRICULUM_OFF + CURRICULUM_IDX_MAX_PITCH]

            # Compute velocity from position change
            var x_position_after = states[env, QPOS_OFF + 0]
            var prev_x = states[env, META_OFF + META_IDX_PREV_X]
            var effective_dt = P.DT * Scalar[gpu_dtype](P.FRAME_SKIP)
            var x_velocity = (x_position_after - prev_x) / effective_dt

            # Compute reward
            var forward_reward = P.FORWARD_REWARD_WEIGHT * x_velocity
            var ctrl_cost = P.CTRL_COST_WEIGHT * ctrl_cost_sum
            var y_angle = states[env, QPOS_OFF + 2]  # rooty
            var abs_y_angle = y_angle
            if abs_y_angle < Scalar[gpu_dtype](0.0):
                abs_y_angle = -abs_y_angle
            var angle_penalty = P.ANGLE_PENALTY_WEIGHT * abs_y_angle

            var reward = forward_reward - ctrl_cost - angle_penalty
            rewards[env] = reward

            # Health check
            var is_healthy = True
            if y_angle > max_pitch or y_angle < -max_pitch:
                is_healthy = False

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
        comptime RESET_NOISE: Scalar[gpu_dtype] = 0.1

        # Use generic Joints reset
        HalfCheetahJoints.reset_env_gpu[gpu_dtype, BATCH_SIZE, STATE_SIZE](
            states, env, RESET_NOISE, seed
        )

        # Reset step counter and prev_x
        comptime META_OFF = metadata_offset[
            HalfCheetah.NQ,
            HalfCheetah.NV,
            HalfCheetah.NUM_BODIES,
            HalfCheetah.MAX_CONTACTS,
        ]()
        comptime QPOS_OFF = qpos_offset[HalfCheetah.NQ, HalfCheetah.NV]()
        states[env, META_OFF + META_IDX_STEP_COUNT] = Scalar[gpu_dtype](0.0)
        states[env, META_OFF + META_IDX_PREV_X] = states[env, QPOS_OFF + 0]
