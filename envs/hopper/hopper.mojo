"""Hopper Environment - thin wrapper around MuJoCoEnv[HopperConfig].

Adds RenderableEnv conformance and env-specific accessors (body positions, quaternions).
All physics, GPU kernels, and shared trait methods live in MuJoCoEnv.
"""

from collections import InlineArray
from memory import alloc

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
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data

from .hopper_def import (
    HopperModel,
    HopperRenderer,
    HopperParams,
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
from .hopper_config import HopperConfig
from ..mujoco_env import MuJoCoEnv


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
    """Hopper environment — thin wrapper around MuJoCoEnv[HopperConfig].

    Adds RenderableEnv conformance and env-specific body position/quaternion accessors.
    All physics, step logic, GPU kernels, and shared trait methods live in MuJoCoEnv.
    """

    # Inner generic env
    comptime _InnerEnv = MuJoCoEnv[HopperConfig, Self.DTYPE, Self.TERMINATE_ON_UNHEALTHY]
    var _env: Self._InnerEnv

    # Renderer (optional)
    var _renderer: UnsafePointer[HopperRenderer, MutAnyOrigin]
    var _renderer_initialized: Bool

    # ---- Forward comptime constants from inner env ----
    comptime dtype = Self.DTYPE
    # Use module-level constants (which resolve to literal Ints from HopperParams)
    # so external code can pass ContAction[3] / ObsState[11] without mismatch
    comptime StateType = ObsState[OBS_DIM]
    comptime ActionType = ContAction[ACTION_DIM]
    comptime OBS_DIM: Int = OBS_DIM
    comptime ACTION_DIM: Int = ACTION_DIM
    comptime NQ: Int = HopperConfig.NQ
    comptime NV: Int = HopperConfig.NV
    comptime NUM_BODIES: Int = HopperConfig.NBODY
    comptime NUM_JOINTS: Int = HopperConfig.NJOINT
    comptime MAX_CONTACTS: Int = HopperConfig.MAX_CONTACTS
    comptime NGEOM: Int = HopperConfig.NGEOM
    comptime STATE_SIZE: Int = Self._InnerEnv.STATE_SIZE
    comptime STEP_WS_SHARED: Int = Self._InnerEnv.STEP_WS_SHARED
    comptime STEP_WS_PER_ENV: Int = Self._InnerEnv.STEP_WS_PER_ENV

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        max_steps: Int = 1000,
        frame_skip: Int = 4,
    ):
        self._env = Self._InnerEnv(max_steps=max_steps, frame_skip=frame_skip)
        self._renderer = UnsafePointer[HopperRenderer, MutAnyOrigin]()
        self._renderer_initialized = False

    # =========================================================================
    # BoxContinuousActionEnv — delegate to inner env
    # =========================================================================

    fn obs_dim(self) -> Int:
        return self._env.obs_dim()

    fn action_dim(self) -> Int:
        return self._env.action_dim()

    fn action_low(self) -> Scalar[Self.dtype]:
        return self._env.action_low()

    fn action_high(self) -> Scalar[Self.dtype]:
        return self._env.action_high()

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        return self._env.step_continuous(action)

    fn step_continuous_vec[
        DTYPE2: DType
    ](
        mut self, action: List[Scalar[DTYPE2]], verbose: Bool = False
    ) -> Tuple[List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool]:
        return self._env.step_continuous_vec(action, verbose=verbose)

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        return self._env.get_obs_list()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        return self._env.reset_obs_list()

    # =========================================================================
    # Env trait — delegate
    # =========================================================================

    fn step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        var actions = List[Float64](capacity=Self.ACTION_DIM)
        for i in range(Self.ACTION_DIM):
            actions.append(Float64(action.data[i]))
        var result = self._env.step_continuous_vec(actions, verbose=verbose)
        var obs = Self.StateType()
        for i in range(Self.OBS_DIM):
            obs.data[i] = Float64(result[0][i])
        return (obs^, Scalar[Self.dtype](result[1]), result[2])

    fn get_state(self) -> Self.StateType:
        var obs_list = self._env.get_obs_list()
        var obs = Self.StateType()
        for i in range(Self.OBS_DIM):
            obs.data[i] = Float64(obs_list[i])
        return obs^

    fn reset(mut self) -> Self.StateType:
        var obs_list = self._env.reset_obs_list()
        var obs = Self.StateType()
        for i in range(Self.OBS_DIM):
            obs.data[i] = Float64(obs_list[i])
        return obs^

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
            qpos[i] = self._env.get_qpos(i)
        return qpos^

    fn get_qvel(self) -> InlineArray[Scalar[Self.DTYPE], 6]:
        var qvel = InlineArray[Scalar[Self.DTYPE], 6](uninitialized=True)
        for i in range(6):
            qvel[i] = self._env.get_qvel(i)
        return qvel^

    fn get_body_position(
        self, body_id: Int
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        return (
            self._env.get_xpos(body_id * 3 + 0),
            self._env.get_xpos(body_id * 3 + 1),
            self._env.get_xpos(body_id * 3 + 2),
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
            self._env.get_xquat(body_id * 4 + 0),
            self._env.get_xquat(body_id * 4 + 1),
            self._env.get_xquat(body_id * 4 + 2),
            self._env.get_xquat(body_id * 4 + 3),
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
        return self._env.get_x_velocity()

    fn get_current_step(self) -> Int:
        return self._env.get_current_step()

    fn get_max_steps(self) -> Int:
        return self._env.get_max_steps()

    fn is_done(self) -> Bool:
        return self._env.is_done()

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

        var xpos = InlineArray[Scalar[Self.DTYPE], NBODY * 3](
            uninitialized=True
        )
        var xquat = InlineArray[Scalar[Self.DTYPE], NBODY * 4](
            uninitialized=True
        )
        for i in range(NBODY * 3):
            xpos[i] = self._env.get_xpos(i)
        for i in range(NBODY * 4):
            xquat[i] = self._env.get_xquat(i)
        self._renderer[].render_from_body_state(
            xpos,
            xquat,
            NBODY,
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
    # GPUContinuousEnv Interface — delegate to MuJoCoEnv static methods
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[gpu_dtype],
        actions: DeviceBuffer[gpu_dtype],
        mut rewards: DeviceBuffer[gpu_dtype],
        mut dones: DeviceBuffer[gpu_dtype],
        mut obs: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[gpu_dtype]] = [],
        workspace_ptr: UnsafePointer[
            Scalar[gpu_dtype], MutAnyOrigin
        ] = UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin](),
    ) raises:
        Self._InnerEnv.step_kernel_gpu[BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM](
            ctx, states, actions, rewards, dones, obs,
            rng_seed, curriculum_values, workspace_ptr,
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        Self._InnerEnv.reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states, rng_seed,
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[gpu_dtype],
        mut dones: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
    ) raises:
        Self._InnerEnv.selective_reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states, dones, rng_seed,
        )

    @staticmethod
    fn extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[gpu_dtype],
        mut obs: DeviceBuffer[gpu_dtype],
    ) raises:
        Self._InnerEnv.extract_obs_kernel_gpu[BATCH_SIZE, STATE_SIZE, OBS_DIM](
            ctx, states, obs,
        )

    @staticmethod
    fn init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype]) raises:
        Self._InnerEnv.init_step_workspace_gpu[BATCH_SIZE](ctx, workspace_buf)

    @staticmethod
    fn update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        Self._InnerEnv.update_curriculum_gpu(ctx, workspace_buf, curriculum_values)
