"""Generic physics3d environment on the PER-FIELD tensor path (migration P5).

`Phyics3dEnvFields[MODEL_DEF, CONFIG]` is the fields-path counterpart of
`Phyics3dEnv`: same MODEL_DEF/CONFIG parameterization, same
`BoxContinuousActionEnv` surface (drop-in for the CPU training drivers), but
the PHYSICS runs through `RK4IntegratorFields` over `DataFields` /
`ModelFields` — no state slab, no workspace slab, no offsets.

Bridge design (transitional, dies at P6): a legacy `Model`/`Data` pair is
kept ONLY as an adapter for the existing comptime hooks that still speak
struct-Data — `MODEL_DEF.reset_data` / `apply_actions` / `extract_obs` and
the CONFIG reward/termination/pre-step hooks. Per step that costs copying
qpos/qvel/qfrc (+ the FK products xpos/xquat/xipos, which the fields step
already computed) between the bridge and the field tensors — O(NQ+NV+NBODY)
scalars, negligible next to the physics. Zero changes to any MODEL_DEF or
CONFIG.

Scope (mirrors what the fields path supports today):
- Contacts + joint limits: `RK4IntegratorFields` runs detection + the PGS
  contact solve (limits inside) after every stage — hopper/walker-class
  locomotion envs are in scope. Equality/tendon-constrained models are not.
- Fluid-force models raise (same guard as the integrators).
- cvel/cfrc_ext-based rewards (ant contact cost) not synced — contact envs
  are out of scope anyway.
- CPU target (single-env driver ABI). The GPU-batched facade
  (`BatchedEnv`-style, driver-owned IO buffers) is the follow-up slice.
"""

from std.collections import InlineArray
from std.memory import alloc
from std.random import random_float64
from std.gpu.host import DeviceContext

from mojo_rl.core.env_traits import BoxContinuousActionEnv, RenderableEnv
from mojo_rl.core.obs_state import ObsState
from mojo_rl.core.cont_action import ContAction

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.model.model_renderer import ModelRenderer
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.integrator.rk4_fields import RK4IntegratorFields
from mojo_rl.physics3d.gpu.buffer_utils import copy_model_to_buffer
from mojo_rl.physics3d.gpu.constants import model_size_with_invweight

from .phyics3d_env_config import Phyics3dEnvConfig


struct Phyics3dEnvFields[
    MODEL_DEF: ModelDefLike,
    CONFIG: Phyics3dEnvConfig,
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
    SOLVER: StaticString = "newton",
](BoxContinuousActionEnv, RenderableEnv):
    """Generic MuJoCo env, physics on the per-field tensor path. See module
    docstring for the bridge design and scope.

    SOLVER defaults to "newton" — the legacy env default physics
    (CONFIG.physics_substep = RK4 + Newton). This facade steps on CPU,
    where the GPU-only PARALLEL_GPU / CRBA_TREEWALK knobs never apply
    (CPU is always serial + dense, matching legacy production's CPU
    side)."""

    comptime dtype = Self.DTYPE
    comptime StateType = ObsState[Self.MODEL_DEF.OBS_DIM]
    comptime ActionType = ContAction[Self.MODEL_DEF.ACTION_DIM]
    comptime NAME: String = "Physics3dEnvFields"

    comptime OBS_DIM: Int = Self.MODEL_DEF.OBS_DIM
    comptime ACTION_DIM: Int = Self.MODEL_DEF.ACTION_DIM
    comptime NQ: Int = Self.MODEL_DEF.NQ
    comptime NV: Int = Self.MODEL_DEF.NV
    comptime NBODY: Int = Self.MODEL_DEF.NBODY
    comptime NJOINT: Int = Self.MODEL_DEF.NJOINT
    comptime MAX_CONTACTS: Int = Self.MODEL_DEF.MAX_CONTACTS
    comptime NGEOM: Int = Self.MODEL_DEF.NGEOM
    comptime NSITE: Int = Self.MODEL_DEF.NSITE
    comptime MS: Int = model_size_with_invweight[
        Self.NBODY,
        Self.NJOINT,
        Self.NV,
        Self.NGEOM,
        NEQUALITY=Self.MODEL_DEF.MAX_EQUALITY,
        NTENDON=Self.MODEL_DEF.MAX_TENDON,
        NSITE=Self.NSITE,
    ]()

    # Legacy bridge (hooks adapter only — physics never touches it)
    var model: Model[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NBODY,
        Self.NJOINT,
        Self.MAX_CONTACTS,
        Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY,
        Self.MODEL_DEF.CONE_TYPE,
        Self.MODEL_DEF.MAX_TENDON,
        Self.NSITE,
    ]
    var data: Data[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NBODY,
        Self.NJOINT,
        Self.MAX_CONTACTS,
        Self.NSITE,
    ]

    # Fields path (the actual physics state)
    var mf: ModelFields[
        Self.DTYPE,
        Self.NV,
        Self.NBODY,
        Self.NJOINT,
        Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY,
        Self.MODEL_DEF.MAX_TENDON,
        Self.NSITE,
    ]
    var d: DataFields[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NBODY,
        Self.MAX_CONTACTS,
        Self.NSITE,
        1,
    ]
    var integ: RK4IntegratorFields[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NBODY,
        Self.NJOINT,
        Self.MAX_CONTACTS,
        Self.NGEOM,
        Self.MODEL_DEF.MAX_EQUALITY,
        Self.MODEL_DEF.MAX_TENDON,
        Self.NSITE,
        0,
        0,
        Self.MODEL_DEF.CONE_TYPE,
        1,
        SOLVER = Self.SOLVER,
    ]

    var max_steps: Int
    var current_step: Int
    var frame_skip: Int
    var _last_terminated: Bool
    var prev_x: Scalar[Self.DTYPE]

    # Renderer (optional; RenderableEnv). Reads the bridge `self.data` FK
    # products, which the fields step re-syncs every frame.
    var _renderer: Optional[
        UnsafePointer[ModelRenderer[Self.MODEL_DEF], MutUntrackedOrigin]
    ]
    var _renderer_initialized: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        max_steps: Int = Self.CONFIG.MAX_STEPS,
        frame_skip: Int = Self.CONFIG.FRAME_SKIP,
    ) raises:
        """`ctx` is used ONCE, for the host staging buffer of the model
        flattening bridge (no device work on the CPU path)."""
        self.max_steps = max_steps
        self.current_step = 0
        self.frame_skip = frame_skip
        self.prev_x = Scalar[Self.DTYPE](0.0)
        self._last_terminated = False
        self._renderer = None
        self._renderer_initialized = False

        self.model = type_of(self.model)()
        self.data = type_of(self.data)()
        Self.MODEL_DEF.setup_model_and_data(self.model, self.data)

        # Bridge the static model into record tensors via the existing
        # flattening (bit-identical by construction; direct parser fill
        # lands at sunset).
        var hb = ctx.enqueue_create_host_buffer[Self.DTYPE](Self.MS)
        ctx.synchronize()
        copy_model_to_buffer(self.model, hb)
        var flat = List[Scalar[Self.DTYPE]](
            length=Self.MS, fill=Scalar[Self.DTYPE](0)
        )
        for i in range(Self.MS):
            flat[i] = hb[i]
        self.mf = type_of(self.mf)()
        self.mf.load_from_slab(flat)

        self.d = type_of(self.d)()
        self.integ = type_of(self.integ)()

        self._sync_fields_from_bridge()
        Self.CONFIG.pre_step_cpu(self.data, self.prev_x)

        # Fluid guard once here (the trait step() is non-raising).
        from mojo_rl.physics3d.gpu.constants import (
            MODEL_META_IDX_DENSITY,
            MODEL_META_IDX_VISCOSITY,
        )
        if (
            self.mf.meta.data[MODEL_META_IDX_DENSITY] != 0
            or self.mf.meta.data[MODEL_META_IDX_VISCOSITY] != 0
        ):
            raise Error(
                "Phyics3dEnvFields: fluid forces not ported to the fields"
                " path yet"
            )

    # ── bridge sync (transitional; O(NQ+NV+NBODY) scalars) ───────────────
    def _sync_fields_from_bridge(mut self):
        for i in range(Self.NQ):
            self.d.qpos.data[i] = self.data.qpos[i]
        for i in range(Self.NV):
            self.d.qvel.data[i] = self.data.qvel[i]

    def _sync_bridge_from_fields(mut self):
        for i in range(Self.NQ):
            self.data.qpos[i] = self.d.qpos.data[i]
        for i in range(Self.NV):
            self.data.qvel[i] = self.d.qvel.data[i]
            self.data.qacc[i] = self.d.qacc.data[i]
        # FK products (already computed by the fields step) so extract_obs /
        # reward hooks that read world poses see fresh values.
        for i in range(Self.NBODY * 3):
            self.data.xpos[i] = self.d.xpos.data[i]
            self.data.xipos[i] = self.d.xipos.data[i]
        for i in range(Self.NBODY * 4):
            self.data.xquat[i] = self.d.xquat.data[i]
        comptime if Self.NSITE > 0:
            for i in range(Self.NSITE * 3):
                self.data.site_xpos[i] = self.d.site_xpos.data[i]

    # ── state management ─────────────────────────────────────────────────
    def _reset_state(mut self):
        """Legacy reset semantics (qpos0 + uniform noise + custom hook),
        then hand the state to the fields path."""
        Self.MODEL_DEF.reset_data(self.data)
        var noise_scale = Self.CONFIG.get_reset_noise()
        if noise_scale > 0.0:
            for i in range(Self.NQ):
                var noise = Scalar[Self.dtype](
                    (random_float64() * 2.0 - 1.0) * noise_scale
                )
                self.data.qpos[i] = self.data.qpos[i] + noise
            for i in range(Self.NV):
                var noise = Scalar[Self.dtype](
                    (random_float64() * 2.0 - 1.0) * noise_scale
                )
                self.data.qvel[i] = self.data.qvel[i] + noise
        Self.CONFIG.custom_reset_cpu(self.model, self.data)
        forward_kinematics(self.model, self.data)  # fresh obs before step 1
        self.current_step = 0
        self.prev_x = Scalar[Self.dtype](0)
        self._last_terminated = False
        Self.CONFIG.pre_step_cpu(self.data, self.prev_x)
        self._sync_fields_from_bridge()

    def set_state(mut self, qpos: List[Float64], qvel: List[Float64]):
        """Deterministic state injection (tests / eval)."""
        for i in range(min(Self.NQ, len(qpos))):
            self.data.qpos[i] = Scalar[Self.dtype](qpos[i])
        for i in range(min(Self.NV, len(qvel))):
            self.data.qvel[i] = Scalar[Self.dtype](qvel[i])
        forward_kinematics(self.model, self.data)
        self._sync_fields_from_bridge()

    def _get_obs(self) -> ObsState[Self.MODEL_DEF.OBS_DIM]:
        var obs_list = List[Scalar[Self.DTYPE]](capacity=Self.OBS_DIM)
        var custom = Self.CONFIG.custom_extract_obs_cpu(self.data, obs_list)
        if not custom:
            Self.MODEL_DEF.extract_obs(self.data, obs_list)
        var obs = ObsState[Self.MODEL_DEF.OBS_DIM]()
        for i in range(Self.OBS_DIM):
            obs.data[i] = Float64(obs_list[i])
        return obs^

    # ── Env interface ─────────────────────────────────────────────────────
    def reset(mut self) -> Self.StateType:
        self._reset_state()
        return self._get_obs()

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        Self.CONFIG.pre_step_cpu(self.data, self.prev_x)

        # Actions via the existing comptime actuator logic (per-motor
        # ctrlrange clamp + gear), then hand qfrc to the fields path.
        var clamped_action = action.copy()
        var action_list = clamped_action.to_list()
        var custom_applied = Self.CONFIG.custom_apply_actions_cpu(
            self.data, action_list
        )
        if not custom_applied:
            Self.MODEL_DEF.apply_actions(self.data, action_list)
        for i in range(Self.NV):
            self.d.qfrc.data[i] = self.data.qfrc[i]

        # Physics: fields RK4 with per-stage contact/limit solving.
        for _ in range(self.frame_skip):
            try:
                # CPU target: cannot actually raise (the `raises` on the
                # dispatchers exists for the GPU branch's ctx handling).
                self.integ.step["cpu"](self.d, self.mf)
            except e:
                print("Phyics3dEnvFields.step: physics error:", e)

        self._sync_bridge_from_fields()
        self.current_step += 1

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
        self._last_terminated = terminated
        return (self._get_obs(), Scalar[Self.dtype](reward), done)

    def was_terminated(self) -> Bool:
        return self._last_terminated

    def get_state(self) -> Self.StateType:
        return self._get_obs()

    def close(mut self):
        pass

    # ── Render accessors (read the bridge `self.data`) ────────────────────
    def get_xpos(self, idx: Int) -> Scalar[Self.DTYPE]:
        return self.data.xpos[idx]

    def get_xquat(self, idx: Int) -> Scalar[Self.DTYPE]:
        return self.data.xquat[idx]

    def get_x_velocity(self) -> Scalar[Self.DTYPE]:
        return self.data.qvel[0]

    # ── RenderableEnv (mirrors Phyics3dEnv; renders the fields physics via
    #    the bridge poses) ──────────────────────────────────────────────────
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

        self._renderer.value().init_pointee_move(renderer^)
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        if not self._renderer.value()[].is_open():
            return

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
        self._renderer.value()[].render_from_body_state(
            xpos,
            xquat,
            Self.MODEL_DEF.NBODY,
            vel_x=Float64(self.get_x_velocity()),
        )

    def close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().free()
        self._renderer_initialized = False

    def is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].is_open()

    def check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].check_quit()

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].delay(ms)

    def renderer_is_paused(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].renderer.is_paused

    def renderer_step_once(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].renderer.step_once

    def start_recording(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer.start_recording(filename, fps, skip)

    def stop_recording(mut self) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer.stop_recording()

    # ── ContinuousStateEnv ────────────────────────────────────────────────
    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=Self.OBS_DIM)
        var custom = Self.CONFIG.custom_extract_obs_cpu(self.data, obs)
        if not custom:
            Self.MODEL_DEF.extract_obs(self.data, obs)
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        self._reset_state()
        return self.get_obs_list()

    def obs_dim(self) -> Int:
        return Self.OBS_DIM

    # ── ContinuousActionEnv ───────────────────────────────────────────────
    def action_dim(self) -> Int:
        return Self.ACTION_DIM

    def action_low(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](Self.MODEL_DEF.CTRL_MIN)

    def action_high(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](Self.MODEL_DEF.CTRL_MAX)

    # ── BoxContinuousActionEnv ────────────────────────────────────────────
    def step_continuous[
        DTYPE2: DType
    ](mut self, action: Scalar[DTYPE2]) -> Tuple[
        List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool
    ]:
        var actions = List[Scalar[DTYPE2]]()
        for _ in range(Self.ACTION_DIM):
            actions.append(Scalar[DTYPE2](action))
        return self.step_continuous_vec[DTYPE2](actions)

    def step_continuous_vec[
        DTYPE2: DType
    ](
        mut self, action: List[Scalar[DTYPE2]], verbose: Bool = False
    ) -> Tuple[List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool]:
        var act = ContAction[Self.MODEL_DEF.ACTION_DIM]()
        for i in range(min(Self.ACTION_DIM, len(action))):
            act.data[i] = Float64(action[i])
        var result = self.step(act, verbose=verbose)
        var obs_list = self.get_obs_list()
        var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
        for i in range(Self.OBS_DIM):
            obs.append(Scalar[DTYPE2](obs_list[i]))
        return (obs^, Scalar[DTYPE2](result[1]), result[2])
