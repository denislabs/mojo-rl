"""Craftax-Classic environment — Phase 1 skeleton.

This file currently provides:
  - Trait-conforming `CraftaxClassicEnv` struct (CPU + GPU paths)
  - No-op reset / step that wires up state, obs, reward, done shapes correctly
  - Zero-filled GPU kernels that exercise the launch path

Game logic, world gen, and observation extraction land in later phases.
See `docs/CRAFTAX_PORT.md`.
"""

from mojo_rl.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from mojo_rl.nn import dtype as gpu_dtype
from layout import LayoutTensor, Layout
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from .constants import (
    MAP_W,
    MAP_SIZE,
    MAX_TIMESTEPS,
    NUM_ACTIONS,
    OBS_DIM,
    DIR_UP,
    DAY_LENGTH,
    PLAYER_MAX_HEALTH,
    INTRINSIC_MAX,
    NUM_INTRINSICS,
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
)
from .state import (
    STATE_SIZE,
    S_MAP_BASE,
    S_PLAYER_POS,
    S_PLAYER_DIR,
    S_INTRINSICS_BASE,
    S_LIGHT_LEVEL,
    S_IS_SLEEPING,
    S_TIMESTEP,
)
from .world_gen import (
    generate_world_cpu,
    generate_world_inline,
    calculate_light_level,
)
from .game_logic import apply_step_inline, extract_obs_inline
from std.random.philox import Random as PhiloxRandom


# Per-env GPU world-gen workspace: 4 noise fields × MAP_SIZE floats.
comptime WORLD_GEN_WS_PER_ENV: Int = 4 * MAP_SIZE


# ============================================================================
# State / Action wrapper types (used for trait conformance only)
# ============================================================================


@fieldwise_init
struct CraftaxState(Copyable, ImplicitlyCopyable, Movable, State):
    """Trait-required handle; the actual state lives in the env's flat array."""

    var index: Int

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit take: Self):
        self.index = take.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct CraftaxAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Discrete action wrapper (0..NUM_ACTIONS-1)."""

    var value: Int

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    def __init__(out self, *, deinit take: Self):
        self.value = take.value


# ============================================================================
# CraftaxClassicEnv
# ============================================================================


struct CraftaxClassicEnv[DTYPE: DType = DType.float32](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Craftax-Classic environment.

    Phase 1: skeleton only. `reset` zero-fills state, `step` increments the
    timestep and returns zero reward; done fires at MAX_TIMESTEPS. This is the
    minimum that exercises trait conformance and the GPU launch path.
    """

    # Trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = CraftaxState
    comptime ActionType = CraftaxAction

    # GPUDiscreteEnv constants
    comptime STATE_SIZE: Int = STATE_SIZE
    comptime OBS_DIM: Int = OBS_DIM
    comptime NUM_ACTIONS: Int = NUM_ACTIONS
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    # CPU state — flat array indexed by offsets from state.mojo
    var state: InlineArray[Scalar[Self.dtype], STATE_SIZE]
    var done: Bool
    var _rng_counter: UInt64

    def __init__(out self):
        self.state = InlineArray[Scalar[Self.dtype], STATE_SIZE](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False
        self._rng_counter = 0

    # ========================================================================
    # CPU: reset + step
    # ========================================================================

    def reset(mut self) -> CraftaxState:
        """Reset env with a freshly generated world.

        Uses an internal counter as seed; call `reset_with_seed` for a
        specific reproducible map.
        """
        self._rng_counter += 1
        return self.reset_with_seed(self._rng_counter)

    def reset_with_seed(
        mut self, seed: UInt64, always_diamond: Bool = False
    ) -> CraftaxState:
        # Zero everything first.
        for i in range(STATE_SIZE):
            self.state[i] = Scalar[Self.dtype](0.0)

        # Generate world directly into the map section of state.
        # `generate_world_cpu` writes block IDs as Float32; cast to dtype.
        var map_ptr = self.state.unsafe_ptr().bitcast[Float32]() + S_MAP_BASE
        var spawn = generate_world_cpu(seed, map_ptr, always_diamond)
        var py = spawn[0]
        var px = spawn[1]

        self.state[S_PLAYER_POS] = Scalar[Self.dtype](py)
        self.state[S_PLAYER_POS + 1] = Scalar[Self.dtype](px)
        self.state[S_PLAYER_DIR] = Scalar[Self.dtype](DIR_UP)

        # Intrinsics: health/food/drink/energy all at max (9).
        for k in range(NUM_INTRINSICS):
            self.state[S_INTRINSICS_BASE + k] = Scalar[Self.dtype](
                INTRINSIC_MAX
            )

        self.state[S_LIGHT_LEVEL] = Scalar[Self.dtype](
            calculate_light_level(0, DAY_LENGTH)
        )
        self.state[S_IS_SLEEPING] = Scalar[Self.dtype](0.0)
        self.state[S_TIMESTEP] = Scalar[Self.dtype](0.0)

        self.done = False
        return CraftaxState(index=0)

    def step(
        mut self, action: CraftaxAction, verbose: Bool = False
    ) -> Tuple[CraftaxState, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            CraftaxState(index=Int(self.state[S_TIMESTEP])),
            result[0],
            result[1],
        )

    def _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Full Phase-3A step: crafting, do, place, move, plants, intrinsics."""
        self._rng_counter += 1
        var rng = PhiloxRandom(seed=self._rng_counter, offset=0)
        var state_ptr = self.state.unsafe_ptr().bitcast[Float32]()
        var result = apply_step_inline(state_ptr, action, rng)
        self.done = result[1]
        return (Scalar[Self.dtype](result[0]), self.done)

    # ========================================================================
    # Env trait methods
    # ========================================================================

    def get_state(self) -> CraftaxState:
        return CraftaxState(index=Int(self.state[S_TIMESTEP]))

    def close(mut self):
        pass

    def action_from_index(self, action_idx: Int) -> CraftaxAction:
        return CraftaxAction(value=action_idx)

    def num_actions(self) -> Int:
        return NUM_ACTIONS

    def obs_dim(self) -> Int:
        return OBS_DIM

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: CraftaxState) -> Int:
        return state.index

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv (CPU)
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs_arr = InlineArray[Float32, OBS_DIM](fill=Float32(0.0))
        var obs_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            obs_arr.unsafe_ptr().bitcast[Float32]()
        )
        var state_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            self.state.unsafe_ptr().bitcast[Float32]()
        )
        extract_obs_inline(state_ptr, obs_ptr)
        var obs = List[Scalar[Self.dtype]](capacity=OBS_DIM)
        for i in range(OBS_DIM):
            obs.append(Scalar[Self.dtype](obs_arr[i]))
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action)
        return (self.get_obs_list(), result[0], result[1])

    # ========================================================================
    # GPU kernels (Phase 1: zero-fill / increment, exercise launch path only)
    # ========================================================================

    comptime TPB: Int = 256

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        # Per-env scratch (4 noise fields). Lives only for this kernel call.
        var scratch_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * WORLD_GEN_WS_PER_ENV
        )
        var scratch = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE * WORLD_GEN_WS_PER_ENV),
            MutAnyOrigin,
        ](scratch_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed_scalar = Scalar[DType.uint64](rng_seed)

        @parameter
        @always_inline
        def reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            scratch: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE * WORLD_GEN_WS_PER_ENV),
                MutAnyOrigin,
            ],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return

            # Zero this env's state slice (mobs, plants, achievements, etc.).
            for s in range(STATE_SIZE):
                states[e, s] = Scalar[gpu_dtype](0.0)

            # Slice per-env scratch and compute pointers.
            var ws_base = e * WORLD_GEN_WS_PER_ENV
            var scratch_ptr = scratch.ptr + ws_base
            var water_ptr = scratch_ptr
            var mountain_ptr = scratch_ptr + MAP_SIZE
            var path_ptr = scratch_ptr + 2 * MAP_SIZE
            var tree_ptr = scratch_ptr + 3 * MAP_SIZE
            var map_ptr = (
                states.ptr + e * STATE_SIZE + S_MAP_BASE
            )

            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(
                e
            ) + UInt64(1)
            var spawn = generate_world_inline(
                per_env_seed,
                water_ptr,
                mountain_ptr,
                path_ptr,
                tree_ptr,
                map_ptr,
                False,
            )
            states[e, S_PLAYER_POS] = Scalar[gpu_dtype](spawn[0])
            states[e, S_PLAYER_POS + 1] = Scalar[gpu_dtype](spawn[1])
            states[e, S_PLAYER_DIR] = Scalar[gpu_dtype](DIR_UP)
            for k in range(NUM_INTRINSICS):
                states[e, S_INTRINSICS_BASE + k] = Scalar[gpu_dtype](
                    INTRINSIC_MAX
                )
            states[e, S_LIGHT_LEVEL] = Scalar[gpu_dtype](0.7969252)
            states[e, S_IS_SLEEPING] = Scalar[gpu_dtype](0.0)
            states[e, S_TIMESTEP] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[reset_wrapper](
            states,
            scratch,
            seed_scalar,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
        workspace_ptr: Optional[
            UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            UnsafePointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        var scratch_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * WORLD_GEN_WS_PER_ENV
        )
        var scratch = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE * WORLD_GEN_WS_PER_ENV),
            MutAnyOrigin,
        ](scratch_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed_scalar = Scalar[DType.uint64](rng_seed)

        @parameter
        @always_inline
        def selective_reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            scratch: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE * WORLD_GEN_WS_PER_ENV),
                MutAnyOrigin,
            ],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return
            if dones[e] <= Scalar[gpu_dtype](0.5):
                return

            for s in range(STATE_SIZE):
                states[e, s] = Scalar[gpu_dtype](0.0)

            var ws_base = e * WORLD_GEN_WS_PER_ENV
            var scratch_ptr = scratch.ptr + ws_base
            var map_ptr = (
                states.ptr + e * STATE_SIZE + S_MAP_BASE
            )
            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(
                e
            ) + UInt64(1)
            var spawn = generate_world_inline(
                per_env_seed,
                scratch_ptr,
                scratch_ptr + MAP_SIZE,
                scratch_ptr + 2 * MAP_SIZE,
                scratch_ptr + 3 * MAP_SIZE,
                map_ptr,
                False,
            )
            states[e, S_PLAYER_POS] = Scalar[gpu_dtype](spawn[0])
            states[e, S_PLAYER_POS + 1] = Scalar[gpu_dtype](spawn[1])
            states[e, S_PLAYER_DIR] = Scalar[gpu_dtype](DIR_UP)
            for k in range(NUM_INTRINSICS):
                states[e, S_INTRINSICS_BASE + k] = Scalar[gpu_dtype](
                    INTRINSIC_MAX
                )
            states[e, S_LIGHT_LEVEL] = Scalar[gpu_dtype](0.7969252)
            states[e, S_IS_SLEEPING] = Scalar[gpu_dtype](0.0)
            states[e, S_TIMESTEP] = Scalar[gpu_dtype](0.0)
            dones[e] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[selective_reset_wrapper](
            states,
            dones,
            scratch,
            seed_scalar,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut terminated_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        workspace_ptr: Optional[
            UnsafePointer[Scalar[gpu_dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            UnsafePointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
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
            gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        var actions = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var seed_scalar = Scalar[DType.uint64](rng_seed)

        @parameter
        @always_inline
        def step_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
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
                gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return

            var action = Int(actions[e])
            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(
                e
            ) + UInt64(1)
            var rng = PhiloxRandom(seed=per_env_seed, offset=0)
            var state_ptr = states.ptr + e * STATE_SIZE
            var result = apply_step_inline(state_ptr, action, rng)

            rewards[e] = Scalar[gpu_dtype](result[0])
            dones[e] = Scalar[gpu_dtype](1.0) if result[1] else Scalar[
                gpu_dtype
            ](0.0)
            terminated_out[e] = Scalar[gpu_dtype](0.0)

            # Symbolic obs: write into this env's slice of the obs buffer.
            var obs_ptr = obs.ptr + e * OBS_DIM
            extract_obs_inline(state_ptr, obs_ptr)

        ctx.enqueue_function[step_wrapper](
            states,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            seed_scalar,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var obs = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @parameter
        @always_inline
        def extract_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            obs: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return
            var state_ptr = states.ptr + e * STATE_SIZE
            var obs_ptr = obs.ptr + e * OBS_DIM
            extract_obs_inline(state_ptr, obs_ptr)

        ctx.enqueue_function[extract_wrapper](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype],) raises:
        pass

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        pass

    # ========================================================================
    # RenderableEnv (Phase 1: all no-ops; renderer comes later)
    # ========================================================================

    def init_renderer(mut self) raises -> Bool:
        return False

    def render_frame(mut self) raises -> None:
        pass

    def close_renderer(mut self) raises -> None:
        pass

    def is_renderer_open(self) -> Bool:
        return False

    def check_renderer_quit(mut self) -> Bool:
        return False

    def renderer_delay(self, ms: Int) -> None:
        pass

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False
