"""Typed workspace: named LayoutTensor views over a flat memory buffer.

Provides compile-time offset computation and named accessor methods,
eliminating manual pointer arithmetic and repetitive view boilerplate.

Works with both CPU (UnsafePointer from List/heap) and GPU (unsafe_ptr
from DeviceBuffer). The workspace does NOT own memory — the caller
manages the underlying buffer's lifetime.

Usage (CPU):
    var data = WS.alloc_cpu()
    var ws = WS(data.unsafe_ptr())
    var obs_t = ws.next_act()  # typed LayoutTensor view

Usage (GPU):
    var buf = WS.alloc_gpu(ctx)
    var ws = WS(buf.unsafe_ptr())
    var obs_t = ws.next_act()  # same API, GPU memory
"""

from layout import Layout, LayoutTensor
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import dtype


# =============================================================================
# OffPolicyTrainWS — Training workspace for off-policy continuous agents
# =============================================================================


struct OffPolicyTrainWS[
    BS: Int,          # Batch size
    OBS: Int,         # Observation dimension
    ACT: Int,         # Action dimension (post-squash for SAC)
    ACTOR_OUT: Int,   # Actor raw output dim (2*ACT for SAC, ACT for DDPG)
    CI: Int,          # Critic input dimension (OBS + ACT)
    CO: Int,          # Critic output dimension
    CCS: Int,         # Critic cache size per sample
    ACS: Int,         # Actor cache size per sample
    CRITIC_WS: Int,   # Critic workspace per sample (for GPU kernels)
    ACTOR_WS: Int,    # Actor workspace per sample (for GPU kernels)
    NUM_CRITICS: Int = 1,
    STRAT_WS: Int = 0,
    TARGET_STRAT_WS: Int = 0,
](ImplicitlyCopyable, Movable):
    """Typed workspace providing named LayoutTensor views over flat memory.

    All offsets are computed at compile time. The struct is just a pointer
    wrapper — zero overhead, zero allocation, works on CPU and GPU.

    Layout:
        Region 1: Target computation (next_act, next_lp, next_ci, q targets)
        Region 2: Critic update (concat input, Q outputs/caches/grads per critic)
        Region 3: Actor update (actor output, cache, workspace)
        Region 4: Strategy workspaces (shared between ActorLoss and TargetAction)
    """

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    # --- Region 1: Target computation ---
    comptime _O_NEXT_ACT: Int = 0
    comptime _O_NEXT_LP: Int = Self._O_NEXT_ACT + Self.BS * Self.ACT
    comptime _O_NEXT_CI: Int = Self._O_NEXT_LP + Self.BS
    comptime _O_NEXT_Q: Int = Self._O_NEXT_CI + Self.BS * Self.CI
    # Per-critic target Q values: contiguous [BS*CO] per critic
    comptime _O_TARGETS: Int = Self._O_NEXT_Q + Self.NUM_CRITICS * Self.BS * Self.CO

    # --- Region 2: Critic update ---
    comptime _O_CI: Int = Self._O_TARGETS + Self.BS
    # Per-critic outputs and caches: contiguous regions indexed by critic_idx
    comptime _O_Q_OUTS: Int = Self._O_CI + Self.BS * Self.CI
    comptime _O_Q_CACHES: Int = Self._O_Q_OUTS + Self.NUM_CRITICS * Self.BS * Self.CO
    comptime _CWS_EACH: Int = max(1, Self.BS * Self.CRITIC_WS)
    comptime _O_CRITIC_WS_START: Int = Self._O_Q_CACHES + Self.NUM_CRITICS * Self.BS * Self.CCS
    comptime _O_Q_GRAD: Int = Self._O_CRITIC_WS_START + Self.NUM_CRITICS * Self._CWS_EACH
    comptime _O_D_CI: Int = Self._O_Q_GRAD + Self.BS * Self.CO

    # --- Region 3: Actor update workspace ---
    comptime _O_ACTOR_WS: Int = Self._O_D_CI + Self.BS * Self.CI

    # --- Region 4: Strategy workspaces ---
    comptime _O_STRAT_WS: Int = Self._O_ACTOR_WS + max(1, Self.BS * Self.ACTOR_WS)
    comptime _O_TARGET_STRAT_WS: Int = Self._O_STRAT_WS + max(1, Self.STRAT_WS)

    # --- Total ---
    comptime TOTAL_SIZE: Int = Self._O_TARGET_STRAT_WS + max(1, Self.TARGET_STRAT_WS)

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    # =========================================================================
    # Allocation helpers
    # =========================================================================

    @staticmethod
    def alloc_cpu() -> List[Scalar[dtype]]:
        """Allocate a zero-initialized CPU buffer for this workspace."""
        var data = List[Scalar[dtype]](capacity=Self.TOTAL_SIZE)
        for _ in range(Self.TOTAL_SIZE):
            data.append(Scalar[dtype](0))
        return data^

    @staticmethod
    def alloc_gpu(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
        """Allocate a GPU buffer for this workspace."""
        return ctx.enqueue_create_buffer[dtype](Self.TOTAL_SIZE)

    # =========================================================================
    # Region 1: Target computation views
    # =========================================================================

    def next_act(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.ACT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.ACT), MutAnyOrigin
        ](self.ptr + Self._O_NEXT_ACT)

    def next_lp(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.BS), MutAnyOrigin](
            self.ptr + Self._O_NEXT_LP
        )

    def next_ci(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.CI), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.CI), MutAnyOrigin
        ](self.ptr + Self._O_NEXT_CI)

    def next_q(self, critic_idx: Int = 0) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.CO), MutAnyOrigin
    ]:
        """Target Q-value output for critic `critic_idx`."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.CO), MutAnyOrigin
        ](self.ptr + Self._O_NEXT_Q + critic_idx * Self.BS * Self.CO)

    def targets(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.BS), MutAnyOrigin](
            self.ptr + Self._O_TARGETS
        )

    # =========================================================================
    # Region 2: Critic update views
    # =========================================================================

    def ci(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.CI), MutAnyOrigin
    ]:
        """Concatenated [obs, act] critic input."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.CI), MutAnyOrigin
        ](self.ptr + Self._O_CI)

    def q_out(self, critic_idx: Int = 0) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.CO), MutAnyOrigin
    ]:
        """Q-value output for critic `critic_idx`."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.CO), MutAnyOrigin
        ](self.ptr + Self._O_Q_OUTS + critic_idx * Self.BS * Self.CO)

    def q_cache(self, critic_idx: Int = 0) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.CCS), MutAnyOrigin
    ]:
        """Activation cache for critic `critic_idx`."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.CCS), MutAnyOrigin
        ](self.ptr + Self._O_Q_CACHES + critic_idx * Self.BS * Self.CCS)

    def critic_ws_ptr(self, critic_idx: Int = 0) -> UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]:
        """Raw pointer to critic workspace for critic `critic_idx`."""
        return self.ptr + Self._O_CRITIC_WS_START + critic_idx * Self._CWS_EACH

    def q_grad(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.CO), MutAnyOrigin
    ]:
        """Shared Q-value gradient (reused across critics)."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.CO), MutAnyOrigin
        ](self.ptr + Self._O_Q_GRAD)

    def d_ci(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.CI), MutAnyOrigin
    ]:
        """Gradient w.r.t. critic input."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.CI), MutAnyOrigin
        ](self.ptr + Self._O_D_CI)

    # =========================================================================
    # Region 3/4: Strategy & actor workspaces (raw pointers)
    # =========================================================================

    def actor_ws_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        """Raw pointer to actor network workspace region."""
        return self.ptr + Self._O_ACTOR_WS

    def strat_ws_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        """Raw pointer to strategy workspace (shared ActorLoss / TargetAction)."""
        return self.ptr + Self._O_STRAT_WS

    def target_strat_ws_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        """Raw pointer to target strategy workspace."""
        return self.ptr + Self._O_TARGET_STRAT_WS


# =============================================================================
# SampleBatch — Typed views over replay buffer sample output
# =============================================================================


struct SampleBatch[
    BS: Int,
    OBS: Int,
    ACT: Int,
](ImplicitlyCopyable, Movable):
    """Typed views over replay buffer sample outputs.

    Works with both CPU (InlineArray) and GPU (DeviceBuffer) backing.
    """

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    comptime _O_OBS: Int = 0
    comptime _O_ACT: Int = Self._O_OBS + Self.BS * Self.OBS
    comptime _O_REW: Int = Self._O_ACT + Self.BS * Self.ACT
    comptime _O_NOBS: Int = Self._O_REW + Self.BS
    comptime _O_DONE: Int = Self._O_NOBS + Self.BS * Self.OBS
    comptime TOTAL_SIZE: Int = Self._O_DONE + Self.BS

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    @staticmethod
    def alloc_cpu() -> List[Scalar[dtype]]:
        var data = List[Scalar[dtype]](capacity=Self.TOTAL_SIZE)
        for _ in range(Self.TOTAL_SIZE):
            data.append(Scalar[dtype](0))
        return data^

    @staticmethod
    def alloc_gpu(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
        return ctx.enqueue_create_buffer[dtype](Self.TOTAL_SIZE)

    def obs(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.OBS), MutAnyOrigin
        ](self.ptr + Self._O_OBS)

    def act(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.ACT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.ACT), MutAnyOrigin
        ](self.ptr + Self._O_ACT)

    def rew(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.BS), MutAnyOrigin](
            self.ptr + Self._O_REW
        )

    def nobs(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.OBS), MutAnyOrigin
        ](self.ptr + Self._O_NOBS)

    def done(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.BS), MutAnyOrigin](
            self.ptr + Self._O_DONE
        )


# =============================================================================
# ExplorationWS — Inference-time buffers (sized by n_envs, not batch_size)
# =============================================================================


struct ExplorationWS[
    MAX_N_ENVS: Int,
    ACTOR_OUT: Int,
    ACTOR_WS: Int,
](ImplicitlyCopyable, Movable):
    """Workspace for inference-time exploration (sized by num environments)."""

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    comptime _O_RAW_ACT: Int = 0
    comptime _O_INF_WS: Int = Self._O_RAW_ACT + Self.MAX_N_ENVS * Self.ACTOR_OUT
    comptime TOTAL_SIZE: Int = Self._O_INF_WS + max(1, Self.MAX_N_ENVS * Self.ACTOR_WS)

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    def raw_act[N_ENVS: Int](self) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS, Self.ACTOR_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTOR_OUT), MutAnyOrigin
        ](self.ptr + Self._O_RAW_ACT)

    def inf_ws_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_INF_WS


# =============================================================================
# RolloutWS — Rollout storage for on-policy agents (PPO / A2C)
# =============================================================================


struct RolloutWS[
    RT: Int,       # ROLLOUT_TOTAL = rollout_len * n_envs
    OBS: Int,      # Observation dimension
](ImplicitlyCopyable, Movable):
    """Typed workspace for on-policy rollout buffers.

    Consolidates obs, actions, log_probs, values, rewards, dones,
    advantages, and returns into a single flat GPU allocation.

    Layout (all contiguous, dtype):
        obs:        [RT * OBS]
        actions:    [RT]
        log_probs:  [RT]
        values:     [RT]
        rewards:    [RT]
        dones:      [RT]
        advantages: [RT]
        returns:    [RT]
    """

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    comptime _O_OBS: Int = 0
    comptime _O_ACTIONS: Int = Self._O_OBS + Self.RT * Self.OBS
    comptime _O_LOG_PROBS: Int = Self._O_ACTIONS + Self.RT
    comptime _O_VALUES: Int = Self._O_LOG_PROBS + Self.RT
    comptime _O_REWARDS: Int = Self._O_VALUES + Self.RT
    comptime _O_DONES: Int = Self._O_REWARDS + Self.RT
    comptime _O_ADVANTAGES: Int = Self._O_DONES + Self.RT
    comptime _O_RETURNS: Int = Self._O_ADVANTAGES + Self.RT
    comptime TOTAL_SIZE: Int = Self._O_RETURNS + Self.RT

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    @staticmethod
    def alloc_gpu(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
        return ctx.enqueue_create_buffer[dtype](Self.TOTAL_SIZE)

    # --- Rollout views (full ROLLOUT_TOTAL) ---

    def obs(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.RT, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.RT, Self.OBS), MutAnyOrigin
        ](self.ptr + Self._O_OBS)

    def obs_at[N_ENVS: Int](self, t_offset: Int) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
    ]:
        """Obs view for a single timestep slice at t_offset * OBS."""
        return LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](self.ptr + Self._O_OBS + t_offset * Self.OBS)

    def actions(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.RT), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.RT), MutAnyOrigin](
            self.ptr + Self._O_ACTIONS
        )

    def actions_at[N_ENVS: Int](self, t_offset: Int) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
            self.ptr + Self._O_ACTIONS + t_offset
        )

    def log_probs(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.RT), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.RT), MutAnyOrigin](
            self.ptr + Self._O_LOG_PROBS
        )

    def log_probs_at[N_ENVS: Int](self, t_offset: Int) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
            self.ptr + Self._O_LOG_PROBS + t_offset
        )

    def values(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.RT), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.RT), MutAnyOrigin](
            self.ptr + Self._O_VALUES
        )

    def values_at[N_ENVS: Int](self, t_offset: Int) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
            self.ptr + Self._O_VALUES + t_offset
        )

    def rewards(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.RT), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.RT), MutAnyOrigin](
            self.ptr + Self._O_REWARDS
        )

    def rewards_at[N_ENVS: Int](self, t_offset: Int) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
            self.ptr + Self._O_REWARDS + t_offset
        )

    def dones(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.RT), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.RT), MutAnyOrigin](
            self.ptr + Self._O_DONES
        )

    def dones_at[N_ENVS: Int](self, t_offset: Int) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
            self.ptr + Self._O_DONES + t_offset
        )

    def advantages(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.RT), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.RT), MutAnyOrigin](
            self.ptr + Self._O_ADVANTAGES
        )

    def returns(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.RT), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.RT), MutAnyOrigin](
            self.ptr + Self._O_RETURNS
        )

    # --- Raw pointer access (for enqueue_copy to/from HostBuffers) ---

    def obs_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_OBS

    def actions_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_ACTIONS

    def log_probs_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_LOG_PROBS

    def values_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_VALUES

    def rewards_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_REWARDS

    def dones_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_DONES

    def advantages_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_ADVANTAGES

    def returns_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ptr + Self._O_RETURNS

    # --- DeviceBuffer sub-views (non-owning, for enqueue_copy) ---

    def rewards_subbuf(self, ctx: DeviceContext) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](
            ctx, self.ptr + Self._O_REWARDS, Self.RT, owning=False
        )

    def values_subbuf(self, ctx: DeviceContext) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](
            ctx, self.ptr + Self._O_VALUES, Self.RT, owning=False
        )

    def dones_subbuf(self, ctx: DeviceContext) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](
            ctx, self.ptr + Self._O_DONES, Self.RT, owning=False
        )

    def advantages_subbuf(self, ctx: DeviceContext) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](
            ctx, self.ptr + Self._O_ADVANTAGES, Self.RT, owning=False
        )

    def returns_subbuf(self, ctx: DeviceContext) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](
            ctx, self.ptr + Self._O_RETURNS, Self.RT, owning=False
        )


# =============================================================================
# MinibatchWS — Minibatch buffers for on-policy agents (PPO / A2C)
# =============================================================================


struct MinibatchWS[
    MB: Int,       # Minibatch size
    OBS: Int,      # Observation dimension
](ImplicitlyCopyable, Movable):
    """Typed workspace for on-policy minibatch buffers.

    Consolidates obs, actions, advantages, returns, old_log_probs, old_values
    into a single flat GPU allocation. Indices buffer is separate (int32).

    Layout (all contiguous, dtype):
        obs:          [MB * OBS]
        actions:      [MB]
        advantages:   [MB]
        returns:      [MB]
        old_log_probs:[MB]
        old_values:   [MB]
    """

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    comptime _O_OBS: Int = 0
    comptime _O_ACTIONS: Int = Self._O_OBS + Self.MB * Self.OBS
    comptime _O_ADVANTAGES: Int = Self._O_ACTIONS + Self.MB
    comptime _O_RETURNS: Int = Self._O_ADVANTAGES + Self.MB
    comptime _O_OLD_LOG_PROBS: Int = Self._O_RETURNS + Self.MB
    comptime _O_OLD_VALUES: Int = Self._O_OLD_LOG_PROBS + Self.MB
    comptime TOTAL_SIZE: Int = Self._O_OLD_VALUES + Self.MB

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    @staticmethod
    def alloc_gpu(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
        return ctx.enqueue_create_buffer[dtype](Self.TOTAL_SIZE)

    def obs(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.OBS), MutAnyOrigin
        ](self.ptr + Self._O_OBS)

    def actions(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.MB), MutAnyOrigin](
            self.ptr + Self._O_ACTIONS
        )

    def advantages(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.MB), MutAnyOrigin](
            self.ptr + Self._O_ADVANTAGES
        )

    def advantages_subbuf(self, ctx: DeviceContext) -> DeviceBuffer[dtype]:
        return DeviceBuffer[dtype](
            ctx, self.ptr + Self._O_ADVANTAGES, Self.MB, owning=False
        )

    def returns(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.MB), MutAnyOrigin](
            self.ptr + Self._O_RETURNS
        )

    def old_log_probs(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.MB), MutAnyOrigin](
            self.ptr + Self._O_OLD_LOG_PROBS
        )

    def old_values(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB), MutAnyOrigin
    ]:
        return LayoutTensor[dtype, Layout.row_major(Self.MB), MutAnyOrigin](
            self.ptr + Self._O_OLD_VALUES
        )


# =============================================================================
# ActorTrainWS — Actor forward/backward workspace for on-policy agents
# =============================================================================


struct ActorTrainWS[
    MB: Int,       # Minibatch size
    ACTIONS: Int,  # Number of actions / actor output dim
    OBS: Int,      # Actor input dim (observation)
    CACHE: Int,    # Actor cache size per sample
](ImplicitlyCopyable, Movable):
    """Typed workspace for actor forward/backward buffers.

    Layout (all contiguous, dtype):
        logits:      [MB * ACTIONS]
        cache:       [MB * CACHE]
        grad_output: [MB * ACTIONS]
        grad_input:  [MB * OBS]
    """

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    comptime _O_LOGITS: Int = 0
    comptime _O_CACHE: Int = Self._O_LOGITS + Self.MB * Self.ACTIONS
    comptime _O_GRAD_OUTPUT: Int = Self._O_CACHE + Self.MB * Self.CACHE
    comptime _O_GRAD_INPUT: Int = Self._O_GRAD_OUTPUT + Self.MB * Self.ACTIONS
    comptime TOTAL_SIZE: Int = Self._O_GRAD_INPUT + Self.MB * Self.OBS

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    @staticmethod
    def alloc_gpu(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
        return ctx.enqueue_create_buffer[dtype](Self.TOTAL_SIZE)

    def logits(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.ACTIONS), MutAnyOrigin
        ](self.ptr + Self._O_LOGITS)

    def cache(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.CACHE), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.CACHE), MutAnyOrigin
        ](self.ptr + Self._O_CACHE)

    def grad_output(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.ACTIONS), MutAnyOrigin
        ](self.ptr + Self._O_GRAD_OUTPUT)

    def grad_input(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.OBS), MutAnyOrigin
        ](self.ptr + Self._O_GRAD_INPUT)


# =============================================================================
# CriticTrainWS — Critic forward/backward workspace for on-policy agents
# =============================================================================


struct CriticTrainWS[
    MB: Int,        # Minibatch size
    CRITIC_OUT: Int, # Critic output dim (usually 1)
    OBS: Int,       # Critic input dim (observation)
    CACHE: Int,     # Critic cache size per sample
](ImplicitlyCopyable, Movable):
    """Typed workspace for critic forward/backward buffers.

    Layout (all contiguous, dtype):
        values:      [MB * CRITIC_OUT]
        cache:       [MB * CACHE]
        grad_output: [MB * CRITIC_OUT]
        grad_input:  [MB * OBS]
    """

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    comptime _O_VALUES: Int = 0
    comptime _O_CACHE: Int = Self._O_VALUES + Self.MB * Self.CRITIC_OUT
    comptime _O_GRAD_OUTPUT: Int = Self._O_CACHE + Self.MB * Self.CACHE
    comptime _O_GRAD_INPUT: Int = Self._O_GRAD_OUTPUT + Self.MB * Self.CRITIC_OUT
    comptime TOTAL_SIZE: Int = Self._O_GRAD_INPUT + Self.MB * Self.OBS

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    @staticmethod
    def alloc_gpu(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
        return ctx.enqueue_create_buffer[dtype](Self.TOTAL_SIZE)

    def values(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.CRITIC_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.CRITIC_OUT), MutAnyOrigin
        ](self.ptr + Self._O_VALUES)

    def cache(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.CACHE), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.CACHE), MutAnyOrigin
        ](self.ptr + Self._O_CACHE)

    def grad_output(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.CRITIC_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.CRITIC_OUT), MutAnyOrigin
        ](self.ptr + Self._O_GRAD_OUTPUT)

    def grad_input(self) -> LayoutTensor[
        dtype, Layout.row_major(Self.MB, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.MB, Self.OBS), MutAnyOrigin
        ](self.ptr + Self._O_GRAD_INPUT)
