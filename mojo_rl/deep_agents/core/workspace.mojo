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
