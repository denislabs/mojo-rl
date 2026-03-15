"""Generic off-policy CPU state container.

Replaces per-agent state structs (DDPGCPUState, TD3CPUState, SACCPUState)
with a single generic struct parameterized by Config. The workspace uses
a single List allocation with compile-time offset views.
"""

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network, NetworkState, NetworkPair
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.deep_agents.core import OffPolicyState
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer


# =============================================================================
# Workspace offset computation
# =============================================================================

# Workspace regions for DDPG-style single critic:
#   next_act:    BATCH * ACTIONS
#   next_ci:     BATCH * CRITIC_IN
#   next_q:      BATCH * 1
#   targets:     BATCH * 1
#   ci:          BATCH * CRITIC_IN
#   q_out:       BATCH * 1
#   q_cache:     BATCH * CRITIC_CACHE_SIZE
#   q_grad:      BATCH * 1
#   d_ci:        BATCH * CRITIC_IN
#   actor_act:   BATCH * ACTIONS
#   actor_cache: BATCH * ACTOR_CACHE_SIZE
#   new_ci:      BATCH * CRITIC_IN
#   new_q:       BATCH * 1
#   dq:          BATCH * 1
#   d_new_ci:    BATCH * CRITIC_IN
#   d_act:       BATCH * ACTIONS
#   d_obs:       BATCH * OBS


# =============================================================================
# GenericOffPolicyCPUState
# =============================================================================


struct GenericOffPolicyCPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
](Movable, OffPolicyState):
    """Generic CPU state for single-critic off-policy agents (DDPG).

    Contains: actor NetworkPair + critic NetworkPair + replay buffer +
    a single contiguous workspace for all scratch buffers.
    """

    # Derive dimensions from Model types to match LayoutTensor expectations
    comptime OBS = Self.ActorModel.IN_DIM
    comptime ACTIONS = Self.ActorModel.OUT_DIM
    comptime BATCH = Self.batch_size
    comptime CRITIC_IN = Self.CriticModel.IN_DIM
    comptime ACTOR_CS = Self.ActorModel.CACHE_SIZE
    comptime CRITIC_CS = Self.CriticModel.CACHE_SIZE
    comptime BUFFER_DTYPE = dtype

    # Workspace region sizes
    comptime R_NEXT_ACT = Self.BATCH * Self.ACTIONS
    comptime R_NEXT_CI = Self.BATCH * Self.CRITIC_IN
    comptime R_NEXT_Q = Self.BATCH
    comptime R_TARGETS = Self.BATCH
    comptime R_CI = Self.BATCH * Self.CRITIC_IN
    comptime R_Q_OUT = Self.BATCH
    comptime R_Q_CACHE = Self.BATCH * Self.CRITIC_CS
    comptime R_Q_GRAD = Self.BATCH
    comptime R_D_CI = Self.BATCH * Self.CRITIC_IN
    comptime R_ACTOR_ACT = Self.BATCH * Self.ACTIONS
    comptime R_ACTOR_CACHE = Self.BATCH * Self.ACTOR_CS
    comptime R_NEW_CI = Self.BATCH * Self.CRITIC_IN
    comptime R_NEW_Q = Self.BATCH
    comptime R_DQ = Self.BATCH
    comptime R_D_NEW_CI = Self.BATCH * Self.CRITIC_IN
    comptime R_D_ACT = Self.BATCH * Self.ACTIONS
    comptime R_D_OBS = Self.BATCH * Self.OBS

    # Cumulative offsets
    comptime O_NEXT_ACT = 0
    comptime O_NEXT_CI = Self.O_NEXT_ACT + Self.R_NEXT_ACT
    comptime O_NEXT_Q = Self.O_NEXT_CI + Self.R_NEXT_CI
    comptime O_TARGETS = Self.O_NEXT_Q + Self.R_NEXT_Q
    comptime O_CI = Self.O_TARGETS + Self.R_TARGETS
    comptime O_Q_OUT = Self.O_CI + Self.R_CI
    comptime O_Q_CACHE = Self.O_Q_OUT + Self.R_Q_OUT
    comptime O_Q_GRAD = Self.O_Q_CACHE + Self.R_Q_CACHE
    comptime O_D_CI = Self.O_Q_GRAD + Self.R_Q_GRAD
    comptime O_ACTOR_ACT = Self.O_D_CI + Self.R_D_CI
    comptime O_ACTOR_CACHE = Self.O_ACTOR_ACT + Self.R_ACTOR_ACT
    comptime O_NEW_CI = Self.O_ACTOR_CACHE + Self.R_ACTOR_CACHE
    comptime O_NEW_Q = Self.O_NEW_CI + Self.R_NEW_CI
    comptime O_DQ = Self.O_NEW_Q + Self.R_NEW_Q
    comptime O_D_NEW_CI = Self.O_DQ + Self.R_DQ
    comptime O_D_ACT = Self.O_D_NEW_CI + Self.R_D_NEW_CI
    comptime O_D_OBS = Self.O_D_ACT + Self.R_D_ACT
    comptime WORKSPACE_SIZE = Self.O_D_OBS + Self.R_D_OBS

    # Networks
    var actor: NetworkPair[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkPair[Self.CriticModel, Self.CriticOpt]

    # Replay buffer
    var buffer: HeapReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

    # Single workspace allocation
    var ws: List[Scalar[dtype]]

    fn __init__(out self):
        """Allocate networks, buffer, and workspace."""
        self.actor = NetworkPair[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier[]]()
        self.critic = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        self.critic.initialize[Kaiming[]]()

        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        # Single contiguous workspace
        self.ws = List[Scalar[dtype]](capacity=Self.WORKSPACE_SIZE)
        for _ in range(Self.WORKSPACE_SIZE):
            self.ws.append(Scalar[dtype](0))

    # =========================================================================
    # OffPolicyState trait
    # =========================================================================

    fn store[
        dtype: DType
    ](
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.OBS](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.OBS](
            uninitialized=True
        )
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        for i in range(Self.ACTIONS):
            act_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(action[i]))
        self.buffer.add(
            obs_arr, act_arr, Scalar[Self.BUFFER_DTYPE](reward), next_arr, done
        )

    fn is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()

    # =========================================================================
    # Workspace view methods — zero-copy LayoutTensor views into ws
    # =========================================================================

    fn next_act(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_NEXT_ACT)

    fn next_ci(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_NEXT_CI)

    fn next_q(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_NEXT_Q)

    fn targets_ptr(mut self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ws.unsafe_ptr() + Self.O_TARGETS

    fn ci(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_CI)

    fn q_out(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_Q_OUT)

    fn q_out_ptr(mut self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ws.unsafe_ptr() + Self.O_Q_OUT

    fn q_cache(mut self) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.BATCH, Self.CRITIC_CS),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CRITIC_CS),
            MutAnyOrigin,
        ](self.ws.unsafe_ptr() + Self.O_Q_CACHE)

    fn q_grad(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_Q_GRAD)

    fn q_grad_ptr(mut self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ws.unsafe_ptr() + Self.O_Q_GRAD

    fn d_ci(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_D_CI)

    fn actor_act(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_ACTOR_ACT)

    fn actor_cache(mut self) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.BATCH, Self.ACTOR_CS),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.ACTOR_CS),
            MutAnyOrigin,
        ](self.ws.unsafe_ptr() + Self.O_ACTOR_CACHE)

    fn new_ci(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_NEW_CI)

    fn new_q(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_NEW_Q)

    fn dq_ptr(mut self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ws.unsafe_ptr() + Self.O_DQ

    fn d_new_ci(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_D_NEW_CI)

    fn d_new_ci_ptr(mut self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ws.unsafe_ptr() + Self.O_D_NEW_CI

    fn d_act(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_D_ACT)

    fn d_act_ptr(mut self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        return self.ws.unsafe_ptr() + Self.O_D_ACT

    fn d_obs(mut self) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](self.ws.unsafe_ptr() + Self.O_D_OBS)
