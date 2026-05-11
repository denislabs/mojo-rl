"""N-Step Transition Buffer for off-policy n-step returns.

Sits between the environment and the replay buffer, accumulating rewards
over N steps and emitting compressed transitions:
  (s_t, a_t, R_n, s_{t+n}, done_any)
  where R_n = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1}

The replay buffer stores these compressed transitions. The agent's TD target
uses γ^n (not γ) for bootstrapping: target = R_n + γ^n * Q(s_{t+n}).

CPU: NStepBuffer[N, OBS_DIM] — single-environment streaming buffer.
GPU: GPUNStepBuffer[N, OBS_DIM, N_ENVS] — per-environment parallel accumulation.

Usage (CPU):
    var nstep = NStepBuffer[3, 4](gamma=0.99)
    # In the training loop:
    var result = nstep.add(obs, action, reward, next_obs, done)
    if result.valid:
        replay_buffer.add(result.obs, result.action, result.reward,
                          result.next_obs, result.done)

Usage (GPU):
    var nstep = GPUNStepBuffer[3, 4, 256](ctx, gamma=0.99)
    # Each collection step:
    nstep.process(ctx, obs_buf, act_buf, rew_buf, nobs_buf, done_buf)
    # Then store valid transitions:
    nstep.store_valid(ctx, replay_buffer)
"""

from mojo_rl.nn.constants import dtype, TPB
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor


# =============================================================================
# GPU Replay Buffer Storable Trait
# =============================================================================


trait GPUReplayBufferStorable:
    """Trait for GPU replay buffers that can store batched transitions."""

    def store[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        rewards: DeviceBuffer[dtype],
        next_states: DeviceBuffer[dtype],
        dones: DeviceBuffer[dtype],
    ) raises:
        ...


# =============================================================================
# CPU NStepBuffer
# =============================================================================


struct NStepTransition[OBS_DIM: Int]:
    """Result of NStepBuffer.add() — a compressed n-step transition."""

    var valid: Bool
    var obs: InlineArray[Scalar[dtype], Self.OBS_DIM]
    var action: Scalar[dtype]
    var reward: Scalar[dtype]  # R_n = discounted sum
    var next_obs: InlineArray[Scalar[dtype], Self.OBS_DIM]
    var done: Bool

    def __init__(out self):
        """Empty (invalid) result."""
        self.valid = False
        self.obs = InlineArray[Scalar[dtype], Self.OBS_DIM](
            fill=Scalar[dtype](0)
        )
        self.action = Scalar[dtype](0)
        self.reward = Scalar[dtype](0)
        self.next_obs = InlineArray[Scalar[dtype], Self.OBS_DIM](
            fill=Scalar[dtype](0)
        )
        self.done = False

    def __init__(
        out self,
        obs: InlineArray[Scalar[dtype], Self.OBS_DIM],
        action: Scalar[dtype],
        reward: Scalar[dtype],
        next_obs: InlineArray[Scalar[dtype], Self.OBS_DIM],
        done: Bool,
    ):
        self.valid = True
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done


struct NStepBuffer[N: Int, OBS_DIM: Int](Movable):
    """CPU streaming n-step transition buffer for a single environment.

    Accumulates up to N transitions, then emits a compressed transition
    with the discounted n-step return. Handles episode boundaries by
    flushing partial transitions.

    Parameters:
        N: Number of steps to accumulate (e.g., 3 for 3-step returns).
        OBS_DIM: Observation dimension.
    """

    # Ring storage (flat arrays, indexed by step position)
    var obs: List[Scalar[dtype]]  # [N * OBS_DIM]
    var actions: List[Scalar[dtype]]  # [N]
    var rewards: List[Scalar[dtype]]  # [N]
    var gamma: Scalar[dtype]
    var count: Int  # Steps currently buffered (0 to N)

    def __init__(out self, gamma: Float64 = 0.99):
        self.obs = List[Scalar[dtype]](capacity=Self.N * Self.OBS_DIM)
        self.actions = List[Scalar[dtype]](capacity=Self.N)
        self.rewards = List[Scalar[dtype]](capacity=Self.N)
        for _ in range(Self.N * Self.OBS_DIM):
            self.obs.append(Scalar[dtype](0))
        for _ in range(Self.N):
            self.actions.append(Scalar[dtype](0))
            self.rewards.append(Scalar[dtype](0))
        self.gamma = Scalar[dtype](gamma)
        self.count = 0

    def __init__(out self, *, deinit take: Self):
        self.obs = take.obs^
        self.actions = take.actions^
        self.rewards = take.rewards^
        self.gamma = take.gamma
        self.count = take.count

    def _compute_return(self, n: Int) -> Scalar[dtype]:
        """Compute discounted return R = r_0 + γr_1 + ... + γ^{n-1}r_{n-1}."""
        var r = Scalar[dtype](0)
        for i in range(n - 1, -1, -1):
            r = r * self.gamma + self.rewards[i]
        return r

    def _shift_left(mut self):
        """Remove oldest entry, shift everything left by 1."""
        for i in range(Self.N - 1):
            for d in range(Self.OBS_DIM):
                self.obs[i * Self.OBS_DIM + d] = self.obs[
                    (i + 1) * Self.OBS_DIM + d
                ]
            self.actions[i] = self.actions[i + 1]
            self.rewards[i] = self.rewards[i + 1]

    def add(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS_DIM],
        action: Scalar[dtype],
        reward: Scalar[dtype],
        next_obs: InlineArray[Scalar[dtype], Self.OBS_DIM],
        done: Bool,
    ) -> NStepTransition[Self.OBS_DIM]:
        """Add a transition. Returns compressed n-step transition if ready.

        Three cases:
        1. count < N and not done: buffer, return invalid.
        2. count == N (or becomes N): emit n-step transition, shift buffer.
        3. done: flush partial n-step transition, reset buffer.

        Args:
            obs: Current observation.
            action: Action taken (as scalar).
            reward: Reward received.
            next_obs: Next observation (s_{t+1}).
            done: Whether the episode ended.

        Returns:
            NStepTransition with valid=True if a transition was emitted.
        """
        # Store at current position
        for d in range(Self.OBS_DIM):
            self.obs[self.count * Self.OBS_DIM + d] = obs[d]
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.count += 1

        if done:
            # Flush: emit whatever steps are accumulated
            var r_n = self._compute_return(self.count)
            var s0 = InlineArray[Scalar[dtype], Self.OBS_DIM](
                uninitialized=True
            )
            for d in range(Self.OBS_DIM):
                s0[d] = self.obs[d]
            var a0 = self.actions[0]
            self.count = 0
            return NStepTransition[Self.OBS_DIM](s0, a0, r_n, next_obs, True)

        if self.count == Self.N:
            # N steps accumulated: emit and shift
            var r_n = self._compute_return(Self.N)
            var s0 = InlineArray[Scalar[dtype], Self.OBS_DIM](
                uninitialized=True
            )
            for d in range(Self.OBS_DIM):
                s0[d] = self.obs[d]
            var a0 = self.actions[0]
            self._shift_left()
            self.count = Self.N - 1
            return NStepTransition[Self.OBS_DIM](s0, a0, r_n, next_obs, False)

        # Not enough steps yet
        return NStepTransition[Self.OBS_DIM]()

    def reset(mut self):
        """Reset the buffer (e.g., at start of new episode)."""
        self.count = 0


# =============================================================================
# GPU NStepBuffer
# =============================================================================


struct GPUNStepBuffer[N: Int, OBS_DIM: Int, N_ENVS: Int](Movable):
    """GPU per-environment n-step accumulation buffer.

    Each of the N_ENVS parallel environments has its own ring buffer of
    size N. A single GPU kernel processes all environments in parallel,
    computing n-step returns and emitting compressed transitions.

    Parameters:
        N: Number of steps to accumulate.
        OBS_DIM: Observation dimension.
        N_ENVS: Number of parallel environments.
    """

    # Per-env ring buffers on GPU
    var obs_ring: DeviceBuffer[dtype]  # [N_ENVS * N * OBS_DIM]
    var act_ring: DeviceBuffer[dtype]  # [N_ENVS * N]
    var rew_ring: DeviceBuffer[dtype]  # [N_ENVS * N]
    var counts: DeviceBuffer[DType.int32]  # [N_ENVS] step count per env

    # Compressed output buffers (overwritten each process() call)
    var out_obs: DeviceBuffer[dtype]  # [N_ENVS * OBS_DIM]
    var out_act: DeviceBuffer[dtype]  # [N_ENVS]
    var out_rew: DeviceBuffer[dtype]  # [N_ENVS]
    var out_nobs: DeviceBuffer[dtype]  # [N_ENVS * OBS_DIM]
    var out_done: DeviceBuffer[dtype]  # [N_ENVS]
    var out_valid: DeviceBuffer[DType.int32]  # [N_ENVS] (1=valid, 0=not ready)

    # CPU-side count of valid transitions (updated after process)
    var num_valid: Int
    var host_valid: HostBuffer[DType.int32]  # [N_ENVS] for GPU→CPU readback

    var gamma: Scalar[dtype]

    def __init__(out self, ctx: DeviceContext, gamma: Float64 = 0.99) raises:
        self.obs_ring = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.N * Self.OBS_DIM
        )
        self.act_ring = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * Self.N)
        self.rew_ring = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * Self.N)
        self.counts = ctx.enqueue_create_buffer[DType.int32](Self.N_ENVS)
        ctx.enqueue_memset(self.obs_ring, 0)
        ctx.enqueue_memset(self.act_ring, 0)
        ctx.enqueue_memset(self.rew_ring, 0)
        ctx.enqueue_memset(self.counts, 0)

        self.out_obs = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.OBS_DIM
        )
        self.out_act = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.out_rew = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.out_nobs = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.OBS_DIM
        )
        self.out_done = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.out_valid = ctx.enqueue_create_buffer[DType.int32](Self.N_ENVS)
        ctx.enqueue_memset(self.out_valid, 0)

        self.host_valid = ctx.enqueue_create_host_buffer[DType.int32](
            Self.N_ENVS
        )
        self.num_valid = 0
        self.gamma = Scalar[dtype](gamma)

    def __init__(out self, *, deinit take: Self):
        self.obs_ring = take.obs_ring^
        self.act_ring = take.act_ring^
        self.rew_ring = take.rew_ring^
        self.counts = take.counts^
        self.out_obs = take.out_obs^
        self.out_act = take.out_act^
        self.out_rew = take.out_rew^
        self.out_nobs = take.out_nobs^
        self.out_done = take.out_done^
        self.out_valid = take.out_valid^
        self.host_valid = take.host_valid^
        self.num_valid = take.num_valid
        self.gamma = take.gamma

    def process(
        mut self,
        ctx: DeviceContext,
        obs: DeviceBuffer[dtype],  # [N_ENVS * OBS_DIM]
        actions: DeviceBuffer[dtype],  # [N_ENVS]
        rewards: DeviceBuffer[dtype],  # [N_ENVS]
        next_obs: DeviceBuffer[dtype],  # [N_ENVS * OBS_DIM]
        dones: DeviceBuffer[dtype],  # [N_ENVS]
    ) raises:
        """Process one step for all environments in parallel.

        For each environment:
        - Store (obs, action, reward) into ring buffer
        - If done or N steps accumulated: compute R_n, emit compressed transition
        - Write results to out_* buffers, set out_valid

        Args:
            ctx: GPU device context.
            obs: Current observations [N_ENVS * OBS_DIM].
            actions: Actions taken [N_ENVS].
            rewards: Rewards received [N_ENVS].
            next_obs: Next observations [N_ENVS * OBS_DIM].
            dones: Done flags [N_ENVS].
        """
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, Self.OBS_DIM), MutAnyOrigin
        ](obs.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](actions.unsafe_ptr())
        var rew_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](rewards.unsafe_ptr())
        var nobs_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, Self.OBS_DIM), MutAnyOrigin
        ](next_obs.unsafe_ptr())
        var done_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](dones.unsafe_ptr())

        # Ring buffers as flat tensors
        var obs_r = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, Self.N * Self.OBS_DIM),
            MutAnyOrigin,
        ](self.obs_ring.unsafe_ptr())
        var act_r = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, Self.N), MutAnyOrigin
        ](self.act_ring.unsafe_ptr())
        var rew_r = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, Self.N), MutAnyOrigin
        ](self.rew_ring.unsafe_ptr())
        var cnt_t = LayoutTensor[
            DType.int32, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.counts.unsafe_ptr())

        # Output buffers
        var o_obs = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, Self.OBS_DIM), MutAnyOrigin
        ](self.out_obs.unsafe_ptr())
        var o_act = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.out_act.unsafe_ptr())
        var o_rew = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.out_rew.unsafe_ptr())
        var o_nobs = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, Self.OBS_DIM), MutAnyOrigin
        ](self.out_nobs.unsafe_ptr())
        var o_done = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.out_done.unsafe_ptr())
        var o_valid = LayoutTensor[
            DType.int32, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.out_valid.unsafe_ptr())

        var gamma_s = self.gamma

        @parameter
        @always_inline
        def nstep_kernel(
            obs_in: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            act_in: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            rew_in: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            nobs_in: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            done_in: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            o_ring: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS, Self.N * Self.OBS_DIM),
                MutAnyOrigin,
            ],
            a_ring: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS, Self.N), MutAnyOrigin
            ],
            r_ring: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS, Self.N), MutAnyOrigin
            ],
            counts: LayoutTensor[
                DType.int32, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            oo: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            oa: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            orw: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            on: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            od: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            ov: LayoutTensor[
                DType.int32, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            gamma: Scalar[dtype],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= Self.N_ENVS:
                return

            var c = Int(rebind[Scalar[DType.int32]](counts[e]))

            # Store incoming transition into ring at position c
            for d in range(Self.OBS_DIM):
                o_ring[e, c * Self.OBS_DIM + d] = obs_in[e, d]
            a_ring[e, c] = act_in[e]
            r_ring[e, c] = rew_in[e]
            c += 1

            var is_done = rebind[Scalar[dtype]](done_in[e]) > Scalar[dtype](0.5)

            if is_done or c == Self.N:
                # Compute R_n = r_0 + γr_1 + ... + γ^{c-1}r_{c-1}
                var r_n = Scalar[dtype](0)
                for i in range(c - 1, -1, -1):
                    r_n = r_n * gamma + rebind[Scalar[dtype]](r_ring[e, i])

                # Emit: (obs_ring[0], act_ring[0], R_n, next_obs, done)
                for d in range(Self.OBS_DIM):
                    oo[e, d] = o_ring[e, d]
                    on[e, d] = nobs_in[e, d]
                oa[e] = a_ring[e, 0]
                orw[e] = r_n
                od[e] = Scalar[dtype](1.0) if is_done else Scalar[dtype](0.0)
                ov[e] = Int32(1)

                if is_done:
                    # Reset ring
                    counts[e] = Int32(0)
                else:
                    # Shift ring left by 1
                    for i in range(Self.N - 1):
                        for d in range(Self.OBS_DIM):
                            o_ring[e, i * Self.OBS_DIM + d] = o_ring[
                                e, (i + 1) * Self.OBS_DIM + d
                            ]
                        a_ring[e, i] = a_ring[e, i + 1]
                        r_ring[e, i] = r_ring[e, i + 1]
                    counts[e] = Int32(Self.N - 1)
            else:
                ov[e] = Int32(0)
                counts[e] = Int32(c)

        comptime ENV_BLOCKS = (Self.N_ENVS + TPB - 1) // TPB
        ctx.enqueue_function[nstep_kernel](
            obs_t,
            act_t,
            rew_t,
            nobs_t,
            done_t,
            obs_r,
            act_r,
            rew_r,
            cnt_t,
            o_obs,
            o_act,
            o_rew,
            o_nobs,
            o_done,
            o_valid,
            gamma_s,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    def store_into[
        B: GPUReplayBufferStorable
    ](self, ctx: DeviceContext, mut buffer: B,) raises:
        """Store all N_ENVS compressed transitions into a GPU replay buffer.

        Stores all transitions (valid and invalid). Invalid ones have zero
        rewards and will be overwritten in the circular buffer.

        Args:
            ctx: GPU device context.
            buffer: GPU replay buffer implementing GPUReplayBufferStorable.
        """
        buffer.store[Self.N_ENVS](
            ctx,
            self.out_obs,
            self.out_act,
            self.out_rew,
            self.out_nobs,
            self.out_done,
        )
