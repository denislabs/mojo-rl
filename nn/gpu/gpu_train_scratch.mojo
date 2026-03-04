"""Pre-allocated GPU device buffers for one off-policy training step.

GPUTrainScratch eliminates the ~25 per-agent ctx.enqueue_create_buffer calls
currently inlined in each agent's train_gpu method. One instance is created
at training setup and reused across all subsequent training steps.

For twin-critic agents (TD3, SAC), compose two GPUTrainScratch instances
(one per critic). Each agent defines its own comptime Scratch alias:

    comptime Scratch = GPUTrainScratch[
        Self.OBS, Self.ACTIONS, Self.BATCH,
        1,                             # CRITIC_OUT_DIM (scalar Q)
        Self.CriticModel.CACHE_SIZE,   # CRITIC_CACHE_SIZE
        Self.CriticNet.WS_PER_SAMPLE,  # CRITIC_WS_PER_SAMPLE
    ]

Usage:
    var scratch = GPUTrainScratch[OBS, ACTIONS, BATCH, 1, CACHE, WS](ctx)
    # Inside train step:
    agent._sample_replay_into(ctx, scratch)   # fills obs_buf, actions_buf, etc.
    agent._compute_targets(ctx, scratch)      # writes to scratch.targets_buf
    agent._backward_critic(ctx, scratch)      # uses grad_output_buf etc.
"""

from gpu.host import DeviceContext, DeviceBuffer
from nn.constants import dtype


struct GPUTrainScratch[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    BATCH_SIZE: Int,
    CRITIC_OUT_DIM: Int,
    CRITIC_CACHE_SIZE: Int,
    CRITIC_WS_PER_SAMPLE: Int,
](Movable):
    """Pre-allocated DeviceBuffers for one GPU off-policy training step.

    Constructed once during train_gpu setup. Eliminates ~25 per-agent
    ctx.enqueue_create_buffer calls currently duplicated in each agent.

    Parameters:
        OBS_DIM: Observation dimension.
        ACTION_DIM: Action dimension.
        BATCH_SIZE: Training batch size.
        CRITIC_OUT_DIM: Critic output dimension (1 for scalar Q-value).
        CRITIC_CACHE_SIZE: Critic forward-pass cache size per sample.
        CRITIC_WS_PER_SAMPLE: Critic GPU workspace elements per sample.
    """

    # ------------------------------------------------------------------
    # Sampled batch buffers (filled by replay buffer sampling)
    # ------------------------------------------------------------------
    var obs_buf: DeviceBuffer[dtype]
    """Current observations [BATCH_SIZE * OBS_DIM]."""

    var actions_buf: DeviceBuffer[dtype]
    """Actions taken [BATCH_SIZE * ACTION_DIM]."""

    var rewards_buf: DeviceBuffer[dtype]
    """Rewards received [BATCH_SIZE]."""

    var next_obs_buf: DeviceBuffer[dtype]
    """Next observations [BATCH_SIZE * OBS_DIM]."""

    var dones_buf: DeviceBuffer[dtype]
    """Done flags [BATCH_SIZE] (1.0 = done, 0.0 = not done)."""

    var indices_buf: DeviceBuffer[DType.int32]
    """Sampled indices [BATCH_SIZE] (for PER priority updates)."""

    # ------------------------------------------------------------------
    # Forward pass scratch buffers
    # ------------------------------------------------------------------
    var q_output_buf: DeviceBuffer[dtype]
    """Critic output [BATCH_SIZE * CRITIC_OUT_DIM]."""

    var q_cache_buf: DeviceBuffer[dtype]
    """Critic forward-pass cache [BATCH_SIZE * CRITIC_CACHE_SIZE]."""

    var q_workspace_buf: DeviceBuffer[dtype]
    """Critic GPU matmul workspace [BATCH_SIZE * CRITIC_WS_PER_SAMPLE]."""

    # ------------------------------------------------------------------
    # Target computation
    # ------------------------------------------------------------------
    var targets_buf: DeviceBuffer[dtype]
    """TD targets [BATCH_SIZE * CRITIC_OUT_DIM]."""

    # ------------------------------------------------------------------
    # Gradient pass
    # ------------------------------------------------------------------
    var grad_output_buf: DeviceBuffer[dtype]
    """Loss gradient w.r.t. critic output [BATCH_SIZE * CRITIC_OUT_DIM]."""

    var grad_input_buf: DeviceBuffer[dtype]
    """Loss gradient w.r.t. critic input [BATCH_SIZE * (OBS_DIM + ACTION_DIM)]."""

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all device buffers on the given GPU context.

        Args:
            ctx: GPU device context (e.g., from DeviceContext()).
        """
        self.obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * Self.OBS_DIM
        )
        self.actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * Self.ACTION_DIM
        )
        self.rewards_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH_SIZE)
        self.next_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * Self.OBS_DIM
        )
        self.dones_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH_SIZE)
        self.indices_buf = ctx.enqueue_create_buffer[DType.int32](
            Self.BATCH_SIZE
        )
        self.q_output_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * Self.CRITIC_OUT_DIM
        )
        self.q_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * Self.CRITIC_CACHE_SIZE
        )
        self.q_workspace_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * Self.CRITIC_WS_PER_SAMPLE
        )
        self.targets_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * Self.CRITIC_OUT_DIM
        )
        self.grad_output_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * Self.CRITIC_OUT_DIM
        )
        self.grad_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_SIZE * (Self.OBS_DIM + Self.ACTION_DIM)
        )

    fn __init__(out self, *, deinit take: Self):
        self.obs_buf = take.obs_buf^
        self.actions_buf = take.actions_buf^
        self.rewards_buf = take.rewards_buf^
        self.next_obs_buf = take.next_obs_buf^
        self.dones_buf = take.dones_buf^
        self.indices_buf = take.indices_buf^
        self.q_output_buf = take.q_output_buf^
        self.q_cache_buf = take.q_cache_buf^
        self.q_workspace_buf = take.q_workspace_buf^
        self.targets_buf = take.targets_buf^
        self.grad_output_buf = take.grad_output_buf^
        self.grad_input_buf = take.grad_input_buf^
