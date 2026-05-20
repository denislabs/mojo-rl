"""GPU-resident replay buffer for EfficientZero V2.

Mirrors the layout of `EZV2DiscreteCPUState`'s buffer + parallel arrays
(SequenceReplayBuffer + mcts_policies + mcts_values + priorities +
step_at_write) on device. Owned by the training loop; upload from CPU
state at sync points (or every step in the future), download for
end-of-run verification.

Step 4 in `docs/EZV2_FULL_GPU_PLAN.md` introduces the struct + bulk
mirror path. Step 5 wires GPU-side priority sampling and gather kernels
that read directly from these buffers, eliminating the host-side
sample/upload path in `train_step_gpu`.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.efficient_zero_v2.configs import EZV2DiscreteConfig
from mojo_rl.deep_agents.efficient_zero_v2.state import EZV2DiscreteCPUState


struct EZV2GPUReplayBuffer[
    CAP: Int,
    OBS: Int,
    ACT: Int,
    K_ROOT: Int,
](Movable):
    """GPU-resident sequence replay + MCTS-target arrays.

    Field layout matches `EZV2DiscreteCPUState`'s `buffer.*` /
    `mcts_*` / `priorities` / `step_at_write` 1:1. All buffers are flat
    `DeviceBuffer`s so a future GPU sample-gather kernel can read them
    via `LayoutTensor` views.

    Parameters:
        CAP: Buffer capacity (must match the CPU state's `_CAP`).
        OBS: Observation dimension.
        ACT: Action dimension (also the width of the one-hot action
            encoding stored in `actions`).
        K_ROOT: Number of MCTS root candidate slots per replay entry.
            Sizes the two fullpi-target buffers (`mcts_sampled_actions`,
            `mcts_improved_policy`). Must match
            `Config.num_root_candidates`.
    """

    # ── Sequence buffer fields (mirrors `SequenceReplayBuffer`) ──────────
    var obs: DeviceBuffer[dtype]              # [CAP * OBS]
    var actions: DeviceBuffer[dtype]          # [CAP * ACT] one-hot
    var rewards: DeviceBuffer[dtype]          # [CAP]
    var dones: DeviceBuffer[dtype]            # [CAP] (term|trunc)
    var terminations: DeviceBuffer[dtype]     # [CAP] (term-only)

    # ── EZ-V2 parallel arrays ────────────────────────────────────────────
    var mcts_policies: DeviceBuffer[dtype]    # [CAP * ACT]
    var mcts_values: DeviceBuffer[dtype]      # [CAP]
    var priorities: DeviceBuffer[dtype]       # [CAP]
    var step_at_write: DeviceBuffer[DType.uint32]  # [CAP]

    # ── Full-π (paper Eq. 6) targets for continuous ACT_DIM==1 ──────────
    # Mirrors `EZV2DiscreteCPUState.mcts_sampled_actions` /
    # `mcts_improved_policy`. Required by
    # `ezv2_policy_loss_grad_continuous_fullpi_kernel` — without these
    # the GPU-sampling path leaves the per-batch fullpi buffers at zero
    # and the fullpi loss collapses to `−ent_scale · H_d ≈ −4.5e-5`,
    # producing no policy gradient. Discrete + ACT_DIM>1 continuous
    # configs allocate these too (harmless overhead — uniform K_ROOT
    # buffer is cheap) and the simple-best loss path simply doesn't
    # read them. Found 2026-05-14.
    var mcts_sampled_actions: DeviceBuffer[dtype]  # [CAP * K_ROOT * ACT]
    var mcts_improved_policy: DeviceBuffer[dtype]  # [CAP * K_ROOT]

    # ── Per-buffer metadata (host-mirrored each upload/download) ─────────
    # Stored as host scalars rather than a single device meta buffer
    # because ptr/size/episode change every flush and the GPU sampler
    # (Step 5) reads them as kernel scalar args.
    var ptr: Int
    var size: Int
    var current_episode: Int
    var max_priority: Float64

    def __init__(out self, ctx: DeviceContext) raises:
        comptime CAP_OBS = Self.CAP * Self.OBS
        comptime CAP_ACT = Self.CAP * Self.ACT
        comptime CAP_K_ACT = Self.CAP * Self.K_ROOT * Self.ACT
        comptime CAP_K = Self.CAP * Self.K_ROOT

        self.obs = ctx.enqueue_create_buffer[dtype](CAP_OBS)
        self.actions = ctx.enqueue_create_buffer[dtype](CAP_ACT)
        self.rewards = ctx.enqueue_create_buffer[dtype](Self.CAP)
        self.dones = ctx.enqueue_create_buffer[dtype](Self.CAP)
        self.terminations = ctx.enqueue_create_buffer[dtype](Self.CAP)
        self.mcts_policies = ctx.enqueue_create_buffer[dtype](CAP_ACT)
        self.mcts_values = ctx.enqueue_create_buffer[dtype](Self.CAP)
        self.priorities = ctx.enqueue_create_buffer[dtype](Self.CAP)
        self.step_at_write = ctx.enqueue_create_buffer[DType.uint32](
            Self.CAP
        )
        self.mcts_sampled_actions = ctx.enqueue_create_buffer[dtype](
            CAP_K_ACT
        )
        self.mcts_improved_policy = ctx.enqueue_create_buffer[dtype](CAP_K)

        ctx.enqueue_memset(self.obs, 0)
        ctx.enqueue_memset(self.actions, 0)
        ctx.enqueue_memset(self.rewards, 0)
        ctx.enqueue_memset(self.dones, 0)
        ctx.enqueue_memset(self.terminations, 0)
        ctx.enqueue_memset(self.mcts_policies, 0)
        ctx.enqueue_memset(self.mcts_values, 0)
        ctx.enqueue_memset(self.priorities, 0)
        ctx.enqueue_memset(self.step_at_write, 0)
        ctx.enqueue_memset(self.mcts_sampled_actions, 0)
        ctx.enqueue_memset(self.mcts_improved_policy, 0)

        self.ptr = 0
        self.size = 0
        self.current_episode = 0
        self.max_priority = 1.0

    def __init__(out self, *, deinit take: Self):
        self.obs = take.obs^
        self.actions = take.actions^
        self.rewards = take.rewards^
        self.dones = take.dones^
        self.terminations = take.terminations^
        self.mcts_policies = take.mcts_policies^
        self.mcts_values = take.mcts_values^
        self.priorities = take.priorities^
        self.step_at_write = take.step_at_write^
        self.mcts_sampled_actions = take.mcts_sampled_actions^
        self.mcts_improved_policy = take.mcts_improved_policy^
        self.ptr = take.ptr
        self.size = take.size
        self.current_episode = take.current_episode
        self.max_priority = take.max_priority

    # ══════════════════════════════════════════════════════════════════════
    # Bulk CPU → GPU mirror
    # ══════════════════════════════════════════════════════════════════════
    #
    # Cheapest correctness-preserving option for Step 4: re-upload the
    # entire CPU buffer every sync. At CAP=50000 and CartPole obs=4,
    # action=2, the total per-call DMA is ~3MB — negligible at the
    # SYNC_INTERVAL=50 cadence (one sync ~every 200 env-steps).
    #
    # Step 5 will replace this with per-flush incremental uploads that
    # only touch the just-written slice, but the bulk path is what Step
    # 4 verifies. As long as the post-upload GPU contents diff to zero
    # against the CPU contents, the storage layout / DMA path are
    # known-good.

    def upload_from_cpu[
        Config: EZV2DiscreteConfig
    ](
        mut self,
        cpu: EZV2DiscreteCPUState[Config, Config.buffer_capacity],
        ctx: DeviceContext,
    ) raises:
        """Bulk-mirror the entire CPU state's buffer + parallel arrays
        to device. Called at the same cadence as the network weight
        sync (`SYNC_INTERVAL` train_steps) in the training loop."""
        ctx.enqueue_copy(self.obs, cpu.buffer.obs.unsafe_ptr())
        ctx.enqueue_copy(self.actions, cpu.buffer.actions.unsafe_ptr())
        ctx.enqueue_copy(self.rewards, cpu.buffer.rewards.unsafe_ptr())
        ctx.enqueue_copy(self.dones, cpu.buffer.dones.unsafe_ptr())
        ctx.enqueue_copy(
            self.terminations, cpu.buffer.terminations.unsafe_ptr()
        )
        ctx.enqueue_copy(self.mcts_policies, cpu.mcts_policies)
        ctx.enqueue_copy(self.mcts_values, cpu.mcts_values)
        ctx.enqueue_copy(self.priorities, cpu.priorities)
        ctx.enqueue_copy(self.step_at_write, cpu.step_at_write)
        ctx.enqueue_copy(
            self.mcts_sampled_actions, cpu.mcts_sampled_actions
        )
        ctx.enqueue_copy(
            self.mcts_improved_policy, cpu.mcts_improved_policy
        )

        self.ptr = cpu.buffer.ptr
        self.size = cpu.buffer.size
        self.current_episode = cpu.buffer.current_episode
        # Note: `cpu.max_priority` is on the agent, not on `cpu` state.
        # Caller refreshes `gpu_replay.max_priority` from agent if needed.

    # ══════════════════════════════════════════════════════════════════════
    # Bulk GPU → CPU download (for verification)
    # ══════════════════════════════════════════════════════════════════════

    def download_to_cpu[
        Config: EZV2DiscreteConfig
    ](
        mut self,
        mut cpu: EZV2DiscreteCPUState[Config, Config.buffer_capacity],
        ctx: DeviceContext,
    ) raises:
        """Mirror the GPU buffer back to a CPU state. Use for end-of-run
        verification; this overwrites the CPU state's buffer fields, so
        do NOT call mid-training unless you want to stomp the source
        of truth."""
        ctx.enqueue_copy(cpu.buffer.obs.unsafe_ptr(), self.obs)
        ctx.enqueue_copy(cpu.buffer.actions.unsafe_ptr(), self.actions)
        ctx.enqueue_copy(cpu.buffer.rewards.unsafe_ptr(), self.rewards)
        ctx.enqueue_copy(cpu.buffer.dones.unsafe_ptr(), self.dones)
        ctx.enqueue_copy(
            cpu.buffer.terminations.unsafe_ptr(), self.terminations
        )
        ctx.enqueue_copy(cpu.mcts_policies, self.mcts_policies)
        ctx.enqueue_copy(cpu.mcts_values, self.mcts_values)
        ctx.enqueue_copy(cpu.priorities, self.priorities)
        ctx.enqueue_copy(cpu.step_at_write, self.step_at_write)
        ctx.enqueue_copy(
            cpu.mcts_sampled_actions, self.mcts_sampled_actions
        )
        ctx.enqueue_copy(
            cpu.mcts_improved_policy, self.mcts_improved_policy
        )

        cpu.buffer.ptr = self.ptr
        cpu.buffer.size = self.size
        cpu.buffer.current_episode = self.current_episode
