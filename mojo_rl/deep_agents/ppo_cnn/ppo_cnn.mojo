"""PPO CNN Agent — Proximal Policy Optimization with CNN for pixel observations.

GPU-only PPO agent using the Nature DQN CNN architecture for processing
pixel observations (4×84×84 stacked grayscale frames).

Actor:  Conv2D[4→32,8,4] → Conv2D[32→64,4,2] → Conv2D[64→64,3,1] →
        Flatten → Dense[3136→512] → Dense[512→num_actions] (logits)
Critic: Same CNN backbone → Dense[512→1] (value)

Reuses PPO state containers (PPODiscreteState, PPODiscreteGPUState) and
PPO kernels — only the model architecture differs from DeepPPOAgent.

Usage:
    from mojo_rl.deep_agents.ppo_cnn import DeepPPOCNNAgent
    from mojo_rl.envs.arcade_games.pong import PongPixelEnv

    var agent = DeepPPOCNNAgent[num_actions=3]()
    with DeviceContext() as ctx:
        var metrics = agent.train_gpu[PongPixelEnv[DType.float32]](
            ctx, num_updates=500, verbose=True, print_every=10,
        )
"""

from std.math import exp, log, sqrt
from std.random import random_float64, seed
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TILE, TPB
from mojo_rl.nn.model import (
    Sequential,
    Model,
    Conv2DReLU,
    FlattenLayer,
    LinearReLU,
    Linear,
)
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.checkpoint import (
    split_lines,
    find_section_start,
    save_checkpoint_file,
    read_checkpoint_file,
)
from mojo_rl.nn.gpu import (
    random_range,
    xorshift32,
    random_uniform,
)
from mojo_rl.deep_agents.core.kernels import (
    zero_buffer_kernel,
    copy_buffer_kernel,
    accumulate_rewards_kernel,
    increment_steps_kernel,
    extract_completed_episodes_kernel,
    selective_reset_tracking_kernel,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from mojo_rl.core.utils.gae import compute_gae_inline
from mojo_rl.core.utils.softmax import (
    softmax_inline,
    sample_from_probs_inline,
    argmax_probs_inline,
)
from mojo_rl.core.utils.normalization import normalize_inline
from mojo_rl.core.utils.shuffle import shuffle_indices_inline
from mojo_rl.deep_agents.ppo.state import PPODiscreteState, PPODiscreteGPUState
from mojo_rl.deep_agents.core.onpolicy_train import (
    OnPolicyAgent,
    OnPolicyDiscreteAgent,
)
from mojo_rl.deep_agents.core.onpolicy_helpers import (
    compute_gae_list,
    normalize_advantages_list,
    fisher_yates_shuffle,
)
from mojo_rl.deep_agents.core.gpu_onpolicy_train import (
    GPUOnPolicyDiscreteAgent,
    run_onpolicy_discrete_train_gpu,
)
from mojo_rl.deep_agents.core.perf_timer import PerfTimer
from mojo_rl.nn.model.model import PerfTimerPtr
from mojo_rl.deep_agents.core.checkpoint_trait import Checkpointable

from mojo_rl.deep_agents.ppo.kernels import (
    ppo_actor_grad_kernel,
    ppo_actor_grad_with_kl_kernel,
    ppo_critic_grad_kernel,
    ppo_critic_grad_clipped_kernel,
    normalize_advantages_kernel,
    _store_post_step_kernel,
    _extract_obs_from_state_kernel,
    _store_pre_step_kernel,
    ppo_gather_minibatch_kernel,
    gradient_norm_kernel,
    gradient_clip_kernel,
    gradient_reduce_and_compute_scale_kernel,
    gradient_apply_scale_kernel,
    gradient_reduce_apply_fused_kernel,
)


# =============================================================================
# Nature DQN CNN Architecture Constants
# =============================================================================

# Input: 4 channels (frame stack) × 84 height × 84 width
# Conv1: 32 filters, 8×8 kernel, stride 4 → 32 × 20 × 20
# Conv2: 64 filters, 4×4 kernel, stride 2 → 64 × 9 × 9
# Conv3: 64 filters, 3×3 kernel, stride 1 → 64 × 7 × 7
# Flatten → 3136
# Dense: 3136 → 512 (ReLU) → num_actions

comptime CONV1_OUT_H: Int = (84 + 2 * 0 - 8) // 4 + 1  # = 20
comptime CONV1_OUT_W: Int = (84 + 2 * 0 - 8) // 4 + 1  # = 20

comptime CONV2_OUT_H: Int = (CONV1_OUT_H + 2 * 0 - 4) // 2 + 1  # = 9
comptime CONV2_OUT_W: Int = (CONV1_OUT_W + 2 * 0 - 4) // 2 + 1  # = 9

comptime CONV3_OUT_H: Int = (CONV2_OUT_H + 2 * 0 - 3) // 1 + 1  # = 7
comptime CONV3_OUT_W: Int = (CONV2_OUT_W + 2 * 0 - 3) // 1 + 1  # = 7
comptime CONV3_FLAT: Int = 64 * CONV3_OUT_H * CONV3_OUT_W  # = 3136

comptime FC_HIDDEN: Int = 512


# =============================================================================
# PPO CNN Agent (GPU-only)
# =============================================================================


struct DeepPPOCNNAgent[
    num_actions: Int,
    rollout_len: Int = 128,
    n_envs: Int = 64,
    gpu_minibatch_size: Int = 256,
    actor_lr: Float64 = 0.00025,
    critic_lr: Float64 = 0.00025,
    profile: Int = 0,
](
    Checkpointable,
    GPUOnPolicyDiscreteAgent,
):
    """PPO agent with CNN for pixel observations (GPU-only).

    Uses the Nature DQN CNN architecture for both actor and critic,
    processing 4×84×84 stacked grayscale frames.

    Parameters:
        num_actions: Number of discrete actions.
        rollout_len: Steps per rollout per environment (default: 128).
        n_envs: Number of parallel GPU environments (default: 64).
        gpu_minibatch_size: Minibatch size for GPU training (default: 256).
        actor_lr: Actor learning rate (default: 2.5e-4).
        critic_lr: Critic learning rate (default: 2.5e-4).
    """

    comptime OBS = 4 * 84 * 84  # 28224 = PIXEL_OBS_DIM
    comptime ACTIONS = Self.num_actions
    comptime ROLLOUT = Self.rollout_len

    comptime TOTAL_ROLLOUT_SIZE: Int = Self.n_envs * Self.rollout_len
    comptime GPU_MINIBATCH = Self.gpu_minibatch_size

    # CNN Actor: shared backbone → logits
    comptime ActorModel = Sequential[
        Conv2DReLU[4, 32, 8, 4, 0, 84, 84],
        Conv2DReLU[32, 64, 4, 2, 0, CONV1_OUT_H, CONV1_OUT_W],
        Conv2DReLU[64, 64, 3, 1, 0, CONV2_OUT_H, CONV2_OUT_W],
        FlattenLayer[CONV3_FLAT],
        LinearReLU[CONV3_FLAT, FC_HIDDEN],
        Linear[FC_HIDDEN, Self.ACTIONS],
    ]
    comptime ActorNet = Network[Self.ActorModel, Adam[Self.actor_lr]]

    # CNN Critic: same backbone → value
    comptime CriticModel = Sequential[
        Conv2DReLU[4, 32, 8, 4, 0, 84, 84],
        Conv2DReLU[32, 64, 4, 2, 0, CONV1_OUT_H, CONV1_OUT_W],
        Conv2DReLU[64, 64, 3, 1, 0, CONV2_OUT_H, CONV2_OUT_W],
        FlattenLayer[CONV3_FLAT],
        LinearReLU[CONV3_FLAT, FC_HIDDEN],
        Linear[FC_HIDDEN, 1],
    ]
    comptime CriticNet = Network[Self.CriticModel, Adam[Self.critic_lr]]

    # GPUOnPolicyDiscreteAgent trait constants
    comptime OBS_DIM: Int = Self.OBS
    comptime NUM_ACTIONS: Int = Self.num_actions
    comptime ROLLOUT_LEN: Int = Self.rollout_len
    comptime MAX_N_ENVS: Int = Self.n_envs

    # State types (reuse PPO state containers parameterized by our CNN models)
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]
    comptime CPUStateType = PPODiscreteState[
        Self.ActorModel,
        Self.ActorOpt,
        Self.CriticModel,
        Self.CriticOpt,
        Self.OBS,
        Self.ACTIONS,
        Self.ROLLOUT,
    ]
    comptime GPUStateType = PPODiscreteGPUState[
        Self.ActorModel,
        Self.ActorOpt,
        Self.CriticModel,
        Self.CriticOpt,
        Self.OBS,
        Self.ACTIONS,
        Self.ROLLOUT,
        Self.n_envs,
        Self.GPU_MINIBATCH,
    ]

    # CPU state (actor + critic networks + rollout buffers)
    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var num_epochs: Int
    var normalize_advantages: Bool

    # Advanced hyperparameters
    var target_kl: Float64
    var max_grad_norm: Float64
    var anneal_lr: Bool
    var anneal_entropy: Bool
    var target_total_steps: Int
    var clip_value: Bool
    var norm_adv_per_minibatch: Bool

    # Training state
    var train_step_count: Int

    # Level-2 profiler: sub-phases of select_actions / compute_advantages / update_epochs
    var train_timer: PerfTimer[Self.profile >= 1]

    # Level-3 profiler: per-layer timing (base slot indices into train_timer)
    var actor_fwd_base: Int
    var actor_bwd_base: Int
    var critic_fwd_base: Int
    var critic_bwd_base: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 4,
        normalize_advantages: Bool = True,
        target_kl: Float64 = 0.015,
        max_grad_norm: Float64 = 0.5,
        anneal_lr: Bool = True,
        anneal_entropy: Bool = False,
        target_total_steps: Int = 0,
        clip_value: Bool = True,
        norm_adv_per_minibatch: Bool = True,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        self.state = Self.CPUStateType()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.normalize_advantages = normalize_advantages
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.anneal_entropy = anneal_entropy
        self.target_total_steps = target_total_steps
        self.clip_value = clip_value
        self.norm_adv_per_minibatch = norm_adv_per_minibatch
        self.train_step_count = 0
        self.train_timer = PerfTimer[Self.profile >= 1]()
        self.actor_fwd_base = 0
        self.actor_bwd_base = 0
        self.critic_fwd_base = 0
        self.critic_bwd_base = 0
        comptime if Self.profile >= 2:
            # select_actions sub-phases (slots 0-2)
            _ = self.train_timer.add_slot("actor_forward")
            _ = self.train_timer.add_slot("critic_forward")
            _ = self.train_timer.add_slot("action_sampling")
            # compute_advantages sub-phases (slots 3-6)
            _ = self.train_timer.add_slot("critic_bootstrap")
            _ = self.train_timer.add_slot("gpu_to_host_copy")
            _ = self.train_timer.add_slot("cpu_gae")
            _ = self.train_timer.add_slot("host_to_gpu_copy")
            # update_epochs sub-phases (slots 7-11)
            _ = self.train_timer.add_slot("gather_minibatch")
            _ = self.train_timer.add_slot("actor_forward_cached")
            _ = self.train_timer.add_slot("actor_backward")
            _ = self.train_timer.add_slot("critic_forward_cached")
            _ = self.train_timer.add_slot("critic_backward")
        comptime if Self.profile >= 3:
            # L3 slots as children of L2 update_epochs sub-phases
            # slot 8 = actor_forward_cached, 9 = actor_backward
            # slot 10 = critic_forward_cached, 11 = critic_backward
            self.actor_fwd_base = Self.ActorModel.register_forward_slots(
                self.train_timer, parent=8
            )
            self.actor_bwd_base = Self.ActorModel.register_backward_slots(
                self.train_timer, parent=9
            )
            self.critic_fwd_base = Self.CriticModel.register_forward_slots(
                self.train_timer, parent=10
            )
            self.critic_bwd_base = Self.CriticModel.register_backward_slots(
                self.train_timer, parent=11
            )
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    fn _perf_ptr(mut self) -> PerfTimerPtr:
        """Return opaque timer pointer for L3 profiling (null when profile < 3)."""
        comptime if Self.profile >= 3:
            return UnsafePointer(to=self.train_timer).bitcast[NoneType]()
        else:
            return PerfTimerPtr(unsafe_from_address=0)

    # =========================================================================
    # GPU State Management
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.gpu_actor.upload_from(self.state.actor, ctx)
        gpu_state.gpu_critic.upload_from(self.state.critic, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.gpu_actor.download_to(self.state.actor, ctx)
        gpu_state.gpu_critic.download_to(self.state.critic, ctx)
        ctx.synchronize()

    # =========================================================================
    # GPU Action Selection
    # =========================================================================

    fn select_actions_with_meta_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        mut log_probs_buf: DeviceBuffer[dtype],
        mut values_buf: DeviceBuffer[dtype],
        rng_seed: UInt32 = 0,
    ) raises -> None:
        """Forward actor + critic CNN on GPU and sample actions."""
        comptime blocks = (N_ENVS + TPB - 1) // TPB

        var actor_params_t = gpu_state.gpu_actor.params_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.logits_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var log_probs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())
        var values_t = LayoutTensor[
            dtype,
            Layout.row_major(N_ENVS, Self.CriticModel.OUT_DIM),
            MutAnyOrigin,
        ](values_buf.unsafe_ptr())

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_mark(ctx)

        Self.ActorModel.forward_gpu_no_cache[N_ENVS](
            ctx,
            logits_t,
            obs_t,
            actor_params_t,
            gpu_state.actor_env_workspace_buf,
        )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(0, ctx)
            self.train_timer.mark()

        Self.CriticModel.forward_gpu_no_cache[N_ENVS](
            ctx,
            values_t,
            obs_t,
            critic_params_t,
            gpu_state.critic_env_workspace_buf,
        )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(1, ctx)
            self.train_timer.mark()

        comptime sample_wrapper = _sample_actions_kernel[
            dtype, N_ENVS, Self.ACTIONS
        ]
        ctx.enqueue_function[sample_wrapper, sample_wrapper](
            logits_t,
            actions_t,
            log_probs_t,
            Scalar[DType.uint32](rng_seed),
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(2, ctx)

    # =========================================================================
    # GPU Advantage Computation
    # =========================================================================

    fn compute_advantages_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        final_obs_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Compute GAE advantages from the collected rollout (CPU-side)."""
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE

        var critic_params_t = gpu_state.gpu_critic.params_view()
        var final_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.OBS), MutAnyOrigin
        ](final_obs_buf.unsafe_ptr())
        var bootstrap_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.CriticModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu_state.values_env_buf.unsafe_ptr())

        # Forward critic on final obs to get bootstrap values
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_mark(ctx)

        Self.CriticModel.forward_gpu_no_cache[Self.n_envs](
            ctx,
            bootstrap_t,
            final_obs_t,
            critic_params_t,
            gpu_state.critic_env_workspace_buf,
        )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(3, ctx)
            self.train_timer.mark()

        ctx.enqueue_copy(
            gpu_state.bootstrap_values_host, gpu_state.values_env_buf
        )
        ctx.enqueue_copy(
            gpu_state.rollout_rewards_host, gpu_state.rollout_rewards_buf
        )
        ctx.enqueue_copy(
            gpu_state.rollout_values_host, gpu_state.rollout_values_buf
        )
        ctx.enqueue_copy(
            gpu_state.rollout_dones_host, gpu_state.rollout_dones_buf
        )
        ctx.synchronize()

        comptime if Self.profile >= 2:
            self.train_timer.accumulate(4)
            self.train_timer.mark()

        for env_idx in range(Self.n_envs):
            var gae = Scalar[dtype](0.0)
            var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
            var bootstrap_val = Scalar[dtype](
                gpu_state.bootstrap_values_host[env_idx]
            )

            for t in range(Self.ROLLOUT - 1, -1, -1):
                var idx = t * Self.n_envs + env_idx
                var reward = gpu_state.rollout_rewards_host[idx]
                var value = gpu_state.rollout_values_host[idx]
                var done = gpu_state.rollout_dones_host[idx]

                var next_val: Scalar[dtype]
                if t == Self.ROLLOUT - 1:
                    next_val = bootstrap_val
                else:
                    var next_idx = (t + 1) * Self.n_envs + env_idx
                    next_val = gpu_state.rollout_values_host[next_idx]

                if done > Scalar[dtype](0.5):
                    next_val = Scalar[dtype](0.0)
                    gae = Scalar[dtype](0.0)

                var delta = (
                    reward + Scalar[dtype](self.gamma) * next_val - value
                )
                gae = delta + gae_decay * gae
                gpu_state.advantages_host[idx] = gae
                gpu_state.returns_host[idx] = gae + value

        if self.normalize_advantages:
            var mean = Scalar[dtype](0.0)
            var var_sum = Scalar[dtype](0.0)
            for i in range(ROLLOUT_TOTAL):
                mean += gpu_state.advantages_host[i]
            mean /= Scalar[dtype](ROLLOUT_TOTAL)
            for i in range(ROLLOUT_TOTAL):
                var diff = gpu_state.advantages_host[i] - mean
                var_sum += diff * diff
            var std = sqrt(
                var_sum / Scalar[dtype](ROLLOUT_TOTAL) + Scalar[dtype](1e-8)
            )
            for i in range(ROLLOUT_TOTAL):
                gpu_state.advantages_host[i] = (
                    gpu_state.advantages_host[i] - mean
                ) / (std + Scalar[dtype](1e-8))

        comptime if Self.profile >= 2:
            self.train_timer.accumulate(5)
            self.train_timer.mark()

        ctx.enqueue_copy(gpu_state.advantages_buf, gpu_state.advantages_host)
        ctx.enqueue_copy(gpu_state.returns_buf, gpu_state.returns_host)
        ctx.synchronize()

        comptime if Self.profile >= 2:
            self.train_timer.accumulate(6)

    # =========================================================================
    # GPU PPO Update Epochs
    # =========================================================================

    fn update_epochs_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        update_idx: Int,
    ) raises -> None:
        """Run PPO multi-epoch minibatch updates on GPU."""
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ACTOR_PARAMS = Self.ActorModel.PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CriticModel.PARAM_SIZE
        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB

        var actor_params_t = gpu_state.gpu_actor.params_view()
        var actor_grads_t = gpu_state.gpu_actor.grads_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()
        var critic_grads_t = gpu_state.gpu_critic.grads_view()

        var mb_obs_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.mb_obs_buf.unsafe_ptr())
        var mb_actions_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_actions_buf.unsafe_ptr())
        var mb_advantages_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_advantages_buf.unsafe_ptr())
        var mb_returns_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_returns_buf.unsafe_ptr())
        var mb_old_log_probs_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_old_log_probs_buf.unsafe_ptr())
        var mb_old_values_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_old_values_buf.unsafe_ptr())
        var mb_indices_t = LayoutTensor[
            DType.int32, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_indices_buf.unsafe_ptr())

        var rollout_obs_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL, Self.OBS), MutAnyOrigin
        ](gpu_state.rollout_obs_buf.unsafe_ptr())
        var rollout_actions_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.rollout_actions_buf.unsafe_ptr())
        var advantages_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.advantages_buf.unsafe_ptr())
        var returns_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.returns_buf.unsafe_ptr())
        var rollout_log_probs_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.rollout_log_probs_buf.unsafe_ptr())
        var rollout_values_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.rollout_values_buf.unsafe_ptr())

        var actor_logits_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.actor_logits_buf.unsafe_ptr())
        var actor_grad_output_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.actor_grad_output_buf.unsafe_ptr())
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu_state.actor_cache_buf.unsafe_ptr())
        var actor_grad_input_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ActorModel.IN_DIM),
            MutAnyOrigin,
        ](gpu_state.actor_grad_input_buf.unsafe_ptr())

        var critic_values_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](gpu_state.critic_values_buf.unsafe_ptr())
        var critic_grad_output_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](gpu_state.critic_grad_output_buf.unsafe_ptr())
        var critic_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu_state.critic_cache_buf.unsafe_ptr())
        var critic_grad_input_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.CriticModel.IN_DIM),
            MutAnyOrigin,
        ](gpu_state.critic_grad_input_buf.unsafe_ptr())

        var kl_divergences_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.kl_divergences_buf.unsafe_ptr())
        var actor_grad_partial_sums_t = LayoutTensor[
            dtype, Layout.row_major(ACTOR_GRAD_BLOCKS), MutAnyOrigin
        ](gpu_state.actor_grad_partial_sums_buf.unsafe_ptr())
        var critic_grad_partial_sums_t = LayoutTensor[
            dtype, Layout.row_major(CRITIC_GRAD_BLOCKS), MutAnyOrigin
        ](gpu_state.critic_grad_partial_sums_buf.unsafe_ptr())
        var actor_scale_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.actor_scale_buf.unsafe_ptr())
        var critic_scale_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.critic_scale_buf.unsafe_ptr())

        # Entropy annealing
        var current_entropy_coef = self.entropy_coef
        if self.anneal_entropy and self.target_total_steps > 0:
            var estimated_steps = update_idx * ROLLOUT_TOTAL
            var progress = Float64(estimated_steps) / Float64(
                self.target_total_steps
            )
            if progress > 1.0:
                progress = 1.0
            current_entropy_coef = self.entropy_coef * (1.0 - progress)

        # Kernel wrappers
        comptime gather_wrapper = ppo_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, ROLLOUT_TOTAL
        ]
        comptime actor_grad_with_kl_wrapper = ppo_actor_grad_with_kl_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime critic_grad_wrapper = ppo_critic_grad_kernel[dtype, MINIBATCH]
        comptime critic_grad_clipped_wrapper = ppo_critic_grad_clipped_kernel[
            dtype, MINIBATCH
        ]
        comptime normalize_advantages_wrapper = normalize_advantages_kernel[
            dtype, MINIBATCH
        ]
        comptime actor_grad_norm_wrapper = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_wrapper = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime actor_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime actor_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, ACTOR_PARAMS
        ]
        comptime critic_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime critic_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, CRITIC_PARAMS
        ]

        var kl_early_stop = False
        var num_minibatches = ROLLOUT_TOTAL // MINIBATCH

        for epoch in range(self.num_epochs):
            if kl_early_stop:
                break

            # Fisher-Yates shuffle on CPU
            var indices_list = List[Int]()
            for i in range(ROLLOUT_TOTAL):
                indices_list.append(i)
            for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                var j = Int(random_float64() * Float64(i + 1))
                var temp = indices_list[i]
                indices_list[i] = indices_list[j]
                indices_list[j] = temp

            for mb_idx in range(num_minibatches):
                if kl_early_stop:
                    break
                var start_idx = mb_idx * MINIBATCH

                comptime if Self.profile >= 2:
                    self.train_timer.sync_and_mark(ctx)

                for i in range(MINIBATCH):
                    gpu_state.mb_indices_host[i] = Int32(
                        indices_list[start_idx + i]
                    )
                ctx.enqueue_copy(
                    gpu_state.mb_indices_buf, gpu_state.mb_indices_host
                )

                # Gather minibatch from rollout
                ctx.enqueue_function[gather_wrapper, gather_wrapper](
                    mb_obs_t,
                    mb_actions_t,
                    mb_advantages_t,
                    mb_returns_t,
                    mb_old_log_probs_t,
                    mb_old_values_t,
                    rollout_obs_t,
                    rollout_actions_t,
                    advantages_t,
                    returns_t,
                    rollout_log_probs_t,
                    rollout_values_t,
                    mb_indices_t,
                    MINIBATCH,
                    grid_dim=(MINIBATCH_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                # Per-minibatch advantage normalization
                if self.norm_adv_per_minibatch:
                    ctx.enqueue_copy(
                        gpu_state.mb_advantages_host,
                        gpu_state.mb_advantages_buf,
                    )
                    ctx.synchronize()
                    var adv_mean = Scalar[dtype](0.0)
                    for i in range(MINIBATCH):
                        adv_mean += gpu_state.mb_advantages_host[i]
                    adv_mean /= Scalar[dtype](MINIBATCH)
                    var adv_var = Scalar[dtype](0.0)
                    for i in range(MINIBATCH):
                        var diff = gpu_state.mb_advantages_host[i] - adv_mean
                        adv_var += diff * diff
                    var adv_std = sqrt(
                        adv_var / Scalar[dtype](MINIBATCH) + Scalar[dtype](1e-8)
                    )
                    ctx.enqueue_function[
                        normalize_advantages_wrapper,
                        normalize_advantages_wrapper,
                    ](
                        mb_advantages_t,
                        adv_mean,
                        adv_std,
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                comptime if Self.profile >= 2:
                    self.train_timer.accumulate(7)
                    self.train_timer.mark()

                # ---- Train actor ----
                gpu_state.gpu_actor.zero_grads(ctx)
                Self.ActorModel.forward_gpu[MINIBATCH](
                    ctx,
                    actor_logits_t,
                    mb_obs_t,
                    actor_params_t,
                    actor_cache_t,
                    gpu_state.actor_mb_workspace_buf,
                    perf=self._perf_ptr(),
                    perf_slot=self.actor_fwd_base,
                )
                ctx.synchronize()

                ctx.enqueue_function[
                    actor_grad_with_kl_wrapper, actor_grad_with_kl_wrapper
                ](
                    actor_grad_output_t,
                    kl_divergences_t,
                    actor_logits_t,
                    mb_old_log_probs_t,
                    mb_advantages_t,
                    mb_actions_t,
                    Scalar[dtype](self.clip_epsilon),
                    Scalar[dtype](current_entropy_coef),
                    MINIBATCH,
                    grid_dim=(MINIBATCH_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                if self.target_kl > 0.0:
                    ctx.enqueue_copy(
                        gpu_state.kl_divergences_host,
                        gpu_state.kl_divergences_buf,
                    )
                    ctx.synchronize()
                    var kl_sum = Scalar[dtype](0.0)
                    for i in range(MINIBATCH):
                        kl_sum += gpu_state.kl_divergences_host[i]
                    if Float64(kl_sum) / Float64(MINIBATCH) > self.target_kl:
                        kl_early_stop = True
                        break

                comptime if Self.profile >= 2:
                    self.train_timer.accumulate(8)
                    self.train_timer.mark()

                Self.ActorModel.backward_gpu[MINIBATCH](
                    ctx,
                    actor_grad_input_t,
                    actor_grad_output_t,
                    actor_params_t,
                    actor_cache_t,
                    actor_grads_t,
                    gpu_state.actor_mb_workspace_buf,
                    perf=self._perf_ptr(),
                    perf_slot=self.actor_bwd_base,
                )

                if self.max_grad_norm > 0.0:
                    ctx.enqueue_function[
                        actor_grad_norm_wrapper, actor_grad_norm_wrapper
                    ](
                        actor_grad_partial_sums_t,
                        actor_grads_t,
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        actor_reduce_scale_wrapper, actor_reduce_scale_wrapper
                    ](
                        actor_scale_t,
                        actor_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        actor_apply_scale_wrapper, actor_apply_scale_wrapper
                    ](
                        actor_grads_t,
                        actor_scale_t,
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                gpu_state.gpu_actor.optimizer_step(ctx)
                ctx.synchronize()

                comptime if Self.profile >= 2:
                    self.train_timer.accumulate(9)
                    self.train_timer.mark()

                # ---- Train critic ----
                gpu_state.gpu_critic.zero_grads(ctx)
                Self.CriticModel.forward_gpu[MINIBATCH](
                    ctx,
                    critic_values_t,
                    mb_obs_t,
                    critic_params_t,
                    critic_cache_t,
                    gpu_state.critic_mb_workspace_buf,
                    perf=self._perf_ptr(),
                    perf_slot=self.critic_fwd_base,
                )
                ctx.synchronize()

                if self.clip_value:
                    ctx.enqueue_function[
                        critic_grad_clipped_wrapper, critic_grad_clipped_wrapper
                    ](
                        critic_grad_output_t,
                        critic_values_t,
                        mb_returns_t,
                        mb_old_values_t,
                        Scalar[dtype](self.clip_epsilon),
                        Scalar[dtype](self.value_loss_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                else:
                    ctx.enqueue_function[
                        critic_grad_wrapper, critic_grad_wrapper
                    ](
                        critic_grad_output_t,
                        critic_values_t,
                        mb_returns_t,
                        Scalar[dtype](self.value_loss_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                ctx.synchronize()

                comptime if Self.profile >= 2:
                    self.train_timer.accumulate(10)
                    self.train_timer.mark()

                Self.CriticModel.backward_gpu[MINIBATCH](
                    ctx,
                    critic_grad_input_t,
                    critic_grad_output_t,
                    critic_params_t,
                    critic_cache_t,
                    critic_grads_t,
                    gpu_state.critic_mb_workspace_buf,
                    perf=self._perf_ptr(),
                    perf_slot=self.critic_bwd_base,
                )

                if self.max_grad_norm > 0.0:
                    ctx.enqueue_function[
                        critic_grad_norm_wrapper, critic_grad_norm_wrapper
                    ](
                        critic_grad_partial_sums_t,
                        critic_grads_t,
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        critic_reduce_scale_wrapper, critic_reduce_scale_wrapper
                    ](
                        critic_scale_t,
                        critic_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        critic_apply_scale_wrapper, critic_apply_scale_wrapper
                    ](
                        critic_grads_t,
                        critic_scale_t,
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                gpu_state.gpu_critic.optimizer_step(ctx)
                ctx.synchronize()

                comptime if Self.profile >= 2:
                    self.train_timer.accumulate(11)

        gpu_state.rollout_step = 0
        self.train_step_count += 1

    # =========================================================================
    # GPU Training
    # =========================================================================

    fn train_gpu[
        EnvType: GPUDiscreteEnv,
    ](
        mut self,
        ctx: DeviceContext,
        num_updates: Int,
        verbose: Bool = False,
        print_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train PPO CNN on GPU with parallel environments.

        Args:
            ctx: GPU device context.
            num_updates: Number of rollout+update cycles.
            verbose: Whether to print progress.
            print_every: Print progress every N updates.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        var timer = PerfTimer[Self.profile >= 1]()
        _ = timer.add_slot("select_actions")
        _ = timer.add_slot("store_pre_step")
        _ = timer.add_slot("env_step")
        _ = timer.add_slot("store_post_step")
        _ = timer.add_slot("episode_tracking")
        _ = timer.add_slot("reset")
        _ = timer.add_slot("compute_advantages")
        _ = timer.add_slot("update_epochs")
        _ = timer.add_slot("gpu_cpu_sync")
        var metrics = run_onpolicy_discrete_train_gpu[EnvType, Self, Self.profile](
            self,
            ctx,
            num_updates,
            timer,
            verbose=verbose,
            print_every=print_every,
        )

        comptime if Self.profile >= 2:
            timer.merge_subtree_range(0, self.train_timer, 0, 3)
            timer.merge_subtree_range(6, self.train_timer, 3, 7)
            timer.merge_subtree_range(7, self.train_timer, 7, 12)

        comptime if Self.profile >= 1:
            timer.print_report("PPO CNN (GPU) Profile")
        return metrics^

    # =========================================================================
    # Checkpoint Save/Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        var actor_param_size = Self.ActorModel.PARAM_SIZE
        var critic_param_size = Self.CriticModel.PARAM_SIZE
        var actor_state_size = (
            actor_param_size * Adam[Self.actor_lr].STATE_PER_PARAM
        )
        var critic_state_size = (
            critic_param_size * Adam[Self.critic_lr].STATE_PER_PARAM
        )

        var content = String("# mojo-rl checkpoint v1\n")
        content += "# type: ppo_cnn_agent\n"
        content += "# actor_param_size: " + String(actor_param_size) + "\n"
        content += "# critic_param_size: " + String(critic_param_size) + "\n"

        content += "actor_params:\n"
        for i in range(actor_param_size):
            content += String(Float64((self.state.actor.params + i)[])) + "\n"

        content += "actor_optimizer_state:\n"
        for i in range(actor_state_size):
            content += (
                String(Float64((self.state.actor.optimizer_state + i)[])) + "\n"
            )

        content += "critic_params:\n"
        for i in range(critic_param_size):
            content += String(Float64((self.state.critic.params + i)[])) + "\n"

        content += "critic_optimizer_state:\n"
        for i in range(critic_state_size):
            content += (
                String(Float64((self.state.critic.optimizer_state + i)[]))
                + "\n"
            )

        content += "metadata:\n"
        content += "gamma=" + String(self.gamma) + "\n"
        content += "gae_lambda=" + String(self.gae_lambda) + "\n"
        content += "clip_epsilon=" + String(self.clip_epsilon) + "\n"
        content += "actor_lr=" + String(Float64(Self.actor_lr)) + "\n"
        content += "critic_lr=" + String(Float64(Self.critic_lr)) + "\n"
        content += "entropy_coef=" + String(self.entropy_coef) + "\n"
        content += "value_loss_coef=" + String(self.value_loss_coef) + "\n"
        content += "num_epochs=" + String(self.num_epochs) + "\n"
        content += (
            "normalize_advantages=" + String(self.normalize_advantages) + "\n"
        )
        content += "target_kl=" + String(self.target_kl) + "\n"
        content += "max_grad_norm=" + String(self.max_grad_norm) + "\n"
        content += "train_step_count=" + String(self.train_step_count) + "\n"

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        var actor_param_size = Self.ActorModel.PARAM_SIZE
        var critic_param_size = Self.CriticModel.PARAM_SIZE
        var actor_state_size = (
            actor_param_size * Adam[Self.actor_lr].STATE_PER_PARAM
        )
        var critic_state_size = (
            critic_param_size * Adam[Self.critic_lr].STATE_PER_PARAM
        )

        var content = read_checkpoint_file(filepath)
        var lines = split_lines(content)

        var actor_params_start = find_section_start(lines, "actor_params:")
        for i in range(actor_param_size):
            (self.state.actor.params + i)[] = Scalar[dtype](
                atof(lines[actor_params_start + i])
            )

        var actor_state_start = find_section_start(
            lines, "actor_optimizer_state:"
        )
        for i in range(actor_state_size):
            (self.state.actor.optimizer_state + i)[] = Scalar[dtype](
                atof(lines[actor_state_start + i])
            )

        var critic_params_start = find_section_start(lines, "critic_params:")
        for i in range(critic_param_size):
            (self.state.critic.params + i)[] = Scalar[dtype](
                atof(lines[critic_params_start + i])
            )

        var critic_state_start = find_section_start(
            lines, "critic_optimizer_state:"
        )
        for i in range(critic_state_size):
            (self.state.critic.optimizer_state + i)[] = Scalar[dtype](
                atof(lines[critic_state_start + i])
            )

        var metadata_start = find_section_start(lines, "metadata:")
        for i in range(metadata_start, len(lines)):
            var line = lines[i]
            if line.startswith("gamma="):
                self.gamma = atof(String(line[6:]))
            elif line.startswith("gae_lambda="):
                self.gae_lambda = atof(String(line[11:]))
            elif line.startswith("clip_epsilon="):
                self.clip_epsilon = atof(String(line[13:]))
            elif line.startswith("entropy_coef="):
                self.entropy_coef = atof(String(line[13:]))
            elif line.startswith("value_loss_coef="):
                self.value_loss_coef = atof(String(line[16:]))
            elif line.startswith("num_epochs="):
                self.num_epochs = Int(atol(String(line[11:])))
            elif line.startswith("normalize_advantages="):
                self.normalize_advantages = String(line[21:]) == "True"
            elif line.startswith("target_kl="):
                self.target_kl = atof(String(line[10:]))
            elif line.startswith("max_grad_norm="):
                self.max_grad_norm = atof(String(line[14:]))
            elif line.startswith("train_step_count="):
                self.train_step_count = Int(atol(String(line[17:])))


# =============================================================================
# GPU Kernel: Sample actions from categorical distribution
# =============================================================================


@always_inline
fn _sample_actions_kernel[
    dtype: DType where dtype.is_floating_point(),
    N_ENVS: Int,
    NUM_ACTIONS: Int,
](
    logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, NUM_ACTIONS), MutAnyOrigin
    ],
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    seed: Scalar[DType.uint32],
):
    """Sample actions from categorical distribution and compute log probs."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    var rng_state = UInt32(seed) ^ (UInt32(i) * 2654435761)
    rng_state = xorshift32(rng_state)

    var max_logit = logits[i, 0]
    for a in range(1, NUM_ACTIONS):
        var l = logits[i, a]
        if l > max_logit:
            max_logit = l

    var sum_exp = logits[i, 0] - logits[i, 0]
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        sum_exp = sum_exp + exp(logit_val)

    var rand_result = random_uniform[dtype](rng_state)
    var rand_val = rand_result[0]
    rng_state = rand_result[1]

    var cumsum_val = Scalar[dtype](0.0)
    var selected_action: actions.element_type = 0
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        var prob = exp(logit_val) / sum_exp
        var prob_scalar = Scalar[dtype](prob[0])
        cumsum_val = cumsum_val + prob_scalar
        if rand_val < cumsum_val:
            selected_action = Scalar[dtype](a)
            break

    actions[i] = selected_action

    var logit_sel = logits[i, Int(selected_action)] - max_logit
    var selected_prob_simd = exp(logit_sel) / sum_exp
    var selected_prob = Float32(selected_prob_simd[0])
    var eps = Float32(1e-8)
    var log_prob_val = log(selected_prob + eps)
    log_probs[i] = Scalar[dtype](log_prob_val)
