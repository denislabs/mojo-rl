"""Generic continuous-action on-policy agent parameterized by ContinuousOnPolicyConfig.

Supports continuous PPO via strategy dispatch on Config.PolicyGrad and Config.EpochSched.
Uses unbounded Gaussian policy (StochasticActor) — actions clipped at environment boundary.

GPU support via GPUOnPolicyContinuousAgent trait + run_onpolicy_continuous_train_gpu.
"""

from std.math import exp, log, sqrt, cos
from std.random import random_float64
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.nn.model.stochastic_actor import LOG_STD_MIN, LOG_STD_MAX, EPS
from mojo_rl.deep_agents.core.eval import (
    run_onpolicy_continuous_eval,
)
from mojo_rl.deep_agents.core import (
    Checkpointable,
    GPUOnPolicyState,
    GPUOnPolicyContinuousAgent,
)
from mojo_rl.deep_agents.core.onpolicy_train import (
    OnPolicyContinuousAgent,
    OnPolicyContinuousState,
)
from mojo_rl.deep_agents.core.gpu_onpolicy_train import (
    run_onpolicy_continuous_train_gpu,
)
from mojo_rl.deep_agents.core.onpolicy_helpers import (
    compute_gae_list,
    normalize_advantages_list,
    fisher_yates_shuffle,
)
from mojo_rl.deep_agents.core.perf_timer import PerfTimer
from mojo_rl.core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.core.logger import Logger, NoOpLogger

# Reuse existing GPU state and kernels from ppo/
from mojo_rl.deep_agents.ppo.state import (
    PPOContinuousState,
    PPOContinuousGPUState,
)
from mojo_rl.deep_agents.ppo.kernels import (
    _sample_continuous_actions_kernel,
    _store_continuous_pre_step_kernel,
    _store_pre_step_obs_parallel_kernel,
    _store_post_step_kernel,
    ppo_continuous_gather_minibatch_kernel,
    ppo_gather_minibatch_obs_parallel_kernel,
    ppo_continuous_actor_grad_kernel,
    ppo_critic_grad_kernel,
    ppo_critic_grad_clipped_kernel,
    normalize_advantages_kernel,
    gradient_norm_kernel,
    gradient_reduce_and_compute_scale_kernel,
    gradient_apply_scale_kernel,
    clamp_log_std_params_kernel,
)

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from .onpolicy_config import ContinuousOnPolicyConfig


# =============================================================================
# GenericOnPolicyContinuousAgent
# =============================================================================


struct GenericOnPolicyContinuousAgent[
    Config: ContinuousOnPolicyConfig,
    n_envs: Int = 64,
    gpu_minibatch_size: Int = 256,
    L: Logger = NoOpLogger,
](OnPolicyContinuousAgent & GPUOnPolicyContinuousAgent & Checkpointable):
    """Generic continuous-action on-policy agent (PPO).

    Uses unbounded Gaussian policy (StochasticActor) with CleanRL-style
    action clipping at environment boundary.

    Parameters:
        Config: Continuous on-policy config (ContinuousPPOConfig).
        n_envs: Number of parallel environments for GPU training (default: 64).
        gpu_minibatch_size: Minibatch size for GPU update epochs (default: 256).
        L: Logger type for diagnostic logging (default: NoOpLogger).
    """

    # Derive dimensions from Config Model types
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.action_dim
    comptime ACTOR_OUT: Int = Self.Config.ActorModel.OUT_DIM  # 2 * action_dim
    comptime ROLLOUT: Int = Self.Config.rollout_len
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]
    comptime ActorModel = Self.Config.ActorModel
    comptime CriticModel = Self.Config.CriticModel
    comptime ActorOpt = Self.Config.ActorOpt
    comptime CriticOpt = Self.Config.CriticOpt

    comptime CPUStateType = PPOContinuousState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.action_dim,
        Self.Config.rollout_len,
    ]

    # GPU-specific comptime constants
    comptime TOTAL_ROLLOUT_SIZE: Int = Self.n_envs * Self.Config.rollout_len
    comptime GPU_MINIBATCH: Int = Self.gpu_minibatch_size

    # GPUOnPolicyContinuousAgent trait constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACTIONS
    comptime ROLLOUT_LEN: Int = Self.ROLLOUT
    comptime MAX_N_ENVS: Int = Self.n_envs

    comptime GPUStateType = PPOContinuousGPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.OBS,
        Self.ACTIONS,
        Self.ROLLOUT,
        Self.n_envs,
        Self.GPU_MINIBATCH,
    ]

    # Persistent CPU state
    var cpu_state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var gae_lambda: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var max_grad_norm: Float64
    var normalize_advantages: Bool

    # PPO-specific
    var clip_epsilon: Float64
    var num_epochs: Int
    var minibatch_size: Int
    var target_kl: Float64
    var clip_value: Bool
    var norm_adv_per_minibatch: Bool

    # Training state
    var train_step_count: Int
    var target_total_steps: Int

    # Checkpoint
    var checkpoint_every: Int
    var checkpoint_path: String

    # Logger
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        max_grad_norm: Float64 = 0.5,
        normalize_advantages: Bool = True,
        clip_epsilon: Float64 = 0.2,
        num_epochs: Int = 4,
        minibatch_size: Int = 64,
        target_kl: Float64 = 0.015,
        clip_value: Bool = True,
        norm_adv_per_minibatch: Bool = True,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        target_total_steps: Int = 0,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.clip_epsilon = clip_epsilon
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.clip_value = clip_value
        self.norm_adv_per_minibatch = norm_adv_per_minibatch
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

        # Initialize CPU state (actor Kaiming, then critic Kaiming)
        self.cpu_state = Self.CPUStateType()
        self.cpu_state.critic.initialize[Kaiming[]]()

    # =========================================================================
    # OnPolicyContinuousAgent trait
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        var s = Self.CPUStateType()
        s.critic.initialize[Kaiming[]]()
        return s^

    fn collect_rollout[
        E: BoxContinuousActionEnv
    ](mut self, mut cpu_state: Self.CPUStateType, mut env: E) -> None:
        if not cpu_state._env_initialized:
            var obs_list = env.reset_obs_list()
            for i in range(Self.OBS):
                cpu_state._current_obs[i] = Scalar[dtype](obs_list[i])
            cpu_state._env_initialized = True

        cpu_state.buffer_idx = 0

        for _ in range(Self.ROLLOUT):
            # Build obs tensor
            var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                uninitialized=True
            )
            for i in range(Self.OBS):
                obs_arr[i] = cpu_state._current_obs[i]
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](obs_arr.unsafe_ptr())

            # Actor forward → mean + log_std
            var actor_out = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
                uninitialized=True
            )
            var actor_out_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
            ](actor_out.unsafe_ptr())
            var a_p = cpu_state.actor.params_view()
            Self.ActorNet.forward[1](obs_t, actor_out_t, a_p)

            # Extract mean and log_std, sample action
            var action = List[Scalar[dtype]](capacity=Self.ACTIONS)
            var total_lp: Float64 = 0.0
            for j in range(Self.ACTIONS):
                var mean = Float64(actor_out[j])
                var raw_ls = Float64(actor_out[Self.ACTIONS + j])
                # Clamp log_std
                if raw_ls < LOG_STD_MIN:
                    raw_ls = LOG_STD_MIN
                elif raw_ls > LOG_STD_MAX:
                    raw_ls = LOG_STD_MAX
                var std = exp(raw_ls)

                # Sample via Box-Muller
                var u1 = random_float64()
                var u2 = random_float64()
                if u1 < 1e-10:
                    u1 = 1e-10
                var noise = sqrt(-2.0 * log(u1)) * cos(
                    2.0 * 3.14159265358979 * u2
                )
                var a = mean + std * noise
                action.append(Scalar[dtype](a))

                # Log prob (Gaussian)
                var diff = (a - mean) / (std + 1e-8)
                total_lp += -0.5 * (
                    diff * diff + 2.0 * raw_ls + 0.9189385332
                )  # log(2*pi)/2

            # Critic forward → value
            var val_arr = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                uninitialized=True
            )
            var val_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.CRITIC_OUT), MutAnyOrigin
            ](val_arr.unsafe_ptr())
            var c_obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
            ](obs_arr.unsafe_ptr())
            var c_p = cpu_state.critic.params_view()
            Self.CriticNet.forward[1](c_obs_t, val_t, c_p)

            # Store step
            var obs_list = List[Scalar[dtype]](capacity=Self.OBS)
            for i in range(Self.OBS):
                obs_list.append(cpu_state._current_obs[i])
            var action_copy = List[Scalar[dtype]](capacity=Self.ACTIONS)
            for i in range(Self.ACTIONS):
                action_copy.append(action[i])
            cpu_state.store_step(
                obs_list,
                action_copy,
                Float64(0),  # reward filled after env step
                val_arr[0],
                Scalar[dtype](total_lp),
                False,
            )

            # Step environment
            var result = env.step_continuous_vec(action)
            var next_obs = result[0].copy()
            var reward = Float64(result[1])
            var done = result[2]

            # Update reward in buffer (overwrite the 0 we stored)
            var idx = cpu_state.buffer_idx - 1
            cpu_state.buffer_rewards[idx] = Scalar[dtype](reward)
            cpu_state.buffer_dones[idx - (idx if idx == 0 else 0)] = done

            # Update current obs
            if done:
                var reset_obs = env.reset_obs_list()
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](reset_obs[i])
            else:
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](next_obs[i])

    fn compute_advantages(mut self, mut cpu_state: Self.CPUStateType) -> None:
        # Bootstrap value from critic on current obs
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_arr[i] = cpu_state._current_obs[i]
        var c_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var val_arr = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
            uninitialized=True
        )
        var val_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.CRITIC_OUT), MutAnyOrigin
        ](val_arr.unsafe_ptr())
        var c_p = cpu_state.critic.params_view()
        Self.CriticNet.forward[1](c_obs_t, val_t, c_p)
        var next_value = val_arr[0]

        compute_gae_list[dtype](
            cpu_state.buffer_rewards,
            cpu_state.buffer_values,
            cpu_state.buffer_dones,
            next_value,
            Self.ROLLOUT,
            self.gamma,
            self.gae_lambda,
            cpu_state._advantages,
            cpu_state._returns,
        )

        if self.normalize_advantages:
            normalize_advantages_list[dtype](
                cpu_state._advantages, Self.ROLLOUT
            )

    fn update_epochs(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """Multi-epoch minibatch PPO update for continuous actions."""
        var n_epochs = Self.Config.EpochSched.get_num_epochs(self.num_epochs)
        var mb_size = Self.Config.EpochSched.get_minibatch_size(
            self.minibatch_size, Self.ROLLOUT
        )
        var num_minibatches = Self.ROLLOUT // mb_size
        var total_loss: Float64 = 0.0
        var n_updates = 0

        for epoch in range(n_epochs):
            # Shuffle indices (PPO)
            comptime if Self.Config.EpochSched.USES_SHUFFLE:
                fisher_yates_shuffle(cpu_state._indices, Self.ROLLOUT)

            for mb_idx in range(num_minibatches):
                var start = mb_idx * mb_size

                # Per-minibatch advantage normalization
                comptime if Self.Config.EpochSched.SUPPORTS_MINIBATCH_NORM:
                    if self.norm_adv_per_minibatch:
                        var mb_mean = Scalar[dtype](0.0)
                        for i in range(mb_size):
                            mb_mean += cpu_state._advantages[
                                cpu_state._indices[start + i]
                            ]
                        mb_mean /= Scalar[dtype](mb_size)
                        var mb_var = Scalar[dtype](0.0)
                        for i in range(mb_size):
                            var diff = (
                                cpu_state._advantages[
                                    cpu_state._indices[start + i]
                                ]
                                - mb_mean
                            )
                            mb_var += diff * diff
                        var mb_std = sqrt(
                            mb_var / Scalar[dtype](mb_size)
                            + Scalar[dtype](1e-8)
                        )
                        for i in range(mb_size):
                            var idx = cpu_state._indices[start + i]
                            cpu_state._advantages[idx] = (
                                cpu_state._advantages[idx] - mb_mean
                            ) / (mb_std + Scalar[dtype](1e-8))

                for s in range(mb_size):
                    var idx = cpu_state._indices[start + s]

                    # Build obs tensor for this sample
                    var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                        uninitialized=True
                    )
                    for d in range(Self.OBS):
                        obs_arr[d] = cpu_state.buffer_obs[idx * Self.OBS + d]
                    var obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](obs_arr.unsafe_ptr())

                    # Actor forward with cache
                    var actor_out = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
                        uninitialized=True
                    )
                    var actor_out_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.ACTOR_OUT),
                        MutAnyOrigin,
                    ](actor_out.unsafe_ptr())
                    var actor_cache = InlineArray[Scalar[dtype], Self.ACTOR_CS](
                        uninitialized=True
                    )
                    var actor_cache_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.ACTOR_CS),
                        MutAnyOrigin,
                    ](actor_cache.unsafe_ptr())
                    var a_p = cpu_state.actor.params_view()
                    Self.ActorNet.forward_with_cache[1](
                        obs_t, actor_out_t, a_p, actor_cache_t
                    )

                    # Compute new log_prob from stored action
                    var new_lp: Float64 = 0.0
                    var entropy: Float64 = 0.0
                    for j in range(Self.ACTIONS):
                        var mean = Float64(actor_out[j])
                        var raw_ls = Float64(actor_out[Self.ACTIONS + j])
                        if raw_ls < LOG_STD_MIN:
                            raw_ls = LOG_STD_MIN
                        elif raw_ls > LOG_STD_MAX:
                            raw_ls = LOG_STD_MAX
                        var std = exp(raw_ls)
                        var stored_a = Float64(
                            cpu_state.buffer_actions[idx * Self.ACTIONS + j]
                        )
                        var diff = (stored_a - mean) / (std + 1e-8)
                        new_lp += -0.5 * (
                            diff * diff + 2.0 * raw_ls + 0.9189385332
                        )
                        entropy += (
                            raw_ls + 0.5 * 1.8378770664
                        )  # 0.5 * (1 + log(2*pi))

                    var old_lp = Float64(cpu_state.buffer_log_probs[idx])
                    var advantage = Float64(cpu_state._advantages[idx])

                    # Clipped surrogate
                    var ratio = exp(new_lp - old_lp)
                    var surr1 = ratio * advantage
                    var clipped_ratio = ratio
                    if clipped_ratio > 1.0 + self.clip_epsilon:
                        clipped_ratio = 1.0 + self.clip_epsilon
                    elif clipped_ratio < 1.0 - self.clip_epsilon:
                        clipped_ratio = 1.0 - self.clip_epsilon
                    var surr2 = clipped_ratio * advantage
                    var policy_loss = -(surr1 if surr1 < surr2 else surr2)

                    # Compute actor gradient (d_loss / d_actor_out)
                    var grad_out = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
                        uninitialized=True
                    )
                    var is_clipped = ratio != clipped_ratio

                    for j in range(Self.ACTIONS):
                        var mean = Float64(actor_out[j])
                        var raw_ls = Float64(actor_out[Self.ACTIONS + j])
                        if raw_ls < LOG_STD_MIN:
                            raw_ls = LOG_STD_MIN
                        elif raw_ls > LOG_STD_MAX:
                            raw_ls = LOG_STD_MAX
                        var std = exp(raw_ls)
                        var stored_a = Float64(
                            cpu_state.buffer_actions[idx * Self.ACTIONS + j]
                        )
                        var diff = (stored_a - mean) / (std + 1e-8)

                        # d(log_prob) / d(mean) = (a - mean) / std^2
                        var dlp_dmean = diff / (std + 1e-8)
                        # d(log_prob) / d(log_std) = (diff^2 - 1)
                        var dlp_dls = diff * diff - 1.0

                        if not is_clipped:
                            # Policy gradient
                            grad_out[j] = Scalar[dtype](
                                -advantage * ratio * dlp_dmean
                            )
                            grad_out[Self.ACTIONS + j] = Scalar[dtype](
                                -advantage * ratio * dlp_dls - self.entropy_coef
                            )
                        else:
                            # Only entropy gradient when clipped
                            grad_out[j] = Scalar[dtype](0.0)
                            grad_out[Self.ACTIONS + j] = Scalar[dtype](
                                -self.entropy_coef
                            )

                    # Backward actor
                    var grad_out_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.ACTOR_OUT),
                        MutAnyOrigin,
                    ](grad_out.unsafe_ptr())
                    var grad_in = InlineArray[Scalar[dtype], Self.OBS](
                        uninitialized=True
                    )
                    var grad_in_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](grad_in.unsafe_ptr())
                    var a_g = cpu_state.actor.grads_view()
                    cpu_state.actor.zero_grads()
                    Self.ActorNet.backward[1](
                        grad_out_t, grad_in_t, a_p, actor_cache_t, a_g
                    )
                    cpu_state.actor.optimizer_step()

                    # Critic forward with cache
                    var val_arr2 = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                        uninitialized=True
                    )
                    var val_t2 = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_OUT),
                        MutAnyOrigin,
                    ](val_arr2.unsafe_ptr())
                    var c_cache = InlineArray[Scalar[dtype], Self.CRITIC_CS](
                        uninitialized=True
                    )
                    var c_cache_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_CS),
                        MutAnyOrigin,
                    ](c_cache.unsafe_ptr())
                    var c_obs_t2 = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_IN),
                        MutAnyOrigin,
                    ](obs_arr.unsafe_ptr())
                    var c_p = cpu_state.critic.params_view()
                    Self.CriticNet.forward_with_cache[1](
                        c_obs_t2, val_t2, c_p, c_cache_t
                    )

                    # Critic gradient (MSE)
                    var ret = cpu_state._returns[idx]
                    var v_grad = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                        uninitialized=True
                    )
                    v_grad[0] = (
                        Scalar[dtype](2.0)
                        * Scalar[dtype](self.value_loss_coef)
                        * (val_arr2[0] - ret)
                    )
                    var v_grad_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_OUT),
                        MutAnyOrigin,
                    ](v_grad.unsafe_ptr())
                    var c_grad_in = InlineArray[Scalar[dtype], Self.CRITIC_IN](
                        uninitialized=True
                    )
                    var c_grad_in_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_IN),
                        MutAnyOrigin,
                    ](c_grad_in.unsafe_ptr())
                    var c_g = cpu_state.critic.grads_view()
                    cpu_state.critic.zero_grads()
                    Self.CriticNet.backward[1](
                        v_grad_t, c_grad_in_t, c_p, c_cache_t, c_g
                    )
                    cpu_state.critic.optimizer_step()

                    total_loss += policy_loss
                    n_updates += 1

        self.train_step_count += 1
        if n_updates > 0:
            return total_loss / Float64(n_updates)
        return 0.0

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        """Select deterministic action (mean of Gaussian policy)."""
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[dtype](obs[i])
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())

        var actor_out = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        var actor_out_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
        ](actor_out.unsafe_ptr())
        var p = cpu_state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, actor_out_t, p)

        # Return mean (deterministic policy)
        var result = List[Float64](capacity=Self.ACTIONS)
        for j in range(Self.ACTIONS):
            result.append(Float64(actor_out[j]))
        return result^

    fn get_explore_rate(self) -> Float64:
        return self.entropy_coef

    # =========================================================================
    # Checkpointable
    # =========================================================================

    fn save_checkpoint(self, path: String) raises -> None:
        from mojo_rl.nn.checkpoint import (
            write_checkpoint_header,
            write_metadata_section,
            save_checkpoint_file,
        )

        var content = write_checkpoint_header(
            "generic_onpolicy_continuous",
            Self.Config.ActorModel.PARAM_SIZE
            + Self.Config.CriticModel.PARAM_SIZE,
            0,
        )
        content += self.cpu_state.actor.write_sections("actor_")
        content += self.cpu_state.critic.write_sections("critic_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("gae_lambda=" + String(self.gae_lambda))
        metadata.append("entropy_coef=" + String(self.entropy_coef))
        metadata.append("clip_epsilon=" + String(self.clip_epsilon))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(path, content)

    fn load_checkpoint(mut self, path: String) raises -> None:
        from mojo_rl.nn.checkpoint import (
            read_checkpoint_file,
            parse_checkpoint_header,
            read_metadata_section,
            get_metadata_value,
        )

        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)
        self.cpu_state.actor.read_sections(content, "actor_")
        self.cpu_state.critic.read_sections(content, "critic_")

        var metadata = read_metadata_section(content)
        var gamma_str = get_metadata_value(metadata, "gamma")
        if len(gamma_str) > 0:
            self.gamma = atof(gamma_str)
        var gae_str = get_metadata_value(metadata, "gae_lambda")
        if len(gae_str) > 0:
            self.gae_lambda = atof(gae_str)
        var entropy_str = get_metadata_value(metadata, "entropy_coef")
        if len(entropy_str) > 0:
            self.entropy_coef = atof(entropy_str)
        var clip_str = get_metadata_value(metadata, "clip_epsilon")
        if len(clip_str) > 0:
            self.clip_epsilon = atof(clip_str)
        var step_str = get_metadata_value(metadata, "train_step_count")
        if len(step_str) > 0:
            self.train_step_count = Int(atol(step_str))

    # =========================================================================
    # CPU Training convenience
    # =========================================================================

    fn train[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_updates: Int = 1000,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train the continuous PPO agent.

        Args:
            env: Environment implementing BoxContinuousActionEnv.
            num_updates: Number of rollout-update cycles.
            verbose: Print progress (default: False).
            print_every: Print every N updates if verbose (default: 10).
            environment_name: Name for metrics labeling.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N updates (default: 0).

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        from mojo_rl.deep_agents.core.onpolicy_train import (
            run_onpolicy_continuous_train,
        )

        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = self.make_cpu_state()
        var ckpt_path = String(self.checkpoint_path)
        var algo_name = String("GenericPPOContinuous")
        var metrics = run_onpolicy_continuous_train[E, Self, Self.L](
            self,
            cpu_state,
            env,
            num_updates,
            checkpoint_every=self.checkpoint_every,
            checkpoint_path=ckpt_path,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
            logger=logger,
        )
        self.cpu_state = cpu_state^
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^

    # =========================================================================
    # Evaluation
    # =========================================================================

    fn evaluate[
        E: BoxContinuousActionEnv & RenderableEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        stochastic: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent with optional rendering and stochastic sampling.

        Args:
            env: Environment to evaluate on (must implement RenderableEnv).
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps_per_episode: Maximum steps per episode (default: 1000).
            verbose: Print per-episode results (default: False).
            stochastic: If True, sample from policy; if False, use mean (default: False).
            render: If True, render each frame (default: False).
            frame_delay_ms: Delay between frames in ms (default: 16 ~60fps).

        Returns:
            Average reward across episodes.
        """
        # Copy weights to avoid aliasing self + self.cpu_state
        var eval_state = self.make_cpu_state()
        eval_state.actor.copy_params_from(self.cpu_state.actor)
        eval_state.critic.copy_params_from(self.cpu_state.critic)

        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for episode in range(num_episodes):
            if quit_requested:
                break

            var obs_raw = env.reset_obs_list()
            var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                uninitialized=True
            )
            for i in range(Self.OBS):
                obs_arr[i] = Scalar[dtype](obs_raw[i])

            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for _ in range(max_steps_per_episode):
                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break
                    if (
                        env.renderer_is_paused()
                        and not env.renderer_step_once()
                    ):
                        continue

                var obs_t = LayoutTensor[
                    dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                ](obs_arr.unsafe_ptr())

                # Actor forward
                var actor_out = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
                    uninitialized=True
                )
                var actor_out_t = LayoutTensor[
                    dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
                ](actor_out.unsafe_ptr())
                var p = eval_state.actor.params_view()
                Self.ActorNet.forward[1](obs_t, actor_out_t, p)

                var action = List[Float64](capacity=Self.ACTIONS)
                if stochastic:
                    # Sample from Gaussian policy
                    for j in range(Self.ACTIONS):
                        var mean = Float64(actor_out[j])
                        var raw_ls = Float64(actor_out[Self.ACTIONS + j])
                        if raw_ls < LOG_STD_MIN:
                            raw_ls = LOG_STD_MIN
                        elif raw_ls > LOG_STD_MAX:
                            raw_ls = LOG_STD_MAX
                        var std = exp(raw_ls)
                        var u1 = random_float64()
                        var u2 = random_float64()
                        if u1 < 1e-10:
                            u1 = 1e-10
                        var noise = sqrt(-2.0 * log(u1)) * cos(
                            2.0 * 3.14159265358979 * u2
                        )
                        action.append(mean + std * noise)
                else:
                    # Deterministic: use mean
                    for j in range(Self.ACTIONS):
                        action.append(Float64(actor_out[j]))

                var result = env.step_continuous_vec(action)
                var reward = Float64(result[1])
                var done = result[2]

                episode_reward += reward
                episode_steps += 1

                if done:
                    var next_obs = env.reset_obs_list()
                    for i in range(Self.OBS):
                        obs_arr[i] = Scalar[dtype](next_obs[i])
                    break
                else:
                    for i in range(Self.OBS):
                        obs_arr[i] = Scalar[dtype](result[0][i])

            total_reward += episode_reward

            if verbose:
                print(
                    "Eval Episode",
                    episode + 1,
                    "| Reward:",
                    String(episode_reward)[:10],
                    "| Steps:",
                    episode_steps,
                )

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)

    fn evaluate_gpu[
        EnvType: GPUContinuousEnv
    ](
        self,
        ctx: DeviceContext,
        num_episodes: Int = 100,
        max_steps: Int = 1000,
        verbose: Bool = False,
        stochastic: Bool = True,
    ) raises -> Float64:
        """Evaluate the agent on GPU parallel environments.

        Uses unbounded Gaussian policy. Actions are clipped to environment
        bounds by the GPU environment kernel.

        Args:
            ctx: GPU device context.
            num_episodes: Target number of evaluation episodes (default: 100).
            max_steps: Maximum steps per episode (default: 1000).
            verbose: Whether to print progress (default: False).
            stochastic: If True, sample from policy; if False, use mean (default: True).

        Returns:
            Average reward over completed episodes.
        """
        comptime N_EVAL_ENVS = Self.n_envs
        comptime ENV_OBS_SIZE = N_EVAL_ENVS * Self.OBS
        comptime ENV_ACTION_SIZE = N_EVAL_ENVS * Self.ACTIONS
        comptime ENV_BLOCKS = (N_EVAL_ENVS + TPB - 1) // TPB

        # Environment state buffers
        var env_states_buf = ctx.enqueue_create_buffer[dtype](
            N_EVAL_ENVS * EnvType.STATE_SIZE
        )
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](N_EVAL_ENVS)
        var dones_buf = ctx.enqueue_create_buffer[dtype](N_EVAL_ENVS)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](N_EVAL_ENVS)

        # Action buffers
        var actions_buf = ctx.enqueue_create_buffer[dtype](ENV_ACTION_SIZE)
        var actor_out_buf = ctx.enqueue_create_buffer[dtype](
            N_EVAL_ENVS * Self.ACTOR_OUT
        )

        # Network parameter buffers (copy from CPU)
        var actor_params_buf = ctx.enqueue_create_buffer[dtype](
            Self.ActorModel.PARAM_SIZE
        )
        ctx.enqueue_copy(actor_params_buf, self.cpu_state.actor.params)

        # Workspace buffer for forward pass
        comptime WORKSPACE_PER_SAMPLE = Self.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
        var actor_workspace_buf = ctx.enqueue_create_buffer[dtype](
            N_EVAL_ENVS * WORKSPACE_PER_SAMPLE
        )

        # Tracking arrays (on CPU)
        var episode_rewards = List[Float64]()
        var current_rewards = InlineArray[Float64, N_EVAL_ENVS](fill=0.0)
        var episodes_completed = 0

        # Step workspace
        comptime EVAL_TOTAL_WS = EnvType.STEP_WS_SHARED + N_EVAL_ENVS * EnvType.STEP_WS_PER_ENV
        comptime EVAL_WS_ALLOC = EVAL_TOTAL_WS if EVAL_TOTAL_WS > 0 else 1
        var eval_ws_buf = ctx.enqueue_create_buffer[dtype](EVAL_WS_ALLOC)
        EnvType.init_step_workspace_gpu[N_EVAL_ENVS](ctx, eval_ws_buf)

        # Initialize environments
        EnvType.reset_kernel_gpu[N_EVAL_ENVS, EnvType.STATE_SIZE](
            ctx, env_states_buf
        )
        EnvType.extract_obs_kernel_gpu[N_EVAL_ENVS, EnvType.STATE_SIZE, Self.OBS](
            ctx, env_states_buf, obs_buf
        )
        ctx.synchronize()

        if verbose:
            print(
                "Running GPU evaluation with", N_EVAL_ENVS, "parallel envs..."
            )

        # Log probs buffer (needed for sampling kernel)
        var log_probs_buf = ctx.enqueue_create_buffer[dtype](N_EVAL_ENVS)

        # Deterministic action extraction kernel
        @always_inline
        fn extract_deterministic_actions(
            actions: LayoutTensor[
                dtype, Layout.row_major(N_EVAL_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            actor_out: LayoutTensor[
                dtype,
                Layout.row_major(N_EVAL_ENVS, Self.ACTOR_OUT),
                ImmutAnyOrigin,
            ],
        ):
            var idx = Int(block_idx.x) * TPB + Int(thread_idx.x)
            if idx >= N_EVAL_ENVS:
                return
            for j in range(Self.ACTIONS):
                actions[idx, j] = actor_out[idx, j]

        comptime sample_k = _sample_continuous_actions_kernel[
            dtype, N_EVAL_ENVS, Self.ACTIONS
        ]

        var step = 0
        while episodes_completed < num_episodes and step < max_steps:
            # Forward actor
            var eval_actor_out_t = LayoutTensor[
                dtype,
                Layout.row_major(N_EVAL_ENVS, Self.ActorModel.OUT_DIM),
                MutAnyOrigin,
            ](actor_out_buf.unsafe_ptr())
            var eval_obs_t = LayoutTensor[
                dtype,
                Layout.row_major(N_EVAL_ENVS, Self.ActorModel.IN_DIM),
                MutAnyOrigin,
            ](obs_buf.unsafe_ptr())
            var eval_params_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.ActorModel.PARAM_SIZE),
                MutAnyOrigin,
            ](actor_params_buf.unsafe_ptr())
            Self.ActorModel.forward_gpu_no_cache[N_EVAL_ENVS](
                ctx,
                eval_actor_out_t,
                eval_obs_t,
                eval_params_t,
                actor_workspace_buf,
            )

            var actions_t = LayoutTensor[
                dtype, Layout.row_major(N_EVAL_ENVS, Self.ACTIONS), MutAnyOrigin
            ](actions_buf.unsafe_ptr())
            var actor_out_t = LayoutTensor[
                dtype,
                Layout.row_major(N_EVAL_ENVS, Self.ACTOR_OUT),
                MutAnyOrigin,
            ](actor_out_buf.unsafe_ptr())

            if stochastic:
                var log_probs_t = LayoutTensor[
                    dtype, Layout.row_major(N_EVAL_ENVS), MutAnyOrigin
                ](log_probs_buf.unsafe_ptr())
                ctx.enqueue_function[sample_k, sample_k](
                    actor_out_t,
                    actions_t,
                    log_probs_t,
                    Scalar[DType.uint32](step * 2654435761),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
            else:
                var actor_out_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(N_EVAL_ENVS, Self.ACTOR_OUT),
                    ImmutAnyOrigin,
                ](actor_out_buf.unsafe_ptr())
                ctx.enqueue_function[
                    extract_deterministic_actions,
                    extract_deterministic_actions,
                ](
                    actions_t,
                    actor_out_immut,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

            # Step environments
            comptime if EVAL_TOTAL_WS > 0:
                EnvType.step_kernel_gpu[
                    N_EVAL_ENVS, EnvType.STATE_SIZE, Self.OBS, Self.ACTIONS
                ](
                    ctx,
                    env_states_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    obs_buf,
                    UInt64(step),
                    List[Scalar[dtype]](),
                    eval_ws_buf.unsafe_ptr(),
                )
            else:
                EnvType.step_kernel_gpu[
                    N_EVAL_ENVS, EnvType.STATE_SIZE, Self.OBS, Self.ACTIONS
                ](
                    ctx,
                    env_states_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    obs_buf,
                    UInt64(step),
                )
            ctx.synchronize()

            # Copy rewards and dones to CPU
            var rewards_host = InlineArray[Scalar[dtype], N_EVAL_ENVS](
                uninitialized=True
            )
            var dones_host = InlineArray[Scalar[dtype], N_EVAL_ENVS](
                uninitialized=True
            )
            ctx.enqueue_copy(rewards_host.unsafe_ptr(), rewards_buf)
            ctx.enqueue_copy(dones_host.unsafe_ptr(), dones_buf)
            ctx.synchronize()

            # Track rewards and episode completion
            for i in range(N_EVAL_ENVS):
                current_rewards[i] += Float64(rewards_host[i])
                if dones_host[i] > 0:
                    episode_rewards.append(current_rewards[i])
                    current_rewards[i] = 0.0
                    episodes_completed += 1
                    if episodes_completed >= num_episodes:
                        break

            # Auto-reset done environments
            EnvType.selective_reset_kernel_gpu[N_EVAL_ENVS, EnvType.STATE_SIZE](
                ctx,
                env_states_buf,
                dones_buf,
                UInt64(step),
                workspace_ptr=eval_ws_buf.unsafe_ptr(),
            )
            EnvType.extract_obs_kernel_gpu[
                N_EVAL_ENVS, EnvType.STATE_SIZE, Self.OBS
            ](ctx, env_states_buf, obs_buf)

            step += 1

        if len(episode_rewards) == 0:
            if verbose:
                print("Warning: No episodes completed!")
            return 0.0

        var total_reward: Float64 = 0.0
        for i in range(len(episode_rewards)):
            total_reward += episode_rewards[i]
        var avg = total_reward / Float64(len(episode_rewards))

        if verbose:
            print(
                "GPU Eval:",
                len(episode_rewards),
                "episodes | Avg reward:",
                String(avg)[:10],
            )

        return avg

    # =========================================================================
    # GPUOnPolicyContinuousAgent trait conformance
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.gpu_actor.upload_from(self.cpu_state.actor, ctx)
        gpu_state.gpu_critic.upload_from(self.cpu_state.critic, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.gpu_actor.download_to(self.cpu_state.actor, ctx)
        gpu_state.gpu_critic.download_to(self.cpu_state.critic, ctx)
        ctx.synchronize()

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
        """Forward actor + critic on GPU and sample continuous actions."""
        comptime blocks = (N_ENVS + TPB - 1) // TPB

        var actor_params_t = gpu_state.gpu_actor.params_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        # Actor forward → [N_ENVS, ACTOR_OUT] (mean + log_std)
        var actor_out_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.sampled_actions_buf.unsafe_ptr())
        Self.ActorModel.forward_gpu_no_cache[N_ENVS](
            ctx,
            actor_out_t,
            obs_t,
            actor_params_t,
            gpu_state.actor_env_workspace_buf,
        )

        # Critic forward → [N_ENVS, 1] values
        var values_t = LayoutTensor[
            dtype,
            Layout.row_major(N_ENVS, Self.CriticModel.OUT_DIM),
            MutAnyOrigin,
        ](values_buf.unsafe_ptr())
        var c_obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.CRITIC_IN), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        Self.CriticModel.forward_gpu_no_cache[N_ENVS](
            ctx,
            values_t,
            c_obs_t,
            critic_params_t,
            gpu_state.critic_env_workspace_buf,
        )

        # Sample continuous actions from Gaussian
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var log_probs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())

        comptime sample_k = _sample_continuous_actions_kernel[
            dtype, N_ENVS, Self.ACTIONS
        ]
        ctx.enqueue_function[sample_k, sample_k](
            actor_out_t,
            actions_t,
            log_probs_t,
            Scalar[DType.uint32](rng_seed),
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

    fn compute_advantages_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        final_obs_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Compute GAE advantages from collected rollout (CPU-side)."""
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE

        var critic_params_t = gpu_state.gpu_critic.params_view()
        var final_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.CRITIC_IN), MutAnyOrigin
        ](final_obs_buf.unsafe_ptr())
        var bootstrap_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.CriticModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu_state.values_env_buf.unsafe_ptr())

        # Forward critic on final obs for bootstrap values
        Self.CriticModel.forward_gpu_no_cache[Self.n_envs](
            ctx,
            bootstrap_t,
            final_obs_t,
            critic_params_t,
            gpu_state.critic_env_workspace_buf,
        )

        # Copy rollout data to host for GAE
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

        # GAE computation per environment
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

        # Normalize advantages globally
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

        # Upload to GPU
        ctx.enqueue_copy(gpu_state.advantages_buf, gpu_state.advantages_host)
        ctx.enqueue_copy(gpu_state.returns_buf, gpu_state.returns_host)
        ctx.synchronize()

    fn update_epochs_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        update_idx: Int,
    ) raises -> None:
        """Multi-epoch minibatch PPO update on GPU (continuous actions)."""
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ACTOR_PARAMS = Self.ActorModel.PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CriticModel.PARAM_SIZE
        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB

        # Param views
        var actor_params_t = gpu_state.gpu_actor.params_view()
        var actor_grads_t = gpu_state.gpu_actor.grads_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()
        var critic_grads_t = gpu_state.gpu_critic.grads_view()

        # Minibatch LayoutTensor views
        var mb_obs_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.mb_obs_buf.unsafe_ptr())
        var mb_actions_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTIONS), MutAnyOrigin
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

        # Rollout LayoutTensor views
        var rollout_obs_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL, Self.OBS), MutAnyOrigin
        ](gpu_state.rollout_obs_buf.unsafe_ptr())
        var rollout_actions_t = LayoutTensor[
            dtype,
            Layout.row_major(ROLLOUT_TOTAL, Self.ACTIONS),
            MutAnyOrigin,
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

        # Training workspace views
        var actor_logits_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.actor_logits_buf.unsafe_ptr())
        var actor_grad_output_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTOR_OUT), MutAnyOrigin
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
            dtype, Layout.row_major(MINIBATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.critic_values_buf.unsafe_ptr())
        var critic_grad_output_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.CRITIC_OUT), MutAnyOrigin
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
        var diag_entropy_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.diag_entropy_buf.unsafe_ptr())
        var diag_clip_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.diag_clip_buf.unsafe_ptr())
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

        # Kernel wrappers
        comptime gather_k = ppo_continuous_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, Self.ACTIONS, ROLLOUT_TOTAL
        ]
        comptime actor_grad_k = ppo_continuous_actor_grad_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime critic_grad_k = ppo_critic_grad_kernel[dtype, MINIBATCH]
        comptime critic_grad_clipped_k = ppo_critic_grad_clipped_kernel[
            dtype, MINIBATCH
        ]
        comptime normalize_adv_k = normalize_advantages_kernel[dtype, MINIBATCH]
        comptime actor_grad_norm_k = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_k = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime actor_reduce_scale_k = gradient_reduce_and_compute_scale_kernel[
            dtype, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime actor_apply_scale_k = gradient_apply_scale_kernel[
            dtype, ACTOR_PARAMS
        ]
        comptime critic_reduce_scale_k = gradient_reduce_and_compute_scale_kernel[
            dtype, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime critic_apply_scale_k = gradient_apply_scale_kernel[
            dtype, CRITIC_PARAMS
        ]

        var kl_early_stop = False
        var n_epochs = Self.Config.EpochSched.get_num_epochs(self.num_epochs)
        var num_minibatches = ROLLOUT_TOTAL // MINIBATCH

        for epoch in range(n_epochs):
            if kl_early_stop:
                break

            # Fisher-Yates shuffle on CPU
            var indices_list = List[Int]()
            for i in range(ROLLOUT_TOTAL):
                indices_list.append(i)
            comptime if Self.Config.EpochSched.USES_SHUFFLE:
                for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                    var j = Int(random_float64() * Float64(i + 1))
                    var temp = indices_list[i]
                    indices_list[i] = indices_list[j]
                    indices_list[j] = temp

            for mb_idx in range(num_minibatches):
                if kl_early_stop:
                    break
                var start_idx = mb_idx * MINIBATCH

                for i in range(MINIBATCH):
                    gpu_state.mb_indices_host[i] = Int32(
                        indices_list[start_idx + i]
                    )
                ctx.enqueue_copy(
                    gpu_state.mb_indices_buf, gpu_state.mb_indices_host
                )

                # Parallel obs gather: 2D grid (OBS_BLOCKS, MINIBATCH)
                comptime gather_obs_k = ppo_gather_minibatch_obs_parallel_kernel[
                    dtype, MINIBATCH, Self.OBS, ROLLOUT_TOTAL
                ]
                comptime GATHER_OBS_BLOCKS = (Self.OBS + TPB - 1) // TPB
                ctx.enqueue_function[gather_obs_k, gather_obs_k](
                    mb_obs_t,
                    rollout_obs_t,
                    mb_indices_t,
                    MINIBATCH,
                    grid_dim=(GATHER_OBS_BLOCKS, MINIBATCH),
                    block_dim=(TPB,),
                )

                # Scalar+action gather (actions are small dim, scalars trivial)
                ctx.enqueue_function[gather_k, gather_k](
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
                comptime if Self.Config.EpochSched.SUPPORTS_MINIBATCH_NORM:
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
                            var diff = (
                                gpu_state.mb_advantages_host[i] - adv_mean
                            )
                            adv_var += diff * diff
                        var adv_std = sqrt(
                            adv_var / Scalar[dtype](MINIBATCH)
                            + Scalar[dtype](1e-8)
                        )
                        ctx.enqueue_function[normalize_adv_k, normalize_adv_k](
                            mb_advantages_t,
                            adv_mean,
                            adv_std,
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.synchronize()

                # ---- Train actor ----
                gpu_state.gpu_actor.zero_grads(ctx)
                Self.ActorModel.forward_gpu[MINIBATCH](
                    ctx,
                    actor_logits_t,
                    mb_obs_t,
                    actor_params_t,
                    actor_cache_t,
                    gpu_state.actor_mb_workspace_buf,
                )
                ctx.synchronize()

                # Continuous PPO actor gradient kernel
                ctx.enqueue_function[actor_grad_k, actor_grad_k](
                    actor_grad_output_t,
                    kl_divergences_t,
                    diag_entropy_t,
                    diag_clip_t,
                    actor_logits_t,
                    mb_old_log_probs_t,
                    mb_advantages_t,
                    mb_actions_t,
                    Scalar[dtype](self.clip_epsilon),
                    Scalar[dtype](self.entropy_coef),
                    MINIBATCH,
                    grid_dim=(MINIBATCH_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                # KL early stopping
                comptime if Self.Config.EpochSched.USES_KL_EARLY_STOP:
                    if self.target_kl > 0.0:
                        ctx.enqueue_copy(
                            gpu_state.kl_divergences_host,
                            gpu_state.kl_divergences_buf,
                        )
                        ctx.synchronize()
                        var kl_sum = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            kl_sum += gpu_state.kl_divergences_host[i]
                        if (
                            Float64(kl_sum) / Float64(MINIBATCH)
                            > self.target_kl
                        ):
                            kl_early_stop = True
                            break

                Self.ActorModel.backward_gpu[MINIBATCH](
                    ctx,
                    actor_grad_input_t,
                    actor_grad_output_t,
                    actor_params_t,
                    actor_cache_t,
                    actor_grads_t,
                    gpu_state.actor_mb_workspace_buf,
                )

                if self.max_grad_norm > 0.0:
                    ctx.enqueue_function[actor_grad_norm_k, actor_grad_norm_k](
                        actor_grad_partial_sums_t,
                        actor_grads_t,
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        actor_reduce_scale_k, actor_reduce_scale_k
                    ](
                        actor_scale_t,
                        actor_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        actor_apply_scale_k, actor_apply_scale_k
                    ](
                        actor_grads_t,
                        actor_scale_t,
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                gpu_state.gpu_actor.optimizer_step(ctx)

                # Clamp log_std params after optimizer step
                comptime LOG_STD_OFFSET = Self.ActorModel.PARAM_SIZE - Self.ACTIONS
                comptime clamp_k = clamp_log_std_params_kernel[
                    dtype, ACTOR_PARAMS, LOG_STD_OFFSET, Self.ACTIONS
                ]
                ctx.enqueue_function[clamp_k, clamp_k](
                    actor_params_t,
                    grid_dim=(1,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                # ---- Train critic ----
                var mb_c_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(MINIBATCH, Self.CRITIC_IN),
                    MutAnyOrigin,
                ](gpu_state.mb_obs_buf.unsafe_ptr())
                gpu_state.gpu_critic.zero_grads(ctx)
                Self.CriticModel.forward_gpu[MINIBATCH](
                    ctx,
                    critic_values_t,
                    mb_c_obs_t,
                    critic_params_t,
                    critic_cache_t,
                    gpu_state.critic_mb_workspace_buf,
                )
                ctx.synchronize()

                if self.clip_value:
                    ctx.enqueue_function[
                        critic_grad_clipped_k, critic_grad_clipped_k
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
                    ctx.enqueue_function[critic_grad_k, critic_grad_k](
                        critic_grad_output_t,
                        critic_values_t,
                        mb_returns_t,
                        Scalar[dtype](self.value_loss_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                ctx.synchronize()

                Self.CriticModel.backward_gpu[MINIBATCH](
                    ctx,
                    critic_grad_input_t,
                    critic_grad_output_t,
                    critic_params_t,
                    critic_cache_t,
                    critic_grads_t,
                    gpu_state.critic_mb_workspace_buf,
                )

                if self.max_grad_norm > 0.0:
                    ctx.enqueue_function[
                        critic_grad_norm_k, critic_grad_norm_k
                    ](
                        critic_grad_partial_sums_t,
                        critic_grads_t,
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        critic_reduce_scale_k, critic_reduce_scale_k
                    ](
                        critic_scale_t,
                        critic_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        critic_apply_scale_k, critic_apply_scale_k
                    ](
                        critic_grads_t,
                        critic_scale_t,
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                gpu_state.gpu_critic.optimizer_step(ctx)
                ctx.synchronize()

        gpu_state.rollout_step = 0
        self.train_step_count += 1

    # =========================================================================
    # GPU Training convenience
    # =========================================================================

    fn train_gpu[
        E: GPUContinuousEnv,
        CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
    ](
        mut self,
        ctx: DeviceContext,
        num_updates: Int,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train on GPU with parallel environments.

        Args:
            ctx: GPU device context.
            num_updates: Number of rollout-update cycles.
            verbose: Print progress (default: False).
            print_every: Print every N updates if verbose (default: 10).
            environment_name: Name for metrics labeling.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N updates (default: 0).

        Returns:
            TrainingMetrics with episode-level statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var timer = PerfTimer[False]()

        var metrics = run_onpolicy_continuous_train_gpu[
            E, Self, CurriculumType, 0, Self.L
        ](
            self,
            ctx,
            num_updates,
            timer,
            target_total_steps=self.target_total_steps,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=String("GenericPPOContinuous (GPU)"),
            logger=logger,
        )
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^
