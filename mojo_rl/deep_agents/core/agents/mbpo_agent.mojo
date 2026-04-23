"""MBPO (Model-Based Policy Optimization) agent.

Combines SAC policy learning with a probabilistic dynamics ensemble.
The dynamics model generates synthetic rollouts to augment the real
replay buffer, achieving ~10x better sample efficiency than SAC alone.

Key components:
- DynamicsEnsemble: N probabilistic networks predicting (reward, delta_obs)
- MBPOCPUState: Dual replay buffers (real + synthetic) + SAC networks + ensemble
- MBPOAgent: SAC training with mixed sampling + model training + rollouts
"""

from std.random import random_float64
from std.math import exp, log, sqrt
from layout import Layout, LayoutTensor
from std.memory import UnsafePointer, alloc, memset
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    read_metadata_section,
    save_checkpoint_file,
    read_checkpoint_file,
    set_metadata_value_float,
    set_metadata_value_int,
)
from mojo_rl.nn.model import Model, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import (
    Network,
    NetworkState,
    NetworkPair,
    GPUNetworkPair,
    GPUNetworkState,
)
from mojo_rl.nn.initializer import Kaiming, Xavier
from mojo_rl.nn.gpu.random import gaussian_noise
from mojo_rl.nn.model.stochastic_actor import rsample, get_deterministic_action

from mojo_rl.deep_agents.core import (
    OffPolicyState,
    OffPolicyContinuousAgent,
    Checkpointable,
)
from mojo_rl.deep_agents.core.workspace import OffPolicyTrainWS, ExplorationWS
from mojo_rl.deep_agents.core.critic_group import CriticGroup, GPUCriticGroup
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.configs.mbpo_config import MBPOConfig
from mojo_rl.deep_agents.core.agents.offpolicy_agent import GenericGPUState
from mojo_rl.deep_agents.core.kernels import (
    concat_obs_action_kernel,
    td_mse_grad_kernel,
    sac_sample_actions_kernel,
    gradient_norm_kernel,
    gradient_reduce_apply_fused_kernel,
    build_dynamics_target_kernel,
    gaussian_nll_grad_kernel,
    gaussian_nll_grad_learnable_kernel,
    reduce_bounds_grad_l2_adam_kernel,
    dynamics_sample_kernel,
    dynamics_sample_learnable_kernel,
    dynamics_sample_ensemble_learnable_kernel,
    sample_elite_assignment_kernel,
    clamp_rewards_kernel,
    sample_indices_kernel,
    increment_rng_counter_kernel,
    alpha_adam_update_kernel,
    reduce_mean_loss_kernel,
    mask_dead_rollouts_kernel,
    update_alive_mask_kernel,
    compute_scaler_mean_kernel,
    compute_scaler_std_kernel,
    normalize_input_kernel,
)
from mojo_rl.cuda.graph import CUDAGraph
from mojo_rl.core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
)
from mojo_rl.deep_agents.core.training.mbpo_train import (
    run_mbpo_train,
    run_mbpo_train_gpu,
)


# =============================================================================
# Helper: concat obs + act into critic input
# =============================================================================


def _concat_obs_act[
    BATCH: Int, OBS: Int, ACT: Int, CI: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    obs_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    act_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    for b in range(BATCH):
        for i in range(OBS):
            (out_ptr + b * CI + i)[] = (obs_ptr + b * OBS + i)[]
        for i in range(ACT):
            (out_ptr + b * CI + OBS + i)[] = (act_ptr + b * ACT + i)[]


# =============================================================================
# DynamicsEnsemble
# =============================================================================


struct DynamicsEnsemble[
    DynModel: Model,
    DynOpt: Optimizer,
    num_ensemble: Int,
    num_elites: Int,
    obs_dim: Int,
    action_dim: Int,
](Movable):
    """Ensemble of probabilistic dynamics models for MBPO.

    Each member predicts [mean(reward, delta_obs), logvar(reward, delta_obs)].
    Output dim = 2 * (1 + obs_dim).
    """

    comptime DYN_IN: Int = Self.obs_dim + Self.action_dim
    comptime DYN_PRED: Int = 1 + Self.obs_dim  # reward + delta_obs
    comptime DYN_OUT: Int = 2 * Self.DYN_PRED  # mean + logvar
    comptime DynNet = Network[Self.DynModel, Self.DynOpt]

    # Ensemble members: heap-allocated array of NetworkState
    var members: List[NetworkState[Self.DynModel, Self.DynOpt]]

    # Elite indices (top num_elites by holdout loss)
    var elite_indices: List[Int]

    # Max/min logvar bounds (trainable in reference, fixed here for simplicity)
    var max_logvar: Float64
    var min_logvar: Float64

    def __init__(out self):
        self.members = List[NetworkState[Self.DynModel, Self.DynOpt]](
            capacity=Self.num_ensemble
        )
        for _ in range(Self.num_ensemble):
            var ns = NetworkState[Self.DynModel, Self.DynOpt]()
            ns.initialize[Xavier[]]()
            # Reference BNN zeros biases (fc.py:137-141:
            # `tf.constant_initializer(0.0)`). Our AutoFused init applies the
            # same initializer to MatMul and BiasAdd; overwrite biases with 0.
            var dyn_p = ns.params_view()
            Self.DynModel.zero_biases[dtype](dyn_p)
            self.members.append(ns^)

        # Initially all members are elite
        self.elite_indices = List[Int](capacity=Self.num_elites)
        for i in range(Self.num_elites):
            self.elite_indices.append(i)

        # Diagnostic tightening: reference inits max=+0.5 but learns it down to
        # roughly [-1, -2] via L2 reg. We don't yet implement learnable bounds
        # + L2 reg, so approximate the converged value with a tighter fixed
        # start. max=-2 → max std ≈ 0.37 in raw delta_obs space (reasonable
        # for HalfCheetah where per-step deltas are small). 0.5 → std ≈ 1.28
        # was too loose and let synthetic rollouts drift far out of dist.
        self.max_logvar = -2.0
        self.min_logvar = -10.0

    def __init__(out self, *, deinit take: Self):
        self.members = take.members^
        self.elite_indices = take.elite_indices^
        self.max_logvar = take.max_logvar
        self.min_logvar = take.min_logvar

    def predict_single(
        self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        elite_idx: Int,
        mut out_next_obs: List[Scalar[dtype]],
        mut out_reward: List[Scalar[dtype]],
    ):
        """Predict (next_obs, reward) using one elite model.

        Samples from the predicted Gaussian and applies residual prediction.
        Results written to out_next_obs and out_reward (single-element list).
        """
        var member_idx = self.elite_indices[elite_idx]

        # Build input: [obs, action]
        var input_arr = InlineArray[Scalar[dtype], Self.DynModel.IN_DIM](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            input_arr[i] = obs[i]
        for i in range(Self.action_dim):
            input_arr[Self.obs_dim + i] = action[i]

        # Forward pass — use DynModel.IN_DIM / OUT_DIM for LayoutTensor types
        var in_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.DynModel.IN_DIM), MutAnyOrigin
        ](input_arr.unsafe_ptr())
        var out_arr = InlineArray[Scalar[dtype], Self.DynModel.OUT_DIM](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.DynModel.OUT_DIM), MutAnyOrigin
        ](out_arr.unsafe_ptr())
        var p = self.members[member_idx].params_view()
        Self.DynNet.forward[1](in_t, out_t, p)

        # Extract mean and logvar, clamp logvar
        var pred_mean = List[Scalar[dtype]](capacity=Self.DYN_PRED)
        var pred_logvar = List[Scalar[dtype]](capacity=Self.DYN_PRED)
        for i in range(Self.DYN_PRED):
            pred_mean.append(out_arr[i])
            var lv = Float64(out_arr[Self.DYN_PRED + i])
            # Clamp logvar to [min_logvar, max_logvar]
            if lv > self.max_logvar:
                lv = self.max_logvar
            elif lv < self.min_logvar:
                lv = self.min_logvar
            pred_logvar.append(Scalar[dtype](lv))

        # Sample from Gaussian: sample = mean + std * noise
        var reward = Float64(pred_mean[0])
        var noise_r = gaussian_noise()
        var std_r = sqrt(exp(Float64(pred_logvar[0])))
        reward += std_r * noise_r
        out_reward.clear()
        out_reward.append(Scalar[dtype](reward))

        out_next_obs.clear()
        for i in range(Self.obs_dim):
            var delta = Float64(pred_mean[1 + i])
            var noise = gaussian_noise()
            var std = sqrt(exp(Float64(pred_logvar[1 + i])))
            delta += std * noise
            # Residual prediction: next_obs = obs + delta
            out_next_obs.append(Scalar[dtype](Float64(obs[i]) + delta))

    def train_model[
        buffer_capacity: Int,
    ](
        mut self,
        buffer: HeapReplayBuffer[
            buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ],
        holdout_ratio: Float64 = 0.2,
        max_epochs: Int = 100,
        batch_size: Int = 256,
        max_epochs_since_update: Int = 5,
    ):
        """Train all ensemble members on real buffer data with early stopping.

        Uses Gaussian NLL loss: 0.5 * (y-mu)^2/var + 0.5 * log(var)
        Selects top num_elites by holdout loss.
        """
        var n_data = buffer.size
        if n_data < batch_size:
            return

        var n_holdout = Int(Float64(n_data) * holdout_ratio)
        if n_holdout < 1:
            n_holdout = 1
        var n_train = n_data - n_holdout

        # Create shuffled indices
        var indices = List[Int](capacity=n_data)
        for i in range(n_data):
            indices.append(i)
        # Fisher-Yates shuffle
        for i in range(n_data - 1, 0, step=-1):
            var j = Int(random_float64() * Float64(i + 1)) % (i + 1)
            var tmp = indices[i]
            indices[i] = indices[j]
            indices[j] = tmp

        # Train each ensemble member
        var holdout_losses = List[Float64](capacity=Self.num_ensemble)
        for m in range(Self.num_ensemble):
            holdout_losses.append(
                self._train_member[buffer_capacity](
                    m,
                    buffer,
                    indices,
                    n_train,
                    n_holdout,
                    max_epochs,
                    batch_size,
                    max_epochs_since_update,
                )
            )

        # Select elites: sort by holdout loss, keep top num_elites
        var sorted_indices = List[Int](capacity=Self.num_ensemble)
        for i in range(Self.num_ensemble):
            sorted_indices.append(i)
        # Simple selection sort
        for i in range(Self.num_ensemble):
            var min_idx = i
            for j in range(i + 1, Self.num_ensemble):
                if (
                    holdout_losses[sorted_indices[j]]
                    < holdout_losses[sorted_indices[min_idx]]
                ):
                    min_idx = j
            var tmp = sorted_indices[i]
            sorted_indices[i] = sorted_indices[min_idx]
            sorted_indices[min_idx] = tmp

        self.elite_indices.clear()
        for i in range(Self.num_elites):
            self.elite_indices.append(sorted_indices[i])

    def _train_member[
        buffer_capacity: Int,
    ](
        mut self,
        member_idx: Int,
        buffer: HeapReplayBuffer[
            buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ],
        indices: List[Int],
        n_train: Int,
        n_holdout: Int,
        max_epochs: Int,
        batch_size: Int,
        max_epochs_since_update: Int,
    ) -> Float64:
        """Train a single ensemble member. Returns holdout loss."""
        var best_holdout = Float64(1e10)
        var epochs_since_update = 0

        for _ in range(max_epochs):
            # Mini-batch training
            var train_loss: Float64 = 0.0
            var n_batches = 0
            var idx = 0
            while idx + batch_size <= n_train:
                var batch_loss = self._train_batch[buffer_capacity](
                    member_idx, buffer, indices, idx, batch_size
                )
                train_loss += batch_loss
                n_batches += 1
                idx += batch_size

            # Compute holdout loss
            var holdout_loss = self._compute_holdout_loss[buffer_capacity](
                member_idx, buffer, indices, n_train, n_holdout
            )

            # Early stopping
            if holdout_loss < best_holdout:
                best_holdout = holdout_loss
                epochs_since_update = 0
            else:
                epochs_since_update += 1
                if epochs_since_update >= max_epochs_since_update:
                    break

        return best_holdout

    def _train_batch[
        buffer_capacity: Int,
    ](
        mut self,
        member_idx: Int,
        buffer: HeapReplayBuffer[
            buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ],
        indices: List[Int],
        start: Int,
        batch_size: Int,
    ) -> Float64:
        """Train on one mini-batch using Gaussian NLL loss.

        Forward pass gives [mean, logvar] for (reward, delta_obs).
        Loss = 0.5 * (target - mean)^2 / var + 0.5 * logvar
        Gradient w.r.t. mean: (mean - target) / var
        Gradient w.r.t. logvar: 0.5 * (1 - (target - mean)^2 / var)
        """
        # We'll process one sample at a time since batch_size is dynamic
        var total_loss: Float64 = 0.0
        self.members[member_idx].zero_grads()

        for b in range(batch_size):
            var data_idx = indices[start + b]

            # Build input: [obs, action]
            var input_arr = InlineArray[Scalar[dtype], Self.DynModel.IN_DIM](
                uninitialized=True
            )
            for i in range(Self.obs_dim):
                input_arr[i] = buffer.obs[data_idx * Self.obs_dim + i]
            for i in range(Self.action_dim):
                input_arr[Self.obs_dim + i] = buffer.actions[
                    data_idx * Self.action_dim + i
                ]

            # Build target: [reward, delta_obs]
            var target_arr = InlineArray[Scalar[dtype], Self.DYN_PRED](
                uninitialized=True
            )
            target_arr[0] = buffer.rewards[data_idx]
            for i in range(Self.obs_dim):
                target_arr[1 + i] = Scalar[dtype](
                    Float64(buffer.next_obs[data_idx * Self.obs_dim + i])
                    - Float64(buffer.obs[data_idx * Self.obs_dim + i])
                )

            # Forward with cache
            var in_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.DynModel.IN_DIM), MutAnyOrigin
            ](input_arr.unsafe_ptr())
            var out_arr = InlineArray[Scalar[dtype], Self.DynModel.OUT_DIM](
                uninitialized=True
            )
            var out_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.DynModel.OUT_DIM), MutAnyOrigin
            ](out_arr.unsafe_ptr())
            var cache_arr = InlineArray[
                Scalar[dtype], Self.DynModel.CACHE_SIZE
            ](uninitialized=True)
            var cache_t = LayoutTensor[
                dtype,
                Layout.row_major(1, Self.DynModel.CACHE_SIZE),
                MutAnyOrigin,
            ](cache_arr.unsafe_ptr())
            var p = self.members[member_idx].params_view()
            Self.DynNet.forward_with_cache[1](in_t, out_t, p, cache_t)

            # Compute Gaussian NLL loss and gradient w.r.t. output
            var grad_out_arr = InlineArray[
                Scalar[dtype], Self.DynModel.OUT_DIM
            ](uninitialized=True)
            var sample_loss: Float64 = 0.0
            for i in range(Self.DYN_PRED):
                var mean = Float64(out_arr[i])
                var logvar = Float64(out_arr[Self.DYN_PRED + i])
                # Clamp logvar
                if logvar > self.max_logvar:
                    logvar = self.max_logvar
                elif logvar < self.min_logvar:
                    logvar = self.min_logvar
                var target = Float64(target_arr[i])
                var var_val = exp(logvar)
                var diff = target - mean
                var diff_sq = diff * diff

                # Loss: 0.5 * diff^2 / var + 0.5 * logvar
                sample_loss += 0.5 * diff_sq / var_val + 0.5 * logvar

                # Gradient w.r.t. mean: (mean - target) / var / batch_size
                grad_out_arr[i] = Scalar[dtype](
                    (mean - target) / var_val / Float64(batch_size)
                )
                # Gradient w.r.t. logvar: 0.5 * (1 - diff^2/var) / batch_size
                grad_out_arr[Self.DYN_PRED + i] = Scalar[dtype](
                    0.5 * (1.0 - diff_sq / var_val) / Float64(batch_size)
                )

            total_loss += sample_loss / Float64(Self.DYN_PRED)

            # Backward pass (accumulate gradients)
            var grad_out_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.DynModel.OUT_DIM), MutAnyOrigin
            ](grad_out_arr.unsafe_ptr())
            var grad_in_arr = InlineArray[Scalar[dtype], Self.DynModel.IN_DIM](
                uninitialized=True
            )
            var grad_in_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.DynModel.IN_DIM), MutAnyOrigin
            ](grad_in_arr.unsafe_ptr())
            var g = self.members[member_idx].grads_view()
            Self.DynNet.backward[1](grad_out_t, grad_in_t, p, cache_t, g)

        # Optimizer step
        self.members[member_idx].optimizer_step()
        return total_loss / Float64(batch_size)

    def _compute_holdout_loss[
        buffer_capacity: Int,
    ](
        self,
        member_idx: Int,
        buffer: HeapReplayBuffer[
            buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ],
        indices: List[Int],
        start: Int,
        n_holdout: Int,
    ) -> Float64:
        """Compute mean Gaussian NLL on holdout set."""
        var total_loss: Float64 = 0.0
        for b in range(n_holdout):
            var data_idx = indices[start + b]

            var input_arr = InlineArray[Scalar[dtype], Self.DynModel.IN_DIM](
                uninitialized=True
            )
            for i in range(Self.obs_dim):
                input_arr[i] = buffer.obs[data_idx * Self.obs_dim + i]
            for i in range(Self.action_dim):
                input_arr[Self.obs_dim + i] = buffer.actions[
                    data_idx * Self.action_dim + i
                ]

            var target_arr = InlineArray[Scalar[dtype], Self.DYN_PRED](
                uninitialized=True
            )
            target_arr[0] = buffer.rewards[data_idx]
            for i in range(Self.obs_dim):
                target_arr[1 + i] = Scalar[dtype](
                    Float64(buffer.next_obs[data_idx * Self.obs_dim + i])
                    - Float64(buffer.obs[data_idx * Self.obs_dim + i])
                )

            var in_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.DynModel.IN_DIM), MutAnyOrigin
            ](input_arr.unsafe_ptr())
            var out_arr = InlineArray[Scalar[dtype], Self.DynModel.OUT_DIM](
                uninitialized=True
            )
            var out_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.DynModel.OUT_DIM), MutAnyOrigin
            ](out_arr.unsafe_ptr())
            var p = self.members[member_idx].params_view()
            Self.DynNet.forward[1](in_t, out_t, p)

            for i in range(Self.DYN_PRED):
                var mean = Float64(out_arr[i])
                var logvar = Float64(out_arr[Self.DYN_PRED + i])
                if logvar > self.max_logvar:
                    logvar = self.max_logvar
                elif logvar < self.min_logvar:
                    logvar = self.min_logvar
                var target = Float64(target_arr[i])
                var var_val = exp(logvar)
                var diff = target - mean
                total_loss += 0.5 * diff * diff / var_val + 0.5 * logvar

        return total_loss / Float64(n_holdout * Self.DYN_PRED)


# =============================================================================
# GPUDynamicsEnsemble
# =============================================================================


struct GPUDynamicsEnsemble[
    DynModel: Model,
    DynOpt: Optimizer,
    num_ensemble: Int,
    num_elites: Int,
    obs_dim: Int,
    action_dim: Int,
    train_batch: Int = 256,
    rollout_batch: Int = 400,
    # Adam learning rate for learnable logvar bounds. Reference MBPO shares
    # the dynamics-optimizer lr (bnn.py puts bounds inside the same minimize
    # call). Default matches DefaultMBPOConfig.model_lr.
    bounds_lr: Float64 = 0.001,
](Movable):
    """GPU-resident ensemble of probabilistic dynamics models.

    Each member predicts [mean(reward, delta_obs), logvar(reward, delta_obs)].
    Training, holdout evaluation, and rollout sampling all run on GPU.
    """

    comptime DYN_PRED: Int = 1 + Self.obs_dim
    comptime DYN_OUT: Int = 2 * Self.DYN_PRED
    comptime DYN_IN: Int = Self.DynModel.IN_DIM
    comptime DynNet = Network[Self.DynModel, Self.DynOpt]

    # GPU network states
    var members: List[GPUNetworkState[Self.DynModel, Self.DynOpt]]
    var elite_indices: List[Int]

    # Per-member, per-dim learnable logvar bounds (MBPO ref bnn.py:169-172, 192).
    # Layout: [num_ensemble * DYN_PRED]. Index (m, d) -> m * DYN_PRED + d.
    # Initialized to +0.5 / -10.0 (reference initial values); learned via
    # L2 regularization (0.01 * sum(max) - 0.01 * sum(min)) added to NLL loss.
    var max_lv_buf: DeviceBuffer[dtype]
    var min_lv_buf: DeviceBuffer[dtype]

    # Adam optimizer state for bounds (shared lr with DynOpt).
    var max_lv_m: DeviceBuffer[dtype]
    var max_lv_v: DeviceBuffer[dtype]
    var min_lv_m: DeviceBuffer[dtype]
    var min_lv_v: DeviceBuffer[dtype]

    # Per-batch per-dim scratch for bounds gradient contributions, reduced
    # across BATCH inside reduce_bounds_grad_l2_adam_kernel.
    var max_lv_grad_scratch: DeviceBuffer[dtype]  # [train_batch * DYN_PRED]
    var min_lv_grad_scratch: DeviceBuffer[dtype]  # [train_batch * DYN_PRED]

    # Pre-allocated GPU buffers for training (fixed train_batch size)
    var t_input: DeviceBuffer[dtype]  # [train_batch * DYN_IN]
    var t_output: DeviceBuffer[dtype]  # [train_batch * DYN_OUT]
    var t_cache: DeviceBuffer[dtype]  # [train_batch * DynModel.CACHE_SIZE]
    var t_target: DeviceBuffer[dtype]  # [train_batch * DYN_PRED]
    var t_grad_out: DeviceBuffer[dtype]  # [train_batch * DYN_OUT]
    var t_grad_in: DeviceBuffer[dtype]  # [train_batch * DYN_IN]
    var t_ws: DeviceBuffer[dtype]  # workspace
    var t_loss: DeviceBuffer[dtype]  # [train_batch]
    var t_loss_host: HostBuffer[dtype]  # [1] reduced mean loss for CPU readback
    var t_loss_scalar: DeviceBuffer[dtype]  # [1] GPU-side reduced mean loss

    # Pre-allocated GPU buffers for rollouts (fixed rollout_batch size)
    var r_obs: DeviceBuffer[dtype]  # [rollout_batch * obs_dim]
    var r_next_obs: DeviceBuffer[dtype]  # [rollout_batch * obs_dim]
    var r_actions: DeviceBuffer[dtype]  # [rollout_batch * action_dim]
    var r_rewards: DeviceBuffer[dtype]  # [rollout_batch]
    var r_dones: DeviceBuffer[dtype]  # [rollout_batch]
    var r_alive: DeviceBuffer[dtype]  # [rollout_batch] alive mask for multi-step
    var r_dyn_input: DeviceBuffer[dtype]  # [rollout_batch * DYN_IN]
    var r_dyn_output: DeviceBuffer[dtype]  # [rollout_batch * DYN_OUT]
    # Stacked outputs of all elite forward passes for one rollout step.
    # Layout: [num_elites, rollout_batch, DYN_OUT]. Each slot i gets filled
    # by forwarding elite member `elite_indices[i]` on r_dyn_input.
    # Enables per-sample random elite selection (matches fake_env.py:54).
    var r_dyn_output_all: DeviceBuffer[dtype]
    # Per-batch random elite-slot index in [0, num_elites). Filled by
    # sample_elite_assignment_kernel each rollout step.
    var r_elite_idx_per_sample: DeviceBuffer[DType.int32]
    # Map elite-slot → ensemble-member index. Used by the sample kernel to
    # look up per-member logvar bounds. Re-uploaded from self.elite_indices
    # after every train_on_buffer call.
    var elite_member_buf: DeviceBuffer[DType.int32]
    var elite_member_host: HostBuffer[DType.int32]
    # GPU-side RNG counter for rollout elite-slot assignment (independent
    # from training rng to avoid coupling rollout randomness with training).
    var r_elite_rng: DeviceBuffer[DType.uint32]
    var r_ws: DeviceBuffer[dtype]  # workspace for rollout forward

    # Scratch buffers for sampling from replay buffer
    var s_obs: DeviceBuffer[dtype]
    var s_act: DeviceBuffer[dtype]
    var s_rew: DeviceBuffer[dtype]
    var s_nobs: DeviceBuffer[dtype]
    var s_done: DeviceBuffer[dtype]
    var s_idx: DeviceBuffer[DType.int32]

    # Per-member GPU-side RNG counters for dynamics training (ensemble bootstrap).
    # Each member draws its own bootstrap sample from the replay buffer so
    # members train on different data subsets, matching MBPO reference
    # bnn.py:328-336. Initialized with distinct seeds [1..num_ensemble].
    var rng_counters: List[DeviceBuffer[DType.uint32]]

    # Input scaler state (MBPO TensorStandardScaler equivalent).
    # Per-dim mean/std over concatenated [obs, act] inputs. Re-fit at the
    # start of each train_on_buffer() call over the populated real buffer,
    # then applied on every dynamics forward (training + rollouts). Without
    # this, dims of very different scale make logvar bounds meaningless.
    var input_mean: DeviceBuffer[dtype]  # [DYN_IN]
    var input_std: DeviceBuffer[dtype]  # [DYN_IN]

    def __init__(out self, ctx: DeviceContext) raises:
        # Initialize ensemble members on GPU
        self.members = List[GPUNetworkState[Self.DynModel, Self.DynOpt]](
            capacity=Self.num_ensemble
        )
        for _ in range(Self.num_ensemble):
            self.members.append(
                GPUNetworkState[Self.DynModel, Self.DynOpt](ctx)
            )

        self.elite_indices = List[Int](capacity=Self.num_elites)
        for i in range(Self.num_elites):
            self.elite_indices.append(i)

        # Learnable per-member, per-dim logvar bounds (reference bnn.py:169-172).
        # Init: max=+0.5, min=-10. Bounds learn via Adam on L2-regularized NLL.
        comptime LV_TOTAL = Self.num_ensemble * Self.DYN_PRED
        self.max_lv_buf = ctx.enqueue_create_buffer[dtype](LV_TOTAL)
        self.min_lv_buf = ctx.enqueue_create_buffer[dtype](LV_TOTAL)
        self.max_lv_buf.enqueue_fill(Scalar[dtype](0.5))
        self.min_lv_buf.enqueue_fill(Scalar[dtype](-10.0))

        self.max_lv_m = ctx.enqueue_create_buffer[dtype](LV_TOTAL)
        self.max_lv_v = ctx.enqueue_create_buffer[dtype](LV_TOTAL)
        self.min_lv_m = ctx.enqueue_create_buffer[dtype](LV_TOTAL)
        self.min_lv_v = ctx.enqueue_create_buffer[dtype](LV_TOTAL)
        self.max_lv_m.enqueue_fill(Scalar[dtype](0.0))
        self.max_lv_v.enqueue_fill(Scalar[dtype](0.0))
        self.min_lv_m.enqueue_fill(Scalar[dtype](0.0))
        self.min_lv_v.enqueue_fill(Scalar[dtype](0.0))

        # Training buffers
        comptime TB = Self.train_batch
        comptime WS = TB * Self.DynModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_SIZE = WS if WS > 0 else 1
        self.t_input = ctx.enqueue_create_buffer[dtype](TB * Self.DYN_IN)
        self.t_output = ctx.enqueue_create_buffer[dtype](TB * Self.DYN_OUT)
        self.t_cache = ctx.enqueue_create_buffer[dtype](
            TB * Self.DynModel.CACHE_SIZE
        )
        self.t_target = ctx.enqueue_create_buffer[dtype](TB * Self.DYN_PRED)
        self.t_grad_out = ctx.enqueue_create_buffer[dtype](TB * Self.DYN_OUT)
        self.t_grad_in = ctx.enqueue_create_buffer[dtype](TB * Self.DYN_IN)
        self.t_ws = ctx.enqueue_create_buffer[dtype](WS_SIZE)
        self.t_loss = ctx.enqueue_create_buffer[dtype](TB)
        self.t_loss_host = ctx.enqueue_create_host_buffer[dtype](1)
        self.t_loss_scalar = ctx.enqueue_create_buffer[dtype](1)

        # Per-batch per-dim scratch for bounds grad contributions. Written once
        # by the NLL kernel (one thread per (b, d)), then reduced across BATCH
        # inside reduce_bounds_grad_l2_adam_kernel. No zeroing needed because
        # every slot is overwritten each batch.
        self.max_lv_grad_scratch = ctx.enqueue_create_buffer[dtype](
            TB * Self.DYN_PRED
        )
        self.min_lv_grad_scratch = ctx.enqueue_create_buffer[dtype](
            TB * Self.DYN_PRED
        )

        # Rollout buffers
        comptime RB = Self.rollout_batch
        comptime RWS = RB * Self.DynModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime RWS_SIZE = RWS if RWS > 0 else 1
        self.r_obs = ctx.enqueue_create_buffer[dtype](RB * Self.obs_dim)
        self.r_next_obs = ctx.enqueue_create_buffer[dtype](RB * Self.obs_dim)
        self.r_actions = ctx.enqueue_create_buffer[dtype](RB * Self.action_dim)
        self.r_rewards = ctx.enqueue_create_buffer[dtype](RB)
        self.r_dones = ctx.enqueue_create_buffer[dtype](RB)
        self.r_alive = ctx.enqueue_create_buffer[dtype](RB)
        self.r_dyn_input = ctx.enqueue_create_buffer[dtype](RB * Self.DYN_IN)
        self.r_dyn_output = ctx.enqueue_create_buffer[dtype](RB * Self.DYN_OUT)
        self.r_dyn_output_all = ctx.enqueue_create_buffer[dtype](
            Self.num_elites * RB * Self.DYN_OUT
        )
        self.r_elite_idx_per_sample = ctx.enqueue_create_buffer[DType.int32](
            RB
        )
        self.elite_member_buf = ctx.enqueue_create_buffer[DType.int32](
            Self.num_elites
        )
        self.elite_member_host = ctx.enqueue_create_host_buffer[DType.int32](
            Self.num_elites
        )
        # Default elite-slot → member mapping is [0, 1, ..., num_elites-1]
        # until train_on_buffer reshuffles it.
        for i in range(Self.num_elites):
            self.elite_member_host[i] = Int32(i)
        ctx.enqueue_copy(self.elite_member_buf, self.elite_member_host)
        self.r_elite_rng = ctx.enqueue_create_buffer[DType.uint32](1)
        self.r_elite_rng.enqueue_fill(UInt32(0xC0FFEE))
        self.r_ws = ctx.enqueue_create_buffer[dtype](RWS_SIZE)

        # Sampling scratch buffers (sized for max of train_batch and rollout_batch)
        comptime SB = TB if TB > RB else RB
        self.s_obs = ctx.enqueue_create_buffer[dtype](SB * Self.obs_dim)
        self.s_act = ctx.enqueue_create_buffer[dtype](SB * Self.action_dim)
        self.s_rew = ctx.enqueue_create_buffer[dtype](SB)
        self.s_nobs = ctx.enqueue_create_buffer[dtype](SB * Self.obs_dim)
        self.s_done = ctx.enqueue_create_buffer[dtype](SB)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](SB)

        # Per-member RNG counters seeded with distinct values so each ensemble
        # member samples a different bootstrap subset from the replay buffer.
        self.rng_counters = List[DeviceBuffer[DType.uint32]](
            capacity=Self.num_ensemble
        )
        for m in range(Self.num_ensemble):
            var rc = ctx.enqueue_create_buffer[DType.uint32](1)
            # Seed with (m + 1) * LARGE_PRIME so distinct members produce
            # distinct Philox streams (Philox seed=0 is fine but ensuring
            # large gaps between member seeds avoids stream overlap).
            rc.enqueue_fill(UInt32((m + 1) * 2654435761))
            self.rng_counters.append(rc^)

        # Scaler state: identity (mean=0, std=1) until first fit_scaler_gpu.
        self.input_mean = ctx.enqueue_create_buffer[dtype](Self.DYN_IN)
        self.input_std = ctx.enqueue_create_buffer[dtype](Self.DYN_IN)
        self.input_mean.enqueue_fill(Scalar[dtype](0.0))
        self.input_std.enqueue_fill(Scalar[dtype](1.0))

    def __init__(out self, *, deinit take: Self):
        self.members = take.members^
        self.elite_indices = take.elite_indices^
        self.max_lv_buf = take.max_lv_buf^
        self.min_lv_buf = take.min_lv_buf^
        self.max_lv_m = take.max_lv_m^
        self.max_lv_v = take.max_lv_v^
        self.min_lv_m = take.min_lv_m^
        self.min_lv_v = take.min_lv_v^
        self.max_lv_grad_scratch = take.max_lv_grad_scratch^
        self.min_lv_grad_scratch = take.min_lv_grad_scratch^
        self.t_input = take.t_input^
        self.t_output = take.t_output^
        self.t_cache = take.t_cache^
        self.t_target = take.t_target^
        self.t_grad_out = take.t_grad_out^
        self.t_grad_in = take.t_grad_in^
        self.t_ws = take.t_ws^
        self.t_loss = take.t_loss^
        self.t_loss_host = take.t_loss_host^
        self.t_loss_scalar = take.t_loss_scalar^
        self.r_obs = take.r_obs^
        self.r_next_obs = take.r_next_obs^
        self.r_actions = take.r_actions^
        self.r_rewards = take.r_rewards^
        self.r_dones = take.r_dones^
        self.r_alive = take.r_alive^
        self.r_dyn_input = take.r_dyn_input^
        self.r_dyn_output = take.r_dyn_output^
        self.r_dyn_output_all = take.r_dyn_output_all^
        self.r_elite_idx_per_sample = take.r_elite_idx_per_sample^
        self.elite_member_buf = take.elite_member_buf^
        self.elite_member_host = take.elite_member_host^
        self.r_elite_rng = take.r_elite_rng^
        self.r_ws = take.r_ws^
        self.s_obs = take.s_obs^
        self.s_act = take.s_act^
        self.s_rew = take.s_rew^
        self.s_nobs = take.s_nobs^
        self.s_done = take.s_done^
        self.s_idx = take.s_idx^
        self.rng_counters = take.rng_counters^
        self.input_mean = take.input_mean^
        self.input_std = take.input_std^

    def upload_from(
        mut self,
        cpu_ensemble: DynamicsEnsemble[
            Self.DynModel,
            Self.DynOpt,
            Self.num_ensemble,
            Self.num_elites,
            Self.obs_dim,
            Self.action_dim,
        ],
        ctx: DeviceContext,
    ) raises:
        """Upload CPU ensemble weights to GPU."""
        for i in range(Self.num_ensemble):
            self.members[i].upload_from(cpu_ensemble.members[i], ctx)
        self.elite_indices.clear()
        for i in range(len(cpu_ensemble.elite_indices)):
            self.elite_indices.append(cpu_ensemble.elite_indices[i])
        self.sync_elite_member_buf(ctx)

    def fit_scaler_gpu[
        BUF_CAP: Int,
    ](
        mut self,
        ctx: DeviceContext,
        buffer: GPUReplayBuffer[BUF_CAP, Self.obs_dim, Self.action_dim],
    ) raises:
        """Fit per-dim mean/std of [obs || act] over the populated buffer.

        Matches `TensorStandardScaler.fit` in the MBPO reference
        (bnn.py:335, utils.py:48-50). Called once per train_on_buffer so
        every gradient step this round uses the same scaler. Cheap: only
        runs every `model_train_freq` env steps, and serial reductions are
        fast at realistic buffer sizes (~1M elements per dim).
        """
        var n = buffer.size
        if n < 1:
            return

        var mean_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.obs_dim), MutAnyOrigin
        ](self.input_mean.unsafe_ptr())
        var mean_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), MutAnyOrigin
        ](self.input_mean.unsafe_ptr() + Self.obs_dim)
        var std_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.obs_dim), MutAnyOrigin
        ](self.input_std.unsafe_ptr())
        var std_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), MutAnyOrigin
        ](self.input_std.unsafe_ptr() + Self.obs_dim)
        var obs_data_t = LayoutTensor[
            dtype, Layout.row_major(BUF_CAP, Self.obs_dim), MutAnyOrigin
        ](buffer.states_buf.unsafe_ptr())
        var act_data_t = LayoutTensor[
            dtype, Layout.row_major(BUF_CAP, Self.action_dim), MutAnyOrigin
        ](buffer.actions_buf.unsafe_ptr())

        # Pass 1: means
        comptime obs_mean_k = compute_scaler_mean_kernel[
            dtype, BUF_CAP, Self.obs_dim
        ]
        comptime act_mean_k = compute_scaler_mean_kernel[
            dtype, BUF_CAP, Self.action_dim
        ]
        ctx.enqueue_function[obs_mean_k, obs_mean_k](
            mean_obs_t, obs_data_t, n,
            grid_dim=(Self.obs_dim,), block_dim=(1,),
        )
        ctx.enqueue_function[act_mean_k, act_mean_k](
            mean_act_t, act_data_t, n,
            grid_dim=(Self.action_dim,), block_dim=(1,),
        )

        # Pass 2: stds (need means first, so these enqueue after by stream order)
        var min_std = Scalar[dtype](1e-12)
        comptime obs_std_k = compute_scaler_std_kernel[
            dtype, BUF_CAP, Self.obs_dim
        ]
        comptime act_std_k = compute_scaler_std_kernel[
            dtype, BUF_CAP, Self.action_dim
        ]
        ctx.enqueue_function[obs_std_k, obs_std_k](
            std_obs_t, obs_data_t, mean_obs_t, n, min_std,
            grid_dim=(Self.obs_dim,), block_dim=(1,),
        )
        ctx.enqueue_function[act_std_k, act_std_k](
            std_act_t, act_data_t, mean_act_t, n, min_std,
            grid_dim=(Self.action_dim,), block_dim=(1,),
        )

    def _enqueue_train_batch[
        BUF_CAP: Int,
    ](
        mut self,
        ctx: DeviceContext,
        buffer: GPUReplayBuffer[BUF_CAP, Self.obs_dim, Self.action_dim],
        m: Int,
    ) raises:
        """Enqueue one dynamics training batch for model m (pure GPU, capturable).

        Sequence: incr_rng → sample → concat → target → forward → NLL_grad
                  → zero_grads → backward → optimizer_step
                  → reduce_bounds_grad + L2 + Adam step on per-dim bounds
        """
        comptime TB = Self.train_batch
        comptime TPB_VAL = 256
        comptime PRED_BLOCKS = (TB * Self.DYN_PRED + TPB_VAL - 1) // TPB_VAL
        comptime DYN_IN_BLOCKS = (TB * Self.DYN_IN + TPB_VAL - 1) // TPB_VAL
        comptime incr_k = increment_rng_counter_kernel
        comptime cat_k = concat_obs_action_kernel[
            dtype, TB, Self.obs_dim, Self.action_dim, Self.DYN_IN
        ]
        comptime tgt_k = build_dynamics_target_kernel[
            dtype, TB, Self.obs_dim, Self.DYN_PRED
        ]
        comptime nll_k = gaussian_nll_grad_learnable_kernel[
            dtype, TB, Self.DYN_PRED, Self.DYN_OUT
        ]
        comptime bounds_k = reduce_bounds_grad_l2_adam_kernel[
            dtype, TB, Self.DYN_PRED
        ]

        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](self.rng_counters[m].unsafe_ptr())
        ctx.enqueue_function[incr_k, incr_k](
            rng_t, grid_dim=(1,), block_dim=(1,),
        )
        buffer.sample[TB](
            ctx, self.rng_counters[m],
            self.s_obs, self.s_act, self.s_rew,
            self.s_nobs, self.s_done, self.s_idx,
        )
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.obs_dim), MutAnyOrigin
        ](self.s_obs.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.action_dim), MutAnyOrigin
        ](self.s_act.unsafe_ptr())
        var input_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.DynModel.IN_DIM), MutAnyOrigin
        ](self.t_input.unsafe_ptr())
        ctx.enqueue_function[cat_k, cat_k](
            input_t, obs_t, act_t,
            grid_dim=(DYN_IN_BLOCKS,), block_dim=(TPB_VAL,),
        )
        # Normalize concatenated input in-place using fitted scaler.
        var mean_full_t = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_IN), MutAnyOrigin
        ](self.input_mean.unsafe_ptr())
        var std_full_t = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_IN), MutAnyOrigin
        ](self.input_std.unsafe_ptr())
        comptime norm_k = normalize_input_kernel[dtype, TB, Self.DYN_IN]
        ctx.enqueue_function[norm_k, norm_k](
            input_t, mean_full_t, std_full_t,
            grid_dim=(DYN_IN_BLOCKS,), block_dim=(TPB_VAL,),
        )
        var nobs_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.obs_dim), MutAnyOrigin
        ](self.s_nobs.unsafe_ptr())
        var rew_t = LayoutTensor[
            dtype, Layout.row_major(TB), MutAnyOrigin
        ](self.s_rew.unsafe_ptr())
        var target_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.DYN_PRED), MutAnyOrigin
        ](self.t_target.unsafe_ptr())
        ctx.enqueue_function[tgt_k, tgt_k](
            target_t, obs_t, nobs_t, rew_t,
            grid_dim=(PRED_BLOCKS,), block_dim=(TPB_VAL,),
        )
        var output_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.DynModel.OUT_DIM), MutAnyOrigin
        ](self.t_output.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.DynModel.CACHE_SIZE), MutAnyOrigin
        ](self.t_cache.unsafe_ptr())
        var p = self.members[m].params_view()
        Self.DynNet.forward_gpu_with_cache[TB](
            ctx, input_t, output_t, p, cache_t, self.t_ws,
        )
        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.DynModel.OUT_DIM), MutAnyOrigin
        ](self.t_grad_out.unsafe_ptr())
        var loss_t = LayoutTensor[
            dtype, Layout.row_major(TB), MutAnyOrigin
        ](self.t_loss.unsafe_ptr())
        # Per-member bounds views (offset pointer into [num_ensemble * DYN_PRED]).
        var max_lv_t = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_PRED), MutAnyOrigin
        ](self.max_lv_buf.unsafe_ptr() + m * Self.DYN_PRED)
        var min_lv_t = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_PRED), MutAnyOrigin
        ](self.min_lv_buf.unsafe_ptr() + m * Self.DYN_PRED)
        var max_grad_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.DYN_PRED), MutAnyOrigin
        ](self.max_lv_grad_scratch.unsafe_ptr())
        var min_grad_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.DYN_PRED), MutAnyOrigin
        ](self.min_lv_grad_scratch.unsafe_ptr())
        ctx.enqueue_function[nll_k, nll_k](
            grad_out_t,
            output_t,
            target_t,
            loss_t,
            max_lv_t,
            min_lv_t,
            max_grad_t,
            min_grad_t,
            grid_dim=(PRED_BLOCKS,),
            block_dim=(TPB_VAL,),
        )
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(TB, Self.DynModel.IN_DIM), MutAnyOrigin
        ](self.t_grad_in.unsafe_ptr())
        var g = self.members[m].grads_view()
        self.members[m].zero_grads(ctx)
        Self.DynNet.backward_gpu[TB](
            ctx, grad_out_t, grad_in_t, p, cache_t, g, self.t_ws,
        )
        self.members[m].optimizer_step(ctx)

        # Reduce per-batch bounds grads, add L2 contribution, Adam step on
        # per-dim bounds. Reference (bnn.py:192): L2 coef = 0.01 on both
        # sum(max) and -sum(min). Shares Adam hyperparams with DynOpt but
        # tracks its own moments. step_num tracks the dynamics network step.
        var max_m_t = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_PRED), MutAnyOrigin
        ](self.max_lv_m.unsafe_ptr() + m * Self.DYN_PRED)
        var max_v_t = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_PRED), MutAnyOrigin
        ](self.max_lv_v.unsafe_ptr() + m * Self.DYN_PRED)
        var min_m_t = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_PRED), MutAnyOrigin
        ](self.min_lv_m.unsafe_ptr() + m * Self.DYN_PRED)
        var min_v_t = LayoutTensor[
            dtype, Layout.row_major(Self.DYN_PRED), MutAnyOrigin
        ](self.min_lv_v.unsafe_ptr() + m * Self.DYN_PRED)

        # Adam hyperparams — mirror Adam defaults, reuse DynOpt's lr so bounds
        # and weights update at the same step size. Bias correction uses the
        # dynamics network's step counter (already incremented above).
        var lv_lr = Scalar[dtype](Self.bounds_lr)
        var lv_beta1 = Scalar[dtype](0.9)
        var lv_beta2 = Scalar[dtype](0.999)
        var lv_eps = Scalar[dtype](1e-8)
        var step_n = self.members[m].step_num
        var bc1 = Scalar[dtype](1.0 - 0.9 ** step_n)
        var bc2 = Scalar[dtype](1.0 - 0.999 ** step_n)
        var l2_coef = Scalar[dtype](0.01)

        ctx.enqueue_function[bounds_k, bounds_k](
            max_lv_t,
            min_lv_t,
            max_m_t,
            max_v_t,
            min_m_t,
            min_v_t,
            max_grad_t,
            min_grad_t,
            l2_coef,
            lv_lr,
            lv_beta1,
            lv_beta2,
            lv_eps,
            bc1,
            bc2,
            grid_dim=(Self.DYN_PRED,),
            block_dim=(1,),
        )

    def sync_elite_member_buf(mut self, ctx: DeviceContext) raises:
        """Copy self.elite_indices (CPU List) to elite_member_buf (GPU buffer).

        Called after train_on_buffer re-ranks elites. The rollout sample kernel
        reads elite_member_buf to map elite-slot → ensemble-member-index for
        per-member logvar bounds lookup.
        """
        for i in range(Self.num_elites):
            self.elite_member_host[i] = Int32(self.elite_indices[i])
        ctx.enqueue_copy(self.elite_member_buf, self.elite_member_host)

    def train_on_buffer[
        BUF_CAP: Int,
    ](
        mut self,
        ctx: DeviceContext,
        buffer: GPUReplayBuffer[BUF_CAP, Self.obs_dim, Self.action_dim],
        max_epochs: Int = 100,
        max_epochs_since_update: Int = 5,
        holdout_check_every: Int = 5,
    ) raises -> Float64:
        """Train all ensemble members on data from GPU replay buffer.

        For each member:
        1. Sample mini-batches from GPU buffer
        2. Build targets [reward, delta_obs] on GPU
        3. forward_gpu_with_cache → [mean, logvar]
        4. gaussian_nll_grad_kernel → grad_output
        5. backward_gpu → accumulate gradients
        6. optimizer_step on GPU
        7. Periodically: holdout loss via GPU reduction + single scalar download
        8. Early stopping based on holdout improvement
        """
        comptime TB = Self.train_batch
        comptime TPB_VAL = 256
        comptime PRED_BLOCKS = (TB * Self.DYN_PRED + TPB_VAL - 1) // TPB_VAL
        comptime BATCH_BLOCKS = (TB + TPB_VAL - 1) // TPB_VAL
        comptime DYN_IN_BLOCKS = (TB * Self.DYN_IN + TPB_VAL - 1) // TPB_VAL

        var n_data = buffer.size
        if n_data < TB:
            return 0.0

        var n_batches_per_epoch = n_data // TB
        if n_batches_per_epoch < 1:
            n_batches_per_epoch = 1

        # Re-fit input scaler over the populated real buffer before training
        # this round (matches MBPO reference: bnn.py:335). Subsequent forward
        # passes (training + rollouts) use these mean/std until next call.
        self.fit_scaler_gpu[BUF_CAP](ctx, buffer)

        # Kernel aliases
        comptime target_k = build_dynamics_target_kernel[
            dtype, TB, Self.obs_dim, Self.DYN_PRED
        ]
        # Holdout uses the learnable NLL kernel so it reads the current
        # per-dim bounds; grad writes to scratch are ignored (we only
        # consume loss_per_sample).
        comptime nll_k = gaussian_nll_grad_learnable_kernel[
            dtype, TB, Self.DYN_PRED, Self.DYN_OUT
        ]
        comptime concat_k = concat_obs_action_kernel[
            dtype, TB, Self.obs_dim, Self.action_dim, Self.DYN_IN
        ]

        var holdout_losses = List[Float64](capacity=Self.num_ensemble)

        # Per-member RNG counter for dynamics training sampling (bootstrap).
        comptime dyn_incr_k = increment_rng_counter_kernel

        for m in range(Self.num_ensemble):
            var dyn_rng_t = LayoutTensor[
                DType.uint32, Layout.row_major(1), MutAnyOrigin
            ](self.rng_counters[m].unsafe_ptr())
            var best_holdout = Float64(1e10)
            var epochs_since_update = 0

            # CUDA graph for this model's training batch (re-captured per model)
            var _dyn_graph: Optional[CUDAGraph] = None

            for epoch in range(max_epochs):
                # Training mini-batches via graph replay
                if not _dyn_graph:
                    # First epoch: warm-up + capture
                    self._enqueue_train_batch[BUF_CAP](
                        ctx, buffer, m
                    )
                    ctx.synchronize()
                    var graph = CUDAGraph(ctx)
                    graph.begin_capture()
                    self._enqueue_train_batch[BUF_CAP](
                        ctx, buffer, m
                    )
                    graph.end_capture()
                    _dyn_graph = graph^
                    # First batch ran in warm-up, replay remaining
                    for _ in range(n_batches_per_epoch - 1):
                        _dyn_graph.value().replay_async()
                    _dyn_graph.value().sync()
                else:
                    # Subsequent epochs: all batches via graph replay
                    for _ in range(n_batches_per_epoch):
                        _dyn_graph.value().replay_async()
                    _dyn_graph.value().sync()

                # Holdout evaluation: check every holdout_check_every epochs
                # to reduce GPU sync overhead (main Phase 2 optimization)
                if (epoch + 1) % holdout_check_every == 0 or epoch == max_epochs - 1:
                    ctx.enqueue_function[dyn_incr_k, dyn_incr_k](
                        dyn_rng_t,
                        grid_dim=(1,),
                        block_dim=(1,),
                    )
                    buffer.sample[TB](
                        ctx,
                        self.rng_counters[m],
                        self.s_obs,
                        self.s_act,
                        self.s_rew,
                        self.s_nobs,
                        self.s_done,
                        self.s_idx,
                    )

                    var h_obs_t = LayoutTensor[
                        dtype, Layout.row_major(TB, Self.obs_dim), MutAnyOrigin
                    ](self.s_obs.unsafe_ptr())
                    var h_act_t = LayoutTensor[
                        dtype,
                        Layout.row_major(TB, Self.action_dim),
                        MutAnyOrigin,
                    ](self.s_act.unsafe_ptr())
                    var h_input_t = LayoutTensor[
                        dtype,
                        Layout.row_major(TB, Self.DynModel.IN_DIM),
                        MutAnyOrigin,
                    ](self.t_input.unsafe_ptr())
                    ctx.enqueue_function[concat_k, concat_k](
                        h_input_t,
                        h_obs_t,
                        h_act_t,
                        grid_dim=(DYN_IN_BLOCKS,),
                        block_dim=(TPB_VAL,),
                    )
                    # Normalize holdout inputs with the same fitted scaler.
                    var h_mean_t = LayoutTensor[
                        dtype, Layout.row_major(Self.DYN_IN), MutAnyOrigin
                    ](self.input_mean.unsafe_ptr())
                    var h_std_t = LayoutTensor[
                        dtype, Layout.row_major(Self.DYN_IN), MutAnyOrigin
                    ](self.input_std.unsafe_ptr())
                    comptime h_norm_k = normalize_input_kernel[
                        dtype, TB, Self.DYN_IN
                    ]
                    ctx.enqueue_function[h_norm_k, h_norm_k](
                        h_input_t, h_mean_t, h_std_t,
                        grid_dim=(DYN_IN_BLOCKS,),
                        block_dim=(TPB_VAL,),
                    )

                    var h_nobs_t = LayoutTensor[
                        dtype, Layout.row_major(TB, Self.obs_dim), MutAnyOrigin
                    ](self.s_nobs.unsafe_ptr())
                    var h_rew_t = LayoutTensor[
                        dtype, Layout.row_major(TB), MutAnyOrigin
                    ](self.s_rew.unsafe_ptr())
                    var h_target_t = LayoutTensor[
                        dtype,
                        Layout.row_major(TB, Self.DYN_PRED),
                        MutAnyOrigin,
                    ](self.t_target.unsafe_ptr())
                    ctx.enqueue_function[target_k, target_k](
                        h_target_t,
                        h_obs_t,
                        h_nobs_t,
                        h_rew_t,
                        grid_dim=(PRED_BLOCKS,),
                        block_dim=(TPB_VAL,),
                    )

                    var h_output_t = LayoutTensor[
                        dtype,
                        Layout.row_major(TB, Self.DynModel.OUT_DIM),
                        MutAnyOrigin,
                    ](self.t_output.unsafe_ptr())
                    var p_h = self.members[m].params_view()
                    Self.DynNet.forward_gpu[TB](
                        ctx,
                        h_input_t,
                        h_output_t,
                        p_h,
                        self.t_ws,
                    )

                    # Compute loss (reuse NLL kernel, ignore gradients)
                    var h_grad_t = LayoutTensor[
                        dtype,
                        Layout.row_major(TB, Self.DynModel.OUT_DIM),
                        MutAnyOrigin,
                    ](self.t_grad_out.unsafe_ptr())
                    var h_loss_t = LayoutTensor[
                        dtype, Layout.row_major(TB), MutAnyOrigin
                    ](self.t_loss.unsafe_ptr())
                    # Use current learnable bounds for this member.
                    var h_max_lv_t = LayoutTensor[
                        dtype, Layout.row_major(Self.DYN_PRED), MutAnyOrigin
                    ](self.max_lv_buf.unsafe_ptr() + m * Self.DYN_PRED)
                    var h_min_lv_t = LayoutTensor[
                        dtype, Layout.row_major(Self.DYN_PRED), MutAnyOrigin
                    ](self.min_lv_buf.unsafe_ptr() + m * Self.DYN_PRED)
                    var h_max_grad_t = LayoutTensor[
                        dtype, Layout.row_major(TB, Self.DYN_PRED), MutAnyOrigin
                    ](self.max_lv_grad_scratch.unsafe_ptr())
                    var h_min_grad_t = LayoutTensor[
                        dtype, Layout.row_major(TB, Self.DYN_PRED), MutAnyOrigin
                    ](self.min_lv_grad_scratch.unsafe_ptr())
                    ctx.enqueue_function[nll_k, nll_k](
                        h_grad_t,
                        h_output_t,
                        h_target_t,
                        h_loss_t,
                        h_max_lv_t,
                        h_min_lv_t,
                        h_max_grad_t,
                        h_min_grad_t,
                        grid_dim=(PRED_BLOCKS,),
                        block_dim=(TPB_VAL,),
                    )

                    # GPU reduction: mean loss → single scalar
                    comptime reduce_k = reduce_mean_loss_kernel[dtype, TB]
                    var loss_scalar_t = LayoutTensor[
                        dtype, Layout.row_major(1), MutAnyOrigin
                    ](self.t_loss_scalar.unsafe_ptr())
                    ctx.enqueue_function[reduce_k, reduce_k](
                        h_loss_t,
                        loss_scalar_t,
                        grid_dim=(1,),
                        block_dim=(1,),
                    )

                    # Download 1 scalar (not TB elements)
                    ctx.enqueue_copy(self.t_loss_host, self.t_loss_scalar)
                    ctx.synchronize()
                    var holdout_loss = Float64(self.t_loss_host[0])

                    if holdout_loss < best_holdout:
                        best_holdout = holdout_loss
                        epochs_since_update = 0
                    else:
                        epochs_since_update += holdout_check_every
                        if epochs_since_update >= max_epochs_since_update:
                            break

            holdout_losses.append(best_holdout)

        # Select elites by holdout loss
        var sorted_indices = List[Int](capacity=Self.num_ensemble)
        for i in range(Self.num_ensemble):
            sorted_indices.append(i)
        for i in range(Self.num_ensemble):
            var min_idx = i
            for j in range(i + 1, Self.num_ensemble):
                if (
                    holdout_losses[sorted_indices[j]]
                    < holdout_losses[sorted_indices[min_idx]]
                ):
                    min_idx = j
            var tmp = sorted_indices[i]
            sorted_indices[i] = sorted_indices[min_idx]
            sorted_indices[min_idx] = tmp

        self.elite_indices.clear()
        for i in range(Self.num_elites):
            self.elite_indices.append(sorted_indices[i])
        # Sync elite→member map to GPU so the rollout sample kernel picks up
        # the new ranking on the next do_model_rollouts_gpu call.
        self.sync_elite_member_buf(ctx)

        # Return mean holdout loss across elites for diagnostic logging.
        var elite_holdout_sum: Float64 = 0.0
        for i in range(Self.num_elites):
            elite_holdout_sum += holdout_losses[self.elite_indices[i]]
        return elite_holdout_sum / Float64(Self.num_elites)


# =============================================================================
# MBPOCPUState
# =============================================================================


struct MBPOCPUState[
    Config: MBPOConfig,
](Movable, OffPolicyState):
    """CPU state for MBPO: SAC networks + dual replay buffers + dynamics ensemble.
    """

    comptime obs_dim: Int = Self.Config.obs_dim
    comptime action_dim: Int = Self.Config.action_dim
    comptime batch_size: Int = Self.Config.batch_size
    comptime BUFFER_DTYPE = dtype

    # SAC networks
    var actor: NetworkPair[Self.Config.ActorModel, Self.Config.ActorOpt]
    var critics: CriticGroup[
        Self.Config.CriticModel, Self.Config.CriticOpt, Self.Config.NUM_CRITICS
    ]

    # Dual replay buffers
    var real_buffer: HeapReplayBuffer[
        Self.Config.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]
    var synth_buffer: HeapReplayBuffer[
        Self.Config.SYNTH_CAPACITY, Self.obs_dim, Self.action_dim, dtype
    ]

    # Dynamics ensemble
    var dynamics: DynamicsEnsemble[
        Self.Config.DynamicsModel,
        Self.Config.DynOpt,
        Self.Config.ENSEMBLE_SIZE,
        Self.Config.ELITE_SIZE,
        Self.obs_dim,
        Self.action_dim,
    ]

    # Workspace for SAC training
    comptime _AL_WS: Int = Self.Config.ActorLoss.ws_size[
        Self.batch_size,
        Self.action_dim,
        Self.Config.ActorModel,
        Self.Config.CriticModel,
    ]()
    comptime _TA_WS: Int = Self.Config.TargetAction.ws_size[
        Self.batch_size,
        Self.action_dim,
        Self.Config.ActorModel.OUT_DIM,
    ]()

    comptime WS = OffPolicyTrainWS[
        Self.batch_size,
        Self.Config.ActorModel.IN_DIM,
        Self.action_dim,
        Self.Config.ActorModel.OUT_DIM,
        Self.Config.CriticModel.IN_DIM,
        Self.Config.CriticModel.OUT_DIM,
        Self.Config.CriticModel.CACHE_SIZE,
        Self.Config.ActorModel.CACHE_SIZE,
        Network[
            Self.Config.CriticModel, Self.Config.CriticOpt
        ].WORKSPACE_SIZE_PER_SAMPLE,
        Network[
            Self.Config.ActorModel, Self.Config.ActorOpt
        ].WORKSPACE_SIZE_PER_SAMPLE,
        Self.Config.NUM_CRITICS,
        Self._AL_WS,
        Self._TA_WS,
    ]

    var ws_data: List[Scalar[dtype]]

    def __init__(out self):
        self.actor = NetworkPair[Self.Config.ActorModel, Self.Config.ActorOpt]()
        self.actor.initialize[Xavier[]]()
        self.critics = CriticGroup[
            Self.Config.CriticModel,
            Self.Config.CriticOpt,
            Self.Config.NUM_CRITICS,
        ]()
        # Match reference MBPO: Keras `Dense` default → Xavier uniform for
        # both actor and critic. Kaiming (which we used previously) has ~1.4×
        # wider init and leads to 2–3× larger initial Q-magnitudes, which
        # exacerbates the high-UTD Q-explosion failure mode.
        self.critics.initialize[Xavier[]]()

        # Zero-init all biases (matches Keras `Dense(bias_initializer='zeros')`,
        # which reference MBPO inherits). AutoFused's default init loop applies
        # the provided initializer uniformly to MatMul + BiasAdd, giving biases
        # a non-zero Xavier distribution. Overwrite with zeros post-init.
        # Opt-in and MBPO-only: other agents (SAC/TD3/DDPG) keep their
        # default non-zero biases via PyTorch/Kaiming conventions.
        var actor_p = self.actor.online.params_view()
        Self.Config.ActorModel.zero_biases[dtype](actor_p)
        # Re-sync target to pick up the zeroed biases.
        self.actor.target.copy_params_from(self.actor.online)
        for i in range(Self.Config.NUM_CRITICS):
            var critic_p = self.critics.pairs[i].online.params_view()
            Self.Config.CriticModel.zero_biases[dtype](critic_p)
            self.critics.pairs[i].target.copy_params_from(
                self.critics.pairs[i].online
            )
        self.real_buffer = HeapReplayBuffer[
            Self.Config.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()
        self.synth_buffer = HeapReplayBuffer[
            Self.Config.SYNTH_CAPACITY, Self.obs_dim, Self.action_dim, dtype
        ]()
        self.dynamics = DynamicsEnsemble[
            Self.Config.DynamicsModel,
            Self.Config.DynOpt,
            Self.Config.ENSEMBLE_SIZE,
            Self.Config.ELITE_SIZE,
            Self.obs_dim,
            Self.action_dim,
        ]()
        self.ws_data = Self.WS.alloc_cpu()

    def __init__(out self, *, deinit take: Self):
        self.actor = take.actor^
        self.critics = take.critics^
        self.real_buffer = take.real_buffer^
        self.synth_buffer = take.synth_buffer^
        self.dynamics = take.dynamics^
        self.ws_data = take.ws_data^

    # OffPolicyState trait
    def store[
        d: DType
    ](
        mut self,
        obs: List[Scalar[d]],
        action: List[Scalar[d]],
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        """Store in real buffer."""
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.action_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        for i in range(Self.action_dim):
            act_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(action[i]))
        self.real_buffer.add(
            obs_arr, act_arr, Scalar[Self.BUFFER_DTYPE](reward), next_arr, done
        )

    def is_ready(self) -> Bool:
        return self.real_buffer.is_ready[Self.batch_size]()


# =============================================================================
# MBPOAgent
# =============================================================================


struct MBPOAgent[
    Config: MBPOConfig,
    L: Logger = NoOpLogger,
    TRAIN_N_ENVS: Int = 1,
    # Fraction of SAC batch drawn from the real buffer (remainder from synth).
    # Reference MBPO uses 0.05 (5%). Must be comptime because it sizes REAL_BS
    # and SYNTH_BS, which are compile-time parameters of the sampling kernels.
    # Pass as an integer basis-points-percent (e.g. 5 for 5%, 25 for 25%) so
    # the batch split is exact and the comptime arithmetic is integer-only.
    REAL_RATIO_PCT: Int = 5,
](OffPolicyContinuousAgent & Checkpointable):
    """MBPO agent: SAC + dynamics ensemble + Dyna-style data augmentation.

    `TRAIN_N_ENVS` is the number of parallel GPU envs stepped per training
    iteration. Reference MBPO uses 1; higher values add env-stepping
    throughput but delay episode feedback (each logged `AvgR` covers
    fewer completed episodes) and shift the paper's updates-per-env-step
    schedule (with `sac_updates_per_step=40` and `TRAIN_N_ENVS=N`, the
    effective updates-per-env-step = 40/N, so N=32 gives only 1.25
    updates/step — far from the paper's 40). Default 1 matches the paper.
    """

    # Dimension aliases — OBS must match ActorModel.IN_DIM for LayoutTensor
    # compatibility with Network.forward / strategy calls.
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.action_dim
    comptime ACTOR_OUT: Int = Self.Config.ActorModel.OUT_DIM
    comptime BATCH: Int = Self.Config.batch_size
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]

    # CPU state type
    comptime CPUStateType = MBPOCPUState[Self.Config]

    # Workspace type
    comptime TrainWS = Self.CPUStateType.WS

    # Strategy workspace sizes (needed for GPU state)
    comptime _AL_WS: Int = Self.CPUStateType._AL_WS
    comptime _TA_WS: Int = Self.CPUStateType._TA_WS

    # GPU buffer: real-only capacity (synthetic buffer is separate)
    comptime GPU_BUF_CAP: Int = Self.Config.buffer_capacity

    # Mixed sampling: REAL_RATIO_PCT% real, (100 - REAL_RATIO_PCT)% synthetic.
    # Derived from the comptime REAL_RATIO_PCT struct parameter — this is what
    # actually controls the batch split on the GPU path (the runtime
    # `real_ratio` constructor arg is CPU-path only and should match).
    comptime REAL_BS: Int = max(
        1, Self.Config.batch_size * Self.REAL_RATIO_PCT // 100
    )
    comptime SYNTH_BS: Int = Self.Config.batch_size - Self.REAL_BS

    # GPU state type — reuses GenericGPUState with real buffer capacity.
    # GPU_N_ENVS is set from the MBPOAgent TRAIN_N_ENVS parameter (default
    # 1, matching reference). Higher values trade episode-feedback latency
    # for env-stepping throughput.
    comptime GPU_N_ENVS: Int = Self.TRAIN_N_ENVS
    comptime GPUStateType = GenericGPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.GPU_BUF_CAP,
        Self.Config.obs_dim,
        Self.Config.action_dim,
        Self.Config.batch_size,
        Self.GPU_N_ENVS,
        Self.Config.NUM_CRITICS,
        Self._AL_WS,
        Self._TA_WS,
    ]

    # Persistent state (for evaluate after train)
    var state: Self.CPUStateType

    # SAC hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64
    var noise_std: Float64

    # Schedule
    var policy_delay: Int
    var update_count: Int

    # SAC alpha
    var alpha: Float64
    var log_alpha: Float64
    var target_entropy: Float64
    var auto_alpha: Bool
    var alpha_lr: Float64
    var alpha_adam_m: Float64
    var alpha_adam_v: Float64
    var alpha_adam_t: Int

    # Training state
    var total_steps: Int
    var train_step_count: Int

    # MBPO-specific
    var model_train_freq: Int
    var rollout_length: Int
    var rollout_min_length: Int
    var rollout_max_length: Int
    var rollout_min_epoch: Int
    var rollout_max_epoch: Int
    var num_rollouts_per_step: Int
    var real_ratio: Float64
    var sac_updates_per_step: Int

    # Global gradient-norm clip for actor + both critics. Critical at high UTD
    # + synthetic-batch mix: un-clipped critic grads let transient α·log_π
    # spikes in the TD target (from tanh saturation) drive Q-values to 10^7+.
    var max_grad_norm: Float64

    # ERE (Emphasizing Recent Experience) — biases sampling toward the most
    # recent transitions. Not in the MBPO paper, but empirically fixes the
    # Q-explosion pattern that triggers when a high-UTD SAC loop trains
    # against a rapidly-cycling replay buffer. Applied to BOTH the real
    # buffer (inside gpu_state) and the synthetic buffer. Disabled by
    # default to stay paper-faithful; enable for low-env-count runs
    # (TRAIN_N_ENVS ≤ 8) where UTD is high.
    var use_ere: Bool
    var ere_eta: Float32

    # Checkpointing
    var checkpoint_every: Int
    var checkpoint_path: String

    # Logging
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    def __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        auto_alpha: Bool = True,
        alpha: Float64 = 0.2,
        alpha_lr: Float64 = 0.0003,
        target_entropy: Float64 = 0.0,
        model_train_freq: Int = 250,
        rollout_min_length: Int = 1,
        rollout_max_length: Int = 1,
        rollout_min_epoch: Int = 20,
        rollout_max_epoch: Int = 150,
        num_rollouts_per_step: Int = 400,
        real_ratio: Float64 = 0.05,
        sac_updates_per_step: Int = 20,
        max_grad_norm: Float64 = 0.0,  # Opt-in; reference MBPO uses no clipping
        use_ere: Bool = False,
        ere_eta: Float32 = Float32(0.996),
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        self.state = Self.CPUStateType()
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.noise_std = 0.0  # SAC uses stochastic sampling, not noise
        self.policy_delay = Self.Config.Schedule.DEFAULT_POLICY_DELAY
        self.update_count = 0

        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.log_alpha = log(alpha)
        self.target_entropy = (
            target_entropy if target_entropy != 0.0 else -Float64(Self.ACTIONS)
        )
        self.alpha_lr = alpha_lr
        self.alpha_adam_m = 0.0
        self.alpha_adam_v = 0.0
        self.alpha_adam_t = 0

        self.total_steps = 0
        self.train_step_count = 0

        # MBPO hyperparameters
        self.model_train_freq = model_train_freq
        self.rollout_length = rollout_min_length
        self.rollout_min_length = rollout_min_length
        self.rollout_max_length = rollout_max_length
        self.rollout_min_epoch = rollout_min_epoch
        self.rollout_max_epoch = rollout_max_epoch
        self.num_rollouts_per_step = num_rollouts_per_step
        # GPU batch sizes are fixed by the comptime REAL_RATIO_PCT struct
        # parameter (see REAL_BS / SYNTH_BS above). Override the runtime
        # `real_ratio` value to match so CPU and GPU paths can't disagree.
        # If the caller passed a different value, emit a one-line warning.
        var rr_from_comptime = Float64(Self.REAL_RATIO_PCT) / 100.0
        if (
            real_ratio > rr_from_comptime + 1e-9
            or real_ratio < rr_from_comptime - 1e-9
        ):
            print(
                "[MBPO WARN] real_ratio=",
                real_ratio,
                " differs from REAL_RATIO_PCT (",
                rr_from_comptime,
                "); using the comptime value. Set MBPOAgent[...,"
                " REAL_RATIO_PCT=N] to change the GPU batch split.",
            )
        self.real_ratio = rr_from_comptime
        self.sac_updates_per_step = sac_updates_per_step
        self.max_grad_norm = max_grad_norm
        self.use_ere = use_ere
        self.ere_eta = ere_eta

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    # =========================================================================
    # OffPolicyContinuousAgent trait
    # =========================================================================

    def make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    def select_action[
        d: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[d]],
    ) -> List[Scalar[d]]:
        """SAC stochastic action selection."""
        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())

        var out_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
        ](out_arr.unsafe_ptr())
        var p = cpu_state.actor.online.params_view()
        Self.ActorNet.forward[1](obs_t, out_t, p)

        # Extract mean + log_std
        var mean_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var ls_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for a in range(Self.ACTIONS):
            var m = Float64(out_arr[a])
            var raw_ls = Float64(out_arr[Self.ACTIONS + a])
            if m != m:
                m = 0.0
            if raw_ls != raw_ls:
                raw_ls = 0.0
            var ls = -5.0 + 0.5 * 7.0 * (raw_ls + 1.0)
            mean_arr[a] = Scalar[dtype](m)
            ls_arr[a] = Scalar[dtype](ls)

        # Sample action via reparameterization
        var noise_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            noise_arr[i] = Scalar[dtype](gaussian_noise())

        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var lp_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var mean_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](mean_arr.unsafe_ptr())
        var ls_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](ls_arr.unsafe_ptr())
        var noise_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](noise_arr.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())
        var lp_t = LayoutTensor[dtype, Layout.row_major(1, 1), MutAnyOrigin](
            lp_arr.unsafe_ptr()
        )
        rsample[1, Self.ACTIONS](mean_t, ls_t, noise_t, act_t, lp_t)

        var result = List[Scalar[d]](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            var a = Float64(act_arr[i]) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            result.append(Scalar[d](a))
        return result^

    def store_transition[
        d: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[d]],
        action: List[Scalar[d]],
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        """Store in real buffer (normalized action)."""
        var normalized = List[Scalar[d]](capacity=len(action))
        for i in range(len(action)):
            normalized.append(Scalar[d](Float64(action[i]) / self.action_scale))
        cpu_state.store[d](obs, normalized, reward, next_obs, done)
        self.total_steps += 1

    def do_cpu_train_step(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
        """One SAC update step with mixed sampling from dual buffers."""
        # Determine how many samples from each buffer
        var real_bs = Int(Float64(Self.BATCH) * self.real_ratio)
        if real_bs < 1:
            real_bs = 1
        var synth_bs = Self.BATCH - real_bs

        # Check if we have enough data
        if cpu_state.real_buffer.size < real_bs:
            return 0.0
        # If synth buffer empty or too small, use all real
        if cpu_state.synth_buffer.size < synth_bs:
            synth_bs = 0
            real_bs = Self.BATCH

        self.update_count += 1

        # Sample mixed batch into pre-allocated arrays
        var b_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_act = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var b_rew = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var b_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_done = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)

        # Sample real data into first real_bs slots
        for b in range(real_bs):
            var idx = (
                Int(random_float64() * Float64(cpu_state.real_buffer.size))
                % cpu_state.real_buffer.size
            )
            for i in range(Self.OBS):
                b_obs[b * Self.OBS + i] = cpu_state.real_buffer.obs[
                    idx * Self.OBS + i
                ]
                b_next[b * Self.OBS + i] = cpu_state.real_buffer.next_obs[
                    idx * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                b_act[b * Self.ACTIONS + i] = cpu_state.real_buffer.actions[
                    idx * Self.ACTIONS + i
                ]
            b_rew[b] = cpu_state.real_buffer.rewards[idx]
            b_done[b] = cpu_state.real_buffer.dones[idx]

        # Sample synthetic data into remaining slots
        for b in range(synth_bs):
            var idx = (
                Int(random_float64() * Float64(cpu_state.synth_buffer.size))
                % cpu_state.synth_buffer.size
            )
            var ob = real_bs + b
            for i in range(Self.OBS):
                b_obs[ob * Self.OBS + i] = cpu_state.synth_buffer.obs[
                    idx * Self.OBS + i
                ]
                b_next[ob * Self.OBS + i] = cpu_state.synth_buffer.next_obs[
                    idx * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                b_act[ob * Self.ACTIONS + i] = cpu_state.synth_buffer.actions[
                    idx * Self.ACTIONS + i
                ]
            b_rew[ob] = cpu_state.synth_buffer.rewards[idx]
            b_done[ob] = cpu_state.synth_buffer.dones[idx]

        # === SAC training step (identical to GenericOffPolicyAgent) ===
        var ws = Self.TrainWS(cpu_state.ws_data.unsafe_ptr())

        # Phase 2: Target actions (SAC uses online actor, no target actor)
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_next.unsafe_ptr())
        var next_act_t = ws.next_act()
        Self.Config.TargetAction.compute_cpu[
            Self.BATCH,
            Self.ACTIONS,
            Self.Config.ActorModel,
            Self.Config.ActorOpt,
        ](
            next_obs_t,
            next_act_t,
            ws.next_lp().ptr,
            cpu_state.actor.online.params_view(),
            ws.strat_ws_ptr(),
            self.action_scale,
        )

        # Concat next_obs + next_act -> next_ci
        _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
            ws.next_ci().ptr, b_next.unsafe_ptr(), ws.next_act().ptr
        )
        var next_ci_t = ws.next_ci()

        # Forward all target critics
        for i in range(Self.Config.NUM_CRITICS):
            var next_qi_t = ws.next_q(i)
            var p_ct = cpu_state.critics.target_params_view(i)
            Self.CriticNet.forward[Self.BATCH](next_ci_t, next_qi_t, p_ct)

        # TD targets
        var q1_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](ws.next_q(0).ptr)
        var q2_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](ws.next_q(Self.Config.NUM_CRITICS - 1).ptr)
        var lp_tv = ws.next_lp()
        var rew_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](b_rew.unsafe_ptr())
        var done_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](b_done.unsafe_ptr())
        var tgt_tv = ws.targets()
        Self.Config.TargetValue.compute_cpu[Self.BATCH](
            q1_tv,
            q2_tv,
            lp_tv,
            rew_tv,
            done_tv,
            tgt_tv,
            self.gamma,
            self.alpha,
        )

        # Phase 3: Critic update
        _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
            ws.ci().ptr, b_obs.unsafe_ptr(), b_act.unsafe_ptr()
        )
        var ci_t = ws.ci()
        var tgt_p = ws.targets().ptr
        var q_grad_t = ws.q_grad()
        var qg_p = ws.q_grad().ptr
        var d_ci_t = ws.d_ci()
        var critic_loss: Float64 = 0.0

        for i in range(Self.Config.NUM_CRITICS):
            var qi_t = ws.q_out(i)
            var qi_cache_t = ws.q_cache(i)
            var p_ci = cpu_state.critics.online_params_view(i)
            Self.CriticNet.forward_with_cache[Self.BATCH](
                ci_t, qi_t, p_ci, qi_cache_t
            )

            var qio_p = ws.q_out(i).ptr
            var ci_loss: Float64 = 0.0
            for b in range(Self.BATCH):
                var td_err = qio_p[b] - tgt_p[b]
                ci_loss += Float64(td_err * td_err)
                qg_p[b] = (
                    Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
                )
            ci_loss /= Float64(Self.BATCH)

            var g_ci = cpu_state.critics.online_grads_view(i)
            cpu_state.critics.pairs[i].zero_grads()
            Self.CriticNet.backward[Self.BATCH](
                q_grad_t, d_ci_t, p_ci, qi_cache_t, g_ci
            )
            cpu_state.critics.pairs[i].optimizer_step()

            if i == 0:
                critic_loss = ci_loss
            else:
                critic_loss = (critic_loss + ci_loss) / 2.0

        # Diagnostic logging
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                self.logger[].log_scalar("critic_loss", critic_loss, step)
                self.logger[].log_scalar("alpha", self.alpha, step)
                self.logger[].log_scalar(
                    "real_buffer_size",
                    Float64(cpu_state.real_buffer.size),
                    step,
                )
                self.logger[].log_scalar(
                    "synth_buffer_size",
                    Float64(cpu_state.synth_buffer.size),
                    step,
                )
                self.logger[].log_scalar(
                    "rollout_length", Float64(self.rollout_length), step
                )
            except:
                pass

        # Phase 4: Actor update
        if Self.Config.Schedule.should_update_actor(
            self.update_count, self.policy_delay
        ):
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
            ](b_obs.unsafe_ptr())
            var a_grads = cpu_state.actor.online.grads_view()
            var c_grads = cpu_state.critics.online_grads_view(0)
            var c2_grads = cpu_state.critics.online_grads_view(0)
            var c2_params = cpu_state.critics.online_params_view(0)
            comptime if Self.Config.NUM_CRITICS == 2:
                c2_grads = cpu_state.critics.online_grads_view(1)
                c2_params = cpu_state.critics.online_params_view(1)
            var mean_lp = Self.Config.ActorLoss.update_actor_cpu[
                Self.BATCH,
                Self.ACTIONS,
                Self.Config.ActorModel,
                Self.Config.ActorOpt,
                Self.Config.CriticModel,
                Self.Config.CriticOpt,
            ](
                obs_t,
                cpu_state.actor.online.params_view(),
                a_grads,
                cpu_state.critics.online_params_view(0),
                c_grads,
                c2_params,
                c2_grads,
                ws.strat_ws_ptr(),
                self.alpha,
            )
            cpu_state.actor.optimizer_step()

            # Alpha auto-tuning
            comptime if Self.Config.ActorLoss.HAS_ALPHA:
                if self.auto_alpha:
                    var grad = -self.alpha * (mean_lp + self.target_entropy)
                    self.alpha_adam_t += 1
                    var beta1: Float64 = 0.9
                    var beta2: Float64 = 0.999
                    var eps: Float64 = 1e-8
                    self.alpha_adam_m = (
                        beta1 * self.alpha_adam_m + (1.0 - beta1) * grad
                    )
                    self.alpha_adam_v = (
                        beta2 * self.alpha_adam_v + (1.0 - beta2) * grad * grad
                    )
                    var m_hat = self.alpha_adam_m / (
                        1.0 - beta1 ** Float64(self.alpha_adam_t)
                    )
                    var v_hat = self.alpha_adam_v / (
                        1.0 - beta2 ** Float64(self.alpha_adam_t)
                    )
                    self.log_alpha -= (
                        self.alpha_lr * m_hat / (sqrt(v_hat) + eps)
                    )
                    # Clamp log_alpha to prevent divergence
                    if self.log_alpha > 2.0:
                        self.log_alpha = 2.0
                    elif self.log_alpha < -10.0:
                        self.log_alpha = -10.0
                    self.alpha = exp(self.log_alpha)

        # Phase 5: Soft update targets
        if Self.Config.Schedule.should_update_targets(
            self.update_count, self.policy_delay
        ):
            cpu_state.critics.soft_update_all(self.tau)

        self.train_step_count += 1
        return critic_loss

    def decay_explore(mut self) -> None:
        pass  # SAC uses entropy, no decay

    def get_explore_rate(self) -> Float64:
        return self.alpha

    def random_action[d: DType](self) -> List[Scalar[d]]:
        var result = List[Scalar[d]](capacity=Self.ACTIONS)
        for _ in range(Self.ACTIONS):
            result.append(
                Scalar[d]((random_float64() * 2.0 - 1.0) * self.action_scale)
            )
        return result^

    def select_greedy_action(
        self,
        cpu_state: Self.CPUStateType,
        obs: List[Float64],
    ) -> List[Float64]:
        """Deterministic action selection (for evaluation)."""
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())

        var out_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
        ](out_arr.unsafe_ptr())
        var p = cpu_state.actor.online.params_view()
        Self.ActorNet.forward[1](obs_t, out_t, p)

        var result = List[Float64](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            var mean = Float64(out_arr[i])
            # Apply tanh squashing
            var a = (exp(2.0 * mean) - 1.0) / (exp(2.0 * mean) + 1.0)
            a *= self.action_scale
            result.append(a)
        return result^

    # =========================================================================
    # High-level train method (matches GenericOffPolicyAgent.train)
    # =========================================================================

    def train[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_epochs: Int = 200,
        steps_per_epoch: Int = 1000,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 5000,
        eval_episodes: Int = 5,
        eval_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 1,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train the MBPO agent on a continuous-action environment.

        Creates CPU state internally, runs the MBPO training loop, and
        stores the final state for later evaluation.

        Args:
            env: Environment implementing BoxContinuousActionEnv.
            num_epochs: Number of training epochs (default: 200).
            steps_per_epoch: Env steps per epoch (default: 1000).
            max_steps_per_episode: Max episode length (default: 1000).
            warmup_steps: Random steps to fill real buffer (default: 5000).
            eval_episodes: Episodes for evaluation (default: 5).
            eval_every: Evaluate every N epochs (default: 1).
            verbose: Print progress (default: False).
            print_every: Print every N epochs if verbose (default: 1).
            environment_name: Name for metrics labeling.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N train steps (default: 0).

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType()
        var metrics = run_mbpo_train[
            E, Self.Config, Self.L, Self.TRAIN_N_ENVS, Self.REAL_RATIO_PCT
        ](
            self,
            cpu_state,
            env,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            max_steps_per_episode=max_steps_per_episode,
            warmup_steps=warmup_steps,
            eval_episodes=eval_episodes,
            eval_every=eval_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            logger=logger,
        )
        self.state = cpu_state^
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^

    # =========================================================================
    # MBPO-specific methods
    # =========================================================================

    def train_dynamics(mut self, mut cpu_state: Self.CPUStateType):
        """Train dynamics ensemble on real buffer data."""
        cpu_state.dynamics.train_model[Self.Config.buffer_capacity](
            cpu_state.real_buffer,
        )

    def do_model_rollouts(mut self, mut cpu_state: Self.CPUStateType):
        """Generate synthetic data via branched rollouts from real states."""
        var num_elites = len(cpu_state.dynamics.elite_indices)
        if num_elites == 0:
            return

        for _ in range(self.num_rollouts_per_step):
            # Sample a random start state from real buffer
            var start_idx = (
                Int(random_float64() * Float64(cpu_state.real_buffer.size))
                % cpu_state.real_buffer.size
            )

            var obs = List[Scalar[dtype]](capacity=Self.OBS)
            for i in range(Self.OBS):
                obs.append(cpu_state.real_buffer.obs[start_idx * Self.OBS + i])

            # Roll k steps
            for _ in range(self.rollout_length):
                # Select action from policy (with exploration)
                var action = self.select_action[dtype](cpu_state, obs)
                # Normalize action for storage
                var norm_action = List[Scalar[dtype]](capacity=Self.ACTIONS)
                for a in range(Self.ACTIONS):
                    norm_action.append(
                        Scalar[dtype](Float64(action[a]) / self.action_scale)
                    )

                # Randomly pick an elite model
                var elite_idx = (
                    Int(random_float64() * Float64(num_elites)) % num_elites
                )

                # Predict next state and reward
                var next_obs = List[Scalar[dtype]](capacity=Self.OBS)
                var reward_list = List[Scalar[dtype]](capacity=1)
                cpu_state.dynamics.predict_single(
                    obs, norm_action, elite_idx, next_obs, reward_list
                )
                var reward = Float64(reward_list[0])

                # Check termination
                var done = Self.Config.TermFn.is_terminal(next_obs)

                # Store synthetic transition — use Config.obs_dim / action_dim
                # to match the buffer's InlineArray type parameters.
                var obs_arr = InlineArray[Scalar[dtype], Self.Config.obs_dim](
                    uninitialized=True
                )
                var next_arr = InlineArray[Scalar[dtype], Self.Config.obs_dim](
                    uninitialized=True
                )
                var act_arr = InlineArray[
                    Scalar[dtype], Self.Config.action_dim
                ](uninitialized=True)
                for i in range(Self.Config.obs_dim):
                    obs_arr[i] = obs[i]
                    next_arr[i] = next_obs[i]
                for i in range(Self.Config.action_dim):
                    act_arr[i] = norm_action[i]
                cpu_state.synth_buffer.add(
                    obs_arr, act_arr, Scalar[dtype](reward), next_arr, done
                )

                if done:
                    break
                obs = next_obs^

    def update_rollout_length(mut self, epoch: Int):
        """Linearly interpolate rollout length based on epoch."""
        if epoch <= self.rollout_min_epoch:
            self.rollout_length = self.rollout_min_length
        elif epoch >= self.rollout_max_epoch:
            self.rollout_length = self.rollout_max_length
        else:
            var progress = Float64(epoch - self.rollout_min_epoch) / Float64(
                self.rollout_max_epoch - self.rollout_min_epoch
            )
            self.rollout_length = self.rollout_min_length + Int(
                progress
                * Float64(self.rollout_max_length - self.rollout_min_length)
            )

    # =========================================================================
    # GPU dynamics training & rollouts
    # =========================================================================

    def train_dynamics_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_dynamics: GPUDynamicsEnsemble[
            Self.Config.DynamicsModel,
            Self.Config.DynOpt,
            Self.Config.ENSEMBLE_SIZE,
            Self.Config.ELITE_SIZE,
            Self.Config.obs_dim,
            Self.Config.action_dim,
        ],
        gpu_buffer: GPUReplayBuffer[
            Self.GPU_BUF_CAP, Self.Config.obs_dim, Self.Config.action_dim
        ],
    ) raises:
        """Train dynamics ensemble on GPU using data from GPU replay buffer."""
        _ = gpu_dynamics.train_on_buffer[Self.GPU_BUF_CAP](
            ctx,
            gpu_buffer,
        )

    def do_model_rollouts_gpu[
        E: GPUContinuousEnv,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_dynamics: GPUDynamicsEnsemble[
            Self.Config.DynamicsModel,
            Self.Config.DynOpt,
            Self.Config.ENSEMBLE_SIZE,
            Self.Config.ELITE_SIZE,
            Self.Config.obs_dim,
            Self.Config.action_dim,
        ],
        mut gpu_state: Self.GPUStateType,
        mut synth_buffer: GPUReplayBuffer[
            Self.Config.SYNTH_CAPACITY, Self.Config.obs_dim, Self.Config.action_dim
        ],
    ) raises:
        """GPU model rollouts: sample starts from real buffer, store in synth buffer.

        For each rollout step:
        1. Sample rollout_batch start obs from GPU buffer
        2. Forward actor → actions
        3. Concat [obs, action] → dynamics input
        4. Forward EACH elite model → r_dyn_output_all[slot_i, b, :]
        5. Pick per-sample random elite slot (matches fake_env.py:54)
        6. dynamics_sample_ensemble_learnable_kernel → next_obs, reward
        7. Store transitions in GPU buffer
        8. Copy next_obs → obs for next step
        """
        comptime RB = gpu_dynamics.rollout_batch
        comptime TPB_VAL = 256
        comptime RB_BLOCKS = (RB + TPB_VAL - 1) // TPB_VAL
        comptime DYN_IN = gpu_dynamics.DYN_IN
        comptime DYN_OUT = gpu_dynamics.DYN_OUT
        comptime DYN_PRED = gpu_dynamics.DYN_PRED
        comptime NUM_ELITES_C = Self.Config.ELITE_SIZE
        comptime NUM_ENSEMBLE_C = Self.Config.ENSEMBLE_SIZE
        comptime DYN_IN_BLOCKS = (RB * DYN_IN + TPB_VAL - 1) // TPB_VAL

        comptime concat_k = concat_obs_action_kernel[
            dtype, RB, Self.Config.obs_dim, Self.Config.action_dim, DYN_IN
        ]
        comptime sample_k = dynamics_sample_ensemble_learnable_kernel[
            dtype,
            RB,
            Self.Config.obs_dim,
            NUM_ELITES_C,
            NUM_ENSEMBLE_C,
            DYN_PRED,
            DYN_OUT,
        ]
        comptime elite_assign_k = sample_elite_assignment_kernel[
            dtype, RB, NUM_ELITES_C
        ]

        var num_elites = len(gpu_dynamics.elite_indices)
        if num_elites == 0 or not gpu_state.buffer.is_ready[RB]():
            return

        # Sample start observations from GPU buffer
        comptime rollout_incr_k = increment_rng_counter_kernel
        var rollout_rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.rng_counter.unsafe_ptr())
        ctx.enqueue_function[rollout_incr_k, rollout_incr_k](
            rollout_rng_t,
            grid_dim=(1,),
            block_dim=(1,),
        )
        gpu_state.buffer.sample[RB](
            ctx,
            gpu_state.rng_counter,
            gpu_dynamics.s_obs,
            gpu_dynamics.s_act,
            gpu_dynamics.s_rew,
            gpu_dynamics.s_nobs,
            gpu_dynamics.s_done,
            gpu_dynamics.s_idx,
        )
        # Copy sampled obs as rollout start states
        ctx.enqueue_copy(gpu_dynamics.r_obs, gpu_dynamics.s_obs)

        # Initialize alive mask (all rollouts start alive)
        gpu_dynamics.r_alive.enqueue_fill(Scalar[dtype](1.0))

        for step in range(self.rollout_length):
            # Forward actor → actions
            # Use Self.OBS (= Config.ActorModel.IN_DIM) for actor LayoutTensors
            var r_obs_t = LayoutTensor[
                dtype,
                Layout.row_major(RB, Self.OBS),
                MutAnyOrigin,
            ](gpu_dynamics.r_obs.unsafe_ptr())
            var r_act_t = LayoutTensor[
                dtype,
                Layout.row_major(RB, Self.ACTIONS),
                MutAnyOrigin,
            ](gpu_dynamics.r_actions.unsafe_ptr())
            var raw_t = LayoutTensor[
                dtype, Layout.row_major(RB, Self.ACTOR_OUT), MutAnyOrigin
            ](
                gpu_dynamics.r_dyn_output.unsafe_ptr()
            )  # reuse buffer for raw output
            var p_actor = gpu_state.actor.online.params_view()

            Self.ActorNet.forward_gpu[RB](
                ctx,
                r_obs_t,
                raw_t,
                p_actor,
                gpu_dynamics.r_ws,
            )

            # SAC sample actions
            comptime sac_k = sac_sample_actions_kernel[
                dtype,
                RB,
                Self.ACTIONS,
                Self.ACTOR_OUT,
            ]
            ctx.enqueue_function[sac_k, sac_k](
                r_act_t,
                raw_t,
                Scalar[dtype](self.action_scale),
                Scalar[dtype](-5.0),
                Scalar[dtype](2.0),
                Scalar[DType.uint32](UInt32(self.total_steps + step)),
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # Concat [obs, action] → dynamics input
            # Use DynamicsModel.IN_DIM for dynamics LayoutTensors
            var r_dyn_in_t = LayoutTensor[
                dtype,
                Layout.row_major(RB, Self.Config.DynamicsModel.IN_DIM),
                MutAnyOrigin,
            ](gpu_dynamics.r_dyn_input.unsafe_ptr())
            ctx.enqueue_function[concat_k, concat_k](
                r_dyn_in_t,
                r_obs_t,
                r_act_t,
                grid_dim=(DYN_IN_BLOCKS,),
                block_dim=(TPB_VAL,),
            )
            # Normalize using the scaler fitted at last train_on_buffer call.
            var r_mean_t = LayoutTensor[
                dtype, Layout.row_major(DYN_IN), MutAnyOrigin
            ](gpu_dynamics.input_mean.unsafe_ptr())
            var r_std_t = LayoutTensor[
                dtype, Layout.row_major(DYN_IN), MutAnyOrigin
            ](gpu_dynamics.input_std.unsafe_ptr())
            comptime r_norm_k = normalize_input_kernel[dtype, RB, DYN_IN]
            ctx.enqueue_function[r_norm_k, r_norm_k](
                r_dyn_in_t, r_mean_t, r_std_t,
                grid_dim=(DYN_IN_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # Forward dynamics model — run ALL elites so the sample kernel
            # can pick a per-sample random elite (matches fake_env.py:43
            # with factored=True, then elite-restricted random selection).
            # Each elite writes to its own slot in r_dyn_output_all.
            comptime DynNet = Network[
                Self.Config.DynamicsModel, Self.Config.DynOpt
            ]
            for i in range(NUM_ELITES_C):
                var elite_member_idx = gpu_dynamics.elite_indices[i]
                var r_dyn_out_slot_t = LayoutTensor[
                    dtype,
                    Layout.row_major(RB, Self.Config.DynamicsModel.OUT_DIM),
                    MutAnyOrigin,
                ](
                    gpu_dynamics.r_dyn_output_all.unsafe_ptr()
                    + i * RB * DYN_OUT
                )
                var p_dyn = gpu_dynamics.members[
                    elite_member_idx
                ].params_view()
                # r_ws is shared across elites — safe because enqueue is FIFO
                # on the stream and each forward finishes before the next
                # starts (both use r_ws, so hardware serializes them).
                DynNet.forward_gpu[RB](
                    ctx,
                    r_dyn_in_t,
                    r_dyn_out_slot_t,
                    p_dyn,
                    gpu_dynamics.r_ws,
                )

            # Per-batch random elite slot assignment. Fresh RNG counter per
            # rollout step so consecutive steps in the same batch don't reuse
            # the same elite pattern.
            var elite_rng_t = LayoutTensor[
                DType.uint32, Layout.row_major(1), MutAnyOrigin
            ](gpu_dynamics.r_elite_rng.unsafe_ptr())
            ctx.enqueue_function[rollout_incr_k, rollout_incr_k](
                elite_rng_t,
                grid_dim=(1,),
                block_dim=(1,),
            )
            var elite_slot_t = LayoutTensor[
                DType.int32, Layout.row_major(RB), MutAnyOrigin
            ](gpu_dynamics.r_elite_idx_per_sample.unsafe_ptr())
            ctx.enqueue_function[elite_assign_k, elite_assign_k](
                elite_slot_t,
                elite_rng_t,
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # Sample next_obs and reward — per-sample random elite (matches
            # softlearning fake_env.py:54-58).
            var r_next_t = LayoutTensor[
                dtype,
                Layout.row_major(RB, Self.OBS),
                MutAnyOrigin,
            ](gpu_dynamics.r_next_obs.unsafe_ptr())
            var r_rew_t = LayoutTensor[
                dtype, Layout.row_major(RB), MutAnyOrigin
            ](gpu_dynamics.r_rewards.unsafe_ptr())
            var r_dyn_out_all_t = LayoutTensor[
                dtype,
                Layout.row_major(
                    NUM_ELITES_C, RB, Self.Config.DynamicsModel.OUT_DIM
                ),
                MutAnyOrigin,
            ](gpu_dynamics.r_dyn_output_all.unsafe_ptr())
            var r_elite_map_t = LayoutTensor[
                DType.int32, Layout.row_major(NUM_ELITES_C), MutAnyOrigin
            ](gpu_dynamics.elite_member_buf.unsafe_ptr())
            var r_max_lv_all_t = LayoutTensor[
                dtype,
                Layout.row_major(NUM_ENSEMBLE_C * DYN_PRED),
                MutAnyOrigin,
            ](gpu_dynamics.max_lv_buf.unsafe_ptr())
            var r_min_lv_all_t = LayoutTensor[
                dtype,
                Layout.row_major(NUM_ENSEMBLE_C * DYN_PRED),
                MutAnyOrigin,
            ](gpu_dynamics.min_lv_buf.unsafe_ptr())
            ctx.enqueue_function[sample_k, sample_k](
                r_next_t,
                r_rew_t,
                r_dyn_out_all_t,
                r_obs_t,
                elite_slot_t,
                r_elite_map_t,
                r_max_lv_all_t,
                r_min_lv_all_t,
                Scalar[DType.uint32](UInt32(self.total_steps * 100 + step)),
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # Clamp synthetic rewards to prevent NaN cascades
            comptime clamp_k = clamp_rewards_kernel[dtype, RB]
            ctx.enqueue_function[clamp_k, clamp_k](
                r_rew_t,
                Scalar[dtype](-100.0),
                Scalar[dtype](100.0),
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # GPU termination check on predicted next_obs (env-side)
            E.is_terminal_obs_gpu[RB, Self.OBS](
                ctx, gpu_dynamics.r_next_obs, gpu_dynamics.r_dones
            )

            # Mask dead rollouts: zero reward + set done=1 for already-dead
            # This prevents storing meaningless transitions from terminated rollouts
            comptime mask_dead_k = mask_dead_rollouts_kernel[dtype, RB]
            var alive_t = LayoutTensor[
                dtype, Layout.row_major(RB), MutAnyOrigin
            ](gpu_dynamics.r_alive.unsafe_ptr())
            var dones_for_mask = LayoutTensor[
                dtype, Layout.row_major(RB), MutAnyOrigin
            ](gpu_dynamics.r_dones.unsafe_ptr())
            ctx.enqueue_function[mask_dead_k, mask_dead_k](
                alive_t,
                r_rew_t,
                dones_for_mask,
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # Store transitions in GPU buffer
            synth_buffer.store[RB](
                ctx,
                gpu_dynamics.r_obs,
                gpu_dynamics.r_actions,
                gpu_dynamics.r_rewards,
                gpu_dynamics.r_next_obs,
                gpu_dynamics.r_dones,
            )

            # Update alive mask: alive[b] *= (1 - dones[b])
            # Must use ORIGINAL dones (before mask_dead zeroed them),
            # but mask_dead only sets dones=1 for already-dead, so
            # dones now has 1.0 for both newly terminated AND already dead.
            # alive * (1-1) = 0 for both cases — correct.
            comptime update_alive_k = update_alive_mask_kernel[dtype, RB]
            ctx.enqueue_function[update_alive_k, update_alive_k](
                alive_t,
                dones_for_mask,
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # Copy next_obs → obs for next rollout step
            ctx.enqueue_copy(gpu_dynamics.r_obs, gpu_dynamics.r_next_obs)

    # =========================================================================
    # GPU SAC methods
    # =========================================================================

    def _upload_buffers_to_gpu(
        self,
        cpu_state: Self.CPUStateType,
        mut gpu_state: Self.GPUStateType,
        mut synth_buffer: GPUReplayBuffer[
            Self.Config.SYNTH_CAPACITY, Self.Config.obs_dim, Self.Config.action_dim
        ],
        ctx: DeviceContext,
    ) raises:
        """Upload CPU real + synth buffers to separate GPU buffers."""
        gpu_state.buffer.upload_from(cpu_state.real_buffer, ctx)
        synth_buffer.upload_from(cpu_state.synth_buffer, ctx)
        ctx.synchronize()

    def select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises:
        """Forward actor on GPU + SAC stochastic sampling."""
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = gpu_state.explore.raw_act[N_ENVS]()
        var p = gpu_state.actor.online.params_view()

        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.explore_buf
        )

        var act_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rng_seed_s = Scalar[DType.uint32](
            UInt32(self.total_steps) * UInt32(Self.ACTIONS)
        )

        # SAC: stochastic sample from actor output
        comptime BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime sac_explore_k = sac_sample_actions_kernel[
            dtype, N_ENVS, Self.ACTIONS, Self.ACTOR_OUT
        ]
        ctx.enqueue_function[sac_explore_k, sac_explore_k](
            act_t,
            raw_t,
            Scalar[dtype](self.action_scale),
            Scalar[dtype](-5.0),
            Scalar[dtype](2.0),
            rng_seed_s,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    def _gpu_train_kernels(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        synth_buffer: GPUReplayBuffer[
            Self.Config.SYNTH_CAPACITY,
            Self.Config.obs_dim,
            Self.Config.action_dim,
        ],
        s_real_idx: DeviceBuffer[DType.int32],
        s_synth_idx: DeviceBuffer[DType.int32],
    ) raises:
        """Pure GPU kernel sequence for one SAC training step.

        Contains ONLY GPU kernel enqueues — no CPU counters, no diagnostics,
        no ctx.synchronize(). Fully CUDA graph capturable.

        Uses dual-buffer mixed sampling: REAL_BS from gpu_state.buffer (real),
        SYNTH_BS from synth_buffer. Output is concatenated into gpu_state.s_*
        scratch buffers at offset regions.
        """
        comptime BS = Self.BATCH
        comptime ELEM_BLOCKS = (BS * Self.CRITIC_IN + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (BS + TPB - 1) // TPB
        comptime concat_k = concat_obs_action_kernel[
            dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN
        ]
        comptime mse_grad_k = td_mse_grad_kernel[dtype, BS, Self.CRITIC_OUT]

        # Phase 1: Mixed sampling — REAL_BS from real, SYNTH_BS from synthetic
        comptime RBS = Self.REAL_BS
        comptime SBS = Self.SYNTH_BS

        # Increment GPU-side RNG counter (CUDA graph compatible)
        comptime incr_k = increment_rng_counter_kernel
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.rng_counter.unsafe_ptr())
        ctx.enqueue_function[incr_k, incr_k](
            rng_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

        # Sample REAL_BS from real buffer into first portion of batch
        var real_s_obs = DeviceBuffer[dtype](
            ctx, gpu_state.s_obs.unsafe_ptr(), RBS * Self.OBS, owning=False
        )
        var real_s_act = DeviceBuffer[dtype](
            ctx, gpu_state.s_act.unsafe_ptr(), RBS * Self.ACTIONS, owning=False
        )
        var real_s_rew = DeviceBuffer[dtype](
            ctx, gpu_state.s_rew.unsafe_ptr(), RBS, owning=False
        )
        var real_s_nobs = DeviceBuffer[dtype](
            ctx, gpu_state.s_nobs.unsafe_ptr(), RBS * Self.OBS, owning=False
        )
        var real_s_done = DeviceBuffer[dtype](
            ctx, gpu_state.s_done.unsafe_ptr(), RBS, owning=False
        )
        gpu_state.buffer.sample[RBS](
            ctx,
            rng_counter=gpu_state.rng_counter,
            sampled_obs=real_s_obs,
            sampled_actions=real_s_act,
            sampled_rewards=real_s_rew,
            sampled_next_obs=real_s_nobs,
            sampled_dones=real_s_done,
            indices=s_real_idx,
        )

        # Increment RNG again for independent synthetic sampling
        ctx.enqueue_function[incr_k, incr_k](
            rng_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

        # Sample SYNTH_BS from synthetic buffer into remaining portion
        var synth_s_obs = DeviceBuffer[dtype](
            ctx,
            gpu_state.s_obs.unsafe_ptr() + RBS * Self.OBS,
            SBS * Self.OBS,
            owning=False,
        )
        var synth_s_act = DeviceBuffer[dtype](
            ctx,
            gpu_state.s_act.unsafe_ptr() + RBS * Self.ACTIONS,
            SBS * Self.ACTIONS,
            owning=False,
        )
        var synth_s_rew = DeviceBuffer[dtype](
            ctx,
            gpu_state.s_rew.unsafe_ptr() + RBS,
            SBS,
            owning=False,
        )
        var synth_s_nobs = DeviceBuffer[dtype](
            ctx,
            gpu_state.s_nobs.unsafe_ptr() + RBS * Self.OBS,
            SBS * Self.OBS,
            owning=False,
        )
        var synth_s_done = DeviceBuffer[dtype](
            ctx,
            gpu_state.s_done.unsafe_ptr() + RBS,
            SBS,
            owning=False,
        )
        synth_buffer.sample[SBS](
            ctx,
            rng_counter=gpu_state.rng_counter,
            sampled_obs=synth_s_obs,
            sampled_actions=synth_s_act,
            sampled_rewards=synth_s_rew,
            sampled_next_obs=synth_s_nobs,
            sampled_dones=synth_s_done,
            indices=s_synth_idx,
        )

        # SAC critic + actor update on whatever is in s_obs, s_act, etc.
        self._gpu_sac_update(ctx, gpu_state)

    def _gpu_sac_update(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises:
        """SAC critic + actor update on pre-filled batch buffers.

        Assumes gpu_state.s_obs, s_act, s_rew, s_nobs, s_done are already
        filled with a training batch (from mixed or real-only sampling).
        Pure GPU kernel sequence — CUDA graph capturable.
        """
        comptime BS = Self.BATCH
        comptime ELEM_BLOCKS = (BS * Self.CRITIC_IN + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (BS + TPB - 1) // TPB
        comptime concat_k = concat_obs_action_kernel[
            dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN
        ]
        comptime mse_grad_k = td_mse_grad_kernel[dtype, BS, Self.CRITIC_OUT]
        comptime incr_k = increment_rng_counter_kernel
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.rng_counter.unsafe_ptr())

        var obs_t = gpu_state.obs_view[BS]()
        var nobs_t = gpu_state.nobs_view[BS]()
        var act_t = gpu_state.act_view[BS]()
        var rew_t = gpu_state.rew_view[BS]()
        var done_t = gpu_state.done_view[BS]()
        var p_actor = gpu_state.actor.online.params_view()
        var p_critic = gpu_state.critics.online_params_view(0)
        var p_critic_t = gpu_state.critics.target_params_view(0)

        # Phase 2: Target actions (SAC: use online actor, no target)
        # Increment RNG counter before target action
        ctx.enqueue_function[incr_k, incr_k](
            rng_t,
            grid_dim=(1,),
            block_dim=(1,),
        )
        var next_act_t = gpu_state.next_act_view[BS]()
        var next_lp_t = gpu_state.next_lp_view[BS]()
        Self.Config.TargetAction.compute_gpu[
            BS,
            Self.ACTIONS,
            Self.Config.ActorModel,
            Self.Config.ActorOpt,
        ](
            ctx,
            nobs_t,
            next_act_t,
            next_lp_t,
            p_actor,
            gpu_state.actor_ws,
            gpu_state.target_strat_ws,
            gpu_state.rng_counter,
            Scalar[dtype](self.action_scale),
        )

        # Concat next_obs + next_act → next_ci, forward target critics
        var next_ci_t = gpu_state.next_ci_view[BS]()
        ctx.enqueue_function[concat_k, concat_k](
            next_ci_t,
            nobs_t,
            next_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        var next_q_t = gpu_state.next_q_view[BS]()
        Self.CriticNet.forward_gpu[BS](
            ctx, next_ci_t, next_q_t, p_critic_t, gpu_state.critic_ws
        )
        comptime if Self.Config.NUM_CRITICS == 2:
            var nq2_t = gpu_state.nq2_view[BS]()
            var p_c2t = gpu_state.critics.target_params_view(1)
            Self.CriticNet.forward_gpu[BS](
                ctx, next_ci_t, nq2_t, p_c2t, gpu_state.critic2_ws
            )

        # Phase 2c: TD targets
        var nq1_flat = LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            gpu_state.next_q.unsafe_ptr()
        )
        var nq2_flat = LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            gpu_state.nq2.unsafe_ptr()
        )
        var targets_t = gpu_state.targets_view[BS]()
        Self.Config.TargetValue.compute_gpu[BS](
            ctx,
            nq1_flat,
            nq2_flat,
            next_lp_t,
            rew_t,
            done_t,
            targets_t,
            self.gamma,
            gpu_state.gpu_scalars,
        )

        # Phase 3: Critic update
        var ci_t = gpu_state.ci_view[BS]()
        var q_t = gpu_state.q_out_view[BS]()
        var q_cache_t = gpu_state.q_cache_view[BS]()
        var q_grad_t = gpu_state.q_grad_view[BS]()
        var d_ci_t = gpu_state.d_ci_view[BS]()

        ctx.enqueue_function[concat_k, concat_k](
            ci_t,
            obs_t,
            act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )
        Self.CriticNet.forward_gpu_with_cache[BS](
            ctx,
            ci_t,
            q_t,
            p_critic,
            q_cache_t,
            gpu_state.critic_ws,
        )
        ctx.enqueue_function[mse_grad_k, mse_grad_k](
            q_grad_t,
            q_t,
            targets_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        var g_critic = gpu_state.critics.online_grads_view(0)
        gpu_state.critics.pairs[0].online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BS](
            ctx,
            q_grad_t,
            d_ci_t,
            p_critic,
            q_cache_t,
            g_critic,
            gpu_state.critic_ws,
        )
        # Clip critic1 gradients. Without this, a single transient target
        # spike (from α·log_π blowing up under tanh saturation) sends the
        # critic weights to ±∞ and Q-values never recover.
        if self.max_grad_norm > 0.0:
            comptime C_PS = Self.Config.CriticModel.PARAM_SIZE
            comptime C_BLOCKS = (C_PS + TPB - 1) // TPB
            comptime c_norm_k = gradient_norm_kernel[dtype, C_PS, C_BLOCKS, TPB]
            comptime c_clip_k = gradient_reduce_apply_fused_kernel[
                dtype, C_PS, C_BLOCKS, TPB
            ]
            var c_ps_t = LayoutTensor[
                dtype, Layout.row_major(C_BLOCKS), MutAnyOrigin
            ](gpu_state.grad_clip_ps.unsafe_ptr())

            ctx.enqueue_function[c_norm_k, c_norm_k](
                c_ps_t,
                g_critic,
                grid_dim=(C_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[c_clip_k, c_clip_k](
                g_critic,
                c_ps_t,
                Scalar[dtype](self.max_grad_norm),
                grid_dim=(C_BLOCKS,),
                block_dim=(TPB,),
            )
        gpu_state.critics.pairs[0].online.optimizer_step(ctx)

        # Critic2 update (twin critics)
        comptime if Self.Config.NUM_CRITICS == 2:
            var q2_out_t = gpu_state.q2_out_view[BS]()
            var q2_cache_t = gpu_state.q2_cache_view[BS]()
            var p_c2 = gpu_state.critics.online_params_view(1)
            Self.CriticNet.forward_gpu_with_cache[BS](
                ctx,
                ci_t,
                q2_out_t,
                p_c2,
                q2_cache_t,
                gpu_state.critic2_ws,
            )
            ctx.enqueue_function[mse_grad_k, mse_grad_k](
                q_grad_t,
                q2_out_t,
                targets_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            var g_c2 = gpu_state.critics.online_grads_view(1)
            gpu_state.critics.pairs[1].online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BS](
                ctx,
                q_grad_t,
                d_ci_t,
                p_c2,
                q2_cache_t,
                g_c2,
                gpu_state.critic2_ws,
            )
            # Clip critic2 gradients
            if self.max_grad_norm > 0.0:
                comptime C_PS2 = Self.Config.CriticModel.PARAM_SIZE
                comptime C_BLOCKS2 = (C_PS2 + TPB - 1) // TPB
                comptime c2_norm_k = gradient_norm_kernel[
                    dtype, C_PS2, C_BLOCKS2, TPB
                ]
                comptime c2_clip_k = gradient_reduce_apply_fused_kernel[
                    dtype, C_PS2, C_BLOCKS2, TPB
                ]
                var c2_ps_t = LayoutTensor[
                    dtype, Layout.row_major(C_BLOCKS2), MutAnyOrigin
                ](gpu_state.grad_clip_ps.unsafe_ptr())

                ctx.enqueue_function[c2_norm_k, c2_norm_k](
                    c2_ps_t,
                    g_c2,
                    grid_dim=(C_BLOCKS2,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[c2_clip_k, c2_clip_k](
                    g_c2,
                    c2_ps_t,
                    Scalar[dtype](self.max_grad_norm),
                    grid_dim=(C_BLOCKS2,),
                    block_dim=(TPB,),
                )
            gpu_state.critics.pairs[1].online.optimizer_step(ctx)

        # Phase 4: Actor update (always runs for SAC/EveryStep schedule)
        if Self.Config.Schedule.should_update_actor(1, self.policy_delay):
            gpu_state.actor.online.zero_grads(ctx)
            gpu_state.critics.pairs[0].online.zero_grads(ctx)
            var a_grads = gpu_state.actor.online.grads_view()
            var c_grads = gpu_state.critics.online_grads_view(0)
            var c2_grads = c_grads
            var p_c2 = p_critic
            var c2_ws = gpu_state.critic_ws
            comptime if Self.Config.NUM_CRITICS == 2:
                gpu_state.critics.pairs[1].online.zero_grads(ctx)
                c2_grads = gpu_state.critics.online_grads_view(1)
                p_c2 = gpu_state.critics.online_params_view(1)
                c2_ws = gpu_state.critic2_ws
            # Increment RNG counter before actor loss
            ctx.enqueue_function[incr_k, incr_k](
                rng_t,
                grid_dim=(1,),
                block_dim=(1,),
            )
            _ = Self.Config.ActorLoss.update_actor_gpu[
                BS,
                Self.ACTIONS,
                Self.Config.ActorModel,
                Self.Config.ActorOpt,
                Self.Config.CriticModel,
                Self.Config.CriticOpt,
            ](
                ctx,
                obs_t,
                p_actor,
                a_grads,
                p_critic,
                c_grads,
                p_c2,
                c2_grads,
                gpu_state.actor_ws,
                gpu_state.critic_ws,
                c2_ws,
                gpu_state.strat_ws,
                gpu_state.dq,
                gpu_state.gpu_scalars,
                gpu_state.rng_counter,
            )

            # Clip actor gradients
            if self.max_grad_norm > 0.0:
                comptime A_PS = Self.Config.ActorModel.PARAM_SIZE
                comptime A_BLOCKS = (A_PS + TPB - 1) // TPB
                comptime a_norm_k = gradient_norm_kernel[
                    dtype, A_PS, A_BLOCKS, TPB
                ]
                comptime a_clip_k = gradient_reduce_apply_fused_kernel[
                    dtype, A_PS, A_BLOCKS, TPB
                ]
                var a_ps_t = LayoutTensor[
                    dtype, Layout.row_major(A_BLOCKS), MutAnyOrigin
                ](gpu_state.grad_clip_ps.unsafe_ptr())

                ctx.enqueue_function[a_norm_k, a_norm_k](
                    a_ps_t,
                    a_grads,
                    grid_dim=(A_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[a_clip_k, a_clip_k](
                    a_grads,
                    a_ps_t,
                    Scalar[dtype](self.max_grad_norm),
                    grid_dim=(A_BLOCKS,),
                    block_dim=(TPB,),
                )
            gpu_state.actor.online.optimizer_step(ctx)

            # Alpha auto-tuning (GPU-side Adam — no D2H or sync)
            comptime if Self.Config.ActorLoss.HAS_ALPHA:
                if self.auto_alpha:
                    comptime LP_OFF = Self.Config.ActorLoss.gpu_lp_offset[
                        BS,
                        Self.ACTIONS,
                        Self.ACTOR_OUT,
                        Self.Config.ActorModel.CACHE_SIZE,
                    ]()
                    var src_lp = LayoutTensor[
                        dtype, Layout.row_major(BS), MutAnyOrigin
                    ](gpu_state.strat_ws.unsafe_ptr() + LP_OFF)

                    comptime GS = Self.GPUStateType
                    comptime mbpo_alpha_k = alpha_adam_update_kernel[
                        dtype,
                        BS,
                        GS.GPU_ALPHA,
                        GS.GPU_LOG_ALPHA,
                        GS.GPU_ADAM_M,
                        GS.GPU_ADAM_V,
                        GS.GPU_ADAM_T,
                        GS.GPU_TARGET_ENT,
                        GS.GPU_ALPHA_LR,
                    ]
                    var scalars_t = LayoutTensor[
                        dtype, Layout.row_major(1), MutAnyOrigin
                    ](gpu_state.gpu_scalars.unsafe_ptr())

                    @always_inline
                    def mbpo_alpha_wrapper(
                        sc: LayoutTensor[
                            dtype, Layout.row_major(1), MutAnyOrigin
                        ],
                        lp: LayoutTensor[
                            dtype, Layout.row_major(BS), MutAnyOrigin
                        ],
                        la_max: Scalar[dtype],
                        la_min: Scalar[dtype],
                        lp_clip: Scalar[dtype],
                    ):
                        mbpo_alpha_k(sc, lp, la_max, la_min, lp_clip)

                    # Tight log_alpha ceiling for MBPO: +0.5 → alpha <= 1.65.
                    # Prevents transient Q-spikes (common at high UTD with
                    # synthetic rollouts) from pinning alpha against the
                    # classic SAC cap of exp(2) = 7.4, where the policy
                    # collapses to max-entropy and never recovers.
                    # lp_clip=50 bounds per-sample log_pi so a single
                    # tanh-saturated batch element can't poison Adam's moments.
                    ctx.enqueue_function[
                        mbpo_alpha_wrapper, mbpo_alpha_wrapper
                    ](
                        scalars_t,
                        src_lp,
                        Scalar[dtype](0.5),
                        Scalar[dtype](-10.0),
                        Scalar[dtype](50.0),
                        grid_dim=(1,),
                        block_dim=(1,),
                    )

    def _gpu_train_diagnostics(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        steps: Int,
        alpha_host: HostBuffer[dtype],
    ) raises:
        """CPU-side bookkeeping + diagnostics. Call outside graph.

        `alpha_host` is a size-1 host buffer pre-allocated by the training
        loop and reused across diagnostic calls (avoids per-call allocs).
        """
        self.train_step_count += steps
        self.update_count += steps

        # GPU Diagnostic logging (periodic)
        comptime BS = Self.BATCH
        if (
            self.logger
            and self.diag_every > 0
            and self.train_step_count % self.diag_every == 0
        ):
            try:
                ctx.enqueue_copy(gpu_state.diag_q_host, gpu_state.q_out)
                ctx.enqueue_copy(gpu_state.diag_tgt_host, gpu_state.targets)
                ctx.enqueue_copy(gpu_state.diag_rew_host, gpu_state.s_rew)
                ctx.enqueue_copy(gpu_state.diag_done_host, gpu_state.s_done)
                ctx.enqueue_copy(gpu_state.diag_act_host, gpu_state.s_act)
                ctx.enqueue_copy(gpu_state.diag_nq_host, gpu_state.next_q)
                ctx.synchronize()
                var diag_q_host = gpu_state.diag_q_host
                var diag_tgt_host = gpu_state.diag_tgt_host
                var diag_rew_host = gpu_state.diag_rew_host
                var diag_done_host = gpu_state.diag_done_host
                var diag_act_host = gpu_state.diag_act_host
                var diag_nq_host = gpu_state.diag_nq_host

                var mean_q: Float64 = 0.0
                var mean_tgt: Float64 = 0.0
                var mean_rew: Float64 = 0.0
                var mean_done: Float64 = 0.0
                var critic_loss: Float64 = 0.0
                var mean_nq: Float64 = 0.0
                var mean_abs_act: Float64 = 0.0
                for b in range(BS):
                    var q_val = Float64(diag_q_host[b])
                    var tgt_val = Float64(diag_tgt_host[b])
                    mean_q += q_val
                    mean_tgt += tgt_val
                    mean_rew += Float64(diag_rew_host[b])
                    mean_done += Float64(diag_done_host[b])
                    mean_nq += Float64(diag_nq_host[b])
                    critic_loss += (q_val - tgt_val) * (q_val - tgt_val)
                for i in range(BS * Self.ACTIONS):
                    var a = Float64(diag_act_host[i])
                    mean_abs_act += a if a >= 0.0 else -a
                mean_q /= Float64(BS)
                mean_tgt /= Float64(BS)
                mean_rew /= Float64(BS)
                mean_done /= Float64(BS)
                mean_nq /= Float64(BS)
                critic_loss /= Float64(BS)
                mean_abs_act /= Float64(BS * Self.ACTIONS)

                var step = self.train_step_count
                self.logger[].log_scalar("critic_loss", critic_loss, step)
                self.logger[].log_scalar("mean_q", mean_q, step)
                self.logger[].log_scalar("mean_target", mean_tgt, step)
                self.logger[].log_scalar("mean_reward", mean_rew, step)
                self.logger[].log_scalar("mean_next_q", mean_nq, step)
                self.logger[].log_scalar("mean_done", mean_done, step)
                self.logger[].log_scalar("mean_abs_action", mean_abs_act, step)
                # Read alpha from GPU (not self.alpha — that's only synced at
                # print boundaries by the training loop). Uses caller-supplied
                # pre-allocated alpha_host to avoid per-diag allocation.
                comptime if Self.Config.ActorLoss.HAS_ALPHA:
                    ctx.enqueue_copy(alpha_host, gpu_state.gpu_scalars)
                    ctx.synchronize()
                    self.logger[].log_scalar(
                        "alpha", Float64(alpha_host[0]), step
                    )
                else:
                    self.logger[].log_scalar("alpha", self.alpha, step)
            except:
                pass

    def do_gpu_train_step_real_only(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises:
        """GPU SAC training step on real data only (no synthetic).

        Used before synthetic data is available. Samples full BATCH
        from gpu_state.buffer (real), then runs the standard SAC
        critic + actor update.
        """
        self.train_step_count += 1
        self.update_count += 1

        comptime BS = Self.BATCH
        comptime incr_k = increment_rng_counter_kernel
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.rng_counter.unsafe_ptr())
        ctx.enqueue_function[incr_k, incr_k](
            rng_t, grid_dim=(1,), block_dim=(1,),
        )
        gpu_state.buffer.sample[BS](
            ctx,
            rng_counter=gpu_state.rng_counter,
            sampled_obs=gpu_state.s_obs,
            sampled_actions=gpu_state.s_act,
            sampled_rewards=gpu_state.s_rew,
            sampled_next_obs=gpu_state.s_nobs,
            sampled_dones=gpu_state.s_done,
            indices=gpu_state.s_idx,
        )
        # Run SAC update on the sampled batch (same code as _gpu_train_kernels
        # post-sampling section)
        self._gpu_sac_update(ctx, gpu_state)

    def do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        synth_buffer: GPUReplayBuffer[
            Self.Config.SYNTH_CAPACITY, Self.Config.obs_dim, Self.Config.action_dim
        ],
        s_real_idx: DeviceBuffer[DType.int32],
        s_synth_idx: DeviceBuffer[DType.int32],
        alpha_host: HostBuffer[dtype],
    ) raises:
        """GPU SAC training step with CPU bookkeeping + diagnostics.

        For CUDA graph capture, use _gpu_train_kernels() instead (pure GPU,
        no CPU counters or D2H copies). Call _gpu_train_diagnostics()
        periodically outside the graph for metrics logging.

        `alpha_host` is a size-1 host buffer reused by the diagnostics.
        """
        self._gpu_train_kernels(
            ctx, gpu_state, synth_buffer, s_real_idx, s_synth_idx
        )
        self._gpu_train_diagnostics(ctx, gpu_state, 1, alpha_host)

    def soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises:
        if Self.Config.Schedule.should_update_targets(
            self.update_count, self.policy_delay
        ):
            gpu_state.critics.soft_update_all(self.tau, ctx)

    # =========================================================================
    # GPU train method — GPU env + GPU SAC + CPU dynamics
    # =========================================================================

    def train_gpu[
        E: GPUContinuousEnv,
        USE_CUDA_GRAPH: Bool = False,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train MBPO with GPU-batched environment stepping + GPU SAC.

        Args:
            ctx: GPU device context.
            num_steps: Total env transitions.
            warmup_steps: Transitions before training (default: 5000).
            verbose: Print progress (default: False).
            print_every: Print interval in transitions (default: 50000).
            environment_name: Name for metrics.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N train steps (default: 0).

        Returns:
            TrainingMetrics with episode-level statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType()
        var metrics = run_mbpo_train_gpu[
            E,
            Self.Config,
            Self.L,
            USE_CUDA_GRAPH,
            Self.TRAIN_N_ENVS,
            Self.REAL_RATIO_PCT,
        ](
            self,
            cpu_state,
            ctx,
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            logger=logger,
        )
        self.state = cpu_state^
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^

    # =========================================================================
    # Checkpointable — saves agent hyperparameters and training state.
    # Network weights require save_cpu_state(cpu_state, path) separately
    # because the Checkpointable trait doesn't include state access.
    # =========================================================================

    def save_checkpoint(self, path: String) raises -> None:
        var content = write_checkpoint_header(
            "mbpo",
            Self.Config.ActorModel.PARAM_SIZE
            + Self.Config.CriticModel.PARAM_SIZE * Self.Config.NUM_CRITICS
            + Self.Config.DynamicsModel.PARAM_SIZE * Self.Config.ENSEMBLE_SIZE,
            0,
        )
        content += self.state.actor.write_sections("actor_")
        content += self.state.critics.pairs[0].write_sections("critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            content += self.state.critics.pairs[1].write_sections("critic2_")
        for m in range(Self.Config.ENSEMBLE_SIZE):
            var prefix = "dyn" + String(m) + "_"
            content += self.state.dynamics.members[m].write_sections(prefix)

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("action_scale=" + String(self.action_scale))
        metadata.append("alpha=" + String(self.alpha))
        metadata.append("log_alpha=" + String(self.log_alpha))
        metadata.append("alpha_adam_t=" + String(self.alpha_adam_t))
        metadata.append("update_count=" + String(self.update_count))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("train_step_count=" + String(self.train_step_count))
        metadata.append("rollout_length=" + String(self.rollout_length))
        var elite_str = String("")
        for i in range(len(self.state.dynamics.elite_indices)):
            if i > 0:
                elite_str += ","
            elite_str += String(self.state.dynamics.elite_indices[i])
        metadata.append("elite_indices=" + elite_str)
        content += write_metadata_section(metadata)
        save_checkpoint_file(path, content)

    def load_checkpoint(mut self, path: String) raises -> None:
        var content = read_checkpoint_file(path)
        self.state.actor.read_sections(content, "actor_")
        self.state.critics.pairs[0].read_sections(content, "critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            self.state.critics.pairs[1].read_sections(content, "critic2_")
        for m in range(Self.Config.ENSEMBLE_SIZE):
            var prefix = "dyn" + String(m) + "_"
            self.state.dynamics.members[m].read_sections(content, prefix)

        var metadata = read_metadata_section(content)
        set_metadata_value_float(metadata, "gamma", self.gamma)
        set_metadata_value_float(metadata, "tau", self.tau)
        set_metadata_value_float(metadata, "action_scale", self.action_scale)
        set_metadata_value_float(metadata, "alpha", self.alpha)
        set_metadata_value_float(metadata, "log_alpha", self.log_alpha)
        set_metadata_value_int(metadata, "alpha_adam_t", self.alpha_adam_t)
        set_metadata_value_int(metadata, "update_count", self.update_count)
        set_metadata_value_int(metadata, "total_steps", self.total_steps)
        set_metadata_value_int(
            metadata, "train_step_count", self.train_step_count
        )
        set_metadata_value_int(metadata, "rollout_length", self.rollout_length)

    def save_cpu_state(self, cpu_state: Self.CPUStateType, path: String) raises:
        """Save network weights and optimizer state from cpu_state.

        Saves actor (online+target), critic(s) (online+target), and dynamics
        ensemble params and optimizer states. Replay buffers are NOT saved.
        """

        var content = write_checkpoint_header(
            "mbpo_state",
            Self.Config.ActorModel.PARAM_SIZE
            + Self.Config.CriticModel.PARAM_SIZE * Self.Config.NUM_CRITICS
            + Self.Config.DynamicsModel.PARAM_SIZE * Self.Config.ENSEMBLE_SIZE,
            0,
        )
        content += cpu_state.actor.write_sections("actor_")
        content += cpu_state.critics.pairs[0].write_sections("critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            content += cpu_state.critics.pairs[1].write_sections("critic2_")
        for m in range(Self.Config.ENSEMBLE_SIZE):
            var prefix = "dyn" + String(m) + "_"
            content += cpu_state.dynamics.members[m].write_sections(prefix)
        save_checkpoint_file(path, content)

    def load_cpu_state(
        self, mut cpu_state: Self.CPUStateType, path: String
    ) raises:
        """Load network weights and optimizer state into cpu_state."""

        var content = read_checkpoint_file(path)
        cpu_state.actor.read_sections(content, "actor_")
        cpu_state.critics.pairs[0].read_sections(content, "critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            cpu_state.critics.pairs[1].read_sections(content, "critic2_")
        for m in range(Self.Config.ENSEMBLE_SIZE):
            var prefix = "dyn" + String(m) + "_"
            cpu_state.dynamics.members[m].read_sections(content, prefix)
