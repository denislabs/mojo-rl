"""Generic on-policy agent parameterized by OnPolicyConfig.

Supports PPO (clipped surrogate, multi-epoch) and A2C (vanilla PG, single pass)
via comptime if branching on Config.IS_PPO.
"""

from std.math import exp, log
from std.random import random_float64
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.core import (
    OnPolicyDiscreteState,
    OnPolicyDiscreteAgent,
    Checkpointable,
)
from mojo_rl.deep_agents.core.onpolicy_helpers import (
    compute_gae_list,
    normalize_advantages_list,
    fisher_yates_shuffle,
)
from mojo_rl.core import TrainingMetrics, BoxDiscreteActionEnv
from mojo_rl.core.utils.softmax import (
    softmax_inline,
    sample_from_probs_inline,
    argmax_probs_inline,
)

from .onpolicy_config import OnPolicyConfig


# =============================================================================
# Generic On-Policy CPU State
# =============================================================================


struct GenericOnPolicyCPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    num_actions: Int,
    rollout_len: Int,
](Movable, OnPolicyDiscreteState):
    """CPU state for discrete on-policy agents (PPO, A2C)."""

    comptime OBS = Self.ActorModel.IN_DIM
    comptime ACTIONS = Self.ActorModel.OUT_DIM
    comptime ROLLOUT = Self.rollout_len
    comptime CRITIC_IN = Self.CriticModel.IN_DIM
    comptime CRITIC_OUT = Self.CriticModel.OUT_DIM

    # Networks
    var actor: NetworkState[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkState[Self.CriticModel, Self.CriticOpt]

    # Rollout buffers
    var buffer_obs: List[Scalar[dtype]]
    var buffer_actions: List[Int]
    var buffer_rewards: List[Scalar[dtype]]
    var buffer_values: List[Scalar[dtype]]
    var buffer_log_probs: List[Scalar[dtype]]
    var buffer_dones: List[Bool]
    var buffer_idx: Int

    # Computed by compute_advantages
    var _advantages: List[Scalar[dtype]]
    var _returns: List[Scalar[dtype]]

    # Bootstrapping state
    var _current_obs: List[Scalar[dtype]]
    var _env_initialized: Bool

    # Minibatch indices
    var _indices: List[Int]

    fn __init__(out self):
        self.actor = NetworkState[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier[]]()
        self.critic = NetworkState[Self.CriticModel, Self.CriticOpt]()
        self.critic.initialize[Xavier[]]()

        # Rollout buffers
        self.buffer_obs = List[Scalar[dtype]](capacity=Self.ROLLOUT * Self.OBS)
        self.buffer_actions = List[Int](capacity=Self.ROLLOUT)
        self.buffer_rewards = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        self.buffer_values = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        self.buffer_log_probs = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        self.buffer_dones = List[Bool](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT * Self.OBS):
            self.buffer_obs.append(Scalar[dtype](0))
        for _ in range(Self.ROLLOUT):
            self.buffer_actions.append(0)
            self.buffer_rewards.append(Scalar[dtype](0))
            self.buffer_values.append(Scalar[dtype](0))
            self.buffer_log_probs.append(Scalar[dtype](0))
            self.buffer_dones.append(False)
        self.buffer_idx = 0

        self._advantages = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        self._returns = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._advantages.append(Scalar[dtype](0))
            self._returns.append(Scalar[dtype](0))

        self._current_obs = List[Scalar[dtype]](capacity=Self.OBS)
        for _ in range(Self.OBS):
            self._current_obs.append(Scalar[dtype](0))
        self._env_initialized = False

        self._indices = List[Int](capacity=Self.ROLLOUT)
        for i in range(Self.ROLLOUT):
            self._indices.append(i)

    # OnPolicyDiscreteState trait
    fn store_step(
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        value: Scalar[dtype],
        log_prob: Scalar[dtype],
        done: Bool,
    ) -> None:
        var idx = self.buffer_idx
        for i in range(Self.OBS):
            self.buffer_obs[idx * Self.OBS + i] = obs[i]
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = Scalar[dtype](reward)
        self.buffer_values[idx] = value
        self.buffer_log_probs[idx] = log_prob
        self.buffer_dones[idx] = done
        self.buffer_idx += 1

    fn is_full(self) -> Bool:
        return self.buffer_idx >= Self.ROLLOUT

    fn clear(mut self) -> None:
        self.buffer_idx = 0


# =============================================================================
# GenericOnPolicyAgent[Config: OnPolicyConfig]
# =============================================================================


struct GenericOnPolicyAgent[
    Config: OnPolicyConfig,
](OnPolicyDiscreteAgent & Checkpointable):
    """Generic on-policy agent. PPO vs A2C via Config.IS_PPO."""

    # Derive ALL dimensions from Model types for unification consistency
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.ActorModel.OUT_DIM  # Must match ActorModel.OUT_DIM
    comptime ROLLOUT: Int = Self.Config.rollout_len
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM  # Should == OBS
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]

    comptime CPUStateType = GenericOnPolicyCPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.rollout_len,
    ]

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

    # Checkpoint
    var checkpoint_every: Int
    var checkpoint_path: String

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
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    # =========================================================================
    # OnPolicyDiscreteAgent trait
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn collect_rollout[
        E: BoxDiscreteActionEnv
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

            # Actor forward → logits
            var logits_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            var logits_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](logits_arr.unsafe_ptr())
            var p_a = cpu_state.actor.params_view()
            Self.ActorNet.forward[1](obs_t, logits_t, p_a)

            # Softmax → probs → sample
            var probs = softmax_inline[dtype, Self.ACTIONS](logits_arr)
            var action = sample_from_probs_inline[dtype, Self.ACTIONS](probs)
            var log_prob = log(probs[action] + Scalar[dtype](1e-8))

            # Critic forward → value (use CRITIC_IN for input dim)
            var c_obs_arr = InlineArray[Scalar[dtype], Self.CRITIC_IN](
                uninitialized=True
            )
            for ci in range(Self.CRITIC_IN):
                c_obs_arr[ci] = obs_arr[ci]
            var c_obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
            ](c_obs_arr.unsafe_ptr())
            var val_arr = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                uninitialized=True
            )
            var val_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.CRITIC_OUT), MutAnyOrigin
            ](val_arr.unsafe_ptr())
            var p_c = cpu_state.critic.params_view()
            Self.CriticNet.forward[1](c_obs_t, val_t, p_c)
            var value = val_arr[0]

            # Step env
            var result = env.step_obs(action)
            var reward = Float64(result[1])
            var done = result[2]

            # Store in buffer
            var idx = cpu_state.buffer_idx
            for i in range(Self.OBS):
                cpu_state.buffer_obs[idx * Self.OBS + i] = obs_arr[i]
            cpu_state.buffer_actions[idx] = action
            cpu_state.buffer_rewards[idx] = Scalar[dtype](reward)
            cpu_state.buffer_values[idx] = value
            cpu_state.buffer_log_probs[idx] = log_prob
            cpu_state.buffer_dones[idx] = done
            cpu_state.buffer_idx += 1

            # Update current obs
            if done:
                var next_obs = env.reset_obs_list()
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](next_obs[i])
            else:
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](result[0][i])

    fn compute_advantages(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> None:
        var buf_len = cpu_state.buffer_idx
        if buf_len == 0:
            return

        # Bootstrap value from current obs (use CRITIC_IN for input)
        var c_obs_arr = InlineArray[Scalar[dtype], Self.CRITIC_IN](
            uninitialized=True
        )
        for i in range(Self.CRITIC_IN):
            c_obs_arr[i] = cpu_state._current_obs[i]
        var c_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
        ](c_obs_arr.unsafe_ptr())
        var val_arr = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
            uninitialized=True
        )
        var val_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.CRITIC_OUT), MutAnyOrigin
        ](val_arr.unsafe_ptr())
        var p_c = cpu_state.critic.params_view()
        Self.CriticNet.forward[1](c_obs_t, val_t, p_c)
        var next_value = val_arr[0]

        compute_gae_list[dtype](
            cpu_state.buffer_rewards,
            cpu_state.buffer_values,
            cpu_state.buffer_dones,
            next_value,
            buf_len,
            self.gamma,
            self.gae_lambda,
            cpu_state._advantages,
            cpu_state._returns,
        )

        if self.normalize_advantages and buf_len > 1:
            normalize_advantages_list[dtype](cpu_state._advantages, buf_len)

    fn update_epochs(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
        var buf_len = cpu_state.buffer_idx
        if buf_len == 0:
            return 0.0

        # Determine number of epochs and minibatch size
        var n_epochs = 1
        var mb_size = buf_len  # A2C: single pass, full rollout
        comptime if Self.Config.IS_PPO:
            n_epochs = self.num_epochs
            mb_size = self.minibatch_size

        var total_loss: Float64 = 0.0
        var sample_count = 0

        for epoch in range(n_epochs):
            # PPO: shuffle indices each epoch
            comptime if Self.Config.IS_PPO:
                fisher_yates_shuffle(cpu_state._indices, buf_len)

            var batch_start = 0
            while batch_start < buf_len:
                var batch_end = batch_start + mb_size
                if batch_end > buf_len:
                    batch_end = buf_len
                var this_mb = batch_end - batch_start

                # Per-minibatch advantage normalization (PPO)
                comptime if Self.Config.IS_PPO:
                    if self.norm_adv_per_minibatch and this_mb > 1:
                        # Gather and normalize
                        var mb_adv = List[Scalar[dtype]](capacity=this_mb)
                        for b in range(batch_start, batch_end):
                            var t = cpu_state._indices[b]
                            mb_adv.append(cpu_state._advantages[t])
                        normalize_advantages_list[dtype](mb_adv, this_mb)
                        for b in range(this_mb):
                            var t = cpu_state._indices[batch_start + b]
                            cpu_state._advantages[t] = mb_adv[b]

                # Process each sample in the minibatch
                for b_idx in range(batch_start, batch_end):
                    var t = b_idx
                    comptime if Self.Config.IS_PPO:
                        t = cpu_state._indices[b_idx]

                    var old_log_prob = cpu_state.buffer_log_probs[t]
                    var advantage = cpu_state._advantages[t]
                    var return_t = cpu_state._returns[t]
                    var action = cpu_state.buffer_actions[t]
                    var old_value = cpu_state.buffer_values[t]

                    # Actor forward
                    var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                        uninitialized=True
                    )
                    for i in range(Self.OBS):
                        obs_arr[i] = cpu_state.buffer_obs[t * Self.OBS + i]
                    var obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](obs_arr.unsafe_ptr())

                    var logits_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    var logits_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](logits_arr.unsafe_ptr())
                    var actor_cache = InlineArray[
                        Scalar[dtype], Self.ACTOR_CS
                    ](uninitialized=True)
                    var actor_cache_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTOR_CS), MutAnyOrigin
                    ](actor_cache.unsafe_ptr())
                    var p_a = cpu_state.actor.params_view()
                    Self.ActorNet.forward_with_cache[1](
                        obs_t, logits_t, p_a, actor_cache_t
                    )

                    var probs = softmax_inline[dtype, Self.ACTIONS](logits_arr)
                    var new_log_prob = log(
                        probs[action] + Scalar[dtype](1e-8)
                    )

                    # Entropy
                    var entropy = Scalar[dtype](0.0)
                    for a in range(Self.ACTIONS):
                        if probs[a] > Scalar[dtype](1e-8):
                            entropy -= probs[a] * log(probs[a])

                    # Policy gradient
                    var d_logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )

                    comptime if Self.Config.IS_PPO:
                        # PPO: clipped surrogate
                        var ratio = exp(new_log_prob - old_log_prob)
                        var surr1 = ratio * advantage
                        var clipped_ratio: Scalar[dtype]
                        if advantage >= Scalar[dtype](0.0):
                            clipped_ratio = min(
                                ratio,
                                Scalar[dtype](1.0 + self.clip_epsilon),
                            )
                        else:
                            clipped_ratio = max(
                                ratio,
                                Scalar[dtype](1.0 - self.clip_epsilon),
                            )
                        var surr2 = clipped_ratio * advantage
                        var use_surr1 = surr1 < surr2

                        var is_clipped = (
                            ratio
                            < Scalar[dtype](1.0 - self.clip_epsilon)
                        ) or (
                            ratio
                            > Scalar[dtype](1.0 + self.clip_epsilon)
                        )

                        for a in range(Self.ACTIONS):
                            var d_lp = Scalar[dtype](0.0)
                            if a == action:
                                d_lp = Scalar[dtype](1.0) - probs[a]
                            else:
                                d_lp = -probs[a]
                            var d_ent = -probs[a] * (
                                Scalar[dtype](1.0)
                                + log(probs[a] + Scalar[dtype](1e-8))
                            )
                            if is_clipped:
                                d_logits[a] = -Scalar[dtype](self.entropy_coef) * d_ent
                            else:
                                var effective_adv = advantage
                                if not use_surr1:
                                    effective_adv = advantage
                                d_logits[a] = (
                                    -effective_adv * ratio * d_lp
                                    - Scalar[dtype](self.entropy_coef) * d_ent
                                )

                    comptime if not Self.Config.IS_PPO:
                        # A2C: vanilla policy gradient
                        for a in range(Self.ACTIONS):
                            var d_lp = Scalar[dtype](0.0)
                            if a == action:
                                d_lp = Scalar[dtype](1.0) - probs[a]
                            else:
                                d_lp = -probs[a]
                            var d_ent = -probs[a] * (
                                Scalar[dtype](1.0)
                                + log(probs[a] + Scalar[dtype](1e-8))
                            )
                            d_logits[a] = (
                                -advantage * d_lp
                                - Scalar[dtype](self.entropy_coef) * d_ent
                            )

                    # Actor backward
                    var d_logits_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](d_logits.unsafe_ptr())
                    var d_obs = InlineArray[Scalar[dtype], Self.OBS](
                        uninitialized=True
                    )
                    var d_obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](d_obs.unsafe_ptr())
                    var g_a = cpu_state.actor.grads_view()
                    Self.ActorNet.backward[1](
                        d_logits_t, d_obs_t, p_a, actor_cache_t, g_a
                    )
                    cpu_state.actor.optimizer_step()

                    # Critic forward + backward (use CRITIC_IN for input)
                    var c_obs = InlineArray[Scalar[dtype], Self.CRITIC_IN](
                        uninitialized=True
                    )
                    for ci in range(Self.CRITIC_IN):
                        c_obs[ci] = obs_arr[ci]
                    var c_obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
                    ](c_obs.unsafe_ptr())
                    var val_out = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                        uninitialized=True
                    )
                    var val_out_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_OUT),
                        MutAnyOrigin,
                    ](val_out.unsafe_ptr())
                    var critic_cache = InlineArray[
                        Scalar[dtype], Self.CRITIC_CS
                    ](uninitialized=True)
                    var critic_cache_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.CRITIC_CS), MutAnyOrigin
                    ](critic_cache.unsafe_ptr())
                    var p_c = cpu_state.critic.params_view()
                    Self.CriticNet.forward_with_cache[1](
                        c_obs_t, val_out_t, p_c, critic_cache_t
                    )
                    var value = val_out[0]

                    var d_value = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                        uninitialized=True
                    )
                    d_value[0] = (
                        Scalar[dtype](2.0)
                        * Scalar[dtype](self.value_loss_coef)
                        * (value - return_t)
                    )
                    var d_value_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_OUT),
                        MutAnyOrigin,
                    ](d_value.unsafe_ptr())
                    var d_obs_c = InlineArray[Scalar[dtype], Self.CRITIC_IN](
                        uninitialized=True
                    )
                    var d_obs_c_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
                    ](d_obs_c.unsafe_ptr())
                    var g_c = cpu_state.critic.grads_view()
                    Self.CriticNet.backward[1](
                        d_value_t, d_obs_c_t, p_c, critic_cache_t, g_c
                    )
                    cpu_state.critic.optimizer_step()

                    total_loss += Float64(-new_log_prob * advantage)
                    sample_count += 1

                batch_start = batch_end

        self.train_step_count += 1
        if sample_count > 0:
            return total_loss / Float64(sample_count)
        return 0.0

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
            uninitialized=True
        )
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[dtype](obs[i])
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var logits_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](logits_arr.unsafe_ptr())
        var p = cpu_state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, logits_t, p)

        var probs = softmax_inline[dtype, Self.ACTIONS](logits_arr)
        var action = argmax_probs_inline[dtype, Self.ACTIONS](probs)

        var result = List[Float64](capacity=1)
        result.append(Float64(action))
        return result^

    fn get_explore_rate(self) -> Float64:
        return self.entropy_coef

    # Checkpointable
    fn save_checkpoint(self, path: String) raises -> None:
        pass

    fn load_checkpoint(mut self, path: String) raises -> None:
        pass

    # Convenience
    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self, mut env: E, num_updates: Int = 1000
    ) raises -> TrainingMetrics:
        from mojo_rl.deep_agents.core.onpolicy_train import (
            run_onpolicy_discrete_train,
        )

        var cpu_state = self.make_cpu_state()
        var ckpt_path = String(self.checkpoint_path)
        return run_onpolicy_discrete_train(
            self,
            cpu_state,
            env,
            num_updates,
            checkpoint_every=self.checkpoint_every,
            checkpoint_path=ckpt_path,
        )
