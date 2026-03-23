"""Rainbow DQN agent variant using host-memory replay buffer.

Identical to GenericRainbowAgent but uses RainbowGPUStateHost (host-memory PER)
instead of RainbowGPUState (GPU-memory PER). Only implements GPUOffPolicyAgent.
"""

from std.math import exp, log, floor, ceil
from std.random import random_float64
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from mojo_rl.deep_agents.core import (
    run_offpolicy_discrete_train_gpu,
    PerfTimer,
)
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Parallel,
    NoisyLinear,
    NoisyLinearReLU,
    Conv2DReLU,
    FlattenLayer,
)
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import (
    Network,
    NetworkState,
    GPUNetworkState,
)
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.loss.two_hot import compute_bins

from mojo_rl.deep_agents.core import (
    OffPolicyDiscreteState,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    Checkpointable,
)
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.deep_agents.core.replay import (
    PrioritizedReplayBuffer,
    GPUPrioritizedReplayBuffer,
    HostPrioritizedReplayBuffer,
    NStepBuffer,
    NStepTransition,
    GPUNStepBuffer,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.eval import run_offpolicy_discrete_eval

from .rainbow_agent import (
    RainbowDQNConfig,
    RainbowGPUStateHost,
    _dueling_dist_combine,
    _dueling_dist_grad_reverse,
    _rainbow_expected_q,
    RainbowCPUState,
)


# =============================================================================
# GenericRainbowHostAgent
# =============================================================================


struct GenericRainbowHostAgent[
    Config: RainbowDQNConfig,
    n_envs: Int = 256,
    L: Logger = NoOpLogger,
](GPUOffPolicyAgent & Checkpointable):
    """Rainbow DQN with host-memory replay buffer.

    Identical to GenericRainbowAgent but uses RainbowGPUStateHost for its
    GPUStateType, enabling much larger buffer capacities for large-observation
    environments (e.g., pixel-based).

    Parameters:
        Config: RainbowConfig with architecture and distributional params.
        n_envs: Parallel environments for GPU training.
        L: Logger type.
    """

    comptime OBS: Int = Self.Config.QModel.IN_DIM
    comptime RAW_OUT: Int = Self.Config.QModel.OUT_DIM
    comptime ACTIONS: Int = Self.Config.num_actions
    comptime NUM_ATOMS: Int = Self.Config.num_atoms
    comptime BATCH: Int = Self.Config.batch_size
    comptime Q_CS: Int = Self.Config.QModel.CACHE_SIZE
    comptime QNet = Network[Self.Config.QModel, Self.Config.QOpt]
    comptime COMBINED: Int = Self.ACTIONS * Self.NUM_ATOMS

    comptime CPUStateType = RainbowCPUState[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.batch_size,
        Self.Config.num_atoms,
        Self.Config.n_step,
        Self.Config.v_min,
        Self.Config.v_max,
    ]

    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = RainbowGPUStateHost[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.num_atoms,
        Self.Config.n_step,
        Self.Config.batch_size,
        Self.n_envs,
    ]

    var state: Self.CPUStateType
    var gamma: Float64
    var tau: Float64
    var target_update_freq: Int
    var train_step_count: Int
    var target_total_steps: Int
    var _target_update_ctr: Int
    var checkpoint_every: Int
    var checkpoint_path: String
    var beta_start: Float64
    var beta_frames: Int
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    def __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 1.0,
        target_update_freq: Int = 500,
        alpha: Float64 = 0.5,
        beta: Float64 = 0.4,
        beta_frames: Int = 100000,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        target_total_steps: Int = 0,
    ):
        self.state = Self.CPUStateType(alpha=alpha, beta=beta, gamma=gamma)
        self.gamma = gamma
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self._target_update_ctr = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.beta_start = beta
        self.beta_frames = beta_frames
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    # =========================================================================
    # GPUOffPolicyAgent trait
    # =========================================================================

    def get_action_scale(self) -> Float64:
        return 1.0

    def get_total_steps(self) -> Int:
        return self.train_step_count

    def set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    def make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx, gamma=self.gamma)

    def upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)
        var bins_host = HostBuffer[dtype](ctx, Self.NUM_ATOMS)
        for i in range(Self.NUM_ATOMS):
            bins_host[i] = Scalar[dtype](self.state.bins[i])
        ctx.enqueue_copy(gpu_state.bins_buf, bins_host)

    def download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.download_to(self.state.online, ctx)
        gpu_state.target.download_to(self.state.target, ctx)

    def select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Noisy forward -> dueling combine -> expected Q -> argmax (no epsilon).
        """
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.env_raw_buf.unsafe_ptr())
        var p = gpu_state.online.params_view()
        Self.QNet.forward_gpu[N_ENVS](ctx, obs_t, raw_t, p, gpu_state.inf_ws)

        # Dueling combine + expected Q + argmax in one kernel
        var q_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.env_q_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var bins_t = LayoutTensor[
            dtype, Layout.row_major(Self.NUM_ATOMS), MutAnyOrigin
        ](gpu_state.bins_buf.unsafe_ptr())

        @always_inline
        def rainbow_select_kernel(
            raw: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
            ],
            eq: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            bins: LayoutTensor[
                dtype, Layout.row_major(Self.NUM_ATOMS), MutAnyOrigin
            ],
            acts: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= N_ENVS:
                return

            # Dueling combine + expected Q for each action
            for a in range(Self.ACTIONS):
                # Combine V + A - mean(A) per atom, then compute expected Q
                var max_val = Scalar[dtype](-1e10)
                for i in range(Self.NUM_ATOMS):
                    var v_i = rebind[Scalar[dtype]](raw[e, i])
                    var mean_a = Scalar[dtype](0)
                    for a2 in range(Self.ACTIONS):
                        mean_a += rebind[Scalar[dtype]](
                            raw[e, Self.NUM_ATOMS + a2 * Self.NUM_ATOMS + i]
                        )
                    mean_a /= Scalar[dtype](Self.ACTIONS)
                    var q_ai = (
                        v_i
                        + rebind[Scalar[dtype]](
                            raw[e, Self.NUM_ATOMS + a * Self.NUM_ATOMS + i]
                        )
                        - mean_a
                    )
                    if q_ai > max_val:
                        max_val = q_ai

                var sum_exp = Scalar[dtype](0)
                var expected = Scalar[dtype](0)
                for i in range(Self.NUM_ATOMS):
                    var v_i = rebind[Scalar[dtype]](raw[e, i])
                    var mean_a = Scalar[dtype](0)
                    for a2 in range(Self.ACTIONS):
                        mean_a += rebind[Scalar[dtype]](
                            raw[e, Self.NUM_ATOMS + a2 * Self.NUM_ATOMS + i]
                        )
                    mean_a /= Scalar[dtype](Self.ACTIONS)
                    var q_ai = (
                        v_i
                        + rebind[Scalar[dtype]](
                            raw[e, Self.NUM_ATOMS + a * Self.NUM_ATOMS + i]
                        )
                        - mean_a
                    )
                    var e_val = exp(q_ai - max_val)
                    sum_exp += e_val
                    expected += e_val * rebind[Scalar[dtype]](bins[i])
                eq[e, a] = expected / sum_exp

            # Argmax (no epsilon)
            var best_q = rebind[Scalar[dtype]](eq[e, 0])
            var best_action = 0
            for a in range(1, Self.ACTIONS):
                var qv = rebind[Scalar[dtype]](eq[e, a])
                if qv > best_q:
                    best_q = qv
                    best_action = a
            acts[e] = Scalar[dtype](best_action)

        ctx.enqueue_function[rainbow_select_kernel, rainbow_select_kernel](
            raw_t,
            q_t,
            bins_t,
            actions_t,
            grid_dim=((N_ENVS + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    def do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Rainbow GPU training step."""
        comptime BATCH = Self.BATCH
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime ATOMS = Self.NUM_ATOMS
        comptime COMB = Self.COMBINED

        # Beta annealing
        var progress = Scalar[dtype](
            Float64(self.train_step_count) / Float64(max(1, self.beta_frames))
        )
        if progress > Scalar[dtype](1.0):
            progress = Scalar[dtype](1.0)
        gpu_state.buffer.anneal_beta(progress, Scalar[dtype](self.beta_start))

        # ---- Phase 1: PER sample ----
        gpu_state.buffer.sample[BATCH](
            ctx,
            gpu_state.s_obs,
            gpu_state.s_act,
            gpu_state.s_rew,
            gpu_state.s_nobs,
            gpu_state.s_done,
            gpu_state.s_idx,
            gpu_state.s_weights,
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())
        var q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.q_raw.unsafe_ptr())
        var next_q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.next_q_raw.unsafe_ptr())
        var online_next_q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.online_next_q_raw.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Q_CS), MutAnyOrigin
        ](gpu_state.cache.unsafe_ptr())
        var grad_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.grad_raw.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.grad_input.unsafe_ptr())

        # Distributional tensors
        var q_comb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
        ](gpu_state.q_combined.unsafe_ptr())
        var next_comb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
        ](gpu_state.next_q_combined.unsafe_ptr())
        var online_next_comb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
        ](gpu_state.online_next_q_combined.unsafe_ptr())
        var expected_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.expected_q.unsafe_ptr())
        var grad_comb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
        ](gpu_state.grad_combined.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_rew.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_done.unsafe_ptr())
        var weights_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_weights.unsafe_ptr())
        var td_errors_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.td_errors.unsafe_ptr())
        var bins_t = LayoutTensor[dtype, Layout.row_major(ATOMS), MutAnyOrigin](
            gpu_state.bins_buf.unsafe_ptr()
        )

        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()

        # ---- Phase 2: Forward passes ----
        Self.QNet.forward_gpu_with_cache[BATCH](
            ctx, obs_t, q_raw_t, p_online, cache_t, gpu_state.train_ws
        )
        Self.QNet.forward_gpu[BATCH](
            ctx, next_obs_t, next_q_raw_t, p_target, gpu_state.train_ws
        )
        Self.QNet.forward_gpu[BATCH](
            ctx, next_obs_t, online_next_q_raw_t, p_online, gpu_state.train_ws
        )

        # ---- Phase 3: Dueling combine (3 kernels) ----
        @always_inline
        def dueling_combine_kernel(
            raw: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
            ],
            comb: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH:
                return
            var b = idx
            for i in range(ATOMS):
                var v_i = rebind[Scalar[dtype]](raw[b, i])
                var mean_a = Scalar[dtype](0)
                for a in range(Self.ACTIONS):
                    mean_a += rebind[Scalar[dtype]](
                        raw[b, ATOMS + a * ATOMS + i]
                    )
                mean_a /= Scalar[dtype](Self.ACTIONS)
                for a in range(Self.ACTIONS):
                    comb[b, a * ATOMS + i] = (
                        v_i
                        + rebind[Scalar[dtype]](raw[b, ATOMS + a * ATOMS + i])
                        - mean_a
                    )

        ctx.enqueue_function[dueling_combine_kernel, dueling_combine_kernel](
            q_raw_t,
            q_comb_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[dueling_combine_kernel, dueling_combine_kernel](
            next_q_raw_t,
            next_comb_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[dueling_combine_kernel, dueling_combine_kernel](
            online_next_q_raw_t,
            online_next_comb_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 4: Expected Q from online-next (for Double DQN action selection) ----
        @always_inline
        def expected_q_kernel(
            comb: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
            bins: LayoutTensor[dtype, Layout.row_major(ATOMS), MutAnyOrigin],
            eq: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            for a in range(Self.ACTIONS):
                var base = a * ATOMS
                var max_val = rebind[Scalar[dtype]](comb[b, base])
                for i in range(1, ATOMS):
                    var v = rebind[Scalar[dtype]](comb[b, base + i])
                    if v > max_val:
                        max_val = v
                var sum_exp = Scalar[dtype](0)
                for i in range(ATOMS):
                    sum_exp += exp(
                        rebind[Scalar[dtype]](comb[b, base + i]) - max_val
                    )
                var expected = Scalar[dtype](0)
                for i in range(ATOMS):
                    var prob = (
                        exp(rebind[Scalar[dtype]](comb[b, base + i]) - max_val)
                        / sum_exp
                    )
                    expected += prob * rebind[Scalar[dtype]](bins[i])
                eq[b, a] = expected

        ctx.enqueue_function[expected_q_kernel, expected_q_kernel](
            online_next_comb_t,
            bins_t,
            expected_q_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 5: Bellman projection + IS-weighted CE grad + dueling reverse ----
        var gamma_n_s = Scalar[dtype](self.gamma)
        for _ in range(Self.Config.n_step - 1):
            gamma_n_s *= Scalar[dtype](self.gamma)
        var v_min_s = Scalar[dtype](Self.Config.v_min)
        var v_max_s = Scalar[dtype](Self.Config.v_max)
        var dz_s = Scalar[dtype](
            (Self.Config.v_max - Self.Config.v_min) / Float64(ATOMS - 1)
        )

        @always_inline
        def rainbow_project_grad_kernel(
            online_comb: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
            target_comb: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
            next_eq: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            bins: LayoutTensor[dtype, Layout.row_major(ATOMS), MutAnyOrigin],
            actions: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            weights: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            grad_c: LayoutTensor[
                dtype, Layout.row_major(BATCH, COMB), MutAnyOrigin
            ],
            grad_r: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
            ],
            td_err: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            gamma_n: Scalar[dtype],
            vmin: Scalar[dtype],
            vmax: Scalar[dtype],
            dz: Scalar[dtype],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return

            # Zero gradients
            for i in range(COMB):
                grad_c[b, i] = Scalar[dtype](0)
            for i in range(Self.RAW_OUT):
                grad_r[b, i] = Scalar[dtype](0)

            # 1. Best next action (Double DQN)
            var best_a = 0
            var best_q = rebind[Scalar[dtype]](next_eq[b, 0])
            for a in range(1, Self.ACTIONS):
                var q = rebind[Scalar[dtype]](next_eq[b, a])
                if q > best_q:
                    best_q = q
                    best_a = a

            # 2. Target softmax for best action
            var t_base = best_a * ATOMS
            var t_max = rebind[Scalar[dtype]](target_comb[b, t_base])
            for i in range(1, ATOMS):
                var v = rebind[Scalar[dtype]](target_comb[b, t_base + i])
                if v > t_max:
                    t_max = v
            var t_sum_exp = Scalar[dtype](0)
            for i in range(ATOMS):
                t_sum_exp += exp(
                    rebind[Scalar[dtype]](target_comb[b, t_base + i]) - t_max
                )

            # 3. Bellman projection with gamma^n
            var projected = InlineArray[Scalar[dtype], ATOMS](
                fill=Scalar[dtype](0)
            )
            var r = rebind[Scalar[dtype]](rewards[b])
            var dm = Scalar[dtype](1.0) - rebind[Scalar[dtype]](dones[b])
            for j in range(ATOMS):
                var t_prob = (
                    exp(
                        rebind[Scalar[dtype]](target_comb[b, t_base + j])
                        - t_max
                    )
                    / t_sum_exp
                )
                var tz = r + gamma_n * rebind[Scalar[dtype]](bins[j]) * dm
                if tz < vmin:
                    tz = vmin
                if tz > vmax:
                    tz = vmax
                var bj = (tz - vmin) / dz
                var l_idx = Int(floor(bj))
                var u_idx = Int(ceil(bj))
                if l_idx == u_idx:
                    projected[l_idx] = projected[l_idx] + t_prob
                else:
                    var u_w = bj - Scalar[dtype](l_idx)
                    projected[l_idx] = projected[l_idx] + t_prob * (
                        Scalar[dtype](1.0) - u_w
                    )
                    projected[u_idx] = projected[u_idx] + t_prob * u_w

            # 4. IS-weighted CE gradient for taken action
            var action = Int(rebind[Scalar[dtype]](actions[b]))
            var p_base = action * ATOMS
            var p_max = rebind[Scalar[dtype]](online_comb[b, p_base])
            for i in range(1, ATOMS):
                var v = rebind[Scalar[dtype]](online_comb[b, p_base + i])
                if v > p_max:
                    p_max = v
            var p_sum_exp = Scalar[dtype](0)
            for i in range(ATOMS):
                p_sum_exp += exp(
                    rebind[Scalar[dtype]](online_comb[b, p_base + i]) - p_max
                )
            var log_sum_exp = p_max + log(p_sum_exp)

            # CE loss for priority
            var sample_loss = Scalar[dtype](0)
            for i in range(ATOMS):
                var log_sm = (
                    rebind[Scalar[dtype]](online_comb[b, p_base + i])
                    - log_sum_exp
                )
                sample_loss = sample_loss - projected[i] * log_sm
            td_err[b] = sample_loss

            var weight = rebind[Scalar[dtype]](weights[b])
            for i in range(ATOMS):
                var sm = (
                    exp(
                        rebind[Scalar[dtype]](online_comb[b, p_base + i])
                        - p_max
                    )
                    / p_sum_exp
                )
                grad_c[b, p_base + i] = (
                    weight * (sm - projected[i]) / Scalar[dtype](BATCH)
                )

            # 5. Dueling gradient reverse (combined -> raw)
            for i in range(ATOMS):
                var sum_dq = Scalar[dtype](0)
                for a in range(Self.ACTIONS):
                    sum_dq += rebind[Scalar[dtype]](grad_c[b, a * ATOMS + i])
                grad_r[b, i] = sum_dq  # dV
                var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](
                    Self.ACTIONS
                )
                for a in range(Self.ACTIONS):
                    grad_r[b, ATOMS + a * ATOMS + i] = (
                        rebind[Scalar[dtype]](grad_c[b, a * ATOMS + i])
                        - one_over_n * sum_dq
                    )

        ctx.enqueue_function[
            rainbow_project_grad_kernel, rainbow_project_grad_kernel
        ](
            q_comb_t,
            next_comb_t,
            expected_q_t,
            bins_t,
            actions_t,
            rewards_t,
            dones_t,
            weights_t,
            grad_comb_t,
            grad_raw_t,
            td_errors_t,
            gamma_n_s,
            v_min_s,
            v_max_s,
            dz_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 6: Backward + optimizer ----
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.QNet.backward_gpu[BATCH](
            ctx,
            grad_raw_t,
            grad_in_t,
            p_online,
            cache_t,
            g,
            gpu_state.train_ws,
        )
        gpu_state.online.optimizer_step(ctx)

        self.train_step_count += 1

        # ---- Phase 7: PER priority update ----
        gpu_state.buffer.update_priorities[BATCH](ctx, gpu_state.td_errors)

        # ---- GPU Diagnostic logging ----
        if (
            self.logger
            and self.diag_every > 0
            and self.train_step_count % self.diag_every == 0
        ):
            try:
                # Copy diagnostic data to host
                ctx.enqueue_copy(gpu_state.diag_comb_host, gpu_state.q_combined)
                ctx.enqueue_copy(gpu_state.diag_act_host, gpu_state.s_act)
                ctx.enqueue_copy(gpu_state.diag_rew_host, gpu_state.s_rew)
                ctx.enqueue_copy(gpu_state.diag_done_host, gpu_state.s_done)
                ctx.enqueue_copy(
                    gpu_state.diag_weights_host, gpu_state.s_weights
                )
                ctx.enqueue_copy(gpu_state.diag_td_host, gpu_state.td_errors)
                ctx.synchronize()

                var step = self.train_step_count

                # Compute expected Q from combined logits on host
                var comb_host_arr = InlineArray[Scalar[dtype], BATCH * COMB](
                    uninitialized=True
                )
                for i in range(BATCH * COMB):
                    comb_host_arr[i] = gpu_state.diag_comb_host[i]
                var bins_host = InlineArray[Scalar[dtype], ATOMS](
                    uninitialized=True
                )
                for i in range(ATOMS):
                    bins_host[i] = Scalar[dtype](
                        Self.Config.v_min
                        + Float64(i)
                        * (Self.Config.v_max - Self.Config.v_min)
                        / Float64(ATOMS - 1)
                    )
                var eq_host = InlineArray[Scalar[dtype], BATCH * Self.ACTIONS](
                    uninitialized=True
                )
                _rainbow_expected_q[BATCH, Self.ACTIONS, ATOMS, COMB](
                    comb_host_arr, bins_host, eq_host
                )

                # Q-value stats (expected Q from distributional)
                var q_min = Float64(eq_host[0])
                var q_max = Float64(eq_host[0])
                var q_sum: Float64 = 0.0
                for i in range(BATCH * Self.ACTIONS):
                    var v = Float64(eq_host[i])
                    q_sum += v
                    if v < q_min:
                        q_min = v
                    if v > q_max:
                        q_max = v
                self.logger[].log_scalar(
                    "q_mean",
                    q_sum / Float64(BATCH * Self.ACTIONS),
                    step,
                )
                self.logger[].log_scalar("q_min", q_min, step)
                self.logger[].log_scalar("q_max", q_max, step)

                # Done fraction and reward stats
                var done_count: Float64 = 0.0
                var rew_sum: Float64 = 0.0
                var rew_min = Float64(gpu_state.diag_rew_host[0])
                var rew_max = Float64(gpu_state.diag_rew_host[0])
                for b in range(BATCH):
                    done_count += Float64(gpu_state.diag_done_host[b])
                    var r = Float64(gpu_state.diag_rew_host[b])
                    rew_sum += r
                    if r < rew_min:
                        rew_min = r
                    if r > rew_max:
                        rew_max = r
                self.logger[].log_scalar(
                    "done_fraction",
                    done_count / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar(
                    "reward_mean",
                    rew_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("reward_min", rew_min, step)
                self.logger[].log_scalar("reward_max", rew_max, step)

                # TD error stats (CE loss per sample, used as PER priority)
                var td_err_abs_sum: Float64 = 0.0
                var td_err_max_abs: Float64 = 0.0
                for b in range(BATCH):
                    var abs_err = Float64(gpu_state.diag_td_host[b])
                    if abs_err < 0:
                        abs_err = -abs_err
                    td_err_abs_sum += abs_err
                    if abs_err > td_err_max_abs:
                        td_err_max_abs = abs_err
                self.logger[].log_scalar(
                    "td_error_abs_mean",
                    td_err_abs_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("td_error_max", td_err_max_abs, step)

                # IS weight stats (importance sampling correction)
                var w_min = Float64(gpu_state.diag_weights_host[0])
                var w_max = Float64(gpu_state.diag_weights_host[0])
                var w_sum: Float64 = 0.0
                for b in range(BATCH):
                    var w = Float64(gpu_state.diag_weights_host[b])
                    w_sum += w
                    if w < w_min:
                        w_min = w
                    if w > w_max:
                        w_max = w
                self.logger[].log_scalar(
                    "is_weight_mean",
                    w_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("is_weight_min", w_min, step)
                self.logger[].log_scalar("is_weight_max", w_max, step)

                # PER beta
                var beta_val = Float64(self.beta_start) + (
                    1.0 - Float64(self.beta_start)
                ) * Float64(self.train_step_count) / Float64(
                    max(1, self.beta_frames)
                )
                if beta_val > 1.0:
                    beta_val = 1.0
                self.logger[].log_scalar("per_beta", beta_val, step)

                # Distribution entropy
                var entropy_sum: Float64 = 0.0
                for b in range(BATCH):
                    var action = Int(Float64(gpu_state.diag_act_host[b]))
                    var pred_base = b * COMB + action * ATOMS
                    var pred_max2 = Float64(gpu_state.diag_comb_host[pred_base])
                    for i in range(1, ATOMS):
                        var v = Float64(gpu_state.diag_comb_host[pred_base + i])
                        if v > pred_max2:
                            pred_max2 = v
                    var se2: Float64 = 0.0
                    for i in range(ATOMS):
                        se2 += exp(
                            Float64(gpu_state.diag_comb_host[pred_base + i])
                            - pred_max2
                        )
                    var h: Float64 = 0.0
                    for i in range(ATOMS):
                        var p = (
                            exp(
                                Float64(gpu_state.diag_comb_host[pred_base + i])
                                - pred_max2
                            )
                            / se2
                        )
                        if p > 1e-8:
                            h -= p * log(p)
                    entropy_sum += h
                self.logger[].log_scalar(
                    "dist_entropy_mean",
                    entropy_sum / Float64(BATCH),
                    step,
                )
            except:
                pass

    def soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        if (
            self.train_step_count - self._target_update_ctr
            >= self.target_update_freq
        ):
            gpu_state.target.soft_update_from_gpu(
                gpu_state.online, self.tau, ctx
            )
            self._target_update_ctr = self.train_step_count

    def decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        pass  # No epsilon

    def train_gpu[
        E: GPUDiscreteEnv,
        CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 0,
        sync_every: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        target_total_steps: Int = 0,
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        self.logger = logger
        self.diag_every = diag_every
        self.target_total_steps = target_total_steps
        var timer = PerfTimer[False]()
        var algo_name = Self.Config.NAME
        return run_offpolicy_discrete_train_gpu[
            E, Self, 0, Self.L, CurriculumType
        ](
            self,
            ctx,
            num_steps,
            timer,
            logger=logger,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
            target_total_steps=target_total_steps,
        )

    # =========================================================================
    # Checkpointable (minimal — delegates to CPU state)
    # =========================================================================

    def save_checkpoint(self, path: String) raises:
        """Save checkpoint — not yet implemented for host-buffer agent."""
        pass

    def load_checkpoint(mut self, path: String) raises:
        """Load checkpoint — not yet implemented for host-buffer agent."""
        pass
