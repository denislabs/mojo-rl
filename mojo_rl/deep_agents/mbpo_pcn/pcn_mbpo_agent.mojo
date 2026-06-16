"""MBPO (Model-Based Policy Optimization) agent.

Combines SAC policy learning with a probabilistic dynamics ensemble.
The dynamics model generates synthetic rollouts to augment the real
replay buffer, achieving ~10x better sample efficiency than SAC alone.

Key components:
- DynamicsEnsemble: N probabilistic networks predicting (reward, delta_obs)
- PCNMBPOCPUState: Dual replay buffers (real + synthetic) + SAC networks + ensemble
- PCNMBPOAgent: SAC training with mixed sampling + model training + rollouts
"""

from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.math import cos, exp, log, sqrt
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
from mojo_rl.deep_agents.mbpo_pcn.pcn_mbpo_config import PCNMBPOConfig
# Direct module-path imports: these MBPO/SAC experiment modules are removed from
# the core PCN package surface (experimental/pcn/__init__) during the nn2
# re-architecture. This legacy agent rides out with the sunset sweep.
from mojo_rl.experimental.pcn.pc_dynamics_ensemble_instance import (
    PCDynamicsEnsembleInstanceCPU,
)
from mojo_rl.experimental.pcn.pc_dynamics_ensemble_instance_gpu import (
    PCDynamicsEnsembleInstanceGPU,
)
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
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
    uniform_random_actions_kernel,
)
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)
from std.sys import has_nvidia_gpu_accelerator
from mojo_rl.cuda.graph import CUDAGraph
from mojo_rl.core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
)

# `run_mbpo_train` / `run_mbpo_train_gpu` from core.training.mbpo_train are
# vanilla-MBPO-typed wrappers; PCN-MBPO callers use `agent.train()` /
# `agent.train_gpu()` (defined further below) which dispatch directly to the
# `_run_train_impl` / `_run_train_gpu_impl` bodies on the agent struct.


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
# PCN-specific GPU kernels (replace vanilla MBPO's NLL-train + Gaussian-sample
# kernels). PCN trains via per-block PC weight rule on (delta_obs, reward)
# targets and samples deterministically from a chosen ensemble member's
# feedforward output.
# =============================================================================


def pcn_build_dyn_target_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    READOUT: Int,  # = OBS_DIM + 1
](
    target: LayoutTensor[dtype, Layout.row_major(BATCH, READOUT), MutAnyOrigin],
    s_obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    s_nobs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    s_rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    reward_mean: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    reward_std: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
):
    """
    Build dynamics training target: [reward, delta_obs].

    target[b, d < OBS_DIM] = next_obs[b,d] - obs[b,d]
    target[b, OBS_DIM]   = (reward[b] - reward_mean) / reward_std

    The reward dim is normalized so PCN's unweighted-MSE loss treats it
    on the same scale as the Δobs dims. Without this, the reward (raw σ
    ≈ 1) gets dwarfed by 17 Δobs dims (raw σ ≈ 0.05) in the MSE average,
    PCN regresses reward to zero, and SAC bootstraps Q with no real
    reward signal. The rollout kernel un-normalizes by `* std + mean`
    before storing in the synth buffer.

    One thread per output element. Used to build the regression target for
    the PCN dynamics from a sampled minibatch of (obs, next_obs, reward).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * READOUT
    if tid >= total:
        return
    var b = tid // READOUT
    var d = tid % READOUT
    if d < OBS_DIM:
        target[b, d] = s_nobs[b, d] - s_obs[b, d]
    else:
        var mu = rebind[Scalar[dtype]](reward_mean[0])
        var sigma = rebind[Scalar[dtype]](reward_std[0])
        target[b, d] = (rebind[Scalar[dtype]](s_rew[b]) - mu) / sigma


def pcn_sample_elite_output_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    READOUT: Int,  # = OBS_DIM + 1 (delta_obs + reward)
    NUM_ELITES: Int,
](
    next_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    dyn_output_all: LayoutTensor[
        dtype, Layout.row_major(NUM_ELITES, BATCH, READOUT), MutAnyOrigin
    ],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    elite_slot: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    rng_counter: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
    noise_scale: Scalar[dtype],
    noise_floor: Scalar[dtype],
    noise_cap: Scalar[dtype],
    reward_mean: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    reward_std: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
):
    """For each (b, d), sample chosen-elite + ensemble-disagreement Gaussian noise.

    Vanilla MBPO's sample kernel adds noise scaled by each elite's *learned*
    Gaussian σ (aleatoric uncertainty). PCN dynamics is deterministic — no
    σ — so we substitute *epistemic uncertainty from ensemble disagreement*:

        μ_d  = mean over elites of dyn_output_all[:, b, d]
        σ_d  = sqrt(mean over elites of (dyn_output_all[:, b, d] - μ_d)²)
        sample = dyn_output_all[chosen_slot, b, d] + clip(σ_d, floor, cap)
                                                      · noise_scale · ε
        ε ~ N(0, 1)  via PhiloxRandom Box-Muller, keyed on (rng_counter, tid)

    Then write:
        next_obs[b, d] = obs[b, d] + sample            for d in 0..OBS_DIM
        rewards[b]     = sample                        for d == OBS_DIM

    Without the noise, SAC's critic over-fits to PCN's perfectly-confident
    point predictions (UTD=40 on 100K deterministic synth transitions per
    env step → critic loss spikes to 1500+ early, Q-values lock at a
    suboptimal stable point, policy entropy collapses). Adding ensemble-
    disagreement noise gives SAC the same kind of uncertainty signal
    vanilla's Gaussian σ does, just sourced from member disagreement
    instead of per-member learned variance.

    `noise_scale` lets the caller tune how aggressively to inject noise.
    `noise_floor` ensures at least *some* noise even in well-converged
    regions (so SAC never sees a perfectly confident target).
    `noise_cap` clips runaway disagreement (early in training, ensemble
    disagreement can be huge — clip prevents the noise from drowning out
    the signal).

    One thread per (batch, output_elem) — total BATCH * (OBS_DIM + 1) threads.
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * READOUT
    if tid >= total:
        return
    var b = tid // READOUT
    var d = tid % READOUT
    var slot = Int(elite_slot[b])

    # Compute ensemble mean + std at (b, d) across all NUM_ELITES members.
    var mean_acc = Scalar[dtype](0.0)
    for k in range(NUM_ELITES):
        mean_acc += rebind[Scalar[dtype]](dyn_output_all[k, b, d])
    var mu = mean_acc / Scalar[dtype](NUM_ELITES)
    var var_acc = Scalar[dtype](0.0)
    for k in range(NUM_ELITES):
        var diff = rebind[Scalar[dtype]](dyn_output_all[k, b, d]) - mu
        var_acc += diff * diff
    var sigma = sqrt(var_acc / Scalar[dtype](NUM_ELITES))
    # Clip to [noise_floor, noise_cap] so we always inject some noise but
    # never let early-training disagreement dominate the signal.
    if sigma < noise_floor:
        sigma = noise_floor
    if sigma > noise_cap:
        sigma = noise_cap
    sigma = sigma * noise_scale

    # Box-Muller Gaussian via PhiloxRandom (matches sac_sample_actions_kernel
    # / ddpg_exploration_counter_kernel pattern). RNG keyed on the rolling
    # counter + per-thread offset so each (b, d) gets independent noise.
    var philox = PhiloxRandom(
        seed=UInt64(Int(rng_counter[0])) + UInt64(tid),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    var u1 = Float32(rand_vals[0]) + Float32(1e-8)
    var u2 = Float32(rand_vals[1])
    var mag = sqrt(Float32(-2.0) * log(u1))
    var z = Scalar[dtype](mag * cos(u2 * Float32(6.283185307179586)))

    var sample = rebind[Scalar[dtype]](dyn_output_all[slot, b, d]) + sigma * z
    if d < OBS_DIM:
        next_obs[b, d] = obs[b, d] + sample
    else:
        # Un-normalize reward: model predicts (r - mean)/std, so
        # actual reward = sample * std + mean. Matches the inverse of
        # `pcn_build_dyn_target_kernel`'s reward normalization.
        var rmu = rebind[Scalar[dtype]](reward_mean[0])
        var rsigma = rebind[Scalar[dtype]](reward_std[0])
        rewards[b] = sample * rsigma + rmu


# =============================================================================
# PCNMBPOCPUState
# =============================================================================


struct PCNMBPOCPUState[
    Config: PCNMBPOConfig,
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

    # Dynamics ensemble — PCN owning wrapper (sibling to vanilla MBPO's
    # `DynamicsEnsemble`). API parity: `train_model[CAP](buffer)`,
    # `predict_single(...)`, `elite_indices`, `write_sections` /
    # `read_sections`. Internal training procedure uses SGLD + per-block PC
    # weight gradients instead of NLL backprop.
    var dynamics: PCDynamicsEnsembleInstanceCPU[
        Self.obs_dim,
        Self.action_dim,
        Self.Config.DYN_HIDDEN_DIM,
        Self.Config.ENSEMBLE_SIZE,
        Self.Config.ELITE_SIZE,
        Self.Config.DYN_BATCH,
        Self.Config.T_INFER,
        Self.Config.LR_X,
        Self.Config.DYN_LR,
        Self.Config.DYN_GRAD_CLIP_NORM,
        10.0,  # OBS_REWARD_SCALE — for now, fixed; revisit per-env.
        dtype,
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
        self.dynamics = PCDynamicsEnsembleInstanceCPU[
            Self.obs_dim,
            Self.action_dim,
            Self.Config.DYN_HIDDEN_DIM,
            Self.Config.ENSEMBLE_SIZE,
            Self.Config.ELITE_SIZE,
            Self.Config.DYN_BATCH,
            Self.Config.T_INFER,
            Self.Config.LR_X,
            Self.Config.DYN_LR,
            Self.Config.DYN_GRAD_CLIP_NORM,
            10.0,
            dtype,
        ](base_seed=UInt64(7))
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
# PCNMBPOAgent
# =============================================================================


struct PCNMBPOAgent[
    Config: PCNMBPOConfig,
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
    comptime CPUStateType = PCNMBPOCPUState[Self.Config]

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
    # GPU_N_ENVS is set from the PCNMBPOAgent TRAIN_N_ENVS parameter (default
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

    # PCN dynamics training schedule. The agent fork originally only ran ONE
    # minibatch per ensemble member per `train_dynamics_gpu` call, vs vanilla
    # MBPO's "train until holdout stops improving" loop (~hundreds of
    # minibatches per call). That under-trained dynamics was the dominant
    # reason synth rollouts were noisy on HalfCheetah. These two knobs set
    # the per-call budget and the warmup-bootstrap budget respectively.
    var dyn_train_minibatches_per_call: Int
    var dyn_warmup_minibatches: Int
    var _dyn_pretrained: Bool  # one-shot guard for warmup pretrain.

    # Synth-rollout noise injection. PCN's deterministic predictions give SAC
    # over-confident Q-targets — adding ensemble-disagreement Gaussian noise
    # to (Δobs, reward) before storing in the synth buffer rescues SAC from
    # the "critic locks in suboptimal Q" failure mode (mirrors the role of
    # vanilla MBPO's per-elite learned σ in `dynamics_sample_ensemble_*`).
    #   noise_scale: multiply ensemble-disagreement std by this factor.
    #   noise_floor: minimum per-dim std to inject (so well-converged
    #                regions still see some uncertainty).
    #   noise_cap:   max per-dim std (so early-training disagreement
    #                doesn't drown out the signal).
    var dyn_noise_scale: Float64
    var dyn_noise_floor: Float64
    var dyn_noise_cap: Float64

    # Checkpointing
    var checkpoint_every: Int
    var checkpoint_path: String

    # Logging
    var logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]]
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
        dyn_train_minibatches_per_call: Int = 50,
        dyn_warmup_minibatches: Int = 500,
        dyn_noise_scale: Float64 = 1.0,
        dyn_noise_floor: Float64 = 0.05,
        dyn_noise_cap: Float64 = 1.0,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        diag_every: Int = 0,
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
                (
                    "); using the comptime value. Set PCNMBPOAgent[...,"
                    " REAL_RATIO_PCT=N] to change the GPU batch split."
                ),
            )
        self.real_ratio = rr_from_comptime
        self.sac_updates_per_step = sac_updates_per_step
        self.max_grad_norm = max_grad_norm
        self.use_ere = use_ere
        self.ere_eta = ere_eta

        self.dyn_train_minibatches_per_call = dyn_train_minibatches_per_call
        self.dyn_warmup_minibatches = dyn_warmup_minibatches
        self._dyn_pretrained = False
        self.dyn_noise_scale = dyn_noise_scale
        self.dyn_noise_floor = dyn_noise_floor
        self.dyn_noise_cap = dyn_noise_cap

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.logger = None
        self.diag_every = diag_every

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
        var s = cpu_state.actor.online.model_state_view()
        Self.ActorNet.forward[1](obs_t, out_t, p, s)

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
        # Zero-length model state slice (critic is stateless; CriticGroup has no model_state_view).
        # Pointer is never read; reuse ws scratch buffer as placeholder.
        var critic_state = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.CriticModel.STATE_SIZE),
            MutAnyOrigin,
        ](rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](cpu_state.ws_data.unsafe_ptr()))
        for i in range(Self.Config.NUM_CRITICS):
            var next_qi_t = ws.next_q(i)
            var p_ct = cpu_state.critics.target_params_view(i)
            Self.CriticNet.forward[Self.BATCH](
                next_ci_t, next_qi_t, p_ct, critic_state
            )

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
                ci_t, qi_t, p_ci, critic_state, qi_cache_t
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
                q_grad_t, d_ci_t, p_ci, critic_state, qi_cache_t, g_ci
            )
            cpu_state.critics.pairs[i].optimizer_step()

            if i == 0:
                critic_loss = ci_loss
            else:
                critic_loss = (critic_loss + ci_loss) / 2.0

        # Diagnostic logging
        if Bool(self.logger) and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                self.logger.value()[].log_scalar(
                    "critic_loss", critic_loss, step
                )
                self.logger.value()[].log_scalar("alpha", self.alpha, step)
                self.logger.value()[].log_scalar(
                    "real_buffer_size",
                    Float64(cpu_state.real_buffer.size),
                    step,
                )
                self.logger.value()[].log_scalar(
                    "synth_buffer_size",
                    Float64(cpu_state.synth_buffer.size),
                    step,
                )
                self.logger.value()[].log_scalar(
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
        var s = cpu_state.actor.online.model_state_view()
        Self.ActorNet.forward[1](obs_t, out_t, p, s)

        var result = List[Float64](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            var mean = Float64(out_arr[i])
            # Apply tanh squashing
            var a = (exp(2.0 * mean) - 1.0) / (exp(2.0 * mean) + 1.0)
            a *= self.action_scale
            result.append(a)
        return result^

    # =========================================================================
    # Evaluation (mirrors DeepSACAgent.evaluate)
    # =========================================================================

    def evaluate[
        E: BoxContinuousActionEnv & RenderableEnv,
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent deterministically (mean action, no sampling).

        Returns average reward across `num_episodes`. Optionally renders each
        frame and honours a user-requested window close.
        """
        var quit_requested = False
        if render:
            _ = env.init_renderer()

        var total_reward: Float64 = 0.0
        var completed: Int = 0
        for ep in range(num_episodes):
            if quit_requested:
                break
            var obs_raw = env.reset_obs_list()
            var obs = List[Float64]()
            for i in range(len(obs_raw)):
                obs.append(Float64(obs_raw[i]))

            var episode_reward: Float64 = 0.0
            for _ in range(max_steps_per_episode):
                var action = self.select_greedy_action(self.state, obs)
                var result = env.step_continuous_vec(action)
                var next_obs = List[Float64]()
                for i in range(len(result[0])):
                    next_obs.append(Float64(result[0][i]))
                episode_reward += Float64(result[1])
                obs = next_obs^

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                if result[2]:
                    break

            total_reward += episode_reward
            completed += 1
            if verbose:
                print(
                    "  Episode "
                    + String(ep + 1)
                    + " | Reward: "
                    + String(episode_reward)[byte=:10]
                )

        if render:
            env.close_renderer()

        if completed == 0:
            return 0.0
        return total_reward / Float64(completed)

    # =========================================================================
    # MBPO-specific methods
    # =========================================================================

    def train_dynamics(mut self, mut cpu_state: Self.CPUStateType) raises:
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
        mut gpu_dynamics: PCDynamicsEnsembleInstanceGPU[
            Self.Config.obs_dim,
            Self.Config.action_dim,
            Self.Config.DYN_HIDDEN_DIM,
            Self.Config.ENSEMBLE_SIZE,
            Self.Config.ELITE_SIZE,
            Self.Config.DYN_BATCH,
            Self.Config.ROLLOUT_BATCH,
            Self.Config.T_INFER,
            Self.Config.LR_X,
            Self.Config.DYN_LR,
        ],
        gpu_buffer: GPUReplayBuffer[
            Self.GPU_BUF_CAP, Self.Config.obs_dim, Self.Config.action_dim
        ],
        n_minibatches: Int = -1,
    ) raises:
        """Train PCN dynamics ensemble on GPU.

        Mirrors vanilla MBPO's `gpu_dynamics.train_on_buffer[CAP](ctx, ...)`
        but uses the PCN training procedure: per minibatch, sample into the
        instance's `s_*` staging buffers, build `(s_a, target=(delta_obs,
        reward))` on device, then call `train_one_minibatch(ctx, m)` per
        ensemble member. Repeats for `n_minibatches` iterations to give
        the local PC weight rule enough gradient steps to actually fit
        (vanilla MBPO does ~hundreds per call). Refresh elite indices
        from a fresh holdout batch at the end.

        `n_minibatches=-1` (default) uses `self.dyn_train_minibatches_per_call`.
        """
        var n_iters = (
            n_minibatches if n_minibatches
            >= 0 else self.dyn_train_minibatches_per_call
        )
        if n_iters <= 0:
            return
        comptime DB = Self.Config.DYN_BATCH
        comptime DYN_IN = Self.Config.obs_dim + Self.Config.action_dim
        comptime READOUT = Self.Config.obs_dim + 1
        comptime TPB_VAL = TPB
        comptime DB_BLOCKS = (DB + TPB_VAL - 1) // TPB_VAL
        comptime DB_IN_BLOCKS = (DB * DYN_IN + TPB_VAL - 1) // TPB_VAL
        comptime DB_OUT_BLOCKS = (DB * READOUT + TPB_VAL - 1) // TPB_VAL

        if not gpu_buffer.is_ready[DB]():
            return

        # Bump the RNG counter once for this train call. The increment
        # kernel takes a LayoutTensor view; the buffer.sample call takes
        # the underlying DeviceBuffer directly.
        var dyn_rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_dynamics.r_elite_rng.unsafe_ptr())
        ctx.enqueue_function[increment_rng_counter_kernel](
            dyn_rng_t, grid_dim=(1,), block_dim=(1,)
        )

        # Refit the input scaler on the current real buffer (per-dim
        # mean/std of [obs || act]). Required for BLOCK0's `tanh(x_below)`
        # to operate in its linear regime — without it, raw HalfCheetah
        # obs (positions/velocities ±10) saturate tanh and the dynamics
        # learns garbage. Same recipe as vanilla MBPO TensorStandardScaler.
        gpu_dynamics.fit_scaler_gpu[Self.GPU_BUF_CAP](ctx, gpu_buffer)
        # Refit the reward scaler too (mean + std on the buffer's reward
        # column). Used by `pcn_build_dyn_target_kernel` to normalize the
        # reward target so PCN's unweighted MSE doesn't drown the reward
        # gradient in the 17-dim Δobs gradient.
        gpu_dynamics.fit_reward_scaler_gpu[Self.GPU_BUF_CAP](ctx, gpu_buffer)

        comptime concat_k = concat_obs_action_kernel[
            dtype, DB, Self.Config.obs_dim, Self.Config.action_dim, DYN_IN
        ]
        comptime build_target_k = pcn_build_dyn_target_kernel[
            dtype, DB, Self.Config.obs_dim, READOUT
        ]
        comptime norm_k = normalize_input_kernel[dtype, DB, DYN_IN]

        # ── Per-iter, per-member training pass ─────────────────────────────
        # Outer loop = `n_iters` minibatches; inner = ensemble members.
        # This is the key fix for the cold-start: vanilla MBPO trains
        # dynamics until holdout converges (~hundreds of minibatches per
        # call), whereas the original fork did one. Without this loop the
        # local PC weight rule produces ~MSE 30 at env step 60K vs vanilla
        # ~MSE 0.3, and SAC trains on garbage synth rollouts.
        for batch_iter in range(n_iters):
            # Bump RNG so each outer iteration draws a fresh minibatch.
            ctx.enqueue_function[increment_rng_counter_kernel](
                dyn_rng_t, grid_dim=(1,), block_dim=(1,)
            )

            for m in range(Self.Config.ENSEMBLE_SIZE):
                # 1. Sample DB transitions from gpu_buffer into s_obs/s_act/...
                gpu_buffer.sample[DB](
                    ctx,
                    gpu_dynamics.r_elite_rng,
                    gpu_dynamics.s_obs,
                    gpu_dynamics.s_act,
                    gpu_dynamics.s_rew,
                    gpu_dynamics.s_nobs,
                    gpu_dynamics.s_done,
                    gpu_dynamics.s_idx,
                )

                # 2. Build s_a_dbuf = concat(s_obs, s_act).
                var s_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(DB, Self.Config.obs_dim),
                    MutAnyOrigin,
                ](gpu_dynamics.s_obs.unsafe_ptr())
                var s_act_t = LayoutTensor[
                    dtype,
                    Layout.row_major(DB, Self.Config.action_dim),
                    MutAnyOrigin,
                ](gpu_dynamics.s_act.unsafe_ptr())
                var s_a_t = LayoutTensor[
                    dtype, Layout.row_major(DB, DYN_IN), MutAnyOrigin
                ](gpu_dynamics.s_a_dbuf.unsafe_ptr())
                ctx.enqueue_function[concat_k](
                    s_a_t,
                    s_obs_t,
                    s_act_t,
                    grid_dim=(DB_IN_BLOCKS,),
                    block_dim=(TPB_VAL,),
                )
                # 2b. Normalize s_a using the just-fitted scaler so the PCN
                # forward operates on roughly-unit-scale inputs.
                var mean_t = LayoutTensor[
                    dtype, Layout.row_major(DYN_IN), MutAnyOrigin
                ](gpu_dynamics.input_mean.unsafe_ptr())
                var std_t = LayoutTensor[
                    dtype, Layout.row_major(DYN_IN), MutAnyOrigin
                ](gpu_dynamics.input_std.unsafe_ptr())
                ctx.enqueue_function[norm_k](
                    s_a_t,
                    mean_t,
                    std_t,
                    grid_dim=(DB_IN_BLOCKS,),
                    block_dim=(TPB_VAL,),
                )

                # 3. Build target = (delta_obs, reward) (raw — unnormalized;
                # PCIdentity readout outputs the same scale).
                var s_nobs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(DB, Self.Config.obs_dim),
                    MutAnyOrigin,
                ](gpu_dynamics.s_nobs.unsafe_ptr())
                var s_rew_t = LayoutTensor[
                    dtype, Layout.row_major(DB), MutAnyOrigin
                ](gpu_dynamics.s_rew.unsafe_ptr())
                var target_t = LayoutTensor[
                    dtype, Layout.row_major(DB, READOUT), MutAnyOrigin
                ](gpu_dynamics.target_dbuf.unsafe_ptr())
                var rmean_t = LayoutTensor[
                    dtype, Layout.row_major(1), MutAnyOrigin
                ](gpu_dynamics.reward_mean.unsafe_ptr())
                var rstd_t = LayoutTensor[
                    dtype, Layout.row_major(1), MutAnyOrigin
                ](gpu_dynamics.reward_std.unsafe_ptr())
                ctx.enqueue_function[build_target_k](
                    target_t,
                    s_obs_t,
                    s_nobs_t,
                    s_rew_t,
                    rmean_t,
                    rstd_t,
                    grid_dim=(DB_OUT_BLOCKS,),
                    block_dim=(TPB_VAL,),
                )

                # 4. SGLD inference + PC weight grads + Adam step on member m.
                gpu_dynamics.train_one_minibatch(ctx, m)

        # ── Refresh elite indices from a fresh holdout batch ───────────────
        gpu_buffer.sample[DB](
            ctx,
            gpu_dynamics.r_elite_rng,
            gpu_dynamics.s_obs,
            gpu_dynamics.s_act,
            gpu_dynamics.s_rew,
            gpu_dynamics.s_nobs,
            gpu_dynamics.s_done,
            gpu_dynamics.s_idx,
        )
        var hs_obs_t = LayoutTensor[
            dtype, Layout.row_major(DB, Self.Config.obs_dim), MutAnyOrigin
        ](gpu_dynamics.s_obs.unsafe_ptr())
        var hs_act_t = LayoutTensor[
            dtype, Layout.row_major(DB, Self.Config.action_dim), MutAnyOrigin
        ](gpu_dynamics.s_act.unsafe_ptr())
        var hs_a_t = LayoutTensor[
            dtype, Layout.row_major(DB, DYN_IN), MutAnyOrigin
        ](gpu_dynamics.s_a_dbuf.unsafe_ptr())
        ctx.enqueue_function[concat_k](
            hs_a_t,
            hs_obs_t,
            hs_act_t,
            grid_dim=(DB_IN_BLOCKS,),
            block_dim=(TPB_VAL,),
        )
        # Normalize the holdout batch the same way as the train batches.
        var h_mean_t = LayoutTensor[
            dtype, Layout.row_major(DYN_IN), MutAnyOrigin
        ](gpu_dynamics.input_mean.unsafe_ptr())
        var h_std_t = LayoutTensor[
            dtype, Layout.row_major(DYN_IN), MutAnyOrigin
        ](gpu_dynamics.input_std.unsafe_ptr())
        ctx.enqueue_function[norm_k](
            hs_a_t,
            h_mean_t,
            h_std_t,
            grid_dim=(DB_IN_BLOCKS,),
            block_dim=(TPB_VAL,),
        )
        var hs_nobs_t = LayoutTensor[
            dtype, Layout.row_major(DB, Self.Config.obs_dim), MutAnyOrigin
        ](gpu_dynamics.s_nobs.unsafe_ptr())
        var hs_rew_t = LayoutTensor[dtype, Layout.row_major(DB), MutAnyOrigin](
            gpu_dynamics.s_rew.unsafe_ptr()
        )
        var ht_t = LayoutTensor[
            dtype, Layout.row_major(DB, READOUT), MutAnyOrigin
        ](gpu_dynamics.target_dbuf.unsafe_ptr())
        var hrmean_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            gpu_dynamics.reward_mean.unsafe_ptr()
        )
        var hrstd_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            gpu_dynamics.reward_std.unsafe_ptr()
        )
        ctx.enqueue_function[build_target_k](
            ht_t,
            hs_obs_t,
            hs_nobs_t,
            hs_rew_t,
            hrmean_t,
            hrstd_t,
            grid_dim=(DB_OUT_BLOCKS,),
            block_dim=(TPB_VAL,),
        )
        gpu_dynamics.refresh_elites(ctx)
        # `refresh_elites` updates the host-resident `elite_indices` list;
        # also push to device for the rollout sample kernel.
        gpu_dynamics.sync_elite_member_buf(ctx)

    def do_model_rollouts_gpu[
        E: GPUContinuousEnv,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_dynamics: PCDynamicsEnsembleInstanceGPU[
            Self.Config.obs_dim,
            Self.Config.action_dim,
            Self.Config.DYN_HIDDEN_DIM,
            Self.Config.ENSEMBLE_SIZE,
            Self.Config.ELITE_SIZE,
            Self.Config.DYN_BATCH,
            Self.Config.ROLLOUT_BATCH,
            Self.Config.T_INFER,
            Self.Config.LR_X,
            Self.Config.DYN_LR,
        ],
        mut gpu_state: Self.GPUStateType,
        mut synth_buffer: GPUReplayBuffer[
            Self.Config.SYNTH_CAPACITY,
            Self.Config.obs_dim,
            Self.Config.action_dim,
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

        TODO(B2-step3 next-turn rewrite). The vanilla MBPO body has been
        removed because it depends on:
        - per-elite `gpu_dynamics.members[i].params_view()` — N/A; PCN
          ensemble holds flat buffers sliced by member index, exposed via
          `predict_rollout_member_into_slot(ctx, m, slot)`.
        - vanilla input scaler (`input_mean` / `input_std`) — PCN doesn't
          fit one (caller normalizes obs/action upstream).
        - `dynamics_sample_ensemble_learnable_kernel` — Gaussian-aware
          (mean+logvar). PCN is deterministic; replaced by a small kernel
          that selects per-sample elite slot and copies the chosen slot's
          `(delta_obs, reward)` into `r_next_obs` / `r_rewards`.
        - `gpu_dynamics.max_lv_buf` / `.min_lv_buf` — N/A.

        PCN deviations from vanilla:
        - Per-elite forward via `gpu_dynamics.predict_rollout_member_into_slot`
          (no per-elite `members[i]` access).
        - No input scaler — PCN training expects raw `[obs|action]`.
        - Deterministic sampling via `pcn_sample_elite_output_kernel`
          (next_obs = obs + delta_obs, reward = chosen elite's reward;
          no Gaussian noise, no logvar bounds).
        """
        comptime RB = Self.Config.ROLLOUT_BATCH
        comptime TPB_VAL = TPB
        comptime RB_BLOCKS = (RB + TPB_VAL - 1) // TPB_VAL
        comptime DYN_IN = Self.Config.obs_dim + Self.Config.action_dim
        comptime READOUT = Self.Config.obs_dim + 1
        comptime NUM_ELITES_C = Self.Config.ELITE_SIZE
        comptime DYN_IN_BLOCKS = (RB * DYN_IN + TPB_VAL - 1) // TPB_VAL
        comptime SAMPLE_BLOCKS = (RB * READOUT + TPB_VAL - 1) // TPB_VAL

        comptime concat_k = concat_obs_action_kernel[
            dtype, RB, Self.Config.obs_dim, Self.Config.action_dim, DYN_IN
        ]
        comptime sample_k = pcn_sample_elite_output_kernel[
            dtype, RB, Self.Config.obs_dim, READOUT, NUM_ELITES_C
        ]
        comptime elite_assign_k = sample_elite_assignment_kernel[
            dtype, RB, NUM_ELITES_C
        ]
        comptime r_norm_k = normalize_input_kernel[dtype, RB, DYN_IN]

        var num_elites = len(gpu_dynamics.elite_indices)
        if num_elites == 0 or not gpu_state.buffer.is_ready[RB]():
            return

        # ── Sample start obs from real GPU buffer ──────────────────────────
        comptime rollout_incr_k = increment_rng_counter_kernel
        var rollout_rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.rng_counter.unsafe_ptr())
        ctx.enqueue_function[rollout_incr_k](
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
        # Use sampled obs as rollout starts.
        ctx.enqueue_copy(gpu_dynamics.r_obs, gpu_dynamics.s_obs)
        # Initialize alive mask (all rollouts start alive).
        gpu_dynamics.r_alive.enqueue_fill(Scalar[dtype](1.0))

        for step in range(self.rollout_length):
            # 1. Actor forward on r_obs → raw actor outputs (reuse r_dyn_output
            #    as scratch; gets overwritten before any read).
            var r_obs_t = LayoutTensor[
                dtype, Layout.row_major(RB, Self.OBS), MutAnyOrigin
            ](gpu_dynamics.r_obs.unsafe_ptr())
            var r_act_t = LayoutTensor[
                dtype, Layout.row_major(RB, Self.ACTIONS), MutAnyOrigin
            ](gpu_dynamics.r_actions.unsafe_ptr())
            var raw_t = LayoutTensor[
                dtype, Layout.row_major(RB, Self.ACTOR_OUT), MutAnyOrigin
            ](gpu_dynamics.r_dyn_output.unsafe_ptr())
            var p_actor = gpu_state.actor.online.params_view()
            var s_actor = gpu_state.actor.online.model_state_view()
            Self.ActorNet.forward_gpu[RB](
                ctx,
                r_obs_t,
                raw_t,
                p_actor,
                s_actor,
                gpu_dynamics.r_ws,
            )

            # 2. SAC stochastic action sample (mean + std → tanh).
            comptime sac_k = sac_sample_actions_kernel[
                dtype,
                RB,
                Self.ACTIONS,
                Self.ACTOR_OUT,
            ]
            ctx.enqueue_function[sac_k](
                r_act_t,
                raw_t,
                Scalar[dtype](self.action_scale),
                Scalar[dtype](-5.0),
                Scalar[dtype](2.0),
                Scalar[DType.uint32](UInt32(self.total_steps + step)),
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # 3. Build dynamics input = concat(r_obs, r_actions) → r_dyn_input.
            var r_dyn_in_t = LayoutTensor[
                dtype, Layout.row_major(RB, DYN_IN), MutAnyOrigin
            ](gpu_dynamics.r_dyn_input.unsafe_ptr())
            ctx.enqueue_function[concat_k](
                r_dyn_in_t,
                r_obs_t,
                r_act_t,
                grid_dim=(DYN_IN_BLOCKS,),
                block_dim=(TPB_VAL,),
            )
            # 3b. Normalize the dynamics input the same way training did.
            # Without this, BLOCK0's tanh saturates on raw obs/action.
            var r_mean_t = LayoutTensor[
                dtype, Layout.row_major(DYN_IN), MutAnyOrigin
            ](gpu_dynamics.input_mean.unsafe_ptr())
            var r_std_t = LayoutTensor[
                dtype, Layout.row_major(DYN_IN), MutAnyOrigin
            ](gpu_dynamics.input_std.unsafe_ptr())
            ctx.enqueue_function[r_norm_k](
                r_dyn_in_t,
                r_mean_t,
                r_std_t,
                grid_dim=(DYN_IN_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # 4. For each elite slot, predict its forward into r_dyn_output_all.
            for i in range(NUM_ELITES_C):
                var elite_member_idx = gpu_dynamics.elite_indices[i]
                gpu_dynamics.predict_rollout_member_into_slot(
                    ctx, elite_member_idx, i
                )

            # 5. Per-sample random elite slot.
            var elite_rng_t = LayoutTensor[
                DType.uint32, Layout.row_major(1), MutAnyOrigin
            ](gpu_dynamics.r_elite_rng.unsafe_ptr())
            ctx.enqueue_function[rollout_incr_k](
                elite_rng_t,
                grid_dim=(1,),
                block_dim=(1,),
            )
            var elite_slot_t = LayoutTensor[
                DType.int32, Layout.row_major(RB), MutAnyOrigin
            ](gpu_dynamics.r_elite_idx_per_sample.unsafe_ptr())
            ctx.enqueue_function[elite_assign_k](
                elite_slot_t,
                elite_rng_t,
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # 6. PCN sample with ensemble-disagreement Gaussian noise →
            #    r_next_obs, r_rewards. The noise replaces vanilla MBPO's
            #    learned-σ Gaussian; without it, SAC's critic over-fits to
            #    PCN's perfectly-confident point predictions.
            var r_next_t = LayoutTensor[
                dtype, Layout.row_major(RB, Self.OBS), MutAnyOrigin
            ](gpu_dynamics.r_next_obs.unsafe_ptr())
            var r_rew_t = LayoutTensor[
                dtype, Layout.row_major(RB), MutAnyOrigin
            ](gpu_dynamics.r_rewards.unsafe_ptr())
            var r_dyn_out_all_t = LayoutTensor[
                dtype,
                Layout.row_major(NUM_ELITES_C, RB, READOUT),
                MutAnyOrigin,
            ](gpu_dynamics.r_dyn_output_all.unsafe_ptr())
            # Bump the elite-RNG counter once per rollout step so each
            # sample-kernel call uses a fresh Philox seed.
            ctx.enqueue_function[rollout_incr_k](
                elite_rng_t,
                grid_dim=(1,),
                block_dim=(1,),
            )
            var rsamp_mean_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_dynamics.reward_mean.unsafe_ptr())
            var rsamp_std_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_dynamics.reward_std.unsafe_ptr())
            ctx.enqueue_function[sample_k](
                r_next_t,
                r_rew_t,
                r_dyn_out_all_t,
                r_obs_t,
                elite_slot_t,
                elite_rng_t,
                Scalar[dtype](self.dyn_noise_scale),
                Scalar[dtype](self.dyn_noise_floor),
                Scalar[dtype](self.dyn_noise_cap),
                rsamp_mean_t,
                rsamp_std_t,
                grid_dim=(SAMPLE_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # 7. Clamp synthetic rewards to prevent NaN cascades.
            comptime clamp_k = clamp_rewards_kernel[dtype, RB]
            ctx.enqueue_function[clamp_k](
                r_rew_t,
                Scalar[dtype](-100.0),
                Scalar[dtype](100.0),
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # 8. GPU termination check on predicted next_obs.
            E.is_terminal_obs_gpu[RB, Self.OBS](
                ctx, gpu_dynamics.r_next_obs, gpu_dynamics.r_dones
            )

            # 9. Mask dead rollouts (zero reward + done=1 for already-dead).
            comptime mask_dead_k = mask_dead_rollouts_kernel[dtype, RB]
            var alive_t = LayoutTensor[
                dtype, Layout.row_major(RB), MutAnyOrigin
            ](gpu_dynamics.r_alive.unsafe_ptr())
            var dones_t = LayoutTensor[
                dtype, Layout.row_major(RB), MutAnyOrigin
            ](gpu_dynamics.r_dones.unsafe_ptr())
            ctx.enqueue_function[mask_dead_k](
                alive_t,
                r_rew_t,
                dones_t,
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # 10. Store synthetic transitions in synth GPU buffer.
            synth_buffer.store[RB](
                ctx,
                gpu_dynamics.r_obs,
                gpu_dynamics.r_actions,
                gpu_dynamics.r_rewards,
                gpu_dynamics.r_next_obs,
                gpu_dynamics.r_dones,
            )

            # 11. Update alive mask: alive[b] *= (1 - dones[b]).
            comptime update_alive_k = update_alive_mask_kernel[dtype, RB]
            ctx.enqueue_function[update_alive_k](
                alive_t,
                dones_t,
                grid_dim=(RB_BLOCKS,),
                block_dim=(TPB_VAL,),
            )

            # 12. r_obs ← r_next_obs for next step.
            ctx.enqueue_copy(gpu_dynamics.r_obs, gpu_dynamics.r_next_obs)

    # =========================================================================
    # GPU SAC methods
    # =========================================================================

    def _upload_buffers_to_gpu(
        self,
        cpu_state: Self.CPUStateType,
        mut gpu_state: Self.GPUStateType,
        mut synth_buffer: GPUReplayBuffer[
            Self.Config.SYNTH_CAPACITY,
            Self.Config.obs_dim,
            Self.Config.action_dim,
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
        var s = gpu_state.actor.online.model_state_view()

        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, s, gpu_state.explore_buf
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
        ctx.enqueue_function[sac_explore_k](
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
        ctx.enqueue_function[incr_k](
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
        ctx.enqueue_function[incr_k](
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
        var s_actor = gpu_state.actor.online.model_state_view()
        var p_critic = gpu_state.critics.online_params_view(0)
        var p_critic_t = gpu_state.critics.target_params_view(0)
        # Zero-length model state for critics (GPUCriticGroup has no model_state_view).
        # Pointer is never read; reuse a valid existing param tensor pointer.
        var s_critic = LayoutTensor[
            dtype,
            Layout.row_major(Self.Config.CriticModel.STATE_SIZE),
            MutAnyOrigin,
        ](rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](p_critic_t.ptr))

        # Phase 2: Target actions (SAC: use online actor, no target)
        # Increment RNG counter before target action
        ctx.enqueue_function[incr_k](
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
        ctx.enqueue_function[concat_k](
            next_ci_t,
            nobs_t,
            next_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        var next_q_t = gpu_state.next_q_view[BS]()
        Self.CriticNet.forward_gpu[BS](
            ctx, next_ci_t, next_q_t, p_critic_t, s_critic, gpu_state.critic_ws
        )
        comptime if Self.Config.NUM_CRITICS == 2:
            var nq2_t = gpu_state.nq2_view[BS]()
            var p_c2t = gpu_state.critics.target_params_view(1)
            Self.CriticNet.forward_gpu[BS](
                ctx, next_ci_t, nq2_t, p_c2t, s_critic, gpu_state.critic2_ws
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

        ctx.enqueue_function[concat_k](
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
            s_critic,
            q_cache_t,
            gpu_state.critic_ws,
        )
        ctx.enqueue_function[mse_grad_k](
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
            s_critic,
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

            ctx.enqueue_function[c_norm_k](
                c_ps_t,
                g_critic,
                grid_dim=(C_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[c_clip_k](
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
                s_critic,
                q2_cache_t,
                gpu_state.critic2_ws,
            )
            ctx.enqueue_function[mse_grad_k](
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
                s_critic,
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

                ctx.enqueue_function[c2_norm_k](
                    c2_ps_t,
                    g_c2,
                    grid_dim=(C_BLOCKS2,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[c2_clip_k](
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
            ctx.enqueue_function[incr_k](
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

                ctx.enqueue_function[a_norm_k](
                    a_ps_t,
                    a_grads,
                    grid_dim=(A_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[a_clip_k](
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

                    @parameter
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
                    ctx.enqueue_function[mbpo_alpha_wrapper](
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
            Bool(self.logger)
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
                self.logger.value()[].log_scalar(
                    "critic_loss", critic_loss, step
                )
                self.logger.value()[].log_scalar("mean_q", mean_q, step)
                self.logger.value()[].log_scalar("mean_target", mean_tgt, step)
                self.logger.value()[].log_scalar("mean_reward", mean_rew, step)
                self.logger.value()[].log_scalar("mean_next_q", mean_nq, step)
                self.logger.value()[].log_scalar("mean_done", mean_done, step)
                self.logger.value()[].log_scalar(
                    "mean_abs_action", mean_abs_act, step
                )
                # Read alpha from GPU (not self.alpha — that's only synced at
                # print boundaries by the training loop). Uses caller-supplied
                # pre-allocated alpha_host to avoid per-diag allocation.
                comptime if Self.Config.ActorLoss.HAS_ALPHA:
                    ctx.enqueue_copy(alpha_host, gpu_state.gpu_scalars)
                    ctx.synchronize()
                    self.logger.value()[].log_scalar(
                        "alpha", Float64(alpha_host[0]), step
                    )
                else:
                    self.logger.value()[].log_scalar("alpha", self.alpha, step)
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
        ctx.enqueue_function[incr_k](
            rng_t,
            grid_dim=(1,),
            block_dim=(1,),
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
            Self.Config.SYNTH_CAPACITY,
            Self.Config.obs_dim,
            Self.Config.action_dim,
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
    # GPU training — call run_mbpo_train_gpu directly (see note above).
    # =========================================================================

    # =========================================================================
    # Checkpointable — saves agent hyperparameters and training state.
    # Network weights require save_cpu_state(cpu_state, path) separately
    # because the Checkpointable trait doesn't include state access.
    # =========================================================================

    def save_checkpoint(self, path: String) raises -> None:
        """Save checkpoint using network weights from `self.state`.

        Note: during GPU training the freshly-trained weights live in the
        training loop's local `cpu_state`, not on `self.state`. The loop
        should call `save_checkpoint(cpu_state, path)` (the overload below)
        instead of this one — otherwise the saved file contains the
        agent's random init weights.
        """
        self._save_checkpoint_impl(self.state, path)

    def save_checkpoint(
        self, cpu_state: Self.CPUStateType, path: String
    ) raises -> None:
        """Save checkpoint using network weights from the provided
        `cpu_state`. Used by the GPU training loop after downloading GPU
        weights into its local cpu_state."""
        self._save_checkpoint_impl(cpu_state, path)

    def _save_checkpoint_impl(
        self, cpu_state: Self.CPUStateType, path: String
    ) raises -> None:
        # Total payload size estimate excludes the dynamics PARAM_SIZE since
        # PCN's per-member layout is internal to the instance wrapper.
        var content = write_checkpoint_header(
            "pcn_mbpo",
            Self.Config.ActorModel.PARAM_SIZE
            + Self.Config.CriticModel.PARAM_SIZE * Self.Config.NUM_CRITICS,
            0,
        )
        content += cpu_state.actor.write_sections("actor_")
        content += cpu_state.critics.pairs[0].write_sections("critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            content += cpu_state.critics.pairs[1].write_sections("critic2_")
        # PCN dynamics: single-call serialization (params + Adam state +
        # opt_global + step_nums for the whole ensemble).
        content += cpu_state.dynamics.write_sections("dyn_")

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
        for i in range(len(cpu_state.dynamics.elite_indices)):
            if i > 0:
                elite_str += ","
            elite_str += String(cpu_state.dynamics.elite_indices[i])
        metadata.append("elite_indices=" + elite_str)
        content += write_metadata_section(metadata)
        save_checkpoint_file(path, content)

    def load_checkpoint(mut self, path: String) raises -> None:
        var content = read_checkpoint_file(path)
        self.state.actor.read_sections(content, "actor_")
        self.state.critics.pairs[0].read_sections(content, "critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            self.state.critics.pairs[1].read_sections(content, "critic2_")
        self.state.dynamics.read_sections(content, "dyn_")

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
            "pcn_mbpo_state",
            Self.Config.ActorModel.PARAM_SIZE
            + Self.Config.CriticModel.PARAM_SIZE * Self.Config.NUM_CRITICS,
            0,
        )
        content += cpu_state.actor.write_sections("actor_")
        content += cpu_state.critics.pairs[0].write_sections("critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            content += cpu_state.critics.pairs[1].write_sections("critic2_")
        content += cpu_state.dynamics.write_sections("dyn_")
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
        cpu_state.dynamics.read_sections(content, "dyn_")

    # =========================================================================
    # High-level training loop implementations
    #
    # These live on the struct (rather than as free functions taking
    # `mut agent: PCNMBPOAgent[Config, L, ...]`) so convenience methods below
    # can call them via simple method dispatch on `self`. The free-function
    # entry points `run_mbpo_train[_gpu]` in mbpo_train.mojo forward to
    # these methods so external callers with an `agent` local continue
    # working unchanged.
    #
    # Why not free functions? Mojo nightly dev2026042305 has an L-value
    # unification bug: passing `self` from a method to a free function
    # typed `mut agent: PCNMBPOAgent[Config, L, TRAIN_N_ENVS, REAL_RATIO_PCT]`
    # fails with "l-value ... cannot be converted to reference" even when
    # both sides textually match.  Method dispatch on `self` (or on an
    # external `agent` l-value of the same concrete type) does not trip
    # this bug, so the loops live here.
    # =========================================================================

    def _run_train_impl[
        E: BoxContinuousActionEnv,
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        mut env: E,
        num_epochs: Int,
        steps_per_epoch: Int = 1000,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 5000,
        eval_episodes: Int = 5,
        eval_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 1,
        environment_name: String = "Environment",
        logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]] = None,
    ) raises -> TrainingMetrics:
        """MBPO CPU training loop body. See `train()` and `run_mbpo_train`."""
        var metrics = TrainingMetrics(
            algorithm_name="MBPO",
            environment_name=environment_name,
        )

        # --- Warmup: fill real buffer with random transitions ---
        var warmup_obs = env.reset_obs_list()
        var warmup_count = 0
        var warmup_ep_steps = 0
        while warmup_count < warmup_steps:
            var action = self.random_action[E.dtype]()
            var result = env.step_continuous_vec(action)
            var next_obs = result[0].copy()
            var reward = Float64(result[1])
            var done = result[2]
            warmup_ep_steps += 1
            var terminated = done and (warmup_ep_steps < max_steps_per_episode)
            self.store_transition(
                cpu_state, warmup_obs, action, reward, next_obs, terminated
            )
            warmup_count += 1
            if done:
                warmup_obs = env.reset_obs_list()
                warmup_ep_steps = 0
            else:
                warmup_obs = next_obs^

        if verbose:
            print(
                "Warmup complete: "
                + String(warmup_steps)
                + " steps in real buffer"
            )

        # --- Training loop ---
        var total_env_steps = 0
        var episode_obs = env.reset_obs_list()
        var episode_reward: Float64 = 0.0
        var episode_steps = 0
        var episode_count = 0

        for epoch in range(num_epochs):
            self.update_rollout_length(epoch)

            for step in range(steps_per_epoch):
                var obs_f64 = List[Float64]()
                for i in range(len(episode_obs)):
                    obs_f64.append(Float64(episode_obs[i]))

                var action = self.select_action(cpu_state, obs_f64)
                var result = env.step_continuous_vec(action)
                var next_obs = List[Float64]()
                for i in range(len(result[0])):
                    next_obs.append(Float64(result[0][i]))
                var reward = Float64(result[1])
                var done = result[2]
                episode_steps += 1
                var terminated = done and (
                    episode_steps < max_steps_per_episode
                )
                self.store_transition(
                    cpu_state, obs_f64, action, reward, next_obs, terminated
                )
                episode_reward += reward
                total_env_steps += 1

                if total_env_steps % self.model_train_freq == 0:
                    self.train_dynamics(cpu_state)
                    self.do_model_rollouts(cpu_state)

                    if verbose:
                        print(
                            "  Model trained at step "
                            + String(total_env_steps)
                            + " | Real buffer: "
                            + String(cpu_state.real_buffer.size)
                            + " | Synth buffer: "
                            + String(cpu_state.synth_buffer.size)
                            + " | Rollout len: "
                            + String(self.rollout_length)
                        )

                if cpu_state.is_ready():
                    for _ in range(self.sac_updates_per_step):
                        _ = self.do_cpu_train_step(cpu_state)

                if done or episode_steps >= max_steps_per_episode:
                    episode_count += 1
                    metrics.log_episode(
                        episode_count - 1,
                        Scalar[DType.float64](episode_reward),
                        episode_steps,
                        self.get_explore_rate(),
                    )
                    if Bool(logger):
                        logger.value()[].log_scalar(
                            "episode_reward", episode_reward, total_env_steps
                        )
                    episode_obs = env.reset_obs_list()
                    episode_reward = 0.0
                    episode_steps = 0
                else:
                    episode_obs = List[Scalar[E.dtype]]()
                    for i in range(len(next_obs)):
                        episode_obs.append(Scalar[E.dtype](next_obs[i]))

            if eval_every > 0 and (epoch + 1) % eval_every == 0:
                var eval_total: Float64 = 0.0
                for _ in range(eval_episodes):
                    var eval_obs_raw = env.reset_obs_list()
                    var eval_obs = List[Float64]()
                    for i in range(len(eval_obs_raw)):
                        eval_obs.append(Float64(eval_obs_raw[i]))
                    var eval_reward: Float64 = 0.0
                    for _ in range(max_steps_per_episode):
                        var eval_action = self.select_greedy_action(
                            cpu_state, eval_obs
                        )
                        var eval_result = env.step_continuous_vec(eval_action)
                        var eval_next = List[Float64]()
                        for i in range(len(eval_result[0])):
                            eval_next.append(Float64(eval_result[0][i]))
                        eval_reward += Float64(eval_result[1])
                        if eval_result[2]:
                            break
                        eval_obs = eval_next^
                    eval_total += eval_reward

                var avg_eval = eval_total / Float64(eval_episodes)

                if Bool(logger):
                    logger.value()[].log_scalar(
                        "eval_reward", avg_eval, total_env_steps
                    )

                if verbose and (epoch + 1) % print_every == 0:
                    print(
                        "Epoch "
                        + String(epoch + 1)
                        + " | Eval reward: "
                        + String(avg_eval)[byte=:8]
                        + " | Env steps: "
                        + String(total_env_steps)
                        + " | Alpha: "
                        + String(self.alpha)[byte=:6]
                        + " | Rollout: "
                        + String(self.rollout_length)
                    )

            if (
                self.checkpoint_every > 0
                and self.checkpoint_path.byte_length() > 0
                and (epoch + 1) % self.checkpoint_every == 0
            ):
                self.save_checkpoint(cpu_state, self.checkpoint_path)

        if Bool(logger):
            logger.value()[].flush()
        return metrics^

    def _run_train_gpu_impl[
        E: GPUContinuousEnv,
        USE_CUDA_GRAPH: Bool = False,
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]] = None,
    ) raises -> TrainingMetrics:
        """MBPO GPU training loop body. See `train_gpu()` and `run_mbpo_train_gpu`.
        """
        comptime n_envs = Self.GPU_N_ENVS

        var metrics = TrainingMetrics(
            algorithm_name="MBPO-GPU",
            environment_name=environment_name,
        )

        comptime GPUState = Self.GPUStateType
        var gpu_state = GPUState(ctx)

        var gpu_dynamics = PCDynamicsEnsembleInstanceGPU[
            Self.Config.obs_dim,
            Self.Config.action_dim,
            Self.Config.DYN_HIDDEN_DIM,
            Self.Config.ENSEMBLE_SIZE,
            Self.Config.ELITE_SIZE,
            Self.Config.DYN_BATCH,
            Self.Config.ROLLOUT_BATCH,
            Self.Config.T_INFER,
            Self.Config.LR_X,
            Self.Config.DYN_LR,
        ](ctx)
        # NOTE: vanilla agent calls `gpu_dynamics.upload_from(cpu_state.dynamics, ctx)`
        # to copy the CPU-trained dynamics weights to GPU. The PCN GPU instance
        # initializes its own params on construction; if we want to match the
        # vanilla flow, we'd add an `upload_from` method (out of scope for the
        # foundation; B2-step3 to-do).

        gpu_state.actor.upload_from(cpu_state.actor, ctx)
        gpu_state.critics.upload_from(cpu_state.critics, ctx)

        comptime GPUStateT = Self.GPUStateType
        var scalars_host = ctx.enqueue_create_host_buffer[dtype](
            GPUStateT.GPU_SCALARS_SIZE
        )
        scalars_host[GPUStateT.GPU_ALPHA] = Scalar[dtype](self.alpha)
        scalars_host[GPUStateT.GPU_LOG_ALPHA] = Scalar[dtype](self.log_alpha)
        scalars_host[GPUStateT.GPU_ADAM_M] = Scalar[dtype](self.alpha_adam_m)
        scalars_host[GPUStateT.GPU_ADAM_V] = Scalar[dtype](self.alpha_adam_v)
        scalars_host[GPUStateT.GPU_ADAM_T] = Scalar[dtype](self.alpha_adam_t)
        scalars_host[GPUStateT.GPU_TARGET_ENT] = Scalar[dtype](
            self.target_entropy
        )
        scalars_host[GPUStateT.GPU_ALPHA_LR] = Scalar[dtype](self.alpha_lr)
        ctx.enqueue_copy(gpu_state.gpu_scalars, scalars_host)

        var alpha_host = ctx.enqueue_create_host_buffer[dtype](
            GPUStateT.GPU_SCALARS_SIZE
        )

        var synth_buffer = GPUReplayBuffer[
            Self.Config.SYNTH_CAPACITY,
            Self.Config.obs_dim,
            Self.Config.action_dim,
        ](ctx)

        if self.use_ere:
            gpu_state.buffer.enable_ere(self.ere_eta)
            synth_buffer.enable_ere(self.ere_eta)
        comptime REAL_BS = Self.REAL_BS
        comptime SYNTH_BS = Self.SYNTH_BS
        var s_real_idx = ctx.enqueue_create_buffer[DType.int32](REAL_BS)
        var s_synth_idx = ctx.enqueue_create_buffer[DType.int32](SYNTH_BS)

        var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
        var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
        var actions_buf = ctx.enqueue_create_buffer[dtype](
            n_envs * E.ACTION_DIM
        )
        var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
        var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
        var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)
        var host_reward_sum = ctx.enqueue_create_host_buffer[dtype](1)
        var host_episode_count = ctx.enqueue_create_host_buffer[dtype](1)

        var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
        if ws_size == 0:
            ws_size = 1
        var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)
        if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
            E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)

        E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=0)
        E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=0,
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )

        ctx.enqueue_memset(episode_rewards_buf, 0)
        ctx.enqueue_memset(episode_steps_buf, 0)
        ctx.enqueue_memset(gpu_reward_sum_buf, 0)
        ctx.enqueue_memset(gpu_episode_count_buf, 0)

        comptime tpb = 256
        comptime env_blocks = (n_envs + tpb - 1) // tpb
        comptime accum_k = accumulate_rewards_kernel[dtype, n_envs]
        comptime incr_k = increment_steps_kernel[dtype, n_envs]
        comptime log_reset_k = log_and_reset_completed_kernel[dtype, n_envs]
        comptime act_blocks = (n_envs * E.ACTION_DIM + tpb - 1) // tpb
        comptime warmup_k = uniform_random_actions_kernel[
            dtype, n_envs, E.ACTION_DIM
        ]
        var action_scale_val = Scalar[dtype](self.action_scale)

        var total_steps = 0
        var total_train_steps = 0
        var step_seed: UInt32 = 42
        var completed_episodes = 0
        var last_avg_reward: Float64 = 0.0
        var next_print = print_every
        var next_model_train = self.model_train_freq
        var epoch = 0

        var _train_graph: Optional[CUDAGraph] = None

        var progress_interval = print_every // 20
        if progress_interval < n_envs:
            progress_interval = n_envs
        var next_progress = progress_interval

        while total_steps < num_steps:
            ctx.enqueue_copy(prev_obs_buf, obs_buf)

            if total_steps < warmup_steps:
                var act_t = LayoutTensor[
                    dtype,
                    Layout.row_major(n_envs, E.ACTION_DIM),
                    MutAnyOrigin,
                ](actions_buf.unsafe_ptr())
                ctx.enqueue_function[warmup_k](
                    act_t,
                    action_scale_val,
                    Scalar[DType.uint32](step_seed),
                    grid_dim=(act_blocks,),
                    block_dim=(tpb,),
                )
            else:
                self.select_actions_gpu[n_envs](
                    ctx, gpu_state, obs_buf, actions_buf
                )

            self.total_steps += n_envs

            E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                terminated_buf,
                obs_buf,
                rng_seed=UInt64(step_seed),
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )

            gpu_state.gpu_store[n_envs](
                ctx,
                prev_obs_buf,
                actions_buf,
                rewards_buf,
                obs_buf,
                terminated_buf,
            )

            var ep_rew_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](episode_rewards_buf.unsafe_ptr())
            var rew_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](rewards_buf.unsafe_ptr())
            var ep_steps_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](episode_steps_buf.unsafe_ptr())
            var dones_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](dones_buf.unsafe_ptr())
            var rsum_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
                gpu_reward_sum_buf.unsafe_ptr()
            )
            var ecount_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_episode_count_buf.unsafe_ptr())

            ctx.enqueue_function[accum_k](
                ep_rew_t,
                rew_t,
                grid_dim=(env_blocks,),
                block_dim=(tpb,),
            )
            ctx.enqueue_function[incr_k](
                ep_steps_t,
                grid_dim=(env_blocks,),
                block_dim=(tpb,),
            )
            ctx.enqueue_function[log_reset_k](
                dones_t,
                ep_rew_t,
                ep_steps_t,
                rsum_t,
                ecount_t,
                grid_dim=(1,),
                block_dim=(1,),
            )

            E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
                ctx,
                states_buf,
                dones_buf,
                rng_seed=UInt64(step_seed + 1),
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )
            E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
                ctx, states_buf, obs_buf
            )

            if gpu_state.gpu_buffer_is_ready():
                if synth_buffer.is_ready[SYNTH_BS]():
                    comptime if USE_CUDA_GRAPH and has_nvidia_gpu_accelerator():
                        if not _train_graph:
                            self._gpu_train_kernels(
                                ctx,
                                gpu_state,
                                synth_buffer,
                                s_real_idx,
                                s_synth_idx,
                            )
                            self.soft_update_targets_gpu(ctx, gpu_state)
                            ctx.synchronize()
                            var graph = CUDAGraph(ctx)
                            graph.begin_capture()
                            self._gpu_train_kernels(
                                ctx,
                                gpu_state,
                                synth_buffer,
                                s_real_idx,
                                s_synth_idx,
                            )
                            self.soft_update_targets_gpu(ctx, gpu_state)
                            graph.end_capture()
                            if verbose:
                                print(
                                    "[CUDA Graph] Captured MBPO SAC train step"
                                    " with "
                                    + String(graph.num_nodes())
                                    + " nodes"
                                )
                            _train_graph = graph^
                        for _ in range(self.sac_updates_per_step):
                            _train_graph.value().replay_async()
                        _train_graph.value().sync()
                        self._gpu_train_diagnostics(
                            ctx,
                            gpu_state,
                            self.sac_updates_per_step,
                            alpha_host,
                        )
                    else:
                        for _ in range(self.sac_updates_per_step):
                            self.do_gpu_train_step(
                                ctx,
                                gpu_state,
                                synth_buffer,
                                s_real_idx,
                                s_synth_idx,
                                alpha_host,
                            )
                            self.soft_update_targets_gpu(ctx, gpu_state)
                else:
                    for _ in range(self.sac_updates_per_step):
                        self.do_gpu_train_step_real_only(ctx, gpu_state)
                        self.soft_update_targets_gpu(ctx, gpu_state)
                total_train_steps += self.sac_updates_per_step

            total_steps += n_envs
            step_seed += 1

            # ── One-shot pretrain right after warmup completes ──────────────
            # Hammers the dynamics on the warmup buffer with many minibatches
            # before SAC starts using synth rollouts. Without this, MSE at
            # env step 5K is still ~30+ and the first ~10K env steps of
            # synth rollouts are noise that SAC overfits to. With the
            # pretrain, MSE starts the post-warmup phase already in single
            # digits so synth data is useful from the first SAC update.
            if (
                not self._dyn_pretrained
                and total_steps >= warmup_steps
                and gpu_state.buffer.is_ready[Self.Config.DYN_BATCH]()
            ):
                if verbose:
                    print(
                        "[PCN-MBPO] Warmup complete (",
                        total_steps,
                        " env steps); pretraining dynamics for ",
                        self.dyn_warmup_minibatches,
                        " minibatches/member ...",
                    )
                self.train_dynamics_gpu(
                    ctx,
                    gpu_dynamics,
                    gpu_state.buffer,
                    n_minibatches=self.dyn_warmup_minibatches,
                )
                self._dyn_pretrained = True
                if verbose:
                    print("[PCN-MBPO] Dynamics pretrain done.")

            if total_steps >= next_model_train and total_steps >= warmup_steps:
                # PCN equivalent of vanilla `gpu_dynamics.train_on_buffer`.
                # Fits scaler → trains each member (× n_minibatches) →
                # refreshes elites.
                self.train_dynamics_gpu(ctx, gpu_dynamics, gpu_state.buffer)
                self.update_rollout_length(epoch)
                epoch += 1

                # ── Dynamics health metrics ────────────────────────────────
                # Holdout MSE is computed inside `train_dynamics_gpu`'s
                # `refresh_elites` call but not returned. Re-run holdout
                # eval per member here for logging only (cheap: NUM_ENSEMBLE
                # forwards on a DYN_BATCH-sized batch already on device).
                if Bool(logger):
                    # Per-member holdout MSE — total + obs-only + reward-only
                    # so we can see if reward MSE is hidden behind aggregate.
                    var loss_sum: Float64 = 0.0
                    var loss_min: Float64 = 1e30
                    var loss_max: Float64 = -1.0
                    var obs_sum: Float64 = 0.0
                    var rew_sum: Float64 = 0.0
                    for m in range(Self.Config.ENSEMBLE_SIZE):
                        var (
                            L,
                            L_obs,
                            L_rew,
                        ) = gpu_dynamics.eval_member_holdout_mse_breakdown(
                            ctx, m
                        )
                        loss_sum += L
                        obs_sum += L_obs
                        rew_sum += L_rew
                        if L < loss_min:
                            loss_min = L
                        if L > loss_max:
                            loss_max = L
                    var n_ens_f = Float64(Self.Config.ENSEMBLE_SIZE)
                    logger.value()[].log_scalar(
                        "dyn_holdout_mse_mean", loss_sum / n_ens_f, total_steps
                    )
                    logger.value()[].log_scalar(
                        "dyn_holdout_mse_min", loss_min, total_steps
                    )
                    logger.value()[].log_scalar(
                        "dyn_holdout_mse_max", loss_max, total_steps
                    )
                    logger.value()[].log_scalar(
                        "dyn_holdout_spread",
                        loss_max - loss_min,
                        total_steps,
                    )
                    logger.value()[].log_scalar(
                        "dyn_holdout_mse_obs",
                        obs_sum / n_ens_f,
                        total_steps,
                    )
                    logger.value()[].log_scalar(
                        "dyn_holdout_mse_reward",
                        rew_sum / n_ens_f,
                        total_steps,
                    )
                    var input_std_mean = gpu_dynamics.download_input_std(ctx)
                    logger.value()[].log_scalar(
                        "dyn_input_std_mean", input_std_mean, total_steps
                    )

                var n_rollout_batches = max(
                    1,
                    self.num_rollouts_per_step // gpu_dynamics.rollout_batch,
                )
                for _ in range(n_rollout_batches):
                    self.do_model_rollouts_gpu[E](
                        ctx, gpu_dynamics, gpu_state, synth_buffer
                    )
                next_model_train += self.model_train_freq

            if verbose and total_steps >= next_progress:
                var interval_start = next_print - print_every
                print_progress_bar(
                    total_steps - interval_start,
                    print_every,
                    total_train_steps,
                    "MBPO-GPU",
                )
                next_progress += progress_interval

            if (
                verbose or (Bool(logger) and logger.value()[].is_active())
            ) and total_steps >= next_print:
                ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
                ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
                ctx.enqueue_copy(alpha_host, gpu_state.gpu_scalars)
                ctx.synchronize()
                self.alpha = Float64(alpha_host[GPUStateT.GPU_ALPHA])
                self.log_alpha = Float64(alpha_host[GPUStateT.GPU_LOG_ALPHA])

                var raw_count = Float64(host_episode_count[0])
                var recent_count = (
                    Int(raw_count) if raw_count > 0.0
                    and raw_count == raw_count else 0
                )
                var raw_sum = Float64(host_reward_sum[0])
                var recent_sum = raw_sum if raw_sum == raw_sum else 0.0
                if recent_count > 0 and raw_sum != raw_sum:
                    print(
                        "[MBPO WARN] NaN in episode reward sum at step "
                        + String(total_steps)
                        + " (count="
                        + String(recent_count)
                        + ")"
                    )
                completed_episodes += recent_count

                if recent_count > 0:
                    last_avg_reward = recent_sum / Float64(recent_count)
                    for _ in range(recent_count):
                        metrics.log_episode(
                            completed_episodes, last_avg_reward, 0, 0.0
                        )

                ctx.enqueue_memset(gpu_reward_sum_buf, 0)
                ctx.enqueue_memset(gpu_episode_count_buf, 0)

                if Bool(logger):
                    logger.value()[].log_scalar(
                        "avg_reward", last_avg_reward, total_steps
                    )
                    logger.value()[].log_scalar(
                        "episodes", Float64(completed_episodes), total_steps
                    )
                    logger.value()[].log_scalar(
                        "train_steps", Float64(total_train_steps), total_steps
                    )
                    logger.value()[].log_scalar(
                        "alpha", self.alpha, total_steps
                    )
                    logger.value()[].log_scalar(
                        "rollout_length",
                        Float64(self.rollout_length),
                        total_steps,
                    )
                    logger.value()[].log_scalar(
                        "real_buffer_size",
                        Float64(gpu_state.buffer.size),
                        total_steps,
                    )
                    logger.value()[].log_scalar(
                        "synth_buffer_size",
                        Float64(synth_buffer.size),
                        total_steps,
                    )
                    logger.value()[].log_scalar(
                        "model_epoch", Float64(epoch), total_steps
                    )

                if verbose:
                    clear_progress_bar()
                    print(
                        "MBPO-GPU | Step "
                        + String(total_steps)
                        + " / "
                        + String(num_steps)
                        + " | Ep: "
                        + String(completed_episodes)
                        + " | AvgR: "
                        + String(last_avg_reward)[byte=:7]
                        + " | Alpha: "
                        + String(self.alpha)[byte=:6]
                        + " | Train: "
                        + String(total_train_steps)
                        + " | R: "
                        + String(gpu_state.buffer.size)
                        + " S: "
                        + String(synth_buffer.size)
                    )

                if (
                    self.checkpoint_every > 0
                    and self.checkpoint_path.byte_length() > 0
                    and total_steps >= self.checkpoint_every
                    and total_steps % self.checkpoint_every < print_every
                ):
                    gpu_state.actor.download_to(cpu_state.actor, ctx)
                    gpu_state.critics.download_to(cpu_state.critics, ctx)
                    ctx.synchronize()
                    self.save_checkpoint(cpu_state, self.checkpoint_path)

                next_print += print_every

        ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
        ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
        ctx.synchronize()
        var final_raw = Float64(host_episode_count[0])
        var final_count = (
            Int(final_raw) if final_raw > 0.0 and final_raw == final_raw else 0
        )
        if final_count > 0:
            var final_avg = Float64(host_reward_sum[0]) / Float64(final_count)
            completed_episodes += final_count
            for _ in range(final_count):
                metrics.log_episode(completed_episodes, final_avg, 0, 0.0)

        gpu_state.actor.download_to(cpu_state.actor, ctx)
        gpu_state.critics.download_to(cpu_state.critics, ctx)
        ctx.synchronize()

        if self.checkpoint_every > 0 and self.checkpoint_path.byte_length() > 0:
            self.save_checkpoint(cpu_state, self.checkpoint_path)

        if Bool(logger):
            logger.value()[].flush()
        return metrics^

    # =========================================================================
    # High-level training — convenience wrappers
    # =========================================================================

    def train[
        E: BoxContinuousActionEnv,
    ](
        mut self,
        mut env: E,
        num_epochs: Int,
        steps_per_epoch: Int = 1000,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 5000,
        eval_episodes: Int = 5,
        eval_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 1,
        environment_name: String = "Environment",
        logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]] = None,
    ) raises -> TrainingMetrics:
        """CPU MBPO training convenience wrapper.

        Allocates a fresh CPU state, runs the training loop, and stores
        the final state on `self.state` so `evaluate()` works immediately.
        """
        var cpu_state = Self.CPUStateType()
        var metrics = self._run_train_impl[E](
            cpu_state,
            env,
            num_epochs,
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
        return metrics^

    def train_gpu[
        E: GPUContinuousEnv,
        USE_CUDA_GRAPH: Bool = False,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 5_000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]] = None,
    ) raises -> TrainingMetrics:
        """GPU MBPO training convenience wrapper.

        Allocates a fresh CPU state, runs the GPU training loop, and
        leaves the downloaded weights on `self.state` so `evaluate()`
        works without the caller juggling cpu_state.
        """
        var cpu_state = Self.CPUStateType()
        var metrics = self._run_train_gpu_impl[E, USE_CUDA_GRAPH](
            cpu_state,
            ctx,
            num_steps,
            warmup_steps=warmup_steps,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            logger=logger,
        )
        self.state = cpu_state^
        return metrics^
