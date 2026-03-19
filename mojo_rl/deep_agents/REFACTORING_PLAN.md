# Deep Agents Refactoring Plan

**Goal:** Remove redundant manual agent implementations, keep only generic+autodiff agents, and reorganize the folder structure for clarity.

**Date:** 2025-03-19

---

## Current State

Each algorithm exists in up to 3 versions:
1. **Manual handwritten** (~1.2MB) — `dqn/`, `ddpg/`, `td3/`, `sac/`, `a2c/`, `ppo/`, `dueling_dqn/`, `dqn_per/`
2. **Generic composable** — `core/generic/` with config structs + strategy traits
3. **Generic + autodiff** — autodiff config variants (`AutodiffDQNConfig`, `AutodiffSACConfig`, etc.)

Usage analysis:
- 91% of examples and 100% of tests use generic agents
- Only 3 `_old` example files and 1 debug test still import from old directories
- `dqn_cnn/` and `ppo_cnn/` are standalone but have generic config equivalents (`DQNCNNConfig`, `PPOCNNConfig`)

**Critical dependency:** Generic agents import kernels from old agent directories (see Phase 1).

---

## Phase 0: Kernel Consolidation ✅ DONE

**Why first:** Generic agents import kernels from old directories. We must relocate these before deleting anything.

### Kernels moved into `core/kernels.mojo` ✅

#### From `dqn/kernels.mojo` (2 kernels)
- [x] `dqn_td_target_kernel` — used by `core/generic/q_target.mojo`
- [x] `dqn_double_td_target_kernel` — used by `core/generic/q_target.mojo`

#### From `dueling_dqn/kernels.mojo` (2 kernels)
- [x] `dueling_combine_kernel` — used by `core/generic/q_output.mojo`
- [x] `dueling_grad_kernel` — used by `core/generic/q_output.mojo`

#### From `td3/kernels.mojo` (1 kernel)
- [x] `add_gaussian_noise_kernel` — used by `core/generic/target_action.mojo`

#### From `sac/kernels.mojo` (5 kernels)
- [x] `sac_sample_actions_kernel` — used by `core/generic/offpolicy_agent.mojo`
- [x] `sac_rsample_with_cache_kernel` — used by `core/generic/actor_loss.mojo`, `core/generic/target_action.mojo`
- [x] `sac_rsample_bwd_kernel` — used by `core/generic/actor_loss.mojo`
- [x] `min_q_dq_kernel` — used by `core/generic/actor_loss.mojo`
- [x] `add_ci_grads_kernel` — used by `core/generic/actor_loss.mojo`

#### From `ppo/kernels.mojo` (18 kernels used by generic)
- [x] `ppo_gather_minibatch_kernel` — used by `core/generic/onpolicy_agent.mojo`
- [x] `ppo_gather_minibatch_obs_parallel_kernel` — used by `core/generic/onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`
- [x] `ppo_critic_grad_kernel` — used by `core/generic/onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`
- [x] `ppo_critic_grad_clipped_kernel` — used by `core/generic/onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`
- [x] `normalize_advantages_kernel` — used by `core/generic/onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`
- [x] `gradient_norm_kernel` — used by `core/generic/offpolicy_agent.mojo`, `onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`
- [x] `gradient_reduce_and_compute_scale_kernel` — used by `core/generic/onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`
- [x] `gradient_apply_scale_kernel` — used by `core/generic/onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`
- [x] `gradient_reduce_apply_fused_kernel` — used by `core/generic/offpolicy_agent.mojo`
- [x] `_store_pre_step_kernel` — used by `core/generic/onpolicy_agent.mojo`, `core/gpu_onpolicy_train.mojo`
- [x] `_store_pre_step_obs_parallel_kernel` — used by `core/generic/onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`
- [x] `_store_post_step_kernel` — used by `core/generic/onpolicy_agent.mojo`, `onpolicy_continuous_agent.mojo`, `core/gpu_onpolicy_train.mojo`
- [x] `_sample_continuous_actions_kernel` — used by `core/generic/onpolicy_continuous_agent.mojo`
- [x] `_store_continuous_pre_step_kernel` — used by `core/generic/onpolicy_continuous_agent.mojo`
- [x] `ppo_continuous_gather_minibatch_kernel` — used by `core/generic/onpolicy_continuous_agent.mojo`
- [x] `ppo_continuous_actor_grad_kernel` — used by `core/generic/onpolicy_continuous_agent.mojo`
- [x] `ppo_actor_grad_with_kl_kernel` — used by `core/generic/policy_gradient.mojo`
- [x] `clamp_log_std_params_kernel` — used by `core/generic/onpolicy_continuous_agent.mojo`

#### Also moved `gradient_norm_kernel` and `gradient_reduce_apply_fused_kernel` from `dreamer_v3/kernels.mojo` ✅
- Previously `offpolicy_agent.mojo` imported these from dreamer_v3 (duplicates of ppo versions)
- Now imported from `core/kernels.mojo`

#### Also moved `PPOContinuousState` and `PPOContinuousGPUState` ✅
- From `ppo/state.mojo` → new file `core/onpolicy_state.mojo`
- `onpolicy_continuous_agent.mojo` updated to import from new location

### Import updates ✅

All imports in `core/generic/` updated to use `core.kernels` instead of old agent-specific paths.
Verified: `grep` for old import paths returns zero matches in `core/`.

---

## Phase 1: Delete Old Manual Agents ✅ DONE

### Directories deleted
- [x] `deep_agents/dqn/` — replaced by `GenericDQNAgent[DoubleDQNConfig]`
- [x] `deep_agents/ddpg/` — replaced by `GenericOffPolicyAgent[DDPGConfig]`
- [x] `deep_agents/td3/` — replaced by `GenericOffPolicyAgent[TD3Config]`
- [x] `deep_agents/sac/` — replaced by `GenericOffPolicyAgent[SACConfig]`
- [x] `deep_agents/a2c/` — replaced by `GenericOnPolicyAgent[A2CConfig]`
- [x] `deep_agents/ppo/` — replaced by `GenericOnPolicyAgent[PPOConfig]` + `GenericOnPolicyContinuousAgent[ContinuousPPOConfig]`
- [x] `deep_agents/dueling_dqn/` — replaced by `GenericDQNAgent[DuelingDQNConfig]`
- [x] `deep_agents/dqn_per/` — replaced by `GenericDQNPERAgent[DQNPERConfig]`
- [x] `deep_agents/dqn_cnn/` — replaced by `GenericDQNAgent[DQNCNNConfig]`
- [x] `deep_agents/ppo_cnn/` — replaced by `GenericOnPolicyAgent[PPOCNNConfig]`

### Files deleted
- [x] `examples/half_cheetah/ppo_half_cheetah_training_gpu_old.mojo`
- [x] `examples/half_cheetah/sac_half_cheetah_training_gpu_old.mojo`
- [x] `examples/half_cheetah/td3_half_cheetah_training_gpu_old.mojo`
- [x] `tests/test_ppo_hopper_continuous_debug.mojo` (uses old PPO import)

### Additional fix
- [x] `tdmpc2/tdmpc2.mojo` — updated to import `gradient_norm_kernel` and `gradient_reduce_apply_fused_kernel` from `core.kernels` (was importing from `ppo.kernels`)

### Verified
- `deep_agents/__init__.mojo` only imports from `core.generic` and `dreamer_v3` — no changes needed
- Zero remaining references to deleted directories in the codebase (only a docstring comment in `aliases.mojo`)

---

## Phase 2: Reorganize `core/generic/` into `core/` ✅ DONE

Split the 18 files in `core/generic/` into logical subdirectories:

### Target structure

```
deep_agents/
├── __init__.mojo              # Public API (aliases + re-exports)
├── core/
│   ├── __init__.mojo
│   │
│   ├── agents/                # Agent structs
│   │   ├── __init__.mojo
│   │   ├── dqn_agent.mojo
│   │   ├── offpolicy_agent.mojo
│   │   ├── onpolicy_agent.mojo
│   │   └── onpolicy_continuous_agent.mojo
│   │
│   ├── configs/               # Algorithm configs
│   │   ├── __init__.mojo
│   │   ├── dqn_configs.mojo       # DQN, DoubleDQN, DuelingDQN, DQNCNN, DQNPER, AutodiffDQN
│   │   ├── offpolicy_configs.mojo # DDPG, TD3, SAC + Autodiff variants
│   │   └── onpolicy_configs.mojo  # PPO, A2C, ContinuousPPO, PPOCNN + Autodiff variants
│   │
│   ├── strategies/            # Strategy traits + implementations
│   │   ├── __init__.mojo
│   │   ├── exploration.mojo       # GaussianNoise, StochasticSample
│   │   ├── update_schedule.mojo   # EveryStep, DelayedAll, DelayedActorOnly
│   │   ├── target_value.mojo      # SingleQTarget, TwinQTarget, EntropicTwinQTarget
│   │   ├── target_action.mojo     # DeterministicTarget, SmoothedTarget, ReparamTarget
│   │   ├── actor_loss.mojo        # DPGLoss, MaxEntLoss + Autodiff variants
│   │   ├── policy_gradient.mojo   # VanillaPG, ClippedSurrogate + Autodiff variants
│   │   ├── epoch_schedule.mojo    # SinglePass, MultiEpochMinibatch
│   │   ├── q_target.mojo          # StandardQTarget, DoubleQTarget
│   │   ├── q_output.mojo          # DirectQ, DuelingQ
│   │   └── q_gradient.mojo        # ManualQGradient, AutodiffQGradient
│   │
│   ├── training/              # Training loops (moved from core/ root)
│   │   ├── __init__.mojo
│   │   ├── offpolicy_train.mojo
│   │   ├── gpu_offpolicy_train.mojo
│   │   ├── onpolicy_train.mojo
│   │   ├── gpu_onpolicy_train.mojo
│   │   ├── offpolicy_helpers.mojo
│   │   └── onpolicy_helpers.mojo
│   │
│   ├── replay/                # Stays as-is
│   │   ├── __init__.mojo
│   │   ├── replay_buffer.mojo
│   │   ├── gpu_replay_buffer.mojo
│   │   ├── sequence_replay_buffer.mojo
│   │   └── gpu_sequence_replay_buffer.mojo
│   │
│   ├── kernels.mojo           # All shared GPU kernels (consolidated in Phase 0)
│   ├── eval.mojo
│   ├── utils.mojo
│   ├── perf_timer.mojo
│   └── checkpoint_trait.mojo
│
├── tdmpc2/                    # Stays as-is
└── dreamer_v3/                # Stays as-is
```

### Migration steps (all completed)
- [x] Create `core/agents/`, `core/configs/`, `core/strategies/`, `core/training/`
- [x] Move agent files from `core/generic/` → `core/agents/` (+ aliases.mojo)
- [x] Move config files → `core/configs/` (DQN configs kept inside dqn_agent.mojo)
- [x] Move strategy files from `core/generic/` → `core/strategies/`
- [x] Move training loops from `core/` root → `core/training/`
- [x] Delete `core/generic/` directory
- [x] Create `__init__.mojo` for agents/, configs/, strategies/, training/
- [x] Update all internal imports (relative + absolute) across moved files
- [x] Update `deep_agents/__init__.mojo` (`core.generic` → `core.agents`)
- [x] Bulk update 77 external files (examples/ + tests/) from `core.generic` → `core.agents`
- [x] Verified: zero references to `core.generic` remain in codebase

---

## Phase 3: Clean up `nn/model/rsample_layer.mojo` ✅ DONE

Renamed to `nn/model/autodiff_layers.mojo` — it's a collection of DiffOp→Model wrappers, not just RSample.

- [x] Renamed `rsample_layer.mojo` → `autodiff_layers.mojo`
- [x] Updated import in `nn/model/__init__.mojo`

---

## Phase 4: Make Autodiff the Default ✅ DONE

All base configs now use autodiff strategies by default. Old `Autodiff`-prefixed configs are retained as `comptime` aliases for backward compatibility.

### Off-policy configs changed
- [x] `DDPGConfig`: `DPGLoss` → `AutodiffDPGLoss`
- [x] `TD3Config`: `DPGLoss` → `AutodiffTD3Loss`
- [x] `SACConfig`: `MaxEntLoss[]` → `AutodiffMaxEntLoss[]`
- [x] `AutodiffSACConfig` / `AutodiffDDPGConfig` / `AutodiffTD3Config` → `comptime` aliases

### On-policy configs changed
- [x] `PPOConfig`: `ClippedSurrogate` → `AutodiffClippedSurrogate[]`, `USE_AUTODIFF_GRAD = True`
- [x] `A2CConfig`: `VanillaPG` → `AutodiffVanillaPG`, `USE_AUTODIFF_GRAD = True`
- [x] `ContinuousPPOConfig`: `ClippedSurrogate` → `AutodiffClippedSurrogate[]`, `USE_AUTODIFF_GRAD = True`
- [x] `PPOCNNConfig`: `ClippedSurrogate` → `AutodiffClippedSurrogate[]`, `USE_AUTODIFF_GRAD = True`
- [x] `AutodiffPPOConfig` / `AutodiffA2CConfig` / `AutodiffContinuousPPOConfig` → `comptime` aliases

### DQN configs changed
- [x] `DQNConfig`, `DoubleDQNConfig`, `DuelingDQNConfig`, `DQNCNNConfig`, `DQNPERConfig`: `ManualQGradient` → `AutodiffQGradient[]`
- [x] `AutodiffDQNConfig` → `comptime` alias for `DoubleDQNConfig`
- `HuberDQNConfig` already used `AutodiffQGradient[HuberLoss]` — unchanged

### Manual strategies retained
- `ManualQGradient`, `DPGLoss`, `MaxEntLoss`, `VanillaPG`, `ClippedSurrogate` still available for custom configs or TDMPC2/Dreamer

---

## Execution Order

1. **Phase 0** — Kernel consolidation (prerequisite, no behavior change)
2. **Phase 1** — Delete old agents (major cleanup, ~1.2MB removed)
3. **Phase 2** — Reorganize folders (structural, many import changes)
4. **Phase 3** — Rename rsample_layer.mojo (trivial)
5. **Phase 4** — Autodiff as default (requires benchmarking)

Each phase should be a separate commit/PR. Test after each phase.

---

## Files NOT touched

- `deep_agents/tdmpc2/` — model-based, can't autodiff yet
- `deep_agents/dreamer_v3/` — world model, can't autodiff yet
- `nn/` structure (except rsample_layer rename)
- All examples and tests using generic imports (just work)
