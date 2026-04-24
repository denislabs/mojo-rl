# REDQ-OFE agent — implementation plan

Status (end of session 2026-04-24): `config.mojo` landed. `__init__.mojo`
landed (config-only, no agent yet). Full agent implementation deferred to
the next session.

## What's in this directory

```
redq_ofe/
├── __init__.mojo          # exports REDQOFEConfig + default configs
├── config.mojo            # REDQOFEConfig trait + DefaultREDQOFEConfig{6,8}
└── PLAN.md                # this file
```

## Background: the integration pattern

See `references/OFENet-main/teflon/policy/SAC.py:train_for_batch`.

- OFE params are updated **only** by the aux next-state-prediction loss
  (`tape.gradient(td_loss, self.qf1.trainable_variables)` lists only
  critic vars — OFE is not in the list). Actor and critic treat OFE
  features as inputs with stop-gradient.
- The `features_from_states` / `features_from_states_actions` methods run
  in `training=False` mode — BatchNorm uses running stats, no update.
- The `call([states, actions])` aux-loss forward runs in `training=True`
  mode — batch stats + EMA update of running stats.

## The three-piece split (proven in `tests/nn/test_ofenet_three_piece.mojo`)

Three independent `NetworkState` objects, each with its own Adam
optimizer, chained manually for aux training:

```
StateBranch6/8  : Sequential of DenseBlocks      [OBS → φ(s)]
ActionBranch6/8 : Sequential of DenseBlocks      [concat(φ(s), a) → φ(s,a)]
Linear          : Linear[PHI_SA_DIM, OBS]        [φ(s,a) → ŝ']
```

All three use a shared aux Adam (single LR, OFE_LR in config).

## Agent implementation — required file: `redq_ofe.mojo`

Copy `../redq/redq.mojo` and apply the following surgical changes.

### 1. Type renames + imports

- `REDQAgent` → `REDQOFEAgent`
- `REDQGPUState` → `REDQOFEGPUState`
- `Config: REDQConfig` → `Config: REDQOFEConfig`
- Replace `from .config import ...` with import from `.config` (REDQOFE)

### 2. New fields in `REDQOFEGPUState`

```mojo
# OFE network states (each = params + grads + Adam m/v buffers)
var ofe_sb_state: GPUNetworkState[Config.OFEStateBranchModel, Adam[Config.OFE_LR]]
var ofe_ab_state: GPUNetworkState[Config.OFEActionBranchModel, Adam[Config.OFE_LR]]
var ofe_pr_state: GPUNetworkState[Config.OFEPredictorModel, Adam[Config.OFE_LR]]

# Intermediate feature buffers (sized for minibatch BS)
var phi_s_buf: DeviceBuffer[dtype]       # [BS, PHI_S_DIM]
var phi_s_next_buf: DeviceBuffer[dtype]  # [BS, PHI_S_DIM]
var phi_sa_in_buf: DeviceBuffer[dtype]   # [BS, PHI_S_DIM + ACT]
var phi_sa_buf: DeviceBuffer[dtype]      # [BS, PHI_SA_DIM]
var phi_sa_next_buf: DeviceBuffer[dtype] # [BS, PHI_SA_DIM] (target path)

# Aux loss buffers
var pred_s_next_buf: DeviceBuffer[dtype] # [BS, OBS]
var aux_grad_pred_buf: DeviceBuffer[dtype] # [BS, OBS]
var aux_grad_phi_sa_buf: DeviceBuffer[dtype]
var aux_grad_phi_sa_in_buf: DeviceBuffer[dtype]
var aux_grad_phi_s_buf: DeviceBuffer[dtype]
```

Also: action-selection path needs a small phi_s buffer for N_ENVS, not BS.

### 3. Modification sites in the training loop

| Line | Context | Change |
|------|---------|--------|
| 58 | Import | Remove `concat_obs_action_kernel` (no longer used) |
| 646 | CPU `select_greedy_action` | Run state-branch forward on CPU, feed φ(s) to actor |
| 803 | GPU `select_actions_gpu` | Before actor forward: run state-branch forward_gpu_no_cache (inference) on obs → phi_s_for_envs |
| 1089 | `_phase_critic_update`, target path | Replace `concat_obs_action_kernel(next_obs, next_action)` with: `StateBranch.forward_no_cache(next_obs) → phi_s_next`; concat(phi_s_next, next_action) → phi_sa_in; `ActionBranch.forward_no_cache → phi_sa_next`. Feed phi_sa_next to target critics. |
| 1114 | Actor forward for target action | Precede with state-branch forward on next_obs → phi_s_next; feed to ActorNet. |
| 1186 | Target critics forward | Input is now phi_sa_next (from above), not concat. |
| 1251 | Online critics forward (with cache) | Replace concat(obs, action) with OFE forward: state-branch → phi_s, concat(phi_s, action) → phi_sa_in, action-branch → phi_sa. Feed phi_sa. |
| 1294 | `_phase_actor_alpha_update`, online path | Same as 1251 for critic-of-sampled-action. |
| 1318 | Actor forward with cache | Input is phi_s (computed from minibatch obs). |
| 1444 | Critics forward for min-Q in actor loss | Input is phi_sa_sampled: phi_s (reused from 1318) + sampled action → concat → ActionBranch → phi_sa_sampled. |
| 1541 | Actor backward | No OFE change here — gradient doesn't flow to OFE because actor's input (phi_s) is treated as stop-gradient. |

### 4. New aux training step

Inserted once per env step (not per UTD iteration), after the usual REDQ
updates. Uses the same minibatch as the RL updates:

```
# Aux forward (training mode — BN uses batch stats, updates running stats)
SB.forward_gpu[BS](obs, phi_s_buf, sb_state.params, sb_state.cache)
concat_phi_s_action_kernel(phi_s_buf, action) → phi_sa_in_buf
AB.forward_gpu[BS](phi_sa_in_buf, phi_sa_buf, ab_state.params, ab_state.cache)
Predictor.forward_gpu[BS](phi_sa_buf, pred_s_next_buf, pr_state.params, pr_state.cache)

# MSE grad: grad_pred = 2 * (pred - next_obs) / (BS * OBS)
mse_grad_kernel[BS, OBS](pred_s_next_buf, next_obs, aux_grad_pred_buf)

# Aux backward
sb_state.zero_grads()
ab_state.zero_grads()
pr_state.zero_grads()

Predictor.backward_gpu[BS](aux_grad_pred_buf, aux_grad_phi_sa_buf,
                           pr_state.params, pr_state.cache, pr_state.grads)
AB.backward_gpu[BS](aux_grad_phi_sa_buf, aux_grad_phi_sa_in_buf,
                    ab_state.params, ab_state.cache, ab_state.grads)
slice_kernel[BS, 0, PHI_S_DIM](aux_grad_phi_sa_in_buf, aux_grad_phi_s_buf)
SB.backward_gpu[BS](aux_grad_phi_s_buf, _discarded_grad_obs,
                    sb_state.params, sb_state.cache, sb_state.grads)

# Optimizer step (all three)
sb_state.optimizer_step()
ab_state.optimizer_step()
pr_state.optimizer_step()
```

### 5. New kernels in `kernels.mojo`

Probably only one:

- `concat_phi_s_action_kernel[dtype, BATCH, PHI_S, ACT]`: concat [phi_s | action]
  into a PHI_S+ACT-wide buffer. Mechanically identical to
  `concat_obs_action_kernel` — just a rename with different dims; we can
  likely reuse `concat_obs_action_kernel` by passing PHI_S_DIM instead of
  obs_dim and phi_s_buf as the "obs" arg. Verify this.

### 6. Checkpointing

Extend `save_checkpoint` / `load_checkpoint` to also save/load the 3 OFE
NetworkStates' params (and optionally their Adam m/v buffers).

### 7. Evaluation path (CPU)

`evaluate` (line 661) needs to run OFE state-branch on CPU before actor.

## Estimated scope of full agent implementation

- Agent file: ~2200 LOC (+150 vs. REDQ for OFE state + aux step)
- New kernel (if needed): ~50 LOC
- `__init__.mojo` (augment): ~20 LOC
- Top-level `deep_agents/__init__.mojo`: ~10 LOC of re-exports
- Smoke test: ~30 LOC (similar to `test_redq_import.mojo`)
- HalfCheetah example: ~180 LOC (copied from REDQ example, swap config)

**Total: ~2500 LOC for the agent + example + test.** Multi-session effort.

## Testing strategy

1. Smoke test: imports resolve (follows `test_redq_import.mojo` pattern).
2. Parity sanity: at init (before any aux training), the OFE state branch
   with gamma=1, beta=0, running_mean=0, running_var=1 is approximately
   identity-ish — critic Q-values should be roughly similar to vanilla
   REDQ's at init. (Not a hard test, just a sanity check.)
3. Example: HalfCheetah 10K-step training. Verify no NaNs, aux loss
   decreases, episode reward trends up. Compare to vanilla REDQ baseline
   at same step count.
