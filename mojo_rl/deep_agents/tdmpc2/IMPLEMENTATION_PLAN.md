# TDMPC2 Implementation Plan for mojo-rl

> Based on: Hansen et al., 2023 — "TD-MPC2: Scalable, Robust World Models for Continuous Control"
> Reference implementation: https://github.com/nicklashansen/tdmpc2

---

## Algorithm Overview

TD-MPC2 is a **model-based RL** algorithm that simultaneously learns:

- an **encoder** : observation → latent state `z`
- a **dynamics model** : `(z, a) → z'` (next latent state prediction)
- a **reward model** : `(z, a) → r` (distributional)
- a **termination model** : `(z) → done`
- a **policy** : `z → (μ, σ)` (Gaussian, used as prior for MPPI)
- an **ensemble of Q-functions** : `(z, a) → Q` (distributional)

At each timestep, TDMPC2 uses **MPPI** (Model Predictive Path Integral) to plan in
latent space over a horizon H, using the world model to evaluate candidate trajectories.

---

## What Already Exists in mojo-rl ✅

### Neural layers (`nn/model/`)

| Component                            | File                    | TDMPC2 usage                             |
| ------------------------------------ | ----------------------- | ---------------------------------------- |
| `Linear`                             | `linear.mojo`           | Base layer for all MLPs                  |
| `LayerNorm`                          | `layer_norm.mojo`       | Used inside `NormedLinear`               |
| `Dropout`                            | `dropout.mojo`          | Used inside `NormedLinear` (Q-functions) |
| `ReLU`, `Tanh`, `Sigmoid`, `Softmax` | dedicated files         | Base activations                         |
| `StochasticActor`                    | `stochastic_actor.mojo` | Foundation for the policy head           |
| `Sequential`                         | `sequential.mojo`       | MLP composition                          |

### Optimizers (`nn/optimizer/`)

| Optimizer | TDMPC2 usage                |
| --------- | --------------------------- |
| `Adam`    | World model + policy        |
| `AdamW`   | Alternative for world model |

### Losses (`nn/loss/`)

| Loss           | TDMPC2 usage                                     |
| -------------- | ------------------------------------------------ |
| `MSE`          | Consistency loss (dynamics prediction)           |
| `CrossEntropy` | Foundation for soft cross-entropy (reward/value) |

### Agents & Infrastructure

| Component            | TDMPC2 usage                                                           |
| -------------------- | ---------------------------------------------------------------------- |
| `ReplayBuffer`       | Transition storage                                                     |
| `SAC`                | Structural reference (twin Q-networks, stochastic actor, soft updates) |
| `Network` (training) | Params + model wrapper                                                 |
| Checkpoint system    | Save/load trained models                                               |

---

## What Needs to Be Built 🔨

### 1. New Neural Layers

#### `nn/model/mish.mojo` — Mish Activation

```
Mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + eˣ))
```

Default activation for all `NormedLinear` blocks in TDMPC2.

Gradient:

```
f'(x) = tanh(sp) + x * σ(x) * (2 - tanh²(sp))   where sp = softplus(x)
```

**Effort**: Low — simple elementwise activation, straightforward forward + backward.

---

#### `nn/model/simnorm.mojo` — Simplicial Normalization

```
SimNorm(simplex_dim)(x):
  1. Reshape x: [..., D] → [..., D / simplex_dim, simplex_dim]
  2. Apply Softmax over the last dimension
  3. Reshape back → [..., D]
```

Used in the **dynamics model output** to stabilize the latent space
(replaces LayerNorm on the dynamics head's final layer).

- **Parameters**: `simplex_dim` (group size, typically 8)
- **No learned parameters** — pure normalization
- **Effort**: Low — Softmax already exists in mojo-rl; this is a reshape + softmax.

---

#### `nn/model/normed_linear.mojo` — NormedLinear Block

```
NormedLinear(in_dim, out_dim, dropout_rate=0., act=Mish):
  Linear(in_dim, out_dim) → Dropout(dropout_rate) → LayerNorm(out_dim) → Mish
```

The **base building block** for all TDMPC2 MLPs (except final projection layers).
Final layers use plain `Linear`, optionally followed by `SimNorm` (dynamics head).

**Effort**: Low — composition of existing layers. Requires Mish first.

---

### 2. New Loss Functions

#### `nn/loss/two_hot.mojo` — Two-Hot Encoding

TDMPC2 uses distributional RL: rewards and values are represented as distributions
over `num_bins` evenly spaced bins in `[v_min, v_max]`.

```
two_hot(x, bins[num_bins]) → target[num_bins]:
  Find adjacent bins: bins[k] ≤ x < bins[k+1]
  target[k]   = (bins[k+1] - x) / (bins[k+1] - bins[k])   # upper weight
  target[k+1] = (x - bins[k])   / (bins[k+1] - bins[k])   # lower weight
  target[i]   = 0  for all other i
```

Produces a soft one-hot vector of size `num_bins`.
Typical values: `num_bins=101`, `v_min=-10.0`, `v_max=10.0`.

**Effort**: Medium — vector computation with careful boundary handling.

---

#### `nn/loss/soft_cross_entropy.mojo` — Soft Cross-Entropy

```
L = -Σᵢ target_i * log(softmax(logits)_i)
  = -Σᵢ target_i * log_softmax(logits)_i
```

Where `target` is the two-hot vector (soft, not hard one-hot).
Used as the loss for both **reward** and **Q-value** heads (distributional).

**Effort**: Low — extends existing `cross_entropy.mojo`.

---

### 3. Sequence Replay Buffer

#### `nn/replay/sequence_replay_buffer.mojo`

The consistency loss requires **unrolling the world model over H steps**.
The replay buffer must store and sample **contiguous sequences of length H+1**,
not individual transitions.

```
Interface:
  add(obs, action, reward, done)               // Continuous insertion
  sample_sequence[BATCH, H]() →
    (obs[H+1], actions[H], rewards[H], dones[H])
```

**Effort**: Medium — circular buffer with sequence handling
(must avoid sequences that cross episode boundaries).

---

### 4. World Model

#### `deep_agents/tdmpc2/world_model.mojo`

```
WorldModel[OBS_DIM, ACTION_DIM, LATENT_DIM, MLP_DIM, NUM_BINS, NUM_Q]:

  encoder:      MLP(OBS_DIM → LATENT_DIM)
                [NormedLinear(OBS_DIM, MLP_DIM),
                 NormedLinear(MLP_DIM, LATENT_DIM)]

  dynamics:     MLP(LATENT_DIM + ACTION_DIM → LATENT_DIM)
                [NormedLinear(LATENT_DIM + ACTION_DIM, MLP_DIM),
                 NormedLinear(MLP_DIM, LATENT_DIM),
                 Linear(LATENT_DIM, LATENT_DIM) + SimNorm(simplex_dim=8)]

  reward:       MLP(LATENT_DIM + ACTION_DIM → NUM_BINS)
                [NormedLinear(LATENT_DIM + ACTION_DIM, MLP_DIM),
                 NormedLinear(MLP_DIM, MLP_DIM),
                 Linear(MLP_DIM, NUM_BINS)]

  termination:  MLP(LATENT_DIM → 1)
                [NormedLinear(LATENT_DIM, MLP_DIM),
                 NormedLinear(MLP_DIM, MLP_DIM),
                 Linear(MLP_DIM, 1) + Sigmoid]

  policy:       MLP(LATENT_DIM → 2 * ACTION_DIM)   # mean + log_std
                [NormedLinear(LATENT_DIM, MLP_DIM),
                 NormedLinear(MLP_DIM, MLP_DIM),
                 Linear(MLP_DIM, 2 * ACTION_DIM)]

  Q_ensemble:   NUM_Q × MLP(LATENT_DIM + ACTION_DIM → NUM_BINS)
                [NormedLinear(..., MLP_DIM, dropout on 1st layer),
                 NormedLinear(MLP_DIM, MLP_DIM),
                 Linear(MLP_DIM, NUM_BINS)]
```

Typical dimensions (single task): `LATENT_DIM=512`, `MLP_DIM=512`,
`NUM_BINS=101`, `NUM_Q=5`.

**Effort**: High — central piece. Requires all layers above.

---

### 5. MPPI Planner

#### `deep_agents/tdmpc2/mppi.mojo`

```
plan(z0, world_model, num_iterations, horizon, num_samples, num_pi_trajs,
     temperature) → action:

  1. Initialization:
     - Sample num_pi_trajs trajectories using the learned policy
     - mean[H, ACTION_DIM] = 0 (or shifted from previous timestep if t > 0)
     - std[H, ACTION_DIM]  = 0.5

  2. For each iteration:
     a. Sample candidates:
        noise ~ N(0, 1)
        actions = clamp(mean + std * noise, -1, 1)     # [num_samples, H, A]

     b. For each candidate, roll out the world model over H steps:
        z_t+1 = dynamics(z_t, a_t)
        r_t   = decode_reward(reward_logits(z_t, a_t))
        G     = Σ_t γ^t * r_t + γ^H * min_Q(z_H, π(z_H))

     c. Top-k elites by return G

     d. Softmax weights:
        w = exp(temperature * (G - max(G)))
        w = w / Σ w

     e. Update distribution:
        mean = Σ wᵢ * actionsᵢ
        std  = sqrt(Σ wᵢ * (actionsᵢ - mean)²)
        std  = clamp(std, std_min, std_max)

  3. Action selection:
     Gumbel-softmax sampling over elite scores → select action_0
     (add exploration noise if not in eval mode)
```

**Effort**: High — nested planning loop with many world model forward passes.
GPU-batched evaluation of `num_samples=512` trajectories is the key performance target.

---

### 6. TDMPC2 Agent

#### `deep_agents/tdmpc2/tdmpc2.mojo`

**Key hyperparameters:**

```
H                 = 3        # planning horizon
gamma             = 0.99     # discount factor
rho               = 0.5      # temporal weight decay in losses
tau               = 0.01     # soft update coefficient for target Q-networks
batch_size        = 256
learning_rate     = 3e-4
enc_lr_scale      = 0.3      # encoder LR multiplier (= 0.3 * lr)
consistency_coef  = 2.0
reward_coef       = 0.5
value_coef        = 0.1
entropy_coef      = 1e-4
num_samples       = 512      # MPPI candidates
num_pi_trajs      = 24       # policy rollout trajectories in MPPI
num_iterations    = 6        # MPPI optimization iterations
temperature       = 0.5      # MPPI softmax temperature
```

**Update step (1 gradient step):**

```
1. Encode initial observation:
   z_0 = encode(obs_0)

2. Compute TD targets (no gradient):
   For each t in [0, H-1]:
     z_next     = sg(encode(obs_t+1))       # stop-gradient
     a_next     ~ policy(z_next)
     Q_targets  = r_t + gamma * (1 - done_t) * min_Q(z_next, a_next)
     td_target_dist[t] = two_hot(Q_targets)

3. Latent rollout + loss accumulation:
   z = z_0
   For t in [0, H-1]:
     z_pred  = dynamics(z, a_t)
     z_enc   = sg(encode(obs_t+1))

     L_consistency += rho^t * MSE(z_pred, z_enc)
     L_reward      += rho^t * soft_CE(reward_logits(z, a_t), two_hot(r_t))
     L_value       += rho^t * soft_CE(Q(z, a_t),             td_target_dist[t])
     L_terminal    += rho^t * BCE(termination(z), done_t)   # if episodic

     z = z_pred   # continue rollout with predicted latent

4. World model update:
   L_wm = consistency_coef * L_consistency
        + reward_coef      * L_reward
        + value_coef       * L_value
        + terminal_coef    * L_terminal
   backprop(L_wm) → Adam step (world_model params, exc. policy)

5. Policy update:
   L_pi = 0
   z = sg(z_0)
   For t in [0, H-1]:
     a_pi ~ policy(z)
     L_pi += -rho^t * (min_Q(z, a_pi) + entropy_coef * H(policy))
     z     = sg(dynamics(z, a_pi))
   Adam step (policy params only)

6. Soft update of target Q-networks:
   theta_target ← tau * theta + (1 - tau) * theta_target
```

---

## Implementation Status

### Completed

- [x] `nn/model/mish.mojo` — Mish activation
- [x] `nn/model/simnorm.mojo` — Simplicial normalization
- [x] `nn/model/normed_linear.mojo` — NormedLinear block
- [x] `nn/loss/soft_cross_entropy.mojo` — Soft cross-entropy
- [x] `nn/loss/two_hot.mojo` — Two-hot encoding + `symlog`/`symexp`
- [x] `nn/replay/sequence_replay_buffer.mojo` — Sequence replay buffer
- [x] `deep_agents/tdmpc2/world_model.mojo` — Full world model (encoder, dynamics, reward, termination, policy, Q-ensemble)
- [x] `deep_agents/tdmpc2/mppi.mojo` — MPPI planner (CPU)
- [x] `deep_agents/tdmpc2/tdmpc2.mojo` — Full agent with CPU + GPU training
- [x] `deep_agents/tdmpc2/kernels.mojo` — GPU kernels for all training operations
- [x] `deep_agents/tdmpc2/state.mojo` — CPU + GPU state management
- [x] MPPI warm-start — shift previous plan's mean forward for consecutive timesteps

### Bug fixes applied

- [x] **Symlog/symexp normalization** — bins represent symlog space; `symexp` applied when decoding Q-values, `symlog` applied to TD targets and reward targets before two-hot encoding. Without this, Q-values saturate at [-10, 10] boundaries.
- [x] **Reward network trained on immediate rewards** — was incorrectly using TD targets (r + γV) instead of raw rewards.
- [x] **Single optimizer step per training iteration** — was doing H optimizer steps inside the horizon loop.
- [x] **Proper DPG policy gradient** — backprop through Q-network to get dQ/d(action), chained through tanh squashing. Original kernel ignored Q-values entirely.
- [x] **Random 2-of-5 Q subsampling** — both TD targets (min of 2) and policy gradient (avg of 2) now randomly subsample 2 of 5 Q-networks per step, matching the reference.
- [x] **Loss coefficients** — `consistency_coef: 20.0` (was 2.0), `reward_coef: 0.1` (was 0.5), matching reference defaults.
- [x] **CPU policy uses avg(Q1, Q2)** — was using min, reference uses average for policy gradient.
- [x] **Symexp derivative in policy gradient** — Q decode backward kernel includes `exp(|Q_symlog|)` factor for correct chain rule through symexp.

---

## Remaining TODOs

### ~~MPPI Warm-Start~~ ✅ DONE

Implemented: `plan()` accepts `t0` flag and `prev_mean` list. When `t0=False`,
shifts `prev_mean[1:]` into `mean[:-1]` (last step zeros). Agent stores
`_prev_mean` and `_episode_t0`, resetting at each episode start. Also fixed
exploration noise to use `std[0]` instead of fixed 0.025 (matches reference).

**Files**: `deep_agents/tdmpc2/mppi.mojo`, `deep_agents/tdmpc2/tdmpc2.mojo`

---

### ~~MPPI Gumbel-Softmax Action Selection~~ ✅ DONE

Implemented: Replaced deterministic argmax with `_weighted_sample()` (multinomial
sampling proportional to softmax weights). Exploration noise now scales by `std[a]`
at t=0 instead of fixed 0.025, matching the reference.

**Files**: `deep_agents/tdmpc2/mppi.mojo`

---

### ~~Dynamic Discount Factor (gamma)~~ ✅ DONE

Implemented: Constructor accepts `episode_length`, `discount_denom` (default 5.0),
`discount_min` (0.95), `discount_max` (0.995). When `episode_length > 0`, computes
`gamma = (L/d - 1) / (L/d)` clamped to [min, max]. E.g. HalfCheetah (1000 steps)
→ gamma=0.995. When `episode_length=0` (default), uses the fixed `gamma` parameter.

**Files**: `deep_agents/tdmpc2/tdmpc2.mojo` (constructor)

---

### MPPI GPU Batching (Performance)

The MPPI loop currently runs on CPU, evaluating trajectories one at a time.
GPU-batching all `num_samples=512` candidates through the world model in parallel
is the single biggest performance opportunity. Target: a single batched
`dynamics_forward[512 * H]` call per MPPI iteration.

**Files**: `deep_agents/tdmpc2/mppi.mojo` (full rewrite for GPU)

---

## Technical Notes

### Stop-Gradient

The consistency loss requires stopping gradients on the encoded `obs_t+1` target.
In mojo-rl this means calling `encode()` in inference mode (no activation caching)
for the target, while the predicted `z_pred` flows through the full computational graph.

### Q-Network Ensemble

TDMPC2 uses 5 Q-networks. Random 2-of-5 subsampling is used for both:
- **TD targets**: min of 2 randomly selected target Q-networks
- **Policy gradient**: avg of 2 randomly selected online Q-networks (with DPG backprop)

Implemented via PhiloxRandom counter-based RNG and pointer arrays for O(1) selection.

### Distributional RL — Symlog Space

Q-functions output logits over `NUM_BINS=101` bins spanning `[v_min, v_max]` in
**symlog space**. The scalar value is recovered as:

```
value_symlog = Σᵢ softmax(logits)ᵢ * bins[i]
value_actual = symexp(value_symlog)
```

Targets are encoded in symlog space before two-hot encoding:
```
td_target_symlog = symlog(r + gamma * (1 - done) * V_next_actual)
reward_target_symlog = symlog(r)
```

Where `symlog(x) = sign(x) * ln(1 + |x|)` and `symexp(x) = sign(x) * (exp(|x|) - 1)`.

### Two Separate Optimizers

- **World model optimizer**: encoder + dynamics + reward + termination + Q-ensemble
  with `enc_lr_scale=0.3` applied to encoder params
- **Policy optimizer**: policy head only, full `learning_rate=3e-4`


Analysis: Why convergence is ~10x slower than reference                                                                                                                                   
                                                                                                                                                                                            
  1. No true BPTT through dynamics (HIGH IMPACT)                                                                                                                                            
                                                                                                                                                                                            
  The reference builds a single computation graph across all H horizon steps and calls .backward() once — gradients from step t+1's consistency loss flow back through step t's dynamics.
  Our Mojo code zeros grad_z_pred_buf at the start of each step (line 1181), so gradient only flows through ONE dynamics step at a time. This severely weakens the world model learning
  signal.

  2. No MPPI for data collection (HIGH IMPACT)

  Reference uses MPPI planning (mpc=true) for data collection by default. Our code uses direct policy sampling (use_mppi=False). MPPI provides much better exploration by look-ahead
  planning through the learned world model.

  3. Entropy scaling by action dimension (MEDIUM IMPACT)

  Reference computes scaled_entropy = -log_prob * (scaled_log_prob / (log_prob + 1e-8)) where scaled_log_prob = log_prob * action_dim. This effectively scales entropy by ACTION_DIM (=6 for
   HalfCheetah). Our code uses raw entropy without this scaling — 6x weaker entropy regularization.

  4. Dynamic discount (LOW IMPACT)

  Reference computes discount = min(max((ep_len/5 - 1)/(ep_len/5), 0.95), 0.995). For 1000-step episodes, discount=0.995. We use gamma=0.99 — slightly lower.

  5. Network capacity (LOW IMPACT)

  Reference defaults: latent_dim=512, mlp_dim=512. We use 256/256 — 4x fewer parameters.