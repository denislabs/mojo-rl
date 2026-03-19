# SAC Actor Loss via Autodiff Composition

## The Problem

SAC's actor loss requires gradient flow through a DAG:

```
obs ──→ Actor ──→ RSample ──→ [action, log_prob]
                                  │
obs ──────────────────────────────┤
                                  ▼
                          [obs, action] ──→ Critic1 ──→ Q1 ──┐
                          [obs, action] ──→ Critic2 ──→ Q2 ──┤
                                                              ▼
                                                          min(Q1, Q2)
                                                              │
                          loss = alpha * log_prob - min_Q  ◄──┘
```

Currently this requires ~400 lines of manual backward code in `actor_loss.mojo`.

## The Solution: Compose Existing Autodiff Primitives

With the new primitives (`RSampleOp`, `MinOp`) and combinators (`SkipConcat`, `DualPath`),
the SAC actor loss becomes a **compile-time Model type definition**:

```mojo
from mojo_rl.nn.model import Sequential, Linear, LinearReLU, LinearTanh
from mojo_rl.nn.autodiff.primitives import RSampleOp, MinOp
from mojo_rl.nn.autodiff.combinators import Parallel, SkipConcat, DualPath

# Example: obs_dim=17, action_dim=6, hidden=256

# Step 1: Define Actor (same as before)
comptime ActorModel = Sequential[
    LinearReLU[17, 256],
    LinearReLU[256, 256],
    Parallel[Linear[256, 6], LinearTanh[256, 6]],  # [mean || tanh(log_std)]
]
# ActorModel: IN_DIM=17, OUT_DIM=12 (6+6)

# Step 2: Define Critic (same as before)
comptime CriticModel = Sequential[
    LinearReLU[23, 256],   # obs_dim + action_dim = 17 + 6 = 23
    LinearReLU[256, 256],
    Linear[256, 1],
]
# CriticModel: IN_DIM=23, OUT_DIM=1

# Step 3: NEW — Define the composed actor-critic loss as a Model type
#
# SkipConcat passes obs through unchanged while Actor+RSample produces [action, log_prob]
# Output: [obs(17) || action(6) || log_prob(1)] = 24 dims
comptime ActorWithSkip = SkipConcat[
    Sequential[ActorModel, RSampleOp[6]],
]
# ActorWithSkip: IN_DIM=17, OUT_DIM=24 (17+6+1)

# Then we need to:
# 1. Extract [obs, action] = first 23 dims → feed to twin critics
# 2. Extract log_prob = last dim → for entropy term
# 3. Forward through twin critics → min(Q1, Q2)
#
# DualPath runs both critics on the same input:
comptime TwinCritic = DualPath[CriticModel, CriticModel]
# TwinCritic: IN_DIM=23, OUT_DIM=2 (Q1, Q2)

# MinOp takes [Q1, Q2] → min
# Sequential[TwinCritic, MinOp[1]] gives the full critic path
```

## What This Replaces

### Before (actor_loss.mojo MaxEntLoss.update_actor_cpu — 260 lines):
```
1. Forward actor with cache
2. Extract mean + log_std from Parallel output
3. Generate noise
4. rsample_with_cache → actions + log_probs
5. Concat obs + actions → critic_input
6. Forward both critics with cache
7. Compute min(Q1, Q2) mask → dQ gradient seeds
8. Backward both critics → d_critic_input
9. Combine d_ci from both critics
10. Extract d_actions
11. Setup entropy gradient
12. rsample_backward → grad_mean, grad_log_std
13. Build actor_grad with AFFINE_SCALE
14. Backward actor
```

### After (using composed Model):
```mojo
# The entire actor loss graph as a single Model
comptime SACActorGraph = Sequential[
    ActorWithSkip,         # obs → [obs, action, log_prob]
    # ... extract [obs,action], forward critics, min, combine with log_prob
]

# Forward + backward is just:
SACActorGraph.forward[BATCH](obs, loss_output, params, cache)
SACActorGraph.backward[BATCH](grad_loss, grad_obs, params, cache, grads)
```

## Remaining Gap: The Slice/Extract Pattern

The composition above works for the actor path and the twin critic path independently.
The remaining challenge is connecting them — we need to:

1. **Slice** [obs, action] from the SkipConcat output (drop log_prob)
2. **Forward through critics** on that slice
3. **Combine** the critic output with log_prob for the loss

This requires a `SliceOp` (extract subset of dimensions) and a way to merge
the log_prob branch back in. Two approaches:

### Approach A: SliceOp DiffOp
```mojo
struct SliceOp[in_dim: Int, start: Int, end: Int](DiffOp):
    """Extract dimensions [start:end] from input."""
    comptime IN_DIM = in_dim
    comptime OUT_DIM = end - start
    # Forward: output = input[:, start:end]
    # Backward: grad_input[:, start:end] = grad_output, rest = 0
```

### Approach B: Custom SACLossModel (pragmatic)
A purpose-built Model that encapsulates the full DAG, parameterized by
Actor and Critic types. This is less general but simpler:

```mojo
struct SACLossModel[Actor: Model, Critic: Model, action_dim: Int](Model):
    """Composed SAC actor-critic loss.

    Forward:  obs → Actor → RSample → [action, log_prob]
              [obs, action] → DualPath[Critic, Critic] → [Q1, Q2]
              → MinOp → min_Q
              output = [-min_Q, log_prob]  (for external loss computation)

    Backward: Full VJP through the DAG automatically.
    """
    comptime IN_DIM = Actor.IN_DIM  # obs_dim
    comptime OUT_DIM = 2            # [-min_Q, log_prob]
    comptime PARAM_SIZE = Actor.PARAM_SIZE + 2 * Critic.PARAM_SIZE
    ...
```

## Phased Roadmap

### Phase 1: New DiffOp Primitives (DONE)

Add the missing differentiable operations that force manual backward code today.

| Component | Status | Purpose |
|-----------|--------|---------|
| `RSampleOp` | ✅ Done | Reparameterized sampling as DiffOp |
| `MinOp` | ✅ Done | Elementwise min with gradient routing |
| `SkipConcat` | ✅ Done | Pass input alongside Inner output |
| `DualPath` | ✅ Done | Twin network forward + gradient split |
| `RSample` / `Min` | ✅ Done | Model wrappers via AutoDiffChain |

These primitives already enable expressing actor and critic paths declaratively.
The remaining gap is **connecting** actor output to critic input when the graph
has fan-out points (obs used by both skip and actor) and merge points (log_prob
and min_Q combined for the loss).

### Phase 2: ComputeGraph — Compile-Time DAG with Fan-Out Support

The current composition tools (Sequential, Parallel, Residual) only support
**linear chains** and **same-input fan-out**. RL loss graphs are **DAGs** where:

- A node's output feeds into **multiple** downstream nodes (fan-out)
- A node takes inputs from **multiple** predecessors (fan-in / merge)
- Gradients must **accumulate** at fan-out points during backward

`ComputeGraph` would be a compile-time DAG builder that composes DiffOps/Models
with explicit dataflow edges.

#### Design Sketch

```mojo
# A node in the graph: an op + which previous nodes feed into it
struct GraphNode[op: DiffOp, *input_indices: Int]:
    """Compile-time DAG node.

    op:            The differentiable operation at this node
    input_indices: Indices of predecessor nodes whose outputs are
                   concatenated to form this node's input.
                   Index -1 = the graph's external input.
    """
    ...

# The graph: a topologically-sorted list of nodes
struct ComputeGraph[*nodes: GraphNode](Model):
    """Compile-time differentiable DAG.

    Nodes are executed in index order (forward) and reverse order (backward).
    All buffer sizes, offsets, and fan-out points are resolved at compile time.

    Memory layout (per sample):
      activations: [node_0_out | node_1_out | ... | node_N-1_out]
      caches:      [node_0_cache | node_1_cache | ... | node_N-1_cache]
      grad_acts:   [node_0_grad | node_1_grad | ... | node_N-1_grad]

    Fan-out handling:
      When node_i's output feeds into nodes j and k, backward accumulates:
        grad_activations[i] = grad_from_j + grad_from_k
      before running node_i's VJP.

    Compile-time constants:
      IN_DIM     = graph input dimension
      OUT_DIM    = last node's output dimension
      PARAM_SIZE = sum of all nodes' PARAM_SIZE
      CACHE_SIZE = sum of all nodes' CACHE_SIZE
    """

    # Forward: execute nodes 0..N-1 in order
    @staticmethod
    fn forward[BATCH](input, output, params, cache):
        comptime for i in range(Self.N):
            # Gather inputs from predecessor nodes (or external input)
            # Concat them into this node's input buffer
            # Run node_i.op.eval()
            ...

    # Backward: execute nodes N-1..0 in reverse
    @staticmethod
    fn backward[BATCH](grad_output, grad_input, params, cache, grads):
        # Initialize grad_activations[last_node] = grad_output
        comptime for _ri in range(Self.N):
            comptime i = Self.N - 1 - _ri
            # Run node_i.op.vjp() using accumulated grad_activations[i]
            # Scatter grad_input contributions to predecessor grad buffers
            # (this is where fan-out accumulation happens)
            ...
```

#### SAC Actor Loss as a ComputeGraph

```mojo
comptime SACLossGraph = ComputeGraph[
    # Node 0: obs → Actor → [mean, tanh_log_std]
    GraphNode[ActorOps, -1],

    # Node 1: [mean, tanh_log_std] → RSample → [action, log_prob]
    GraphNode[RSampleOp[6], 0],

    # Node 2: concat(obs, action) → critic_input
    #   inputs: external(-1) for obs, node 1 for action
    #   Needs a ConcatGatherOp that takes specific slices from multiple sources
    GraphNode[ConcatGatherOp[obs=(-1, 0, 17), act=(1, 0, 6)], -1, 1],

    # Node 3: critic_input → Critic1 → Q1
    GraphNode[Critic1Ops, 2],

    # Node 4: critic_input → Critic2 → Q2 (fan-out from node 2)
    GraphNode[Critic2Ops, 2],

    # Node 5: [Q1, Q2] → Min → min_Q
    GraphNode[MinOp[1], 3, 4],

    # Node 6: [min_Q, log_prob] → weighted sum → loss
    #   loss = alpha * log_prob - min_Q
    GraphNode[SACLossOp[alpha], 5, 1],
]
```

#### Key Implementation Challenges

1. **Compile-time input gathering**: Each node's input may come from multiple
   predecessors. The graph must compute concat offsets at compile time and
   generate the appropriate copy/view code.

2. **Fan-out gradient accumulation**: When node 2's output feeds into both
   node 3 and node 4, backward must sum `grad_from_3 + grad_from_4` before
   running node 2's VJP. This requires a compile-time analysis of which nodes
   consume each node's output (the "consumers list").

3. **Partial slicing**: Node 2 needs `action` from node 1's output (dims 0..5)
   but NOT `log_prob` (dim 6). Similarly, node 6 needs `log_prob` from node 1
   and `min_Q` from node 5. This requires either:
   - A `SliceOp` / `GatherOp` that extracts specific dimensions
   - Or encoding slice ranges in the `GraphNode` definition

4. **Multi-network params**: The graph contains params for Actor, Critic1, and
   Critic2. Each node's params must be at a known offset within the graph's
   flat param tensor. This is similar to how Sequential handles per-layer offsets.

5. **Mojo variadic constraints**: Mojo's `Variadic` system works well for
   homogeneous iteration (`comptime for`), but GraphNode has heterogeneous
   `input_indices`. May need to encode fan-in as fixed-size fields
   (e.g., `max_inputs=4`) rather than variadic.

#### Incremental Path to ComputeGraph

Rather than building the full general-purpose ComputeGraph at once, an
incremental approach:

**Phase 2a**: `SliceOp[in_dim, start, end]` — extract dimension range.
Completes the SAC graph using existing combinators (Sequential, SkipConcat, etc.)
without needing a full DAG system.

**Phase 2b**: `FanOut[Inner, *Consumers]` combinator — a Model that forwards
Inner's output to multiple downstream Models and sums their grad_inputs.
This is a generalization of DualPath that handles arbitrary fan-out.

**Phase 2c**: Full `ComputeGraph[*nodes]` — the general DAG builder. At this
point the existing combinators (Sequential, Parallel, Residual, SkipConcat,
DualPath, FanOut) become syntactic sugar over ComputeGraph node patterns.

### Phase 3: Rewrite Agent Losses as ComputeGraph Definitions

Once ComputeGraph is available, each algorithm's actor loss becomes a
**type alias** instead of a hand-written strategy:

| Algorithm | Manual Code | ComputeGraph Definition |
|-----------|-------------|------------------------|
| DDPG | `DPGLoss.update_actor_cpu` (~150 lines) | ~10 lines |
| TD3 | Similar to DDPG + target noise | ~12 lines |
| SAC | `MaxEntLoss.update_actor_cpu` (~260 lines) | ~15 lines |

The old strategy types (`ActorLoss`, `TargetAction`) could be replaced by
ComputeGraph-based implementations, or kept as wrappers that delegate to
the graph's `forward()` / `backward()`.

This also opens the door for **new algorithms** to be expressed purely as
graph definitions — no manual backward code needed.

## Summary

| Phase | Component | Status | Purpose |
|-------|-----------|--------|---------|
| 1 | `RSampleOp` | ✅ Done | Reparameterized sampling as DiffOp |
| 1 | `MinOp` | ✅ Done | Elementwise min with gradient routing |
| 1 | `SkipConcat` | ✅ Done | Pass input alongside Inner output |
| 1 | `DualPath` | ✅ Done | Twin network forward + gradient split |
| 1 | `RSample` / `Min` | ✅ Done | Model wrappers via AutoDiffChain |
| 2a | `SliceOp` | ✅ Done | Extract dimension range |
| 2a | `NegateOp` | ✅ Done | Elementwise negation |
| 2a | `SplitApply` | ✅ Done | Split input, apply different Models |
| 2a | `AutodiffMaxEntLoss` | ✅ Done | SAC actor loss via composed graph |
| 2a | `AutodiffSACConfig` | ✅ Done | Drop-in SAC config using autodiff loss |
| 2b | `FanOut` | ✅ Done | N copies of same Model, concat outputs |
| 3 | `AutodiffDPGLoss` | ✅ Done | DDPG actor loss via composed graph |
| 3 | `AutodiffTD3Loss` | ✅ Done | TD3 actor loss via composed graph |
| 3 | `AutodiffDDPGConfig` | ✅ Done | Drop-in DDPG config using autodiff |
| 3 | `AutodiffTD3Config` | ✅ Done | Drop-in TD3 config using autodiff |
| 3 | GPU autodiff path | ✅ Done | GPU forward/backward (CUDA; Metal has nested generic limits) |
| 2c | `ComputeGraph` | Future | Full compile-time DAG builder |

## Extending to Other Algorithm Families

The autodiff composition system is not limited to off-policy continuous agents.
Every RL algorithm family has a loss that can be expressed as a composed graph.

### DQN Family

DQN's loss is the simplest — the Q-network IS the actor:

```
obs → QNetwork → Q_values[num_actions] → Gather(action_idx) → Q(s,a)
loss = MSE(Q(s,a), target)
```

| New DiffOp | Purpose |
|-----------|---------|
| `GatherOp` | Select Q-value at action index; backward = sparse gradient at that index |

```mojo
comptime DQNGraph = Sequential[QNetwork, GatherOp]
// loss = MSE(output, target) — handled by existing LossFunction trait
```

**Double DQN**: Same graph — only the target computation changes (argmax from
online network, value from target network), which is outside the loss graph.

**Dueling DQN**: The Q-network itself becomes a composed Model:
```mojo
comptime DuelingQ = Sequential[
    SharedTrunk,
    Parallel[ValueStream, AdvantageStream],  // → [V(s), A(s,a)]
    DuelingCombineOp,                        // Q = V + A - mean(A)
]
```
Needs `DuelingCombineOp` (subtract advantage mean, add value).

### PPO (Discrete + Continuous)

PPO's actor loss uses the clipped surrogate objective:

```
obs → Actor → logits/[mean,std]
    → LogProb(action) → log_pi
    → ratio = exp(log_pi - old_log_pi)
    → surrogate = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
```

| New DiffOp | Purpose |
|-----------|---------|
| `LogProbOp` | Compute log probability of taken action from logits/distribution |
| `RatioOp` | `exp(log_prob - old_log_prob)` with old_log_prob as frozen input |
| `ClipSurrogateOp` | `min(ratio * A, clip(ratio) * A)` with gradient passthrough inside clip range |

```mojo
comptime PPOActorGraph = Sequential[
    ActorModel,
    LogProbOp[action_dim],
    RatioOp,               // uses old_log_prob from rollout buffer
    ClipSurrogateOp[eps],   // uses advantages from rollout buffer
]
```

Key difference from off-policy: PPO needs "external" frozen inputs (old_log_prob,
advantages) that come from the rollout buffer, not from the graph. These can be
passed via the workspace/cache mechanism or as additional DiffOp parameters.

### Algorithm Coverage Summary

| Algorithm Family | Graph Shape | DiffOps | Status |
|-----------------|-------------|---------|--------|
| DDPG | Chain | `SkipConcat`, `Negate` | ✅ Done |
| TD3 | Chain + twin fan-out | + `DualPath`, `Min` | ✅ Done |
| SAC | Chain + twin fan-out + split | + `RSample`, `SplitApply` | ✅ Done |
| DQN / Double DQN | Chain + gather | `GatherOp` | ✅ Primitives done |
| Dueling DQN | Parallel branches | `DuelingCombineOp` | TODO (1 op) |
| PPO (discrete) | Chain + external inputs | `CategoricalLogProbOp`, `RatioOp`, `ClipSurrogateOp` | ✅ Primitives done |
| PPO (continuous) | Chain + external inputs | Same + `RSampleOp` | ✅ Primitives done |
| Dreamer / TD-MPC2 | Multi-head + recurrent | Recurrent ops, multi-loss | Needs ComputeGraph |

### When ComputeGraph Becomes Necessary

The existing combinators handle all algorithms where the loss graph is a
**tree** (possibly with fan-out and split points). ComputeGraph is only needed
for **true DAGs** with arbitrary fan-in from non-adjacent nodes — primarily
model-based RL with world models:

```
obs → Encoder → latent ──→ TransitionModel → next_latent
                    │──→ RewardPredictor → reward_hat
                    │──→ ValuePredictor → value_hat
                    └──→ Decoder → obs_hat

loss = reconstruction_loss + reward_loss + value_loss + KL_loss
```

Here `latent` fans out to 4 downstream consumers, and the loss combines
outputs from all of them. The existing combinators CAN express this
(via nested FanOut + SplitApply), but a ComputeGraph would be more natural.

**Recommendation**: Build ComputeGraph when a model-based algorithm (Dreamer,
TD-MPC2 world model training) is the next target. The combinator approach
covers all current model-free algorithms cleanly.

## Benefits

1. **Correctness by construction**: No manual gradient stitching → no gradient bugs
2. **Automatic AFFINE_SCALE**: RSampleOp's VJP handles chain rule through log_std rescaling
3. **GPU auto-generation**: Each DiffOp has both CPU and GPU implementations
4. **Reusable**: MinOp, SkipConcat, DualPath are useful beyond SAC (TD3, etc.)
5. **~400 lines → ~20 lines**: Actor loss strategy becomes a type alias
6. **New algorithms for free**: Express any actor-critic loss as a graph definition
7. **Zero runtime overhead**: All graph topology resolved at compile time
8. **Incremental**: Each algorithm family needs only 1-3 new DiffOps, not a full rewrite
