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

`ComputeGraph` is a compile-time DAG builder that composes Models
with explicit dataflow edges.

#### Design: Fixed-Arity Nodes via GraphNode Trait

Rather than variadic input indices (which hit Mojo limitations), each node
has a fixed maximum of 2 input sources. This covers all RL algorithm graphs:

```mojo
trait GraphNode:
    """A node in a ComputeGraph DAG.

    Each node wraps a Model and declares its input sources:
      IN0: Index of first predecessor (-1 = graph external input)
      IN1: Index of second predecessor (-2 = unused, single input)

    When IN1 != -2, the outputs of IN0 and IN1 are concatenated to form
    this node's input. The concatenated dimension must equal the inner
    Model's IN_DIM.

    Nodes with 3+ inputs can be handled by chaining concat nodes.
    """
    comptime IN0: Int      # First input source (-1 = graph input, -2 = unused)
    comptime IN1: Int      # Second input source (-2 = unused)

    # Expose inner Model's compile-time constants
    comptime OP_IN_DIM: Int
    comptime OP_OUT_DIM: Int
    comptime OP_PARAM_SIZE: Int
    comptime OP_CACHE_SIZE: Int
    comptime OP_WORKSPACE_SIZE_PER_SAMPLE: Int

    # Delegate to inner Model
    @staticmethod
    fn initialize_params[INIT](mut params): ...
    @staticmethod
    fn forward[BATCH](input, mut output, params, mut cache): ...
    @staticmethod
    fn backward[BATCH](grad_output, mut grad_input, params, cache, mut grads): ...
    # + GPU variants


struct GNode[Op: Model, in0: Int = -1, in1: Int = -2](GraphNode):
    """Concrete graph node wrapping any Model type.

    Usage:
        GNode[LinearReLU[17, 256], -1]      # Single input from graph input
        GNode[CriticModel, 2]                # Single input from node 2
        GNode[MinOp[1], 3, 4]               # Concat outputs of nodes 3 and 4
    """
    comptime IN0 = in0
    comptime IN1 = in1
    comptime OP_IN_DIM = Op.IN_DIM
    comptime OP_OUT_DIM = Op.OUT_DIM
    comptime OP_PARAM_SIZE = Op.PARAM_SIZE
    comptime OP_CACHE_SIZE = Op.CACHE_SIZE
    comptime OP_WORKSPACE_SIZE_PER_SAMPLE = Op.WORKSPACE_SIZE_PER_SAMPLE
    # All methods delegate to Op
```

#### ComputeGraph Structure

```mojo
struct ComputeGraph[*NODES: GraphNode](Model):
    """Compile-time differentiable DAG.

    Nodes are executed in index order (forward) and reverse (backward).
    All buffer sizes, offsets, and fan-out points resolved at compile time.

    Memory layout:
      Cache (per sample):
        [act_0 | act_1 | ... | act_{N-1} | cache_0 | cache_1 | ... | cache_{N-1}]
        Activations stored for backward pass + per-node op caches.

      Workspace (temporary, per sample):
        Grad activations:  [grad_act_0 | ... | grad_act_{N-1}]
        Concat buffer:     [max_concat_dim]  (reusable across nodes)
        Op workspace:      [max(ws_per_node)]  (reusable)

    Fan-out handling (automatic):
      During backward, processing nodes in reverse order:
      1. Run node_i's VJP: grad_act[i] → grad_input
      2. For each input source (in0, in1):
         - ADD grad_input portion to grad_act[source]
      Multiple consumers naturally accumulate into the same grad_act buffer.

    Compile-time constants:
      IN_DIM     = inferred from nodes referencing -1 (graph input)
      OUT_DIM    = last node's OP_OUT_DIM
      PARAM_SIZE = sum of _align4(node.OP_PARAM_SIZE) for each node
      CACHE_SIZE = sum of (node.OP_OUT_DIM + node.OP_CACHE_SIZE) per node
    """

    comptime node_types = Variadic.types[T=GraphNode, *Self.NODES]
    comptime N = Variadic.size(Self.node_types)

    # --- Dimension inference ---
    # IN_DIM: dimension of graph's external input.
    # Inferred from nodes whose IN0 == -1. When such a node has a single
    # input (IN1 == -2), IN_DIM = node.OP_IN_DIM. When it concats with
    # another source, the portion from -1 is computed by subtraction.
    # For simplicity, the first node with IN0 == -1 and IN1 == -2 sets IN_DIM.

    comptime OUT_DIM: Int = Self.node_types[Self.N - 1].OP_OUT_DIM

    # --- Offset helpers (same pattern as Sequential/AutoDiffChain) ---

    @staticmethod
    fn _act_offset[idx: Int]() -> Int:
        """Per-sample offset to node idx's activation in cache."""
        var total = 0
        comptime for j in range(idx):
            total += Self.node_types[j].OP_OUT_DIM
        return total

    @staticmethod
    fn _cache_offset[idx: Int]() -> Int:
        """Per-sample offset to node idx's op cache in cache."""
        var total = Self._total_act_size()  # After all activations
        comptime for j in range(idx):
            total += Self.node_types[j].OP_CACHE_SIZE
        return total

    @staticmethod
    fn _param_offset[idx: Int]() -> Int:
        """Aligned offset to node idx's params."""
        var total = 0
        comptime for j in range(idx):
            total += _align4(Self.node_types[j].OP_PARAM_SIZE)
        return total
```

#### SAC Actor Loss as a ComputeGraph

```mojo
comptime SACGraph = ComputeGraph[
    GNode[ActorModel, -1],              # 0: obs → [mean, log_std]
    GNode[RSample[6], 0],               # 1: → [action(6), log_prob(1)]
    GNode[Slice[7, 0, 6], 1],           # 2: → action(6)  (drop log_prob)
    GNode[ConcatOp, -1, 2],             # 3: [obs(17), action(6)] = critic_input(23)
    GNode[Critic1, 3],                  # 4: → Q1
    GNode[Critic2, 3],                  # 5: → Q2   (fan-out from node 3!)
    GNode[Min[1], 4, 5],                # 6: → min_Q
    GNode[Slice[7, 6, 7], 1],           # 7: → log_prob(1)  (from node 1)
    GNode[SACLossOp, 6, 7],             # 8: → alpha * log_prob - min_Q
]
# Fan-out: node 3 feeds both 4 and 5 → grads auto-accumulated
# Fan-out: node 1 feeds both 2 and 7 → grads auto-accumulated
```

#### Dreamer World Model as a ComputeGraph

This is the primary motivation — true DAGs with 4+ fan-out:

```mojo
comptime DreamerGraph = ComputeGraph[
    GNode[Encoder, -1],                 # 0: obs → latent
    GNode[TransitionModel, 0],          # 1: latent → next_latent
    GNode[RewardPredictor, 0],          # 2: latent → reward_hat
    GNode[ValuePredictor, 0],           # 3: latent → value_hat
    GNode[Decoder, 0],                  # 4: latent → obs_hat
    GNode[MultiLossOp, 1, 2, 3, 4],    # 5: combined loss
]
# Fan-out: node 0 feeds 1, 2, 3, 4 → grads accumulated from all 4
```

Note: Dreamer's 4-way fan-out from latent would need either:
- A `FanOut4` that outputs concat(1,2,3,4) then route to MultiLossOp
- Or extend GNode to support IN2/IN3 (still fixed arity, just higher)
- Or chain: `GNode[ConcatOp, 1, 2]` → `GNode[ConcatOp, prev, 3]` → etc.

#### Key Design Decisions

1. **Fixed 2-input arity** avoids Mojo variadic-per-node issues. Covers
   DDPG/TD3/SAC/DQN/PPO directly. Dreamer needs concat chains (still works).

2. **Nodes use Model trait** (not DiffOp). This means existing composed types
   (Sequential, SkipConcat, etc.) work as node ops — no reimplementation needed.

3. **SliceOp handles partial reads**. Instead of encoding slice ranges in
   GraphNode, use Slice nodes to extract subsets from predecessor outputs.
   This keeps GraphNode simple and reuses existing SliceOp.

4. **ConcatOp for multi-input**. When IN1 != -2, the graph concatenates
   IN0 and IN1 outputs into a temporary buffer before feeding to the node's
   Model. This is implicit in the graph — no explicit ConcatOp needed for
   2-input nodes. A dedicated ConcatOp (identity with 2 inputs) is only
   needed when you want to name the concat result for further fan-out.

5. **Activation storage in cache**. All node outputs are stored in the cache
   buffer during forward (needed for backward). This is like Sequential's
   intermediate buffers but persisted across forward/backward calls.

6. **Grad accumulation is natural**. Processing backward in reverse order,
   each node's VJP writes grad_input. This is then ADDED to the predecessor's
   grad_act buffer. Multiple consumers add to the same buffer = fan-out.

#### Incremental Path to ComputeGraph

**Phase 2a** ✅: `SliceOp`, `NegateOp`, `SplitApply` — completed.
Enables all model-free algorithms via combinator trees.

**Phase 2b** ✅: `FanOut` — completed. N-copy fan-out with separate params.

**Phase 2c** (current): `ComputeGraph[*NODES: GraphNode]` — the general DAG builder.
Start with CPU-only prototype, then add GPU. The existing combinators remain
available as convenience sugar for simple topologies.

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
| 2c | `ComputeGraph` | ✅ Done (CPU + GPU) | Full compile-time DAG builder |
| — | `CompositeParams` | Future | Reduce param assembly/scatter boilerplate |

## CompositeParams — Boilerplate Reduction (Future)

The existing autodiff actor losses (AutodiffMaxEntLoss, AutodiffDPGLoss, AutodiffTD3Loss)
share ~60 lines of identical param assembly/scatter boilerplate per algorithm:

```mojo
# Current: 30+ lines per algorithm for assembly
var combined_params = InlineArray[Scalar[dtype], TOTAL_PS](uninitialized=True)
for i in range(TOTAL_PS): combined_params[i] = 0.0
for i in range(ACTOR_PS): combined_params[i] = actor_params.ptr[i]
for i in range(CRITIC_PS): combined_params[CRITIC1_OFF + i] = critic_params.ptr[i]
for i in range(CRITIC_PS): combined_params[CRITIC2_OFF + i] = critic2_params.ptr[i]
# ... and mirror for gradient scattering
```

`CompositeParams` would be a compile-time helper that auto-manages alignment and offsets:

```mojo
struct CompositeParams[*MODELS: Model]:
    """Auto-aligned param layout for multi-model compositions.

    Handles:
    - 4-element GPU alignment padding between models
    - Offset computation
    - Assembly (copy separate → combined) and scatter (combined → separate)
    """
    comptime model_types = Variadic.types[T=Model, *Self.MODELS]
    comptime N = Variadic.size(Self.model_types)

    @staticmethod
    fn _offset[idx: Int]() -> Int:
        var total = 0
        comptime for j in range(idx):
            total += _align4(Self.model_types[j].PARAM_SIZE)
        return total

    @staticmethod
    fn _total_size() -> Int:
        var total = 0
        comptime for j in range(Self.N - 1):
            total += _align4(Self.model_types[j].PARAM_SIZE)
        total += Self.model_types[Self.N - 1].PARAM_SIZE
        return total

    comptime TOTAL_SIZE: Int = Self._total_size()

    @staticmethod
    fn assemble(
        mut dst: InlineArray[Scalar[dtype], Self.TOTAL_SIZE],
        *sources: UnsafePointer[Scalar[dtype]],
    ):
        """Copy N separate param buffers into one combined buffer."""
        for i in range(Self.TOTAL_SIZE):
            dst[i] = 0.0  # Zero padding
        comptime for m in range(Self.N):
            var off = Self._offset[m]()
            var sz = Self.model_types[m].PARAM_SIZE
            for i in range(sz):
                dst[off + i] = sources[m][i]

    @staticmethod
    fn scatter(
        src: InlineArray[Scalar[dtype], Self.TOTAL_SIZE],
        *dsts: UnsafePointer[Scalar[dtype]],
    ):
        """Copy combined grads back to N separate buffers."""
        comptime for m in range(Self.N):
            var off = Self._offset[m]()
            var sz = Self.model_types[m].PARAM_SIZE
            for i in range(sz):
                dsts[m][i] = src[off + i]
```

**Impact**: Each actor loss drops ~60 lines of manual param/grad management to ~4 lines.
This is independent of ComputeGraph and can be added whenever convenient.

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
