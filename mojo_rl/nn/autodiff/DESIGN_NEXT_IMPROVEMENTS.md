# Autodiff Next Improvements — DX & Performance

## Context

Research done 2026-03-19 on latest Mojo features (v0.26.1-v0.26.2) to identify
improvements to the autodiff composition system beyond what's in DESIGN_SAC_COMPOSITION.md.

**Current state**: ComputeGraph, all model-free autodiff losses, GPU support — all done.
Dreamer port to ComputeGraph in progress.

---

## New Mojo Features (Relevant)

| Feature | Version | What It Enables |
|---------|---------|-----------------|
| **Compile-time reflection** (`struct_field_count`, `struct_field_types`, `__struct_field_ref`) | v0.26.1+ | Auto-derive PARAM_SIZE, auto-serialize, auto-init |
| **Conditional trait conformance** (`where conforms_to(T, Trait)`) | v0.26.2 | GPU support opt-in at type level |
| **`trait_downcast`** | v0.26.2 | Single training loop for CPU+GPU |
| **Typed errors** | v0.26.1 | Dimension mismatch errors with structured info |
| **`@register_passable` → `RegisterPassable` trait** | v0.26.2 | Cleaner DiffOp definitions |

**Still NOT possible** (fundamental Mojo gaps):
- **No closures/lambdas** → Can't build tape-based autograd (`loss.backward()` with dynamic graph)
- **No runtime polymorphism** → Can't do `layers.append(Linear(64))` at runtime
- These block eager-mode PyTorch-style DX. Compile-time graph definition remains the only path.

**Note**: For simple MLPs, our DX is already at or better than PyTorch:
```mojo
# Ours (more concise)
comptime DQN = Sequential[LinearReLU[4, 128], LinearReLU[128, 128], Linear[128, 2]]
```
```python
# PyTorch (more verbose)
class DQN(nn.Module):
    def __init__(self, n_obs, n_act):
        self.layer1 = nn.Linear(n_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_act)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
```

The DX gap is in **complex DAGs** (ComputeGraph with opaque node indices) and **boilerplate** (param assembly).

---

## Improvement 1: Named ComputeGraph Nodes (DX — Huge Win)

**Priority: HIGH** | Status: TODO

### Problem

Current ComputeGraph uses integer indices for node references — hard to read, error-prone:
```mojo
comptime SACGraph = ComputeGraph[
    GNode[ActorModel, -1],              # 0
    GNode[RSample[6], 0],               # 1
    GNode[Slice[7, 0, 6], 1],           # 2
    GNode[ConcatOp, -1, 2],             # 3 — what's node 2 again?
    GNode[Critic1, 3],                  # 4
    GNode[Critic2, 3],                  # 5
    GNode[Min[1], 4, 5],               # 6
    GNode[Slice[7, 6, 7], 1],           # 7
    GNode[SACLossOp, 6, 7],             # 8
]
```

### Proposed Solution

Use `StringLiteral` compile-time params as node names. Name→index resolution at compile time
via `comptime for` over the node list:

```mojo
comptime SACGraph = ComputeGraph[
    GNode["actor",       ActorModel,     "input"],
    GNode["rsample",     RSample[6],     "actor"],
    GNode["action",      Slice[7, 0, 6], "rsample"],
    GNode["critic_in",   Identity[23],   "input", "action"],
    GNode["Q1",          Critic1,        "critic_in"],
    GNode["Q2",          Critic2,        "critic_in"],
    GNode["min_q",       Min[1],         "Q1", "Q2"],
    GNode["log_prob",    Slice[7, 6, 7], "rsample"],
    GNode["loss",        SACLossOp,      "min_q", "log_prob"],
]
```

### Implementation Sketch

```mojo
struct GNode[
    name: StringLiteral,
    Op: Model,
    in0_name: StringLiteral = "input",   # "input" = graph external input
    in1_name: StringLiteral = "",         # "" = unused (single input)
](GraphNode):
    # Name→index resolution done by ComputeGraph at graph construction time
    # via comptime for loop over all nodes, string comparison
    ...
```

Inside `ComputeGraph`, resolve names to indices:
```mojo
@staticmethod
fn _resolve_index[target_name: StringLiteral]() -> Int:
    """Resolve node name to index. 'input' = -1, '' = -2."""
    comptime if target_name == "input":
        return -1
    comptime if target_name == "":
        return -2
    comptime for i in range(Self.N):
        comptime if Self.node_types[i].NAME == target_name:
            return i
    # Compile-time error if name not found
    comptime assert False, "Unknown node name: " + target_name
    return -1
```

### Open Question

Need to verify that `StringLiteral` comparison works in `comptime for` loops.
If not, could fall back to integer IDs but keep names as documentation-only annotations.
Could also use integer enum-style constants:
```mojo
comptime ACTOR = 0
comptime RSAMPLE = 1
# ... then GNode[ActorModel, INPUT, UNUSED] etc.
```
Less elegant but guaranteed to work.

---

## Improvement 2: Compile-Time Dimension Checking (DX — Bug Prevention)

**Priority: HIGH** | Status: TODO

### Problem

Dimension mismatches between composed layers are silent — only caught at runtime (or worse,
produce wrong results silently).

### Approach: `comptime assert` (NOT `where` clauses)

`where` clauses on variadic types (like `Sequential[*LAYERS]`) are tricky — Mojo's constraint
solver may not handle variadic iteration well. Instead, use `comptime assert` which we've
already validated works well (e.g., for checking `dtype` is floating point):

```mojo
struct Sequential[*LAYERS: Model](Model):
    # Validate dimension chain at type instantiation
    comptime for i in range(Self.N - 1):
        comptime assert(
            Self.layer_types[i].OUT_DIM == Self.layer_types[i + 1].IN_DIM,
            "Sequential dimension mismatch: layer "
            + str(i) + " OUT_DIM=" + str(Self.layer_types[i].OUT_DIM)
            + " != layer " + str(i + 1) + " IN_DIM=" + str(Self.layer_types[i + 1].IN_DIM)
        )
```

Similarly for `AutoDiffChain`, `ComputeGraph` (validate that each node's OP_IN_DIM matches
the sum of its input sources' OUT_DIMs), etc.

### Impact

Catches shape bugs at compile time with clear error messages instead of silent corruption
or cryptic runtime crashes.

---

## Improvement 3: CompositeParams (DX — Boilerplate Reduction)

**Priority: MEDIUM** | Status: TODO (design in DESIGN_SAC_COMPOSITION.md)

### Problem

Each autodiff actor loss (MaxEnt, DPG, TD3) has ~60 lines of identical param assembly/scatter:
```mojo
var combined_params = InlineArray[Scalar[dtype], TOTAL_PS](uninitialized=True)
for i in range(TOTAL_PS): combined_params[i] = 0.0
for i in range(ACTOR_PS): combined_params[i] = actor_params.ptr[i]
for i in range(CRITIC_PS): combined_params[CRITIC1_OFF + i] = critic_params.ptr[i]
for i in range(CRITIC_PS): combined_params[CRITIC2_OFF + i] = critic2_params.ptr[i]
# ... and 30+ lines for gradient scattering
```

### Solution

`CompositeParams[*MODELS: Model]` — compile-time helper with auto-aligned offsets.
Full design already in DESIGN_SAC_COMPOSITION.md § CompositeParams.

**Impact**: ~60 lines → ~4 lines per algorithm.

Could potentially use new reflection APIs (`struct_field_types`) for auto-discovery,
but explicit variadic params are simpler and sufficient.

---

## Improvement 4: Element-Wise Fusion Pass (Performance)

**Priority: MEDIUM** | Status: TODO

### Problem

Current `AutoFused` only fuses `MatMul+BiasAdd(+Activation)`. But complex autodiff graphs
(SAC, Dreamer) have chains of element-wise ops that each launch separate GPU kernels:

```
Scale → NegateOp → BiasAdd  = 3 kernels (3 global memory round-trips)
Could be:  output[i] = -(input[i] * scale) + bias  = 1 kernel
```

### Approach

Add a second fusion pass after the existing MatMul+BiasAdd pass. Merge consecutive
**element-wise ops** (ops where `IN_DIM == OUT_DIM`, `CACHE_SIZE == 0`,
`OP_WORKSPACE_PER_SAMPLE == 0`) into a single `FusedElementWise` kernel.

Candidates: Scale, NegateOp, BiasAdd (standalone), ReLU, Tanh, Sigmoid, Mish.

### Challenge

Composing the element-wise functions at compile time. Need a way to chain:
```mojo
# Pseudocode for fused Scale[d,2,1] → NegateOp[d] → BiasAdd[d]:
fn fused_forward(input: Scalar, bias: Scalar) -> Scalar:
    return -(input * 2.0) + bias
```

Could use an `ElementWiseOp` sub-trait with `fn apply(x: Scalar, params...) -> Scalar`
and `fn grad(x: Scalar, dy: Scalar, params...) -> (dx, dparams...)`. Then the fused
kernel iterates the chain at compile time.

With `comptime if conforms_to(Op, ElementWiseOp)` (new in v0.26.2), the fusion pass
can detect fusable ops cleanly.

### Note

This mainly matters for GPU where kernel launch overhead is significant.
On CPU, the element-wise loops are already fast. Prioritize this when GPU-heavy
graphs (Dreamer) show kernel-launch bottlenecks in profiling.

---

## Improvement 5: Conditional GPU Conformance (DX — Architecture)

**Priority: LOW** | Status: TODO

### Problem

Currently, CPU and GPU code paths are separate — models either have GPU methods or don't.
Training code must know which path to use.

### Approach (New Mojo Feature)

With conditional trait conformance:
```mojo
struct AutoFused[*OPS: DiffOp](
    Model,
    GPUModel where all_ops_have_gpu[*OPS](),
):
    # GPU methods only compiled when all ops support GPU
```

### Impact

Eliminates separate CPU/GPU model definitions. A single training loop can use
`comptime if conforms_to(M, GPUModel)` to dispatch.

Lower priority because the current dual-path approach works fine —
this is a cleanliness improvement, not a capability one.

---

## Summary

| # | Improvement | Priority | Effort | Impact |
|---|-----------|----------|--------|--------|
| 1 | Named ComputeGraph nodes | HIGH | Medium | Huge DX win for DAGs (Dreamer, SAC) |
| 2 | `comptime assert` dimension checking | HIGH | Small | Catches shape bugs at compile time |
| 3 | CompositeParams | MEDIUM | Medium | -60 lines/algorithm |
| 4 | Element-wise fusion pass | MEDIUM | Large | GPU perf for complex graphs |
| 5 | Conditional GPU conformance | LOW | Medium | Cleaner architecture |

## What's Blocked (Waiting on Mojo)

- **Eager-mode DX** (`x = relu(linear(x))` with automatic backward): Needs closures/lambdas
- **Dynamic model building** (`layers.append(Linear(64))`): Needs runtime polymorphism
- These are language-level gaps. Our compile-time approach is the best possible given current Mojo.
