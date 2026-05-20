"""GPU Gumbel-search MCTS (EfficientZero V2 discrete planner kernels).

**Phase 3 move**: this file is the verbatim port of
``efficient_zero_v2/gpu_mcts.mojo``. The old location is now a thin
re-export shim. Same parallelism strategy as MuZero's GPU MCTS — per-env
trees flat across one DeviceBuffer, one thread per env, traversal
sequential within a tree.

Key differences vs MuZero PUCT GPU MCTS (preserved during the move):
  • Root expansion is restricted to K candidates sampled by Gumbel-Top-k.
  • Sequential Halving runs in `log2(K)` host-orchestrated phases.
  • Non-root selection uses the deterministic visit-balance rule
    `argmax_a [π_improved(a) − N(s,a)/(1+ΣN(s,b))]`, which needs the raw
    policy logits at every node — so we store `node_logits` (not `prior`).
  • The improved policy `π̂ = softmax(logits + σ(completed_Q))` is computed
    over the full action space at the end of search.

The driver (`run_gumbel_search_gpu`) issues one simulation per kernel-launch
batch. The MuZero-style virtual-loss BATCH_SIMS fan-out is not implemented
here — it's a perf optimisation deferred alongside the EZ-V2 training-loop
rewiring on top of the new `GumbelGPUMCTS` orchestrator.

References:
  Danihelka, Guez, Schrittwieser, Silver — *Policy improvement by planning
  with Gumbel*, ICLR 2022.
  Wang, Sun, Li et al. — *EfficientZero V2*, ICML 2024.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from std.math import sqrt, log, exp
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network, GPUNetworkState


comptime MAX_DEPTH: Int = 32


# ═════════════════════════════════════════════════════════════════════════
# State container
# ═════════════════════════════════════════════════════════════════════════


struct EZV2GPUMCTSState[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_K: Int,
](Movable):
    """GPU-resident per-env Gumbel-search trees + scratch buffers.

    All buffers are flat DeviceBuffers; the per-env layout matches MuZero's
    GPU MCTS so that integration with the existing rep/dyn/pred forward
    pipeline is straightforward.

    Parameters:
        N_ENVS: Parallel envs (one tree per env).
        MAX_NODES: Hard upper bound on tree size per env.
        ACT: Discrete action count.
        LATENT: Hidden-state dimension (= RepModel.OUT_DIM = DynModel.OUT_DIM
            minus reward bins).
        BINS: Categorical value/reward bin count.
        MAX_K: Maximum Gumbel-Top-k root candidates. Must be ≤ ACT and a
            power of two for Sequential Halving to use exactly log2(K)
            phases. The driver clips at runtime.
    """

    comptime PRED_OUT: Int = Self.ACT + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS

    # ─── Tree node arrays [N_ENVS × MAX_NODES × ACT] ─────────────────────
    var visit_count: DeviceBuffer[dtype]
    var total_value: DeviceBuffer[dtype]
    var node_logits: DeviceBuffer[dtype]      # raw policy logits per node
    var reward: DeviceBuffer[dtype]
    var child_idx: DeviceBuffer[dtype]

    # ─── Tree node scalars [N_ENVS × MAX_NODES] ──────────────────────────
    var total_visits: DeviceBuffer[dtype]
    var node_value: DeviceBuffer[dtype]       # decoded scalar value at node

    # ─── Hidden state pool [N_ENVS × MAX_NODES × LATENT] ─────────────────
    var hidden_states: DeviceBuffer[dtype]

    # ─── Per-env scalars [N_ENVS] ────────────────────────────────────────
    var node_count: DeviceBuffer[dtype]
    var min_q: DeviceBuffer[dtype]
    var max_q: DeviceBuffer[dtype]

    # ─── Root legality + Gumbel-Top-k state [N_ENVS × *] ─────────────────
    var legal_mask: DeviceBuffer[dtype]       # [N_ENVS × ACT]
    var root_candidates: DeviceBuffer[dtype]  # [N_ENVS × MAX_K]
    var root_gumbels: DeviceBuffer[dtype]     # [N_ENVS × MAX_K]
    var root_active: DeviceBuffer[dtype]      # [N_ENVS × MAX_K]; entry = candidate slot index or -1

    # ─── Selection / expansion scratch (one sim at a time) ───────────────
    var pending_parent: DeviceBuffer[dtype]   # [N_ENVS]
    var pending_action: DeviceBuffer[dtype]   # [N_ENVS]
    var path_lengths: DeviceBuffer[dtype]     # [N_ENVS]
    var leaf_values: DeviceBuffer[dtype]      # [N_ENVS]
    var search_paths: DeviceBuffer[dtype]     # [N_ENVS × MAX_DEPTH]
    var action_paths: DeviceBuffer[dtype]     # [N_ENVS × MAX_DEPTH]

    # ─── Network I/O scratch ─────────────────────────────────────────────
    var root_hidden: DeviceBuffer[dtype]      # [N_ENVS × LATENT] — rep output
    var dyn_input: DeviceBuffer[dtype]        # [N_ENVS × DYN_IN]
    var dyn_output: DeviceBuffer[dtype]       # [N_ENVS × DYN_OUT]
    var pred_input: DeviceBuffer[dtype]       # [N_ENVS × LATENT]
    var pred_output: DeviceBuffer[dtype]      # [N_ENVS × PRED_OUT]

    # ─── Final policy [N_ENVS × ACT] ─────────────────────────────────────
    var policies_out: DeviceBuffer[dtype]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime NA = Self.N_ENVS * Self.MAX_NODES * Self.ACT
        comptime NS = Self.N_ENVS * Self.MAX_NODES
        comptime NH = Self.N_ENVS * Self.MAX_NODES * Self.LATENT

        self.visit_count = ctx.enqueue_create_buffer[dtype](NA)
        self.total_value = ctx.enqueue_create_buffer[dtype](NA)
        self.node_logits = ctx.enqueue_create_buffer[dtype](NA)
        self.reward = ctx.enqueue_create_buffer[dtype](NA)
        self.child_idx = ctx.enqueue_create_buffer[dtype](NA)
        self.total_visits = ctx.enqueue_create_buffer[dtype](NS)
        self.node_value = ctx.enqueue_create_buffer[dtype](NS)
        self.hidden_states = ctx.enqueue_create_buffer[dtype](NH)

        self.node_count = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.min_q = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.max_q = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)

        self.legal_mask = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )
        self.root_candidates = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.MAX_K
        )
        self.root_gumbels = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.MAX_K
        )
        self.root_active = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.MAX_K
        )

        self.pending_parent = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.pending_action = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.path_lengths = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.leaf_values = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.search_paths = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * MAX_DEPTH
        )
        self.action_paths = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * MAX_DEPTH
        )

        self.root_hidden = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.LATENT
        )
        self.dyn_input = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.DYN_IN
        )
        self.dyn_output = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.DYN_OUT
        )
        self.pred_input = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.LATENT
        )
        self.pred_output = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PRED_OUT
        )

        self.policies_out = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )

    def __init__(out self, *, deinit take: Self):
        self.visit_count = take.visit_count^
        self.total_value = take.total_value^
        self.node_logits = take.node_logits^
        self.reward = take.reward^
        self.child_idx = take.child_idx^
        self.total_visits = take.total_visits^
        self.node_value = take.node_value^
        self.hidden_states = take.hidden_states^
        self.node_count = take.node_count^
        self.min_q = take.min_q^
        self.max_q = take.max_q^
        self.legal_mask = take.legal_mask^
        self.root_candidates = take.root_candidates^
        self.root_gumbels = take.root_gumbels^
        self.root_active = take.root_active^
        self.pending_parent = take.pending_parent^
        self.pending_action = take.pending_action^
        self.path_lengths = take.path_lengths^
        self.leaf_values = take.leaf_values^
        self.search_paths = take.search_paths^
        self.action_paths = take.action_paths^
        self.root_hidden = take.root_hidden^
        self.dyn_input = take.dyn_input^
        self.dyn_output = take.dyn_output^
        self.pred_input = take.pred_input^
        self.pred_output = take.pred_output^
        self.policies_out = take.policies_out^

    def zero_tree(self, ctx: DeviceContext) raises:
        """Zero all per-node arrays + the per-env scalars. Required before
        each search to avoid carrying state from the previous call (in
        particular the parent→child links)."""
        ctx.enqueue_memset(self.visit_count, 0)
        ctx.enqueue_memset(self.total_value, 0)
        ctx.enqueue_memset(self.node_logits, 0)
        ctx.enqueue_memset(self.reward, 0)
        ctx.enqueue_memset(self.total_visits, 0)
        ctx.enqueue_memset(self.node_value, 0)
        ctx.enqueue_memset(self.policies_out, 0)
        self.child_idx.enqueue_fill(Scalar[dtype](-1.0))
        self.root_active.enqueue_fill(Scalar[dtype](-1.0))


# ═════════════════════════════════════════════════════════════════════════
# Kernels
# ═════════════════════════════════════════════════════════════════════════
#
# Layout conventions:
#   tree_off = e * MAX_NODES * ACT
#   tv_off   = e * MAX_NODES                  (per-node scalars)
#   h_off    = e * MAX_NODES * LATENT + node_idx * LATENT
#   na_off   = tree_off + node_idx * ACT + a
#   path_off = e * MAX_DEPTH


def gz_scatter_root_hidden_kernel[
    N_ENVS: Int, MAX_NODES: Int, LATENT: Int, dtype: DType,
](
    root_hidden: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ],
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Scatter the contiguous rep-forward output `[N_ENVS × LATENT]` into
    each env's slot-0 hidden state in the strided pool. One thread per env.

    The rep forward writes a contiguous batch buffer; the tree storage
    expects per-env stride `MAX_NODES * LATENT`."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var src = e * LATENT
    var dst = e * MAX_NODES * LATENT  # node 0 for env e
    for i in range(LATENT):
        hidden_states[dst + i] = root_hidden[src + i]


def gz_init_root_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BINS: Int,
    MAX_K: Int,
    PRED_OUT: Int,
    dtype: DType,
](
    node_logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    node_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    legal_mask: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    root_candidates: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ],
    root_gumbels: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ],
    root_active: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ],
    pred_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ],
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    k_actual: Scalar[DType.int32],
    apply_legal: Scalar[DType.uint8],
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Initialize the root node:
        • copies policy logits from pred_output into node_logits[e][0],
          masking illegal actions to a very negative value;
        • decodes the root value into node_value[e][0];
        • samples K Gumbel-Top-k root candidates via Philox + writes their
          action ids and underlying g(a) values; populates root_active.

    One thread per env."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var pred_off = e * PRED_OUT
    var na_off = e * MAX_NODES * ACT
    var ns_off = e * MAX_NODES
    var lm_off = e * ACT
    var k_off = e * MAX_K

    # ── 1. Copy logits + apply legal mask ────────────────────────────────
    for a in range(ACT):
        var raw = rebind[Scalar[dtype]](pred_output[pred_off + a])
        var is_legal = True
        if apply_legal > Scalar[DType.uint8](0):
            var lm = rebind[Scalar[dtype]](legal_mask[lm_off + a])
            is_legal = lm > Scalar[dtype](0.5)
        if is_legal:
            node_logits[na_off + a] = raw
        else:
            node_logits[na_off + a] = Scalar[dtype](-1e9)

    # ── 2. Decode root value (categorical → scalar via inverse h⁻¹) ──────
    var v_max_logit = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
        if v > v_max_logit:
            v_max_logit = v
    var v_sum = Scalar[dtype](0.0)
    for i in range(BINS):
        v_sum += exp(
            rebind[Scalar[dtype]](pred_output[pred_off + ACT + i]) - v_max_logit
        )
    var step_v = (v_max - v_min) / Scalar[dtype](BINS - 1)
    var v_expected = Scalar[dtype](0.0)
    for i in range(BINS):
        var prob = (
            exp(
                rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                - v_max_logit
            )
            / v_sum
        )
        v_expected += prob * (v_min + Scalar[dtype](i) * step_v)
    # Inverse scalar transform h⁻¹ (matches MuZero's `inverse_scalar_transform`)
    var sgn = (
        Scalar[dtype](1.0)
        if v_expected >= Scalar[dtype](0.0)
        else Scalar[dtype](-1.0)
    )
    var abs_y = (
        v_expected
        if v_expected >= Scalar[dtype](0.0)
        else -v_expected
    )
    var eps_h = Scalar[dtype](0.001)
    var inner = sqrt(
        Scalar[dtype](1.0)
        + Scalar[dtype](4.0) * eps_h * (abs_y + Scalar[dtype](1.0) + eps_h)
    )
    var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps_h)
    node_value[ns_off] = sgn * (f * f - Scalar[dtype](1.0))

    # ── 3. Gumbel-Top-k sampling ─────────────────────────────────────────
    # Per-action g_a = -log(-log(U)). Score = logits + g_a. Top-K by repeated
    # argmax with masking. Illegal actions excluded by their -1e9 logit
    # (legal mask was already folded in above).
    # Stronger inter-call seed mixing via multiplication by the 64-bit
    # golden ratio (matches `gpu_sampling.mojo:ezv2_sample_starts_kernel`).
    # Philox already has good avalanche from a +1 seed delta, but adding
    # this defends against any subtle correlation across consecutive
    # calls (rng_seed=0,1,2,…) and against any future change to a less
    # avalanche-y RNG. The per-env LCG offset stays for explicit env-axis
    # decorrelation.
    var philox = PhiloxRandom(
        seed=(
            UInt64(rng_seed) * UInt64(0x9E3779B97F4A7C15)
        )
        + UInt64(e * 1664525 + 1013904223),
        offset=0,
    )
    # Inline buffer for the Gumbel noise + scores; ACT is comptime small.
    var noises = InlineArray[Scalar[dtype], ACT](uninitialized=True)
    var scores = InlineArray[Scalar[dtype], ACT](uninitialized=True)
    var taken = InlineArray[Bool, ACT](uninitialized=True)
    for a in range(ACT):
        var u = philox.step_uniform()
        # Clamp to (1e-9, 1-1e-9) to keep -log(-log(.)) finite.
        var uv = Scalar[dtype](u[0])
        if uv < Scalar[dtype](1e-9):
            uv = Scalar[dtype](1e-9)
        if uv > Scalar[dtype](1.0) - Scalar[dtype](1e-9):
            uv = Scalar[dtype](1.0) - Scalar[dtype](1e-9)
        var g = -log(-log(uv))
        noises[a] = g
        scores[a] = rebind[Scalar[dtype]](node_logits[na_off + a]) + g
        # Exclude illegal actions from Gumbel-Top-k entirely. Without this,
        # if K > legal_count the loop below would still hand the leftover
        # slots to illegal actions (their logits are -1e9, but g(a) is
        # finite, so they're still the best of "what's left").
        var is_legal = True
        if apply_legal > Scalar[DType.uint8](0):
            var lm = rebind[Scalar[dtype]](legal_mask[lm_off + a])
            is_legal = lm > Scalar[dtype](0.5)
        taken[a] = not is_legal

    var k_int = Int(k_actual)
    if k_int > MAX_K:
        k_int = MAX_K
    if k_int < 1:
        k_int = 1
    for slot in range(MAX_K):
        root_candidates[k_off + slot] = Scalar[dtype](-1.0)
        root_gumbels[k_off + slot] = Scalar[dtype](0.0)
        root_active[k_off + slot] = Scalar[dtype](-1.0)
    for slot in range(k_int):
        var best_a = -1
        var best_s = Scalar[dtype](-1e18)
        for a in range(ACT):
            if taken[a]:
                continue
            # Treat very-negative logits as "illegal" — those have been
            # clamped to -1e9; their g+logit is still hugely negative.
            if scores[a] > best_s:
                best_s = scores[a]
                best_a = a
        if best_a < 0:
            break
        root_candidates[k_off + slot] = Scalar[dtype](best_a)
        root_gumbels[k_off + slot] = noises[best_a]
        root_active[k_off + slot] = Scalar[dtype](slot)
        taken[best_a] = True

    # ── 4. Per-env scalar reset ──────────────────────────────────────────
    node_count[e] = Scalar[dtype](1.0)
    min_q[e] = Scalar[dtype](1e18)
    max_q[e] = Scalar[dtype](-1e18)


# ── helpers used by select / halve / extract ─────────────────────────────
#
# Computed inline inside the kernels to avoid sharing state across kernels.
# They mirror the CPU helpers in efficient_zero_v2/mcts.mojo.


def gz_select_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    MAX_K: Int,
    LATENT: Int,
    DYN_IN: Int,
    dtype: DType,
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    node_logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    legal_mask: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    root_candidates: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ],
    root_active: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ],
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ],
    dyn_input: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_IN), MutAnyOrigin
    ],
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    pending_action: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    current_slot: Scalar[DType.int32],
    apply_legal: Scalar[DType.uint8],
    c_visit: Scalar[dtype],
    c_scale: Scalar[dtype],
) where dtype.is_floating_point():
    """One simulation's selection phase, per env.

    Steps:
        1. Read root action from `root_candidates[e][active[e][slot]]`.
        2. If unexpanded, this is the leaf — record pending and build dyn
           input from root hidden state.
        3. Otherwise descend, applying the visit-balance rule at every
           non-root node until we hit an unexpanded action.
        4. Build dyn_input = [parent_hidden ‖ one_hot(leaf_action)].
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var ns_off = e * MAX_NODES
    var k_off = e * MAX_K
    var path_off = e * MAX_DEPTH
    var lm_off = e * ACT

    # Read this sim's root candidate slot.
    var slot_idx = Int(current_slot)
    if slot_idx < 0:
        slot_idx = 0
    var cand_idx = Int(rebind[Scalar[dtype]](root_active[k_off + slot_idx]))
    if cand_idx < 0:
        # No active candidate at this slot — should not happen if the host
        # respects the active set size; fall back to slot 0.
        cand_idx = 0
    var root_action = Int(
        rebind[Scalar[dtype]](root_candidates[k_off + cand_idx])
    )
    if root_action < 0:
        root_action = 0

    # Walk down the tree.
    var node_idx = 0
    var depth = 0
    search_paths[path_off] = Scalar[dtype](0.0)
    action_paths[path_off] = Scalar[dtype](root_action)
    var current_action = root_action

    while depth < MAX_DEPTH - 1:
        var child = rebind[Scalar[dtype]](
            child_idx[tree_off + node_idx * ACT + current_action]
        )
        if child < Scalar[dtype](0.0):
            break  # this is the leaf to expand

        node_idx = Int(child)
        depth += 1
        search_paths[path_off + depth] = Scalar[dtype](node_idx)

        # Non-root selection: visit-balance rule on improved policy.
        # 1. completed_Q[a] = empirical Q if visited, else v_mix.
        # 2. σ_q[a] = (c_visit + max_b N(s,b)) * c_scale * normalize(Q[a]).
        # 3. π_improved[a] = softmax(node_logits[a] + σ_q[a]).
        # 4. a* = argmax_a [π_improved[a] - N(s,a)/(1+ΣN(s,b))].
        var ns_idx = ns_off + node_idx
        var na_base = tree_off + node_idx * ACT
        var n_total = rebind[Scalar[dtype]](total_visits[ns_idx])
        var v_self = rebind[Scalar[dtype]](node_value[ns_idx])

        # ─── v_mix ────────────────────────────────────────────────────────
        # Renormalized prior over visited children weights the visited Q's;
        # then mix with v_self in proportion to total visits.
        var max_visited_logit = Scalar[dtype](-1e18)
        var any_visited = False
        for a in range(ACT):
            var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
            if nva > Scalar[dtype](0.5):
                var l = rebind[Scalar[dtype]](node_logits[na_base + a])
                if l > max_visited_logit:
                    max_visited_logit = l
                any_visited = True
        var v_mix = v_self
        if any_visited:
            var sum_w = Scalar[dtype](0.0)
            var weighted_q = Scalar[dtype](0.0)
            for a in range(ACT):
                var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
                if nva > Scalar[dtype](0.5):
                    var l = rebind[Scalar[dtype]](node_logits[na_base + a])
                    var w = exp(l - max_visited_logit)
                    sum_w += w
                    var qa = (
                        rebind[Scalar[dtype]](total_value[na_base + a])
                        / nva
                    )
                    weighted_q += w * qa
            if sum_w > Scalar[dtype](1e-12):
                var mean_visited_q = weighted_q / sum_w
                v_mix = (v_self + n_total * mean_visited_q) / (
                    Scalar[dtype](1.0) + n_total
                )

        # ─── completed_Q + σ ─────────────────────────────────────────────
        var max_visit = Scalar[dtype](0.0)
        for a in range(ACT):
            var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
            if nva > max_visit:
                max_visit = nva
        var sigma_scale = (c_visit + max_visit) * c_scale
        var mn = rebind[Scalar[dtype]](min_q[e])
        var mx = rebind[Scalar[dtype]](max_q[e])
        var q_range = mx - mn

        # Compute z[a] = node_logits[a] + σ(completed_Q[a]).
        # Stable softmax → π_improved.
        var z = InlineArray[Scalar[dtype], ACT](uninitialized=True)
        var max_z = Scalar[dtype](-1e18)
        for a in range(ACT):
            var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
            var qa: Scalar[dtype]
            if nva > Scalar[dtype](0.5):
                qa = rebind[Scalar[dtype]](total_value[na_base + a]) / nva
            else:
                qa = v_mix
            var qn: Scalar[dtype]
            if q_range > Scalar[dtype](1e-8):
                qn = (qa - mn) / q_range
            else:
                qn = qa
            z[a] = rebind[Scalar[dtype]](node_logits[na_base + a]) + (
                sigma_scale * qn
            )
            if z[a] > max_z:
                max_z = z[a]
        var sum_e = Scalar[dtype](0.0)
        var probs = InlineArray[Scalar[dtype], ACT](uninitialized=True)
        for a in range(ACT):
            var ev = exp(z[a] - max_z)
            probs[a] = ev
            sum_e += ev
        if sum_e <= Scalar[dtype](1e-12):
            sum_e = Scalar[dtype](1.0)
        for a in range(ACT):
            probs[a] = probs[a] / sum_e

        # Visit-balance argmax. Legal mask is root-only; non-root nodes do
        # not have explicit legality info (the dynamics network is expected
        # to make illegal actions unrewarding).
        var denom = Scalar[dtype](1.0) + n_total
        var best_a = 0
        var best_s = Scalar[dtype](-1e18)
        for a in range(ACT):
            var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
            var s = probs[a] - nva / denom
            if s > best_s:
                best_s = s
                best_a = a
        current_action = best_a
        action_paths[path_off + depth] = Scalar[dtype](best_a)

    # Record leaf info.
    var parent_idx = node_idx
    pending_parent[e] = Scalar[dtype](parent_idx)
    pending_action[e] = Scalar[dtype](current_action)
    path_lengths[e] = Scalar[dtype](depth + 1)

    # Build dyn_input = [parent_hidden ‖ one_hot(leaf_action)].
    var d_off = e * DYN_IN
    var h_off = e * MAX_NODES * LATENT + parent_idx * LATENT
    for i in range(LATENT):
        dyn_input[d_off + i] = hidden_states[h_off + i]
    for a in range(ACT):
        dyn_input[d_off + LATENT + a] = Scalar[dtype](0.0)
    dyn_input[d_off + LATENT + current_action] = Scalar[dtype](1.0)


def gz_copy_pred_input_kernel[
    N_ENVS: Int, LATENT: Int, DYN_OUT: Int, dtype: DType,
](
    pred_input: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ],
    dyn_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Copy the LATENT prefix of dyn_output into pred_input. The dyn output
    is `[hidden ‖ reward_logits]`; we feed only the hidden part into the
    prediction network."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var src = e * DYN_OUT
    var dst = e * LATENT
    for i in range(LATENT):
        pred_input[dst + i] = dyn_output[src + i]


def gz_expand_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    PRED_OUT: Int,
    DYN_OUT: Int,
    dtype: DType,
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    node_logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ],
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    pending_action: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    dyn_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin
    ],
    pred_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ],
    leaf_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
) where dtype.is_floating_point():
    """Expand the leaf for each env: write hidden_state, decode reward,
    populate child node_logits, decode child value into both `node_value`
    and `leaf_values`."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var action = Int(rebind[Scalar[dtype]](pending_action[e]))
    var child = Int(rebind[Scalar[dtype]](node_count[e]))
    if child >= MAX_NODES:
        leaf_values[e] = Scalar[dtype](0.0)
        return

    var tree_off = e * MAX_NODES * ACT
    var ns_off = e * MAX_NODES
    var dyn_off = e * DYN_OUT
    var pred_off = e * PRED_OUT
    var child_h_off = e * MAX_NODES * LATENT + child * LATENT

    # ── Hidden state: copy first LATENT values from dyn_output ──────────
    # No min-max scaling here — the dynamics net's MinMaxNorm handles it.
    for i in range(LATENT):
        hidden_states[child_h_off + i] = dyn_output[dyn_off + i]

    # ── Reward decoding (categorical → scalar with h⁻¹) ─────────────────
    comptime NUM_REW_BINS = DYN_OUT - LATENT
    var rew_decoded: Scalar[dtype]
    if NUM_REW_BINS == 1:
        # Scalar reward path (rare here — MuZero typically uses categorical).
        rew_decoded = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
    else:
        var r_max = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
        for i in range(1, NUM_REW_BINS):
            var v = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
            if v > r_max:
                r_max = v
        var r_sum = Scalar[dtype](0.0)
        for i in range(NUM_REW_BINS):
            r_sum += exp(
                rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i]) - r_max
            )
        var r_step = (v_max - v_min) / Scalar[dtype](NUM_REW_BINS - 1)
        var r_expected = Scalar[dtype](0.0)
        for i in range(NUM_REW_BINS):
            var p = (
                exp(
                    rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                    - r_max
                )
                / r_sum
            )
            r_expected += p * (v_min + Scalar[dtype](i) * r_step)
        # Inverse scalar transform
        var sgn_r = (
            Scalar[dtype](1.0)
            if r_expected >= Scalar[dtype](0.0)
            else Scalar[dtype](-1.0)
        )
        var abs_r = (
            r_expected
            if r_expected >= Scalar[dtype](0.0)
            else -r_expected
        )
        var eps_r = Scalar[dtype](0.001)
        var inner_r = sqrt(
            Scalar[dtype](1.0)
            + Scalar[dtype](4.0) * eps_r * (
                abs_r + Scalar[dtype](1.0) + eps_r
            )
        )
        var f_r = (inner_r - Scalar[dtype](1.0)) / (
            Scalar[dtype](2.0) * eps_r
        )
        rew_decoded = sgn_r * (f_r * f_r - Scalar[dtype](1.0))
    reward[tree_off + parent * ACT + action] = rew_decoded

    # ── Copy child policy logits + initialize per-action stats ──────────
    var child_na_base = tree_off + child * ACT
    for a in range(ACT):
        node_logits[child_na_base + a] = pred_output[pred_off + a]
        visit_count[child_na_base + a] = Scalar[dtype](0.0)
        total_value[child_na_base + a] = Scalar[dtype](0.0)
        reward[child_na_base + a] = Scalar[dtype](0.0)
        child_idx[child_na_base + a] = Scalar[dtype](-1.0)
    total_visits[ns_off + child] = Scalar[dtype](0.0)

    # ── Decode child value into node_value[child] + leaf_values[e] ──────
    var v_max_logit = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
        if v > v_max_logit:
            v_max_logit = v
    var v_sum = Scalar[dtype](0.0)
    for i in range(BINS):
        v_sum += exp(
            rebind[Scalar[dtype]](pred_output[pred_off + ACT + i]) - v_max_logit
        )
    var step_v = (v_max - v_min) / Scalar[dtype](BINS - 1)
    var v_expected = Scalar[dtype](0.0)
    for i in range(BINS):
        var prob = (
            exp(
                rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                - v_max_logit
            )
            / v_sum
        )
        v_expected += prob * (v_min + Scalar[dtype](i) * step_v)
    var sgn = (
        Scalar[dtype](1.0)
        if v_expected >= Scalar[dtype](0.0)
        else Scalar[dtype](-1.0)
    )
    var abs_y = (
        v_expected
        if v_expected >= Scalar[dtype](0.0)
        else -v_expected
    )
    var eps_h = Scalar[dtype](0.001)
    var inner = sqrt(
        Scalar[dtype](1.0)
        + Scalar[dtype](4.0) * eps_h * (abs_y + Scalar[dtype](1.0) + eps_h)
    )
    var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps_h)
    var v_decoded = sgn * (f * f - Scalar[dtype](1.0))
    node_value[ns_off + child] = v_decoded
    leaf_values[e] = v_decoded

    # ── Link parent → child + bump node count ───────────────────────────
    child_idx[tree_off + parent * ACT + action] = Scalar[dtype](child)
    node_count[e] = Scalar[dtype](child + 1)


def gz_backup_kernel[
    N_ENVS: Int, MAX_NODES: Int, ACT: Int, dtype: DType,
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    leaf_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    gamma: Scalar[dtype],
) where dtype.is_floating_point():
    """Walk the search path from leaf to root and accumulate the discounted
    return into per-action stats; refresh min_q/max_q for σ(Q)
    normalization. One thread per env."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var path_off = e * MAX_DEPTH
    var path_len = Int(rebind[Scalar[dtype]](path_lengths[e]))

    var value = rebind[Scalar[dtype]](leaf_values[e])
    for i in range(path_len):
        var idx = path_len - 1 - i
        var node_idx = Int(rebind[Scalar[dtype]](search_paths[path_off + idx]))
        var action = Int(rebind[Scalar[dtype]](action_paths[path_off + idx]))
        var na_off = tree_off + node_idx * ACT + action
        value = rebind[Scalar[dtype]](reward[na_off]) + gamma * value
        visit_count[na_off] = rebind[Scalar[dtype]](
            visit_count[na_off]
        ) + Scalar[dtype](1.0)
        total_value[na_off] = (
            rebind[Scalar[dtype]](total_value[na_off]) + value
        )
        total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
            total_visits[tv_off + node_idx]
        ) + Scalar[dtype](1.0)

        var n_a = rebind[Scalar[dtype]](visit_count[na_off])
        var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
        if mean_q < rebind[Scalar[dtype]](min_q[e]):
            min_q[e] = mean_q
        if mean_q > rebind[Scalar[dtype]](max_q[e]):
            max_q[e] = mean_q


def gz_halve_active_kernel[
    N_ENVS: Int, MAX_NODES: Int, ACT: Int, MAX_K: Int, dtype: DType,
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    node_logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    root_candidates: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ],
    root_gumbels: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ],
    root_active: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ],
    old_size: Scalar[DType.int32],
    keep: Scalar[DType.int32],
    c_visit: Scalar[dtype],
    c_scale: Scalar[dtype],
) where dtype.is_floating_point():
    """Sequential-Halving phase boundary: keep the top-`keep` active root
    candidates by score `g(a) + logits(a) + σ(completed_Q(a))`."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var ns_off = e * MAX_NODES
    var k_off = e * MAX_K
    var na_base = tree_off  # root is node 0
    var ns_idx = ns_off  # root node 0

    # Compute completed_Q for the root.
    var v_self = rebind[Scalar[dtype]](node_value[ns_idx])
    var n_total = rebind[Scalar[dtype]](total_visits[ns_idx])

    # ─── v_mix ────────────────────────────────────────────────────────────
    var max_visited_logit = Scalar[dtype](-1e18)
    var any_visited = False
    for a in range(ACT):
        var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
        if nva > Scalar[dtype](0.5):
            var l = rebind[Scalar[dtype]](node_logits[na_base + a])
            if l > max_visited_logit:
                max_visited_logit = l
            any_visited = True
    var v_mix = v_self
    if any_visited:
        var sum_w = Scalar[dtype](0.0)
        var weighted_q = Scalar[dtype](0.0)
        for a in range(ACT):
            var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
            if nva > Scalar[dtype](0.5):
                var l = rebind[Scalar[dtype]](node_logits[na_base + a])
                var w = exp(l - max_visited_logit)
                sum_w += w
                var qa = (
                    rebind[Scalar[dtype]](total_value[na_base + a]) / nva
                )
                weighted_q += w * qa
        if sum_w > Scalar[dtype](1e-12):
            var mean_visited_q = weighted_q / sum_w
            v_mix = (v_self + n_total * mean_visited_q) / (
                Scalar[dtype](1.0) + n_total
            )

    var max_visit = Scalar[dtype](0.0)
    for a in range(ACT):
        var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
        if nva > max_visit:
            max_visit = nva
    var sigma_scale = (c_visit + max_visit) * c_scale
    var mn = rebind[Scalar[dtype]](min_q[e])
    var mx = rebind[Scalar[dtype]](max_q[e])
    var q_range = mx - mn

    var old_n = Int(old_size)
    if old_n > MAX_K:
        old_n = MAX_K
    var keep_n = Int(keep)
    if keep_n < 1:
        keep_n = 1
    if keep_n > old_n:
        keep_n = old_n

    # Score the active candidates.
    var scores = InlineArray[Scalar[dtype], MAX_K](uninitialized=True)
    var active_idx = InlineArray[Int, MAX_K](uninitialized=True)
    for i in range(MAX_K):
        scores[i] = Scalar[dtype](-1e18)
        active_idx[i] = -1
    for i in range(old_n):
        var cand = Int(rebind[Scalar[dtype]](root_active[k_off + i]))
        if cand < 0:
            continue
        var act = Int(rebind[Scalar[dtype]](root_candidates[k_off + cand]))
        if act < 0:
            continue
        var nva = rebind[Scalar[dtype]](visit_count[na_base + act])
        var qa: Scalar[dtype]
        if nva > Scalar[dtype](0.5):
            qa = rebind[Scalar[dtype]](total_value[na_base + act]) / nva
        else:
            qa = v_mix
        var qn: Scalar[dtype]
        if q_range > Scalar[dtype](1e-8):
            qn = (qa - mn) / q_range
        else:
            qn = qa
        var sigma_q = sigma_scale * qn
        var l = rebind[Scalar[dtype]](node_logits[na_base + act])
        var g = rebind[Scalar[dtype]](root_gumbels[k_off + cand])
        scores[i] = g + l + sigma_q
        active_idx[i] = cand

    # Top-K selection sort (small K, O(K * old_n)).
    for slot in range(keep_n):
        var best = slot
        for j in range(slot + 1, old_n):
            if scores[j] > scores[best]:
                best = j
        if best != slot:
            var ts = scores[slot]
            scores[slot] = scores[best]
            scores[best] = ts
            var ti = active_idx[slot]
            active_idx[slot] = active_idx[best]
            active_idx[best] = ti

    # Write back.
    for i in range(MAX_K):
        root_active[k_off + i] = Scalar[dtype](-1.0)
    for i in range(keep_n):
        root_active[k_off + i] = Scalar[dtype](active_idx[i])


def gz_extract_policy_kernel[
    N_ENVS: Int, MAX_NODES: Int, ACT: Int, dtype: DType,
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    node_logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    legal_mask: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    policies_out: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    apply_legal: Scalar[DType.uint8],
    c_visit: Scalar[dtype],
    c_scale: Scalar[dtype],
) where dtype.is_floating_point():
    """Compute the improved policy at the root and write to policies_out."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var ns_off = e * MAX_NODES
    var lm_off = e * ACT
    var po_off = e * ACT
    var na_base = tree_off
    var ns_idx = ns_off

    var v_self = rebind[Scalar[dtype]](node_value[ns_idx])
    var n_total = rebind[Scalar[dtype]](total_visits[ns_idx])

    # v_mix
    var max_visited_logit = Scalar[dtype](-1e18)
    var any_visited = False
    for a in range(ACT):
        var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
        if nva > Scalar[dtype](0.5):
            var l = rebind[Scalar[dtype]](node_logits[na_base + a])
            if l > max_visited_logit:
                max_visited_logit = l
            any_visited = True
    var v_mix = v_self
    if any_visited:
        var sum_w = Scalar[dtype](0.0)
        var weighted_q = Scalar[dtype](0.0)
        for a in range(ACT):
            var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
            if nva > Scalar[dtype](0.5):
                var l = rebind[Scalar[dtype]](node_logits[na_base + a])
                var w = exp(l - max_visited_logit)
                sum_w += w
                var qa = (
                    rebind[Scalar[dtype]](total_value[na_base + a]) / nva
                )
                weighted_q += w * qa
        if sum_w > Scalar[dtype](1e-12):
            var mean_visited_q = weighted_q / sum_w
            v_mix = (v_self + n_total * mean_visited_q) / (
                Scalar[dtype](1.0) + n_total
            )

    var max_visit = Scalar[dtype](0.0)
    for a in range(ACT):
        var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
        if nva > max_visit:
            max_visit = nva
    var sigma_scale = (c_visit + max_visit) * c_scale
    var mn = rebind[Scalar[dtype]](min_q[e])
    var mx = rebind[Scalar[dtype]](max_q[e])
    var q_range = mx - mn

    var z = InlineArray[Scalar[dtype], ACT](uninitialized=True)
    var max_z = Scalar[dtype](-1e18)
    for a in range(ACT):
        var is_legal = True
        if apply_legal > Scalar[DType.uint8](0):
            var lm = rebind[Scalar[dtype]](legal_mask[lm_off + a])
            is_legal = lm > Scalar[dtype](0.5)
        if not is_legal:
            z[a] = Scalar[dtype](-1e9)
            continue
        var nva = rebind[Scalar[dtype]](visit_count[na_base + a])
        var qa: Scalar[dtype]
        if nva > Scalar[dtype](0.5):
            qa = rebind[Scalar[dtype]](total_value[na_base + a]) / nva
        else:
            qa = v_mix
        var qn: Scalar[dtype]
        if q_range > Scalar[dtype](1e-8):
            qn = (qa - mn) / q_range
        else:
            qn = qa
        var l = rebind[Scalar[dtype]](node_logits[na_base + a])
        z[a] = l + sigma_scale * qn
        if z[a] > max_z:
            max_z = z[a]

    var sum_e = Scalar[dtype](0.0)
    for a in range(ACT):
        sum_e += exp(z[a] - max_z)
    if sum_e <= Scalar[dtype](1e-12):
        sum_e = Scalar[dtype](1.0)
    for a in range(ACT):
        policies_out[po_off + a] = exp(z[a] - max_z) / sum_e


# ═════════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════════


def run_gumbel_search_gpu[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_K: Int,
    NUM_SIMULATIONS: Int,
    RepModel: Model,
    DynModel: Model,
    PredModel: Model,
    RepOpt: Optimizer,
    DynOpt: Optimizer,
    PredOpt: Optimizer,
](
    ctx: DeviceContext,
    mut state: EZV2GPUMCTSState[N_ENVS, MAX_NODES, ACT, LATENT, BINS, MAX_K],
    obs_buf: DeviceBuffer[dtype],
    rep_state: GPUNetworkState[RepModel, RepOpt],
    dyn_state: GPUNetworkState[DynModel, DynOpt],
    pred_state: GPUNetworkState[PredModel, PredOpt],
    workspace_buf: DeviceBuffer[dtype],
    v_min: Float64,
    v_max: Float64,
    apply_legal: Bool = False,
    k_actual: Int = MAX_K,
    c_visit: Float64 = 50.0,
    c_scale: Float64 = 0.1,
    gamma: Float64 = 0.997,
    rng_seed: UInt32 = UInt32(0),
) raises:
    """Run Gumbel search across all envs in `state`. Writes the improved
    policy distribution to `state.policies_out`.

    Caller is responsible for:
      • populating `obs_buf` with `[N_ENVS × OBS]` (contiguous batch);
      • optionally populating `state.legal_mask` if `apply_legal=True`;
      • calling `state.zero_tree(ctx)` is done internally;
      • allocating `workspace_buf` sized for the max of the three networks'
        per-sample workspace * `N_ENVS`.
    """
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB

    # ── 0. Reset tree ────────────────────────────────────────────────────
    state.zero_tree(ctx)

    # ── 1. Rep forward (obs → root_hidden, contiguous [N_ENVS × LATENT]) ─
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, RepModel.IN_DIM), MutAnyOrigin
    ](obs_buf.unsafe_ptr())
    var root_hidden_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, RepModel.OUT_DIM), MutAnyOrigin
    ](state.root_hidden.unsafe_ptr())
    Network[RepModel, RepOpt].forward_gpu[N_ENVS](
        ctx,
        obs_t,
        root_hidden_t,
        rep_state.params_view(),
        rep_state.model_state_view(),
        workspace_buf,
    )

    # ── 2. Pred forward (root_hidden → pred_output, contiguous) ──────────
    var pred_in_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PredModel.IN_DIM), MutAnyOrigin
    ](state.root_hidden.unsafe_ptr())
    var pred_out_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PredModel.OUT_DIM), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())
    Network[PredModel, PredOpt].forward_gpu[N_ENVS](
        ctx,
        pred_in_t,
        pred_out_t,
        pred_state.params_view(),
        pred_state.model_state_view(),
        workspace_buf,
    )

    # ── 3. Scatter root_hidden into hidden_states[e][0] ──────────────────
    var rh_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ](state.root_hidden.unsafe_ptr())
    var hs_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    comptime run_scatter = gz_scatter_root_hidden_kernel[
        N_ENVS, MAX_NODES, LATENT, dtype
    ]
    ctx.enqueue_function[run_scatter](
        rh_flat,
        hs_flat,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── 4. Init root: logits + Gumbel-Top-k + value + per-env scalars ────
    var nl_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.node_logits.unsafe_ptr())
    var nv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.node_value.unsafe_ptr())
    var nc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.node_count.unsafe_ptr())
    var miq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.min_q.unsafe_ptr())
    var mxq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.max_q.unsafe_ptr())
    var lm_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](state.legal_mask.unsafe_ptr())
    var rc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_candidates.unsafe_ptr())
    var rg_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_gumbels.unsafe_ptr())
    var ra_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_active.unsafe_ptr())
    var po_full_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())

    var k_clipped = k_actual
    if k_clipped > MAX_K:
        k_clipped = MAX_K
    if k_clipped > ACT:
        k_clipped = ACT
    # Round down to power of two for clean log2(K) phases.
    k_clipped = _largest_power_of_two_le(k_clipped)
    if k_clipped < 1:
        k_clipped = 1

    comptime run_init = gz_init_root_kernel[
        N_ENVS, MAX_NODES, ACT, BINS, MAX_K, PRED_OUT, dtype
    ]
    ctx.enqueue_function[run_init](
        nl_t,
        nv_t,
        nc_t,
        miq_t,
        mxq_t,
        lm_t,
        rc_t,
        rg_t,
        ra_t,
        po_full_t,
        Scalar[dtype](v_min),
        Scalar[dtype](v_max),
        Scalar[DType.int32](k_clipped),
        Scalar[DType.uint8](1 if apply_legal else 0),
        rng_seed,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── 5. Sequential Halving simulation loop ────────────────────────────
    var num_phases = _ilog2(k_clipped)
    if num_phases < 1:
        num_phases = 1
    var per_phase_budget = NUM_SIMULATIONS // num_phases
    if per_phase_budget < 1:
        per_phase_budget = 1

    var sims_used = 0
    var active_size = k_clipped
    for phase in range(num_phases):
        var per_action = per_phase_budget // active_size
        if per_action < 1:
            per_action = 1

        for _rep in range(per_action):
            for slot in range(active_size):
                if sims_used >= NUM_SIMULATIONS:
                    break
                _run_one_sim_gpu[
                    N_ENVS,
                    MAX_NODES,
                    ACT,
                    LATENT,
                    BINS,
                    MAX_K,
                    DynModel,
                    PredModel,
                    DynOpt,
                    PredOpt,
                ](
                    ctx,
                    state,
                    dyn_state,
                    pred_state,
                    workspace_buf,
                    slot,
                    apply_legal,
                    v_min,
                    v_max,
                    c_visit,
                    c_scale,
                    gamma,
                )
                sims_used += 1

        # Halve the active set, except in the last phase.
        if phase + 1 < num_phases and active_size > 1:
            var keep = active_size // 2
            if keep < 1:
                keep = 1
            comptime run_halve = gz_halve_active_kernel[
                N_ENVS, MAX_NODES, ACT, MAX_K, dtype
            ]
            var vc_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
            ](state.visit_count.unsafe_ptr())
            var tv_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
            ](state.total_value.unsafe_ptr())
            var tvis_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
            ](state.total_visits.unsafe_ptr())
            ctx.enqueue_function[run_halve](
                vc_t,
                tv_t,
                nl_t,
                tvis_t,
                nv_t,
                miq_t,
                mxq_t,
                rc_t,
                rg_t,
                ra_t,
                Scalar[DType.int32](active_size),
                Scalar[DType.int32](keep),
                Scalar[dtype](c_visit),
                Scalar[dtype](c_scale),
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            active_size = keep

    # Spend any leftover simulations on slot 0 of the (now size-1) active set.
    while sims_used < NUM_SIMULATIONS:
        _run_one_sim_gpu[
            N_ENVS,
            MAX_NODES,
            ACT,
            LATENT,
            BINS,
            MAX_K,
            DynModel,
            PredModel,
            DynOpt,
            PredOpt,
        ](
            ctx,
            state,
            dyn_state,
            pred_state,
            workspace_buf,
            0,
            apply_legal,
            v_min,
            v_max,
            c_visit,
            c_scale,
            gamma,
        )
        sims_used += 1

    # ── 6. Extract improved policy ───────────────────────────────────────
    var po_extract_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](state.policies_out.unsafe_ptr())
    var vc_t2 = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.visit_count.unsafe_ptr())
    var tv_t2 = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.total_value.unsafe_ptr())
    var tvis_t2 = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.total_visits.unsafe_ptr())
    comptime run_extract = gz_extract_policy_kernel[
        N_ENVS, MAX_NODES, ACT, dtype
    ]
    ctx.enqueue_function[run_extract](
        vc_t2,
        tv_t2,
        nl_t,
        tvis_t2,
        nv_t,
        miq_t,
        mxq_t,
        lm_t,
        po_extract_t,
        Scalar[DType.uint8](1 if apply_legal else 0),
        Scalar[dtype](c_visit),
        Scalar[dtype](c_scale),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


def _run_one_sim_gpu[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_K: Int,
    DynModel: Model,
    PredModel: Model,
    DynOpt: Optimizer,
    PredOpt: Optimizer,
](
    ctx: DeviceContext,
    mut state: EZV2GPUMCTSState[N_ENVS, MAX_NODES, ACT, LATENT, BINS, MAX_K],
    dyn_state: GPUNetworkState[DynModel, DynOpt],
    pred_state: GPUNetworkState[PredModel, PredOpt],
    workspace_buf: DeviceBuffer[dtype],
    slot: Int,
    apply_legal: Bool,
    v_min: Float64,
    v_max: Float64,
    c_visit: Float64,
    c_scale: Float64,
    gamma: Float64,
) raises:
    """One simulation across all envs: select → dyn → pred → expand →
    backup. The root candidate slot is shared across envs (that's safe
    because Sequential Halving keeps the active sets the same size for all
    envs in any given phase)."""
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB

    var vc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.visit_count.unsafe_ptr())
    var tv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.total_value.unsafe_ptr())
    var nl_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.node_logits.unsafe_ptr())
    var rw_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.reward.unsafe_ptr())
    var ci_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.child_idx.unsafe_ptr())
    var tvis_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.total_visits.unsafe_ptr())
    var nv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.node_value.unsafe_ptr())
    var nc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.node_count.unsafe_ptr())
    var miq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.min_q.unsafe_ptr())
    var mxq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.max_q.unsafe_ptr())
    var lm_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](state.legal_mask.unsafe_ptr())
    var rc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_candidates.unsafe_ptr())
    var ra_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_active.unsafe_ptr())
    var hs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    var di_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_IN), MutAnyOrigin
    ](state.dyn_input.unsafe_ptr())
    var pp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.pending_parent.unsafe_ptr())
    var pa_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.pending_action.unsafe_ptr())
    var sp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ](state.search_paths.unsafe_ptr())
    var ap_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ](state.action_paths.unsafe_ptr())
    var pl_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.path_lengths.unsafe_ptr())

    # Selection.
    comptime run_select = gz_select_kernel[
        N_ENVS, MAX_NODES, ACT, MAX_K, LATENT, DYN_IN, dtype
    ]
    ctx.enqueue_function[run_select](
        vc_t,
        tv_t,
        nl_t,
        ci_t,
        tvis_t,
        nv_t,
        miq_t,
        mxq_t,
        lm_t,
        rc_t,
        ra_t,
        hs_t,
        di_t,
        pp_t,
        pa_t,
        sp_t,
        ap_t,
        pl_t,
        Scalar[DType.int32](slot),
        Scalar[DType.uint8](1 if apply_legal else 0),
        Scalar[dtype](c_visit),
        Scalar[dtype](c_scale),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Dynamics forward.
    var dyn_in_b = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, DynModel.IN_DIM), MutAnyOrigin
    ](state.dyn_input.unsafe_ptr())
    var dyn_out_b = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, DynModel.OUT_DIM), MutAnyOrigin
    ](state.dyn_output.unsafe_ptr())
    Network[DynModel, DynOpt].forward_gpu[N_ENVS](
        ctx,
        dyn_in_b,
        dyn_out_b,
        dyn_state.params_view(),
        dyn_state.model_state_view(),
        workspace_buf,
    )

    # Copy dyn_output's hidden prefix into pred_input, then prediction forward.
    comptime run_copy = gz_copy_pred_input_kernel[
        N_ENVS, LATENT, DYN_OUT, dtype
    ]
    var pred_in_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ](state.pred_input.unsafe_ptr())
    var dyn_out_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin
    ](state.dyn_output.unsafe_ptr())
    ctx.enqueue_function[run_copy](
        pred_in_flat,
        dyn_out_flat,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    var pred_in_b = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PredModel.IN_DIM), MutAnyOrigin
    ](state.pred_input.unsafe_ptr())
    var pred_out_b = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PredModel.OUT_DIM), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())
    Network[PredModel, PredOpt].forward_gpu[N_ENVS](
        ctx,
        pred_in_b,
        pred_out_b,
        pred_state.params_view(),
        pred_state.model_state_view(),
        workspace_buf,
    )

    # Expand.
    var lv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.leaf_values.unsafe_ptr())
    var po_full_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())
    comptime run_expand = gz_expand_kernel[
        N_ENVS, MAX_NODES, ACT, LATENT, BINS, PRED_OUT, DYN_OUT, dtype
    ]
    ctx.enqueue_function[run_expand](
        vc_t,
        tv_t,
        nl_t,
        rw_t,
        ci_t,
        tvis_t,
        nv_t,
        nc_t,
        hs_t,
        pp_t,
        pa_t,
        dyn_out_flat,
        po_full_t,
        lv_t,
        Scalar[dtype](v_min),
        Scalar[dtype](v_max),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Backup.
    comptime run_backup = gz_backup_kernel[
        N_ENVS, MAX_NODES, ACT, dtype
    ]
    ctx.enqueue_function[run_backup](
        vc_t,
        tv_t,
        rw_t,
        tvis_t,
        miq_t,
        mxq_t,
        sp_t,
        ap_t,
        pl_t,
        lv_t,
        Scalar[dtype](gamma),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


# ═════════════════════════════════════════════════════════════════════════
# Host helpers (mirror efficient_zero_v2/mcts.mojo)
# ═════════════════════════════════════════════════════════════════════════


def _ilog2(n: Int) -> Int:
    var x = n
    var r = 0
    while x > 1:
        x = x // 2
        r += 1
    return r


def _largest_power_of_two_le(n: Int) -> Int:
    var x = 1
    while x * 2 <= n:
        x *= 2
    return x
