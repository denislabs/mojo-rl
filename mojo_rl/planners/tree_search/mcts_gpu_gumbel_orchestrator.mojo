"""GumbelGPUMCTS — discrete-action Gumbel-MCTS orchestrator for EZv2.

The agent-side counterpart of ``GenericGPUMCTS`` for the EfficientZero V2
discrete planner. Owns an ``EZV2GPUMCTSState`` plus a ``root_value_out``
device buffer, exposes ``search_gpu[REP, DYN, PRED]`` that drives the
full encode → predict (root) → init_root (Gumbel-Top-k) → Sequential
Halving phases → extract policy pipeline.

Same trait surface as ``GenericGPUMCTS`` (``RepresentationGPU`` /
``DynamicsGPU`` / ``PredictionGPU``), so the EZv2 agent's existing
GPUNetworkState + the EZv2 trait adapters drop into the orchestrator
without any agent-side network code changes.

What's different vs ``GenericGPUMCTS.search_gpu``:
  • Selection rule: deterministic visit-balance
    ``argmax_a [π_improved(a) − N(s,a)/(1+ΣN(s,b))]`` instead of PUCT.
  • Root expansion: restricted to ``K`` candidates via Gumbel-Top-k —
    ``init_root`` samples them, ``halve_active`` halves the active set
    between Sequential Halving phases.
  • Sims structure: ``log2(K)`` host-orchestrated phases; per-action
    budget is ``NUM_SIMULATIONS / num_phases / active_size``. Leftover
    sims spend on slot 0 of the final survivor.
  • No virtual-loss batching — sims run one at a time within a phase.
    Kept verbatim from ``run_gumbel_search_gpu`` so the EZv2 perf
    profile is unchanged through the rewiring.
  • Policy readout: ``gz_extract_policy_kernel`` builds the
    ``π̂ = softmax(logits + σ(completed_Q))`` improved policy. The agent
    samples (or argmaxes) from it host-side — the orchestrator does NOT
    write a separate ``actions_out`` because EZv2 selects stochastically
    during data collection.

Output buffers exposed:
  • ``policies_view()`` → ``[N_ENVS × ACT]`` improved policy.
  • ``root_value_view()`` → ``[N_ENVS]`` scalar root value (scattered
    from ``state.node_value[e * MAX_NODES]``).
  • ``legal_mask_view()`` → ``[N_ENVS × ACT]``; the caller populates it
    before ``search_gpu`` when calling with ``apply_legal=True``.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT as dtype
comptime TPB = 256  # preserved from legacy nn.constants (nn.TPB == 128)

from .model_traits_gpu import (
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
    EnvStepGPU,
)
from .mcts_gpu_gumbel import (
    MAX_DEPTH,
    EZV2GPUMCTSState,
    gz_scatter_root_hidden_kernel,
    gz_init_root_kernel,
    gz_select_kernel,
    gz_copy_pred_input_kernel,
    gz_expand_kernel,
    gz_backup_kernel,
    gz_halve_active_kernel,
    gz_extract_policy_kernel,
    gz_az_copy_root_state_kernel,
    gz_az_stage_state_kernel,
    gz_az_expand_kernel,
)
from .strategies import PlayerMode, SinglePlayer
from std.random.philox import Random as PhiloxRandom
from std.math import log, exp


# ═════════════════════════════════════════════════════════════════════════
# Helpers (moved from `run_gumbel_search_gpu` so the orchestrator method is
# pure orchestration). Kept module-private so the legacy driver still
# imports its own copies from `mcts_gpu_gumbel.mojo`.
# ═════════════════════════════════════════════════════════════════════════


def _ilog2(n: Int) -> Int:
    var x = n
    var r = 0
    while x > 1:
        x = x // 2
        r += 1
    return r


def _largest_power_of_two_le(n: Int) -> Int:
    if n < 1:
        return 1
    var p = 1
    while p * 2 <= n:
        p *= 2
    return p


# ═════════════════════════════════════════════════════════════════════════
# Root-value extraction kernel
# ═════════════════════════════════════════════════════════════════════════
#
# Lives next to the orchestrator (not in `mcts_gpu_gumbel.mojo`) because it
# only exists to populate the orchestrator's `root_value_out` buffer — the
# legacy driver writes nothing equivalent.


def gz_extract_root_value_kernel[
    N_ENVS: Int, MAX_NODES: Int, ACT: Int, dtype: DType,
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    node_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    root_value_out: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
):
    """MCTS-improved root value: ``Σ_a total_value[root,a] / Σ_a N[root,a]``.

    Mirrors ``gpu_mcts_extract_root_value_kernel`` in the vanilla
    orchestrator. We CANNOT just read ``node_value[root]`` because the
    Gumbel backup kernel only updates ``total_value`` / ``visit_count``
    edge stats, not ``node_value``; the latter stays at the network's
    bare prediction from ``gz_init_root_kernel``. Without this fix the
    n-step value target bootstraps off the network's own prediction →
    no learning signal for the value head.

    Falls back to ``node_value[root]`` when total_visits = 0 (e.g. if a
    search was skipped). One thread per env."""
    comptime assert dtype.is_floating_point(), (
        "gz_extract_root_value_kernel: dtype must be floating-point"
    )
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var total_v = Scalar[dtype](0.0)
    var total_n = Scalar[dtype](0.0)
    for a in range(ACT):
        var n = rebind[Scalar[dtype]](visit_count[tree_off + a])
        if n > Scalar[dtype](0.0):
            total_v += rebind[Scalar[dtype]](total_value[tree_off + a])
            total_n += n

    if total_n > Scalar[dtype](0.5):
        root_value_out[e] = total_v / total_n
    else:
        # No sims — fall back to the network's raw prediction.
        root_value_out[e] = node_value[e * MAX_NODES]


def gz_extract_actions_gumbel_kernel[
    N_ENVS: Int, ACT: Int, dtype: DType,
](
    policies_out: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    legal_masks: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    actions_out: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    rng_seed: Scalar[DType.uint32],
    gumbel_scale: Scalar[dtype] = Scalar[dtype](1.0),
) where dtype.is_floating_point():
    """Gumbel-argmax action selection (mctx convention for Gumbel-MuZero).

    Computes ``argmax_a [gumbel_scale · g_a + log(policies_out[a])]`` per
    env, where ``g_a = -log(-log(u))`` is a fresh Gumbel sample. Because
    ``log(softmax(scores)) = scores + const``, this is equivalent to
    ``argmax_a [gumbel_scale · g_a + logits[a] + σ(Q[a])]`` — the
    canonical Full Gumbel MuZero action choice from mctx.

    Why argmax-with-noise vs sampling-from-policy: the improved policy
    ``softmax(logits + σ(Q))`` is extremely peaked (σ_scale ≈ 6 typical),
    so direct sampling collapses all envs onto the same action and
    kills trajectory diversity. Gumbel-argmax preserves diversity
    because each env's noise is independent.

    ``gumbel_scale=1.0`` at training, ``0.0`` at eval (mctx default —
    deterministic argmax over the improved policy).

    One thread per env. Illegal actions are excluded by adding a large
    negative offset to their log-policy."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var p_off = e * ACT

    var philox = PhiloxRandom(
        seed=(
            UInt64(rng_seed) * UInt64(0x9E3779B97F4A7C15)
        )
        + UInt64(e * 2654435761 + 1442695040888963407),
        offset=0,
    )

    var best_score = Scalar[dtype](-1e18)
    var best_action = -1
    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[p_off + a])
        if legal <= Scalar[dtype](0.5):
            continue
        var p = rebind[Scalar[dtype]](policies_out[p_off + a])
        # Clamp probability to keep log finite for near-zero entries.
        var p_safe = p
        if p_safe < Scalar[dtype](1e-12):
            p_safe = Scalar[dtype](1e-12)
        var log_p = log(p_safe)

        var g = Scalar[dtype](0.0)
        if gumbel_scale > Scalar[dtype](0.0):
            var u = philox.step_uniform()
            var uv = Scalar[dtype](u[0])
            if uv < Scalar[dtype](1e-9):
                uv = Scalar[dtype](1e-9)
            if uv > Scalar[dtype](1.0) - Scalar[dtype](1e-9):
                uv = Scalar[dtype](1.0) - Scalar[dtype](1e-9)
            g = gumbel_scale * -log(-log(uv))

        var score = g + log_p
        if score > best_score:
            best_score = score
            best_action = a

    if best_action < 0:
        best_action = 0
    actions_out[e] = Scalar[dtype](best_action)


def gz_extract_actions_temp_kernel[
    N_ENVS: Int, ACT: Int, dtype: DType,
](
    policies_out: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    legal_masks: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    ep_steps: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    actions_out: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    temp_threshold: Int,
    rng_seed: Scalar[DType.uint32],
    temp_min: Scalar[dtype] = Scalar[dtype](0.0),
) where dtype.is_floating_point():
    """Temperature-controlled action sampling from the *improved policy*.

    Reads from the Gumbel-MCTS improved policy in ``policies_out``
    (already legal-masked + renormalized by ``gz_extract_policy_kernel``)
    and writes a chosen action per env into ``actions_out``. The
    ``policies_out`` buffer is left UNTOUCHED — it's still the training
    target.

    Schedule (per env, matches the vanilla MuZero / AlphaZero convention
    via ``GenericGPUMCTS.extract_actions_temp``):
      * ``ep_steps[e] < temp_threshold``  → sample ∝ ``policies_out``.
      * Else if ``temp_min > 0``           → sample ∝ ``policies_out^(1/τ)``.
      * Else                               → argmax over legal actions.

    One thread per env. ``USE_LEGAL_MASK=True`` callers (board games)
    additionally re-mask before sampling; for single-player envs the
    legal mask is all-ones and the renorm is a no-op."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var p_off = e * ACT
    var move_count = Int(rebind[Scalar[dtype]](ep_steps[e]))

    # ── 1. Apply legal mask to a local probability buffer ───────────────
    var probs = InlineArray[Scalar[dtype], ACT](uninitialized=True)
    var total = Scalar[dtype](0.0)
    var best_action = -1
    var best_p = Scalar[dtype](-1.0)
    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[p_off + a])
        if legal > Scalar[dtype](0.5):
            var p = rebind[Scalar[dtype]](policies_out[p_off + a])
            probs[a] = p
            total += p
            if p > best_p:
                best_p = p
                best_action = a
        else:
            probs[a] = Scalar[dtype](0.0)

    if best_action < 0:
        # No legal moves — fall back to action 0 (caller should have
        # ensured at least one legal action exists).
        actions_out[e] = Scalar[dtype](0.0)
        return

    # ── 2. Branch on move_count ────────────────────────────────────────
    if move_count < temp_threshold:
        # Sample ∝ policies_out (legal-masked).
        var philox = PhiloxRandom(
            seed=(
                UInt64(rng_seed) * UInt64(0x9E3779B97F4A7C15)
            )
            + UInt64(e * 2654435761 + 1442695040888963407),
            offset=0,
        )
        var u = philox.step_uniform()
        var r = Scalar[dtype](u[0]) * total
        var cum = Scalar[dtype](0.0)
        var picked = best_action
        for a in range(ACT):
            cum += probs[a]
            if cum >= r:
                picked = a
                break
        actions_out[e] = Scalar[dtype](picked)
    elif temp_min > Scalar[dtype](0.0):
        # Sample ∝ p^(1/τ); rebuild a temp-distorted total.
        var inv_temp = Scalar[dtype](1.0) / temp_min
        var sharp_total = Scalar[dtype](0.0)
        for a in range(ACT):
            if probs[a] > Scalar[dtype](0.0):
                # ``p^(1/τ)`` — via exp(log) so any τ works.
                probs[a] = exp(inv_temp * log(probs[a]))
                sharp_total += probs[a]
        var philox = PhiloxRandom(
            seed=(
                UInt64(rng_seed) * UInt64(0x9E3779B97F4A7C15)
            )
            + UInt64(e * 2654435761 + 1442695040888963407),
            offset=0,
        )
        var u = philox.step_uniform()
        var r = Scalar[dtype](u[0]) * sharp_total
        var cum = Scalar[dtype](0.0)
        var picked = best_action
        for a in range(ACT):
            cum += probs[a]
            if cum >= r:
                picked = a
                break
        actions_out[e] = Scalar[dtype](picked)
    else:
        # Greedy argmax over legal actions.
        actions_out[e] = Scalar[dtype](best_action)


# ═════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════


struct GumbelGPUMCTS[
    N_ENVS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    NUM_SIMULATIONS: Int,
    PLAYER: PlayerMode = SinglePlayer,
    # > 0 enables the Gumbel AlphaZero path (`search_gpu_alphazero`): tree
    # nodes carry true game-state payloads and expansion is `env.step_gpu`
    # instead of the dynamics net. 0 (default) allocates no AZ buffers.
    STATE_SIZE: Int = 0,
](Movable, ImplicitlyDeletable):
    """GPU Gumbel-MCTS orchestrator (shared across EZv2 + MuZero).

    Comptime params:
        N_ENVS: Number of parallel envs / trees.
        ACT: Discrete action count.
        LATENT: Hidden state dim.
        BINS: Categorical reward / value bins.
        MAX_NODES: Per-tree node arena size.
        MAX_K: Max Gumbel-Top-k root candidates (must be a power of two
            and ≤ ``ACT``; the driver clips at runtime).
        NUM_SIMULATIONS: Total sims per ``search_gpu`` call. Budget split
            across ``log2(K)`` phases by Sequential Halving.
        PLAYER: ``SinglePlayer`` (default, EZv2 / single-player MuZero)
            or ``SelfPlay`` (two-player zero-sum, board-game MuZero —
            backup negates value at each ply; per-edge reward is
            ignored, matching ``GenericGPUMCTS.search_gpu_selfplay``).

    Runtime ctor args:
        gamma, v_min, v_max — categorical decode + discounting.
        c_visit, c_scale — π-improvement σ(Q) scaling (paper defaults
            50.0 and 0.1).
        gumbel_scale — scale on the root Gumbel noise. mctx convention:
            1.0 at training / data-collection (default), 0.0 at eval /
            arena (deterministic root-action ranking).
    """

    comptime PRED_OUT: Int = Self.ACT + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime ENV_BLOCKS: Int = (Self.N_ENVS + TPB - 1) // TPB

    var state: EZV2GPUMCTSState[
        Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.BINS, Self.MAX_K,
    ]
    """Per-env trees + scratch (legal mask, root candidates, search
    paths, network I/O staging, ``policies_out``). See
    ``EZV2GPUMCTSState`` for field-by-field layout."""

    var root_value_out: DeviceBuffer[dtype]
    """``[N_ENVS]`` scalar root value scattered from
    ``state.node_value[e * MAX_NODES]`` (decoded by
    ``gz_init_root_kernel``)."""

    var actions_out: DeviceBuffer[dtype]
    """``[N_ENVS]`` chosen action per env, populated by
    ``extract_actions_argmax`` / ``extract_actions_temp``. EZv2 does
    not use this (it samples host-side from ``policies_out``); MuZero
    consumes it directly via the ``mcts_step_*`` agent buffers."""

    # ── Gumbel AlphaZero buffers (size 1 when STATE_SIZE == 0) ──────────
    comptime AZ_GS_SIZE: Int = (
        Self.N_ENVS * Self.MAX_NODES * Self.STATE_SIZE
        if Self.STATE_SIZE > 0 else 1
    )
    comptime AZ_EXP_SIZE: Int = (
        Self.N_ENVS * Self.STATE_SIZE if Self.STATE_SIZE > 0 else 1
    )
    comptime AZ_ENV_SIZE: Int = Self.N_ENVS if Self.STATE_SIZE > 0 else 1
    comptime AZ_LEGAL_SIZE: Int = (
        Self.N_ENVS * Self.ACT if Self.STATE_SIZE > 0 else 1
    )

    var az_game_states: DeviceBuffer[dtype]
    """``[N_ENVS × MAX_NODES × STATE_SIZE]`` per-node game states (AZ path)."""
    var az_expansion_states: DeviceBuffer[dtype]
    """``[N_ENVS × STATE_SIZE]`` staged parent states; mutated in place by
    ``env.step_gpu`` into the child states."""
    var az_step_rewards: DeviceBuffer[dtype]
    var az_step_dones: DeviceBuffer[dtype]
    var az_step_terminated: DeviceBuffer[dtype]
    var az_exp_legal: DeviceBuffer[dtype]
    """``[N_ENVS × ACT]`` post-step legal mask (masks child priors)."""

    var gamma: Float64
    var v_min: Float64
    var v_max: Float64
    var c_visit: Float64
    var c_scale: Float64
    var gumbel_scale: Float64
    var qnorm_per_node: Bool
    """σ(completed_Q) normalization mode. True (default) = per-NODE
    completed-Q min/max rescale (mctx qtransform_completed_by_mix_value):
    the node's best child maps to qn=1, worst to qn=0 — full-strength
    ranking, REQUIRED for two-player ±1-value games whose tree-global range
    dwarfs sibling ΔQ (C4 fix). False = tree-GLOBAL min/max (classic MuZero
    normalization): preserves gap MAGNITUDE relative to the tree's value
    spread — required for tiny action spaces (ACT=2: per-node quantizes qn
    to exactly {0,1}, destroying magnitude → targets become confident
    one-hots toward Q-estimate noise; MZ CartPole regressed from 500 to
    thrashing ~130-300 with target_max_prob 0.997 from step 600).
    Single-player small-ACT drivers (MZ / EZv2 discrete) pass False."""

    def __init__(
        out self,
        ctx: DeviceContext,
        gamma: Float64 = 0.997,
        v_min: Float64 = -10.0,
        v_max: Float64 = 10.0,
        c_visit: Float64 = 50.0,
        c_scale: Float64 = 0.1,
        gumbel_scale: Float64 = 1.0,
        qnorm_per_node: Bool = True,
    ) raises:
        if Self.MAX_K > Self.ACT:
            raise Error("GumbelGPUMCTS: MAX_K must be <= ACT")
        if Self.MAX_K < 1:
            raise Error("GumbelGPUMCTS: MAX_K must be >= 1")
        self.state = EZV2GPUMCTSState[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.BINS,
            Self.MAX_K,
        ](ctx)
        # Default legal_mask = all ones so callers that don't populate
        # it (single-player paths, eval_argmax callers) get all-legal
        # behavior. Self-play / board games overwrite per-step from
        # the env's legal mask before each ``search_gpu_selfplay`` call.
        self.state.legal_mask.enqueue_fill(Scalar[dtype](1.0))
        self.root_value_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.actions_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.az_game_states = ctx.enqueue_create_buffer[dtype](
            Self.AZ_GS_SIZE
        )
        self.az_expansion_states = ctx.enqueue_create_buffer[dtype](
            Self.AZ_EXP_SIZE
        )
        self.az_step_rewards = ctx.enqueue_create_buffer[dtype](
            Self.AZ_ENV_SIZE
        )
        self.az_step_dones = ctx.enqueue_create_buffer[dtype](
            Self.AZ_ENV_SIZE
        )
        self.az_step_terminated = ctx.enqueue_create_buffer[dtype](
            Self.AZ_ENV_SIZE
        )
        self.az_exp_legal = ctx.enqueue_create_buffer[dtype](
            Self.AZ_LEGAL_SIZE
        )
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.gumbel_scale = gumbel_scale
        self.qnorm_per_node = qnorm_per_node

    def __init__(out self, *, deinit take: Self):
        self.state = take.state^
        self.root_value_out = take.root_value_out^
        self.actions_out = take.actions_out^
        self.az_game_states = take.az_game_states^
        self.az_expansion_states = take.az_expansion_states^
        self.az_step_rewards = take.az_step_rewards^
        self.az_step_dones = take.az_step_dones^
        self.az_step_terminated = take.az_step_terminated^
        self.az_exp_legal = take.az_exp_legal^
        self.gamma = take.gamma
        self.v_min = take.v_min
        self.v_max = take.v_max
        self.c_visit = take.c_visit
        self.c_scale = take.c_scale
        self.gumbel_scale = take.gumbel_scale
        self.qnorm_per_node = take.qnorm_per_node

    # ══════════════════════════════════════════════════════════════════════
    # Views
    # ══════════════════════════════════════════════════════════════════════

    def policies_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS × ACT]`` improved-policy distribution."""
        return self.state.policies_out

    def root_value_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS]`` scalar root value (decoded from value bins at
        init_root time)."""
        return self.root_value_out

    def legal_mask_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS × ACT]`` legal-action mask; caller populates before
        ``search_gpu(apply_legal=True)`` and re-reads any time."""
        return self.state.legal_mask

    def actions_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS]`` chosen action per env. Valid only after a call
        to ``extract_actions_argmax`` / ``extract_actions_temp``."""
        return self.actions_out

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def search_gpu[
        REP: RepresentationGPU,
        DYN: DynamicsGPU,
        PRED: PredictionGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut rep: REP,
        mut dyn: DYN,
        mut pred: PRED,
        obs: LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, REP.OBS_DIM), MutAnyOrigin
        ],
        apply_legal: Bool = False,
        k_actual: Int = Self.MAX_K,
        rng_seed: UInt32 = UInt32(0),
    ) raises:
        """Run Gumbel-MCTS for all ``N_ENVS`` envs in parallel.

        Pipeline (mirrors ``run_gumbel_search_gpu``):
          1. ``rep.encode_gpu`` → root hidden (contiguous ``[N_ENVS × LATENT]``)
          2. ``pred.predict_gpu`` → root pred output (logits + value bins)
          3. Scatter root hidden into ``state.hidden_states[e][0]``
          4. ``gz_init_root_kernel`` — logits + Gumbel-Top-k + decoded value
          5. Sequential Halving: ``log2(K)`` phases, each
             ``per_phase_budget // active_size`` sims per slot, then
             ``gz_halve_active_kernel``.
             Each sim: ``gz_select_kernel`` → ``dyn.step_gpu`` →
             ``gz_copy_pred_input_kernel`` → ``pred.predict_gpu`` →
             ``gz_expand_kernel`` → ``gz_backup_kernel``.
          6. Leftover sims on slot 0 of the size-1 survivor.
          7. ``gz_extract_policy_kernel`` → improved policy.
          8. ``gz_extract_root_value_kernel`` → root_value_out.

        ``apply_legal=True`` reads the caller-populated
        ``state.legal_mask`` and applies it inside ``init_root`` (Gumbel
        sampling skips illegal actions) and ``extract_policy``.

        ``k_actual`` is clipped to ``[1, MAX_K]`` and rounded down to a
        power of two by the driver.
        """

        # ── 0. Reset tree ────────────────────────────────────────────────
        self.state.zero_tree(ctx)

        # ── 1. Rep forward ───────────────────────────────────────────────
        var root_hidden_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, REP.LATENT_DIM), MutAnyOrigin,
        ](self.state.root_hidden.unsafe_ptr())
        rep.encode_gpu[Self.N_ENVS](ctx, obs, root_hidden_t)

        # ── 2. Pred forward at the root ──────────────────────────────────
        var pred_root_in = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, PRED.LATENT_DIM), MutAnyOrigin,
        ](self.state.root_hidden.unsafe_ptr())
        var pred_root_out = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, PRED.PRED_OUT_DIM), MutAnyOrigin,
        ](self.state.pred_output.unsafe_ptr())
        pred.predict_gpu[Self.N_ENVS](ctx, pred_root_in, pred_root_out)

        # ── 3. Scatter root_hidden → hidden_states[e][0] ─────────────────
        var rh_flat = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.LATENT), MutAnyOrigin
        ](self.state.root_hidden.unsafe_ptr())
        var hs_flat = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        comptime run_scatter = gz_scatter_root_hidden_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.LATENT, dtype,
        ]
        ctx.enqueue_function[run_scatter](
            rh_flat,
            hs_flat,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 4. Init root ─────────────────────────────────────────────────
        var nl_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.node_logits.unsafe_ptr())
        var nv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.node_value.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.node_count.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.max_q.unsafe_ptr())
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.legal_mask.unsafe_ptr())
        var rc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_candidates.unsafe_ptr())
        var rg_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_gumbels.unsafe_ptr())
        var ra_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_active.unsafe_ptr())
        var po_full_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.PRED_OUT), MutAnyOrigin
        ](self.state.pred_output.unsafe_ptr())

        var k_clipped = k_actual
        if k_clipped > Self.MAX_K:
            k_clipped = Self.MAX_K
        if k_clipped > Self.ACT:
            k_clipped = Self.ACT
        k_clipped = _largest_power_of_two_le(k_clipped)
        if k_clipped < 1:
            k_clipped = 1

        comptime run_init = gz_init_root_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.BINS, Self.MAX_K,
            Self.PRED_OUT, dtype,
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
            Scalar[dtype](self.v_min),
            Scalar[dtype](self.v_max),
            Scalar[DType.int32](k_clipped),
            Scalar[DType.uint8](1 if apply_legal else 0),
            rng_seed,
            Scalar[dtype](self.gumbel_scale),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 5. Sequential Halving simulation loop ────────────────────────
        var num_phases = _ilog2(k_clipped)
        if num_phases < 1:
            num_phases = 1
        var per_phase_budget = Self.NUM_SIMULATIONS // num_phases
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
                    if sims_used >= Self.NUM_SIMULATIONS:
                        break
                    self._run_one_sim_gpu[REP, DYN, PRED](
                        ctx, dyn, pred, slot, apply_legal
                    )
                    sims_used += 1

            # Halve the active set, except in the last phase.
            if phase + 1 < num_phases and active_size > 1:
                var keep = active_size // 2
                if keep < 1:
                    keep = 1
                comptime run_halve = gz_halve_active_kernel[
                    Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.MAX_K, dtype,
                ]
                var vc_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
                    MutAnyOrigin,
                ](self.state.visit_count.unsafe_ptr())
                var tv_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
                    MutAnyOrigin,
                ](self.state.total_value.unsafe_ptr())
                var tvis_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES),
                    MutAnyOrigin,
                ](self.state.total_visits.unsafe_ptr())
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
                    Scalar[dtype](self.c_visit),
                    Scalar[dtype](self.c_scale),
                    Scalar[DType.uint8](1 if self.qnorm_per_node else 0),
                    grid_dim=(Self.ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                active_size = keep

        # Spend any leftover simulations on slot 0 of the size-1 survivor.
        while sims_used < Self.NUM_SIMULATIONS:
            self._run_one_sim_gpu[REP, DYN, PRED](
                ctx, dyn, pred, 0, apply_legal
            )
            sims_used += 1

        # ── 6. Extract improved policy ───────────────────────────────────
        var po_extract_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.policies_out.unsafe_ptr())
        var vc_t2 = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.visit_count.unsafe_ptr())
        var tv_t2 = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.total_value.unsafe_ptr())
        var tvis_t2 = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.total_visits.unsafe_ptr())
        comptime run_extract = gz_extract_policy_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype,
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
            Scalar[dtype](self.c_visit),
            Scalar[dtype](self.c_scale),
            Scalar[DType.uint8](1 if self.qnorm_per_node else 0),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 7. Extract root scalar value (MCTS-improved, not the raw
        #       network prediction) ──────────────────────────────────────
        var rv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.root_value_out.unsafe_ptr())
        comptime run_root_value = gz_extract_root_value_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype,
        ]
        ctx.enqueue_function[run_root_value](
            vc_t2,
            tv_t2,
            nv_t,
            rv_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    def search_gpu_selfplay[
        REP: RepresentationGPU,
        DYN: DynamicsGPU,
        PRED: PredictionGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut rep: REP,
        mut dyn: DYN,
        mut pred: PRED,
        obs: LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, REP.OBS_DIM), MutAnyOrigin
        ],
        k_actual: Int = Self.MAX_K,
        rng_seed: UInt32 = UInt32(0),
    ) raises:
        """Two-player self-play variant of ``search_gpu`` for board games.

        Caller must populate ``state.legal_mask`` ([N_ENVS × ACT]) before
        invoking — the kernel honors it both at root Gumbel-Top-k
        sampling and in the improved-policy extraction.

        Identical to ``search_gpu(apply_legal=True)`` except the
        orchestrator's ``PLAYER`` comptime param is asserted to be
        ``SelfPlay``-compatible: ``gz_backup_kernel`` uses the negated
        recurrence (``value = -value`` at every parent), and per-edge
        rewards are ignored — terminal reward is folded into the leaf
        value by the expand kernel.

        Use this in MuZero board-game training / evaluation; for
        single-player envs (CartPole / Atari arcade) call ``search_gpu``
        directly with ``apply_legal=False``.
        """
        # The orchestrator template will fail to instantiate with the
        # wrong PLAYER (NEGATE_BACKUP threaded into ``gz_backup_kernel``
        # via ``Self.PLAYER.NEGATE_BACKUP``). Reuse ``search_gpu`` —
        # ``apply_legal=True`` is the only behavioral difference at the
        # orchestrator level, the negation lives in the kernel choice.
        self.search_gpu[REP, DYN, PRED](
            ctx, rep, dyn, pred, obs,
            apply_legal=True,
            k_actual=k_actual,
            rng_seed=rng_seed,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Action extraction (composes after search_gpu* — visit-target stays
    # in ``policies_out``, this only writes ``actions_out``).
    # ══════════════════════════════════════════════════════════════════════

    def extract_actions_argmax(mut self, ctx: DeviceContext) raises:
        """Greedy argmax over the legal-masked improved policy.

        For eval / arena, where Gumbel noise is disabled. Equivalent to
        ``extract_actions_gumbel(gumbel_scale=0.0)`` — argmax over
        ``log(policies_out)`` ignoring noise."""
        self.extract_actions_gumbel(
            ctx, rng_seed=UInt32(0), gumbel_scale=0.0
        )

    def extract_actions_gumbel(
        mut self,
        ctx: DeviceContext,
        rng_seed: UInt32 = UInt32(0),
        gumbel_scale: Float64 = 1.0,
    ) raises:
        """Run mctx-style action selection: ``argmax_a [g_a + log(π̂[a])]``.

        Adds per-env Gumbel noise to the log-improved-policy and picks
        the argmax over legal actions. The Gumbel noise preserves
        trajectory diversity across envs (each env's noise is
        independent) — direct sampling from ``policies_out`` collapses
        envs onto the same peaked-policy action and kills the
        state-discriminative learning signal.

        Pass ``gumbel_scale=0.0`` at eval / arena for deterministic
        argmax over the improved policy."""
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.legal_mask.unsafe_ptr())
        var po_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.policies_out.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.actions_out.unsafe_ptr())
        comptime run_extract = gz_extract_actions_gumbel_kernel[
            Self.N_ENVS, Self.ACT, dtype,
        ]
        ctx.enqueue_function[run_extract](
            po_t, lm_t, act_t,
            rng_seed,
            Scalar[dtype](gumbel_scale),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    def extract_actions_temp[
        TEMP_THRESHOLD: Int = 0,
    ](
        mut self,
        ctx: DeviceContext,
        ep_steps: LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ],
        rng_seed: UInt32 = UInt32(0),
        temp_min: Float64 = 0.0,
    ) raises:
        """Temperature-controlled action sampling from the improved policy.

        Mirrors ``GenericGPUMCTS.extract_actions_temp`` but reads from
        ``policies_out`` (the Gumbel-MuZero improved policy) instead of
        raw visit counts. The policy target stored in replay stays the
        improved policy — this method only writes ``actions_out``."""
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.legal_mask.unsafe_ptr())
        var po_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.policies_out.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.actions_out.unsafe_ptr())
        comptime run_extract = gz_extract_actions_temp_kernel[
            Self.N_ENVS, Self.ACT, dtype,
        ]
        ctx.enqueue_function[run_extract](
            po_t, lm_t, ep_steps, act_t,
            TEMP_THRESHOLD,
            rng_seed,
            Scalar[dtype](temp_min),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Internal — one MCTS simulation across all envs.
    # ══════════════════════════════════════════════════════════════════════

    def _run_one_sim_gpu[
        REP: RepresentationGPU,
        DYN: DynamicsGPU,
        PRED: PredictionGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut dyn: DYN,
        mut pred: PRED,
        slot: Int,
        apply_legal: Bool,
    ) raises:
        """One sim across all envs: select → dyn → pred → expand → backup.

        Mirrors the per-sim body of the legacy ``_run_one_sim_gpu``.
        ``slot`` is the Gumbel-Top-k root candidate slot to descend into
        (shared across envs — Sequential Halving keeps active sets in
        sync, so the same slot index is valid for every env).
        """
        var vc_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.visit_count.unsafe_ptr())
        var tv_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.total_value.unsafe_ptr())
        var nl_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.node_logits.unsafe_ptr())
        var rw_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.reward.unsafe_ptr())
        var ci_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.child_idx.unsafe_ptr())
        var tvis_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.total_visits.unsafe_ptr())
        var nv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.node_value.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.node_count.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.max_q.unsafe_ptr())
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.legal_mask.unsafe_ptr())
        var rc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_candidates.unsafe_ptr())
        var ra_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_active.unsafe_ptr())
        var hs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        var di_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.DYN_IN), MutAnyOrigin
        ](self.state.dyn_input.unsafe_ptr())
        var pp_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.pending_parent.unsafe_ptr())
        var pa_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.pending_action.unsafe_ptr())
        var sp_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * MAX_DEPTH), MutAnyOrigin
        ](self.state.search_paths.unsafe_ptr())
        var ap_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * MAX_DEPTH), MutAnyOrigin
        ](self.state.action_paths.unsafe_ptr())
        var pl_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.path_lengths.unsafe_ptr())

        # Selection.
        comptime run_select = gz_select_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.MAX_K, Self.LATENT,
            Self.DYN_IN, dtype,
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
            Scalar[dtype](self.c_visit),
            Scalar[dtype](self.c_scale),
            Scalar[DType.uint8](1 if self.qnorm_per_node else 0),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Dynamics forward (via trait adapter).
        var dyn_in_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, DYN.DYN_IN_DIM),
            MutAnyOrigin,
        ](self.state.dyn_input.unsafe_ptr())
        var dyn_out_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, DYN.DYN_OUT_DIM),
            MutAnyOrigin,
        ](self.state.dyn_output.unsafe_ptr())
        dyn.step_gpu[Self.N_ENVS](ctx, dyn_in_b, dyn_out_b)

        # Copy dyn_output's hidden prefix into pred_input.
        comptime run_copy = gz_copy_pred_input_kernel[
            Self.N_ENVS, Self.LATENT, Self.DYN_OUT, dtype,
        ]
        var pred_in_flat = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.LATENT), MutAnyOrigin
        ](self.state.pred_input.unsafe_ptr())
        var dyn_out_flat = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.DYN_OUT), MutAnyOrigin
        ](self.state.dyn_output.unsafe_ptr())
        ctx.enqueue_function[run_copy](
            pred_in_flat,
            dyn_out_flat,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Prediction forward.
        var pred_in_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.LATENT_DIM),
            MutAnyOrigin,
        ](self.state.pred_input.unsafe_ptr())
        var pred_out_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.PRED_OUT_DIM),
            MutAnyOrigin,
        ](self.state.pred_output.unsafe_ptr())
        pred.predict_gpu[Self.N_ENVS](ctx, pred_in_b, pred_out_b)

        # Expand.
        var lv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.leaf_values.unsafe_ptr())
        var po_full_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.PRED_OUT), MutAnyOrigin
        ](self.state.pred_output.unsafe_ptr())
        comptime run_expand = gz_expand_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.BINS,
            Self.PRED_OUT, Self.DYN_OUT, dtype,
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
            Scalar[dtype](self.v_min),
            Scalar[dtype](self.v_max),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Backup. NEGATE_BACKUP comes from the PLAYER comptime param —
        # SelfPlay flips perspective at every ply; SinglePlayer uses the
        # standard ``reward + gamma · value`` recurrence.
        comptime run_backup = gz_backup_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype,
            Self.PLAYER.NEGATE_BACKUP,
        ]
        ctx.enqueue_function[run_backup](
            vc_t,
            tv_t,
            rw_t,
            tvis_t,
            nv_t,
            miq_t,
            mxq_t,
            sp_t,
            ap_t,
            pl_t,
            lv_t,
            Scalar[dtype](self.gamma),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Gumbel AlphaZero — true-env-rules expansion (STATE_SIZE > 0)
    # ══════════════════════════════════════════════════════════════════════

    def search_gpu_alphazero[
        PRED: PredictionGPU,
        ENV: EnvStepGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut pred: PRED,
        mut env: ENV,
        root_obs: LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, PRED.LATENT_DIM), MutAnyOrigin
        ],
        root_states: DeviceBuffer[dtype],
        root_legal: DeviceBuffer[dtype],
        k_actual: Int = Self.MAX_K,
        rng_seed: Scalar[DType.uint32] = Scalar[DType.uint32](0),
    ) raises:
        """Gumbel AlphaZero search (Danihelka et al.): Gumbel-Top-k root
        candidates + Sequential Halving + visit-balance in-tree selection —
        the SAME planner rules as ``search_gpu`` — but the model is the true
        game: tree nodes carry ``STATE_SIZE`` game-state payloads, expansion
        is ``env.step_gpu`` on the parent state, child priors are masked by
        the post-step legal mask, the value head is scalar (``BINS == 1`` →
        tanh squash), and the backup negates per ply (``PLAYER=SelfPlay``).

        Contracts: ``PRED.LATENT_DIM == ENV.OBS_DIM`` (prediction reads obs
        directly — no representation net), ``ENV.STATE_SIZE == STATE_SIZE``,
        ``root_legal`` is ``[N_ENVS × ACT]`` and is always applied (board
        games). Results land in ``policies_view()`` (improved policy) and
        ``root_value_view()``. Serial sims — no virtual loss, no frozen-tree
        duplicate problem by construction."""
        comptime assert Self.STATE_SIZE > 0, (
            "search_gpu_alphazero needs STATE_SIZE > 0 (game-state payloads)"
        )
        comptime assert Self.BINS == 1, (
            "Gumbel AlphaZero uses a scalar value head (BINS == 1, tanh)"
        )

        # ── 1. Reset tree + root legality ────────────────────────────────
        self.state.zero_tree(ctx)
        ctx.enqueue_copy(self.state.legal_mask, root_legal)

        # ── 2. Root predict on obs (no representation net) ───────────────
        var pred_out_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.PRED_OUT_DIM),
            MutAnyOrigin,
        ](self.state.pred_output.unsafe_ptr())
        pred.predict_gpu[Self.N_ENVS](ctx, root_obs, pred_out_b)

        # ── 3. Init root: masked logits + tanh value + Gumbel-Top-k ──────
        var nl_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.node_logits.unsafe_ptr())
        var nv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.node_value.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.node_count.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.max_q.unsafe_ptr())
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.legal_mask.unsafe_ptr())
        var rc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_candidates.unsafe_ptr())
        var rg_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_gumbels.unsafe_ptr())
        var ra_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_active.unsafe_ptr())
        var po_full_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.PRED_OUT), MutAnyOrigin
        ](self.state.pred_output.unsafe_ptr())

        var k_clipped = k_actual
        if k_clipped > Self.MAX_K:
            k_clipped = Self.MAX_K
        if k_clipped > Self.ACT:
            k_clipped = Self.ACT
        k_clipped = _largest_power_of_two_le(k_clipped)
        if k_clipped < 1:
            k_clipped = 1

        comptime run_init = gz_init_root_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.BINS, Self.MAX_K,
            Self.PRED_OUT, dtype,
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
            Scalar[dtype](self.v_min),
            Scalar[dtype](self.v_max),
            Scalar[DType.int32](k_clipped),
            Scalar[DType.uint8](1),
            rng_seed,
            Scalar[dtype](self.gumbel_scale),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 4. Root game state → node 0 ──────────────────────────────────
        comptime AZ_BLOCKS = (
            Self.N_ENVS * Self.STATE_SIZE + TPB - 1
        ) // TPB
        var gs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.STATE_SIZE),
            MutAnyOrigin,
        ](self.az_game_states.unsafe_ptr())
        var rs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.STATE_SIZE),
            MutAnyOrigin,
        ](root_states.unsafe_ptr())
        comptime run_copy_root = gz_az_copy_root_state_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.STATE_SIZE, dtype,
        ]
        ctx.enqueue_function[run_copy_root](
            gs_t, rs_t, grid_dim=(AZ_BLOCKS,), block_dim=(TPB,),
        )

        # ── 5. Sequential Halving simulation loop ────────────────────────
        var num_phases = _ilog2(k_clipped)
        if num_phases < 1:
            num_phases = 1
        var per_phase_budget = Self.NUM_SIMULATIONS // num_phases
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
                    if sims_used >= Self.NUM_SIMULATIONS:
                        break
                    self._run_one_sim_az_gpu[PRED, ENV](
                        ctx, pred, env, slot,
                        UInt64(rng_seed) * UInt64(1000003)
                        + UInt64(sims_used),
                    )
                    sims_used += 1

            if phase + 1 < num_phases and active_size > 1:
                var keep = active_size // 2
                if keep < 1:
                    keep = 1
                comptime run_halve = gz_halve_active_kernel[
                    Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.MAX_K, dtype,
                ]
                var vc_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
                    MutAnyOrigin,
                ](self.state.visit_count.unsafe_ptr())
                var tv_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
                    MutAnyOrigin,
                ](self.state.total_value.unsafe_ptr())
                var tvis_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES),
                    MutAnyOrigin,
                ](self.state.total_visits.unsafe_ptr())
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
                    Scalar[dtype](self.c_visit),
                    Scalar[dtype](self.c_scale),
                    Scalar[DType.uint8](1 if self.qnorm_per_node else 0),
                    grid_dim=(Self.ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                active_size = keep

        while sims_used < Self.NUM_SIMULATIONS:
            self._run_one_sim_az_gpu[PRED, ENV](
                ctx, pred, env, 0,
                UInt64(rng_seed) * UInt64(1000003) + UInt64(sims_used),
            )
            sims_used += 1

        # ── 6. Extract improved policy (legal-masked) + root value ───────
        var po_extract_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.policies_out.unsafe_ptr())
        var vc_t2 = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.visit_count.unsafe_ptr())
        var tv_t2 = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.total_value.unsafe_ptr())
        var tvis_t2 = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.total_visits.unsafe_ptr())
        comptime run_extract = gz_extract_policy_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype,
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
            Scalar[DType.uint8](1),
            Scalar[dtype](self.c_visit),
            Scalar[dtype](self.c_scale),
            Scalar[DType.uint8](1 if self.qnorm_per_node else 0),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        var rv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.root_value_out.unsafe_ptr())
        comptime run_root_value = gz_extract_root_value_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype,
        ]
        ctx.enqueue_function[run_root_value](
            vc_t2,
            tv_t2,
            nv_t,
            rv_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    def _run_one_sim_az_gpu[
        PRED: PredictionGPU,
        ENV: EnvStepGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut pred: PRED,
        mut env: ENV,
        slot: Int,
        step_seed: UInt64,
    ) raises:
        """One AZ sim across all envs: select → stage state → env.step →
        pred → expand (masked, tanh leaf) → negated backup."""
        var vc_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.visit_count.unsafe_ptr())
        var tv_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.total_value.unsafe_ptr())
        var nl_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.node_logits.unsafe_ptr())
        var rw_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.reward.unsafe_ptr())
        var ci_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.child_idx.unsafe_ptr())
        var tvis_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.total_visits.unsafe_ptr())
        var nv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.node_value.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.node_count.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.max_q.unsafe_ptr())
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.legal_mask.unsafe_ptr())
        var rc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_candidates.unsafe_ptr())
        var ra_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_active.unsafe_ptr())
        var hs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        var di_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.DYN_IN), MutAnyOrigin
        ](self.state.dyn_input.unsafe_ptr())
        var pp_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.pending_parent.unsafe_ptr())
        var pa_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.pending_action.unsafe_ptr())
        var sp_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * MAX_DEPTH), MutAnyOrigin
        ](self.state.search_paths.unsafe_ptr())
        var ap_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * MAX_DEPTH), MutAnyOrigin
        ](self.state.action_paths.unsafe_ptr())
        var pl_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.path_lengths.unsafe_ptr())

        # Selection — the shared Gumbel visit-balance kernel. It also builds
        # a dynamics input from the (unused, uninitialized) hidden pool; that
        # write is dead on the AZ path and harmless.
        comptime run_select = gz_select_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.MAX_K, Self.LATENT,
            Self.DYN_IN, dtype,
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
            Scalar[DType.uint8](1),
            Scalar[dtype](self.c_visit),
            Scalar[dtype](self.c_scale),
            Scalar[DType.uint8](1 if self.qnorm_per_node else 0),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Stage the leaf-parent's game state for expansion.
        comptime AZ_BLOCKS = (
            Self.N_ENVS * Self.STATE_SIZE + TPB - 1
        ) // TPB
        var gs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.STATE_SIZE),
            MutAnyOrigin,
        ](self.az_game_states.unsafe_ptr())
        var es_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.STATE_SIZE),
            MutAnyOrigin,
        ](self.az_expansion_states.unsafe_ptr())
        comptime run_stage = gz_az_stage_state_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.STATE_SIZE, dtype,
        ]
        ctx.enqueue_function[run_stage](
            es_t, gs_t, pp_t, grid_dim=(AZ_BLOCKS,), block_dim=(TPB,),
        )

        # True-rules expansion: env.step on the staged states. Post-step obs
        # lands directly in pred_input (PRED.LATENT_DIM == ENV.OBS_DIM).
        env.step_gpu[Self.N_ENVS](
            ctx,
            self.az_expansion_states,
            self.state.pending_action,
            self.az_step_rewards,
            self.az_step_dones,
            self.az_step_terminated,
            self.state.pred_input,
            self.az_exp_legal,
            step_seed,
        )

        # Prediction on the post-step obs.
        var pred_in_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.LATENT_DIM),
            MutAnyOrigin,
        ](self.state.pred_input.unsafe_ptr())
        var pred_out_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.PRED_OUT_DIM),
            MutAnyOrigin,
        ](self.state.pred_output.unsafe_ptr())
        pred.predict_gpu[Self.N_ENVS](ctx, pred_in_b, pred_out_b)

        # Expand (masked child logits, tanh/terminal leaf value).
        var lv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.leaf_values.unsafe_ptr())
        var el_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.az_exp_legal.unsafe_ptr())
        var sr_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.az_step_rewards.unsafe_ptr())
        var sd_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.az_step_dones.unsafe_ptr())
        var po_full_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.PRED_OUT), MutAnyOrigin
        ](self.state.pred_output.unsafe_ptr())
        comptime run_expand = gz_az_expand_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.STATE_SIZE,
            Self.PRED_OUT, dtype,
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
            gs_t,
            es_t,
            el_t,
            pp_t,
            pa_t,
            sr_t,
            sd_t,
            po_full_t,
            lv_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Backup — negated per ply for SelfPlay (from the PLAYER trait).
        comptime run_backup = gz_backup_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype,
            Self.PLAYER.NEGATE_BACKUP,
        ]
        ctx.enqueue_function[run_backup](
            vc_t,
            tv_t,
            rw_t,
            tvis_t,
            nv_t,
            miq_t,
            mxq_t,
            sp_t,
            ap_t,
            pl_t,
            lv_t,
            Scalar[dtype](self.gamma),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )
