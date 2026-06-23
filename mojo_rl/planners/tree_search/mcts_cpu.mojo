"""Generic CPU MCTS — Monte Carlo Tree Search parameterized by the model
contract and a handful of strategy traits.

The previous home for this code, ``mojo_rl/deep_agents/muzero/mcts.mojo``,
hard-coded MuZero's networks (``Network[RepModel, RepOpt]`` etc.),
MuZero's categorical reward / value decode, MuZero's PUCT formula, and
MuZero's MinMax hidden-state scaling. AlphaZero already needed half of
those to be different (true game rules instead of dynamics, no MinMax
scaling, no scalar transform), and EZv2 wants further variants. The
agent-coupled struct made every variation a fork.

This file is Phase 3 of the planners refactor (CPU half — see
``docs/PLANNERS_PACKAGE.md``). It replaces the agent coupling with two
sets of comptime parameters:

* **Model traits** — ``Representation`` / ``Dynamics`` / ``Prediction``
  (from ``model_traits.mojo``). The adapter that wraps each agent's
  networks lives next to the agent; MCTS sees only ``List[Float64]``.
  Adapters own encoding/decoding/scaling, so MCTS is independent of
  ``ValueEncoding`` and ``HiddenScaling`` — those choices live with the
  agent.

* **Strategy traits** — ``PUCTFormula`` / ``ExplorationNoise`` /
  ``PlayerMode`` (from ``strategies.mojo``). Compile-time dispatch into
  the formulas inside the hot selection / backup loops.

What MCTS still owns:
  * Tree topology (``MCTSNode`` per-action stats).
  * MinMax Q-normalization (always on; trivial cost; if a future
    variant wants raw Q values it can pre-clamp the range).
  * Root prior building (legal mask + exploration noise + softmax-free
    re-normalization).
  * Visit-count policy at the end.

What MCTS no longer touches:
  * Network forward passes.
  * Categorical reward / value decode.
  * Hidden-state min-max scaling.
  * Scalar transforms.

The old ``MCTS`` struct in ``muzero/mcts.mojo`` is left in place for the
strangler migration (see the doc). When MuZero / EZv2 / AlphaZero get
trait adapters, the legacy struct can be deleted.
"""

from std.math import sqrt, log, exp
from std.random import random_float64

from mojo_rl.planners.common.min_max_stats import MinMaxStats

from .model_traits import Representation, Dynamics, Prediction
from .strategies import (
    PUCTFormula,
    ExplorationNoise,
    PlayerMode,
)


# ═══════════════════════════════════════════════════════════════════════════
# MCTSNode — per-action statistics
# ═══════════════════════════════════════════════════════════════════════════


struct MCTSNode[ACTION_DIM: Int](ImplicitlyCopyable, Movable):
    """A single node in the MCTS search tree.

    Layout mirrors ``muzero/mcts.MCTSNode`` so behavior parity is
    obvious in code review. Per-action arrays are ``InlineArray`` for
    cache locality — typical ACTION_DIM is small (≤ 64) for the agents
    that use MCTS.

    Parameters:
        ACTION_DIM: Number of discrete actions available.
    """

    var visit_count: InlineArray[Int, Self.ACTION_DIM]
    """N(s, a) — visit count per action."""

    var total_value: InlineArray[Float64, Self.ACTION_DIM]
    """W(s, a) — sum of backed-up values per action."""

    var prior: InlineArray[Float64, Self.ACTION_DIM]
    """P(s, a) — policy prior (probabilities, sum to 1)."""

    var reward: InlineArray[Float64, Self.ACTION_DIM]
    """R(s, a) — scalar reward observed when taking action ``a`` from
    this node. Filled in by ``Dynamics.step_cpu`` at expansion time."""

    var child_idx: InlineArray[Int, Self.ACTION_DIM]
    """Index into ``MCTS.nodes`` of the child reached by action ``a``,
    or ``-1`` if unexpanded."""

    var total_visits: Int
    """N(s) = sum_a N(s, a)."""

    var hidden_state_idx: Int
    """Slot in the ``hidden_states`` pool that holds this node's state.
    Same as the node's own index in the typical case."""

    def __init__(out self, hidden_idx: Int):
        self.visit_count = InlineArray[Int, Self.ACTION_DIM](uninitialized=True)
        self.total_value = InlineArray[Float64, Self.ACTION_DIM](
            uninitialized=True
        )
        self.prior = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        self.reward = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        self.child_idx = InlineArray[Int, Self.ACTION_DIM](uninitialized=True)
        for a in range(Self.ACTION_DIM):
            self.visit_count[a] = 0
            self.total_value[a] = Float64(0.0)
            self.prior[a] = Float64(1.0) / Float64(Self.ACTION_DIM)
            self.reward[a] = Float64(0.0)
            self.child_idx[a] = -1
        self.total_visits = 0
        self.hidden_state_idx = hidden_idx

    def __init__(out self, *, copy: Self):
        self.visit_count = copy.visit_count
        self.total_value = copy.total_value
        self.prior = copy.prior
        self.reward = copy.reward
        self.child_idx = copy.child_idx
        self.total_visits = copy.total_visits
        self.hidden_state_idx = copy.hidden_state_idx

    def __init__(out self, *, deinit take: Self):
        self.visit_count = take.visit_count
        self.total_value = take.total_value
        self.prior = take.prior
        self.reward = take.reward
        self.child_idx = take.child_idx
        self.total_visits = take.total_visits
        self.hidden_state_idx = take.hidden_state_idx

    def mean_value(self, action: Int) -> Float64:
        """Q(s, a) = W(s, a) / N(s, a), or 0 if unvisited."""
        if self.visit_count[action] > 0:
            return self.total_value[action] / Float64(self.visit_count[action])
        return Float64(0.0)

    def is_expanded(self, action: Int) -> Bool:
        return self.child_idx[action] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


@always_inline
def _sample_dirichlet_approx[
    ACTION_DIM: Int
]() -> InlineArray[Float64, ACTION_DIM]:
    """Approximate symmetric Dirichlet by normalizing Gamma(1, 1) samples.

    Matches the existing ``muzero/mcts.search`` approach: ``-log(U)``
    is an Exp(1) sample, which is Gamma(1, 1); normalizing N of them
    gives a Dirichlet(1, ..., 1). For small alpha (Atari = 0.25) this
    is approximate but the only difference vs a true Gamma(alpha)
    sampler is concentration; for the noise-fraction blend at the root
    it doesn't measurably affect policy. Kept for behavior parity.
    """
    var out = InlineArray[Float64, ACTION_DIM](uninitialized=True)
    var s = Float64(0.0)
    for a in range(ACTION_DIM):
        var u = random_float64(0.0001, 0.9999)
        var g = -log(u)
        out[a] = g
        s += g
    if s > 0.0:
        for a in range(ACTION_DIM):
            out[a] /= s
    else:
        var inv = Float64(1.0) / Float64(ACTION_DIM)
        for a in range(ACTION_DIM):
            out[a] = inv
    return out


@always_inline
def _apply_legal_mask[
    ACTION_DIM: Int
](mut prior: InlineArray[Float64, ACTION_DIM], legal_mask: List[Bool],):
    """Zero out illegal-action prior entries and renormalize.

    If all legal actions had zero prior, fall back to uniform over the
    legal set (matches behavior of the old ``muzero/mcts.search``).
    """
    if len(legal_mask) != ACTION_DIM:
        return
    var mask_sum = Float64(0.0)
    for a in range(ACTION_DIM):
        if not legal_mask[a]:
            prior[a] = Float64(0.0)
        else:
            mask_sum += prior[a]
    if mask_sum > 0.0:
        for a in range(ACTION_DIM):
            prior[a] /= mask_sum
    else:
        var legal_count = Float64(0.0)
        for a in range(ACTION_DIM):
            if legal_mask[a]:
                legal_count += 1.0
        if legal_count > 0.0:
            for a in range(ACTION_DIM):
                if legal_mask[a]:
                    prior[a] = 1.0 / legal_count


# ═══════════════════════════════════════════════════════════════════════════
# Generic CPU MCTS
# ═══════════════════════════════════════════════════════════════════════════


struct GenericCPUMCTS[
    ACTION_DIM: Int,
    LATENT_DIM: Int,
    NUM_SIMULATIONS: Int,
    MAX_NODES: Int,
    PUCT: PUCTFormula,
    NOISE: ExplorationNoise,
    PLAYER: PlayerMode,
    BATCH_SIMS: Int = 1,
    VIRTUAL_LOSS: Int = 3,
    NORMALIZE_Q: Bool = True,
](ImplicitlyDeletable, Movable):
    """CPU MCTS parameterized by the model contract + strategy traits.

    Comptime params:
        ACTION_DIM: Discrete action count.
        LATENT_DIM: Hidden state dimension exchanged with the adapter.
        NUM_SIMULATIONS: Simulations per ``search`` call.
        MAX_NODES: Pool size for the node + hidden-state arenas.
        PUCT: PUCT formula trait (e.g. ``MuZeroPUCT``, ``AlphaGoPUCT``).
        NOISE: Root exploration noise (``DirichletNoise``,
            ``EpsilonNoise``, ``NoNoise``).
        PLAYER: ``SinglePlayer`` or ``SelfPlay`` — toggles negation in
            ``_backup`` (matches reference AlphaZero behavior).
        BATCH_SIMS: Number of simulations selected+expanded per round
            before any backup runs. ``1`` (default) = pure sequential
            (each sim's backup runs immediately, vloss has no net
            effect within a sim). Larger values diversify root-level
            exploration by holding virtual loss in place across
            multiple in-flight sims — required to compensate for
            spiky Dirichlet root priors (e.g. CartPole's
            ``DirichletNoise[0.25, 0.25]``) that otherwise lock a
            sequential PUCT walk onto a single noise-picked action.
        VIRTUAL_LOSS: Magnitude added to ``visit_count`` (and
            subtracted from per-action mean Q) at every descent step
            during selection. ``3`` is the canonical AlphaGo Zero
            value. Net visit change per sim per step is ``+1``
            (``+VIRTUAL_LOSS`` during select, ``-(VIRTUAL_LOSS-1)``
            during backup), so the final tree state matches the
            no-vloss reference. Setting to ``0`` disables vloss
            entirely.

    Vloss is applied at EVERY descent step (Phase D Bug 2 correct fix
    from ``docs/PHASE_D_GPU_MCTS_BUG_HUNT.md``), not just at the leaf
    parent — without this, all sims in a batch share the same root
    action and only diversify at the deepest level.

    Runtime ctor args carry only what's left after strategy traits:
        gamma: discount factor used in the backup recursion.

    Strategy hyperparams (``c_base``, ``c_init``, noise alpha/fraction)
    are pulled directly from the trait comptime constants — no runtime
    duplication.
    """

    var nodes: List[MCTSNode[Self.ACTION_DIM]]
    """Node arena. Cleared and rebuilt per ``search`` call."""

    var hidden_states: List[Float64]
    """``MAX_NODES × LATENT_DIM`` flat pool — one slot per node (owned List,
    RAII — no manual alloc/free, no MutAnyOrigin)."""

    var min_max: MinMaxStats
    """Q-value range tracker for PUCT normalization."""

    var gamma: Float64
    """Discount factor used in the per-step backup recursion."""

    def __init__(out self, gamma: Float64 = 0.997):
        self.nodes = List[MCTSNode[Self.ACTION_DIM]](capacity=Self.MAX_NODES)
        self.hidden_states = List[Float64](
            length=Self.MAX_NODES * Self.LATENT_DIM, fill=Float64(0.0)
        )
        self.min_max = MinMaxStats()
        self.gamma = gamma

    def __init__(out self, *, deinit take: Self):
        self.nodes = take.nodes^
        self.hidden_states = take.hidden_states^
        self.min_max = take.min_max
        self.gamma = take.gamma

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def search[
        REP: Representation,
        DYN: Dynamics,
        PRED: Prediction,
    ](
        mut self,
        mut rep: REP,
        mut dyn: DYN,
        mut pred: PRED,
        root_obs: List[Float64],
        add_noise: Bool = True,
        legal_mask: List[Bool] = List[Bool](),
    ) raises -> InlineArray[Float64, Self.ACTION_DIM]:
        """Run ``NUM_SIMULATIONS`` MCTS simulations from ``root_obs``.

        Phases:
          1. Reset tree + MinMax stats.
          2. Encode root obs via ``Representation`` adapter.
          3. Predict root prior + value via ``Prediction`` adapter.
          4. Apply legal mask (renormalize over legal actions).
          5. Blend in exploration noise per the ``NOISE`` trait.
          6. Run simulations: select (PUCT) → expand (Dynamics +
             Prediction) → backup (per ``PLAYER`` transform).
          7. Return the normalized visit-count policy.

        Args:
            rep: Trait-implementing adapter wrapping the agent's networks representation (or stub).
            dyn: Trait-implementing adapter wrapping the agent's networks dynamics (or stub).
            pred: Trait-implementing adapter wrapping the agent's networks prediction (or stub).
            root_obs: Length ``OBS_DIM`` observation.
            add_noise: Whether to apply root exploration noise. Set
                ``False`` for evaluation runs; ``True`` for self-play /
                training.
            legal_mask: Optional length-``ACTION_DIM`` mask of legal
                actions. Empty list means all legal (back-compat with
                the old MuZero default).

        Returns:
            Visit-count probabilities at the root (sum to 1).
        """
        # ── 1. Reset state ───────────────────────────────────────────
        self.nodes.clear()
        self.min_max = MinMaxStats()

        # ── 2. Encode root obs ───────────────────────────────────────
        var root_hidden = List[Float64](
            length=Self.LATENT_DIM, fill=Float64(0.0)
        )
        rep.encode_cpu(root_obs, root_hidden)

        # Copy root hidden into slot 0 of the pool.
        for i in range(Self.LATENT_DIM):
            self.hidden_states[i] = root_hidden[i]

        # ── 3. Predict root prior + value ────────────────────────────
        var root_prior_list = List[Float64](
            length=Self.ACTION_DIM, fill=Float64(0.0)
        )
        var _root_value = pred.predict_cpu(root_hidden, root_prior_list)
        # The root's predicted scalar value is unused: the visit-count
        # policy is what gets returned; the value is consumed by the
        # *caller* via the root node's mean_value (see muzero.mojo:746).

        var root = MCTSNode[Self.ACTION_DIM](hidden_idx=0)
        for a in range(Self.ACTION_DIM):
            root.prior[a] = root_prior_list[a]

        # ── 4. Apply legal mask ──────────────────────────────────────
        _apply_legal_mask[Self.ACTION_DIM](root.prior, legal_mask)

        # ── 5. Exploration noise ─────────────────────────────────────
        if add_noise:
            comptime if Self.NOISE.NOISE_TYPE == 0:
                # Dirichlet
                var noise = _sample_dirichlet_approx[Self.ACTION_DIM]()
                var frac = Self.NOISE.NOISE_FRACTION
                for a in range(Self.ACTION_DIM):
                    root.prior[a] = (1.0 - frac) * root.prior[a] + frac * noise[
                        a
                    ]
            elif Self.NOISE.NOISE_TYPE == 1:
                # Uniform epsilon
                var frac = Self.NOISE.NOISE_FRACTION
                var u = Float64(1.0) / Float64(Self.ACTION_DIM)
                for a in range(Self.ACTION_DIM):
                    root.prior[a] = (1.0 - frac) * root.prior[a] + frac * u
            # NOISE_TYPE == 2: no-op.

        self.nodes.append(root^)

        # ── 6. Simulations — batched, with virtual loss ─────────────
        # Each batch round selects up to ``BATCH_SIMS`` leaves with
        # virtual loss at every descent step, expands each, then runs
        # backup for the whole batch. Holding vloss across the batch
        # is what diversifies root-level action choice when the
        # network's prior + Dirichlet noise concentrates strongly on
        # one action. With ``BATCH_SIMS=1`` the math degenerates to
        # plain sequential MCTS (vloss applied then immediately
        # removed within the same sim's backup — net zero effect).
        comptime BSIMS: Int = Self.BATCH_SIMS
        comptime VLOSS: Int = Self.VIRTUAL_LOSS

        var sims_done: Int = 0
        while sims_done < Self.NUM_SIMULATIONS:
            var batch_n: Int = BSIMS
            if sims_done + batch_n > Self.NUM_SIMULATIONS:
                batch_n = Self.NUM_SIMULATIONS - sims_done

            # Per-batch scratch — each sim's path/actions/leaf_value
            # held until phase 2 so vloss stays in place during all
            # selections in this round.
            var batch_paths = List[List[Int]](capacity=batch_n)
            var batch_actions = List[List[Int]](capacity=batch_n)
            var batch_leafs = List[Float64](capacity=batch_n)
            var batch_valid = List[Bool](capacity=batch_n)

            # Phase 1: select + expand each sim in the batch.
            for _ in range(batch_n):
                var path = List[Int](capacity=64)
                var actions = List[Int](capacity=64)
                var node_idx: Int = 0
                path.append(node_idx)

                while True:
                    var action = self._select_action(node_idx)
                    actions.append(action)

                    # Virtual loss at every descent step. Without this,
                    # all sims in a batch follow the same root action
                    # and only diversify at the deepest level (the
                    # Phase D Bug 2 pattern).
                    comptime if VLOSS != 0:
                        self.nodes[node_idx].visit_count[
                            action
                        ] += VLOSS
                        self.nodes[node_idx].total_visits += VLOSS

                    if not self.nodes[node_idx].is_expanded(action):
                        break
                    node_idx = self.nodes[node_idx].child_idx[action]
                    path.append(node_idx)

                var parent_idx = path[len(path) - 1]
                var leaf_action = actions[len(actions) - 1]

                # Tree overflow → undo vloss along this path and skip
                # the sim. Mirrors the legacy MuZero MCTS overflow path
                # and the AZ Phase D Bug 2 fix's "remove vloss along
                # the FULL path, not just at the leaf parent".
                if len(self.nodes) >= Self.MAX_NODES:
                    comptime if VLOSS != 0:
                        for i in range(len(path)):
                            var nd = path[i]
                            var ac = actions[i]
                            self.nodes[nd].visit_count[ac] -= VLOSS
                            self.nodes[nd].total_visits -= VLOSS
                    batch_valid.append(False)
                    batch_paths.append(path^)
                    batch_actions.append(actions^)
                    batch_leafs.append(Float64(0.0))
                    continue

                var child_hidden_idx = len(self.nodes)
                var leaf_value = self._expand_node[REP, DYN, PRED](
                    parent_idx, leaf_action, child_hidden_idx, dyn, pred
                )
                batch_valid.append(True)
                batch_paths.append(path^)
                batch_actions.append(actions^)
                batch_leafs.append(leaf_value)

            # Phase 2: backup all valid sims. Each backup nets out the
            # vloss it applied during selection so the final tree state
            # matches a no-vloss reference run.
            for b in range(batch_n):
                if batch_valid[b]:
                    self._backup_batched(
                        batch_paths[b], batch_actions[b], batch_leafs[b]
                    )

            sims_done += batch_n

        # ── 7. Visit-count policy ────────────────────────────────────
        var policy = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        var total = Float64(0.0)
        for a in range(Self.ACTION_DIM):
            policy[a] = Float64(self.nodes[0].visit_count[a])
            total += policy[a]
        if total > 0.0:
            for a in range(Self.ACTION_DIM):
                policy[a] /= total
        else:
            for a in range(Self.ACTION_DIM):
                policy[a] = 1.0 / Float64(Self.ACTION_DIM)
        return policy

    def root_value(self) -> Float64:
        """Visit-weighted Q at the root after ``search`` completed.

        Mirrors ``muzero/muzero.mojo:746`` — caller uses this as the
        agent's ``V(root)`` estimate for n-step bootstrapping. Safe to
        call after ``search``; returns 0 before any search has run.
        """
        if len(self.nodes) == 0:
            return Float64(0.0)
        var root = self.nodes[0]
        if root.total_visits == 0:
            return Float64(0.0)
        var v = Float64(0.0)
        for a in range(Self.ACTION_DIM):
            if root.visit_count[a] > 0:
                v += (
                    Float64(root.visit_count[a]) / Float64(root.total_visits)
                ) * root.mean_value(a)
        return v

    # ══════════════════════════════════════════════════════════════════════
    # Internals
    # ══════════════════════════════════════════════════════════════════════

    def _select_action(self, node_idx: Int) -> Int:
        """PUCT: argmax_a [Q̂(s, a) + c(s) · P(s, a) · √N(s) / (1+N(s,a))].

        Q̂ is MinMax-normalized to [0, 1]. ``c(s)`` comes from the
        compile-time ``PUCT`` trait so MuZero / AlphaGo / UCB1 all live
        in one branch.

        For ``PLAYER.USE_LEGAL_MASK = True`` (zero-sum AlphaZero-style),
        zero-prior actions are skipped — they encode hard illegality
        (``predict_cpu`` zeroes illegal entries before renormalizing).
        At an unvisited node ``sqrt(N) = 0`` forces every explore term
        to 0, so without this skip the argmax would tie-break to
        ``action 0`` even when illegal. For ``USE_LEGAL_MASK = False``
        (single-player MuZero etc.) the skip is bypassed to preserve
        bit-parity with the legacy ``muzero/mcts.MCTS`` — softmax
        priors there are strictly positive anyway, so the practical
        difference is the legal-mask edge case that the parity test in
        ``tests/planners/tree_search/test_mcts_cpu_parity_muzero.mojo``
        documents.
        """
        var node = self.nodes[node_idx]
        var sqrt_total = sqrt(Float64(node.total_visits))

        var c = Self.PUCT.compute_c(
            Float64(node.total_visits),
            Self.PUCT.C_BASE,
            Self.PUCT.C_INIT,
        )

        var best_action: Int = 0
        var best_score = Float64(-1.0e18)
        for a in range(Self.ACTION_DIM):
            comptime if Self.PLAYER.USE_LEGAL_MASK:
                if node.prior[a] <= Float64(0.0):
                    continue

            var q: Float64
            if node.visit_count[a] > 0:
                comptime if Self.NORMALIZE_Q:
                    q = self.min_max.normalize(node.mean_value(a))
                else:
                    q = node.mean_value(a)
            else:
                q = Float64(0.0)

            var explore = (
                c
                * node.prior[a]
                * sqrt_total
                / (1.0 + Float64(node.visit_count[a]))
            )
            var score = q + explore
            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def _expand_node[
        REP: Representation,
        DYN: Dynamics,
        PRED: Prediction,
    ](
        mut self,
        parent_idx: Int,
        action: Int,
        child_hidden_idx: Int,
        mut dyn: DYN,
        mut pred: PRED,
    ) raises -> Float64:
        """Apply Dynamics + Prediction at the leaf, return leaf value.

        Writes the new child node into ``self.nodes`` and links it from
        the parent. Returns the prediction's scalar value at the child
        hidden state — the seed for ``_backup``.
        """
        # Pull parent hidden out of the pool.
        var parent_h = List[Float64](length=Self.LATENT_DIM, fill=Float64(0.0))
        var parent_off = (
            self.nodes[parent_idx].hidden_state_idx * Self.LATENT_DIM
        )
        for i in range(Self.LATENT_DIM):
            parent_h[i] = self.hidden_states[parent_off + i]

        # Dynamics — reward + next hidden.
        var child_h = List[Float64](length=Self.LATENT_DIM, fill=Float64(0.0))
        var reward = dyn.step_cpu(parent_h, action, child_h)

        # Store child hidden in the pool slot the caller chose.
        var child_off = child_hidden_idx * Self.LATENT_DIM
        for i in range(Self.LATENT_DIM):
            self.hidden_states[child_off + i] = child_h[i]

        # Prediction — prior + leaf value (scalar).
        var child_prior_list = List[Float64](
            length=Self.ACTION_DIM, fill=Float64(0.0)
        )
        var leaf_value = pred.predict_cpu(child_h, child_prior_list)

        # Build the child node.
        var child = MCTSNode[Self.ACTION_DIM](hidden_idx=child_hidden_idx)
        for a in range(Self.ACTION_DIM):
            child.prior[a] = child_prior_list[a]

        # Link parent → child.
        self.nodes[parent_idx].reward[action] = reward
        self.nodes[parent_idx].child_idx[action] = child_hidden_idx
        self.nodes.append(child^)

        return leaf_value

    def _backup_batched(
        mut self,
        search_path: List[Int],
        actions_path: List[Int],
        leaf_value: Float64,
    ):
        """Propagate ``leaf_value`` up the search path, netting out vloss.

        Per-step transform is delegated to ``PLAYER.backup_transform``:
            SinglePlayer: value ← reward + gamma * value
            SelfPlay:     value ← -value (zero-sum, no per-edge reward)

        Visit-count delta per step is ``1 - VIRTUAL_LOSS`` so the
        ``+VIRTUAL_LOSS`` applied during selection nets to exactly
        ``+1`` (one real visit per sim per step) once both phases of
        the batch round complete. For ``VIRTUAL_LOSS=0`` this reduces
        to the classic ``+1`` sequential backup.
        """
        comptime VLOSS: Int = Self.VIRTUAL_LOSS
        comptime DELTA: Int = 1 - VLOSS

        var value = leaf_value
        var path_len = len(search_path)
        for i in range(path_len):
            var idx = path_len - 1 - i
            var node_idx = search_path[idx]
            var action = actions_path[idx]

            var edge_reward = self.nodes[node_idx].reward[action]
            value = Self.PLAYER.backup_transform(value, edge_reward, self.gamma)

            self.nodes[node_idx].visit_count[action] += DELTA
            self.nodes[node_idx].total_value[action] += value
            self.nodes[node_idx].total_visits += DELTA
            self.min_max.update(self.nodes[node_idx].mean_value(action))
