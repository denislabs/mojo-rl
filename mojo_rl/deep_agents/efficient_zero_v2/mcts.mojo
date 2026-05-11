"""EfficientZero V2 Gumbel-search MCTS — Phase 1 (CPU).

Replaces MuZero's PUCT MCTS with the Gumbel scheme of Danihelka et al. 2021:

  • Root sampling: draw g(a) ~ Gumbel(0) for every action; expand only the
    top-K of (logits(a) + g(a)) — Gumbel-Top-k sampling without replacement.
  • Sequential Halving bandit at root: log2(K) phases; each phase distributes
    its share of the simulation budget evenly across the surviving
    candidates, then keeps the top half (by g(a) + logits(a) + σ(Q̂(a))).
  • Non-root selection: deterministic visit-balance rule that drives the
    empirical visit distribution toward the improved policy

        π_improved(a) = softmax(logits(a) + σ(completed_Q(a)))
        σ(q)           = (c_visit + max_b N(s,b)) · c_scale · normalize(q)

    Concretely we pick a* = argmax_a [π_improved(a) − N(s,a)/(1+ΣN(s,b))].
    `completed_Q(a)` uses the empirical mean Q for visited children and a
    visit-weighted v_mix for unvisited children.

The search reuses MuZero's `MinMaxStats` (Q normalization), the inverse
scalar transform (categorical → scalar values), and the same
representation/dynamics/prediction Network triple — no new training code is
introduced in Phase 1.

References:
    Danihelka, Guez, Schrittwieser, Silver — *Policy improvement by planning
    with Gumbel*, ICLR 2022.
    Wang, Sun, Li et al. — *EfficientZero V2*, ICML 2024.
"""

from std.math import sqrt, log, exp
from std.memory import alloc, memset
from std.random import random_float64
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.deep_agents.muzero.utils import (
    MinMaxStats,
    inverse_scalar_transform,
)


# ═════════════════════════════════════════════════════════════════════════
# Tree node
# ═════════════════════════════════════════════════════════════════════════


struct GumbelMCTSNode[ACTION_DIM: Int](ImplicitlyCopyable, Movable):
    """A node in the Gumbel-search tree.

    Stores per-action statistics, raw policy logits (kept un-softmaxed because
    the improved-policy formula in Gumbel MuZero adds σ(Q) to *logits* before
    re-softmaxing), the network value estimate at this node (used as the
    "v_root" term in v_mix for unvisited children), and child-node indices.

    Parameters:
        ACTION_DIM: Number of discrete actions.
    """

    var visit_count: InlineArray[Int, Self.ACTION_DIM]
    var total_value: InlineArray[Float64, Self.ACTION_DIM]
    var logits: InlineArray[Float64, Self.ACTION_DIM]
    var reward: InlineArray[Float64, Self.ACTION_DIM]
    var child_idx: InlineArray[Int, Self.ACTION_DIM]
    var legal: InlineArray[Bool, Self.ACTION_DIM]

    var total_visits: Int
    var hidden_state_idx: Int
    var value_estimate: Float64

    def __init__(out self, hidden_idx: Int):
        self.visit_count = InlineArray[Int, Self.ACTION_DIM](uninitialized=True)
        self.total_value = InlineArray[Float64, Self.ACTION_DIM](
            uninitialized=True
        )
        self.logits = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        self.reward = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        self.child_idx = InlineArray[Int, Self.ACTION_DIM](uninitialized=True)
        self.legal = InlineArray[Bool, Self.ACTION_DIM](uninitialized=True)
        for a in range(Self.ACTION_DIM):
            self.visit_count[a] = 0
            self.total_value[a] = Float64(0.0)
            self.logits[a] = Float64(0.0)
            self.reward[a] = Float64(0.0)
            self.child_idx[a] = -1
            self.legal[a] = True
        self.total_visits = 0
        self.hidden_state_idx = hidden_idx
        self.value_estimate = Float64(0.0)

    def __init__(out self, *, copy: Self):
        self.visit_count = copy.visit_count
        self.total_value = copy.total_value
        self.logits = copy.logits
        self.reward = copy.reward
        self.child_idx = copy.child_idx
        self.legal = copy.legal
        self.total_visits = copy.total_visits
        self.hidden_state_idx = copy.hidden_state_idx
        self.value_estimate = copy.value_estimate

    def __init__(out self, *, deinit take: Self):
        self.visit_count = take.visit_count
        self.total_value = take.total_value
        self.logits = take.logits
        self.reward = take.reward
        self.child_idx = take.child_idx
        self.legal = take.legal
        self.total_visits = take.total_visits
        self.hidden_state_idx = take.hidden_state_idx
        self.value_estimate = take.value_estimate

    def mean_value(self, action: Int) -> Float64:
        if self.visit_count[action] > 0:
            return self.total_value[action] / Float64(
                self.visit_count[action]
            )
        return Float64(0.0)

    def is_expanded(self, action: Int) -> Bool:
        return self.child_idx[action] >= 0


# ═════════════════════════════════════════════════════════════════════════
# Gumbel search engine
# ═════════════════════════════════════════════════════════════════════════


struct GumbelMCTS[
    ACTION_DIM: Int,
    LATENT_DIM: Int,
    NUM_BINS: Int = 101,
    NUM_SIMULATIONS: Int = 32,
    NUM_ROOT_CANDIDATES: Int = 16,
    MAX_NODES: Int = 256,
](Movable):
    """Gumbel-Top-k MCTS with Sequential Halving bandit at the root.

    Parameters:
        ACTION_DIM: Number of discrete actions.
        LATENT_DIM: Hidden-state dimension (must match dynamics output).
        NUM_BINS: Categorical value/reward bin count.
        NUM_SIMULATIONS: Total simulation budget per `search()` call.
        NUM_ROOT_CANDIDATES: Maximum K — sampled candidates at root. Clipped
            to ACTION_DIM at runtime if larger.
        MAX_NODES: Maximum tree nodes. Includes both the root and any non-
            root expansions.
    """

    var nodes: List[GumbelMCTSNode[Self.ACTION_DIM]]
    var hidden_states: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var min_max: MinMaxStats

    # Hyperparameters (Danihelka 2021 default values).
    var gamma: Float64
    var c_visit: Float64
    var c_scale: Float64

    def __init__(
        out self,
        gamma: Float64 = 0.997,
        c_visit: Float64 = 50.0,
        c_scale: Float64 = 0.1,
    ):
        self.nodes = List[GumbelMCTSNode[Self.ACTION_DIM]](
            capacity=Self.MAX_NODES
        )
        self.hidden_states = alloc[Scalar[dtype]](
            Self.MAX_NODES * Self.LATENT_DIM
        )
        memset(self.hidden_states, 0, Self.MAX_NODES * Self.LATENT_DIM)
        self.min_max = MinMaxStats()
        self.gamma = gamma
        self.c_visit = c_visit
        self.c_scale = c_scale

    def __init__(out self, *, deinit take: Self):
        self.nodes = take.nodes^
        self.hidden_states = take.hidden_states
        self.min_max = take.min_max
        self.gamma = take.gamma
        self.c_visit = take.c_visit
        self.c_scale = take.c_scale

    def __del__(deinit self):
        self.hidden_states.free()

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def search[
        RepModel: Model,
        DynModel: Model,
        PredModel: Model,
        RepOpt: Optimizer,
        DynOpt: Optimizer,
        PredOpt: Optimizer,
    ](
        mut self,
        root_obs: List[Scalar[dtype]],
        rep_state: NetworkState[RepModel, RepOpt],
        dyn_state: NetworkState[DynModel, DynOpt],
        pred_state: NetworkState[PredModel, PredOpt],
        v_min: Float64,
        v_max: Float64,
        legal_mask: List[Bool] = List[Bool](),
    ) -> InlineArray[Float64, Self.ACTION_DIM]:
        """Run Gumbel search and return the improved policy.

        The returned distribution sums to 1 over legal actions and is the
        Gumbel-MuZero π̂ = softmax(logits + σ(completed_Q)) — the appropriate
        target for both action selection and policy training.

        Args:
            root_obs: Current observation (length = RepModel.IN_DIM, or shorter
                — extra entries are zero-padded).
            rep_state: Representation network state.
            dyn_state: Dynamics network state.
            pred_state: Prediction network state.
            v_min: Minimum value-support bin.
            v_max: Maximum value-support bin.
            legal_mask: Optional mask over root actions. Empty = all legal.

        Returns:
            Improved policy distribution over the full action space.
        """
        # ---- Reset tree --------------------------------------------------
        self.nodes.clear()
        self.min_max = MinMaxStats()

        # ---- Encode root observation ------------------------------------
        comptime B: Int = 1
        comptime REP_IN = RepModel.IN_DIM
        comptime REP_OUT = RepModel.OUT_DIM

        var obs_ptr = alloc[Scalar[dtype]](REP_IN)
        for i in range(REP_IN):
            if i < len(root_obs):
                obs_ptr[i] = root_obs[i]
            else:
                obs_ptr[i] = Scalar[dtype](0.0)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(B, REP_IN), MutAnyOrigin
        ](obs_ptr)

        var h0_ptr = self.hidden_states  # node 0 uses slot 0
        var h0_t = LayoutTensor[
            dtype, Layout.row_major(B, REP_OUT), MutAnyOrigin
        ](h0_ptr)
        Network[RepModel, RepOpt].forward[B](
            obs_t,
            h0_t,
            rep_state.params_view(),
            rep_state.model_state_view(),
        )
        obs_ptr.free()

        # ---- Predict root prior + value ---------------------------------
        comptime PRED_OUT = PredModel.OUT_DIM
        comptime PRED_IN = PredModel.IN_DIM
        var pred_out_ptr = alloc[Scalar[dtype]](PRED_OUT)
        memset(pred_out_ptr, 0, PRED_OUT)
        var pred_out_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_OUT), MutAnyOrigin
        ](pred_out_ptr)
        var h0_view = LayoutTensor[
            dtype, Layout.row_major(B, PRED_IN), MutAnyOrigin
        ](h0_ptr)
        Network[PredModel, PredOpt].forward[B](
            h0_view,
            pred_out_t,
            pred_state.params_view(),
            pred_state.model_state_view(),
        )

        var root_value = self._decode_value(
            pred_out_ptr + Self.ACTION_DIM, v_min, v_max
        )

        # ---- Build root node --------------------------------------------
        var root = GumbelMCTSNode[Self.ACTION_DIM](hidden_idx=0)
        for a in range(Self.ACTION_DIM):
            root.logits[a] = Float64(
                rebind[Scalar[dtype]](pred_out_t[0, a])
            )
        root.value_estimate = root_value

        # Apply legality at root by setting illegal-action logits to a very
        # negative value (Gumbel-Top-k will never pick them).
        if len(legal_mask) == Self.ACTION_DIM:
            for a in range(Self.ACTION_DIM):
                root.legal[a] = legal_mask[a]
                if not legal_mask[a]:
                    root.logits[a] = Float64(-1e9)

        pred_out_ptr.free()
        self.nodes.append(root^)

        # ---- Pick K candidates via Gumbel-Top-k -------------------------
        # K is capped by both NUM_ROOT_CANDIDATES and the number of legal
        # actions, then rounded down to a power of two so Sequential Halving
        # is well-defined.
        var legal_count = 0
        for a in range(Self.ACTION_DIM):
            if self.nodes[0].legal[a]:
                legal_count += 1
        var k_cap = (
            Self.NUM_ROOT_CANDIDATES
            if Self.NUM_ROOT_CANDIDATES < legal_count
            else legal_count
        )
        if k_cap < 1:
            k_cap = 1
        var k_actual = _largest_power_of_two_le(k_cap)

        var candidates = InlineArray[Int, Self.NUM_ROOT_CANDIDATES](
            uninitialized=True
        )
        var gumbels = InlineArray[Float64, Self.NUM_ROOT_CANDIDATES](
            uninitialized=True
        )
        for i in range(Self.NUM_ROOT_CANDIDATES):
            candidates[i] = -1
            gumbels[i] = Float64(0.0)
        self._sample_gumbel_top_k(0, k_actual, candidates, gumbels)

        # ---- Sequential Halving simulation budget allocation ------------
        var num_phases = _ilog2(k_actual)
        if num_phases < 1:
            # K=1 — no halving, still spend the budget on the lone candidate
            num_phases = 1
        var per_phase_budget = Self.NUM_SIMULATIONS // num_phases
        if per_phase_budget < 1:
            per_phase_budget = 1

        # Active set: indices into `candidates[0..k_actual)`. We preserve the
        # entries; an entry is dropped by replacing with -1 once eliminated.
        var active = InlineArray[Int, Self.NUM_ROOT_CANDIDATES](
            uninitialized=True
        )
        for i in range(Self.NUM_ROOT_CANDIDATES):
            active[i] = -1
        for i in range(k_actual):
            active[i] = i  # candidate index
        var active_size = k_actual

        var sims_used = 0
        for phase in range(num_phases):
            var per_action = per_phase_budget // active_size
            if per_action < 1:
                per_action = 1

            for _ in range(per_action):
                for i in range(active_size):
                    if sims_used >= Self.NUM_SIMULATIONS:
                        break
                    var cand_idx = active[i]
                    var root_action = candidates[cand_idx]
                    self._simulate[
                        DynModel, PredModel, DynOpt, PredOpt
                    ](
                        root_action, dyn_state, pred_state, v_min, v_max
                    )
                    sims_used += 1

            # Halve the active set, except in the last phase (already at 1).
            if phase + 1 < num_phases and active_size > 1:
                var keep = active_size // 2
                if keep < 1:
                    keep = 1
                self._halve_active_set(
                    candidates, gumbels, active, active_size, keep
                )
                active_size = keep

        # Use any remaining sims on the last surviving candidate so the
        # bandit's best estimate is sharpened.
        var leftover_target = active[0]
        while sims_used < Self.NUM_SIMULATIONS and leftover_target >= 0:
            self._simulate[
                DynModel, PredModel, DynOpt, PredOpt
            ](
                candidates[leftover_target],
                dyn_state,
                pred_state,
                v_min,
                v_max,
            )
            sims_used += 1

        # ---- Build improved policy target -------------------------------
        var policy = self._improved_policy(0)
        return policy

    # ─────────────────────────────────────────────────────────────────────
    # Sampling + scoring helpers
    # ─────────────────────────────────────────────────────────────────────

    def _sample_gumbel_top_k(
        self,
        node_idx: Int,
        k: Int,
        mut candidates: InlineArray[Int, Self.NUM_ROOT_CANDIDATES],
        mut gumbels: InlineArray[Float64, Self.NUM_ROOT_CANDIDATES],
    ):
        """Gumbel-Top-k sampling without replacement.

        For each legal action a draw g(a) ~ Gumbel(0) = -log(-log(U)) and
        keep the top-k of (logits(a) + g(a)). Stores both the action indices
        (in `candidates`) and the underlying g(a) values (in `gumbels`) since
        Sequential Halving needs g(a) for scoring.
        """
        # All-action Gumbel scores (illegal = -infinity already).
        var node = self.nodes[node_idx]
        var scores = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        var noises = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        for a in range(Self.ACTION_DIM):
            if not node.legal[a]:
                noises[a] = Float64(0.0)
                scores[a] = Float64(-1e18)
                continue
            var u = random_float64(1e-9, 1.0 - 1e-9)
            var g = -log(-log(u))
            noises[a] = g
            scores[a] = node.logits[a] + g

        # Repeated argmax (k is small, ACTION_DIM is small — O(k * ACT) is fine).
        var taken = InlineArray[Bool, Self.ACTION_DIM](uninitialized=True)
        for a in range(Self.ACTION_DIM):
            taken[a] = False

        for slot in range(k):
            var best_a = -1
            var best_s = Float64(-1e18)
            for a in range(Self.ACTION_DIM):
                if taken[a]:
                    continue
                if not node.legal[a]:
                    continue
                if scores[a] > best_s:
                    best_s = scores[a]
                    best_a = a
            if best_a < 0:
                break
            candidates[slot] = best_a
            gumbels[slot] = noises[best_a]
            taken[best_a] = True

    def _halve_active_set(
        self,
        candidates: InlineArray[Int, Self.NUM_ROOT_CANDIDATES],
        gumbels: InlineArray[Float64, Self.NUM_ROOT_CANDIDATES],
        mut active: InlineArray[Int, Self.NUM_ROOT_CANDIDATES],
        active_size: Int,
        keep: Int,
    ):
        """Keep the top-`keep` active indices by Sequential-Halving score
        s(a) = g(a) + logits(a) + σ(completed_Q(a)).
        """
        var root = self.nodes[0]
        var completed_q = self._completed_q(0)
        var sigma = self._sigma_q_array(0, completed_q)

        # Compute scores for active entries.
        var scored = InlineArray[Float64, Self.NUM_ROOT_CANDIDATES](
            uninitialized=True
        )
        var indices = InlineArray[Int, Self.NUM_ROOT_CANDIDATES](
            uninitialized=True
        )
        for i in range(Self.NUM_ROOT_CANDIDATES):
            scored[i] = Float64(-1e18)
            indices[i] = -1
        for i in range(active_size):
            var cand = active[i]
            var act = candidates[cand]
            scored[i] = gumbels[cand] + root.logits[act] + sigma[act]
            indices[i] = cand

        # Selection sort top `keep` (active_size is small).
        for slot in range(keep):
            var best = slot
            for j in range(slot + 1, active_size):
                if scored[j] > scored[best]:
                    best = j
            if best != slot:
                var tmp_s = scored[slot]
                scored[slot] = scored[best]
                scored[best] = tmp_s
                var tmp_i = indices[slot]
                indices[slot] = indices[best]
                indices[best] = tmp_i

        # Write back into `active`.
        for i in range(Self.NUM_ROOT_CANDIDATES):
            active[i] = -1
        for i in range(keep):
            active[i] = indices[i]

    # ─────────────────────────────────────────────────────────────────────
    # Completed-Q + improved policy machinery (Gumbel MuZero, App. D / G)
    # ─────────────────────────────────────────────────────────────────────

    def _v_mix(self, node_idx: Int) -> Float64:
        """Visit-weighted mixture used as the Q estimate for unvisited
        children. From Danihelka et al. App. D:

            v_mix = (1/(1+ΣN(s,a))) · [ V(s) + ΣN(s,a) · w(a) · Q̂(a) ]

        where w(a) is the prior over visited children, renormalized. We
        approximate the original formula with the standard mctx form."""
        var node = self.nodes[node_idx]

        # Prior over visited children, renormalized.
        var visited_logits_max = Float64(-1e18)
        var visited_count = 0
        for a in range(Self.ACTION_DIM):
            if node.visit_count[a] > 0:
                if node.logits[a] > visited_logits_max:
                    visited_logits_max = node.logits[a]
                visited_count += 1
        if visited_count == 0:
            return node.value_estimate

        var sum_w = Float64(0.0)
        var weighted_q = Float64(0.0)
        for a in range(Self.ACTION_DIM):
            if node.visit_count[a] > 0:
                var w = exp(node.logits[a] - visited_logits_max)
                sum_w += w
                weighted_q += w * node.mean_value(a)
        if sum_w <= 0.0:
            return node.value_estimate
        var mean_visited_q = weighted_q / sum_w

        var total = Float64(node.total_visits)
        return (node.value_estimate + total * mean_visited_q) / (1.0 + total)

    def _completed_q(
        self, node_idx: Int
    ) -> InlineArray[Float64, Self.ACTION_DIM]:
        """Build completed-Q array: Q̂(a) for visited children, v_mix for
        unvisited."""
        var node = self.nodes[node_idx]
        var v_mix = self._v_mix(node_idx)
        var q = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        for a in range(Self.ACTION_DIM):
            if node.visit_count[a] > 0:
                q[a] = node.mean_value(a)
            else:
                q[a] = v_mix
        return q

    def _sigma_q_array(
        self,
        node_idx: Int,
        q: InlineArray[Float64, Self.ACTION_DIM],
    ) -> InlineArray[Float64, Self.ACTION_DIM]:
        """σ(Q): paper Eq. 3 — (c_visit + max_b N(s,b)) · c_scale · norm(Q).

        Q is normalized into [0, 1] via the running MinMax over backed-up
        Q-values (shared with the rest of the search)."""
        var node = self.nodes[node_idx]
        var max_visit = 0
        for a in range(Self.ACTION_DIM):
            if node.visit_count[a] > max_visit:
                max_visit = node.visit_count[a]
        var scale = (self.c_visit + Float64(max_visit)) * self.c_scale

        var out = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        for a in range(Self.ACTION_DIM):
            out[a] = scale * self.min_max.normalize(q[a])
        return out

    def _improved_policy(
        self, node_idx: Int
    ) -> InlineArray[Float64, Self.ACTION_DIM]:
        """π̂ = softmax(logits + σ(completed_Q)) over the full action space.

        Illegal-action logits are -1e9 from the legal mask, so they remain
        zero in the output."""
        var node = self.nodes[node_idx]
        var completed_q = self._completed_q(node_idx)
        var sigma = self._sigma_q_array(node_idx, completed_q)

        var z = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        var max_z = Float64(-1e18)
        for a in range(Self.ACTION_DIM):
            if not node.legal[a]:
                z[a] = Float64(-1e18)
            else:
                z[a] = node.logits[a] + sigma[a]
            if z[a] > max_z:
                max_z = z[a]

        var sum_exp = Float64(0.0)
        var probs = InlineArray[Float64, Self.ACTION_DIM](uninitialized=True)
        for a in range(Self.ACTION_DIM):
            var e = exp(z[a] - max_z)
            probs[a] = e
            sum_exp += e
        if sum_exp <= 0.0:
            # Fallback to uniform over legal — should not happen unless every
            # logit is -inf.
            var legal_count = 0
            for a in range(Self.ACTION_DIM):
                if node.legal[a]:
                    legal_count += 1
            if legal_count == 0:
                for a in range(Self.ACTION_DIM):
                    probs[a] = 1.0 / Float64(Self.ACTION_DIM)
            else:
                for a in range(Self.ACTION_DIM):
                    probs[a] = 1.0 / Float64(legal_count) if node.legal[
                        a
                    ] else 0.0
            return probs
        for a in range(Self.ACTION_DIM):
            probs[a] = probs[a] / sum_exp
        return probs

    def _select_non_root_action(self, node_idx: Int) -> Int:
        """Visit-balance rule (Danihelka 2021, App. B).

        a* = argmax_a [ π_improved(a) − N(s,a) / (1 + ΣN(s,b)) ]

        Drives empirical visit counts toward the improved policy. No Gumbel
        noise here — those samples are root-only."""
        var probs = self._improved_policy(node_idx)
        var node = self.nodes[node_idx]
        var denom = 1.0 + Float64(node.total_visits)

        var best_a = -1
        var best_s = Float64(-1e18)
        for a in range(Self.ACTION_DIM):
            if not node.legal[a]:
                continue
            var s = probs[a] - Float64(node.visit_count[a]) / denom
            if s > best_s:
                best_s = s
                best_a = a
        if best_a < 0:
            best_a = 0
        return best_a

    # ─────────────────────────────────────────────────────────────────────
    # Simulation: select → expand → backup
    # ─────────────────────────────────────────────────────────────────────

    def _simulate[
        DynModel: Model,
        PredModel: Model,
        DynOpt: Optimizer,
        PredOpt: Optimizer,
    ](
        mut self,
        root_action: Int,
        dyn_state: NetworkState[DynModel, DynOpt],
        pred_state: NetworkState[PredModel, PredOpt],
        v_min: Float64,
        v_max: Float64,
    ):
        """One simulation: take `root_action` from the root, traverse the
        non-root subtree by the visit-balance rule until an unexpanded leaf,
        expand it via the dynamics + prediction networks, then back the value
        up through the path."""
        var search_path = List[Int](capacity=64)
        var actions_path = List[Int](capacity=64)

        var node_idx = 0
        search_path.append(node_idx)
        actions_path.append(root_action)

        # Traverse non-root nodes by the visit-balance rule until an
        # unexpanded action is hit.
        while True:
            var act = actions_path[len(actions_path) - 1]
            if not self.nodes[node_idx].is_expanded(act):
                break
            node_idx = self.nodes[node_idx].child_idx[act]
            search_path.append(node_idx)
            var next_act = self._select_non_root_action(node_idx)
            actions_path.append(next_act)

        var parent_idx = search_path[len(search_path) - 1]
        var leaf_action = actions_path[len(actions_path) - 1]

        if len(self.nodes) >= Self.MAX_NODES:
            # Tree budget exhausted — back up the parent's prediction so the
            # current path's stats stay consistent.
            self._backup(search_path, actions_path, Float64(0.0))
            return

        var child_hidden_idx = len(self.nodes)
        var leaf_value = self._expand_node[
            DynModel, PredModel, DynOpt, PredOpt
        ](
            parent_idx,
            leaf_action,
            child_hidden_idx,
            dyn_state,
            pred_state,
            v_min,
            v_max,
        )

        self._backup(search_path, actions_path, leaf_value)

    def _expand_node[
        DynModel: Model,
        PredModel: Model,
        DynOpt: Optimizer,
        PredOpt: Optimizer,
    ](
        mut self,
        parent_idx: Int,
        action: Int,
        child_hidden_idx: Int,
        dyn_state: NetworkState[DynModel, DynOpt],
        pred_state: NetworkState[PredModel, PredOpt],
        v_min: Float64,
        v_max: Float64,
    ) -> Float64:
        """Run dynamics + prediction at (parent.hidden, action), append the
        new child to the tree, and return the predicted scalar value at the
        new node."""
        comptime B: Int = 1
        comptime DYN_IN = DynModel.IN_DIM
        comptime DYN_OUT = DynModel.OUT_DIM

        var dyn_input_ptr = alloc[Scalar[dtype]](DYN_IN)
        memset(dyn_input_ptr, 0, DYN_IN)

        var parent_h_offset = (
            self.nodes[parent_idx].hidden_state_idx * Self.LATENT_DIM
        )
        for i in range(Self.LATENT_DIM):
            dyn_input_ptr[i] = (self.hidden_states + parent_h_offset + i)[]
        dyn_input_ptr[Self.LATENT_DIM + action] = Scalar[dtype](1.0)

        var dyn_input_t = LayoutTensor[
            dtype, Layout.row_major(B, DYN_IN), MutAnyOrigin
        ](dyn_input_ptr)
        var dyn_output_ptr = alloc[Scalar[dtype]](DYN_OUT)
        memset(dyn_output_ptr, 0, DYN_OUT)
        var dyn_output_t = LayoutTensor[
            dtype, Layout.row_major(B, DYN_OUT), MutAnyOrigin
        ](dyn_output_ptr)

        Network[DynModel, DynOpt].forward[B](
            dyn_input_t,
            dyn_output_t,
            dyn_state.params_view(),
            dyn_state.model_state_view(),
        )

        var child_h_offset = child_hidden_idx * Self.LATENT_DIM
        for i in range(Self.LATENT_DIM):
            (self.hidden_states + child_h_offset + i)[] = dyn_output_ptr[i]
        var reward = self._decode_value(
            dyn_output_ptr + Self.LATENT_DIM, v_min, v_max
        )

        dyn_input_ptr.free()
        dyn_output_ptr.free()

        # Predict child policy + value.
        comptime PRED_OUT = PredModel.OUT_DIM
        comptime PRED_IN = PredModel.IN_DIM
        var pred_out_ptr = alloc[Scalar[dtype]](PRED_OUT)
        memset(pred_out_ptr, 0, PRED_OUT)
        var pred_out_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_OUT), MutAnyOrigin
        ](pred_out_ptr)
        var child_h_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_IN), MutAnyOrigin
        ](self.hidden_states + child_h_offset)
        Network[PredModel, PredOpt].forward[B](
            child_h_t,
            pred_out_t,
            pred_state.params_view(),
            pred_state.model_state_view(),
        )

        var child = GumbelMCTSNode[Self.ACTION_DIM](
            hidden_idx=child_hidden_idx
        )
        for a in range(Self.ACTION_DIM):
            child.logits[a] = Float64(
                rebind[Scalar[dtype]](pred_out_t[0, a])
            )
        var leaf_value = self._decode_value(
            pred_out_ptr + Self.ACTION_DIM, v_min, v_max
        )
        child.value_estimate = leaf_value

        pred_out_ptr.free()

        self.nodes[parent_idx].reward[action] = reward
        self.nodes[parent_idx].child_idx[action] = child_hidden_idx
        self.nodes.append(child^)

        return leaf_value

    def _backup(
        mut self,
        search_path: List[Int],
        actions_path: List[Int],
        leaf_value: Float64,
    ):
        """Standard MuZero backup along the path with discount γ."""
        var value = leaf_value
        var path_len = len(search_path)
        for i in range(path_len):
            var idx = path_len - 1 - i
            var node_idx = search_path[idx]
            var action = actions_path[idx]

            value = self.nodes[node_idx].reward[action] + self.gamma * value

            self.nodes[node_idx].visit_count[action] += 1
            self.nodes[node_idx].total_value[action] += value
            self.nodes[node_idx].total_visits += 1

            self.min_max.update(self.nodes[node_idx].mean_value(action))

    # ─────────────────────────────────────────────────────────────────────
    # Categorical-value decode (matches MuZero's _decode_value)
    # ─────────────────────────────────────────────────────────────────────

    def _decode_value(
        self,
        logits_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        v_min: Float64,
        v_max: Float64,
    ) -> Float64:
        var step = (v_max - v_min) / Float64(
            Self.NUM_BINS - 1
        ) if Self.NUM_BINS > 1 else Float64(0.0)

        var max_val = Float64(logits_ptr[0])
        for i in range(1, Self.NUM_BINS):
            var v = Float64(logits_ptr[i])
            if v > max_val:
                max_val = v

        var sum_exp = Float64(0.0)
        for i in range(Self.NUM_BINS):
            sum_exp += exp(Float64(logits_ptr[i]) - max_val)

        var result = Float64(0.0)
        for i in range(Self.NUM_BINS):
            var prob = exp(Float64(logits_ptr[i]) - max_val) / sum_exp
            result += prob * (v_min + Float64(i) * step)

        return inverse_scalar_transform(result)


# ═════════════════════════════════════════════════════════════════════════
# File-local helpers
# ═════════════════════════════════════════════════════════════════════════


def _ilog2(n: Int) -> Int:
    """Integer floor(log2(n)) for n ≥ 1."""
    var x = n
    var r = 0
    while x > 1:
        x = x // 2
        r += 1
    return r


def _largest_power_of_two_le(n: Int) -> Int:
    """Largest power of two ≤ n, for n ≥ 1."""
    var x = 1
    while x * 2 <= n:
        x *= 2
    return x
