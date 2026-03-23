"""MuZero MCTS — Monte Carlo Tree Search with learned model.

Instead of using a true game simulator, MuZero's MCTS expands nodes using
the learned dynamics and prediction networks. Each simulation:
  1. Selection: Traverse from root to leaf using PUCT (with MinMax Q-normalization)
  2. Expansion: Use dynamics network g(s, a) -> (r, s') and prediction f(s') -> (p, v)
  3. Backup: Propagate discounted returns back up the search path

After N simulations, the visit count distribution at the root is used as the
action selection policy.

Reference: Schrittwieser et al., 2020 (Nature)
"""

from std.math import sqrt, log, exp
from std.memory import alloc, memset
from std.random import random_float64
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network, NetworkState
from .utils import MinMaxStats, inverse_scalar_transform


# ═══════════════════════════════════════════════════════════════════════════
# MCTS Node
# ═══════════════════════════════════════════════════════════════════════════


struct MCTSNode[ACTION_DIM: Int](ImplicitlyCopyable, Movable):
    """A single node in the MCTS search tree.

    Stores per-action statistics and links to child hidden states via
    indices into a shared hidden state pool.

    Parameters:
        ACTION_DIM: Number of discrete actions available.
    """

    # Per-action statistics
    var visit_count: InlineArray[Int, Self.ACTION_DIM]  # N(s, a)
    var total_value: InlineArray[
        Float64, Self.ACTION_DIM
    ]  # W(s, a) = sum of values
    var prior: InlineArray[
        Float64, Self.ACTION_DIM
    ]  # P(s, a) from prediction net
    var reward: InlineArray[
        Float64, Self.ACTION_DIM
    ]  # R(s, a) from dynamics net
    var child_idx: InlineArray[
        Int, Self.ACTION_DIM
    ]  # Index of child node (-1 = unexpanded)

    # Node-level
    var total_visits: Int  # N(s) = sum of N(s, a)
    var hidden_state_idx: Int  # Index into hidden state pool

    def __init__(out self, hidden_idx: Int):
        """Initialize node with zero statistics.

        Args:
            hidden_idx: Index of this node's hidden state in the pool.
        """
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
        """Mean Q-value for an action: Q(s, a) = W(s, a) / N(s, a).

        Args:
            action: Action index.

        Returns:
            Mean value, or 0.0 if the action has never been visited.
        """
        if self.visit_count[action] > 0:
            return self.total_value[action] / Float64(self.visit_count[action])
        return Float64(0.0)

    def is_expanded(self, action: Int) -> Bool:
        """Check whether a child node exists for the given action.

        Args:
            action: Action index.

        Returns:
            True if the action has been expanded.
        """
        return self.child_idx[action] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# MCTS Search
# ═══════════════════════════════════════════════════════════════════════════


struct MCTS[
    ACTION_DIM: Int,
    LATENT_DIM: Int,
    NUM_BINS: Int = 101,
    NUM_SIMULATIONS: Int = 50,
    MAX_NODES: Int = 512,
](Movable):
    """Monte Carlo Tree Search with learned dynamics and prediction models.

    Maintains a tree of MCTSNode instances and a pool of hidden states.
    Each simulation traverses/expands the tree using learned networks.

    Parameters:
        ACTION_DIM: Number of discrete actions.
        LATENT_DIM: Hidden state dimension.
        NUM_BINS: Categorical distribution bins for value/reward (default: 101).
        NUM_SIMULATIONS: Number of MCTS simulations per search (default: 50).
        MAX_NODES: Maximum nodes in the tree (default: 512).
    """

    # Node storage
    var nodes: List[MCTSNode[Self.ACTION_DIM]]

    # Hidden state pool [MAX_NODES * LATENT_DIM]
    var hidden_states: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    # MinMax Q-value normalization
    var min_max: MinMaxStats

    # Search hyperparameters
    var c_base: Float64
    var c_init: Float64
    var gamma: Float64
    var dirichlet_alpha: Float64
    var noise_fraction: Float64

    def __init__(
        out self,
        gamma: Float64 = 0.997,
        c_base: Float64 = 19652.0,
        c_init: Float64 = 1.25,
        dirichlet_alpha: Float64 = 0.25,
        noise_fraction: Float64 = 0.25,
    ):
        """Initialize MCTS search engine.

        Args:
            gamma: Discount factor for backup (default: 0.997).
            c_base: PUCT base constant (default: 19652).
            c_init: PUCT initial exploration constant (default: 1.25).
            dirichlet_alpha: Dirichlet noise alpha for root exploration.
            noise_fraction: Fraction of Dirichlet noise at root.
        """
        self.nodes = List[MCTSNode[Self.ACTION_DIM]](capacity=Self.MAX_NODES)
        self.hidden_states = alloc[Scalar[dtype]](
            Self.MAX_NODES * Self.LATENT_DIM
        )
        memset(self.hidden_states, 0, Self.MAX_NODES * Self.LATENT_DIM)
        self.min_max = MinMaxStats()
        self.c_base = c_base
        self.c_init = c_init
        self.gamma = gamma
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction

    def __init__(out self, *, deinit take: Self):
        self.nodes = take.nodes^
        self.hidden_states = take.hidden_states
        self.min_max = take.min_max
        self.c_base = take.c_base
        self.c_init = take.c_init
        self.gamma = take.gamma
        self.dirichlet_alpha = take.dirichlet_alpha
        self.noise_fraction = take.noise_fraction

    def __del__(deinit self):
        self.hidden_states.free()

    # ══════════════════════════════════════════════════════════════════════
    # Core MCTS Search
    # ══════════════════════════════════════════════════════════════════════

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
        add_noise: Bool = True,
        legal_mask: List[Bool] = List[Bool](),
    ) -> InlineArray[Float64, Self.ACTION_DIM]:
        """Run MCTS from an observation, return visit count policy.

        1. Encode observation with representation network -> root hidden state
        2. Run prediction network to get root prior and value
        3. Apply legal action mask to root prior (if provided)
        4. Add Dirichlet noise to root prior (if training)
        5. Run NUM_SIMULATIONS simulations (select, expand, backup)
        6. Return normalized visit counts as policy

        Args:
            root_obs: Current observation [obs_dim].
            rep_state: Representation network state.
            dyn_state: Dynamics network state.
            pred_state: Prediction network state.
            v_min: Minimum value support.
            v_max: Maximum value support.
            add_noise: Whether to add Dirichlet noise at root.
            legal_mask: Optional legal action mask (length ACTION_DIM).
                       If empty, all actions are legal (backward compatible).

        Returns:
            Visit count distribution over actions (sums to 1).
        """
        # Reset tree
        self.nodes.clear()
        self.min_max = MinMaxStats()

        # ── Step 1: Encode observation -> root hidden state ──────────
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

        # Output: root hidden state
        var h0_ptr = self.hidden_states  # Node 0 uses slot 0
        var h0_t = LayoutTensor[
            dtype, Layout.row_major(B, REP_OUT), MutAnyOrigin
        ](h0_ptr)

        Network[RepModel, RepOpt].forward[B](
            obs_t, h0_t, rep_state.params_view()
        )

        # Scale hidden state to [0, 1]
        self._scale_hidden_state(0)

        obs_ptr.free()

        # ── Step 2: Predict root prior and value ─────────────────────
        comptime PRED_OUT = PredModel.OUT_DIM

        var pred_out_ptr = alloc[Scalar[dtype]](PRED_OUT)
        memset(pred_out_ptr, 0, PRED_OUT)
        var pred_out_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_OUT), MutAnyOrigin
        ](pred_out_ptr)

        comptime PRED_IN = PredModel.IN_DIM
        var h0_view = LayoutTensor[
            dtype, Layout.row_major(B, PRED_IN), MutAnyOrigin
        ](h0_ptr)
        Network[PredModel, PredOpt].forward[B](
            h0_view, pred_out_t, pred_state.params_view()
        )

        # Parse prediction: [policy_logits(ACTION_DIM) | value_logits(NUM_BINS)]
        var root_value = self._decode_value(
            pred_out_ptr + Self.ACTION_DIM, v_min, v_max
        )

        # Create root node
        var root = MCTSNode[Self.ACTION_DIM](hidden_idx=0)

        # Set prior from softmax of policy logits
        var policy_logits_ptr = alloc[Float64](Self.ACTION_DIM)
        for a in range(Self.ACTION_DIM):
            policy_logits_ptr[a] = Float64(
                rebind[Scalar[dtype]](pred_out_t[0, a])
            )
        # Softmax
        var max_logit = policy_logits_ptr[0]
        for a in range(1, Self.ACTION_DIM):
            if policy_logits_ptr[a] > max_logit:
                max_logit = policy_logits_ptr[a]
        var sum_exp = Float64(0.0)
        for a in range(Self.ACTION_DIM):
            policy_logits_ptr[a] = exp(policy_logits_ptr[a] - max_logit)
            sum_exp += policy_logits_ptr[a]
        for a in range(Self.ACTION_DIM):
            root.prior[a] = policy_logits_ptr[a] / sum_exp

        pred_out_ptr.free()

        # ── Step 2b: Apply legal action mask to root prior ─────────
        if len(legal_mask) == Self.ACTION_DIM:
            var mask_sum = Float64(0.0)
            for a in range(Self.ACTION_DIM):
                if not legal_mask[a]:
                    root.prior[a] = Float64(0.0)
                else:
                    mask_sum += root.prior[a]
            # Renormalize
            if mask_sum > 0.0:
                for a in range(Self.ACTION_DIM):
                    root.prior[a] /= mask_sum
            else:
                # All legal actions have 0 prior — uniform over legal
                var legal_count = Float64(0.0)
                for a in range(Self.ACTION_DIM):
                    if legal_mask[a]:
                        legal_count += 1.0
                if legal_count > 0.0:
                    for a in range(Self.ACTION_DIM):
                        if legal_mask[a]:
                            root.prior[a] = 1.0 / legal_count

        # ── Step 3: Add Dirichlet noise to root ─────────────────────
        if add_noise:
            # Approximate Dirichlet by sampling Gamma(alpha, 1) then normalizing
            var noise = InlineArray[Float64, Self.ACTION_DIM](
                uninitialized=True
            )
            var noise_sum = Float64(0.0)
            for a in range(Self.ACTION_DIM):
                # Gamma approximation: for small alpha, use -log(U) * alpha
                var u = random_float64(0.0001, 0.9999)
                var g = -log(u)  # Exponential(1), rough Gamma for alpha~0.25
                noise[a] = g
                noise_sum += g
            for a in range(Self.ACTION_DIM):
                noise[a] /= noise_sum
                root.prior[a] = (1.0 - self.noise_fraction) * root.prior[
                    a
                ] + self.noise_fraction * noise[a]

        self.nodes.append(root^)

        policy_logits_ptr.free()

        # ── Step 4: Run simulations ──────────────────────────────────
        for _ in range(Self.NUM_SIMULATIONS):
            self._run_simulation(dyn_state, pred_state, v_min, v_max)

        # ── Step 5: Return visit count policy ────────────────────────
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

    # ══════════════════════════════════════════════════════════════════════
    # Batched MCTS Search (Virtual Loss)
    # ══════════════════════════════════════════════════════════════════════

    def search_batched[
        RepModel: Model,
        DynModel: Model,
        PredModel: Model,
        RepOpt: Optimizer,
        DynOpt: Optimizer,
        PredOpt: Optimizer,
        BATCH_SIMS: Int = 8,
    ](
        mut self,
        root_obs: List[Scalar[dtype]],
        rep_state: NetworkState[RepModel, RepOpt],
        dyn_state: NetworkState[DynModel, DynOpt],
        pred_state: NetworkState[PredModel, PredOpt],
        v_min: Float64,
        v_max: Float64,
        add_noise: Bool = True,
    ) -> InlineArray[Float64, Self.ACTION_DIM]:
        """Run batched MCTS — select BATCH_SIMS leaves per round, expand in one forward call.

        Uses virtual losses during selection to encourage diverse leaf selection.
        Network forward calls are batched: instead of NUM_SIMULATIONS x BATCH=1,
        we do ceil(NUM_SIMULATIONS / BATCH_SIMS) x BATCH=BATCH_SIMS calls.

        This provides ~BATCH_SIMS x speedup on network evaluation, which is the
        dominant cost in MCTS with learned models.

        Args:
            root_obs: Current observation [obs_dim].
            rep_state: Representation network state.
            dyn_state: Dynamics network state.
            pred_state: Prediction network state.
            v_min: Minimum value support.
            v_max: Maximum value support.
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Visit count distribution over actions (sums to 1).
        """
        # ── Initialize tree (same as unbatched) ──────────────────────
        self.nodes.clear()
        self.min_max = MinMaxStats()

        # Encode root observation
        comptime B1: Int = 1
        comptime REP_IN = RepModel.IN_DIM
        comptime REP_OUT = RepModel.OUT_DIM

        var obs_ptr = alloc[Scalar[dtype]](REP_IN)
        for i in range(REP_IN):
            if i < len(root_obs):
                obs_ptr[i] = root_obs[i]
            else:
                obs_ptr[i] = Scalar[dtype](0.0)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(B1, REP_IN), MutAnyOrigin
        ](obs_ptr)

        var h0_ptr = self.hidden_states
        var h0_t = LayoutTensor[
            dtype, Layout.row_major(B1, REP_OUT), MutAnyOrigin
        ](h0_ptr)
        Network[RepModel, RepOpt].forward[B1](
            obs_t, h0_t, rep_state.params_view()
        )
        self._scale_hidden_state(0)
        obs_ptr.free()

        # Predict root prior and value
        comptime PRED_OUT = PredModel.OUT_DIM
        comptime PRED_IN_R = PredModel.IN_DIM
        var pred_out_ptr = alloc[Scalar[dtype]](PRED_OUT)
        memset(pred_out_ptr, 0, PRED_OUT)
        var pred_out_t = LayoutTensor[
            dtype, Layout.row_major(B1, PRED_OUT), MutAnyOrigin
        ](pred_out_ptr)
        var h0_view = LayoutTensor[
            dtype, Layout.row_major(B1, PRED_IN_R), MutAnyOrigin
        ](h0_ptr)
        Network[PredModel, PredOpt].forward[B1](
            h0_view, pred_out_t, pred_state.params_view()
        )

        # Create root node with softmax prior
        var root = MCTSNode[Self.ACTION_DIM](hidden_idx=0)
        var policy_logits_ptr = alloc[Float64](Self.ACTION_DIM)
        for a in range(Self.ACTION_DIM):
            policy_logits_ptr[a] = Float64(
                rebind[Scalar[dtype]](pred_out_t[0, a])
            )
        var max_logit = policy_logits_ptr[0]
        for a in range(1, Self.ACTION_DIM):
            if policy_logits_ptr[a] > max_logit:
                max_logit = policy_logits_ptr[a]
        var sum_exp = Float64(0.0)
        for a in range(Self.ACTION_DIM):
            policy_logits_ptr[a] = exp(policy_logits_ptr[a] - max_logit)
            sum_exp += policy_logits_ptr[a]
        for a in range(Self.ACTION_DIM):
            root.prior[a] = policy_logits_ptr[a] / sum_exp

        pred_out_ptr.free()

        # Dirichlet noise
        if add_noise:
            var noise = InlineArray[Float64, Self.ACTION_DIM](
                uninitialized=True
            )
            var noise_sum = Float64(0.0)
            for a in range(Self.ACTION_DIM):
                var u = random_float64(0.0001, 0.9999)
                noise[a] = -log(u)
                noise_sum += noise[a]
            for a in range(Self.ACTION_DIM):
                noise[a] /= noise_sum
                root.prior[a] = (1.0 - self.noise_fraction) * root.prior[
                    a
                ] + self.noise_fraction * noise[a]

        self.nodes.append(root^)
        policy_logits_ptr.free()

        # ── Batched simulations ──────────────────────────────────────
        comptime DYN_IN = DynModel.IN_DIM
        comptime DYN_OUT = DynModel.OUT_DIM
        comptime VIRTUAL_LOSS: Int = 3  # Virtual loss value

        # Pre-allocate batched forward buffers
        var batch_dyn_input = alloc[Scalar[dtype]](BATCH_SIMS * DYN_IN)
        var batch_dyn_output = alloc[Scalar[dtype]](BATCH_SIMS * DYN_OUT)
        var batch_pred_input = alloc[Scalar[dtype]](
            BATCH_SIMS * PredModel.IN_DIM
        )
        var batch_pred_output = alloc[Scalar[dtype]](BATCH_SIMS * PRED_OUT)

        # Per-simulation tracking
        var sim_parent_idx = alloc[Int](BATCH_SIMS)
        var sim_action = alloc[Int](BATCH_SIMS)
        var sim_search_paths = List[List[Int]](capacity=BATCH_SIMS)
        var sim_action_paths = List[List[Int]](capacity=BATCH_SIMS)
        var sim_valid = alloc[Bool](BATCH_SIMS)

        var sims_done = 0
        while sims_done < Self.NUM_SIMULATIONS:
            # How many to batch this round
            var batch_count = Self.NUM_SIMULATIONS - sims_done
            if batch_count > BATCH_SIMS:
                batch_count = BATCH_SIMS

            # ── Phase 1: Select leaves with virtual losses ───────────
            sim_search_paths.clear()
            sim_action_paths.clear()
            var pending_expansions = 0

            for s in range(batch_count):
                var search_path = List[Int](capacity=64)
                var actions_path = List[Int](capacity=64)
                var node_idx = 0
                search_path.append(node_idx)

                while True:
                    var action = self._select_action(node_idx)
                    actions_path.append(action)

                    if not self.nodes[node_idx].is_expanded(action):
                        break

                    node_idx = self.nodes[node_idx].child_idx[action]
                    search_path.append(node_idx)

                var parent_idx = search_path[len(search_path) - 1]
                var leaf_action = actions_path[len(actions_path) - 1]

                # Check tree overflow
                if len(self.nodes) + pending_expansions >= Self.MAX_NODES:
                    sim_valid[s] = False
                    sim_search_paths.append(search_path^)
                    sim_action_paths.append(actions_path^)
                    continue

                sim_parent_idx[s] = parent_idx
                sim_action[s] = leaf_action
                sim_valid[s] = True

                # Apply virtual loss to discourage same path
                self.nodes[parent_idx].visit_count[leaf_action] += VIRTUAL_LOSS
                self.nodes[parent_idx].total_visits += VIRTUAL_LOSS

                # Build dynamics input for this leaf
                var parent_h_offset = (
                    self.nodes[parent_idx].hidden_state_idx * Self.LATENT_DIM
                )
                for i in range(Self.LATENT_DIM):
                    batch_dyn_input[pending_expansions * DYN_IN + i] = (
                        self.hidden_states + parent_h_offset + i
                    )[]
                # Zero the action part, then set one-hot
                for a in range(Self.ACTION_DIM):
                    batch_dyn_input[
                        pending_expansions * DYN_IN + Self.LATENT_DIM + a
                    ] = Scalar[dtype](0.0)
                batch_dyn_input[
                    pending_expansions * DYN_IN + Self.LATENT_DIM + leaf_action
                ] = Scalar[dtype](1.0)

                pending_expansions += 1
                sim_search_paths.append(search_path^)
                sim_action_paths.append(actions_path^)

            if pending_expansions == 0:
                sims_done += batch_count
                continue

            # ── Phase 2: Batched dynamics forward ────────────────────
            # Pad remaining slots with zeros if pending < BATCH_SIMS
            for i in range(pending_expansions * DYN_IN, BATCH_SIMS * DYN_IN):
                batch_dyn_input[i] = Scalar[dtype](0.0)
            memset(batch_dyn_output, 0, BATCH_SIMS * DYN_OUT)

            var dyn_in_t = LayoutTensor[
                dtype, Layout.row_major(BATCH_SIMS, DYN_IN), MutAnyOrigin
            ](batch_dyn_input)
            var dyn_out_t = LayoutTensor[
                dtype, Layout.row_major(BATCH_SIMS, DYN_OUT), MutAnyOrigin
            ](batch_dyn_output)
            Network[DynModel, DynOpt].forward[BATCH_SIMS](
                dyn_in_t, dyn_out_t, dyn_state.params_view()
            )

            # ── Phase 3: Extract hidden states, batched prediction ───
            # Copy dynamics outputs to hidden state pool + build pred input
            var expansion_idx = 0
            for s in range(batch_count):
                if not sim_valid[s]:
                    continue

                var child_hidden_idx = len(self.nodes) + expansion_idx
                if child_hidden_idx >= Self.MAX_NODES:
                    sim_valid[s] = False
                    expansion_idx += 1
                    continue

                # Extract hidden state
                var child_h_offset = child_hidden_idx * Self.LATENT_DIM
                for i in range(Self.LATENT_DIM):
                    (
                        self.hidden_states + child_h_offset + i
                    )[] = batch_dyn_output[expansion_idx * DYN_OUT + i]
                self._scale_hidden_state(child_hidden_idx)

                # Copy to prediction input
                comptime PRED_IN = PredModel.IN_DIM
                for i in range(PRED_IN):
                    batch_pred_input[expansion_idx * PRED_IN + i] = (
                        self.hidden_states + child_h_offset + i
                    )[]

                expansion_idx += 1

            # Pad prediction input
            comptime PRED_IN_B = PredModel.IN_DIM
            for i in range(expansion_idx * PRED_IN_B, BATCH_SIMS * PRED_IN_B):
                batch_pred_input[i] = Scalar[dtype](0.0)
            memset(batch_pred_output, 0, BATCH_SIMS * PRED_OUT)

            var pred_in_t = LayoutTensor[
                dtype, Layout.row_major(BATCH_SIMS, PRED_IN_B), MutAnyOrigin
            ](batch_pred_input)
            var pred_out_t_b = LayoutTensor[
                dtype, Layout.row_major(BATCH_SIMS, PRED_OUT), MutAnyOrigin
            ](batch_pred_output)
            Network[PredModel, PredOpt].forward[BATCH_SIMS](
                pred_in_t, pred_out_t_b, pred_state.params_view()
            )

            # ── Phase 4: Create nodes + backup ───────────────────────
            expansion_idx = 0
            for s in range(batch_count):
                if not sim_valid[s]:
                    # Remove virtual loss for invalid simulations
                    if sim_valid[s] == False and s < len(sim_search_paths):
                        pass  # Virtual loss wasn't applied for invalid ones
                    continue

                var child_hidden_idx = len(self.nodes)
                var parent = sim_parent_idx[s]
                var act = sim_action[s]

                # Extract reward from dynamics output
                var reward = self._decode_value(
                    batch_dyn_output
                    + expansion_idx * DYN_OUT
                    + Self.LATENT_DIM,
                    v_min,
                    v_max,
                )

                # Create child node with softmax prior from prediction
                var child = MCTSNode[Self.ACTION_DIM](
                    hidden_idx=child_hidden_idx
                )
                var max_l = Float64(batch_pred_output[expansion_idx * PRED_OUT])
                for a in range(1, Self.ACTION_DIM):
                    var v = Float64(
                        batch_pred_output[expansion_idx * PRED_OUT + a]
                    )
                    if v > max_l:
                        max_l = v
                var se = Float64(0.0)
                for a in range(Self.ACTION_DIM):
                    var v = Float64(
                        batch_pred_output[expansion_idx * PRED_OUT + a]
                    )
                    child.prior[a] = exp(v - max_l)
                    se += child.prior[a]
                for a in range(Self.ACTION_DIM):
                    child.prior[a] /= se

                # Get leaf value
                var leaf_value = self._decode_value(
                    batch_pred_output
                    + expansion_idx * PRED_OUT
                    + Self.ACTION_DIM,
                    v_min,
                    v_max,
                )

                # Link parent -> child
                self.nodes[parent].reward[act] = reward
                self.nodes[parent].child_idx[act] = child_hidden_idx
                self.nodes.append(child^)

                # Remove virtual loss before backup
                self.nodes[parent].visit_count[act] -= VIRTUAL_LOSS
                self.nodes[parent].total_visits -= VIRTUAL_LOSS

                # Backup
                self._backup(
                    sim_search_paths[s], sim_action_paths[s], leaf_value
                )

                expansion_idx += 1

            sims_done += batch_count

        # Free batched buffers
        batch_dyn_input.free()
        batch_dyn_output.free()
        batch_pred_input.free()
        batch_pred_output.free()
        sim_parent_idx.free()
        sim_action.free()
        sim_valid.free()

        # Return visit count policy
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

    # ══════════════════════════════════════════════════════════════════════
    # Internal Methods
    # ══════════════════════════════════════════════════════════════════════

    def _run_simulation[
        DynModel: Model,
        PredModel: Model,
        DynOpt: Optimizer,
        PredOpt: Optimizer,
    ](
        mut self,
        dyn_state: NetworkState[DynModel, DynOpt],
        pred_state: NetworkState[PredModel, PredOpt],
        v_min: Float64,
        v_max: Float64,
    ):
        """Run one MCTS simulation: select -> expand -> backup.

        Args:
            dyn_state: Dynamics network state.
            pred_state: Prediction network state.
            v_min: Minimum value support.
            v_max: Maximum value support.
        """
        # ── Selection: traverse from root to leaf ────────────────────
        var search_path = List[Int](capacity=64)  # Node indices
        var actions_path = List[Int](capacity=64)  # Actions taken
        var node_idx = 0
        search_path.append(node_idx)

        while True:
            var action = self._select_action(node_idx)
            actions_path.append(action)

            if not self.nodes[node_idx].is_expanded(action):
                break

            node_idx = self.nodes[node_idx].child_idx[action]
            search_path.append(node_idx)

        # ── Expansion: use dynamics + prediction ─────────────────────
        var parent_idx = search_path[len(search_path) - 1]
        var action = actions_path[len(actions_path) - 1]

        # Guard against tree overflow
        if len(self.nodes) >= Self.MAX_NODES:
            return

        var child_hidden_idx = len(self.nodes)

        # Dynamics: g(parent_hidden, action) -> (reward, child_hidden)
        self._expand_node[DynModel, PredModel, DynOpt, PredOpt](
            parent_idx,
            action,
            child_hidden_idx,
            dyn_state,
            pred_state,
            v_min,
            v_max,
        )

        # Get leaf value from prediction
        var leaf_value = self._get_leaf_value(
            child_hidden_idx, pred_state, v_min, v_max
        )

        # ── Backup: propagate value up the search path ───────────────
        self._backup(search_path, actions_path, leaf_value)

    def _select_action(self, node_idx: Int) -> Int:
        """Select action using PUCT formula.

        a* = argmax_a [ Q(s,a) + c(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]

        Q-values are normalized to [0, 1] via MinMax.

        Args:
            node_idx: Index of current node.

        Returns:
            Selected action index.
        """
        var node = self.nodes[node_idx]
        var sqrt_total = sqrt(Float64(node.total_visits))

        # Exploration rate c(s)
        var c = (
            log((1.0 + Float64(node.total_visits) + self.c_base) / self.c_base)
            + self.c_init
        )

        var best_action = 0
        var best_score = Float64(-1e18)

        for a in range(Self.ACTION_DIM):
            var q_value: Float64
            if node.visit_count[a] > 0:
                q_value = self.min_max.normalize(node.mean_value(a))
            else:
                q_value = Float64(0.0)

            var prior_score = (
                c
                * node.prior[a]
                * sqrt_total
                / (1.0 + Float64(node.visit_count[a]))
            )
            var score = q_value + prior_score

            if score > best_score:
                best_score = score
                best_action = a

        return best_action

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
    ):
        """Expand a node by applying dynamics and prediction networks.

        Args:
            parent_idx: Parent node index.
            action: Action taken.
            child_hidden_idx: Index for the new child node.
            dyn_state: Dynamics network state.
            pred_state: Prediction network state.
            v_min: Minimum value support.
            v_max: Maximum value support.
        """
        comptime B: Int = 1
        comptime DYN_IN = DynModel.IN_DIM  # LATENT_DIM + ACTION_DIM
        comptime DYN_OUT = DynModel.OUT_DIM  # LATENT_DIM + NUM_BINS (reward)

        # Build dynamics input: [hidden_state || one_hot_action]
        var dyn_input_ptr = alloc[Scalar[dtype]](DYN_IN)
        memset(dyn_input_ptr, 0, DYN_IN)

        # Copy parent hidden state
        var parent_h_offset = (
            self.nodes[parent_idx].hidden_state_idx * Self.LATENT_DIM
        )
        for i in range(Self.LATENT_DIM):
            dyn_input_ptr[i] = (self.hidden_states + parent_h_offset + i)[]

        # One-hot action
        dyn_input_ptr[Self.LATENT_DIM + action] = Scalar[dtype](1.0)

        var dyn_input_t = LayoutTensor[
            dtype, Layout.row_major(B, DYN_IN), MutAnyOrigin
        ](dyn_input_ptr)

        # Dynamics forward
        var dyn_output_ptr = alloc[Scalar[dtype]](DYN_OUT)
        memset(dyn_output_ptr, 0, DYN_OUT)
        var dyn_output_t = LayoutTensor[
            dtype, Layout.row_major(B, DYN_OUT), MutAnyOrigin
        ](dyn_output_ptr)

        Network[DynModel, DynOpt].forward[B](
            dyn_input_t, dyn_output_t, dyn_state.params_view()
        )

        # Extract child hidden state (first LATENT_DIM elements)
        var child_h_offset = child_hidden_idx * Self.LATENT_DIM
        for i in range(Self.LATENT_DIM):
            (self.hidden_states + child_h_offset + i)[] = dyn_output_ptr[i]

        # Scale child hidden state
        self._scale_hidden_state(child_hidden_idx)

        # Extract reward (remaining NUM_BINS elements -> decode)
        var reward = self._decode_value(
            dyn_output_ptr + Self.LATENT_DIM, v_min, v_max
        )

        dyn_input_ptr.free()
        dyn_output_ptr.free()

        # Predict child prior and value
        comptime PRED_OUT = PredModel.OUT_DIM

        var pred_out_ptr = alloc[Scalar[dtype]](PRED_OUT)
        memset(pred_out_ptr, 0, PRED_OUT)
        var pred_out_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_OUT), MutAnyOrigin
        ](pred_out_ptr)

        comptime PRED_IN_2 = PredModel.IN_DIM
        var child_h_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_IN_2), MutAnyOrigin
        ](self.hidden_states + child_h_offset)

        Network[PredModel, PredOpt].forward[B](
            child_h_t, pred_out_t, pred_state.params_view()
        )

        # Create child node
        var child = MCTSNode[Self.ACTION_DIM](hidden_idx=child_hidden_idx)

        # Set prior from softmax of policy logits
        var max_logit = Float64(rebind[Scalar[dtype]](pred_out_t[0, 0]))
        for a in range(1, Self.ACTION_DIM):
            var v = Float64(rebind[Scalar[dtype]](pred_out_t[0, a]))
            if v > max_logit:
                max_logit = v
        var sum_exp = Float64(0.0)
        for a in range(Self.ACTION_DIM):
            var v = Float64(rebind[Scalar[dtype]](pred_out_t[0, a]))
            child.prior[a] = exp(v - max_logit)
            sum_exp += child.prior[a]
        for a in range(Self.ACTION_DIM):
            child.prior[a] /= sum_exp

        pred_out_ptr.free()

        # Link parent -> child
        self.nodes[parent_idx].reward[action] = reward
        self.nodes[parent_idx].child_idx[action] = child_hidden_idx
        self.nodes.append(child^)

    def _get_leaf_value[
        PredModel: Model,
        PredOpt: Optimizer,
    ](
        self,
        hidden_idx: Int,
        pred_state: NetworkState[PredModel, PredOpt],
        v_min: Float64,
        v_max: Float64,
    ) -> Float64:
        """Get value prediction for a leaf node.

        Args:
            hidden_idx: Hidden state pool index for the leaf.
            pred_state: Prediction network state.
            v_min: Minimum value support.
            v_max: Maximum value support.

        Returns:
            Decoded scalar value prediction.
        """
        comptime B: Int = 1
        comptime PRED_OUT = PredModel.OUT_DIM

        var pred_out_ptr = alloc[Scalar[dtype]](PRED_OUT)
        memset(pred_out_ptr, 0, PRED_OUT)
        var pred_out_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_OUT), MutAnyOrigin
        ](pred_out_ptr)

        comptime PRED_IN_3 = PredModel.IN_DIM
        var h_offset = hidden_idx * Self.LATENT_DIM
        var h_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_IN_3), MutAnyOrigin
        ](self.hidden_states + h_offset)

        Network[PredModel, PredOpt].forward[B](
            h_t, pred_out_t, pred_state.params_view()
        )

        var value = self._decode_value(
            pred_out_ptr + Self.ACTION_DIM, v_min, v_max
        )
        pred_out_ptr.free()
        return value

    def _backup(
        mut self,
        search_path: List[Int],
        actions_path: List[Int],
        leaf_value: Float64,
    ):
        """Backup value through the search path from leaf to root.

        For each node on the path, accumulates the discounted return
        and updates visit counts and Q-values.

        Args:
            search_path: Node indices from root to parent of leaf.
            actions_path: Actions taken at each node.
            leaf_value: Value prediction at the leaf.
        """
        var value = leaf_value

        # Walk backwards from leaf to root
        var path_len = len(search_path)
        for i in range(path_len):
            var idx = path_len - 1 - i
            var node_idx = search_path[idx]
            var action = actions_path[idx]

            if idx < path_len - 1:
                # Not the leaf parent — discount through intermediate steps
                value = self.nodes[node_idx].reward[action] + self.gamma * value
            else:
                # Leaf parent — reward + gamma * leaf_value
                value = self.nodes[node_idx].reward[action] + self.gamma * value

            self.nodes[node_idx].visit_count[action] += 1
            self.nodes[node_idx].total_value[action] += value
            self.nodes[node_idx].total_visits += 1

            self.min_max.update(self.nodes[node_idx].mean_value(action))

    def _scale_hidden_state(mut self, node_idx: Int):
        """Scale hidden state to [0, 1] via min-max normalization.

        Prevents hidden state magnitudes from growing unboundedly through
        repeated dynamics applications.

        Args:
            node_idx: Node index (same as hidden state pool index).
        """
        var offset = node_idx * Self.LATENT_DIM
        var min_val = (self.hidden_states + offset)[0]
        var max_val = min_val
        for i in range(1, Self.LATENT_DIM):
            var v = (self.hidden_states + offset + i)[]
            if Float64(v) < Float64(min_val):
                min_val = v
            if Float64(v) > Float64(max_val):
                max_val = v

        var delta = Float64(max_val) - Float64(min_val)
        if delta > 1e-8:
            for i in range(Self.LATENT_DIM):
                var v = (self.hidden_states + offset + i)[]
                (self.hidden_states + offset + i)[] = Scalar[dtype](
                    (Float64(v) - Float64(min_val)) / delta
                )

    def _decode_value(
        self,
        logits_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        v_min: Float64,
        v_max: Float64,
    ) -> Float64:
        """Decode categorical logits to a scalar value via softmax + expectation.

        Args:
            logits_ptr: Pointer to NUM_BINS logits.
            v_min: Minimum support value.
            v_max: Maximum support value.

        Returns:
            Decoded scalar value (after inverse scalar transform).
        """
        var step = (v_max - v_min) / Float64(
            Self.NUM_BINS - 1
        ) if Self.NUM_BINS > 1 else Float64(0.0)

        # Softmax
        var max_val = Float64(logits_ptr[0])
        for i in range(1, Self.NUM_BINS):
            var v = Float64(logits_ptr[i])
            if v > max_val:
                max_val = v

        var sum_exp = Float64(0.0)
        for i in range(Self.NUM_BINS):
            sum_exp += exp(Float64(logits_ptr[i]) - max_val)

        # Expected value in transformed space
        var result = Float64(0.0)
        for i in range(Self.NUM_BINS):
            var prob = exp(Float64(logits_ptr[i]) - max_val) / sum_exp
            result += prob * (v_min + Float64(i) * step)

        # Inverse transform to get original scale
        return inverse_scalar_transform(result)
