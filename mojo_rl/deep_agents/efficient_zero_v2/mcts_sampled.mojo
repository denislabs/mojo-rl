"""EfficientZero V2 sampled-Gumbel MCTS — CPU (Phase 3.2.1).

Continuous-action sibling of `mcts.mojo`. Same Sequential Halving + visit-
balance template, but the per-action enumeration is replaced by K candidate
action vectors sampled from the squashed-Gaussian policy at each node.

Algorithm (paper App. A, Wang et al. 2024):

  • Root sampling: draw K_ROOT continuous candidates a_i. Half from
        N(μ, σ),
    half from
        N(μ, STD_MAGNIFICATION · σ)
    for an exploration boost (paper App. A "more diverse root candidates").
    The squashed-Gaussian parameterization matches
    `ezv2_policy_loss_grad_continuous_kernel`:

        μ      = SOFT_CLAMP · tanh(μ_raw / SOFT_CLAMP)   (Dreamer-v3, ref. 5.0)
        σ      = softplus(σ_raw + INIT_STD) + MIN_STD    (ref. INIT_STD=1.0)
        u      = μ + σ · ε              (ε ~ N(0, 1))
        a      = MAX_ACTION · tanh(u)

    The corresponding log-prior log π(a_i | s) is evaluated dim-wise via
        c_d    = a_i,d / MAX_ACTION
        u*_d   = atanh(c_d)
        η_d    = (u*_d − μ_d) / σ_d
        log π  = Σ_d  −0.5 η_d² − log σ_d − 0.5 log(2π) − log(1 − c_d²)

    (constant offset vs the true squashed-normal density — irrelevant for
    softmax-based selection).

  • Sequential Halving at root: log₂(K_ROOT) phases, each phase distributes
    its share of the simulation budget evenly across surviving candidates,
    then keeps the top half by score
        s_i = g_i + log π(a_i | s) + σ(completed_Q(a_i))
    where g_i ~ Gumbel(0). Identical formula to the discrete sibling, just
    over candidate indices instead of action indices.

  • Non-root selection: K_NON_ROOT candidates per node (paper App. A:
    K_NON_ROOT = K_ROOT // 2, all from N(μ, σ) — no magnification).
    Visit-balance rule
        a* = argmax_i [π_improved(i) − N(s,i) / (1 + ΣN(s,j))]
    drives empirical visits toward the improved-policy distribution.

The improved-policy *training target* (paper Eq. 8 — simple-best-action
loss) is just the chosen action vector itself, not a distribution. The
accompanying loss kernel
(`ezv2_policy_loss_grad_continuous_kernel`) consumes it directly.

Reuses MuZero's `MinMaxStats` (Q normalization) and
`inverse_scalar_transform` (categorical → scalar value). No new training
code is introduced here — this is the acting-side search engine.

References:
    Wang, Sun, Li et al. — *EfficientZero V2*, ICML 2024 (App. A).
    Danihelka et al. — *Policy improvement by planning with Gumbel*,
    ICLR 2022 (Sequential Halving + visit-balance rule).
"""

from std.math import sqrt, log, exp, tanh, cos, pi
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


comptime LOG_2PI: Float64 = 1.8378770664093453


# ═════════════════════════════════════════════════════════════════════════
# Helpers — squashed-Gaussian sampling + log-prob, file-local
# ═════════════════════════════════════════════════════════════════════════


@always_inline
def _stdnormal() -> Float64:
    """One Box-Muller draw from N(0, 1). Cheap enough for K=16 per node."""
    var u1 = random_float64(1e-9, 1.0 - 1e-9)
    var u2 = random_float64(0.0, 1.0)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


@always_inline
def _softplus(x: Float64) -> Float64:
    """Numerically stable softplus."""
    if x > 0.0:
        return x + log(1.0 + exp(-x))
    return log(1.0 + exp(x))


@always_inline
def _atanh_clipped(c_in: Float64) -> Float64:
    """atanh on `c_in` with hard clip to ±0.999 — matches the loss kernel
    exactly so log-prob computed here lines up with the loss it'll feed."""
    var c = c_in
    if c > 0.999:
        c = 0.999
    if c < -0.999:
        c = -0.999
    return 0.5 * log((1.0 + c) / (1.0 - c))


def _sample_squashed_gaussian_dim(
    mu_d: Float64, sigma_d: Float64, max_action: Float64
) -> Float64:
    """Sample one action coordinate from the squashed-Gaussian policy:

        u = mu_d + sigma_d * eps,     eps ~ N(0, 1)
        a = max_action * tanh(u)

    The mean `mu_d` is the soft-clamped policy mean
        mu_d = SOFT_CLAMP * tanh(mu_raw_d / SOFT_CLAMP)
    (Dreamer-v3 soft clamp, reference `ez_dmc_state.py:421`) so the sampled
    `a` lies in (−max_action, max_action).
    """
    var eps = _stdnormal()
    var u = mu_d + sigma_d * eps
    return max_action * tanh(u)


@always_inline
def _logp_squashed_gaussian_dim(
    a_d: Float64, mu_d: Float64, sigma_d: Float64, max_action: Float64
) -> Float64:
    """log π(a_d | s) per dim. Matches the loss kernel's density up to an
    additive constant (`log MAX_ACTION`) — irrelevant for softmax-based
    Sequential Halving / improved-policy scoring."""
    var c = a_d / max_action
    var u_star = _atanh_clipped(c)
    var diff = u_star - mu_d
    var inv_sg = 1.0 / sigma_d
    var eta = diff * inv_sg
    # Clamp c again for the squash log-correction to stay finite at the
    # boundary (matches kernel).
    var c_clip = c
    if c_clip > 0.999:
        c_clip = 0.999
    if c_clip < -0.999:
        c_clip = -0.999
    return (
        -0.5 * eta * eta
        - log(sigma_d)
        - 0.5 * LOG_2PI
        - log(1.0 - c_clip * c_clip)
    )


# ═════════════════════════════════════════════════════════════════════════
# Tree node — K_PAD candidate slots per node
# ═════════════════════════════════════════════════════════════════════════


struct SampledGumbelMCTSNode[ACT_DIM: Int, K_PAD: Int](
    ImplicitlyCopyable, Movable
):
    """Tree node carrying up to K_PAD candidate action vectors.

    Root nodes use K_ROOT slots, non-root nodes use K_NON_ROOT.
    `active_k` records how many slots are populated so the search loops
    only iterate over the live candidates.

    Per-candidate state (visit count, total value, log prior, reward,
    child node index) is stored in flat InlineArray buffers keyed by
    candidate index `i ∈ [0, active_k)`.
    """

    var actions: InlineArray[Float64, Self.K_PAD * Self.ACT_DIM]
    var visit_count: InlineArray[Int, Self.K_PAD]
    var total_value: InlineArray[Float64, Self.K_PAD]
    var log_prior: InlineArray[Float64, Self.K_PAD]
    var reward: InlineArray[Float64, Self.K_PAD]
    var child_idx: InlineArray[Int, Self.K_PAD]

    var active_k: Int
    var total_visits: Int
    var hidden_state_idx: Int
    var value_estimate: Float64

    def __init__(out self, hidden_idx: Int):
        self.actions = InlineArray[Float64, Self.K_PAD * Self.ACT_DIM](
            uninitialized=True
        )
        self.visit_count = InlineArray[Int, Self.K_PAD](uninitialized=True)
        self.total_value = InlineArray[Float64, Self.K_PAD](uninitialized=True)
        self.log_prior = InlineArray[Float64, Self.K_PAD](uninitialized=True)
        self.reward = InlineArray[Float64, Self.K_PAD](uninitialized=True)
        self.child_idx = InlineArray[Int, Self.K_PAD](uninitialized=True)
        for i in range(Self.K_PAD):
            self.visit_count[i] = 0
            self.total_value[i] = 0.0
            self.log_prior[i] = 0.0
            self.reward[i] = 0.0
            self.child_idx[i] = -1
        for j in range(Self.K_PAD * Self.ACT_DIM):
            self.actions[j] = 0.0
        self.active_k = 0
        self.total_visits = 0
        self.hidden_state_idx = hidden_idx
        self.value_estimate = 0.0

    def __init__(out self, *, copy: Self):
        self.actions = copy.actions
        self.visit_count = copy.visit_count
        self.total_value = copy.total_value
        self.log_prior = copy.log_prior
        self.reward = copy.reward
        self.child_idx = copy.child_idx
        self.active_k = copy.active_k
        self.total_visits = copy.total_visits
        self.hidden_state_idx = copy.hidden_state_idx
        self.value_estimate = copy.value_estimate

    def __init__(out self, *, deinit take: Self):
        self.actions = take.actions
        self.visit_count = take.visit_count
        self.total_value = take.total_value
        self.log_prior = take.log_prior
        self.reward = take.reward
        self.child_idx = take.child_idx
        self.active_k = take.active_k
        self.total_visits = take.total_visits
        self.hidden_state_idx = take.hidden_state_idx
        self.value_estimate = take.value_estimate

    def mean_value(self, i: Int) -> Float64:
        if self.visit_count[i] > 0:
            return self.total_value[i] / Float64(self.visit_count[i])
        return 0.0

    def is_expanded(self, i: Int) -> Bool:
        return self.child_idx[i] >= 0


# ═════════════════════════════════════════════════════════════════════════
# Search engine
# ═════════════════════════════════════════════════════════════════════════


struct SampledGumbelMCTS[
    ACT_DIM: Int,
    LATENT_DIM: Int,
    NUM_BINS: Int = 51,
    NUM_SIMULATIONS: Int = 32,
    K_ROOT: Int = 16,
    K_NON_ROOT: Int = 8,
    MAX_NODES: Int = 256,
    MAX_ACTION: Float64 = 1.0,
    MIN_STD: Float64 = 0.1,
    STD_MAGNIFICATION: Float64 = 3.0,
    # Number of root candidates drawn from the policy `N(μ, σ)`.
    # The remaining `K_ROOT - N_POLICY_AT_ROOT` candidates come from
    # `Uniform(-MAX_ACTION, MAX_ACTION)` (reference `cy_mcts.py:127-128`
    # with `policy_action_num=4, random_action_num=12` for K_ROOT=16).
    # Default `K_ROOT` (all policy) preserves the legacy magnified-policy
    # behavior — when `N_POLICY_AT_ROOT == K_ROOT`, the second half of
    # candidates uses `STD_MAGNIFICATION · σ` exactly as before.
    N_POLICY_AT_ROOT: Int = K_ROOT,
    # Dreamer-v3 soft clamp on μ_pre (reference `ez_dmc_state.py:421`).
    # MUST match the value used in the training loss kernel and acting-side
    # GPU MCTS, otherwise train/act densities diverge.
    SOFT_CLAMP: Float64 = 5.0,
    # Bias inside softplus on σ_raw (reference `ez_dmc_state.py:422`).
    # Same parity caveat as SOFT_CLAMP.
    INIT_STD: Float64 = 1.0,
](Movable):
    """Sampled-Gumbel MCTS for continuous actions.

    Parameters:
        ACT_DIM: Real action vector dimension.
        LATENT_DIM: Hidden-state dimension (must match dynamics output).
        NUM_BINS: Categorical value/reward bin count.
        NUM_SIMULATIONS: Total simulation budget per `search()` call.
        K_ROOT: Number of root candidates (paper default 16). When
            `N_POLICY_AT_ROOT == K_ROOT`, half drawn from `N(μ, σ)` and
            half from `N(μ, STD_MAGNIFICATION · σ)` (legacy magnified
            mode). When `N_POLICY_AT_ROOT < K_ROOT`, the first
            `N_POLICY_AT_ROOT` from `N(μ, σ)` and the remainder from
            `Uniform(-MAX_ACTION, MAX_ACTION)` (reference DMC mode).
        K_NON_ROOT: Number of non-root candidates per node (paper default
            K_ROOT // 2 = 8, all from N(μ, σ)). Must be ≤ K_ROOT.
        MAX_NODES: Maximum tree node budget (root + expansions).
        MAX_ACTION: Action |a_d| upper bound (squashed-Gaussian param).
        MIN_STD: Floor on σ added after softplus (squashed-Gaussian param).
        STD_MAGNIFICATION: Multiplier applied to σ for the second half of
            root candidates (legacy magnified mode only).
        N_POLICY_AT_ROOT: See module-level docstring above.
        SOFT_CLAMP: Dreamer-v3 soft clamp on μ_pre.
        INIT_STD: Bias added inside softplus on σ_raw.

    The squashed-Gaussian hyperparameters must match the agent's
    `Config.ActSpace` impl so the loss kernel sees the same density at
    training time.
    """

    var nodes: List[SampledGumbelMCTSNode[Self.ACT_DIM, Self.K_ROOT]]
    var hidden_states: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var min_max: MinMaxStats

    var gamma: Float64
    var c_visit: Float64
    var c_scale: Float64

    def __init__(
        out self,
        gamma: Float64 = 0.997,
        c_visit: Float64 = 50.0,
        c_scale: Float64 = 0.1,
    ):
        self.nodes = List[SampledGumbelMCTSNode[Self.ACT_DIM, Self.K_ROOT]](
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
    # Public entry — returns chosen action vector + visit distribution
    # over the K_ROOT root candidates + root value estimate
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
        reward_min: Float64 = -0.732_050_807_568_877_3,
        reward_max: Float64 = 0.732_050_807_568_877_3,
        deterministic: Bool = False,
    ) -> Tuple[
        InlineArray[Float64, Self.ACT_DIM],
        InlineArray[Float64, Self.K_ROOT],
        Float64,
    ]:
        """Run sampled-Gumbel search.

        Returns `(chosen_action, visit_distribution, root_value)`:
          - `chosen_action`: ACT_DIM-vector — most-visited candidate at
            root if `deterministic`, else visit-count-weighted soft pick.
          - `visit_distribution`: K_ROOT-vector — N(s, i) / total at root.
            Useful for diagnostics; the loss-kernel target is just
            `chosen_action`.
          - `root_value`: Decoded categorical value at root (V(s)).

        Args:
            root_obs: Current observation (length ≤ RepModel.IN_DIM, zero
                padded). Mirrors `mcts.GumbelMCTS.search`.
            rep_state: Representation network state.
            dyn_state: Dynamics network state.
            pred_state: Prediction network state.
            v_min: Minimum value-support bin.
            v_max: Maximum value-support bin.
            reward_min: Minimum reward-support bin. Decouples reward decode
                from the value range to match the reference (paper uses
                separate `reward_support` and `value_support`). Default is
                reference DMC `h(-2) ≈ -0.732`.
            reward_max: Maximum reward-support bin. Default `h(2) ≈ 0.732`.
            deterministic: If True, pick the argmax-visit candidate
                (eval mode). If False, draw weighted by visit counts
                (training mode).
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

        # ---- Predict root policy params + value -------------------------
        comptime PRED_OUT = PredModel.OUT_DIM
        comptime PRED_IN = PredModel.IN_DIM
        # PRED_OUT = 2*ACT_DIM (μ_raw ‖ σ_raw) + NUM_BINS
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
            pred_out_ptr + 2 * Self.ACT_DIM, v_min, v_max
        )

        # ---- Build root node + sample K_ROOT candidates -----------------
        var root = SampledGumbelMCTSNode[Self.ACT_DIM, Self.K_ROOT](
            hidden_idx=0
        )
        root.value_estimate = root_value
        root.active_k = Self.K_ROOT

        var root_mu = InlineArray[Float64, Self.ACT_DIM](uninitialized=True)
        var root_sg = InlineArray[Float64, Self.ACT_DIM](uninitialized=True)
        for d in range(Self.ACT_DIM):
            var mu_raw = Float64(rebind[Scalar[dtype]](pred_out_t[0, d]))
            var sg_raw = Float64(
                rebind[Scalar[dtype]](pred_out_t[0, Self.ACT_DIM + d])
            )
            root_mu[d] = Self.SOFT_CLAMP * tanh(mu_raw / Self.SOFT_CLAMP)
            root_sg[d] = _softplus(sg_raw + Self.INIT_STD) + Self.MIN_STD

        # Root candidate sampling. Two modes, selected by N_POLICY_AT_ROOT:
        #   • Legacy magnified (N_POLICY_AT_ROOT == K_ROOT): half from
        #     N(μ, σ), half from N(μ, STD_MAG · σ). Matches the original
        #     Pendulum baseline.
        #   • Reference DMC (N_POLICY_AT_ROOT < K_ROOT): first
        #     N_POLICY_AT_ROOT from N(μ, σ), rest from
        #     Uniform(-MAX_ACTION, MAX_ACTION). Matches `cy_mcts.py:127-128`
        #     with `policy_action_num=4, random_action_num=12`. Uniform
        #     samples decouple exploration from the policy's current
        #     `μ`, breaking the action-saturation feedback loop where a
        #     biased policy traps MCTS into evaluating only candidates
        #     near the same biased μ.
        # Both modes score every candidate under the policy density
        # `N(μ, σ)` for `log_prior` so the Sequential-Halving prior is
        # comparable across modes.
        comptime LEGACY_MAGNIFIED = Self.N_POLICY_AT_ROOT == Self.K_ROOT
        for i in range(Self.K_ROOT):
            var is_policy_sample: Bool
            var is_magnified: Bool

            comptime if LEGACY_MAGNIFIED:
                # Legacy mode: every candidate from policy; second half magnified.
                is_policy_sample = True
                is_magnified = i >= Self.K_ROOT // 2
            else:
                is_policy_sample = i < Self.N_POLICY_AT_ROOT
                is_magnified = False

            var lp = 0.0
            if is_policy_sample:
                var sg_eff = root_sg
                if is_magnified:
                    var sg_widened = InlineArray[Float64, Self.ACT_DIM](
                        uninitialized=True
                    )
                    for d in range(Self.ACT_DIM):
                        sg_widened[d] = root_sg[d] * Self.STD_MAGNIFICATION
                    sg_eff = sg_widened
                for d in range(Self.ACT_DIM):
                    var a_d = _sample_squashed_gaussian_dim(
                        root_mu[d], sg_eff[d], Self.MAX_ACTION
                    )
                    root.actions[i * Self.ACT_DIM + d] = a_d
                    # Score under the unmagnified policy density — widened
                    # samples would otherwise be double-penalized for being
                    # in the tail of `N(μ, σ)`.
                    lp += _logp_squashed_gaussian_dim(
                        a_d, root_mu[d], root_sg[d], Self.MAX_ACTION
                    )
            else:
                # Uniform random in [-MAX_ACTION, MAX_ACTION] per dim.
                # Score under the policy density `N(μ, σ)` so these tail
                # samples still get a meaningful (typically low) log_prior
                # vs the policy-centered ones.
                for d in range(Self.ACT_DIM):
                    var a_d = random_float64(-Self.MAX_ACTION, Self.MAX_ACTION)
                    root.actions[i * Self.ACT_DIM + d] = a_d
                    lp += _logp_squashed_gaussian_dim(
                        a_d, root_mu[d], root_sg[d], Self.MAX_ACTION
                    )
            root.log_prior[i] = lp

        pred_out_ptr.free()
        self.nodes.append(root^)

        # ---- Gumbel noise per root candidate (paper Sequential Halving) -
        var gumbels = InlineArray[Float64, Self.K_ROOT](uninitialized=True)
        for i in range(Self.K_ROOT):
            var u = random_float64(1e-9, 1.0 - 1e-9)
            gumbels[i] = -log(-log(u))

        # ---- Sequential Halving over K_ROOT candidates ------------------
        var k_actual = _largest_power_of_two_le(Self.K_ROOT)
        var num_phases = _ilog2(k_actual)
        if num_phases < 1:
            num_phases = 1
        var per_phase_budget = Self.NUM_SIMULATIONS // num_phases
        if per_phase_budget < 1:
            per_phase_budget = 1

        var active = InlineArray[Int, Self.K_ROOT](uninitialized=True)
        for i in range(Self.K_ROOT):
            active[i] = -1
        for i in range(k_actual):
            active[i] = i
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
                    self._simulate[DynModel, PredModel, DynOpt, PredOpt](
                        cand_idx,
                        dyn_state,
                        pred_state,
                        v_min,
                        v_max,
                        reward_min,
                        reward_max,
                    )
                    sims_used += 1

            if phase + 1 < num_phases and active_size > 1:
                var keep = active_size // 2
                if keep < 1:
                    keep = 1
                self._halve_active_set(gumbels, active, active_size, keep)
                active_size = keep

        # Spend any leftover budget on the last surviving candidate.
        while sims_used < Self.NUM_SIMULATIONS and active_size > 0:
            var leftover = active[0]
            self._simulate[DynModel, PredModel, DynOpt, PredOpt](
                leftover,
                dyn_state,
                pred_state,
                v_min,
                v_max,
                reward_min,
                reward_max,
            )
            sims_used += 1

        # ---- Build outputs ----------------------------------------------
        var visits = InlineArray[Float64, Self.K_ROOT](uninitialized=True)
        var total_visits = 0
        for i in range(Self.K_ROOT):
            visits[i] = Float64(self.nodes[0].visit_count[i])
            total_visits += self.nodes[0].visit_count[i]
        if total_visits > 0:
            for i in range(Self.K_ROOT):
                visits[i] = visits[i] / Float64(total_visits)
        else:
            for i in range(Self.K_ROOT):
                visits[i] = 1.0 / Float64(Self.K_ROOT)

        var chosen_idx = self._pick_chosen(visits, deterministic)
        var chosen = InlineArray[Float64, Self.ACT_DIM](uninitialized=True)
        for d in range(Self.ACT_DIM):
            chosen[d] = self.nodes[0].actions[chosen_idx * Self.ACT_DIM + d]
        return (chosen, visits, root_value)

    # ─────────────────────────────────────────────────────────────────────
    # Action choice from visit distribution
    # ─────────────────────────────────────────────────────────────────────

    def _pick_chosen(
        self,
        visits: InlineArray[Float64, Self.K_ROOT],
        deterministic: Bool,
    ) -> Int:
        if deterministic:
            var best_i = 0
            var best_v = -1.0
            for i in range(Self.K_ROOT):
                if visits[i] > best_v:
                    best_v = visits[i]
                    best_i = i
            return best_i
        # Visit-weighted soft pick — sample from the visit distribution.
        var u = random_float64(0.0, 1.0)
        var acc = 0.0
        for i in range(Self.K_ROOT):
            acc += visits[i]
            if u <= acc:
                return i
        return Self.K_ROOT - 1

    # ─────────────────────────────────────────────────────────────────────
    # Halving — drop bottom half of `active` by Sequential-Halving score
    # ─────────────────────────────────────────────────────────────────────

    def _halve_active_set(
        self,
        gumbels: InlineArray[Float64, Self.K_ROOT],
        mut active: InlineArray[Int, Self.K_ROOT],
        active_size: Int,
        keep: Int,
    ):
        var root = self.nodes[0]
        var completed_q = self._completed_q(0)
        var sigma = self._sigma_q_array(0, completed_q)

        var scored = InlineArray[Float64, Self.K_ROOT](uninitialized=True)
        var indices = InlineArray[Int, Self.K_ROOT](uninitialized=True)
        for i in range(Self.K_ROOT):
            scored[i] = -1e18
            indices[i] = -1
        for i in range(active_size):
            var cand = active[i]
            scored[i] = gumbels[cand] + root.log_prior[cand] + sigma[cand]
            indices[i] = cand

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

        for i in range(Self.K_ROOT):
            active[i] = -1
        for i in range(keep):
            active[i] = indices[i]

    # ─────────────────────────────────────────────────────────────────────
    # Completed-Q + improved policy
    # ─────────────────────────────────────────────────────────────────────

    def _v_mix(self, node_idx: Int) -> Float64:
        var node = self.nodes[node_idx]
        var visited_logp_max = -1e18
        var visited_count = 0
        for i in range(node.active_k):
            if node.visit_count[i] > 0:
                if node.log_prior[i] > visited_logp_max:
                    visited_logp_max = node.log_prior[i]
                visited_count += 1
        if visited_count == 0:
            return node.value_estimate

        var sum_w = 0.0
        var weighted_q = 0.0
        for i in range(node.active_k):
            if node.visit_count[i] > 0:
                var w = exp(node.log_prior[i] - visited_logp_max)
                sum_w += w
                weighted_q += w * node.mean_value(i)
        if sum_w <= 0.0:
            return node.value_estimate
        var mean_visited_q = weighted_q / sum_w

        var total = Float64(node.total_visits)
        return (node.value_estimate + total * mean_visited_q) / (1.0 + total)

    def _completed_q(self, node_idx: Int) -> InlineArray[Float64, Self.K_ROOT]:
        var node = self.nodes[node_idx]
        var v_mix = self._v_mix(node_idx)
        var q = InlineArray[Float64, Self.K_ROOT](uninitialized=True)
        for i in range(Self.K_ROOT):
            q[i] = 0.0
        for i in range(node.active_k):
            if node.visit_count[i] > 0:
                q[i] = node.mean_value(i)
            else:
                q[i] = v_mix
        return q

    def _sigma_q_array(
        self,
        node_idx: Int,
        q: InlineArray[Float64, Self.K_ROOT],
    ) -> InlineArray[Float64, Self.K_ROOT]:
        var node = self.nodes[node_idx]
        var max_visit = 0
        for i in range(node.active_k):
            if node.visit_count[i] > max_visit:
                max_visit = node.visit_count[i]
        var scale = (self.c_visit + Float64(max_visit)) * self.c_scale

        var out = InlineArray[Float64, Self.K_ROOT](uninitialized=True)
        for i in range(Self.K_ROOT):
            out[i] = 0.0
        for i in range(node.active_k):
            out[i] = scale * self.min_max.normalize(q[i])
        return out

    def _improved_policy_at(
        self,
        node_idx: Int,
    ) -> InlineArray[Float64, Self.K_ROOT]:
        var node = self.nodes[node_idx]
        var completed_q = self._completed_q(node_idx)
        var sigma = self._sigma_q_array(node_idx, completed_q)

        var z = InlineArray[Float64, Self.K_ROOT](uninitialized=True)
        var max_z = -1e18
        for i in range(Self.K_ROOT):
            z[i] = -1e18
        for i in range(node.active_k):
            z[i] = node.log_prior[i] + sigma[i]
            if z[i] > max_z:
                max_z = z[i]

        var sum_exp = 0.0
        var probs = InlineArray[Float64, Self.K_ROOT](uninitialized=True)
        for i in range(Self.K_ROOT):
            probs[i] = 0.0
        for i in range(node.active_k):
            var e = exp(z[i] - max_z)
            probs[i] = e
            sum_exp += e
        if sum_exp <= 0.0:
            for i in range(node.active_k):
                probs[i] = 1.0 / Float64(node.active_k)
            return probs
        for i in range(node.active_k):
            probs[i] = probs[i] / sum_exp
        return probs

    def _select_non_root_candidate(self, node_idx: Int) -> Int:
        var probs = self._improved_policy_at(node_idx)
        var node = self.nodes[node_idx]
        var denom = 1.0 + Float64(node.total_visits)

        var best_i = 0
        var best_s = -1e18
        for i in range(node.active_k):
            var s = probs[i] - Float64(node.visit_count[i]) / denom
            if s > best_s:
                best_s = s
                best_i = i
        return best_i

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
        root_cand_idx: Int,
        dyn_state: NetworkState[DynModel, DynOpt],
        pred_state: NetworkState[PredModel, PredOpt],
        v_min: Float64,
        v_max: Float64,
        reward_min: Float64,
        reward_max: Float64,
    ):
        """One simulation: take root candidate `root_cand_idx`, traverse
        the non-root subtree by visit-balance until an unexpanded leaf,
        expand via dynamics + prediction, back the value up the path."""
        var search_path = List[Int](capacity=64)
        var cand_path = List[Int](capacity=64)

        var node_idx = 0
        search_path.append(node_idx)
        cand_path.append(root_cand_idx)

        while True:
            var ci = cand_path[len(cand_path) - 1]
            if not self.nodes[node_idx].is_expanded(ci):
                break
            node_idx = self.nodes[node_idx].child_idx[ci]
            search_path.append(node_idx)
            var next_ci = self._select_non_root_candidate(node_idx)
            cand_path.append(next_ci)

        var parent_idx = search_path[len(search_path) - 1]
        var leaf_cand = cand_path[len(cand_path) - 1]

        if len(self.nodes) >= Self.MAX_NODES:
            self._backup(search_path, cand_path, 0.0)
            return

        var child_hidden_idx = len(self.nodes)
        var leaf_value = self._expand_node[
            DynModel, PredModel, DynOpt, PredOpt
        ](
            parent_idx,
            leaf_cand,
            child_hidden_idx,
            dyn_state,
            pred_state,
            v_min,
            v_max,
            reward_min,
            reward_max,
        )
        self._backup(search_path, cand_path, leaf_value)

    def _expand_node[
        DynModel: Model,
        PredModel: Model,
        DynOpt: Optimizer,
        PredOpt: Optimizer,
    ](
        mut self,
        parent_idx: Int,
        cand_idx: Int,
        child_hidden_idx: Int,
        dyn_state: NetworkState[DynModel, DynOpt],
        pred_state: NetworkState[PredModel, PredOpt],
        v_min: Float64,
        v_max: Float64,
        reward_min: Float64,
        reward_max: Float64,
    ) -> Float64:
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
        # Continuous: write the raw action vector into the action slots.
        var act_off = cand_idx * Self.ACT_DIM
        for d in range(Self.ACT_DIM):
            dyn_input_ptr[Self.LATENT_DIM + d] = Scalar[dtype](
                self.nodes[parent_idx].actions[act_off + d]
            )

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
        # Reward decode uses the reward support (paper `dmc_state.yaml`
        # carries a separate `reward_support: range=[-2, 2]`). Sharing
        # `v_min/v_max` with the value head left MCTS reading rewards
        # ~100× too coarse on DMC envs.
        var reward = self._decode_value(
            dyn_output_ptr + Self.LATENT_DIM, reward_min, reward_max
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

        var child = SampledGumbelMCTSNode[Self.ACT_DIM, Self.K_ROOT](
            hidden_idx=child_hidden_idx
        )
        child.active_k = Self.K_NON_ROOT

        # Sample K_NON_ROOT candidates from N(μ, σ) at the child node.
        var child_mu = InlineArray[Float64, Self.ACT_DIM](uninitialized=True)
        var child_sg = InlineArray[Float64, Self.ACT_DIM](uninitialized=True)
        for d in range(Self.ACT_DIM):
            var mu_raw = Float64(rebind[Scalar[dtype]](pred_out_t[0, d]))
            var sg_raw = Float64(
                rebind[Scalar[dtype]](pred_out_t[0, Self.ACT_DIM + d])
            )
            child_mu[d] = Self.SOFT_CLAMP * tanh(mu_raw / Self.SOFT_CLAMP)
            child_sg[d] = _softplus(sg_raw + Self.INIT_STD) + Self.MIN_STD

        for i in range(Self.K_NON_ROOT):
            var lp = 0.0
            for d in range(Self.ACT_DIM):
                var a_d = _sample_squashed_gaussian_dim(
                    child_mu[d], child_sg[d], Self.MAX_ACTION
                )
                child.actions[i * Self.ACT_DIM + d] = a_d
                lp += _logp_squashed_gaussian_dim(
                    a_d, child_mu[d], child_sg[d], Self.MAX_ACTION
                )
            child.log_prior[i] = lp

        var leaf_value = self._decode_value(
            pred_out_ptr + 2 * Self.ACT_DIM, v_min, v_max
        )
        child.value_estimate = leaf_value

        pred_out_ptr.free()

        self.nodes[parent_idx].reward[cand_idx] = reward
        self.nodes[parent_idx].child_idx[cand_idx] = child_hidden_idx
        self.nodes.append(child^)

        return leaf_value

    def _backup(
        mut self,
        search_path: List[Int],
        cand_path: List[Int],
        leaf_value: Float64,
    ):
        var value = leaf_value
        var path_len = len(search_path)
        for i in range(path_len):
            var idx = path_len - 1 - i
            var node_idx = search_path[idx]
            var ci = cand_path[idx]

            value = self.nodes[node_idx].reward[ci] + self.gamma * value

            self.nodes[node_idx].visit_count[ci] += 1
            self.nodes[node_idx].total_value[ci] += value
            self.nodes[node_idx].total_visits += 1

            self.min_max.update(self.nodes[node_idx].mean_value(ci))

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
        ) if Self.NUM_BINS > 1 else 0.0

        var max_val = Float64(logits_ptr[0])
        for i in range(1, Self.NUM_BINS):
            var v = Float64(logits_ptr[i])
            if v > max_val:
                max_val = v

        var sum_exp = 0.0
        for i in range(Self.NUM_BINS):
            sum_exp += exp(Float64(logits_ptr[i]) - max_val)

        var result = 0.0
        for i in range(Self.NUM_BINS):
            var prob = exp(Float64(logits_ptr[i]) - max_val) / sum_exp
            result += prob * (v_min + Float64(i) * step)

        return inverse_scalar_transform(result)


# ═════════════════════════════════════════════════════════════════════════
# File-local helpers
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
