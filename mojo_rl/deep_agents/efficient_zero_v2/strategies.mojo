"""EfficientZero V2 strategies — value targets and policy losses.

Two trait families that the EZ-V2 training loop dispatches over at compile
time, mirroring the strategy-trait pattern used elsewhere in mojo-rl
(`muzero/strategies.mojo`, `core/strategies/`):

  • `ValueTarget` — selects/blends between **Search-Based Value Estimation**
    (SVE) and the standard **multi-step TD** target. SVE is the empirical
    mean of the bootstrap returns sampled by every simulation in the search
    tree, which equals the visit-weighted root mean Q. Mixed value linearly
    blends SVE → TD as data ages; the rationale (paper Eq. 16) is that fresh
    data has search distributions matching the current policy, so SVE is
    accurate, while older data drifts off-policy and TD with reanalyze (run
    by the agent's training loop with the latest target net) is more
    reliable. Defaults are paper Table 3: T_FRESH=20000, T_STALE=40000.

  • `PolicyLoss` — `FullCrossEntropy` for low-dim discrete action spaces,
    `SimpleBestAction` (-log π(a*)) for high-dim cases where computing the
    full target distribution is too expensive (paper App. A,
    `ez/utils/loss.py:88`).

Both traits expose `@staticmethod` `compute(...)` methods so the entire
dispatch is comptime — no per-step branch on a config flag at runtime.

Helpers (free functions) compute the inputs the strategies select between:

    compute_sve(total_value_sum, total_visits)   — root SVE from MCTS stats
    compute_multistep_td[N](rewards, valid, term, gamma, bootstrap) — n-step
"""

from std.math import exp, log


# ═════════════════════════════════════════════════════════════════════════
# Helpers — pre-computation of the candidate targets
# ═════════════════════════════════════════════════════════════════════════


def compute_sve(total_value_sum: Float64, total_visits: Int) -> Float64:
    """Search-Based Value Estimation (Eq. 13).

    SVE(s_0) = (1/N) Σ_i V_i

    where V_i is the bootstrap return collected by simulation i and N is
    the simulation budget. Algebraically this equals
    `Σ_a total_value(s_0, a) / Σ_a visit_count(s_0, a)` because every
    simulation's V_i is added to `total_value` of the root action it took
    (and `visit_count` counts simulations).

    This is the same quantity that MuZero's
    `gpu_mcts_extract_root_value_kernel` already computes — SVE just
    re-uses it as the value target instead of as a logging output.

    Args:
        total_value_sum: Σ_a root.total_value[a] after search.
        total_visits: Σ_a root.visit_count[a] after search (= N_sim).

    Returns:
        Scalar SVE estimate in raw value space.
    """
    if total_visits <= 0:
        return Float64(0.0)
    return total_value_sum / Float64(total_visits)


def compute_multistep_td[
    N: Int,
](
    rewards: InlineArray[Float64, N],
    valid_steps: Int,
    terminated: Bool,
    gamma: Float64,
    bootstrap: Float64,
) -> Float64:
    """Standard n-step TD return.

        G^n_t = Σ_{k=0}^{K-1} γ^k r_{t+k}  +  γ^K · V̂(s_{t+K}) · 𝟙[¬term]

    where K = min(N, valid_steps). The bootstrap term is dropped if a
    terminal state was reached within the window. `valid_steps < N` happens
    when the trajectory ends inside the K-step window (replay padding).

    Args:
        rewards: Up to N future rewards starting at step t (0-indexed).
        valid_steps: How many of those rewards are real (≤ N).
        terminated: True if the trajectory hit a terminal state inside
            the window.
        gamma: Discount factor.
        bootstrap: V̂(s_{t+K}) — typically the target net's value estimate
            on the bootstrap state (decoded to scalar space).

    Returns:
        Scalar return estimate in raw value space.
    """
    var ret = Float64(0.0)
    var disc = Float64(1.0)
    var k_max = valid_steps if valid_steps < N else N
    for k in range(k_max):
        ret += disc * rewards[k]
        disc *= gamma
    if not terminated:
        ret += disc * bootstrap
    return ret


# ═════════════════════════════════════════════════════════════════════════
# ValueTarget — selects/blends SVE vs TD
# ═════════════════════════════════════════════════════════════════════════


trait ValueTarget:
    """Strategy producing the scalar value target V_target(s_t) used in the
    L_V loss term.

    The caller pre-computes SVE and TD (see `compute_sve` /
    `compute_multistep_td`) plus the data age (in train steps) and asks
    the strategy how to combine them.
    """

    comptime TARGET_TYPE: Int  # 0=SVE, 1=MultiStepTD, 2=Mixed

    @staticmethod
    def compute(sve: Float64, td: Float64, age: Int) -> Float64:
        """Combine SVE and TD into a single scalar target.

        Args:
            sve: Pre-computed SVE estimate.
            td: Pre-computed multi-step TD return.
            age: Number of train steps since the transition was collected.

        Returns:
            Selected/blended scalar target.
        """
        ...


struct SVETarget(ValueTarget):
    """Pure SVE target. Right when the search distribution faithfully
    matches the current policy (i.e., fresh data) — paper §4.2."""

    comptime TARGET_TYPE: Int = 0

    @staticmethod
    def compute(sve: Float64, td: Float64, age: Int) -> Float64:
        return sve


struct MultiStepTDTarget(ValueTarget):
    """Pure n-step TD return target. Standard MuZero choice; right when
    the data is stale enough that the snapshotted search has drifted off
    the current policy and Reanalyze + target-net bootstrap are more
    reliable."""

    comptime TARGET_TYPE: Int = 1

    @staticmethod
    def compute(sve: Float64, td: Float64, age: Int) -> Float64:
        return td


struct MixedValueTarget[
    T_FRESH: Int = 20000,
    T_STALE: Int = 40000,
](ValueTarget):
    """Reference-parity mixed value target (paper Eq. 16 +
    `EfficientZeroV2-main/ez/agents/base.py:419-424` +
    `ez/worker/batch_worker.py:580`).

    The reference combines TWO decisions:

      1. **Training-step gate** (`start_use_mix_training_steps=40000`):
         Before this many training steps, use pure TD bootstrap (= SARSA).
         After, fall through to the per-sample decision below.
         Maps to our `T_STALE` parameter. This gate is applied **by the
         caller** (not by `compute()`) — `compute()` always returns the
         per-sample blend; the caller decides whether to consult it.

      2. **Per-sample age switch** (`mixed_value_threshold=20000`):
         For recent samples (age < threshold), use TD bootstrap — the
         target-net's value estimate is current. For old samples
         (age ≥ threshold), use stored SVE — the search value was
         reliable when computed, and the current target-net estimate of
         those old states would mix in policy that has drifted.
         Maps to our `T_FRESH` parameter. Hard switch (no interpolation)
         — matches reference's binary `top_value_mask ∈ {0, 1}`.

    Direction note (2026-05-13 fix): prior version had this reversed
    (low age → SVE). Reference is the opposite — recent samples use TD,
    old samples use SVE.

    Args:
        T_FRESH: Per-sample age threshold. Sample-age < T_FRESH ⇒ TD;
            age ≥ T_FRESH ⇒ SVE. Reference `mixed_value_threshold=20000`.
        T_STALE: Training-step gate. Read by the caller (not by
            `compute()`) to decide when to switch from pure TD to the
            per-sample blend. Reference `start_use_mix_training_steps=40000`.
    """

    comptime TARGET_TYPE: Int = 2
    comptime FRESH: Int = Self.T_FRESH
    comptime STALE: Int = Self.T_STALE

    @staticmethod
    def compute(sve: Float64, td: Float64, age: Int) -> Float64:
        if age < Self.T_FRESH:
            return td
        return sve


# ═════════════════════════════════════════════════════════════════════════
# PolicyLoss — discrete-action variants
# ═════════════════════════════════════════════════════════════════════════


trait PolicyLoss:
    """Strategy producing the L_P loss term for discrete actions.

    Both implementations consume the same arguments — they ignore the
    fields they don't need — so the trait signature is uniform:

        compute[ACT](logits, target_policy, target_action) -> Float64
    """

    comptime LOSS_TYPE: Int  # 0=FullCrossEntropy, 1=SimpleBestAction

    @staticmethod
    def compute[
        ACT: Int,
    ](
        logits: InlineArray[Float64, ACT],
        target_policy: InlineArray[Float64, ACT],
        target_action: Int,
    ) -> Float64:
        """Negative log likelihood under the predicted policy distribution.

        Args:
            logits: Raw policy logits (pre-softmax) of length ACT.
            target_policy: Search-improved policy distribution
                (`π̂ = softmax(logits + σ(completed_Q))`); ignored by
                `SimpleBestAction`.
            target_action: Search-selected action `a*_S`; ignored by
                `FullCrossEntropy`.

        Returns:
            Scalar loss (mean over the batch is the caller's job).
        """
        ...


struct FullCrossEntropy(PolicyLoss):
    """L = -Σ_a π_target(a) · log π(a). Default for low-dim discrete."""

    comptime LOSS_TYPE: Int = 0

    @staticmethod
    def compute[
        ACT: Int,
    ](
        logits: InlineArray[Float64, ACT],
        target_policy: InlineArray[Float64, ACT],
        target_action: Int,
    ) -> Float64:
        # Numerically stable log-softmax.
        var max_l = logits[0]
        for i in range(1, ACT):
            if logits[i] > max_l:
                max_l = logits[i]
        var sum_e = Float64(0.0)
        for i in range(ACT):
            sum_e += exp(logits[i] - max_l)
        var log_sum = log(sum_e) + max_l
        var loss = Float64(0.0)
        for i in range(ACT):
            if target_policy[i] > 0.0:
                loss -= target_policy[i] * (logits[i] - log_sum)
        return loss


struct SimpleBestAction(PolicyLoss):
    """L = -log π(a*_S). Used in EZ-V2 for action spaces large enough
    that sampling targets the search-selected action only (paper App. A
    `simple_loss`)."""

    comptime LOSS_TYPE: Int = 1

    @staticmethod
    def compute[
        ACT: Int,
    ](
        logits: InlineArray[Float64, ACT],
        target_policy: InlineArray[Float64, ACT],
        target_action: Int,
    ) -> Float64:
        var max_l = logits[0]
        for i in range(1, ACT):
            if logits[i] > max_l:
                max_l = logits[i]
        var sum_e = Float64(0.0)
        for i in range(ACT):
            sum_e += exp(logits[i] - max_l)
        var log_sum = log(sum_e) + max_l
        return -(logits[target_action] - log_sum)
