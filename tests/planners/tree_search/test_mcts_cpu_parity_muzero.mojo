"""Phase 3 planners: CPU MCTS bit-parity vs the legacy MuZero ``MCTS``.

This is the parity test the design doc calls out as Phase 3's safety
net: stand the legacy ``mojo_rl.deep_agents.muzero.mcts.MCTS`` and the
new ``mojo_rl.planners.tree_search.GenericCPUMCTS`` side-by-side on the
same tiny MuZero-style network setup, same root observation, no
exploration noise, and assert that the visit-count distributions at the
root come out identical.

What this falsifies:
  * Trait adapter precision drift (Float32 round-trip mismatch).
  * Off-by-one in the value/reward decode.
  * Sign / discount bugs in the new ``_backup`` path.
  * Mis-wiring of MinMax Q-normalization in PUCT selection.

Setup (small enough to be exhaustive in 16 simulations):
  OBS=4, ACT=2, LATENT=4, BINS=3, HIDDEN=8.
  RepModel/DynModel/PredModel are plain ``Sequential[LinearMish, Linear]`` —
  no MinMaxNorm layer, so MCTS's post-hoc scaling does real work (the
  bit-parity case we care about). Kaiming init with a fixed seed makes
  both MCTS instances see identical params.

Adapters in this file (``MuZeroRepresentationCPU`` etc.) round through
``Scalar[dtype]`` (float32) on every hidden-state write so the new
MCTS's Float64 hidden pool stores values that are *bit-identical* to
the legacy MCTS's Float32 pool when widened to Float64. Without that
round-trip the two trees diverge after a few sims due to ULP-level
hidden-state differences flipping a PUCT tie.

Usage:
    pixi run mojo run -I . tests/planners/tree_search/test_mcts_cpu_parity_muzero.mojo
"""

from std.math import abs as math_abs, sqrt, exp, log
from std.memory import alloc, memset
from std.random import seed as _set_seed
from std.testing import assert_true, assert_equal

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState

from mojo_rl.deep_agents.muzero.mcts import MCTS as LegacyMCTS
from mojo_rl.deep_agents.muzero.utils import inverse_scalar_transform

from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    NoNoise,
    SinglePlayer,
    Representation,
    Dynamics,
    Prediction,
)


# ─── Tiny MuZero-style config ─────────────────────────────────────────────


comptime OBS: Int = 4
comptime ACT: Int = 2
comptime LATENT: Int = 4
comptime BINS: Int = 3
comptime HIDDEN: Int = 8
comptime DYN_IN: Int = LATENT + ACT
comptime DYN_OUT: Int = LATENT + BINS
comptime PRED_OUT: Int = ACT + BINS

# Plain Sequential with no MinMaxNorm — MCTS's post-hoc MinMax scaling
# does observable work here, which is what the parity test exercises.
comptime RepModel = Sequential[
    LinearMish[OBS, HIDDEN], Linear[HIDDEN, LATENT]
]
comptime DynModel = Sequential[
    LinearMish[DYN_IN, HIDDEN], Linear[HIDDEN, DYN_OUT]
]
comptime PredModel = Sequential[
    LinearMish[LATENT, HIDDEN], Linear[HIDDEN, PRED_OUT]
]
comptime OptType = Adam[LR=1e-3]


# ─── Adapter helpers — float32 round-trip + categorical decode ────────────


@always_inline
def _f32_round(x: Float64) -> Float64:
    """Round-trip through Float32 so adapter writes match what the
    legacy MCTS would store in its ``Scalar[dtype]`` hidden pool.
    """
    return Float64(Scalar[dtype](x))


def _minmax_scale_f32_inplace[
    DIM: Int
](mut buf: List[Float64]):
    """MinMax scale ``buf`` to [0, 1] with float32 precision throughout.

    Mirrors ``muzero/mcts._scale_hidden_state``: walk the buffer once
    for min/max (computed from float32-rounded values) and once for
    the divide. ``delta < 1e-8`` is a no-op, matching legacy.
    """
    var min_v = _f32_round(buf[0])
    var max_v = min_v
    for i in range(1, DIM):
        var v = _f32_round(buf[i])
        if v < min_v:
            min_v = v
        if v > max_v:
            max_v = v
    var delta = max_v - min_v
    if delta > 1e-8:
        for i in range(DIM):
            var v = _f32_round(buf[i])
            buf[i] = _f32_round((v - min_v) / delta)
    else:
        # Leave buf untouched but still float32-quantize so subsequent
        # reads see the same bits the legacy pool would store.
        for i in range(DIM):
            buf[i] = _f32_round(buf[i])


def _decode_value_categorical[
    NBINS: Int,
](
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    v_min: Float64,
    v_max: Float64,
) raises -> Float64:
    """Categorical decode — softmax + expectation + inverse scalar
    transform. Matches ``muzero/mcts.MCTS._decode_value`` line-for-line.
    """
    var step = (v_max - v_min) / Float64(NBINS - 1) if NBINS > 1 else 0.0
    var max_val = Float64(logits[0])
    for i in range(1, NBINS):
        var v = Float64(logits[i])
        if v > max_val:
            max_val = v
    var sum_exp = Float64(0.0)
    for i in range(NBINS):
        sum_exp += exp(Float64(logits[i]) - max_val)
    var result = Float64(0.0)
    for i in range(NBINS):
        var prob = exp(Float64(logits[i]) - max_val) / sum_exp
        result += prob * (v_min + Float64(i) * step)
    return inverse_scalar_transform(result)


# ─── Adapters wrapping NetworkState[Model, Adam] ──────────────────────────


@fieldwise_init
struct MuZeroRepresentationCPU(
    Movable, ImplicitlyDestructible, Representation,
):
    """Adapter: obs (List[Float64]) → MCTS hidden (List[Float64]) via
    the representation network plus post-hoc MinMax scaling.

    Owns raw pointers into the caller's ``NetworkState`` — no lifetime
    management. The caller is responsible for keeping the
    ``NetworkState`` alive for as long as any adapter holds a pointer.
    """

    comptime OBS_DIM: Int = OBS
    comptime LATENT_DIM: Int = LATENT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    def encode_cpu(
        mut self,
        obs: List[Float64],
        mut hidden_out: List[Float64],
    ) raises:
        comptime B: Int = 1
        var inp_ptr = alloc[Scalar[dtype]](RepModel.IN_DIM)
        for i in range(RepModel.IN_DIM):
            inp_ptr[i] = Scalar[dtype](
                obs[i] if i < len(obs) else Float64(0.0)
            )
        var inp_t = LayoutTensor[
            dtype, Layout.row_major(B, RepModel.IN_DIM), MutAnyOrigin
        ](inp_ptr)

        var out_ptr = alloc[Scalar[dtype]](RepModel.OUT_DIM)
        memset(out_ptr, 0, RepModel.OUT_DIM)
        var out_t = LayoutTensor[
            dtype, Layout.row_major(B, RepModel.OUT_DIM), MutAnyOrigin
        ](out_ptr)

        var params_t = LayoutTensor[
            dtype, Layout.row_major(RepModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(RepModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)

        Network[RepModel, OptType].forward[B](
            inp_t, out_t, params_t, state_t
        )

        for i in range(LATENT):
            hidden_out[i] = Float64(out_ptr[i])

        inp_ptr.free()
        out_ptr.free()

        # Post-hoc MinMax scaling — float32-precision throughout so the
        # Float64 pool slot ends up storing the same bits the legacy
        # Float32 pool would.
        _minmax_scale_f32_inplace[LATENT](hidden_out)


@fieldwise_init
struct MuZeroDynamicsCPU(
    Movable, ImplicitlyDestructible, Dynamics,
):
    """Adapter: (hidden, action) → (next_hidden, reward_scalar).

    Encodes ``action`` as one-hot, forwards through the dynamics net,
    splits the output into [LATENT | NUM_BINS], applies MinMax scaling
    to the hidden portion, and decodes the categorical reward to a
    scalar. ``v_min`` / ``v_max`` are the same reward support used by
    the agent at training time.
    """

    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var v_min: Float64
    var v_max: Float64

    def step_cpu(
        mut self,
        hidden_in: List[Float64],
        action: Int,
        mut hidden_out: List[Float64],
    ) raises -> Float64:
        comptime B: Int = 1
        var inp_ptr = alloc[Scalar[dtype]](DynModel.IN_DIM)
        memset(inp_ptr, 0, DynModel.IN_DIM)
        for i in range(LATENT):
            inp_ptr[i] = Scalar[dtype](hidden_in[i])
        inp_ptr[LATENT + action] = Scalar[dtype](1.0)

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(B, DynModel.IN_DIM), MutAnyOrigin
        ](inp_ptr)

        var out_ptr = alloc[Scalar[dtype]](DynModel.OUT_DIM)
        memset(out_ptr, 0, DynModel.OUT_DIM)
        var out_t = LayoutTensor[
            dtype, Layout.row_major(B, DynModel.OUT_DIM), MutAnyOrigin
        ](out_ptr)

        var params_t = LayoutTensor[
            dtype, Layout.row_major(DynModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(DynModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)

        Network[DynModel, OptType].forward[B](
            inp_t, out_t, params_t, state_t
        )

        for i in range(LATENT):
            hidden_out[i] = Float64(out_ptr[i])

        var reward = _decode_value_categorical[BINS](
            out_ptr + LATENT, self.v_min, self.v_max
        )

        inp_ptr.free()
        out_ptr.free()

        _minmax_scale_f32_inplace[LATENT](hidden_out)

        return reward


@fieldwise_init
struct MuZeroPredictionCPU(
    Movable, ImplicitlyDestructible, Prediction,
):
    """Adapter: hidden → (softmax policy, scalar value).

    Splits the prediction output into [ACT | NUM_BINS], softmaxes the
    policy logits in Float64, decodes the value bins through the
    inverse scalar transform.
    """

    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var v_min: Float64
    var v_max: Float64

    def predict_cpu(
        mut self,
        hidden: List[Float64],
        mut policy_out: List[Float64],
    ) raises -> Float64:
        comptime B: Int = 1
        var inp_ptr = alloc[Scalar[dtype]](PredModel.IN_DIM)
        for i in range(LATENT):
            inp_ptr[i] = Scalar[dtype](hidden[i])

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(B, PredModel.IN_DIM), MutAnyOrigin
        ](inp_ptr)

        var out_ptr = alloc[Scalar[dtype]](PredModel.OUT_DIM)
        memset(out_ptr, 0, PredModel.OUT_DIM)
        var out_t = LayoutTensor[
            dtype, Layout.row_major(B, PredModel.OUT_DIM), MutAnyOrigin
        ](out_ptr)

        var params_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)

        Network[PredModel, OptType].forward[B](
            inp_t, out_t, params_t, state_t
        )

        # Softmax policy logits in Float64.
        var max_l = Float64(out_ptr[0])
        for a in range(1, ACT):
            var v = Float64(out_ptr[a])
            if v > max_l:
                max_l = v
        var sum_exp = Float64(0.0)
        for a in range(ACT):
            policy_out[a] = exp(Float64(out_ptr[a]) - max_l)
            sum_exp += policy_out[a]
        for a in range(ACT):
            policy_out[a] /= sum_exp

        var value = _decode_value_categorical[BINS](
            out_ptr + ACT, self.v_min, self.v_max
        )

        inp_ptr.free()
        out_ptr.free()

        return value


# ─── Test harness ─────────────────────────────────────────────────────────


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


def test_visit_count_parity_no_noise() raises:
    """Legacy MCTS and GenericCPUMCTS produce identical visit counts
    at the root when run on the same root_obs, same NetworkStates, and
    ``add_noise=False`` (no Dirichlet randomness in either path).
    """
    _set_seed(0x70E12025)

    var rep_state = NetworkState[RepModel, OptType]()
    rep_state.initialize[Kaiming[]]()
    var dyn_state = NetworkState[DynModel, OptType]()
    dyn_state.initialize[Kaiming[]]()
    var pred_state = NetworkState[PredModel, OptType]()
    pred_state.initialize[Kaiming[]]()

    var root_obs: List[Float64] = [0.1, -0.2, 0.3, -0.4]

    var v_min: Float64 = -5.0
    var v_max: Float64 = 5.0

    # ── Legacy MCTS ──────────────────────────────────────────────────
    var legacy = LegacyMCTS[ACT, LATENT, BINS, 16, 32](
        gamma=0.997,
        c_base=19652.0,
        c_init=1.25,
        dirichlet_alpha=0.25,
        noise_fraction=0.0,
    )
    # Legacy expects List[Scalar[dtype]] for root_obs.
    var legacy_obs = List[Scalar[dtype]](length=OBS, fill=Scalar[dtype](0))
    for i in range(OBS):
        legacy_obs[i] = Scalar[dtype](root_obs[i])

    var legacy_policy = legacy.search[
        RepModel, DynModel, PredModel, OptType, OptType, OptType,
    ](
        legacy_obs,
        rep_state,
        dyn_state,
        pred_state,
        v_min,
        v_max,
        add_noise=False,
    )

    # Snapshot legacy root visit counts.
    var legacy_visits = InlineArray[Int, ACT](uninitialized=True)
    for a in range(ACT):
        legacy_visits[a] = legacy.nodes[0].visit_count[a]

    # ── New MCTS via trait adapters ──────────────────────────────────
    var new_mcts = GenericCPUMCTS[
        ACT, LATENT, 16, 32, MuZeroPUCT[], NoNoise, SinglePlayer,
    ](gamma=0.997)

    var rep_adapter = MuZeroRepresentationCPU(
        params=rep_state.params, model_state=rep_state.model_state,
    )
    var dyn_adapter = MuZeroDynamicsCPU(
        params=dyn_state.params, model_state=dyn_state.model_state,
        v_min=v_min, v_max=v_max,
    )
    var pred_adapter = MuZeroPredictionCPU(
        params=pred_state.params, model_state=pred_state.model_state,
        v_min=v_min, v_max=v_max,
    )

    var new_policy = new_mcts.search[
        MuZeroRepresentationCPU, MuZeroDynamicsCPU, MuZeroPredictionCPU,
    ](
        rep_adapter, dyn_adapter, pred_adapter,
        root_obs, add_noise=False,
    )

    var new_visits = InlineArray[Int, ACT](uninitialized=True)
    for a in range(ACT):
        new_visits[a] = new_mcts.nodes[0].visit_count[a]

    # ── Assertions ───────────────────────────────────────────────────
    # Sum of visits must be NUM_SIMULATIONS on both sides.
    var legacy_sum: Int = 0
    var new_sum: Int = 0
    for a in range(ACT):
        legacy_sum += legacy_visits[a]
        new_sum += new_visits[a]
    assert_equal(legacy_sum, 16, "legacy root total_visits should = NUM_SIMS")
    assert_equal(new_sum, 16, "new root total_visits should = NUM_SIMS")

    # Bit-parity: per-action visit counts must match exactly.
    for a in range(ACT):
        assert_equal(
            legacy_visits[a],
            new_visits[a],
            "visit count mismatch at a="
            + String(a) + ": legacy="
            + String(legacy_visits[a]) + " new="
            + String(new_visits[a]),
        )

    # And visit-count probability distribution should match in float
    # (with one ULP slack — visit counts are integer so this is just
    # a structural sanity check on the policy normalization).
    for a in range(ACT):
        assert_true(
            _approx(legacy_policy[a], new_policy[a], tol=1e-9),
            "policy mismatch at a=" + String(a)
            + ": legacy=" + String(legacy_policy[a])
            + " new=" + String(new_policy[a]),
        )


def test_visit_count_parity_legal_mask() raises:
    """Parity holds when legal mask is in play. ``legal_mask=[False,
    True]`` forces both engines onto action 1 — both should have all
    NUM_SIMS visits on the right arm.
    """
    _set_seed(0x70E12026)

    var rep_state = NetworkState[RepModel, OptType]()
    rep_state.initialize[Kaiming[]]()
    var dyn_state = NetworkState[DynModel, OptType]()
    dyn_state.initialize[Kaiming[]]()
    var pred_state = NetworkState[PredModel, OptType]()
    pred_state.initialize[Kaiming[]]()

    var root_obs: List[Float64] = [0.5, 0.5, -0.5, -0.5]

    var legacy = LegacyMCTS[ACT, LATENT, BINS, 12, 24](
        gamma=0.99, c_base=19652.0, c_init=1.25,
        dirichlet_alpha=0.0, noise_fraction=0.0,
    )
    var legacy_obs = List[Scalar[dtype]](length=OBS, fill=Scalar[dtype](0))
    for i in range(OBS):
        legacy_obs[i] = Scalar[dtype](root_obs[i])

    var legal: List[Bool] = [False, True]
    var _legacy_policy = legacy.search[
        RepModel, DynModel, PredModel, OptType, OptType, OptType,
    ](
        legacy_obs, rep_state, dyn_state, pred_state,
        -5.0, 5.0, add_noise=False, legal_mask=legal,
    )

    var new_mcts = GenericCPUMCTS[
        ACT, LATENT, 12, 24, MuZeroPUCT[], NoNoise, SinglePlayer,
    ](gamma=0.99)

    var rep_a = MuZeroRepresentationCPU(
        params=rep_state.params, model_state=rep_state.model_state,
    )
    var dyn_a = MuZeroDynamicsCPU(
        params=dyn_state.params, model_state=dyn_state.model_state,
        v_min=-5.0, v_max=5.0,
    )
    var pred_a = MuZeroPredictionCPU(
        params=pred_state.params, model_state=pred_state.model_state,
        v_min=-5.0, v_max=5.0,
    )

    var _new_policy = new_mcts.search[
        MuZeroRepresentationCPU, MuZeroDynamicsCPU, MuZeroPredictionCPU,
    ](
        rep_a, dyn_a, pred_a, root_obs, add_noise=False, legal_mask=legal,
    )

    for a in range(ACT):
        assert_equal(
            legacy.nodes[0].visit_count[a],
            new_mcts.nodes[0].visit_count[a],
            "legal-mask visit count mismatch at a=" + String(a),
        )
    # NB: muzero-general's legal-mask convention zeroes the *prior* but
    # not the selection score. On the first simulation, PUCT exploration
    # is ``c · P(a) · √N(s) / (1 + N(s,a))`` and ``√N(s) = 0``, so the
    # prior-zero term has no effect and the tie-break still picks the
    # lowest-index action — even when it's illegal. The bit-parity
    # assertion above proves the new MCTS replicates that quirk. The
    # legal action (1) dominates from sim 2 onward.
    assert_true(
        new_mcts.nodes[0].visit_count[1] >= 11,
        "legal action 1 should claim ≥ 11 of 12 visits, got "
        + String(new_mcts.nodes[0].visit_count[1]),
    )


def main() raises:
    print("=== Phase 3 planners: CPU MCTS bit-parity vs legacy MuZero ===")
    test_visit_count_parity_no_noise()
    print("  PASS visit counts match legacy MCTS exactly (no noise)")
    test_visit_count_parity_legal_mask()
    print("  PASS visit counts match legacy MCTS exactly (legal mask)")
    print("OK")
