"""Phase R.0 bit-identity gate — CriticEnsemble[CRITIC, 2] is
behaviourally byte-identical to a pair of standalone
`OnlineTargetPair[CRITIC]`s through hard-copy + polyak.

Strategy (avoids depending on Mojo's global-RNG seeding semantics):
  1. Build `CriticEnsemble[CriticNet, 2]` + two standalone
     `OnlineTargetPair[CriticNet]`s. Each call internally draws random
     init, so the four pair states differ from each other initially.
  2. Overwrite every leaf via a sequential-fill visitor — same pattern
     in `ensemble.pairs[i].online` and `standalone_i.online`, a
     different (also matched) pattern in the target_nets. After this,
     ensemble pair `i` ≡ standalone pair `i` byte-for-byte AND
     online ≠ target inside each pair (so polyak actually moves
     something, not a vacuous τ-on-equal no-op).
  3. Run K polyak τ-updates: `ensemble.soft_update_all["cpu"](tau)` vs
     `standalone_i.polyak_step["cpu"](tau)`. Both paths route to the
     same `map_params.polyak_update`, so the test gates "the
     ensemble's container indexing + loop doesn't reorder or drop a
     leaf" — failure modes that are easy to introduce.
  4. Byte-compare every leaf of online + target of pair 0 and pair 1.

CPU-only (R.0 lifecycle gate; GPU goes through the same
`polyak_step` path so the CPU gate already exercises the indexing
behaviour the GPU port needs).
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.named_params import named_params, NamedParam
from mojo_rl.nn.core.map_params import hard_copy_params
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.core.online_target_pair import OnlineTargetPair
from mojo_rl.deep_agents.redq import CriticEnsemble


comptime OBS = 3
comptime ACT = 1
comptime SA_DIM = OBS + ACT

comptime CriticNet = Sequential[
    Linear[SA_DIM, 16],
    ReLU[16],
    Linear[16, 1],
]


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _fill_params_sequential[
    M: Module,
](mut model: M, start: Float64, step: Float64) raises:
    """Overwrite every leaf parameter with a deterministic sequence:
    leaf 0 → [start, start+step, start+2·step, …]
    leaf 1 → continues from where leaf 0 ended.

    Walks `named_params` in the canonical order, so two models with the
    same module shape get byte-identical fills."""
    var ps = named_params["cpu", M](model)
    var v = start
    for i in range(len(ps)):
        ref p = ps[i]
        for k in range(p.n_elems):
            p.param_ptr[k] = Scalar[DT](v)
            v += step


def _max_abs_diff[
    M: Module,
](mut a: M, mut b: M) raises -> Float64:
    """Return max |a.param − b.param| across every leaf. 0.0 ⇔ byte-
    identical for the deterministic fills we use. Also raises on
    structural mismatch (different leaf count, n_elems, or names)."""
    var ap = named_params["cpu", M](a)
    var bp = named_params["cpu", M](b)
    if len(ap) != len(bp):
        raise Error(
            "param-count mismatch: a="
            + String(len(ap))
            + " b=" + String(len(bp))
        )
    var worst: Float64 = 0.0
    for i in range(len(ap)):
        ref pa = ap[i]
        ref pb = bp[i]
        if pa.n_elems != pb.n_elems:
            raise Error(
                "n_elems mismatch at leaf " + String(i)
                + " (a=" + String(pa.n_elems)
                + ", b=" + String(pb.n_elems) + ")"
            )
        if pa.name != pb.name:
            raise Error(
                "leaf-name mismatch at " + String(i)
                + " (a='" + pa.name + "', b='" + pb.name + "')"
            )
        for k in range(pa.n_elems):
            var d = Float64(pa.param_ptr[k]) - Float64(pb.param_ptr[k])
            if d < 0.0:
                d = -d
            if d > worst:
                worst = d
    return worst


# ─────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────


def test_critic_ensemble_n2_bit_identity() raises:
    print("--- CriticEnsemble[CriticNet, 2] vs twin OnlineTargetPair ---")

    # 1. Construct both layouts. Both run Xavier init internally; their
    #    starting param values DIFFER (independent random draws).
    var ensemble = CriticEnsemble[CriticNet, 2].make["cpu", Xavier]()
    var standalone0 = OnlineTargetPair[CriticNet].make["cpu", Xavier]()
    var standalone1 = OnlineTargetPair[CriticNet].make["cpu", Xavier]()

    # 2. Overwrite both layouts with the same deterministic fills.
    #    Pair 0 online ≡ standalone0 online, pair 0 target ≡ standalone0
    #    target — but online ≠ target within each pair so polyak moves
    #    something.
    _fill_params_sequential[CriticNet](ensemble.pairs[0].online, 0.0, 0.001)
    _fill_params_sequential[CriticNet](standalone0.online, 0.0, 0.001)
    _fill_params_sequential[CriticNet](ensemble.pairs[0].target_net, 1.0, -0.0007)
    _fill_params_sequential[CriticNet](standalone0.target_net, 1.0, -0.0007)

    _fill_params_sequential[CriticNet](ensemble.pairs[1].online, -0.5, 0.0013)
    _fill_params_sequential[CriticNet](standalone1.online, -0.5, 0.0013)
    _fill_params_sequential[CriticNet](ensemble.pairs[1].target_net, 0.5, -0.0009)
    _fill_params_sequential[CriticNet](standalone1.target_net, 0.5, -0.0009)

    # Post-fill sanity: byte-identical AND online ≠ target.
    var d0_on_init = _max_abs_diff[CriticNet](
        ensemble.pairs[0].online, standalone0.online
    )
    var d0_tg_init = _max_abs_diff[CriticNet](
        ensemble.pairs[0].target_net, standalone0.target_net
    )
    var d1_on_init = _max_abs_diff[CriticNet](
        ensemble.pairs[1].online, standalone1.online
    )
    var d1_tg_init = _max_abs_diff[CriticNet](
        ensemble.pairs[1].target_net, standalone1.target_net
    )
    var d0_diverge = _max_abs_diff[CriticNet](
        ensemble.pairs[0].online, ensemble.pairs[0].target_net
    )
    var d1_diverge = _max_abs_diff[CriticNet](
        ensemble.pairs[1].online, ensemble.pairs[1].target_net
    )
    print("  post-fill max|e[0].online - s0.online|     =", d0_on_init)
    print("  post-fill max|e[0].target - s0.target|     =", d0_tg_init)
    print("  post-fill max|e[1].online - s1.online|     =", d1_on_init)
    print("  post-fill max|e[1].target - s1.target|     =", d1_tg_init)
    print("  post-fill max|e[0].online - e[0].target|   =", d0_diverge)
    print("  post-fill max|e[1].online - e[1].target|   =", d1_diverge)
    assert_true(d0_on_init == 0.0, "e[0].online ≡ s0.online after fill")
    assert_true(d0_tg_init == 0.0, "e[0].target ≡ s0.target after fill")
    assert_true(d1_on_init == 0.0, "e[1].online ≡ s1.online after fill")
    assert_true(d1_tg_init == 0.0, "e[1].target ≡ s1.target after fill")
    assert_true(
        d0_diverge > 0.1,
        "online vs target inside pair 0 must differ (else polyak is a no-op)",
    )
    assert_true(
        d1_diverge > 0.1,
        "online vs target inside pair 1 must differ (else polyak is a no-op)",
    )

    # 3. Run K polyak τ-updates. Same τ, no ctx (CPU).
    var tau = Scalar[DT](0.3)
    comptime K = 5
    for _ in range(K):
        ensemble.soft_update_all["cpu"](tau)
        standalone0.polyak_step["cpu"](tau)
        standalone1.polyak_step["cpu"](tau)

    # 4. Byte-compare every leaf.
    var d0_on = _max_abs_diff[CriticNet](
        ensemble.pairs[0].online, standalone0.online
    )
    var d0_tg = _max_abs_diff[CriticNet](
        ensemble.pairs[0].target_net, standalone0.target_net
    )
    var d1_on = _max_abs_diff[CriticNet](
        ensemble.pairs[1].online, standalone1.online
    )
    var d1_tg = _max_abs_diff[CriticNet](
        ensemble.pairs[1].target_net, standalone1.target_net
    )
    print("  post-K=" + String(K) + " max|e[0].online - s0.online|  =", d0_on)
    print("  post-K=" + String(K) + " max|e[0].target - s0.target|  =", d0_tg)
    print("  post-K=" + String(K) + " max|e[1].online - s1.online|  =", d1_on)
    print("  post-K=" + String(K) + " max|e[1].target - s1.target|  =", d1_tg)
    # Polyak only writes target — online should be untouched (== 0.0
    # already, but assert as a regression gate against future bugs that
    # accidentally also mutate online).
    assert_true(d0_on == 0.0, "polyak must not touch e[0].online")
    assert_true(d1_on == 0.0, "polyak must not touch e[1].online")
    # Target byte-identity is the actual ensemble-vs-twin gate.
    assert_true(d0_tg == 0.0, "ensemble pair 0 target ≡ standalone 0 target")
    assert_true(d1_tg == 0.0, "ensemble pair 1 target ≡ standalone 1 target")

    # 5. Also confirm: after polyak, target really moved toward online
    #    (i.e. the post-polyak target-vs-online gap is smaller than the
    #    initial 0.1+ gap). This guards against the test silently
    #    passing if `polyak_update` were stubbed out.
    var moved_0 = _max_abs_diff[CriticNet](
        ensemble.pairs[0].online, ensemble.pairs[0].target_net
    )
    var moved_1 = _max_abs_diff[CriticNet](
        ensemble.pairs[1].online, ensemble.pairs[1].target_net
    )
    print("  post-polyak max|online - target| pair 0    =", moved_0)
    print("  post-polyak max|online - target| pair 1    =", moved_1)
    assert_true(
        moved_0 < d0_diverge,
        "pair 0 target must have moved toward online (else polyak no-op'd)",
    )
    assert_true(
        moved_1 < d1_diverge,
        "pair 1 target must have moved toward online (else polyak no-op'd)",
    )

    print("PASS — CriticEnsemble[CRITIC, 2] bit-identical to twin pair.")


def main() raises:
    test_critic_ensemble_n2_bit_identity()
