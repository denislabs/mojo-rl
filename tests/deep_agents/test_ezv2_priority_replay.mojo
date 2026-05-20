"""Phase-2 priority-replay wiring test.

Verifies that `train_step` actually does priority-weighted sampling and
updates per-transition priorities (paper App. A "Priority
Precalculation"):

  1. Before training, all stored priorities equal `agent.max_priority`
     (the default-on-write value), so sampling is approximately uniform.
  2. After a `train_step`, the priorities at the sampled buffer indices
     have been overwritten with the per-sample value-CE loss; positions
     with high TD error end up with much higher priority than untouched
     positions.
  3. Forcing one buffer index to have an extreme priority and
     zero-ing the rest pins all subsequent samples to that index — the
     smoking gun that the priority weighting drives sampling, not just
     the legal-window mask.

If these checks fail the priority array is dead-weight (uniform
sampling silently bypasses it).
"""

from std.math import abs
from std.random import seed
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.nn.constants import dtype


def _expect(
    cond: Bool,
    label: String,
    mut passed: Int,
    mut total: Int,
):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def main():
    print("=== EZ-V2 priority-replay wiring test ===")
    var passed = 0
    var total = 0

    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=32,
        HIDDEN=32,
        PROJ=64,
        PRED_BOTTLENECK=32,
        BINS=21,
        BS=8,
        K_UNROLL=3,
        N_TD=5,
        SIMS=8,
        NODES=32,
        K_GUMBEL=2,
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
    )
    var env = CartPoleEnv[DType.float32]()

    var num_episodes = 50
    var max_steps_per_ep = 60
    for _ep in range(num_episodes):
        var obs = env.reset_obs_list()
        for _step in range(max_steps_per_ep):
            var result = agent.select_action(obs, training=True)
            var action = result[0]
            var policy = result[1]
            var root_value = result[2]
            var step_result = env.step_obs(action)
            var next_obs = step_result[0].copy()
            var reward = Float64(step_result[1])
            var done = step_result[2]
            agent.store_transition(
                obs, action, reward, policy, root_value, done
            )
            obs = next_obs^
            if done:
                break

    var n = agent.state.buffer.size
    print()
    print("--- After rollout ---")
    print("    buf size           =", n)
    print("    agent.max_priority =", agent.max_priority)

    # ── 1. Pre-training: all priorities = max_priority (the default) ─────
    var pre_min = Float64(1e18)
    var pre_max = Float64(-1e18)
    for i in range(n):
        var p = Float64(agent.state.priorities[i])
        if p < pre_min:
            pre_min = p
        if p > pre_max:
            pre_max = p
    print("    pre-training priority range  =", pre_min, "..", pre_max)

    _expect(
        pre_max == pre_min and pre_max == 1.0,
        "fresh transitions all stamped with max_priority=1.0",
        passed, total,
    )

    # ── 2. Run one train_step → some priorities get overwritten ──────────
    var t = agent.train_step()
    var post_train_max_priority = agent.max_priority
    print()
    print("--- After 1 train_step ---")
    print("    L_total                          =", t[0])
    print("    agent.max_priority               =", post_train_max_priority)

    var n_updated = 0
    var min_updated = Float64(1e18)
    var max_updated = Float64(-1e18)
    for i in range(n):
        var p = Float64(agent.state.priorities[i])
        # An entry that was overwritten by train_step has a different
        # value than the default 1.0. Most likely it'll differ by far
        # more than float noise.
        var d = p - 1.0
        if d < 0:
            d = -d
        if d > 1e-3:
            n_updated += 1
            if p < min_updated:
                min_updated = p
            if p > max_updated:
                max_updated = p
    print("    indices updated by train_step    =", n_updated)
    print("    updated priority range           =", min_updated, "..", max_updated)

    _expect(
        n_updated > 0,
        "train_step overwrote at least one priority entry",
        passed, total,
    )
    # BATCH=8 windows at K=3 means up to 8 root indices get updated.
    # Most likely it's exactly 8 (no duplicate samples), but with
    # priority-uniform first-step sampling there could be repeats.
    _expect(
        n_updated <= Config.batch_size,
        "no more priorities updated than BATCH_SIZE (= 8)",
        passed, total,
    )

    # ── 3. Force a single index extreme + zero others; confirm that
    #       index gets sampled deterministically. ─────────────────────────
    # Pick the oldest valid (non-boundary) start index.
    var buf_ptr = agent.state.buffer.ptr
    comptime CAP = 50000
    var oldest = (buf_ptr - n + CAP) % CAP
    var pinned_idx = -1
    for offset in range(n - Config.unroll_steps):
        var idx = (oldest + offset) % CAP
        var ok = True
        for k in range(Config.unroll_steps):
            var iidx = (idx + k) % CAP
            if Float64(agent.state.buffer.dones[iidx]) > 0.5:
                ok = False
                break
        if ok:
            pinned_idx = idx
            break
    print("    pinned_idx (forced extreme)      =", pinned_idx)
    _expect(
        pinned_idx >= 0,
        "found at least one valid window-start in buffer",
        passed, total,
    )

    # Zero every priority, then make the pinned index huge.
    for i in range(n):
        agent.state.priorities[i] = Scalar[dtype](0.0)
    agent.state.priorities[pinned_idx] = Scalar[dtype](1.0e6)
    # Phase 1 (sum-tree PER): the sum-tree is the source of truth for
    # sampling weights now; rebuild it from the freshly-mutated
    # `priorities` + `dones` state so the next train_step honours the
    # pinned weight.
    agent.state.rebuild_priority_tree()

    # Run another train_step and see which indices got overwritten.
    seed(42)
    var t2 = agent.train_step()
    print("    L_total under pinned priority    =", t2[0])

    # The pinned index should have its priority overwritten (as part
    # of the BATCH=8 sample); the other priorities should remain 0.
    var pinned_priority_after = Float64(
        agent.state.priorities[pinned_idx]
    )
    var n_other_nonzero = 0
    for i in range(n):
        if i == pinned_idx:
            continue
        if Float64(agent.state.priorities[i]) > 1e-6:
            n_other_nonzero += 1

    print(
        "    pinned priority post-train       =", pinned_priority_after
    )
    print(
        "    other-index nonzero priorities   =", n_other_nonzero
    )
    _expect(
        pinned_priority_after != 1.0e6,
        "pinned index's priority was overwritten by train_step (sampled)",
        passed, total,
    )
    # Every sample in the BATCH should have hit the pinned index, since
    # every other index has priority 0. So no OTHER index should have
    # been touched.
    _expect(
        n_other_nonzero == 0,
        "with one extreme priority, no other index was sampled",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
