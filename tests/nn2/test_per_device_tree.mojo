"""Part A (docs/DEVICE_PER_TREE_PLAN.md) — device-resident PER sum-tree
unit tests, run against the DEFAULT `GPUPrioritizedReplay` (DEVICE_TREE_
= True). The host-tree oracle path is covered by `test_per_replay.mojo`.

Verifies (plan §5):
  1. Insert leaves (single `add` path): D2H the device tree and compare
     EVERY node against a host reference tree rebuilt from the same leaf
     priorities (≤ 1e-4). CAP=100 (non-power-of-two) stresses the
     level-clamped propagate kernel.
  2. Sample: indices in range; uniform priorities ⇒ all normalized IS
     weights == 1 (closed form (N·P)^{−β} with P = 1/N).
  3. `update_priorities` with known td_errors: updated leaves match
     `(|TD| + ε)^α` (sequential last-duplicate-wins, like the host
     loop), internal sums match the reference rebuild, and
     `max_priority_dev` tracks the raw `|TD| + ε` ceiling.
  4. New inserts after a priority spike get `max_priority^α` (device
     ceiling read), and a follow-up sample concentrates on the spiked
     slot.

RNG note: the device tree samples via Philox, the host tree via
`random_float64` — statistically equivalent, NOT bit-identical. The
convergence gate for the swap is the Rainbow Pong run (NVIDIA).

Run:
    pixi run -e apple mojo run -I . tests/nn2/test_per_device_tree.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs as fabs, pow as fpow
from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.data.per_replay import GPUPrioritizedReplay


comptime OBS = 2
comptime ACT = 1
comptime CAP = 100  # non-power-of-two: stresses the propagate level clamp
comptime BATCH = 32
comptime TREE_N = 2 * CAP - 1
comptime ALPHA = Float64(0.6)
comptime BETA = Float64(0.4)
comptime EPS = Float64(1e-6)

comptime DevTreePER = GPUPrioritizedReplay[OBS, ACT, CAP]  # default = device


def _new_rb(ctx: DeviceContext) raises -> DevTreePER:
    return DevTreePER.new(
        ctx,
        alpha=Scalar[DT](ALPHA),
        beta=Scalar[DT](BETA),
        epsilon=Scalar[DT](EPS),
        batch_capacity=BATCH,
    )


def _fill_buffer(
    mut rb: DevTreePER, ctx: DeviceContext, n: Int
) raises:
    var obs = alloc[Scalar[DT]](OBS)
    var act = alloc[Scalar[DT]](ACT)
    var nxt = alloc[Scalar[DT]](OBS)
    for step in range(n):
        obs[0] = Scalar[DT](Float64(step))
        obs[1] = Scalar[DT](Float64(step) + 0.1)
        act[0] = Scalar[DT](Float64(step) * 10.0)
        nxt[0] = Scalar[DT](Float64(step) + 1.0)
        nxt[1] = Scalar[DT](Float64(step) + 1.1)
        rb.add(
            ctx, obs, act,
            Scalar[DT](Float64(step) + 1000.0),
            nxt,
            Scalar[DT](0.0),
        )
    ctx.synchronize()


def _read_tree(
    rb: DevTreePER, ctx: DeviceContext
) raises -> List[Float64]:
    """D2H the device tree (test-only path)."""
    var h = alloc[Scalar[DT]](TREE_N)
    ctx.enqueue_copy(h, rb.tree_dev)
    ctx.synchronize()
    var out = List[Float64](capacity=TREE_N)
    for i in range(TREE_N):
        out.append(Float64(h[i]))
    h.free()
    return out^


def _reference_rebuild(ref leaves: List[Float64]) -> List[Float64]:
    """Host reference: internal node i = child(2i+1) + child(2i+2)."""
    var t = List[Float64](length=TREE_N, fill=0.0)
    for i in range(CAP):
        t[CAP - 1 + i] = leaves[i]
    for i in range(CAP - 2, -1, -1):
        t[i] = t[2 * i + 1] + t[2 * i + 2]
    return t^


def _assert_tree_matches(
    ref dev: List[Float64], ref reference: List[Float64], name: String
) raises:
    for i in range(TREE_N):
        assert_true(
            fabs(dev[i] - reference[i]) <= 1e-4 * (1.0 + fabs(reference[i])),
            name + ": tree node " + String(i) + " mismatch: dev="
            + String(dev[i]) + " ref=" + String(reference[i]),
        )


def test_insert_propagate_matches_reference() raises:
    var ctx = DeviceContext()
    var rb = _new_rb(ctx)
    var n_adds = 60
    _fill_buffer(rb, ctx, n_adds)

    var dev = _read_tree(rb, ctx)
    var leaves = List[Float64](length=CAP, fill=0.0)
    var p0 = fpow(1.0, ALPHA)  # max_priority starts at 1.0
    for i in range(n_adds):
        leaves[i] = p0
    var reference = _reference_rebuild(leaves)
    _assert_tree_matches(dev, reference, "insert")
    assert_true(
        fabs(dev[0] - Float64(n_adds) * p0) < 1e-3,
        "root should be n_adds; got " + String(dev[0]),
    )
    print(
        "  test_insert_propagate_matches_reference PASSED (root=",
        dev[0], ")",
    )


def test_sample_indices_and_uniform_weights() raises:
    var ctx = DeviceContext()
    var rb = _new_rb(ctx)
    var n_adds = 60
    _fill_buffer(rb, ctx, n_adds)

    var mb_s = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](BATCH)
    rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)

    var h_idx = alloc[Int32](BATCH)
    var h_w = alloc[Scalar[DT]](BATCH)
    var h_s = alloc[Scalar[DT]](BATCH * OBS)
    var h_r = alloc[Scalar[DT]](BATCH)
    ctx.enqueue_copy(h_idx, rb.base.indices)
    ctx.enqueue_copy(h_w, rb.weights)
    ctx.enqueue_copy(h_s, mb_s)
    ctx.enqueue_copy(h_r, mb_r)
    ctx.synchronize()

    for i in range(BATCH):
        var idx = Int(h_idx[i])
        assert_true(
            idx >= 0 and idx < n_adds,
            "Sampled idx " + String(idx) + " out of range",
        )
        # Uniform priorities → P = 1/N → w = (N·P)^{−β} = 1 for every
        # lane, and the normalizer is 1 → all weights ≈ 1.
        assert_true(
            fabs(Float64(h_w[i]) - 1.0) < 1e-3,
            "Uniform-priority IS weight should be 1; got "
            + String(Float64(h_w[i])),
        )
        # Gather content maps back to the stored step.
        assert_true(
            fabs(Float64(h_s[i * OBS]) - Float64(idx)) < 1e-3,
            "Gathered obs[0] mismatch at lane " + String(i),
        )
        assert_true(
            fabs(Float64(h_r[i]) - (Float64(idx) + 1000.0)) < 1e-3,
            "Gathered reward mismatch at lane " + String(i),
        )
    print("  test_sample_indices_and_uniform_weights PASSED")


def test_update_priorities_matches_formula() raises:
    var ctx = DeviceContext()
    var rb = _new_rb(ctx)
    var n_adds = 60
    _fill_buffer(rb, ctx, n_adds)

    var mb_s = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](BATCH)
    rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)

    var h_idx = alloc[Int32](BATCH)
    ctx.enqueue_copy(h_idx, rb.base.indices)
    ctx.synchronize()

    # Known signed TD errors, varied per lane.
    var td_host = alloc[Scalar[DT]](BATCH)
    for i in range(BATCH):
        var v = 0.25 * Float64(i + 1)
        td_host[i] = Scalar[DT](v if i % 2 == 0 else -v)
    var td_dev = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_copy(td_dev, td_host)
    rb.update_priorities[BATCH](ctx, td_dev)

    var dev = _read_tree(rb, ctx)

    # Host reference: leaves start at 1.0 for live slots, then the lane
    # loop applies sequentially (last duplicate wins — the device kernel
    # mirrors this).
    var leaves = List[Float64](length=CAP, fill=0.0)
    for i in range(n_adds):
        leaves[i] = fpow(1.0, ALPHA)
    var max_raw = 1.0
    for i in range(BATCH):
        var raw = 0.25 * Float64(i + 1) + EPS
        if raw > max_raw:
            max_raw = raw
        leaves[Int(h_idx[i])] = fpow(raw, ALPHA)
    var reference = _reference_rebuild(leaves)
    _assert_tree_matches(dev, reference, "update")

    # Device max-priority ceiling tracks raw |TD| + ε.
    var h_max = alloc[Scalar[DT]](1)
    ctx.enqueue_copy(h_max, rb.max_priority_dev)
    ctx.synchronize()
    assert_true(
        fabs(Float64(h_max[0]) - max_raw) < 1e-4 * max_raw,
        "max_priority_dev mismatch: got " + String(Float64(h_max[0]))
        + " expected " + String(max_raw),
    )
    print("  test_update_priorities_matches_formula PASSED")


def test_spike_dominates_and_new_inserts_get_max() raises:
    var ctx = DeviceContext()
    var rb = _new_rb(ctx)
    var n_adds = 60
    _fill_buffer(rb, ctx, n_adds)

    var mb_s = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](BATCH)
    rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)

    var h_idx = alloc[Int32](BATCH)
    ctx.enqueue_copy(h_idx, rb.base.indices)
    ctx.synchronize()
    var target_idx = Int(h_idx[0])

    # Spike lane 0 with a huge TD error.
    var td_host = alloc[Scalar[DT]](BATCH)
    for i in range(BATCH):
        td_host[i] = Scalar[DT](0.0)
    td_host[0] = Scalar[DT](1000.0)
    var td_dev = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_copy(td_dev, td_host)
    rb.update_priorities[BATCH](ctx, td_dev)

    # Caveat: if lane 0's leaf reappears later in the batch, the
    # last-duplicate-wins rule overwrites the spike — skip the dominance
    # check in that (rare) case but still validate the insert rule.
    var spiked = True
    for i in range(1, BATCH):
        if Int(h_idx[i]) == target_idx:
            spiked = False

    if spiked:
        rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)
        ctx.enqueue_copy(h_idx, rb.base.indices)
        ctx.synchronize()
        var hits = 0
        for i in range(BATCH):
            if Int(h_idx[i]) == target_idx:
                hits += 1
        # Spike weight (1000+ε)^0.6 ≈ 63.2 vs ~59 background units →
        # ~52% of the mass; stratified sampling should land ≥ half - 4
        # of the lanes there.
        assert_true(
            hits >= BATCH // 2 - 4,
            "Spiked leaf should dominate sampling; got "
            + String(hits) + "/" + String(BATCH),
        )

    # New insert (next `add`) takes the bumped ceiling: leaf at the
    # write position must be (1000 + ε)^α — read from max_priority_dev
    # ON DEVICE by the leaf-set kernel.
    var write_pos = rb.base.pos
    var obs = alloc[Scalar[DT]](OBS)
    var act = alloc[Scalar[DT]](ACT)
    var nxt = alloc[Scalar[DT]](OBS)
    obs[0] = Scalar[DT](7.0)
    obs[1] = Scalar[DT](7.0)
    act[0] = Scalar[DT](7.0)
    nxt[0] = Scalar[DT](7.0)
    nxt[1] = Scalar[DT](7.0)
    rb.add(ctx, obs, act, Scalar[DT](7.0), nxt, Scalar[DT](0.0))

    var dev = _read_tree(rb, ctx)
    var expected = fpow(1000.0 + EPS, ALPHA)
    var got = dev[CAP - 1 + write_pos]
    assert_true(
        fabs(got - expected) < 1e-3 * expected,
        "New insert should get max_priority^α: got " + String(got)
        + " expected " + String(expected),
    )
    print(
        "  test_spike_dominates_and_new_inserts_get_max PASSED",
        "(dominance checked:", spiked, ")",
    )


def main() raises:
    print("=" * 60)
    print("Part A — device-resident PER sum-tree unit tests")
    print("=" * 60)
    test_insert_propagate_matches_reference()
    test_sample_indices_and_uniform_weights()
    test_update_priorities_matches_formula()
    test_spike_dominates_and_new_inserts_get_max()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
