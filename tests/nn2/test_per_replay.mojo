"""C.3 — Prioritized Experience Replay unit tests.

Verifies `GPUPrioritizedReplay[OBS=2, ACT=1, CAP=64]`:

  1. Fresh buffer: tree empty, sample raises.
  2. Add N transitions: tree leaves match `max_priority^α` for each;
     tree root equals N · `max_priority^α`.
  3. Sample BATCH=16: indices land in `[0, N)`; gather produces valid
     content (sampled rows map back to stored values via obs[0] → step).
  4. After `update_priorities`, the sum-tree leaves at the updated
     indices reflect the new `(|TD| + ε)^α` priorities, and a follow-up
     sample preferentially picks the high-priority slots.

Run:
    pixi run -e apple mojo run -I . tests/nn2/test_per_replay.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs as fabs, pow as fpow
from std.memory import alloc
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.data.per_replay import GPUPrioritizedReplay


comptime OBS = 2
comptime ACT = 1
comptime CAP = 64
comptime BATCH = 16
comptime ALPHA = Float64(0.6)
comptime EPS = Float64(1e-6)

# This file tests the HOST sum-tree path (the debugging / A-B oracle) —
# it asserts host-internal state (`tree`, `_host_indices`,
# `_host_weights`). The device-resident tree (now the default,
# `DEVICE_TREE_=True`) is covered by `test_per_device_tree.mojo`.
comptime HostTreePER = GPUPrioritizedReplay[
    OBS, ACT, CAP, DT, False
]


def _approx(a: Scalar[DT], b: Float64) -> Bool:
    return fabs(Float64(a) - b) < 1e-3


def _fill_buffer(
    mut rb: HostTreePER,
    ctx: DeviceContext,
    n: Int,
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


def test_fresh_buffer() raises:
    var ctx = DeviceContext()
    var rb = HostTreePER.new(
        ctx,
        alpha=Scalar[DT](ALPHA),
        beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](EPS),
        batch_capacity=BATCH,
    )
    assert_true(rb.base.size == 0, "Fresh buffer size should be 0")
    assert_true(rb.base.pos == 0, "Fresh buffer pos should be 0")
    assert_true(
        _approx(rb._tree_total(), 0.0),
        "Fresh tree total should be 0; got " + String(Float64(rb._tree_total())),
    )
    print("  test_fresh_buffer PASSED")


def test_add_initializes_tree_leaves() raises:
    var ctx = DeviceContext()
    var rb = HostTreePER.new(
        ctx,
        alpha=Scalar[DT](ALPHA),
        beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](EPS),
        batch_capacity=BATCH,
    )
    var n_adds = 32
    _fill_buffer(rb, ctx, n_adds)

    # Every leaf in [0, n_adds) should be max_priority^alpha.
    # max_priority starts at 1.0 → leaf value = 1.0^alpha = 1.0.
    var expected_leaf = fpow(1.0, ALPHA)
    for i in range(n_adds):
        var leaf_value = rb.tree[i + CAP - 1]
        assert_true(
            _approx(leaf_value, expected_leaf),
            "Leaf " + String(i) + " should be "
            + String(expected_leaf) + ", got "
            + String(Float64(leaf_value)),
        )
    # Leaves [n_adds, CAP) should still be 0.
    for i in range(n_adds, CAP):
        var leaf_value = rb.tree[i + CAP - 1]
        assert_true(
            _approx(leaf_value, 0.0),
            "Empty leaf " + String(i) + " should be 0",
        )
    # Root = sum of all leaves = n_adds * expected_leaf.
    var expected_total = Float64(n_adds) * expected_leaf
    assert_true(
        _approx(rb._tree_total(), expected_total),
        "Tree root mismatch: got "
        + String(Float64(rb._tree_total()))
        + " expected " + String(expected_total),
    )
    print(
        "  test_add_initializes_tree_leaves PASSED (",
        n_adds, " leaves, total=",
        rb._tree_total(), ")",
    )


def test_sample_returns_valid_indices() raises:
    seed(42)
    var ctx = DeviceContext()
    var rb = HostTreePER.new(
        ctx,
        alpha=Scalar[DT](ALPHA),
        beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](EPS),
        batch_capacity=BATCH,
    )
    _fill_buffer(rb, ctx, 48)

    var mb_s = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](BATCH)
    rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)
    ctx.synchronize()

    # Indices live on device; also accessible host-side via _host_indices.
    for i in range(BATCH):
        var idx = Int(rb._host_indices[i])
        assert_true(
            idx >= 0 and idx < 48,
            "Sampled idx " + String(idx) + " out of range [0, 48)",
        )

    # IS weights should be normalised so max ≤ 1.0.
    var max_w = Float64(0.0)
    for i in range(BATCH):
        var w = Float64(rb._host_weights[i])
        if w > max_w:
            max_w = w
        assert_true(
            w > 0.0 and w <= 1.0 + 1e-5,
            "Weight " + String(i) + " out of (0, 1]: " + String(w),
        )
    assert_true(
        fabs(max_w - 1.0) < 1e-4,
        "max IS weight should be 1.0 after normalisation; got " + String(max_w),
    )

    # Verify gather content: sampled rows should map back to valid steps.
    var h_s = alloc[Scalar[DT]](BATCH * OBS)
    var h_r = alloc[Scalar[DT]](BATCH)
    ctx.enqueue_copy(h_s, mb_s)
    ctx.enqueue_copy(h_r, mb_r)
    ctx.synchronize()
    for i in range(BATCH):
        var idx = Int(rb._host_indices[i])
        var expected_obs0 = Float64(idx)
        assert_true(
            _approx(h_s[i * OBS + 0], expected_obs0),
            "Gathered obs[0] mismatch at i=" + String(i)
            + ": got " + String(Float64(h_s[i * OBS + 0]))
            + " expected " + String(expected_obs0),
        )
        var expected_r = Float64(idx) + 1000.0
        assert_true(
            _approx(h_r[i], expected_r),
            "Gathered reward mismatch at i=" + String(i)
            + ": got " + String(Float64(h_r[i]))
            + " expected " + String(expected_r),
        )
    print("  test_sample_returns_valid_indices PASSED")


def test_update_priorities_shifts_sampling() raises:
    """After updating priorities so a single slot dominates, the next
    sample should return that slot in (almost) every BATCH lane."""
    seed(42)
    var ctx = DeviceContext()
    var rb = HostTreePER.new(
        ctx,
        alpha=Scalar[DT](ALPHA),
        beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](EPS),
        batch_capacity=BATCH,
    )
    _fill_buffer(rb, ctx, 48)

    # First sample → produces self._host_indices; we'll spike priority
    # of the first sampled index with a large TD error.
    var mb_s = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](BATCH)
    rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)
    ctx.synchronize()

    var target_idx = Int(rb._host_indices[0])

    # Build TD errors: lane 0 huge, others zero.
    var td_host = alloc[Scalar[DT]](BATCH)
    for i in range(BATCH):
        td_host[i] = Scalar[DT](0.0)
    td_host[0] = Scalar[DT](1000.0)  # huge TD error on lane 0
    var td_dev = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_copy(td_dev, td_host)
    rb.update_priorities[BATCH](ctx, td_dev)

    # The leaf at target_idx should now dominate the tree.
    var leaf_value = rb.tree[target_idx + CAP - 1]
    var expected_leaf = fpow(1000.0 + EPS, ALPHA)
    assert_true(
        _approx(leaf_value, expected_leaf),
        "Updated leaf at " + String(target_idx) + " mismatch: got "
        + String(Float64(leaf_value)) + " expected "
        + String(expected_leaf),
    )

    # Subsequent sample: count how many BATCH lanes land on target_idx.
    rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)
    ctx.synchronize()
    var hits = 0
    for i in range(BATCH):
        if Int(rb._host_indices[i]) == target_idx:
            hits += 1
    # With ~63 priority units total + 1 of value 1000^0.6 ≈ 63.1 at
    # the spiked leaf, the spike's weight is ~50% of total. Stratified
    # sampling spreads picks across BATCH segments; expect at least
    # ~half of lanes to land on the target.
    assert_true(
        hits >= BATCH // 2,
        "Spiked leaf should dominate sampling; got "
        + String(hits) + "/" + String(BATCH) + " hits",
    )
    # max_priority should track the new ceiling (raw |TD|+eps).
    assert_true(
        _approx(rb.max_priority, 1000.0 + EPS),
        "max_priority should track new ceiling; got "
        + String(Float64(rb.max_priority)),
    )
    print(
        "  test_update_priorities_shifts_sampling PASSED (",
        hits, "/", BATCH, " hits on spiked slot )",
    )


def main() raises:
    print("=" * 60)
    print("C.3 PER unit tests")
    print("=" * 60)
    test_fresh_buffer()
    test_add_initializes_tree_leaves()
    test_sample_returns_valid_indices()
    test_update_priorities_shifts_sampling()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
