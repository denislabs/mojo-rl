"""C.4 — ERE recency-bias sampling tests.

Verifies the ERE branch of `GPUReplay.sample[BATCH]`:

  1. Disabled (default): sample distribution is uniform over the
     whole buffer — `pos`-anchored window has no effect.
  2. Enabled with `eta = 0.5`: at the FIRST `sample` call after
     `enable_ere(...)`, `η^k = 1.0` so `c_k = size` — distribution
     equals uniform (just rotated). At the SECOND call `η^k = 0.5`
     so `c_k = size/2` — every sampled index is within the recent
     half of the buffer.
  3. With `eta = 1.0`: ERE is a no-op (samples uniform over whole
     buffer).

Bias is verified by computing the histogram of sampled indices
relative to `pos` (the head) — recency bias should concentrate the
histogram in `[pos − c_k, pos)` (mod CAP).
"""

from std.gpu.host import DeviceContext
from std.memory import alloc
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.data.gpu_replay import GPUReplay


comptime OBS = 2
comptime ACT = 1
comptime CAP = 1_000
comptime BATCH = 256
comptime FILL_N = 800   # < CAP so wrap is not exercised in this test


def _fill(
    mut rb: GPUReplay[OBS, ACT, CAP], ctx: DeviceContext, n: Int,
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
            Scalar[DT](Float64(step)),
            nxt,
            Scalar[DT](0.0),
        )
    ctx.synchronize()


def _sampled_indices(
    mut rb: GPUReplay[OBS, ACT, CAP],
    ctx: DeviceContext,
) raises -> List[Int]:
    var mb_s = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](BATCH)
    rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)
    var h_s = alloc[Scalar[DT]](BATCH * OBS)
    ctx.enqueue_copy(h_s, mb_s)
    ctx.synchronize()
    # Recover sampled step from obs[0] (= step).
    var out = List[Int]()
    for i in range(BATCH):
        out.append(Int(Float64(h_s[i * OBS + 0])))
    return out^


def test_ere_disabled_is_uniform() raises:
    seed(42)
    var ctx = DeviceContext()
    var rb = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    _fill(rb, ctx, FILL_N)
    assert_true(not rb.ere_enabled, "ERE should be disabled by default")

    var idxs = _sampled_indices(rb, ctx)
    # With uniform sampling over 800 slots, expect roughly equal counts
    # in the lower-half [0, 400) and upper-half [400, 800).
    var lo = 0
    var hi = 0
    for i in range(BATCH):
        if idxs[i] < FILL_N // 2:
            lo += 1
        else:
            hi += 1
    # Allow generous slack (BATCH=256 has noisy bounds; expect ~128 each).
    assert_true(
        lo > BATCH // 6 and hi > BATCH // 6,
        "Uniform sampling should hit both halves: lo="
        + String(lo) + " hi=" + String(hi),
    )
    print(
        "  test_ere_disabled_is_uniform PASSED (lo=", lo, " hi=", hi, ")",
    )


def test_ere_with_eta_half_biases_recent() raises:
    """Spike ERE bias by setting eta=0.5: the second sample (η^1=0.5)
    restricts c_k to size/2 → every sampled index must be in the
    recent half `[pos − size/2, pos)`."""
    seed(42)
    var ctx = DeviceContext()
    var rb = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    _fill(rb, ctx, FILL_N)
    rb.enable_ere(eta=Scalar[DT](0.5), c_min=1, k_max=1000)

    # First sample: η^k=1.0 → c_k=size → uniform.
    var idxs_first = _sampled_indices(rb, ctx)
    assert_true(
        rb._ere_k == 1,
        "After first sample _ere_k should be 1, got "
        + String(rb._ere_k),
    )

    # Second sample: η^k=0.5 → c_k≈size/2. The recent window is
    # `[pos − c_k, pos)` = `[400, 800)` since pos=FILL_N=800.
    var idxs_second = _sampled_indices(rb, ctx)
    var pos = rb.pos
    var size = rb.size
    var c_k = size // 2  # eta=0.5 → η^1=0.5
    # All sampled indices must be in [pos − c_k, pos) mod CAP.
    var hits_recent = 0
    for i in range(BATCH):
        var step = idxs_second[i]
        # Recent window starts at (pos - c_k + CAP) % CAP.
        var window_start = (pos - c_k + CAP) % CAP
        var window_end = pos  # exclusive
        var in_window = False
        if window_start < window_end:
            if step >= window_start and step < window_end:
                in_window = True
        else:
            if step >= window_start or step < window_end:
                in_window = True
        if in_window:
            hits_recent += 1
    assert_true(
        hits_recent == BATCH,
        "ERE with eta=0.5 should land all "
        + String(BATCH) + " samples in recent half; got "
        + String(hits_recent),
    )
    print(
        "  test_ere_with_eta_half_biases_recent PASSED (",
        hits_recent, "/", BATCH, " in recent half)",
    )


def test_ere_with_eta_one_is_uniform_like() raises:
    """`eta=1.0`: η^k stays at 1.0 so `c_k = size` every call —
    distribution equals uniform-over-whole-buffer (just rotated).
    Verify both halves get hit roughly proportionally."""
    seed(42)
    var ctx = DeviceContext()
    var rb = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    _fill(rb, ctx, FILL_N)
    rb.enable_ere(eta=Scalar[DT](1.0), c_min=1, k_max=1000)
    var idxs = _sampled_indices(rb, ctx)
    var lo = 0
    var hi = 0
    for i in range(BATCH):
        if idxs[i] < FILL_N // 2:
            lo += 1
        else:
            hi += 1
    assert_true(
        lo > BATCH // 6 and hi > BATCH // 6,
        "ERE with eta=1.0 should be uniform-like; lo="
        + String(lo) + " hi=" + String(hi),
    )
    print(
        "  test_ere_with_eta_one_is_uniform_like PASSED (lo=", lo,
        " hi=", hi, ")",
    )


def test_ere_cycle_wrap() raises:
    """After `k_max` calls, `_ere_k` should reset to 0 and
    `_ere_eta_pow_k` to 1.0."""
    seed(42)
    var ctx = DeviceContext()
    var rb = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    _fill(rb, ctx, FILL_N)
    rb.enable_ere(eta=Scalar[DT](0.99), c_min=1, k_max=5)
    for _ in range(5):
        _ = _sampled_indices(rb, ctx)
    assert_true(
        rb._ere_k == 0,
        "After k_max calls _ere_k should wrap to 0; got "
        + String(rb._ere_k),
    )
    var pow_ok = (
        (Float64(rb._ere_eta_pow_k) - 1.0).__abs__() < 1e-6
    )
    assert_true(
        pow_ok,
        "After wrap _ere_eta_pow_k should reset to 1.0; got "
        + String(Float64(rb._ere_eta_pow_k)),
    )
    print("  test_ere_cycle_wrap PASSED")


def main() raises:
    print("=" * 60)
    print("C.4 ERE recency-bias tests")
    print("=" * 60)
    test_ere_disabled_is_uniform()
    test_ere_with_eta_half_biases_recent()
    test_ere_with_eta_one_is_uniform_like()
    test_ere_cycle_wrap()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
