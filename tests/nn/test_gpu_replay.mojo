"""C.1 — GPUReplay standalone unit test.

Verifies GPUReplay[OBS=3, ACT=1, CAP=100] in isolation:

  1. Empty buffer: size=0, pos=0.
  2. Add 64 transitions: size advances, content gathers correctly.
  3. Add over CAP: size saturates at CAP, pos wraps.
  4. Sample BATCH=32: outputs match stored content at sampled indices.

Run:
    pixi run mojo run -I . tests/nn/test_gpu_replay.mojo
"""

from std.gpu.host import DeviceContext
from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.data.gpu_replay import GPUReplay


comptime OBS = 3
comptime ACT = 1
comptime CAP = 100
comptime BATCH = 32


def _make_obs(out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin], step: Int):
    """Encode `step` into a known pattern so we can verify which slot
    was sampled. obs[d] = step * 10 + d."""
    for d in range(OBS):
        out_ptr[d] = Scalar[DT](Float64(step) * 10.0 + Float64(d))


def _make_nxt(out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin], step: Int):
    for d in range(OBS):
        out_ptr[d] = Scalar[DT](Float64(step) * 10.0 + Float64(d) + 0.5)


def _make_act(out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin], step: Int):
    for j in range(ACT):
        out_ptr[j] = Scalar[DT](Float64(step) * 100.0 + Float64(j))


def test_empty_buffer() raises:
    var ctx = DeviceContext()
    var rb = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    assert_true(rb.size == 0, "Empty buffer size should be 0")
    assert_true(rb.pos == 0, "Empty buffer pos should be 0")
    assert_true(not rb.is_ready[BATCH](), "Empty buffer is not ready")
    print("  test_empty_buffer PASSED")


def test_add_and_size_tracking() raises:
    var ctx = DeviceContext()
    var rb = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    var obs_p = alloc[Scalar[DT]](OBS)
    var act_p = alloc[Scalar[DT]](ACT)
    var nxt_p = alloc[Scalar[DT]](OBS)

    var n_adds = 64
    for step in range(n_adds):
        _make_obs(obs_p, step)
        _make_act(act_p, step)
        _make_nxt(nxt_p, step)
        rb.add(
            ctx, obs_p, act_p,
            Scalar[DT](Float64(step) + 1000.0),
            nxt_p,
            Scalar[DT](Float64(step % 7 == 0)),  # deterministic done pattern
        )
    ctx.synchronize()
    assert_true(
        rb.size == n_adds,
        "After " + String(n_adds) + " adds size should equal n_adds, got " + String(rb.size),
    )
    assert_true(
        rb.pos == n_adds % CAP,
        "pos should be n_adds % CAP, got " + String(rb.pos),
    )
    assert_true(rb.is_ready[BATCH](), "Buffer should be ready after enough adds")
    print("  test_add_and_size_tracking PASSED (size=", rb.size, ", pos=", rb.pos, ")")


def test_circular_wraparound() raises:
    var ctx = DeviceContext()
    var rb = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    var obs_p = alloc[Scalar[DT]](OBS)
    var act_p = alloc[Scalar[DT]](ACT)
    var nxt_p = alloc[Scalar[DT]](OBS)

    # Add CAP + 25 transitions; verify wraparound semantics.
    var n_adds = CAP + 25
    for step in range(n_adds):
        _make_obs(obs_p, step)
        _make_act(act_p, step)
        _make_nxt(nxt_p, step)
        rb.add(
            ctx, obs_p, act_p,
            Scalar[DT](Float64(step)),
            nxt_p,
            Scalar[DT](0.0),
        )
    ctx.synchronize()
    assert_true(
        rb.size == CAP,
        "After overflow size should saturate at CAP, got " + String(rb.size),
    )
    assert_true(
        rb.pos == 25,
        "After CAP+25 adds pos should be 25, got " + String(rb.pos),
    )
    print("  test_circular_wraparound PASSED (size=", rb.size, ", pos=", rb.pos, ")")


def test_sample_content() raises:
    """Fill buffer with a known pattern, sample, verify sampled rows
    each correspond to *some* stored transition (round-trip identity:
    sampled obs[d] should equal `step*10 + d` for some step in [0, n_adds))."""
    var ctx = DeviceContext()
    var rb = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    var obs_p = alloc[Scalar[DT]](OBS)
    var act_p = alloc[Scalar[DT]](ACT)
    var nxt_p = alloc[Scalar[DT]](OBS)

    var n_adds = 50
    for step in range(n_adds):
        _make_obs(obs_p, step)
        _make_act(act_p, step)
        _make_nxt(nxt_p, step)
        rb.add(
            ctx, obs_p, act_p,
            Scalar[DT](Float64(step) + 1000.0),
            nxt_p,
            Scalar[DT](Float64(step % 7 == 0)),
        )
    ctx.synchronize()

    # Allocate device minibatch buffers.
    var mb_s = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](BATCH)

    rb.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)
    ctx.synchronize()

    # D2H download for validation.
    var h_s = alloc[Scalar[DT]](BATCH * OBS)
    var h_a = alloc[Scalar[DT]](BATCH * ACT)
    var h_r = alloc[Scalar[DT]](BATCH)
    var h_sp = alloc[Scalar[DT]](BATCH * OBS)
    var h_d = alloc[Scalar[DT]](BATCH)
    ctx.enqueue_copy(h_s, mb_s)
    ctx.enqueue_copy(h_a, mb_a)
    ctx.enqueue_copy(h_r, mb_r)
    ctx.enqueue_copy(h_sp, mb_sp)
    ctx.enqueue_copy(h_d, mb_d)
    ctx.synchronize()

    # Each sampled row must correspond to some step in [0, n_adds).
    # Reverse-engineer step from obs[0] (= step*10).
    var distinct = List[Int]()
    for i in range(BATCH):
        var s0 = Float64(h_s[i * OBS + 0])
        var step = Int(s0 / 10.0 + 0.5)
        assert_true(
            step >= 0 and step < n_adds,
            "Sampled step out of range: i=" + String(i)
            + " obs[0]=" + String(s0) + " inferred step=" + String(step),
        )
        # Cross-check all fields.
        for d in range(OBS):
            var expected = Float64(step) * 10.0 + Float64(d)
            assert_true(
                Float64(h_s[i * OBS + d]) == expected,
                "obs mismatch i=" + String(i) + " d=" + String(d),
            )
            var expected_nxt = expected + 0.5
            assert_true(
                Float64(h_sp[i * OBS + d]) == expected_nxt,
                "nxt mismatch i=" + String(i) + " d=" + String(d),
            )
        for j in range(ACT):
            var expected_a = Float64(step) * 100.0 + Float64(j)
            assert_true(
                Float64(h_a[i * ACT + j]) == expected_a,
                "act mismatch i=" + String(i) + " j=" + String(j),
            )
        var expected_r = Float64(step) + 1000.0
        assert_true(
            Float64(h_r[i]) == expected_r,
            "rew mismatch i=" + String(i) + " got=" + String(Float64(h_r[i])),
        )
        var expected_d = Float64(step % 7 == 0)
        assert_true(
            Float64(h_d[i]) == expected_d,
            "dne mismatch i=" + String(i) + " got=" + String(Float64(h_d[i])),
        )
        var seen = False
        for k in range(len(distinct)):
            if distinct[k] == step:
                seen = True
                break
        if not seen:
            distinct.append(step)

    # Sanity: with BATCH=32 sampled with replacement from 50 slots, we
    # expect at least 10 distinct steps with very high probability.
    assert_true(
        len(distinct) >= 10,
        "Sample distribution looks pathological: only "
        + String(len(distinct)) + " distinct steps in " + String(BATCH) + " samples",
    )
    print(
        "  test_sample_content PASSED (",
        len(distinct), " distinct steps over ", BATCH, " samples)",
    )


def main() raises:
    print("=" * 60)
    print("C.1 GPUReplay unit tests")
    print("=" * 60)
    test_empty_buffer()
    test_add_and_size_tracking()
    test_circular_wraparound()
    test_sample_content()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
