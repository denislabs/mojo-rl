"""Gate for data-platform Stage 3 — index-policy PARITY against the legacy buffers.

This is the load-bearing gate of the whole migration. A sampler that is
subtly *differently* random does not fail loudly: training still converges,
just worse, and only visibly so once several algorithms have moved. So each
new policy must reproduce its legacy counterpart's index sequence
**bit-for-bit** under a fixed seed.

**How the legacy index sequence is recovered without touching legacy code:**
each stored row's first observation element is set to its own row number, so
the gathered minibatch column 0 *is* the sequence of sampled indices. No
instrumentation, no forked copy of the sampler.

⚠ Host and device uniform are different sequences by construction (global
`random_float64()` vs per-lane Philox). Parity is per-backend against its own
legacy counterpart — comparing CPU to GPU would be meaningless.

Run:
    pixi run mojo run -I . tests/data/test_sampler_parity.mojo
"""

from std.gpu.host import DeviceContext
from std.random import random_float64, seed
from std.testing import assert_almost_equal, assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.data import (
    IndexBatch,
    PrioritizedSampler,
    SequenceWindowSampler,
    UniformDeviceSampler,
    UniformSampler,
)
from mojo_rl.deep_agents.data.cpu_replay import CPUReplay
from mojo_rl.deep_agents.data.cpu_per_replay import CPUPrioritizedReplay
from mojo_rl.deep_agents.data.gpu_replay import GPUReplay
from mojo_rl.deep_agents.data.sequence_replay import SequenceReplay


comptime OBS: Int = 2
comptime ACT: Int = 1
comptime CAP: Int = 64
comptime N_FILL: Int = 50
comptime BATCH: Int = 32
comptime SEED: Int = 20260805


def _row_obs(row: Int) -> List[Scalar[DT]]:
    """obs[0] = row number, so a gathered minibatch reveals the indices."""
    var o = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    o[0] = Scalar[DT](row)
    o[1] = Scalar[DT](-row)
    return o^


def _one_act() -> List[Scalar[DT]]:
    return List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.5))


def _mb() -> List[Scalar[DT]]:
    return List[Scalar[DT]](length=BATCH * OBS, fill=Scalar[DT](0))


def _scalar_mb() -> List[Scalar[DT]]:
    return List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))


# ══════════════════════════════════════════════════════════════════════════
# 1. Uniform, host
# ══════════════════════════════════════════════════════════════════════════

def test_uniform_host_parity() raises:
    print("[1] uniform (host) vs CPUReplay ...")
    var legacy = CPUReplay[OBS, ACT, CAP].new()
    for r in range(N_FILL):
        var o = _row_obs(r)
        var a = _one_act()
        var nx = _row_obs(r)
        legacy.add(o, a, Scalar[DT](r), nx, Scalar[DT](0))

    var s_out = _mb()
    var a_out = List[Scalar[DT]](length=BATCH * ACT, fill=Scalar[DT](0))
    var r_out = _scalar_mb()
    var sp_out = _mb()
    var d_out = _scalar_mb()

    seed(SEED)
    legacy.sample(BATCH, s_out, a_out, r_out, sp_out, d_out)
    var legacy_idx = List[Int]()
    for k in range(BATCH):
        legacy_idx.append(Int(Float64(s_out[k * OBS])))

    seed(SEED)
    var sampler = UniformSampler(legacy.count())
    var got = sampler.draw(BATCH)

    assert_equal(got.size(), BATCH, "batch size")
    for k in range(BATCH):
        assert_equal(
            Int(got.host[k]), legacy_idx[k],
            "uniform-host index " + String(k) + " differs from legacy",
        )
    # A sampler that returned a constant would pass a weak check; assert the
    # sequence actually varies.
    var distinct = 0
    for k in range(1, BATCH):
        if legacy_idx[k] != legacy_idx[0]:
            distinct += 1
    assert_true(distinct > 0, "legacy sequence is degenerate — bad fixture")
    print("     ", BATCH, "indices identical  OK")


# ══════════════════════════════════════════════════════════════════════════
# 2. Prioritized, host
# ══════════════════════════════════════════════════════════════════════════

def test_per_host_parity() raises:
    print("[2] prioritized (host) vs CPUPrioritizedReplay ...")
    var legacy = CPUPrioritizedReplay[OBS, ACT, CAP].new(
        alpha=Scalar[DT](0.6), beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](1e-6), batch_capacity=BATCH,
    )
    var mine = PrioritizedSampler(
        CAP, alpha=Scalar[DT](0.6), beta=Scalar[DT](0.4),
        epsilon=Scalar[DT](1e-6),
    )
    for r in range(N_FILL):
        var o = _row_obs(r)
        var a = _one_act()
        var nx = _row_obs(r)
        legacy.add(o, a, Scalar[DT](r), nx, Scalar[DT](0))
        mine.note_added(r)

    var s_out = _mb()
    var a_out = List[Scalar[DT]](length=BATCH * ACT, fill=Scalar[DT](0))
    var r_out = _scalar_mb()
    var sp_out = _mb()
    var d_out = _scalar_mb()

    # ── first draw, uniform priorities ────────────────────────────────
    seed(SEED)
    legacy.sample[BATCH](s_out, a_out, r_out, sp_out, d_out)
    var legacy_idx = List[Int]()
    for k in range(BATCH):
        legacy_idx.append(Int(Float64(s_out[k * OBS])))

    seed(SEED)
    var got = mine.draw(BATCH)
    for k in range(BATCH):
        assert_equal(
            Int(got.host[k]), legacy_idx[k],
            "PER index " + String(k) + " differs on the first draw",
        )
    for k in range(BATCH):
        assert_almost_equal(
            Float64(mine.last_weights[k]),
            Float64(legacy._host_weights[k]),
            atol=1e-12,
            msg="PER IS weight " + String(k),
        )
    print("      first draw + IS weights OK")

    # ── priority update, then a second draw ───────────────────────────
    # The second draw is the real test: it only matches if the sum-tree
    # UPDATE reproduced the legacy tree exactly, not merely the descent.
    var td = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    for k in range(BATCH):
        td[k] = Scalar[DT](Float64(k + 1) * 0.37 - 5.0)   # mixed signs

    legacy.update_priorities[BATCH](td)
    mine.update_priorities(td)

    seed(SEED + 1)
    legacy.sample[BATCH](s_out, a_out, r_out, sp_out, d_out)
    var legacy_idx2 = List[Int]()
    for k in range(BATCH):
        legacy_idx2.append(Int(Float64(s_out[k * OBS])))

    seed(SEED + 1)
    var got2 = mine.draw(BATCH)
    for k in range(BATCH):
        assert_equal(
            Int(got2.host[k]), legacy_idx2[k],
            "PER index " + String(k) + " differs AFTER a priority update",
        )
    # The update must actually have changed the distribution, or the second
    # draw proves nothing about the tree write path.
    var changed = 0
    for k in range(BATCH):
        if legacy_idx2[k] != legacy_idx[k]:
            changed += 1
    assert_true(
        changed > 0,
        "priority update did not change the draw — the second comparison"
        " would be vacuous",
    )
    print("      post-update draw OK (", changed, "of", BATCH, "indices moved)")


# ══════════════════════════════════════════════════════════════════════════
# 3. Sequence windows, host
# ══════════════════════════════════════════════════════════════════════════

def test_sequence_host_parity() raises:
    print("[3] sequence windows (host) vs SequenceReplay ...")
    comptime T: Int = 6
    comptime B: Int = 8

    var legacy = SequenceReplay[OBS, ACT, CAP].new()
    for r in range(N_FILL):
        var o = _row_obs(r)
        var a = _one_act()
        legacy.record(
            o.unsafe_ptr().as_unsafe_any_origin(),
            a.unsafe_ptr().as_unsafe_any_origin(),
            Scalar[DT](r),
            Scalar[DT](0),
        )

    var obs_out = List[Scalar[DT]](
        length=B * (T + 1) * OBS, fill=Scalar[DT](0)
    )
    var act_out = List[Scalar[DT]](length=B * T * ACT, fill=Scalar[DT](0))
    var rew_out = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))
    var dne_out = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))
    var fst_out = List[Scalar[DT]](length=B * (T + 1), fill=Scalar[DT](0))

    seed(SEED)
    legacy.sample_batch_fst[B, T](
        obs_out.unsafe_ptr().as_unsafe_any_origin(),
        act_out.unsafe_ptr().as_unsafe_any_origin(),
        rew_out.unsafe_ptr().as_unsafe_any_origin(),
        dne_out.unsafe_ptr().as_unsafe_any_origin(),
        fst_out.unsafe_ptr().as_unsafe_any_origin(),
    )
    var legacy_starts = List[Int]()
    for b in range(B):
        legacy_starts.append(Int(Float64(obs_out[b * (T + 1) * OBS])))

    seed(SEED)
    var sampler = SequenceWindowSampler(legacy.count(), T)
    var got = sampler.draw_starts(B)

    assert_equal(
        sampler.n_valid(), legacy.count() - T, "n_valid must match size - T"
    )
    for b in range(B):
        assert_equal(
            Int(got.host[b]), legacy_starts[b],
            "sequence start " + String(b) + " differs from legacy",
        )

    # The expansion must reproduce the legacy window's frame rows.
    var exp_idx = List[Scalar[DType.int32]]()
    sampler.expand_window(got, exp_idx)
    assert_equal(len(exp_idx), B * (T + 1), "expanded window size")
    for b in range(B):
        for k in range(T + 1):
            var want = Int(Float64(obs_out[(b * (T + 1) + k) * OBS]))
            assert_equal(
                Int(exp_idx[b * (T + 1) + k]), want,
                "window frame (" + String(b) + "," + String(k) + ")",
            )
    print("      starts + window expansion OK")


def test_within_episode_option() raises:
    """The non-spanning option is OPT-IN; the default must keep the legacy
    (spanning) behaviour, since DreamerV3 relies on it."""
    print("[4] within_episode is opt-in, not the default ...")
    var s = SequenceWindowSampler(100, 8)
    assert_true(not s.within_episode, "default must be spanning (legacy)")
    var s2 = SequenceWindowSampler(100, 8, True)
    assert_true(s2.within_episode, "opt-in flag must stick")
    print("      OK")


# ══════════════════════════════════════════════════════════════════════════
# 4. Uniform, device
# ══════════════════════════════════════════════════════════════════════════

def test_uniform_device_parity() raises:
    print("[5] uniform (device) vs GPUReplay Philox ...")
    var ctx = DeviceContext()
    var legacy = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    for r in range(N_FILL):
        var o = _row_obs(r)
        var a = _one_act()
        var nx = _row_obs(r)
        legacy.add(o, a, Scalar[DT](r), nx, Scalar[DT](0), ctx=ctx)

    var mb_s = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d = ctx.enqueue_create_buffer[DT](BATCH)
    legacy.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)

    var host_s = List[Scalar[DT]](unsafe_uninit_length=BATCH * OBS)
    ctx.enqueue_copy(host_s.unsafe_ptr(), mb_s)
    ctx.synchronize()

    var legacy_idx = List[Int]()
    for k in range(BATCH):
        legacy_idx.append(Int(Float64(host_s[k * OBS])))

    # Same seed and starting offset as GPUReplay.new().
    var mine = UniformDeviceSampler(
        legacy.count(), seed=UInt64(0xC0FFEE_DECADE_0042), offset=UInt64(0)
    )
    var got = mine.draw(ctx, BATCH)

    for k in range(BATCH):
        assert_equal(
            Int(got.host[k]), legacy_idx[k],
            "uniform-device index " + String(k) + " differs from legacy",
        )
    var distinct = 0
    for k in range(1, BATCH):
        if legacy_idx[k] != legacy_idx[0]:
            distinct += 1
    assert_true(distinct > 0, "device sequence is degenerate — bad fixture")

    # The offset must advance the same way, or the SECOND batch diverges.
    legacy.sample[BATCH](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)
    ctx.enqueue_copy(host_s.unsafe_ptr(), mb_s)
    ctx.synchronize()
    var got2 = mine.draw(ctx, BATCH)
    for k in range(BATCH):
        assert_equal(
            Int(got2.host[k]), Int(Float64(host_s[k * OBS])),
            "uniform-device index " + String(k) + " differs on the SECOND"
            " batch (RNG offset advance mismatch)",
        )
    print("     ", BATCH, "x2 indices identical  OK")


def main() raises:
    test_uniform_host_parity()
    test_per_host_parity()
    test_sequence_host_parity()
    test_within_episode_option()
    test_uniform_device_parity()
    print("\n[PASS] sampler parity — Stage 3")
