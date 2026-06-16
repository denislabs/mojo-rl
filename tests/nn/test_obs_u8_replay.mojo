"""Part B (docs/DEVICE_PER_TREE_PLAN.md) — uint8 pixel-obs storage
round-trip tests.

The claim: for obs that are exact `k/255` (what the pixel resize kernel
emits), storing as uint8 (`round(x·255)`) and dequantizing on gather
(`k / 255.0`) is LOSSLESS — the gathered minibatch is bit-identical to
the float32-stored buffer's.

Verified by running a `GPUReplay[…, DT]` and a `GPUReplay[…, uint8]`
side by side: same adds, same fresh Philox stream (same seed + offset ⇒
same sampled indices), then exact `==` comparison of the gathered
obs/next_obs minibatches. A second test does the same through the
uint8 `GPUSequenceReplay` host bridge (frame-tag contiguity + exact
`k/255` values).

Capacity context (NOT speed): uint8 cuts the Pong-pixel
(OBS = 4·84·84) obs+nxt footprint 4× — ~2.7 GB → ~677 MB at CAP=12k.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_obs_u8_replay.mojo
"""

from std.gpu.host import DeviceContext
from std.memory import alloc
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.data.gpu_replay import GPUReplay
from mojo_rl.deep_agents.data.gpu_sequence_replay import GPUSequenceReplay


comptime OBS = 8
comptime ACT = 1
comptime CAP = 32
comptime BATCH = 16


def _obs_val(step: Int, d: Int) -> Scalar[DT]:
    """Exact k/255 'pixel' value, k in [0, 255]."""
    var k = (step * OBS + d * 7) % 256
    return Scalar[DT](k) / Scalar[DT](255.0)


def test_u8_gather_bit_identical_to_f32() raises:
    var ctx = DeviceContext()
    var rb_f = GPUReplay[OBS, ACT, CAP].new(ctx, batch_capacity=BATCH)
    var rb_u = GPUReplay[OBS, ACT, CAP, DType.uint8].new(
        ctx, batch_capacity=BATCH
    )

    var obs = alloc[Scalar[DT]](OBS)
    var act = alloc[Scalar[DT]](ACT)
    var nxt = alloc[Scalar[DT]](OBS)
    for step in range(24):
        for d in range(OBS):
            obs[d] = _obs_val(step, d)
            nxt[d] = _obs_val(step + 1, d)
        act[0] = Scalar[DT](Float64(step))
        rb_f.add(ctx, obs, act, Scalar[DT](Float64(step)), nxt,
                 Scalar[DT](0.0))
        rb_u.add(ctx, obs, act, Scalar[DT](Float64(step)), nxt,
                 Scalar[DT](0.0))

    var mb_s_f = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a_f = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r_f = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp_f = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d_f = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_s_u = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_a_u = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var mb_r_u = ctx.enqueue_create_buffer[DT](BATCH)
    var mb_sp_u = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var mb_d_u = ctx.enqueue_create_buffer[DT](BATCH)

    # Both buffers are fresh → same Philox seed + offset → identical
    # sampled indices → the gathers must agree element-for-element.
    rb_f.sample[BATCH](ctx, mb_s_f, mb_a_f, mb_r_f, mb_sp_f, mb_d_f)
    rb_u.sample[BATCH](ctx, mb_s_u, mb_a_u, mb_r_u, mb_sp_u, mb_d_u)

    var h_f = alloc[Scalar[DT]](BATCH * OBS)
    var h_u = alloc[Scalar[DT]](BATCH * OBS)
    var hp_f = alloc[Scalar[DT]](BATCH * OBS)
    var hp_u = alloc[Scalar[DT]](BATCH * OBS)
    var h_idx = alloc[Int32](BATCH)
    ctx.enqueue_copy(h_f, mb_s_f)
    ctx.enqueue_copy(h_u, mb_s_u)
    ctx.enqueue_copy(hp_f, mb_sp_f)
    ctx.enqueue_copy(hp_u, mb_sp_u)
    ctx.enqueue_copy(h_idx, rb_u.indices)
    ctx.synchronize()

    for i in range(BATCH):
        var step = Int(h_idx[i])
        for d in range(OBS):
            # Exact bitwise equality — quantization is lossless on k/255.
            assert_equal(
                h_u[i * OBS + d], h_f[i * OBS + d],
                "u8 obs round-trip != f32 at lane " + String(i)
                + " d " + String(d),
            )
            assert_equal(
                hp_u[i * OBS + d], hp_f[i * OBS + d],
                "u8 nxt round-trip != f32 at lane " + String(i)
                + " d " + String(d),
            )
            # And both equal the analytically expected k/255.
            assert_equal(
                h_u[i * OBS + d], _obs_val(step, d),
                "u8 obs != expected k/255 at lane " + String(i),
            )
    print("  test_u8_gather_bit_identical_to_f32 PASSED")


def test_u8_sequence_replay_round_trip() raises:
    comptime B = 4
    comptime T = 3
    var ctx = DeviceContext()
    var buf = GPUSequenceReplay[OBS, ACT, CAP, DType.uint8].new(ctx)

    var s = alloc[Scalar[DT]](OBS)
    var a = alloc[Scalar[DT]](ACT)
    for slot in range(20):
        for d in range(OBS):
            s[d] = Scalar[DT](slot) / Scalar[DT](255.0)  # frame tag k/255
        a[0] = Scalar[DT](Float64(slot))
        buf.record(s, a, Scalar[DT](Float64(slot)), Scalar[DT](0.0))

    var obs_out = alloc[Scalar[DT]](B * (T + 1) * OBS)
    var act_out = alloc[Scalar[DT]](B * T * ACT)
    var rew_out = alloc[Scalar[DT]](B * T)
    var dne_out = alloc[Scalar[DT]](B * T)
    buf.sample_batch[B, T](obs_out, act_out, rew_out, dne_out)

    for b in range(B):
        var t0 = Int(
            Float64(obs_out[b * (T + 1) * OBS]) * 255.0 + 0.5
        )
        for k in range(T + 1):
            var expected = Scalar[DT](t0 + k) / Scalar[DT](255.0)
            for i in range(OBS):
                # Exact k/255 dequant + window contiguity.
                assert_equal(
                    obs_out[b * (T + 1) * OBS + k * OBS + i], expected,
                    "seq u8 obs mismatch at b " + String(b)
                    + " k " + String(k),
                )
        for k in range(T):
            # act/rew ride the DT path — untouched by quantization.
            assert_equal(
                act_out[b * T * ACT + k], Scalar[DT](t0 + k),
                "seq act mismatch",
            )
            assert_equal(
                rew_out[b * T + k], Scalar[DT](t0 + k),
                "seq rew mismatch",
            )
    print("  test_u8_sequence_replay_round_trip PASSED")


def main() raises:
    print("=" * 60)
    print("Part B — uint8 pixel-obs storage round-trip tests")
    print("=" * 60)
    test_u8_gather_bit_identical_to_f32()
    test_u8_sequence_replay_round_trip()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
