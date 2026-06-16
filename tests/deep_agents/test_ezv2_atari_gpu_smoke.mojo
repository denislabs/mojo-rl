"""EZv2-Atari spatial model — GPU compile + forward smoke (Stage 2 polish #16).

The Stage-2 spatial nets were CPU-verified only; this confirms the full
rep → dynamics → prediction stack COMPILES and runs a finite forward on the
GPU (Apple Metal / NVIDIA), at the real Atari dims (IN_CH=12, ACT=18,
BINS=601, latent [64,6,6]=2304). It also doubles as the GPU verification of
`init_zero`: after zeroing the head/reward output layers, the prediction
output (all 619 logits) and the dynamics reward half (last 601) must read
back EXACTLY zero, while the dynamics next-state half stays finite.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_gpu_smoke.mojo
"""

from std.math import abs, isnan, isinf
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.testing import assert_true, assert_equal
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import (
    EZRepNetResNetAtari, EZDynNetAtari, EZPredNetAtari,
    EZ_LATENT, EZ_C,
    ez_atari_init_zero_pred, ez_atari_init_zero_dyn,
)


comptime IN_CH = 12
comptime ACT = 18
comptime BINS = 601
comptime B = 2

comptime Rep = EZRepNetResNetAtari[IN_CH, EZ_C]
comptime Dyn = EZDynNetAtari[ACT, BINS]
comptime Pred = EZPredNetAtari[ACT, BINS]

comptime OBS = IN_CH * 96 * 96          # 110592
comptime DYN_IN = EZ_LATENT + ACT       # 2322
comptime DYN_OUT = EZ_LATENT + BINS      # 2905
comptime PRED_OUT = ACT + BINS           # 619


def main() raises:
    print("=" * 70)
    print("EZv2-Atari spatial model — GPU compile + init_zero smoke")
    print("=" * 70)

    assert_equal(Rep.OUT_DIM, EZ_LATENT, "rep out == 2304")
    assert_equal(Dyn.OUT_DIM, DYN_OUT, "dyn out == LATENT+BINS")
    assert_equal(Pred.OUT_DIM, PRED_OUT, "pred out == ACT+BINS")

    with DeviceContext() as ctx:
        var rep = Rep.make[target="gpu", INIT=Kaiming](ctx)
        var dyn = Dyn.make[target="gpu", INIT=Kaiming](ctx)
        var pred = Pred.make[target="gpu", INIT=Kaiming](ctx)
        dyn.set_attr["training"](Scalar[DT](1.0))

        # ── input obs slab → rep → latent ──────────────────────────────
        var obs = ctx.enqueue_create_buffer[DT](B * OBS)
        var lat = ctx.enqueue_create_buffer[DT](B * EZ_LATENT)
        var obs_h = ctx.enqueue_create_host_buffer[DT](B * OBS)
        ctx.synchronize()
        for i in range(B * OBS):
            obs_h.unsafe_ptr()[i] = Scalar[DT]((i % 255)) / Scalar[DT](255.0)
        ctx.enqueue_copy(obs, obs_h)
        ctx.synchronize()

        var obs_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](obs.unsafe_ptr()),
            row_major[B, OBS]())
        var lat_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](lat.unsafe_ptr()),
            row_major[B, EZ_LATENT]())
        rep.forward["gpu", B](obs_t, output=lat_t)

        var lat_h = ctx.enqueue_create_host_buffer[DT](B * EZ_LATENT)
        ctx.enqueue_copy(lat_h, lat)
        ctx.synchronize()
        var lat_finite = True
        for i in range(B * EZ_LATENT):
            var v = lat_h.unsafe_ptr()[i]
            if isnan(v) or isinf(v):
                lat_finite = False
        assert_true(lat_finite, "rep latent finite on GPU")
        print("  ✓ rep forward (110592 → 2304) finite on GPU")

        # ── init_zero, then forwards must give exactly-zero head outputs ─
        ez_atari_init_zero_pred["gpu", ACT, BINS](pred, ctx)
        ez_atari_init_zero_dyn["gpu", ACT, BINS](dyn, ctx)
        ctx.synchronize()

        # prediction: latent → [policy(18) | value(601)] — all zero post-init.
        var pout = ctx.enqueue_create_buffer[DT](B * PRED_OUT)
        var pout_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](pout.unsafe_ptr()),
            row_major[B, PRED_OUT]())
        pred.forward["gpu", B](lat_t, output=pout_t)
        var pout_h = ctx.enqueue_create_host_buffer[DT](B * PRED_OUT)
        ctx.enqueue_copy(pout_h, pout)
        ctx.synchronize()
        var pmax = Scalar[DT](0.0)
        for i in range(B * PRED_OUT):
            var a = abs(pout_h.unsafe_ptr()[i])
            if a > pmax:
                pmax = a
        print("  pred max|logit| after init_zero =", pmax)
        assert_true(pmax == Scalar[DT](0.0),
                    "pred outputs exactly zero after init_zero (GPU)")

        # dynamics: [z|onehot] → [z'(2304) | reward(601)]; reward half zero.
        var din = ctx.enqueue_create_buffer[DT](B * DYN_IN)
        var din_h = ctx.enqueue_create_host_buffer[DT](B * DYN_IN)
        ctx.synchronize()
        for i in range(B * DYN_IN):
            din_h.unsafe_ptr()[i] = Scalar[DT]((i % 17)) / Scalar[DT](17.0)
        ctx.enqueue_copy(din, din_h)
        ctx.synchronize()
        var din_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](din.unsafe_ptr()),
            row_major[B, DYN_IN]())
        var dout = ctx.enqueue_create_buffer[DT](B * DYN_OUT)
        var dout_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dout.unsafe_ptr()),
            row_major[B, DYN_OUT]())
        dyn.forward["gpu", B](din_t, output=dout_t)
        var dout_h = ctx.enqueue_create_host_buffer[DT](B * DYN_OUT)
        ctx.enqueue_copy(dout_h, dout)
        ctx.synchronize()

        var rew_max = Scalar[DT](0.0)
        var zp_finite = True
        for b in range(B):
            for j in range(DYN_OUT):
                var v = dout_h.unsafe_ptr()[b * DYN_OUT + j]
                if j < EZ_LATENT:
                    if isnan(v) or isinf(v):
                        zp_finite = False
                else:
                    var a = abs(v)
                    if a > rew_max:
                        rew_max = a
        print("  dyn next-state finite =", zp_finite,
              " reward max|logit| after init_zero =", rew_max)
        assert_true(zp_finite, "dyn next-state (z') finite on GPU")
        assert_true(rew_max == Scalar[DT](0.0),
                    "dyn reward outputs exactly zero after init_zero (GPU)")

        _ = rep^
        _ = dyn^
        _ = pred^

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
