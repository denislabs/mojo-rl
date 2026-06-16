"""DynamicsEnsembleBlock GPU smoke (Phase 4.3a).

Builds a GPU dynamics ensemble (MBPO world model) and verifies the new
device paths:
  1. `predict_member['gpu']` produces finite means + logvars, with the
     logvars clamped into [LOGVAR_MIN, LOGVAR_MAX].
  2. `train_member_step['gpu']` produces a finite Gaussian-NLL loss that
     DECREASES over repeated steps on a fixed (input, target) batch — i.e.
     the device forward → loss → vjp → member.vjp → Adam.step chain learns.
  3. `eval_member_loss['gpu']` matches the post-train loss (forward-only).

Strong end-to-end check of the GPU ensemble on Apple Metal; full numeric
CPU↔GPU parity is a separate NVIDIA-gated step (TF32 / FD unreliable on
Metal).

Run (Apple): pixi run -e apple mojo run -I . \
    tests/nn/test_dynamics_ensemble_gpu_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import random_float64, seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.mbpo.dynamics_ensemble_block import (
    DynamicsEnsembleBlock,
)


comptime OBS = 3
comptime ACT = 1
comptime IN = OBS + ACT          # 4
comptime PRED = 1 + OBS          # 4 (reward + Δobs)
comptime OUT = 2 * PRED          # 8 (mean + logvar)
comptime BATCH = 32
comptime N = 3
comptime ELITES = 2
comptime LV_MIN = -10.0
comptime LV_MAX = -2.0

comptime DynNet = Sequential[
    Linear[IN, 32], ReLU[32], Linear[32, 32], ReLU[32], Linear[32, OUT],
]
comptime Ensemble = DynamicsEnsembleBlock[
    DynNet, N, ELITES, IN, OUT, BATCH, LV_MIN, LV_MAX,
]


def _finite(v: Float64, tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_dynamics_ensemble_gpu_smoke() raises:
    print("--- DynamicsEnsembleBlock GPU smoke ---")
    seed(42)
    var ctx = DeviceContext()
    var ens = Ensemble.make["gpu", Kaiming](ctx)

    # Fixed input + target batch (host → device).
    var in_host = List[Scalar[DT]](length=BATCH * IN, fill=Scalar[DT](0.0))
    var tgt_host = List[Scalar[DT]](length=BATCH * PRED, fill=Scalar[DT](0.0))
    for k in range(BATCH * IN):
        in_host[k] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for k in range(BATCH * PRED):
        tgt_host[k] = Scalar[DT](random_float64() * 0.5 - 0.25)

    var in_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var tgt_dev = ctx.enqueue_create_buffer[DT](BATCH * PRED)
    ctx.enqueue_copy(in_dev, in_host.unsafe_ptr())
    ctx.enqueue_copy(tgt_dev, tgt_host.unsafe_ptr())
    ctx.synchronize()
    var in_t = TileTensor(in_dev.unsafe_ptr(), row_major[BATCH, IN]())
    var tgt_t = TileTensor(tgt_dev.unsafe_ptr(), row_major[BATCH, PRED]())

    # ── 1. predict_member: finite mu + clamped lv.
    var mu_dev = ctx.enqueue_create_buffer[DT](BATCH * PRED)
    var lv_dev = ctx.enqueue_create_buffer[DT](BATCH * PRED)
    var mu_t = TileTensor(mu_dev.unsafe_ptr(), row_major[BATCH, PRED]())
    var lv_t = TileTensor(lv_dev.unsafe_ptr(), row_major[BATCH, PRED]())
    ens.predict_member["gpu"](0, in_t, mu_t, lv_t)
    var mu_host = ctx.enqueue_create_host_buffer[DT](BATCH * PRED)
    var lv_host = ctx.enqueue_create_host_buffer[DT](BATCH * PRED)
    ctx.enqueue_copy(mu_host, mu_dev)
    ctx.enqueue_copy(lv_host, lv_dev)
    ctx.synchronize()
    for k in range(BATCH * PRED):
        _finite(Float64(mu_host.unsafe_ptr()[k]), "mu")
        var lv = Float64(lv_host.unsafe_ptr()[k])
        _finite(lv, "lv")
        assert_true(
            lv >= LV_MIN - 1e-5 and lv <= LV_MAX + 1e-5,
            "logvar not clamped into [" + String(LV_MIN) + ", "
            + String(LV_MAX) + "]: " + String(lv),
        )
    print("  predict_member OK (finite mu, lv clamped)")

    # ── 2. train_member_step: finite, decreasing loss.
    var first_loss = Float64(ens.train_member_step["gpu"](0, in_t, tgt_t))
    _finite(first_loss, "first_loss")
    var last_loss = first_loss
    for _ in range(60):
        last_loss = Float64(ens.train_member_step["gpu"](0, in_t, tgt_t))
    _finite(last_loss, "last_loss")
    print("  train_member_step: first=", first_loss, " last=", last_loss)
    assert_true(
        last_loss < first_loss,
        "GPU NLL loss did not decrease over training (first="
        + String(first_loss) + ", last=" + String(last_loss) + ")",
    )

    # ── 3. eval_member_loss forward-only on the trained weights. Note
    # train_member_step returns the PRE-update loss, so the post-update
    # eval forward should be no worse than `last_loss` (one more gradient
    # step of improvement), and far below the initial loss.
    var eval_loss = Float64(ens.eval_member_loss["gpu"](0, in_t, tgt_t))
    _finite(eval_loss, "eval_loss")
    assert_true(
        eval_loss <= last_loss + 1e-3,
        "eval (post-update) loss should be <= last pre-update loss: eval="
        + String(eval_loss) + ", last=" + String(last_loss),
    )
    assert_true(
        eval_loss < first_loss,
        "eval loss should be well below the initial loss",
    )
    print("  eval_member_loss OK (=", eval_loss, ")")
    print("PASS")


def main() raises:
    print("=" * 70)
    print("DynamicsEnsembleBlock GPU smoke (Phase 4.3a)")
    print("=" * 70)
    test_dynamics_ensemble_gpu_smoke()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
