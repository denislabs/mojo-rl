"""Smoke test for nn_pc — small fixed-seed PCN end-to-end on synthetic data.

Architecture:
    PCLinear[3, 5]              # hidden 1 — predicts input (3) from x^(1) (5)
    PCLinear[5, 4]              # hidden 2 — predicts x^(1) from x^(2) (4)
    PCLinear[2, 4, PCIdentity]  # readout  — y_hat (2) from x^(2) (4)

Verifies:
  - The whole stack compiles
  - One full train_one_batch (T_infer=20, T_learn=50) runs without crashing
  - Supervised loss DECREASES across batches (using the same x_input/y_target
    so we should see the net memorize)
  - Total energy is finite

Run:
    pixi run mojo run -I . tests/nn_pc/test_pc_smoke.mojo
"""

from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.experimental.nn_pc import (
    PCLinear, PCSequential, PCIdentity, PCTrainer
)


comptime BATCH = 2
comptime N_BATCHES = 8

comptime MODEL = PCSequential[
    PCLinear[3, 5],
    PCLinear[5, 4],
    PCLinear[2, 4, PCIdentity],
]
# PCTrainer takes the same variadic LAYERS (constructs MODEL internally).
comptime TRAINER = PCTrainer[
    PCLinear[3, 5],
    PCLinear[5, 4],
    PCLinear[2, 4, PCIdentity],
    dtype=dtype,
]


def main() raises:
    print("=== nn_pc smoke test ===")
    print("  N_LINEARS  =", MODEL.N_LINEARS)
    print("  N_LATENTS  =", MODEL.N_LATENTS)
    print("  IN_DIM     =", MODEL.IN_DIM)
    print("  OUT_DIM    =", MODEL.OUT_DIM)
    print("  TOP_LATENT =", MODEL.TOP_LATENT_DIM)
    print("  PARAM_SIZE =", MODEL.PARAM_SIZE)
    print("  LATENT_SIZE_PER_SAMPLE =", MODEL.LATENT_SIZE_PER_SAMPLE)

    # ── Allocate params + initialize via Xavier ──
    var params_buf = alloc[Scalar[dtype]](MODEL.PARAM_SIZE)
    memset(params_buf, 0, MODEL.PARAM_SIZE)
    var params = LayoutTensor[
        dtype, Layout.row_major(MODEL.PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    MODEL.initialize_params[Xavier[], dtype](params)

    # ── Allocate input + target ──
    var x_in_buf = alloc[Scalar[dtype]](BATCH * MODEL.IN_DIM)
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * MODEL.OUT_DIM)
    memset(x_in_buf, 0, BATCH * MODEL.IN_DIM)
    memset(y_tgt_buf, 0, BATCH * MODEL.OUT_DIM)

    # Fixed inputs (uniform-ish via Philox)
    var rng = PhiloxRandom(seed=UInt64(42), offset=UInt64(0))
    for i in range(BATCH * MODEL.IN_DIM):
        var r = rng.step_uniform()
        x_in_buf[i] = Scalar[dtype](Float32(r[0]) * 2.0 - 1.0)
    # One-hot targets: sample 0 -> class 0, sample 1 -> class 1
    y_tgt_buf[0 * MODEL.OUT_DIM + 0] = Scalar[dtype](1.0)
    y_tgt_buf[1 * MODEL.OUT_DIM + 1] = Scalar[dtype](1.0)

    var x_input = LayoutTensor[
        dtype, Layout.row_major(BATCH, MODEL.IN_DIM), MutAnyOrigin
    ](x_in_buf)
    var y_target = LayoutTensor[
        dtype, Layout.row_major(BATCH, MODEL.OUT_DIM), MutAnyOrigin
    ](y_tgt_buf)

    # ── Allocate latents (per batch — re-init each batch) ──
    var lat_size = BATCH * MODEL.LATENT_SIZE_PER_SAMPLE
    var lat_buf = alloc[Scalar[dtype]](lat_size)
    var latents = LayoutTensor[
        dtype, Layout.row_major(BATCH, MODEL.LATENT_SIZE_PER_SAMPLE), MutAnyOrigin
    ](lat_buf)

    # ── Run multiple train batches; expect sup_loss to decrease ──
    print("\n  batch | energy       sup_loss")
    print("  ------+----------------------")
    var first_loss: Float64 = 0.0
    var last_loss: Float64 = 0.0
    for b in range(N_BATCHES):
        # Re-initialize latents ~ N(0, 1) for each batch (PyTorch convention)
        TRAINER.randn_init_latents[BATCH](
            latents, seed=UInt64(100 + b), offset=UInt64(0)
        )
        var result = TRAINER.train_one_batch[BATCH](
            params, latents, x_input, y_target,
            T_infer=20, T_learn=50,
            eta_infer=Scalar[dtype](0.05),
            eta_learn=Scalar[dtype](0.005),
        )
        print(
            "    ", b, "  ",
            String(result.energy)[byte=:8], " ",
            String(result.sup_loss)[byte=:8],
        )
        if b == 0:
            first_loss = result.sup_loss
        last_loss = result.sup_loss

    print("\n  first sup_loss:", first_loss)
    print("  last  sup_loss:", last_loss)

    if last_loss < first_loss:
        print("  [PASS] supervised loss decreased over batches")
    else:
        print("  [FAIL] supervised loss did NOT decrease")
        raise Error("smoke test failed")

    params_buf.free()
    x_in_buf.free()
    y_tgt_buf.free()
    lat_buf.free()
    print("=== Done ===")
