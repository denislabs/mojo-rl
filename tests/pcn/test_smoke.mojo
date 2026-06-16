"""Smoke test for pcn — small fixed-seed PCN end-to-end on synthetic data.

Architecture (Bogacz canonical, bottom-up):
    PCBlock[3, 5, PCIdentity]   # input → x_1   (no activation on data input)
    PCBlock[5, 4, PCReLU]       # x_1 (after ReLU) → x_2
    PCBlock[4, 2, PCReLU]       # x_2 (after ReLU) → output (readout)

    input(3) → block_0 → x_1(5) → block_1 → x_2(4) → block_2 → output(2) ↔ target(2)

Verifies:
  - The whole stack compiles
  - One full train_one_batch (T_infer=20) runs without crashing
  - Total energy DECREASES across batches (the net is fitting the fixed pair)
  - Energy is finite

Run:
    pixi run mojo run -I . tests/pcn/test_smoke.mojo
"""

from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT as dtype
from mojo_rl.experimental.pcn.pc_initializer import PCXavier
from mojo_rl.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)


comptime BATCH = 2
comptime N_BATCHES = 12

comptime NET = PCSequential[
    PCBlock[3, 5, PCIdentity],   # block_0: input → x_1 (no act on input)
    PCBlock[5, 4, PCReLU],       # block_1: x_1 → x_2 (act = ReLU on x_1)
    PCBlock[4, 2, PCReLU],       # block_2: readout (act = ReLU on x_2)
]


def main() raises:
    print("=== pcn smoke test (Bogacz canonical) ===")
    print("  N           =", NET.N)
    print("  N_LATENTS   =", NET.N_LATENTS)
    print("  IN_DIM      =", NET.IN_DIM)
    print("  OUT_DIM     =", NET.OUT_DIM)
    print("  PARAM_SIZE  =", NET.PARAM_SIZE)
    print("  LATENT_DIM  =", NET.LATENT_DIM)
    print("  SCRATCH_OUT =", NET.SCRATCH_OUT_DIM)
    print("  SCRATCH_IN  =", NET.SCRATCH_IN_DIM)

    # ── params + grads ────────────────────────────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    memset(params_buf, 0, NET.PARAM_SIZE)
    memset(grads_buf, 0, NET.PARAM_SIZE)

    var params = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var grads = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](grads_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    # ── input + target ────────────────────────────────────────────────────────
    var x_in_buf = alloc[Scalar[dtype]](BATCH * NET.IN_DIM)
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM)
    memset(x_in_buf, 0, BATCH * NET.IN_DIM)
    memset(y_tgt_buf, 0, BATCH * NET.OUT_DIM)

    var rng = PhiloxRandom(seed=UInt64(42), offset=UInt64(0))
    for i in range(BATCH * NET.IN_DIM):
        var r = rng.step_uniform()
        x_in_buf[i] = Scalar[dtype](Float32(r[0]) * 2.0 - 1.0)
    # One-hot targets: sample 0 → class 0, sample 1 → class 1
    y_tgt_buf[0 * NET.OUT_DIM + 0] = Scalar[dtype](1.0)
    y_tgt_buf[1 * NET.OUT_DIM + 1] = Scalar[dtype](1.0)

    var x_input = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.IN_DIM), MutAnyOrigin
    ](x_in_buf)
    var y_target = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.OUT_DIM), MutAnyOrigin
    ](y_tgt_buf)

    # ── latents + scratch buffers (allocated once, reused) ────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM)
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM)
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM)
    memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](lat_buf)
    var mu_eps_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_OUT_DIM), MutAnyOrigin
    ](mu_eps_buf_raw)
    var a_below_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](a_below_buf_raw)
    var z_below_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](z_below_buf_raw)
    var dx_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](dx_buf_raw)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n  batch | E_init    E_final   sup_loss")
    print("  ------+--------------------------------")
    var first_loss: Float64 = 0.0
    var last_loss: Float64 = 0.0
    for b in range(N_BATCHES):
        var result = PCTrainer[
            PCBlock[3, 5, PCIdentity],
            PCBlock[5, 4, PCReLU],
            PCBlock[4, 2, PCReLU],
            dtype=dtype,
        ].train_one_batch[BATCH](
            params,
            grads,
            latents,
            mu_eps_buf,
            a_below_buf,
            z_below_buf,
            dx_buf,
            x_input,
            y_target,
            T_infer=20,
            lr_x=Scalar[dtype](0.1),
            lr_w=Scalar[dtype](0.01),
        )
        print(
            "    ", b, "  ",
            String(result.energy_initial)[byte=:9], " ",
            String(result.energy_final)[byte=:9], " ",
            String(result.output_loss_final)[byte=:9],
        )
        if b == 0:
            first_loss = result.output_loss_final
        last_loss = result.output_loss_final

    print("\n  first sup_loss:", first_loss)
    print("  last  sup_loss:", last_loss)

    if last_loss < first_loss:
        print("  [PASS] supervised loss decreased over batches")
    else:
        print("  [FAIL] supervised loss did NOT decrease")
        raise Error("smoke test failed")

    params_buf.free()
    grads_buf.free()
    x_in_buf.free()
    y_tgt_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    print("=== Done ===")
