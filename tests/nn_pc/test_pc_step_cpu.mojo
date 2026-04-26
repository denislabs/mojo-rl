"""Hand-verified numerical test for nn_pc on CPU.

Tiny architecture, BATCH=1, all weights/latent/inputs hand-set so that one
inference step + one learning step can be compared element-wise to hand
computation. Catches sign/index bugs that the smoke test would silently
absorb.

Architecture:
    PCLinear[2, 3]              # hidden — predicts input(2) from x^(1)(3)
    PCLinear[2, 3, PCIdentity]  # readout — y_hat(2) from x^(1)(3)

Latent layout (per sample):
    x^(1) of dim 3 (only hidden latent)

Param layout (row-major, [in_dim, out_dim]):
    W_0  shape [2, 3] at offset 0
    W_R  shape [2, 3] at offset 6

Setup (BATCH=1):
    x_input  = [1.0, 2.0]
    y_target = [1.0, 0.0]
    W_0      = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    W_R      = [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
    x^(1)    = [0.1, 0.2, 0.3]
    eta_infer = 0.05, eta_learn = 0.01, T_infer = 1, T_learn = 1

Hand computation (Float64, see test source for full derivation):
    INFERENCE STEP:
      a_0 = x^(1) @ W_0^T = [0.14, 0.32]                  (>0 → ReLU passes through)
      x_hat_0 = a_0 = [0.14, 0.32]
      eps_0 = x_input - x_hat_0 = [0.86, 1.68]
      h_0 = eps_0 * 1 = [0.86, 1.68]

      a_R = x^(1) @ W_R^T = [0.50, 0.68]                  (identity → x_hat = a)
      eps_sup = x_hat_R - y_target = [-0.50, 0.68]
      h_R = eps_sup = [-0.50, 0.68]
      eps_L_pulled = h_R @ W_R = [0.33, 0.348, 0.366]

      pb (h_0 @ W_0)       = [0.758, 1.012, 1.266]
      grad_X = eps_L_pulled - pb = [-0.428, -0.664, -0.900]
      x^(1)_new = x^(1) - 0.05 * grad_X = [0.1214, 0.2332, 0.345]

    LEARNING STEP (recomputed with x^(1)_new):
      a_0 = [0.16228, 0.37216];  x_hat_0 = a_0;  eps_0 = [0.83772, 1.62784]
      h_0 = [0.83772, 1.62784]
      a_R = [0.58204, 0.79192];  y_hat = a_R;  eps_sup = [-0.41796, 0.79192]
      h_R = eps_sup

      W_0 += +0.01 * (h_0^T @ x_above):
        W_0[0,0] = 0.1 + 0.01 * 0.83772 * 0.1214 = 0.1010170...
        ... (full table in expected values below)

      W_R += -0.01 * (h_R^T @ x_above):
        W_R[0,0] = 0.7 - 0.01 * (-0.41796) * 0.1214 = 0.7005074...
        ...

Run:
    pixi run mojo run -I . tests/nn_pc/test_pc_step_cpu.mojo
"""

from std.math import abs
from std.memory import alloc, memset
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn_pc import PCLinear, PCSequential, PCIdentity, PCTrainer


comptime BATCH = 1

comptime TRAINER = PCTrainer[
    PCLinear[2, 3],
    PCLinear[2, 3, PCIdentity],
    dtype=dtype,
]


def main() raises:
    print("=== nn_pc step CPU correctness ===")

    # ── Architecture sanity ──
    if TRAINER.MODEL.N_LINEARS != 2:
        raise Error("expected N_LINEARS=2")
    if TRAINER.MODEL.N_LATENTS != 1:
        raise Error("expected N_LATENTS=1")
    if TRAINER.MODEL.PARAM_SIZE != 12:
        raise Error("expected PARAM_SIZE=12")
    if TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE != 3:
        raise Error("expected LATENT_SIZE_PER_SAMPLE=3")

    # ── Allocate buffers ──
    var params = alloc[Scalar[dtype]](TRAINER.MODEL.PARAM_SIZE)
    memset(params, 0, TRAINER.MODEL.PARAM_SIZE)
    var lat = alloc[Scalar[dtype]](BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE)
    memset(lat, 0, BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE)
    var x_in = alloc[Scalar[dtype]](BATCH * TRAINER.MODEL.IN_DIM)
    memset(x_in, 0, BATCH * TRAINER.MODEL.IN_DIM)
    var y_tgt = alloc[Scalar[dtype]](BATCH * TRAINER.MODEL.OUT_DIM)
    memset(y_tgt, 0, BATCH * TRAINER.MODEL.OUT_DIM)

    # ── Hand-set initial state ──
    # W_0 [2,3] row-major at offset 0
    params[0] = Scalar[dtype](0.1)  # W_0[0,0]
    params[1] = Scalar[dtype](0.2)  # W_0[0,1]
    params[2] = Scalar[dtype](0.3)  # W_0[0,2]
    params[3] = Scalar[dtype](0.4)  # W_0[1,0]
    params[4] = Scalar[dtype](0.5)  # W_0[1,1]
    params[5] = Scalar[dtype](0.6)  # W_0[1,2]
    # W_R [2,3] row-major at offset 6
    params[6] = Scalar[dtype](0.7)
    params[7] = Scalar[dtype](0.8)
    params[8] = Scalar[dtype](0.9)
    params[9] = Scalar[dtype](1.0)
    params[10] = Scalar[dtype](1.1)
    params[11] = Scalar[dtype](1.2)
    # latent x^(1) of dim 3
    lat[0] = Scalar[dtype](0.1)
    lat[1] = Scalar[dtype](0.2)
    lat[2] = Scalar[dtype](0.3)
    # input
    x_in[0] = Scalar[dtype](1.0)
    x_in[1] = Scalar[dtype](2.0)
    # one-hot label class 0
    y_tgt[0] = Scalar[dtype](1.0)
    y_tgt[1] = Scalar[dtype](0.0)

    # ── Wrap in LayoutTensor and run train_one_batch ──
    var p_t = LayoutTensor[
        dtype, Layout.row_major(TRAINER.MODEL.PARAM_SIZE), MutAnyOrigin
    ](params)
    var lat_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE),
        MutAnyOrigin,
    ](lat)
    var x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TRAINER.MODEL.IN_DIM), MutAnyOrigin
    ](x_in)
    var y_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TRAINER.MODEL.OUT_DIM), MutAnyOrigin
    ](y_tgt)

    var _ = TRAINER.train_one_batch[BATCH](
        p_t, lat_t, x_t, y_t,
        T_infer=1, T_learn=1,
        eta_infer=Scalar[dtype](0.05),
        eta_learn=Scalar[dtype](0.01),
    )

    # ── Expected values (Float64 hand computation, see header) ──
    var exp_lat: List[Float64] = [0.1214, 0.2332, 0.345]

    # Recomputed snapshot at x^(1)_new for the learning step:
    # a_0 = [0.16228, 0.37216]; eps_0 = h_0 = [0.83772, 1.62784]
    # a_R = [0.58204, 0.79192]; eps_sup = h_R = [-0.41796, 0.79192]
    # x_above_R = x_above_0 = x^(1)_new = [0.1214, 0.2332, 0.345]
    #
    # Non-readout: W_0 += +0.01 * (h_0^T @ x_above)
    #   h_0[0]=0.83772; h_0[1]=1.62784
    #   xa[0]=0.1214; xa[1]=0.2332; xa[2]=0.345
    #   delta_W_0[i,j] = 0.01 * h_0[i] * xa[j]
    #
    # Readout: W_R += -0.01 * (h_R^T @ x_above)
    #   h_R[0]=-0.41796; h_R[1]=0.79192
    #   delta_W_R[i,j] = -0.01 * h_R[i] * xa[j]

    var h0_0: Float64 = 0.83772
    var h0_1: Float64 = 1.62784
    var hR_0: Float64 = -0.41796
    var hR_1: Float64 = 0.79192
    var xa_0: Float64 = 0.1214
    var xa_1: Float64 = 0.2332
    var xa_2: Float64 = 0.345

    var exp_W0: List[Float64] = [
        0.1 + 0.01 * h0_0 * xa_0,
        0.2 + 0.01 * h0_0 * xa_1,
        0.3 + 0.01 * h0_0 * xa_2,
        0.4 + 0.01 * h0_1 * xa_0,
        0.5 + 0.01 * h0_1 * xa_1,
        0.6 + 0.01 * h0_1 * xa_2,
    ]
    var exp_WR: List[Float64] = [
        0.7 - 0.01 * hR_0 * xa_0,
        0.8 - 0.01 * hR_0 * xa_1,
        0.9 - 0.01 * hR_0 * xa_2,
        1.0 - 0.01 * hR_1 * xa_0,
        1.1 - 0.01 * hR_1 * xa_1,
        1.2 - 0.01 * hR_1 * xa_2,
    ]

    # ── Compare ──
    var TOL: Float64 = 1.0e-5
    var max_lat_err: Float64 = 0.0
    var max_W0_err: Float64 = 0.0
    var max_WR_err: Float64 = 0.0

    for i in range(3):
        var actual = Float64(lat[i])
        var err = abs(actual - exp_lat[i])
        if err > max_lat_err:
            max_lat_err = err

    for i in range(6):
        var actual = Float64(params[i])
        var err = abs(actual - exp_W0[i])
        if err > max_W0_err:
            max_W0_err = err

    for i in range(6):
        var actual = Float64(params[6 + i])
        var err = abs(actual - exp_WR[i])
        if err > max_WR_err:
            max_WR_err = err

    print("  max latent error:", max_lat_err)
    print("  max W_0 error   :", max_W0_err)
    print("  max W_R error   :", max_WR_err)

    var ok = True
    if max_lat_err >= TOL:
        print("  [FAIL] latent x^(1) mismatch")
        for i in range(3):
            print("    [", i, "] got", Float64(lat[i]), "expected", exp_lat[i])
        ok = False
    else:
        print("  [PASS] latent x^(1)")

    if max_W0_err >= TOL:
        print("  [FAIL] W_0 mismatch")
        for i in range(6):
            print("    [", i, "] got", Float64(params[i]), "expected", exp_W0[i])
        ok = False
    else:
        print("  [PASS] W_0 weight update")

    if max_WR_err >= TOL:
        print("  [FAIL] W_R mismatch")
        for i in range(6):
            print(
                "    [", i, "] got", Float64(params[6 + i]),
                "expected", exp_WR[i],
            )
        ok = False
    else:
        print("  [PASS] W_R weight update (readout sign convention)")

    params.free()
    lat.free()
    x_in.free()
    y_tgt.free()

    if not ok:
        raise Error("step-correctness test failed")
    print("=== Done ===")
