"""Compile-check smoke test for PCDynamics + PCDynamicsEnsemble.

Just allocates buffers, runs init/predict/compute_grads. No training,
no convergence assertions — purely a "does it build and not crash"
gate before wiring into PCN-MBPO.
"""

from std.memory import alloc, memset
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.experimental.nn_pc_v2 import (
    PCDynamics,
    PCDynamicsEnsemble,
)


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 32
comptime BATCH = 4
comptime N_ENSEMBLE = 3
comptime N_ELITES = 2
comptime DYN = PCDynamics[OBS, ACT, HIDDEN, dtype]
comptime ENS = PCDynamicsEnsemble[OBS, ACT, HIDDEN, N_ENSEMBLE, N_ELITES, dtype]
comptime OPT = Adam[LR=0.001]


def main() raises:
    print("Smoke: PCDynamics + PCDynamicsEnsemble compile-check")
    print("  PER_MEMBER_PARAM_SIZE =", ENS.PER_MEMBER_PARAM_SIZE)
    print("  TOTAL_PARAM_SIZE      =", ENS.TOTAL_PARAM_SIZE)

    # Allocate ensemble buffers.
    var params_buf = alloc[Scalar[dtype]](ENS.TOTAL_PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](ENS.TOTAL_PARAM_SIZE)
    var opt_state_buf = alloc[Scalar[dtype]](
        ENS.TOTAL_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var opt_global_buf = alloc[Scalar[dtype]](
        N_ENSEMBLE * OPT.GLOBAL_STATE_SIZE
    )
    memset(params_buf, 0, ENS.TOTAL_PARAM_SIZE)
    memset(grads_buf, 0, ENS.TOTAL_PARAM_SIZE)
    memset(opt_state_buf, 0, ENS.TOTAL_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(opt_global_buf, 0, N_ENSEMBLE * OPT.GLOBAL_STATE_SIZE)

    ENS.init_all(params_buf, base_seed=UInt64(42))
    print("  init_all OK")

    # Allocate per-member SGLD scratch (shared, reused across members).
    var lat_buf = alloc[Scalar[dtype]](BATCH * DYN.SCRATCH_LAT)
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * DYN.SCRATCH_OUT)
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * DYN.SCRATCH_IN)
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * DYN.SCRATCH_IN)
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * DYN.SCRATCH_LAT)
    memset(lat_buf, 0, BATCH * DYN.SCRATCH_LAT)
    memset(mu_eps_buf_raw, 0, BATCH * DYN.SCRATCH_OUT)
    memset(a_below_buf_raw, 0, BATCH * DYN.SCRATCH_IN)
    memset(z_below_buf_raw, 0, BATCH * DYN.SCRATCH_IN)
    memset(dx_buf_raw, 0, BATCH * DYN.SCRATCH_LAT)

    var latents = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_LAT), MutAnyOrigin
    ](lat_buf)
    var mu_eps_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_OUT), MutAnyOrigin
    ](mu_eps_buf_raw)
    var a_below_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_IN), MutAnyOrigin
    ](a_below_buf_raw)
    var z_below_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_IN), MutAnyOrigin
    ](z_below_buf_raw)
    var dx_buf = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_LAT), MutAnyOrigin
    ](dx_buf_raw)

    # Inputs: (s, a) and target (s_next, r). Just zeros for smoke test.
    var s_a_buf = alloc[Scalar[dtype]](BATCH * DYN.AUG_DIM)
    var target_buf = alloc[Scalar[dtype]](BATCH * DYN.READOUT)
    memset(s_a_buf, 0, BATCH * DYN.AUG_DIM)
    memset(target_buf, 0, BATCH * DYN.READOUT)
    # Set some non-zero values so SGLD has something to settle.
    for b in range(BATCH):
        s_a_buf[b * DYN.AUG_DIM + 0] = Scalar[dtype](0.5)
        target_buf[b * DYN.READOUT + 0] = Scalar[dtype](0.6)
    var s_a = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.AUG_DIM), MutAnyOrigin
    ](s_a_buf)
    var target = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.READOUT), MutAnyOrigin
    ](target_buf)

    # Predict scratch.
    var a_aug_buf = alloc[Scalar[dtype]](BATCH * DYN.AUG_DIM)
    var z_hidden_buf = alloc[Scalar[dtype]](BATCH * DYN.HIDDEN_DIM)
    var a_z_buf = alloc[Scalar[dtype]](BATCH * DYN.HIDDEN_DIM)
    var out_buf = alloc[Scalar[dtype]](BATCH * DYN.READOUT)
    var a_aug = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.AUG_DIM), MutAnyOrigin
    ](a_aug_buf)
    var z_hidden = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.HIDDEN_DIM), MutAnyOrigin
    ](z_hidden_buf)
    var a_z = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.HIDDEN_DIM), MutAnyOrigin
    ](a_z_buf)
    var out = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.READOUT), MutAnyOrigin
    ](out_buf)

    # Predict member 0 (smoke test only).
    ENS.predict_member[BATCH](
        0, s_a, params_buf, a_aug, z_hidden, a_z, out
    )
    print("  predict_member(0) OK; out[0,0]=", Float64(out_buf[0]))

    # Train member 0 for a few steps. Adam step counter is per-member.
    var step_num: Int = 0
    for _ in range(3):
        var loss = ENS.train_member[BATCH, OPT](
            0, s_a, target,
            params_buf, grads_buf, opt_state_buf, opt_global_buf,
            latents, mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
            step_num,
            T_infer=5,
            lr_x=Scalar[dtype](0.01),
        )
        print("  train_member(0) loss=", loss)

    # Eval each member's holdout loss.
    var losses = List[Float64](capacity=N_ENSEMBLE)
    for m in range(N_ENSEMBLE):
        var L = ENS.eval_member_loss[BATCH](
            m, s_a, target, params_buf, a_aug, z_hidden, a_z, out
        )
        losses.append(L)
        print("  member", m, " holdout loss=", L)

    # Elite selection.
    var elites = List[Int]()
    ENS.select_elites(losses, elites)
    print("  elites:", elites)

    # Cleanup
    params_buf.free()
    grads_buf.free()
    opt_state_buf.free()
    opt_global_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    s_a_buf.free()
    target_buf.free()
    a_aug_buf.free()
    z_hidden_buf.free()
    a_z_buf.free()
    out_buf.free()
    print("=== Smoke test OK ===")
