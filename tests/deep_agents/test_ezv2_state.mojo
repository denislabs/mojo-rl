"""Phase-2 Step 4 unit tests — EZ-V2 config + state.

Smoke checks that the new pieces tie together:

  1. `EZV2DiscreteMLPConfig` resolves to the expected dimensions
     (rep / dyn / pred / projector / predictor IN/OUT lines up with
     paper App. G).
  2. `EZV2DiscreteCPUState` constructs cleanly — all networks initialize,
     all scratch buffers allocate, no crashes.
  3. End-to-end forward pass of the consistency-loss path:

         rep(o) ──► projector ──► predictor ──►   …
                                        proj(rep(o)) ──► (target, detached)
                                                  ──► cosine_consistency_loss

     Verifies network-shape consistency from the encoder all the way
     through the SimSiam loss, on a single random batch with one of the
     branches synthetically forced to match the other (the loss should
     be near −1 in that case).

If (3) fails, the K-step training loop in step 5 won't have correct
plumbing for the consistency loss term.
"""

from std.math import abs
from std.random import seed
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteConfig,
    EZV2DiscreteMLPConfig,
    EZV2DiscreteCPUState,
    cosine_consistency_loss_forward,
)




def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _expect(
    cond: Bool,
    label: String,
    mut passed: Int,
    mut total: Int,
):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def main():
    print("=== EZ-V2 Phase 2 / Step 4 — config + state smoke test ===")
    var passed = 0
    var total = 0

    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 32
    comptime HIDDEN = 32
    comptime PROJ = 64
    comptime BOTTLE = 32
    comptime BINS = 21

    comptime Config = EZV2DiscreteMLPConfig[
        OBS=OBS,
        ACT=ACT,
        LATENT=LATENT,
        HIDDEN=HIDDEN,
        PROJ=PROJ,
        PRED_BOTTLENECK=BOTTLE,
        BINS=BINS,
        BS=8,
        K_UNROLL=3,
        N_TD=5,
        SIMS=8,
        NODES=32,
        K_GUMBEL=2,
    ]

    # ── 1. Compile-time dimensions ──────────────────────────────────────
    print()
    print("--- 1. Resolved Config dimensions ---")
    print(
        "    obs_dim     =", Config.obs_dim,
        " action_dim =", Config.action_dim,
    )
    print(
        "    latent_dim  =", Config.latent_dim,
        " proj_dim   =", Config.proj_dim,
    )
    print(
        "    DYN_IN/OUT  =", Config.DYN_IN, "/", Config.DYN_OUT,
    )
    print(
        "    PRED_OUT    =", Config.PRED_OUT,
    )
    _expect(Config.obs_dim == OBS, "Config.obs_dim", passed, total)
    _expect(Config.action_dim == ACT, "Config.action_dim", passed, total)
    _expect(Config.latent_dim == LATENT, "Config.latent_dim", passed, total)
    _expect(Config.proj_dim == PROJ, "Config.proj_dim", passed, total)
    _expect(
        Config.DYN_IN == LATENT + ACT,
        "Config.DYN_IN = LATENT + ACT",
        passed, total,
    )
    _expect(
        Config.DYN_OUT == LATENT + BINS,
        "Config.DYN_OUT = LATENT + BINS",
        passed, total,
    )
    _expect(
        Config.PRED_OUT == ACT + BINS,
        "Config.PRED_OUT = ACT + BINS",
        passed, total,
    )

    print(
        "    Projector IN/OUT =",
        Config.ProjectorModel.IN_DIM,
        "/",
        Config.ProjectorModel.OUT_DIM,
    )
    print(
        "    Predictor IN/OUT =",
        Config.PredictorModel.IN_DIM,
        "/",
        Config.PredictorModel.OUT_DIM,
    )
    _expect(
        Config.ProjectorModel.IN_DIM == LATENT,
        "Projector.IN_DIM == latent_dim",
        passed, total,
    )
    _expect(
        Config.ProjectorModel.OUT_DIM == PROJ,
        "Projector.OUT_DIM == proj_dim",
        passed, total,
    )
    _expect(
        Config.PredictorModel.IN_DIM == PROJ,
        "Predictor.IN_DIM == proj_dim",
        passed, total,
    )
    _expect(
        Config.PredictorModel.OUT_DIM == PROJ,
        "Predictor.OUT_DIM == proj_dim (asymmetric bottleneck inside)",
        passed, total,
    )

    # Loss weights from paper Table 3 (defaults).
    _expect(
        Config.lambda_reward == 1.0
        and Config.lambda_policy == 1.0
        and Config.lambda_value == 0.25
        and Config.lambda_consistency == 2.0,
        "default loss weights match paper Eq. 3",
        passed, total,
    )
    _expect(
        Config.t_fresh == 20000 and Config.t_stale == 40000,
        "default mixed-value-target thresholds match paper Table 3",
        passed, total,
    )

    # ── 2. State construction ────────────────────────────────────────────
    print()
    print("--- 2. EZV2DiscreteCPUState construction ---")
    seed(2026)
    var state = EZV2DiscreteCPUState[Config, _CAP=128]()
    _expect(True, "state constructed without crash", passed, total)

    print(
        "    Rep PARAM_SIZE       =",
        Config.RepModel.PARAM_SIZE,
    )
    print(
        "    Projector PARAM_SIZE =",
        Config.ProjectorModel.PARAM_SIZE,
    )
    print(
        "    Predictor PARAM_SIZE =",
        Config.PredictorModel.PARAM_SIZE,
    )
    _expect(
        Config.ProjectorModel.PARAM_SIZE > 0,
        "Projector params allocated",
        passed, total,
    )
    _expect(
        Config.PredictorModel.PARAM_SIZE > 0,
        "Predictor params allocated",
        passed, total,
    )

    # ── 3. End-to-end consistency-loss forward path ──────────────────────
    print()
    print("--- 3. End-to-end consistency-loss forward ---")
    comptime BATCH = 4

    # Random observations.
    var obs_arr = InlineArray[Scalar[dtype], BATCH * OBS](uninitialized=True)
    for i in range(BATCH * OBS):
        obs_arr[i] = Scalar[dtype](0.13 * Float64(i % 11) - 0.5)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](obs_arr.unsafe_ptr())

    # Build params/model-state LayoutTensor views from the raw pointers
    # to bypass `params_view()` — Mojo nightly's overload resolution
    # rejects the method call when the struct field's declared type
    # carries an alias chain that the method's `self` was instantiated
    # against, even though both sides print as the same type.
    var rep_params = LayoutTensor[
        dtype, Layout.row_major(Config.RepModel.PARAM_SIZE), MutAnyOrigin
    ](state.representation.params)
    var rep_state_buf = LayoutTensor[
        dtype, Layout.row_major(Config.RepModel.STATE_SIZE), MutAnyOrigin
    ](state.representation.model_state)
    var lat_arr = InlineArray[Scalar[dtype], BATCH * LATENT](
        uninitialized=True
    )
    var lat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](lat_arr.unsafe_ptr())
    Network[Config.RepModel, Config.OptType].forward[BATCH](
        obs_t, lat_t, rep_params, rep_state_buf
    )

    var proj_params = LayoutTensor[
        dtype,
        Layout.row_major(Config.ProjectorModel.PARAM_SIZE),
        MutAnyOrigin,
    ](state.projector.params)
    var proj_state_buf = LayoutTensor[
        dtype,
        Layout.row_major(Config.ProjectorModel.STATE_SIZE),
        MutAnyOrigin,
    ](state.projector.model_state)
    var proj_arr = InlineArray[Scalar[dtype], BATCH * PROJ](uninitialized=True)
    var proj_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PROJ), MutAnyOrigin
    ](proj_arr.unsafe_ptr())
    Network[Config.ProjectorModel, Config.OptType].forward[BATCH](
        lat_t, proj_t, proj_params, proj_state_buf
    )

    var pred_params = LayoutTensor[
        dtype,
        Layout.row_major(Config.PredictorModel.PARAM_SIZE),
        MutAnyOrigin,
    ](state.predictor.params)
    var pred_state_buf = LayoutTensor[
        dtype,
        Layout.row_major(Config.PredictorModel.STATE_SIZE),
        MutAnyOrigin,
    ](state.predictor.model_state)
    var online_arr = InlineArray[Scalar[dtype], BATCH * PROJ](
        uninitialized=True
    )
    var online_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PROJ), MutAnyOrigin
    ](online_arr.unsafe_ptr())
    Network[Config.PredictorModel, Config.OptType].forward[BATCH](
        proj_t, online_t, pred_params, pred_state_buf
    )

    # Step 4: cosine loss between online (predictor output) and proj
    # (projector output), treating proj as detached. With a *fresh*
    # randomly-initialized predictor whose output happens to be
    # uncorrelated with proj, the loss should be small in magnitude
    # (close to 0, the orthogonal regime), not near ±1.
    var loss_random = cosine_consistency_loss_forward[BATCH, PROJ](
        online_t, proj_t
    )
    print("    fresh-init consistency loss =", loss_random)
    _expect(
        _abs(loss_random) <= 1.0 + 1e-6,
        "loss bounded in [-1, 1]",
        passed, total,
    )

    # Step 5: when online ≡ proj (forced by copying), loss should be ≈ −1.
    for i in range(BATCH * PROJ):
        online_arr[i] = proj_arr[i]
    var loss_match = cosine_consistency_loss_forward[BATCH, PROJ](
        online_t, proj_t
    )
    print("    online == proj loss        =", loss_match)
    _expect(
        _abs(loss_match - (-1.0)) < 1e-3,
        "online == proj → loss ≈ −1",
        passed, total,
    )

    # ── 4. Replay buffer is live + empty ─────────────────────────────────
    print()
    print("--- 4. Replay buffer ---")
    print("    is_ready =", state.is_ready())
    _expect(
        not state.is_ready(),
        "fresh state has empty replay buffer (not ready)",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
