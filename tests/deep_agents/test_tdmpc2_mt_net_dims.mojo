"""MT net widths — the task embedding widens the fan-IN, never the OUTPUT.

`nets_mt.mojo` builds each multi-task net by reusing the single-task alias with
`LATENT' = LATENT + TASK_EMB`. That substitution is only valid when the widened
parameter does not also set the net's OUTPUT width:

    reward / Q     → BINS          unaffected
    termination    → 1             unaffected
    policy         → 2*MAX_ACT     unaffected
    encoder        → takes LATENT as a SEPARATE parameter
    dynamics       → `NormedLinearSimNorm[MLP, LATENT, SN]`  ← ALSO the output

So the alias silently gave `TDMPC2DynamicsMT.OUT_DIM = LATENT + TASK_EMB` (544
at the walker's dims) while the encoder — and therefore the consistency target
`z_enc_next` — stayed at 512. Nothing raised, because `MSELossPlain[LATENT]`
and the carry `Concat[8, LATENT]` each read only the first LATENT columns. The
world model just never converged: `consistency_loss` sat at ~0.16 and ROSE,
against 0.033 → 0.008 for single-task, while `reward_loss` and `value_loss`
looked completely normal — precisely because their outputs were unaffected.

The gate is a width identity, not a training run, so it costs milliseconds and
would have caught this before any GPU time was spent:

    dynamics.OUT_DIM == encoder.OUT_DIM == LATENT

Both are the latent space the consistency MSE compares and BPTT carries; if
they ever disagree again, the world model is broken no matter what the loss
curves look like.

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_mt_net_dims.mojo`
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.deep_agents.tdmpc2.nets import TDMPC2Dynamics, TDMPC2Encoder
from mojo_rl.deep_agents.tdmpc2.nets_mt import (
    TDMPC2DynamicsMT, TDMPC2EncoderMT, TDMPC2RewardMT, TDMPC2QNetMT,
    TDMPC2TerminationMT, TDMPC2PolicyMT,
)

# The walker multi-task dims — the configuration that failed.
comptime LATENT = 512
comptime MAX_ACT = 6
comptime MLP = 512
comptime SN = 8
comptime EMB = 32
comptime MAX_OBS = 24
comptime ENC = 256
comptime BINS = 101


def test_dynamics_out_dim_is_latent_not_widened() raises:
    """The defect gate. `OUT_DIM` must be LATENT, never LATENT + TASK_EMB."""
    comptime got = TDMPC2DynamicsMT[LATENT, MAX_ACT, MLP, SN, EMB].OUT_DIM
    print("  MT dynamics OUT_DIM =", got, " (LATENT =", LATENT, ")")
    assert_equal(
        got, LATENT,
        msg=(
            "MT dynamics output is widened by TASK_EMB — `znext` no longer"
            " matches the encoder latent that the consistency MSE compares it"
            " against, and the world model cannot converge"
        ),
    )
    assert_true(
        got != LATENT + EMB,
        "OUT_DIM is exactly LATENT+TASK_EMB — this is the aliasing bug",
    )


def test_dynamics_and_encoder_agree_on_the_latent_space() raises:
    """The invariant that actually matters: both ends of the consistency MSE."""
    comptime d = TDMPC2DynamicsMT[LATENT, MAX_ACT, MLP, SN, EMB].OUT_DIM
    comptime e = TDMPC2EncoderMT[MAX_OBS, ENC, LATENT, SN, EMB].OUT_DIM
    print("  dynamics =", d, "  encoder =", e)
    assert_equal(
        d, e,
        msg=(
            "dynamics and encoder disagree on the latent width — `znext` and"
            " `z_enc_next` are not in the same space"
        ),
    )


def test_mt_matches_single_task_output_widths() raises:
    """Every MT net must agree with its single-task counterpart on OUT_DIM —
    the embedding conditions the input and changes nothing downstream."""
    assert_equal(
        TDMPC2DynamicsMT[LATENT, MAX_ACT, MLP, SN, EMB].OUT_DIM,
        TDMPC2Dynamics[LATENT, MAX_ACT, MLP, SN].OUT_DIM,
        msg="MT dynamics OUT_DIM must equal single-task dynamics OUT_DIM",
    )
    assert_equal(
        TDMPC2EncoderMT[MAX_OBS, ENC, LATENT, SN, EMB].OUT_DIM,
        TDMPC2Encoder[MAX_OBS, ENC, LATENT, SN].OUT_DIM,
        msg="MT encoder OUT_DIM must equal single-task encoder OUT_DIM",
    )


def test_heads_emit_their_own_widths() raises:
    """The nets the alias trick IS valid for — asserted so a future 'fix' that
    over-corrects by widening these too gets caught."""
    assert_equal(
        TDMPC2RewardMT[LATENT, MAX_ACT, MLP, BINS, EMB].OUT_DIM, BINS,
        msg="reward head emits BINS",
    )
    assert_equal(
        TDMPC2QNetMT[LATENT, MAX_ACT, MLP, BINS, EMB].OUT_DIM, BINS,
        msg="Q head emits BINS",
    )
    assert_equal(
        TDMPC2TerminationMT[LATENT, MAX_ACT, MLP, EMB].OUT_DIM, 1,
        msg="termination head emits 1 logit",
    )
    assert_equal(
        TDMPC2PolicyMT[LATENT, MAX_ACT, MLP, EMB].OUT_DIM, 2 * MAX_ACT,
        msg="policy emits [mean | log_std]",
    )
    print("  heads: BINS =", BINS, " term = 1  policy =", 2 * MAX_ACT)


def main() raises:
    print("=" * 70)
    print("TD-MPC2 multi-task net widths")
    print("=" * 70)
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("=" * 70)
    print("MT NET DIMS PASSED")
    print("=" * 70)
