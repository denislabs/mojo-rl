"""TD-MPC2 multi-task network aliases (item C, §14.3).

Multi-task conditions every net on a learned task embedding concatenated into
its input (reference `cfg.multitask`: a per-task `_task_emb` row appended to the
obs / latent inputs). Because the embedding *widens the first-layer fan-in*
(`OBS → MAX_OBS+TASK_EMB`, `LATENT+ACT → LATENT+ACT+TASK_EMB`), this is a
different `Linear[I,O]` *type* — Mojo cannot conditionally alias it against the
single-task type (see memory `feedback_mojo_conditional_type_alias_blocked`). So
the multi-task path is a parallel alias/struct set; the single-task nets stay at
zero diff and are bit-identical by construction.

No new layer types are needed: each MT net is just the existing net with a wider
first layer. The 3-way input concat `[z | a | task_emb]` (width
`LATENT+ACT+TASK_EMB`) feeds `Linear[(LATENT+TASK_EMB)+ACT, MLP]` — i.e. the
existing alias with `LATENT' = LATENT+TASK_EMB`. The encoder takes
`MAX_OBS+TASK_EMB`; the policy takes `LATENT+TASK_EMB`.

⚠⚠ That `LATENT' = LATENT+TASK_EMB` substitution is valid ONLY for nets whose
OUTPUT width does not depend on the parameter being widened — reward and Q emit
`BINS`, termination emits 1, the policy emits `2*MAX_ACT`, and the encoder takes
`LATENT` as a separate parameter. `TDMPC2Dynamics` is the one exception: its
first parameter sets BOTH the fan-in and the final `NormedLinearSimNorm[MLP,
LATENT, SN]`, so aliasing it widened the OUTPUT too —
`TDMPC2DynamicsMT.OUT_DIM` was `LATENT+TASK_EMB` (544 at the walker's dims)
while the encoder, and therefore the consistency target `z_enc_next`, stayed at
512. `MSELossPlain[LATENT]` and the carry `Concat[8, LATENT]` both read only
the first `LATENT` columns, so nothing raised: the world model simply never
converged. Measured cost — `consistency_loss` 0.16 rising vs 0.033 → 0.008 for
single-task, with `reward_loss` and `value_loss` NORMAL because their outputs
were unaffected. `TDMPC2DynamicsMT` is therefore written out in full below,
NOT aliased. Gated by `tests/deep_agents/test_tdmpc2_mt_net_dims.mojo`.
"""

from mojo_rl.nn.combinators.sequential import Sequential

from .nets import (
    TDMPC2Encoder, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy,
    TDMPC2Termination, NormedLinear, NormedLinearSimNorm,
)


comptime TDMPC2EncoderMT[
    MAX_OBS: Int, ENC: Int, LATENT: Int, SN: Int, TASK_EMB: Int
] = TDMPC2Encoder[MAX_OBS + TASK_EMB, ENC, LATENT, SN]

# ⚠ Written out, NOT `TDMPC2Dynamics[LATENT + TASK_EMB, ...]`. The task
# embedding widens the fan-IN only; the output must stay `LATENT` so `znext`
# matches the encoder's `z_enc_next` that the consistency MSE compares it to,
# and the carry that feeds the next BPTT step. See the module docstring.
comptime TDMPC2DynamicsMT[
    LATENT: Int, MAX_ACT: Int, MLP: Int, SN: Int, TASK_EMB: Int
] = Sequential[
    NormedLinear[LATENT + TASK_EMB + MAX_ACT, MLP],
    NormedLinear[MLP, MLP],
    NormedLinearSimNorm[MLP, LATENT, SN],
]

comptime TDMPC2RewardMT[
    LATENT: Int, MAX_ACT: Int, MLP: Int, BINS: Int, TASK_EMB: Int
] = TDMPC2Reward[LATENT + TASK_EMB, MAX_ACT, MLP, BINS]

comptime TDMPC2QNetMT[
    LATENT: Int, MAX_ACT: Int, MLP: Int, BINS: Int, TASK_EMB: Int,
    QP: Float64 = 0.0,
] = TDMPC2QNet[LATENT + TASK_EMB, MAX_ACT, MLP, BINS, QP]

comptime TDMPC2TerminationMT[
    LATENT: Int, MAX_ACT: Int, MLP: Int, TASK_EMB: Int
] = TDMPC2Termination[LATENT + TASK_EMB, MAX_ACT, MLP]

comptime TDMPC2PolicyMT[
    LATENT: Int, MAX_ACT: Int, MLP: Int, TASK_EMB: Int
] = TDMPC2Policy[LATENT + TASK_EMB, MAX_ACT, MLP]
