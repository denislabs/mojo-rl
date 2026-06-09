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
"""

from .nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy,
    TDMPC2Termination,
)


comptime TDMPC2EncoderMT[
    MAX_OBS: Int, ENC: Int, LATENT: Int, SN: Int, TASK_EMB: Int
] = TDMPC2Encoder[MAX_OBS + TASK_EMB, ENC, LATENT, SN]

comptime TDMPC2DynamicsMT[
    LATENT: Int, MAX_ACT: Int, MLP: Int, SN: Int, TASK_EMB: Int
] = TDMPC2Dynamics[LATENT + TASK_EMB, MAX_ACT, MLP, SN]

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
