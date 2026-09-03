# +--------------------------------------------------------------------------+ #
# | SmolVLA — connector, token embedding, LM head, and the action heads
# +--------------------------------------------------------------------------+ #
"""The thirteen tensors that are neither tower nor expert.

    model.connector.modality_projection.proj.weight   [960, 12288]   BF16
    model.text_model.embed_tokens.weight              [49280, 960]   BF16
    lm_head.weight                                    [49280, 960]   BF16
    model.state_proj.{weight,bias}                    [960, 32]      F32
    model.action_in_proj.{weight,bias}                [720, 32]      F32
    model.action_out_proj.{weight,bias}               [32, 720]      F32
    model.action_time_mlp_in.{weight,bias}            [720, 1440]    F32
    model.action_time_mlp_out.{weight,bias}           [720, 720]     F32

⚠ **32 is padding, not the robot.** `max_state_dim` and `max_action_dim` are
both 32 in `config.json`; the SO-101's 6 joints sit inside a 32-vector with the
rest zero. The projections are sized for 32 and must stay that way — narrowing
them to 6 would be a differently shaped model that no checkpoint fits.

⚠ **`action_time_mlp_in` takes 1440 = 2 x 720**: the noisy action embedding
concatenated with the time embedding. The time half is what makes the expert a
flow-matching denoiser rather than a plain decoder.

The connector consumes `PixelShuffle`'s output — 64 tokens of 12288 — and is the
only place the 12288 appears; it is `768 * 4^2`, the vision width times the
square of `scale_factor`.

Bias handling splits: the three BF16 tensors are bias-free in the checkpoint
(`TN_ZEROS` for our `Linear`'s always-present bias), while all five F32 action
heads DO ship a bias. Getting that backwards zeroes a real, trained bias.
"""

from mojo_rl.nn.combinators.tokenwise import Tokenwise
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.embedding import Embedding


comptime SMOLVLA_STATE_DIM: Int = 32   # max_state_dim — padded, not the robot
comptime SMOLVLA_ACTION_DIM: Int = 32  # max_action_dim
comptime SMOLVLA_EXPERT_W: Int = 720
comptime SMOLVLA_CONNECTOR_IN: Int = 12288  # 768 * scale_factor^2


comptime SmolVLMConnector[
    TOKENS: Int, IN_C: Int = SMOLVLA_CONNECTOR_IN, DIM: Int = 960
] = Tokenwise[TOKENS, Linear[IN_C, DIM]]

comptime SmolVLATokenEmbed[VOCAB: Int = 49280, DIM: Int = 960] = Embedding[
    VOCAB, DIM
]

comptime SmolVLALMHead[DIM: Int = 960, VOCAB: Int = 49280] = Linear[DIM, VOCAB]

comptime SmolVLAStateProj[
    S: Int = SMOLVLA_STATE_DIM, DIM: Int = 960
] = Linear[S, DIM]

comptime SmolVLAActionIn[
    A: Int = SMOLVLA_ACTION_DIM, W: Int = SMOLVLA_EXPERT_W
] = Linear[A, W]

comptime SmolVLAActionOut[
    W: Int = SMOLVLA_EXPERT_W, A: Int = SMOLVLA_ACTION_DIM
] = Linear[W, A]

comptime SmolVLATimeMLPIn[W: Int = SMOLVLA_EXPERT_W] = Linear[2 * W, W]

comptime SmolVLATimeMLPOut[W: Int = SMOLVLA_EXPERT_W] = Linear[W, W]
