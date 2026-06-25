"""bf16 compile-smoke for the EZv2 Atari VP-path nets (EZv2-bf16 step 2a).

Confirms the value-prefix-path nets are bf16-INSTANTIABLE after ADT threading:
`ACT_DT == bfloat16` at `ADT=bf16`, and the fp32 default is unchanged. Compile +
comptime asserts only (Apple bf16 GEMM is broken → no bf16 forward run).

Covers rep (EZRepNetResNetAtari), pred (EZPredNetAtari), the action-plane
(EZActionPlane), the z'-only dynamics graph (EZDynZGraph) and its wrapper
(EZDynZNetAtari). The dynamics path needed two ops to gain an ADT seam first:
`BroadcastTokens` (broadcast_tokens.mojo) and `InputSlot` (graph_decl.mojo) — both
now carry `ADT: DType = DT`. The reward LSTM (EZRewardLSTMAtari) stays fp32
(LSTMCell has no bf16 path yet — deferred).

Run: pixi run mojo run -I . tests/deep_agents/test_ezv2_nets_bf16_smoke.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import (
    EZRepNetResNetAtari, EZPredNetAtari, EZActionPlane, EZDynZGraph,
    EZDynZNetAtari,
)

comptime BF16 = DType.bfloat16


def main() raises:
    comptime IN_CH = 12     # n_stack*3 (RGB)
    comptime C = 64         # num_channels
    comptime ACT = 6        # e.g. Pong
    comptime BINS = 601

    # bf16-flow: ACT_DT propagates to bf16 through Sequential / Parallel /
    # ComputeGraph (all decls incl. InputSlot now match the nodes' ADT).
    comptime assert (
        EZRepNetResNetAtari[IN_CH, C, ADT=BF16].ACT_DT == BF16
    ), "rep must flow bf16"
    comptime assert (
        EZPredNetAtari[ACT, BINS, ADT=BF16].ACT_DT == BF16
    ), "pred must flow bf16"
    comptime assert (
        EZActionPlane[ACT, ADT=BF16].ACT_DT == BF16
    ), "action-plane must flow bf16"
    comptime assert (
        EZDynZGraph[ACT, ADT=BF16].ACT_DT == BF16
    ), "dyn-z graph must flow bf16"
    comptime assert (
        EZDynZNetAtari[ACT, ADT=BF16].ACT_DT == BF16
    ), "dyn-z net must flow bf16"

    # fp32 default unchanged (ADT defaults to DT).
    comptime assert (
        EZRepNetResNetAtari[IN_CH, C].ACT_DT == DT
    ), "rep fp32 default"
    comptime assert (
        EZPredNetAtari[ACT, BINS].ACT_DT == DT
    ), "pred fp32 default"
    comptime assert (
        EZDynZNetAtari[ACT].ACT_DT == DT
    ), "dyn-z fp32 default"

    print(
        "bf16 nets compile: rep+pred+action+dyn-z ACT_DT==bf16; fp32 default"
        " intact"
    )
