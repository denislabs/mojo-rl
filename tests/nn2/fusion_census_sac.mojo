"""Static fusion census of the SAC ComputeGraphs (target_y + actor_loss).

Runs `FusionReportExporter` over the *real* graph definitions (referenced
via the block comptime aliases so they stay in sync). ACTOR / CRITIC are
opaque ExternalNodes in the DAG, so minimal `Linear` stand-ins with the
right dims suffice — the census measures the graph-level element-wise tail,
not the nets' internals. Walker2d dims: OBS=17, ACT=6 (SA=23).

Run: pixi run mojo run -I . tests/nn2/fusion_census_sac.mojo
"""

from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.combinators.graph_export import FusionReportExporter
from mojo_rl.deep_agents2.sac.target_y_block import TargetYBlock
from mojo_rl.deep_agents2.sac.actor_loss import SACActorLoss


comptime OBS = 17
comptime ACT = 6
comptime SA = OBS + ACT
comptime BATCH = 256

# Opaque stand-ins for the actor (→ 2·ACT) and critic (→ 1).
comptime ActorStub = Linear[OBS, 2 * ACT]
comptime CriticStub = Linear[SA, 1]

comptime TargetYGraphT = TargetYBlock[
    ActorStub, CriticStub, BATCH, OBS, ACT
].TargetYGraph
comptime ActorGraphT = SACActorLoss[ActorStub, CriticStub, BATCH].ActorGraph


def main() raises:
    print("=" * 70)
    print("SAC ComputeGraph fusion census (walker2d dims OBS=17 ACT=6)")
    print("=" * 70)

    var g1 = TargetYGraphT()
    var rep1 = FusionReportExporter()
    g1.describe(rep1, String("SAC target_y"))
    print(rep1.out)

    var g2 = ActorGraphT()
    var rep2 = FusionReportExporter()
    g2.describe(rep2, String("SAC actor_loss"))
    print(rep2.out)
