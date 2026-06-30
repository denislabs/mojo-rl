"""Storage ComputeGraph export: Text + Mermaid + FusionReport exporters.

Builds a storage ComputeGraph with real op labels (Linear / ReLU / Tanh /
Concat), a pointwise chain (ReLU→Tanh, a fusable region), a memory-glue node
(Concat), and a container node (a `Node` wrapping a `Sequential`, exercising the
`display_steps` → `node_inner` expansion). Runs `graph.describe` into each
exporter and asserts the rendered topology, container expansion, and fusion
census. Pure comptime topology — no make/forward/device.

Run: pixi run mojo run -I . tests/nn/test_storage_graph_export.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.concat import Concat2
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_export import (
    TextExporter, MermaidExporter, FusionReportExporter,
)


comptime IN = 4
comptime H = 8

comptime G = ComputeGraph[
    InputSlot["x", IN],
    Node["lin", Linear[IN, H], "x"],  # Linear (matmul)
    Node["r", ReLU[H], "lin"],  # ReLU (pointwise)
    Node["t", Tanh[H], "r"],  # Tanh (pointwise) → {r,t} fusable
    Node["cat", Concat2[IN, H], "x", "t"],  # Concat (memory), reads x + t → 12
    # Container: a Sequential op → describe expands it into node_inner steps.
    Node["head", Sequential[Linear[IN + H, H], ReLU[H]], "cat"],
]


def _has(haystack: String, needle: String) -> Bool:
    return haystack.find(needle) >= 0


def main() raises:
    print("=" * 60)
    print("storage ComputeGraph export (Text / Mermaid / Fusion)")
    print("=" * 60)
    var g = G()

    var tex = TextExporter()
    g.describe("demo", tex)
    print("\n--- TextExporter ---")
    print(tex.out)
    assert_true(_has(tex.out, "6 nodes"), "text node count")
    # Real op labels (not category tags).
    assert_true(_has(tex.out, "Linear"), "text Linear label")
    assert_true(_has(tex.out, "ReLU"), "text ReLU label")
    assert_true(_has(tex.out, "Tanh"), "text Tanh label")
    assert_true(_has(tex.out, "Concat"), "text Concat label")
    assert_true(_has(tex.out, "<- x, t"), "text multi-edge into cat")
    # Container expansion (node_inner) — Sequential children listed under head.
    assert_true(_has(tex.out, "└ Linear"), "text container inner Linear")
    assert_true(_has(tex.out, "└ ReLU"), "text container inner ReLU")

    var mer = MermaidExporter()
    g.describe("demo", mer)
    print("\n--- MermaidExporter ---")
    print(mer.out)
    assert_true(_has(mer.out, "graph TD"), "mermaid header")
    assert_true(_has(mer.out, 'x(['), "mermaid input stadium")
    assert_true(_has(mer.out, "subgraph head"), "mermaid container subgraph")

    var fus = FusionReportExporter()
    g.describe("demo", fus)
    print("\n--- FusionReportExporter ---")
    print(fus.out)
    assert_true(_has(fus.out, "FusionReport"), "fusion header")
    # r (ReLU) → t (Tanh) is a pointwise chain of size 2 → 1 launch saved.
    assert_true(_has(fus.out, "{r, t}"), "fusion region {r, t}")
    assert_true(_has(fus.out, "saves 1"), "fusion saves 1 launch")
    assert_true(_has(fus.out, "1 region(s), 1 launch(es) eliminable"),
                "fusion summary")
    assert_true(_has(fus.out, "memory-glue nodes (Concat/Slice): 1"),
                "fusion memory count")

    print("\nALL PASSED")
