"""Storage ComputeGraph topology export: TextExporter + MermaidExporter.

Builds a small 3-node storage ComputeGraph (InputSlot → Linear → ReLU) and runs
`graph.describe` into each exporter, asserting the rendered topology mentions
every node, the edges, and the right per-node output widths. Pure comptime
topology — no make/forward/device needed.

Run: pixi run mojo run -I . tests/nn/test_storage_graph_export.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.primitives.concat import Concat2
from mojo_rl.nn.storage.combinators.graph_decl import InputSlot, Node
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.graph_export import (
    TextExporter, MermaidExporter,
)


comptime IN = 4
comptime H = 8

comptime G = ComputeGraph[
    InputSlot["x", IN],
    Node["lin", Linear[IN, H], "x"],
    Node["act", ReLU[H], "lin"],
    # Multi-input (ARITY=2): reads the raw input "x" (4) and "act" (8) → 12.
    Node["cat", Concat2[IN, H], "x", "act"],
]


def _has(haystack: String, needle: String) -> Bool:
    return haystack.find(needle) >= 0


def main() raises:
    print("=" * 60)
    print("storage ComputeGraph topology export")
    print("=" * 60)
    var g = G()

    var tex = TextExporter()
    g.describe("mlp3", tex)
    print("\n--- TextExporter ---")
    print(tex.out)
    assert_true(_has(tex.out, "ComputeGraph"), "text header")
    assert_true(_has(tex.out, "mlp3"), "text graph name")
    assert_true(_has(tex.out, "4 nodes"), "text node count")
    assert_true(_has(tex.out, "x"), "text input node")
    assert_true(_has(tex.out, "lin"), "text lin node")
    assert_true(_has(tex.out, "act"), "text act node")
    assert_true(_has(tex.out, "cat"), "text cat node")
    # lin reads x; act reads lin; cat reads x AND act (two srcs).
    assert_true(_has(tex.out, "<- x"), "text edge x->lin")
    assert_true(_has(tex.out, "<- lin"), "text edge lin->act")
    assert_true(_has(tex.out, "<- x, act"), "text multi-edge into cat")
    assert_true(_has(tex.out, "-> 12"), "text cat out width 12")

    var mer = MermaidExporter()
    g.describe("mlp3", mer)
    print("\n--- MermaidExporter ---")
    print(mer.out)
    assert_true(_has(mer.out, "graph TD"), "mermaid header")
    assert_true(_has(mer.out, 'x(['), "mermaid input stadium")
    assert_true(_has(mer.out, "x -->|4| lin"), "mermaid edge x->lin (dim 4)")
    assert_true(_has(mer.out, "lin -->|8| act"), "mermaid edge lin->act (dim 8)")
    assert_true(_has(mer.out, "x -->|4| cat"), "mermaid edge x->cat (dim 4)")
    assert_true(_has(mer.out, "act -->|8| cat"), "mermaid edge act->cat (dim 8)")

    print("\nALL PASSED")
