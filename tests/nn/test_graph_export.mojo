"""ComputeGraph.describe — TextExporter + MermaidExporter smoke / demo.

`describe` reads only the graph's comptime topology, so it runs on a
default-constructed graph (no `make`, no device context, no buffers).
This exercises the GraphVisitor seam on:

  1. a tiny hand-built 3-node chain (asserts the event stream), and
  2. the real `WMLossGraph` from DreamerV3 (prints the actual diagram so
     we can eyeball the 30-node RSSM loss as text + mermaid).
"""

from std.testing import assert_true

from mojo_rl.nn.combinators import (
    ComputeGraph, InputSlot, Node, TextExporter, MermaidExporter,
)
from mojo_rl.nn.primitives.linear import Linear

from mojo_rl.deep_agents.dreamerv3.wm import WMLossGraph


def test_text_exporter_tiny() raises:
    comptime Tiny = ComputeGraph[
        2,
        InputSlot["x", 4],
        Node["h", Linear[4, 8], "x"],
        Node["y", Linear[8, 2], "h"],
    ]
    var g = Tiny()  # default ctor — describe needs no make
    var e = TextExporter()
    g.describe(e, "Tiny")
    print(e.out)

    # Event-stream sanity: 3 nodes, the slot has no inputs, the two
    # compute nodes each have one.
    assert_true(len(e.names) == 3, "expected 3 nodes")
    assert_true(e.labels[0] == "input", "node 0 is an InputSlot")
    assert_true(e.labels[1] == "Linear", "node 1 is a Linear")
    assert_true(e.ins[0].byte_length() == 0, "slot has no inputs")
    assert_true(e.ins[1] == "x", "h <- x")
    assert_true(e.ins[2] == "h", "y <- h")
    assert_true(e.outs[2] == 2, "graph out dim 2")


def test_mermaid_exporter_tiny() raises:
    comptime Tiny = ComputeGraph[
        2,
        InputSlot["x", 4],
        Node["h", Linear[4, 8], "x"],
        Node["y", Linear[8, 2], "h"],
    ]
    var g = Tiny()
    var m = MermaidExporter()
    g.describe(m, "Tiny")
    print(m.out)
    assert_true(m.out.find("graph TD") >= 0, "mermaid header present")
    assert_true(m.out.find("x -->|4| h") >= 0, "labelled edge x->h")
    assert_true(m.out.find("h -->|8| y") >= 0, "labelled edge h->y")


def test_describe_wm_loss_graph() raises:
    # Tiny RSSM dims — only the topology matters for describe.
    comptime WM = WMLossGraph[
        8,   # DETER
        16,  # H
        4,   # STOCH
        4,   # CLASSES
        2,   # BLOCKS
        3,   # ACT
        12,  # TOKEN
        10,  # OBS
        16,  # DEC_U
        16,  # HU
        5,   # BINS
    ]
    var g = WM()

    print("\n================= WMLossGraph (text) =================")
    var e = TextExporter()
    g.describe(e, "WMLossGraph")
    print(e.out)

    print("\n================= WMLossGraph (mermaid) =================")
    var m = MermaidExporter()
    g.describe(m, "WMLossGraph")
    print(m.out)

    assert_true(len(e.names) == WM.N, "text exporter saw every node")


def main() raises:
    test_text_exporter_tiny()
    test_mermaid_exporter_tiny()
    test_describe_wm_loss_graph()
    print("\nAll graph-export tests passed.")
