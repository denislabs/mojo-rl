"""ComputeGraph exporters — `GraphVisitor` conformers that render a storage
`ComputeGraph`'s topology (run via `graph.describe[Exporter](name, exporter)`).

  - `TextExporter`     — a readable indented table (`#idx name label <- ins -> out`).
  - `MermaidExporter`  — a `graph TD` Mermaid diagram (paste into any Mermaid
    renderer); InputSlots render as stadiums, owned/external nodes as boxes.

Transformed from the legacy `nn/combinators/graph_export.mojo`. The legacy
container `node_inner`/subgraph expansion and the `FusionReportExporter` (static
fusion census) are NOT ported — both relied on per-Op `display_steps()`, which
the storage leaves do not expose; these are topology-only exporters. Read the
graph by NODE NAME (descriptive in practice — "rsample", "concat", "q1", …); the
`label` is the decl category.
"""

from ..core.graph_visitor import GraphVisitor


def _pad(s: String, width: Int) -> String:
    """Right-pad `s` with spaces to at least `width` bytes (ASCII names)."""
    var out = s
    var n = s.byte_length()
    while n < width:
        out += " "
        n += 1
    return out


def _idx2(i: Int) -> String:
    """Zero-padded 2-digit node index."""
    if i < 10:
        return "0" + String(i)
    return String(i)


struct TextExporter(GraphVisitor):
    """Buffers node rows + their input lists, formats an indented table in
    `end()`. Read the result from `.out`."""

    var out: String
    var gname: String
    var names: List[String]
    var labels: List[String]
    var outs: List[Int]
    var ins: List[String]  # accumulated "src1, src2, ..." per node

    def __init__(out self):
        self.out = String("")
        self.gname = String("")
        self.names = List[String]()
        self.labels = List[String]()
        self.outs = List[Int]()
        self.ins = List[String]()

    def begin(mut self, graph_name: String, n_nodes: Int) raises:
        self.gname = graph_name

    def node(
        mut self, idx: Int, name: String, label: String, kind: Int,
        out_dim: Int,
    ) raises:
        self.names.append(name)
        self.labels.append(label)
        self.outs.append(out_dim)
        self.ins.append(String(""))

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int
    ) raises:
        var last = len(self.ins) - 1
        if self.ins[last].byte_length() > 0:
            self.ins[last] = self.ins[last] + ", " + src
        else:
            self.ins[last] = src

    def end(mut self) raises:
        var maxname = 4
        var maxlabel = 5
        for i in range(len(self.names)):
            if self.names[i].byte_length() > maxname:
                maxname = self.names[i].byte_length()
            if self.labels[i].byte_length() > maxlabel:
                maxlabel = self.labels[i].byte_length()

        var s = String("ComputeGraph")
        if self.gname.byte_length() > 0:
            s += ' "' + self.gname + '"'
        s += "  (" + String(len(self.names)) + " nodes)\n"

        for i in range(len(self.names)):
            s += "  #" + _idx2(i) + "  " + _pad(self.names[i], maxname)
            s += "  " + _pad(self.labels[i], maxlabel)
            if self.ins[i].byte_length() > 0:
                s += _pad("  <- " + self.ins[i], maxname + 28)
            else:
                s += _pad("", maxname + 28)
            s += "  -> " + String(self.outs[i]) + "\n"
        self.out = s


struct MermaidExporter(GraphVisitor):
    """Buffers nodes + edges, renders a `graph TD` Mermaid diagram in `end()`.
    Read the result from `.out`."""

    var out: String
    var gname: String
    var names: List[String]
    var labels: List[String]
    var kinds: List[Int]
    var outs: List[Int]
    var esrc: List[String]
    var edst: List[String]
    var ein: List[Int]

    def __init__(out self):
        self.out = String("")
        self.gname = String("")
        self.names = List[String]()
        self.labels = List[String]()
        self.kinds = List[Int]()
        self.outs = List[Int]()
        self.esrc = List[String]()
        self.edst = List[String]()
        self.ein = List[Int]()

    def begin(mut self, graph_name: String, n_nodes: Int) raises:
        self.gname = graph_name

    def node(
        mut self, idx: Int, name: String, label: String, kind: Int,
        out_dim: Int,
    ) raises:
        self.names.append(name)
        self.labels.append(label)
        self.kinds.append(kind)
        self.outs.append(out_dim)

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int
    ) raises:
        self.esrc.append(src)
        self.edst.append(dst)
        self.ein.append(in_dim)

    def end(mut self) raises:
        var s = String("")
        if self.gname.byte_length() > 0:
            s += "%% " + self.gname + "\n"
        s += "graph TD\n"

        for i in range(len(self.names)):
            var name = self.names[i]
            var label = self.labels[i]
            var od = String(self.outs[i])
            if self.kinds[i] == 0:
                # InputSlot — stadium.
                s += "  " + name + '(["' + name + "<br/>in " + od + '"])\n'
            else:
                # Owned / external node — box labelled name + category + width.
                s += (
                    "  " + name + '["' + name + "<br/>" + label
                    + " &rarr; " + od + '"]\n'
                )

        for e in range(len(self.esrc)):
            s += (
                "  " + self.esrc[e] + " -->|" + String(self.ein[e]) + "| "
                + self.edst[e] + "\n"
            )
        self.out = s
