"""Graph exporters — GraphVisitor sinks for `ComputeGraph.describe`.

`describe` walks the graph's comptime topology and emits, per node, a
`node` event (with the op's display label), zero or more `node_inner`
events (container children — e.g. each layer of a `Sequential`), and one
`edge` event per input slot. Each exporter here turns that event stream
into a different representation, sharing the single traversal:

  * `TextExporter`    — aligned table for the terminal / CI logs, with
        container children listed underneath their node. The debugging
        default: no rendering dependency, prints straight out.
  * `MermaidExporter` — `graph TD` source for docs (GitHub renders it
        natively). Input slots are stadiums, leaves are boxes, and
        container nodes become labelled `subgraph`s holding their chained
        children. Edges are labelled with the feature width they carry.

Both buffer events and render in `end()`, so neither depends on emission
order. Adding a third exporter (DOT, a shape/param report, …) is a new
struct conforming to `GraphVisitor` — it never touches `ComputeGraph` or
the others.
"""

from ..core import GraphVisitor


# ──────────────────────────────────────────────────────────────────────
# Small formatting helpers (module-level — shared by both exporters).
# ──────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────
# TextExporter — buffer node rows + children, format a table in `end`.
# ──────────────────────────────────────────────────────────────────────


struct TextExporter(GraphVisitor):
    var out: String
    var gname: String
    var names: List[String]
    var labels: List[String]
    var outs: List[Int]
    var ins: List[String]          # accumulated "src1, src2, ..." per node
    var subs: List[List[String]]   # container children ("label →out") per node

    def __init__(out self):
        self.out = String("")
        self.gname = String("")
        self.names = List[String]()
        self.labels = List[String]()
        self.outs = List[Int]()
        self.ins = List[String]()
        self.subs = List[List[String]]()

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
        self.subs.append(List[String]())

    def node_inner(
        mut self, parent: String, step_idx: Int, label: String,
        out_dim: Int,
    ) raises:
        self.subs[len(self.subs) - 1].append(label + " →" + String(out_dim))

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int,
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
            # Container children, indented under the node.
            for j in range(len(self.subs[i])):
                s += _pad("", maxname + 10) + "└ " + self.subs[i][j] + "\n"
        self.out = s


# ──────────────────────────────────────────────────────────────────────
# MermaidExporter — buffer, then render `graph TD` with subgraphs.
# ──────────────────────────────────────────────────────────────────────


struct MermaidExporter(GraphVisitor):
    var out: String
    var gname: String
    var names: List[String]
    var labels: List[String]
    var kinds: List[Int]
    var outs: List[Int]
    var subs: List[List[String]]   # container children ("label →out") per node
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
        self.subs = List[List[String]]()
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
        self.subs.append(List[String]())

    def node_inner(
        mut self, parent: String, step_idx: Int, label: String,
        out_dim: Int,
    ) raises:
        self.subs[len(self.subs) - 1].append(label + " →" + String(out_dim))

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int,
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
                # External input — stadium.
                s += "  " + name + '(["' + name + "<br/>in " + od + '"])\n'
            elif len(self.subs[i]) > 0:
                # Container — subgraph holding the chained children.
                s += "  subgraph " + name + " [" + name + ": " + label + "]\n"
                var nsub = len(self.subs[i])
                for j in range(nsub):
                    s += (
                        "    " + name + "__" + String(j) + '["'
                        + self.subs[i][j] + '"]\n'
                    )
                for j in range(nsub - 1):
                    s += (
                        "    " + name + "__" + String(j) + " --> "
                        + name + "__" + String(j + 1) + "\n"
                    )
                s += "  end\n"
            else:
                # Leaf — box labelled with op name + output width.
                s += "  " + name + '["' + name + "<br/>" + label + " &rarr; " + od + '"]\n'

        for e in range(len(self.esrc)):
            s += (
                "  " + self.esrc[e] + " -->|" + String(self.ein[e]) + "| "
                + self.edst[e] + "\n"
            )
        self.out = s
