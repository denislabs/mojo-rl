"""GraphVisitor — a pluggable sink for a storage `ComputeGraph` topology walk.

`ComputeGraph.describe[V: GraphVisitor](name, visitor)` drives the visitor over
the comptime node list: one `begin`, then per node a `node` call followed by one
`edge` call per input, then `end`. Exporters (text dump, Mermaid diagram) are
just `GraphVisitor` conformers — they never touch the graph internals, buffers,
or the device, so describe is target-agnostic (pure comptime topology, no `ctx`).

Transformed from the legacy `nn/core/graph_visitor.mojo`. The legacy
`node_inner` callback (per-Op inner display-step expansion) is intentionally
dropped: it relied on `Op.display_steps()`, which the storage leaves do not
expose. The storage walk reports the node-level topology (name, category, output
width, edges) — enough for visualization; the per-Op fusion-step census would
need per-leaf `display_steps` first.
"""


trait GraphVisitor(ImplicitlyDeletable):
    """Sink for a ComputeGraph topology walk (begin → node*/edge* → end)."""

    def begin(mut self, graph_name: String, n_nodes: Int) raises:
        """Called once before any node. `n_nodes` = total decls in the graph."""
        ...

    def node(
        mut self,
        idx: Int,
        name: String,
        label: String,
        kind: Int,
        out_dim: Int,
    ) raises:
        """One graph node. `idx` is the topological (decl) order; `name` is the
        decl name; `label` is a category tag; `kind` is the decl category
        (0=InputSlot, 1=owned Node, 2=ExternalNode); `out_dim` is the per-sample
        output width."""
        ...

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int
    ) raises:
        """A data edge `src → dst` feeding input slot `slot` of `dst`; `in_dim`
        is the source node's per-sample output width."""
        ...

    def end(mut self) raises:
        """Called once after the last node/edge — finalize the rendering."""
        ...
