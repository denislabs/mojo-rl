"""GraphVisitor — a pluggable sink for a storage `ComputeGraph` topology walk.

`ComputeGraph.describe[V: GraphVisitor](name, visitor)` drives the visitor over
the comptime node list: one `begin`, then per node a `node` call followed by one
`edge` call per input, then `end`. Exporters (text dump, Mermaid diagram) are
just `GraphVisitor` conformers — they never touch the graph internals, buffers,
or the device, so describe is target-agnostic (pure comptime topology, no `ctx`).

Transformed from the legacy `nn/core/graph_visitor.mojo`. `node`'s `label` is the
op's display name (`Module.display_label`); container nodes (a `Node` wrapping a
`Sequential`) expand into one `node_inner` call per child (`Module.display_steps`)
so they don't collapse to one opaque box. `begin`/`end`/`node_inner` carry `pass`
defaults so an exporter overrides only what it needs.
"""


@fieldwise_init
struct DisplayStep(Copyable, Movable):
    """One inner display step of a container module — its child's display label
    and per-sample output width. Returned by `Module.display_steps`."""

    var label: String
    var out_dim: Int


trait GraphVisitor(Deinitable):
    """Sink for a ComputeGraph topology walk
    (begin → node*/node_inner*/edge* → end)."""

    def begin(mut self, graph_name: String, n_nodes: Int) raises:
        """Called once before any node. `n_nodes` = total decls in the graph."""
        pass

    def node(
        mut self,
        idx: Int,
        name: String,
        label: String,
        kind: Int,
        out_dim: Int,
    ) raises:
        """One graph node. `idx` is the topological (decl) order; `name` is the
        decl name; `label` is the op's display name; `kind` is the decl category
        (0=InputSlot, 1=owned Node, 2=ExternalNode); `out_dim` is the per-sample
        output width."""
        ...

    def node_inner(
        mut self, parent: String, step_idx: Int, label: String, out_dim: Int
    ) raises:
        """Called once per inner step of a container node (e.g. each child of a
        `Sequential`), right after that node's `node` call. Default no-op —
        exporters that don't expand containers ignore it."""
        pass

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int
    ) raises:
        """A data edge `src → dst` feeding input slot `slot` of `dst`; `in_dim`
        is the source node's per-sample output width."""
        ...

    def end(mut self) raises:
        """Called once after the last node/edge — finalize the rendering."""
        pass
