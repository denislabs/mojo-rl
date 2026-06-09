"""GraphVisitor trait — invoked during a `ComputeGraph.describe` walk.

The topology sink, mirroring `ParamVisitor` for parameters. `describe`
walks the graph's `*NODES` in declaration (topological) order exactly
the way `_forward_cpu` does, but routes the comptime node metadata
(`NAME`, `KIND`, `OUT_DIM`, `IN_DIMS`, `IN_NAMES`) plus each op's
display label / inner steps into a runtime sink instead of into kernels.
One traversal, many exporters: `TextExporter`, `MermaidExporter`, … each
conform to this trait and accumulate their own representation. Adding a
new exporter never touches the graph.

Call sequence (per `describe`):
    begin(graph_name, n_nodes)
    for each node i in topo order:
        node(i, name, label, kind, out_dim)
        for each inner display step s of node i:        # containers only
            node_inner(name, s, step_label, step_out_dim)
        for each input slot k of node i:
            edge(name, src_name, k, in_dim)             # src = predecessor
    end()

`label` is the op's display name (see `Module.display_label`). Container
ops (`Sequential`, and its aliases like `DreamerDecoder`) expand into
`node_inner` steps — one per child — so a `Node` wrapping
`Sequential[Linear, RMSNorm, GELU]` no longer collapses to one opaque
box. Leaves emit no inner steps. `kind` is the node arity: 0 = external
`InputSlot`, 1 = unary … 4 = quaternary.

`begin` / `end` / `node_inner` carry `pass` defaults so an exporter only
overrides what it needs; `node` / `edge` are abstract.
"""


@fieldwise_init
struct DisplayStep(Copyable, Movable):
    """One inner display step of a container module — its child's display
    label and output width. Returned by `Module.display_steps`."""

    var label: String
    var out_dim: Int


trait GraphVisitor(ImplicitlyDestructible):
    def begin(mut self, graph_name: String, n_nodes: Int) raises:
        """Called once before any node. Default no-op."""
        pass

    def node(
        mut self,
        idx: Int,
        name: String,
        label: String,
        kind: Int,
        out_dim: Int,
    ) raises:
        """Called once per node in topological order. `label` is the op's
        display name; `kind` is its arity (0 = InputSlot)."""
        ...

    def node_inner(
        mut self,
        parent: String,
        step_idx: Int,
        label: String,
        out_dim: Int,
    ) raises:
        """Called once per inner step of a container node (e.g. each child
        of a `Sequential`), right after that node's `node` call. Default
        no-op — exporters that don't expand containers ignore it."""
        pass

    def edge(
        mut self,
        dst: String,
        src: String,
        slot: Int,
        in_dim: Int,
    ) raises:
        """Called once per input slot of each compute node: `src`
        (predecessor name) feeds `dst` at input slot `slot`, carrying
        `in_dim` features. Never called for `InputSlot`s (KIND=0)."""
        ...

    def end(mut self) raises:
        """Called once after the last node. Default no-op."""
        pass
