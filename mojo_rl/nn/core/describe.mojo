"""Describe — a module-introspection helper for the storage `nn` framework.

An ADDITIVE debugging / migration aid. `DescribeVisitor` is a `ParamVisitor`
conformer (mirroring `_NamedCollector` in named_params.mojo) that accumulates,
per visited param, the dotted `name` and the comptime size `N`, plus a running
total of param count (sum of N) and tensor count (number of rows). It touches
neither the value/grad buffers nor the device, so it is target-agnostic (no
`ctx` needed).

`describe[target](model)` runs the param + state walks and returns a formatted
multi-line table String (one `"<name>: <size>"` line per param, then a footer
`"total params: <N> across <K> tensors"`) — returning a String (not printing)
makes it testable. `print_describe` is a thin print convenience.
"""

from max.gpu.host import DeviceContext

from .tensor import Tensor
from .param import ParamVisitor
from .module import Module


struct DescribeRow(Copyable):
    """One visited param/state: its dotted name and comptime size."""

    var name: String
    var size: Int

    def __init__(out self, name: String, size: Int):
        self.name = name
        self.size = size


struct DescribeVisitor(ParamVisitor):
    """Accumulates a `(name, size)` row per visited param/state plus running
    totals. Metadata-only — never reads buffers or the device."""

    var rows: List[DescribeRow]
    var total_params: Int

    def __init__(out self):
        self.rows = List[DescribeRow]()
        self.total_params = 0

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.rows.append(DescribeRow(name, N))
        self.total_params += N

    def render(self) -> String:
        """Multi-line table: one `"<name>: <size>"` line per row, then a footer
        `"total params: <N> across <K> tensors"`."""
        var s = String("")
        for ref r in self.rows:
            s += r.name + ": " + String(r.size) + "\n"
        s += (
            "total params: "
            + String(self.total_params)
            + " across "
            + String(len(self.rows))
            + " tensors"
        )
        return s


def describe[
    target: StaticString, M: Module
](mut model: M, ctx: Optional[DeviceContext] = None) raises -> String:
    """Formatted introspection table of every trainable Param AND persisted
    State of `model`, in `for_each_param` then `for_each_state` walk order with
    dotted names. Returns the table String (footer counts every visited
    tensor)."""
    var v = DescribeVisitor()
    model.for_each_param[target](v, ctx)
    model.for_each_state[target](v, ctx)
    return v.render()


def print_describe[
    target: StaticString, M: Module
](mut model: M, ctx: Optional[DeviceContext] = None) raises:
    """Thin convenience: build the describe table and print it."""
    print(describe[target](model, ctx))
