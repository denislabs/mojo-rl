"""Checkpoint — positional param save/load over the storage surface.

`save_params` / `load_params` walk a Module's `Param`s in `for_each_param` order
and (de)serialize each Param's VALUES, one float per line, to a text file (the
legacy `nn` checkpoint format minus the named sections — positional, so the
model's topology must match at save/load). Optimizer moment state (m/v) is NOT
saved yet (Adam re-warms on resume — fine for SAC); add a second pass when
optimizer-resume is needed.

Storage-clean: the visitor OWNS its accumulator (String on save / value List on
load) and the caller reads it back after the `for_each_param` walk — no
`UnsafePointer[String]` (cf. legacy's `out_ptr`). GPU params download on save /
upload on load.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from .tensor import Tensor
from .param import ParamVisitor
from .module import Module


def _split_lines(content: String) -> List[String]:
    var lines = List[String]()
    var cur = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(cur)
            cur = String("")
        else:
            cur += chr(Int(c))
    if cur.byte_length() > 0:
        lines.append(cur)
    return lines^


struct CheckpointWriter(ParamVisitor):
    """Appends each visited Param's VALUES (one float/line) to `content`."""
    var content: String

    def __init__(out self):
        self.content = String("storage-ckpt v1\n")

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        for i in range(N):
            self.content += String(param.data[i]) + "\n"


struct CheckpointReader(ParamVisitor):
    """Consumes N values per visited Param from `values` (walk order)."""
    var values: List[Scalar[DT]]
    var off: Int

    def __init__(out self, var values: List[Scalar[DT]]):
        self.values = values^
        self.off = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if self.off + N > len(self.values):
            raise Error("load_params: short read (checkpoint/model mismatch)")
        for i in range(N):
            param.data[i] = self.values[self.off + i]
        self.off += N
        comptime if target == "gpu":
            param.upload(ctx.value())


def save_params[
    target: StaticString, M: Module
](mut model: M, path: String, ctx: Optional[DeviceContext] = None) raises:
    var w = CheckpointWriter()
    model.for_each_param[target](w, ctx)
    # States (e.g. BatchNorm running stats) ride the same stream, right after
    # the params — state-less models append nothing (format-compatible).
    model.for_each_state[target](w, ctx)
    with open(path, "w") as f:
        f.write(w.content)


def load_params[
    target: StaticString, M: Module
](mut model: M, path: String, ctx: Optional[DeviceContext] = None) raises:
    var content: String
    with open(path, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var values = List[Scalar[DT]]()
    for li in range(len(lines)):
        var line = lines[li]
        if line.byte_length() == 0 or line.startswith("storage-ckpt"):
            continue
        values.append(Scalar[DT](atof(line)))
    var r = CheckpointReader(values^)
    model.for_each_param[target](r, ctx)
    model.for_each_state[target](r, ctx)
