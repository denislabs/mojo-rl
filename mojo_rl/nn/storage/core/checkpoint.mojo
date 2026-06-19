"""Checkpoint — named-section param/state + optimizer-moment save/load (v2).

Walks a Module's Params via `for_each_param` then States via `for_each_state`,
writing one named section per field:

    storage-ckpt v2
    P <dotted-name> <size> <has_moments:0|1>
    <size value lines>
    [if has_moments: <size> m lines, then <size> v lines]
    ...
    S <dotted-name> <size>
    <size value lines>
    ...

The dotted name (from the A2 name-threading walker, e.g. "0.weight") is VALIDATED
against the in-memory walk order on load — a name/size mismatch raises, catching
topology drift between save and load (the legacy named-section guarantee the
positional v1 format lacked).

Optimizer moments (Adam's per-param `m`/`v`, co-located on the Param) ride the
same param section when populated (`m.n >= N`), enabling exact training resume.
`save_moments=False` writes a model-only checkpoint (moments skipped). GPU params
download on save / upload on load.

Storage-clean: the visitor OWNS its accumulator (String on save / parsed lines +
cursor on load); no UnsafePointer.
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
    """Appends a named section per visited Param/State. `mode`: 0 = Param (P,
    with optional moments), 1 = State (S). `save_moments` gates m/v output."""
    var content: String
    var mode: Int
    var save_moments: Bool

    def __init__(out self, save_moments: Bool = True):
        self.content = String("storage-ckpt v2\n")
        self.mode = 0
        self.save_moments = save_moments

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        if self.mode == 1:  # State
            self.content += "S " + name + " " + String(N) + "\n"
            for i in range(N):
                self.content += String(param.data[i]) + "\n"
            return
        # Param: include moments when populated (optimizer has stepped).
        var has_m = self.save_moments and m.n >= N and v.n >= N
        comptime if target == "gpu":
            if has_m:
                m.download(ctx.value())
                v.download(ctx.value())
        self.content += (
            "P " + name + " " + String(N) + " " + ("1" if has_m else "0") + "\n"
        )
        for i in range(N):
            self.content += String(param.data[i]) + "\n"
        if has_m:
            for i in range(N):
                self.content += String(m.data[i]) + "\n"
            for i in range(N):
                self.content += String(v.data[i]) + "\n"


struct CheckpointReader(ParamVisitor):
    """Consumes one named section per visited Param/State, validating the
    section kind + dotted name + size against the in-memory walk (topology-drift
    catch). Restores values and, for Params, the m/v moments if present."""
    var lines: List[String]
    var cur: Int
    var mode: Int

    def __init__(out self, var lines: List[String]):
        self.lines = lines^
        self.cur = 0
        self.mode = 0

    def _next(mut self) raises -> String:
        if self.cur >= len(self.lines):
            raise Error("checkpoint: unexpected end of file")
        var s = self.lines[self.cur]
        self.cur += 1
        return s

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var hdr = self._next()
        var toks = hdr.split(" ")
        var expected_kind = String("S") if self.mode == 1 else String("P")
        if len(toks) < 3 or toks[0] != expected_kind:
            raise Error(
                "checkpoint: expected '" + expected_kind + " " + name
                + "' section, got header `" + hdr + "`"
            )
        if toks[1] != name:
            raise Error(
                "checkpoint: name mismatch — model expects `" + name
                + "`, checkpoint has `" + toks[1] + "` (topology drift)"
            )
        if atol(toks[2]) != N:
            raise Error(
                "checkpoint: size mismatch for `" + name + "` — model "
                + String(N) + ", checkpoint " + toks[2]
            )
        for i in range(N):
            param.data[i] = Scalar[DT](atof(self._next()))
        if self.mode == 0 and len(toks) >= 4 and toks[3] == "1":
            m.ensure(N)
            v.ensure(N)
            for i in range(N):
                m.data[i] = Scalar[DT](atof(self._next()))
            for i in range(N):
                v.data[i] = Scalar[DT](atof(self._next()))
            comptime if target == "gpu":
                m.upload(ctx.value())
                v.upload(ctx.value())
        comptime if target == "gpu":
            param.upload(ctx.value())


def save_params[
    target: StaticString, M: Module
](
    mut model: M, path: String,
    ctx: Optional[DeviceContext] = None,
    save_moments: Bool = True,
) raises:
    """Write a v2 named checkpoint: Params (+ moments if populated) then States."""
    var w = CheckpointWriter(save_moments)
    w.mode = 0
    model.for_each_param[target](w, ctx)
    w.mode = 1
    model.for_each_state[target](w, ctx)
    with open(path, "w") as f:
        f.write(w.content)


def load_params[
    target: StaticString, M: Module
](mut model: M, path: String, ctx: Optional[DeviceContext] = None) raises:
    """Load a v2 named checkpoint, validating names/sizes against `model`."""
    var content: String
    with open(path, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    # Drop the format header line.
    var body = List[String]()
    for li in range(len(lines)):
        if lines[li].startswith("storage-ckpt"):
            continue
        body.append(lines[li])
    var r = CheckpointReader(body^)
    r.mode = 0
    model.for_each_param[target](r, ctx)
    r.mode = 1
    model.for_each_state[target](r, ctx)
