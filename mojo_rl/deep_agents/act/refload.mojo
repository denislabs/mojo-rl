# +--------------------------------------------------------------------------+ #
# | Reference-dump plumbing for the ACT gates
# +--------------------------------------------------------------------------+ #
"""Read a `tools/act/dump_act_reference.py` dump, and push its weights into a
module BY NAME.

Gating a port against a reference means running both on the SAME weights.
Positional loading would do it in ten fewer lines and would also silently
succeed when two same-sized params swap — the q/k/v projections in a DETR layer
are all `[DIM, DIM]`, so exactly the confusion that matters most is the one a
positional loader cannot see. Everything here is addressed by the dotted
`for_each_param` name, and a name present on one side but not the other is an
error naming the name.

Test-only. Nothing in a training or inference path reads it.

## Dump format

    <dir>/manifest.txt      `name<TAB>d0,d1,...` per array
    <dir>/<name>.bin        raw little-endian float32, C order

## Weight layout

⚠ **torch `nn.Linear.weight` is `[out_features, in_features]`.** The dump
script transposes to whatever this framework's `Linear` expects and records the
transposed shape, so `RefDump` itself does no reinterpretation — a blob is
written in the destination's layout or the load fails on size. The one place
that convention is decided is `dump_act_reference.py`.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor


struct RefDump(Movable & Deinitable):
    """A reference dump directory: names, sizes, and lazily-read blobs."""

    var root: String
    var names: List[String]
    var sizes: List[Int]

    def __init__(out self, var root: String) raises:
        self.root = root^
        self.names = List[String]()
        self.sizes = List[Int]()
        var text: String
        with open(String(self.root) + "/manifest.txt", "r") as f:
            var raw = f.read_bytes()
            text = String()
            for i in range(len(raw)):
                text += chr(Int(raw[i]))
            _ = raw^
        var line = String()
        var bytes = text.as_bytes()
        for i in range(len(bytes)):
            if bytes[i] == UInt8(ord("\n")):
                if line.byte_length() > 0:
                    self._add(line)
                line = String()
            else:
                line += chr(Int(bytes[i]))
        if line.byte_length() > 0:
            self._add(line)

    def __init__(out self, *, deinit move: Self):
        self.root = move.root^
        self.names = move.names^
        self.sizes = move.sizes^

    def _add(mut self, line: String) raises:
        """`name<TAB>d0,d1,...` -> (name, product of dims)."""
        var b = line.as_bytes()
        var tab = -1
        for i in range(len(b)):
            if b[i] == UInt8(ord("\t")):
                tab = i
                break
        if tab < 0:
            raise Error("RefDump: manifest line has no tab: " + line)
        var name = String()
        for i in range(tab):
            name += chr(Int(b[i]))
        var n = 1
        var cur = String()
        for i in range(tab + 1, len(b)):
            if b[i] == UInt8(ord(",")):
                n *= Int(cur)
                cur = String()
            else:
                cur += chr(Int(b[i]))
        if cur.byte_length() > 0:
            n *= Int(cur)
        self.names.append(name^)
        self.sizes.append(n)

    def has(self, name: String) -> Bool:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return True
        return False

    def size_of(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.sizes[i]
        raise Error("RefDump: no array named '" + name + "'")

    def get(self, name: String) raises -> List[Scalar[DT]]:
        var n = self.size_of(name)
        var out = List[Scalar[DT]](unsafe_uninit_length=n)
        with open(String(self.root) + "/" + name + ".bin", "r") as f:
            var bytes = f.read_bytes()
            if len(bytes) != n * 4:
                raise Error(
                    "RefDump: " + name + ".bin is " + String(len(bytes))
                    + " bytes but the manifest says " + String(n * 4)
                )
            var p = bytes.unsafe_ptr().unsafe_bitcast[Scalar[DT]]()
            for i in range(n):
                out[i] = p[unsafe_offset=i]
            _ = bytes^
        return out^


struct LoadRefParams[PREFIX: StaticString](ParamVisitor):
    """`ParamVisitor` filling each param from `<PREFIX><dotted name>`.

    Records what it touched so the caller can assert full coverage. A param the
    dump does not mention keeps its random init, and the comparison — which is
    supposed to be running identical weights on both sides — then reads as a
    small numerical disagreement rather than as the missing weight it is.
    """

    var dump: RefDump
    var loaded: List[String]
    var missing: List[String]

    def __init__(out self, var dump: RefDump):
        self.dump = dump^
        self.loaded = List[String]()
        self.missing = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.dump = move.dump^
        self.loaded = move.loaded^
        self.missing = move.missing^

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
        comptime assert target == "cpu", "LoadRefParams: CPU only"
        var key = String(Self.PREFIX) + name
        if not self.dump.has(key):
            self.missing.append(key^)
            return
        var vals = self.dump.get(key)
        if len(vals) != N:
            raise Error(
                "LoadRefParams: '" + key + "' has " + String(len(vals))
                + " values but the param holds " + String(N)
            )
        param.ensure(N)
        for i in range(N):
            param.data[i] = vals[i]
        self.loaded.append(key^)


struct ListParams(ParamVisitor):
    """Collects `(name, size)` for every param — used to author the dump's name
    mapping and to check it stays in step."""

    var names: List[String]
    var sizes: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.sizes = List[Int]()

    def __init__(out self, *, deinit move: Self):
        self.names = move.names^
        self.sizes = move.sizes^

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
        self.names.append(String(name))
        self.sizes.append(N)

    def show(self) raises:
        for i in range(len(self.names)):
            print("  " + self.names[i] + "\t" + String(self.sizes[i]))
