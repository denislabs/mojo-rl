# +--------------------------------------------------------------------------+ #
# | Our parameter names and layout <-> PyTorch's
# +--------------------------------------------------------------------------+ #
"""`TorchNameMap` — the three per-architecture facts a generic walk cannot know.

`safetensors_io.mojo` moves weights by their walked name, in our layout, with a
rank-1 shape. That is exactly right between two of our own files and useless
across the fence, because a PyTorch state dict differs on all three counts:

  1. **Names.** Our `0.0.0.weight` is their `conv1.weight`. There is no rule —
     ours come from `Sequential` child indices, theirs from attribute names.
  2. **Layout.** `nn.Linear.weight` is `[out, in]` and computes `y = x @ Wt`;
     ours is `[in, out]` and computes `y = x @ W`. Same numbers, transposed.
     Convolutions agree (`[OC, IC, KH, KW]` on both sides), which is what makes
     this easy to get wrong: a map that transposes nothing is right about most
     of the tensors.
  3. **Shapes.** `Param[NAME, DECAY, SIZE]` is flat. A consumer that reshapes
     by the file's shape needs the real one.

All three live in one table per architecture, used in BOTH directions, so an
import and an export cannot disagree about what a name means.

## The fourth fact: tensors one side does not have

`TN_ZEROS` covers a parameter of ours with no counterpart. The case that
forced it is real and easy to miss: torchvision's ResNet convolutions are
`bias=False` (a BatchNorm follows and its beta subsumes any bias) while this
framework's `Conv2D` always has one. Leaving those biases at their random
initialisation makes the loaded model a DIFFERENT FUNCTION from the pretrained
one — quietly, at a magnitude that reads as a numerical disagreement rather
than as the missing tensor it is.

## Direction

    load:  their file -> our params      (`LoadTorchNamed`)
    save:  our params -> their file      (`SaveTorchNamed`)

`TN_ZEROS` entries are filled on load and SKIPPED on save: writing a bias
torchvision does not have would produce a file `load_state_dict` rejects.
"""

from max.gpu.host import DeviceContext

from mojo_rl.io.safetensors import SafeTensors, SafeTensorsWriter
from mojo_rl.nn.constants import DT
from .param import ParamVisitor
from .safetensors_io import fill_param
from .tensor import Tensor


comptime TN_PLAIN = 0
"""Same numbers, same order — a rename only."""
comptime TN_TRANSPOSE = 1
"""Their `[R, C]` is our `[C, R]`. `nn.Linear` and nothing else, so far."""
comptime TN_ZEROS = 2
"""Ours exists, theirs does not: fill with zeros on load, skip on save."""


struct TorchNameMap(Movable):
    """One architecture's (our name, their name, layout, their shape) table.

    Parallel arrays with a flat `shape_data`, matching `io/safetensors.mojo` —
    shapes are variable-length and a `List[List[Int]]` buys nothing here.
    """

    var ours: List[String]
    var theirs: List[String]
    var kind: List[Int]
    var shape_data: List[Int]
    var shape_start: List[Int]
    var shape_rank: List[Int]

    def __init__(out self):
        self.ours = List[String]()
        self.theirs = List[String]()
        self.kind = List[Int]()
        self.shape_data = List[Int]()
        self.shape_start = List[Int]()
        self.shape_rank = List[Int]()

    def __init__(out self, *, deinit move: Self):
        self.ours = move.ours^
        self.theirs = move.theirs^
        self.kind = move.kind^
        self.shape_data = move.shape_data^
        self.shape_start = move.shape_start^
        self.shape_rank = move.shape_rank^

    def size(self) -> Int:
        return len(self.ours)

    def add(
        mut self,
        var ours: String,
        var theirs: String,
        ref shape: List[Int],
        kind: Int = TN_PLAIN,
    ) raises:
        """`shape` is THEIR shape — the one that goes in the file."""
        for i in range(len(self.ours)):
            if self.ours[i] == ours:
                raise Error(
                    "TorchNameMap: '" + ours + "' mapped twice (to '"
                    + self.theirs[i] + "' and '" + theirs + "')"
                )
        if kind == TN_TRANSPOSE and len(shape) != 2:
            raise Error(
                "TorchNameMap: '" + ours + "' is marked TN_TRANSPOSE but its"
                " shape has rank " + String(len(shape))
            )
        var start = len(self.shape_data)
        for i in range(len(shape)):
            self.shape_data.append(shape[i])
        self.ours.append(ours^)
        self.theirs.append(theirs^)
        self.kind.append(kind)
        self.shape_start.append(start)
        self.shape_rank.append(len(shape))

    def add_linear(
        mut self, var ours: String, var theirs: String, out_features: Int,
        in_features: Int,
    ) raises:
        """A `nn.Linear` weight: theirs `[out, in]`, ours `[in, out]`."""
        var s: List[Int] = [out_features, in_features]
        self.add(ours^, theirs^, s, TN_TRANSPOSE)

    def index_of_ours(self, name: String) -> Int:
        for i in range(len(self.ours)):
            if self.ours[i] == name:
                return i
        return -1

    def their_shape(self, i: Int) -> List[Int]:
        var out = List[Int]()
        for k in range(self.shape_rank[i]):
            out.append(self.shape_data[self.shape_start[i] + k])
        return out^

    def numel(self, i: Int) -> Int:
        var n = 1
        for k in range(self.shape_rank[i]):
            n *= self.shape_data[self.shape_start[i] + k]
        return n


def transpose_2d(ref src: List[Float32], rows: Int, cols: Int) -> List[Float32]:
    """`src` is row-major `[rows, cols]`; the result is row-major
    `[cols, rows]`."""
    var out = List[Float32](unsafe_uninit_length=rows * cols)
    for r in range(rows):
        for c in range(cols):
            out[c * rows + r] = src[r * cols + c]
    return out^


struct LoadTorchNamed[GRAPH_PREFIX: StaticString](ParamVisitor):
    """Fill one subtree of a model from a PyTorch-named safetensors file.

    Safe to run over `for_each_param` AND `for_each_state` in turn — the same
    instance accumulates both, so the header is parsed once and the coverage
    check sees the whole backbone rather than half of it.

    Four counters, because the four ways this goes wrong are different:
    `loaded` (matched and filled), `zeroed` (`TN_ZEROS`), `missing` (the map
    names a tensor the file does not have), `unmapped` (the model has a
    parameter under the prefix that the map does not mention — the topology
    moved and the map did not). `skipped` is everything outside the subtree and
    is expected.
    """

    var file: SafeTensors
    var map: TorchNameMap
    var loaded: List[String]
    var zeroed: List[String]
    var missing: List[String]
    var unmapped: List[String]
    var skipped: Int

    def __init__(out self, var file: SafeTensors, var map: TorchNameMap):
        self.file = file^
        self.map = map^
        self.loaded = List[String]()
        self.zeroed = List[String]()
        self.missing = List[String]()
        self.unmapped = List[String]()
        self.skipped = 0

    def __init__(out self, *, deinit move: Self):
        self.file = move.file^
        self.map = move.map^
        self.loaded = move.loaded^
        self.zeroed = move.zeroed^
        self.missing = move.missing^
        self.unmapped = move.unmapped^
        self.skipped = move.skipped

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
        comptime GP = String(Self.GRAPH_PREFIX)
        if not name.startswith(GP):
            self.skipped += 1
            return
        var local = String(name[byte = GP.byte_length() :])
        var i = self.map.index_of_ours(local)
        if i < 0:
            self.unmapped.append(local^)
            return

        if self.map.kind[i] == TN_ZEROS:
            var zeros = List[Float32](unsafe_uninit_length=N)
            for k in range(N):
                zeros[k] = Float32(0.0)
            fill_param(param, zeros, ctx)
            self.zeroed.append(local^)
            return

        var key = String(self.map.theirs[i])
        if not self.file.has(key):
            self.missing.append(key^)
            return
        var vals = self.file.read_f32(key)

        var want = self.map.their_shape(i)
        var got = self.file.shape(key)
        # ⚠ The SHAPE, not just the count. A `[512, 256]` and a `[256, 512]`
        # hold the same number of floats, and confusing them is precisely what
        # the transpose flag exists to prevent — so a map that is right about
        # the flag and wrong about the shape must not pass quietly.
        if not _same_shape(want, got):
            raise Error(
                "LoadTorchNamed: '" + key + "' has shape "
                + self.file.shape_str(key) + " but the map declares "
                + _shape_str(want)
            )

        if self.map.kind[i] == TN_TRANSPOSE:
            var t = transpose_2d(vals, want[0], want[1])
            if len(t) != N:
                raise Error(_size_err(key, local, len(t), N))
            fill_param(param, t, ctx)
        else:
            if len(vals) != N:
                raise Error(_size_err(key, local, len(vals), N))
            fill_param(param, vals, ctx)
        self.loaded.append(local^)

    def report(self, what: String) raises:
        """Raise unless every mapped tensor under the prefix was filled."""
        if len(self.unmapped) > 0:
            raise Error(
                what + ": " + String(len(self.unmapped)) + " parameter(s)"
                " under '" + String(Self.GRAPH_PREFIX) + "' are not in the"
                " name map, first '" + self.unmapped[0] + "' — the model's"
                " topology and the map have drifted apart, and those weights"
                " would have kept their initialisation"
            )
        if len(self.missing) > 0:
            raise Error(
                what + ": " + String(len(self.missing)) + " tensor(s) named by"
                " the map are absent from the file, first '" + self.missing[0]
                + "' — the file is a different variant of this architecture"
            )
        if len(self.loaded) == 0:
            raise Error(
                what + ": matched NOTHING under '" + String(Self.GRAPH_PREFIX)
                + "' — the prefix and the walk have drifted apart, so every"
                " weight would have been silently discarded"
            )


struct SaveTorchNamed[GRAPH_PREFIX: StaticString](ParamVisitor):
    """Collect one subtree into a `SafeTensorsWriter` under PyTorch names,
    shapes and layout — a file `load_state_dict` can take.

    `TN_ZEROS` entries are skipped: they exist because THEY do not have the
    tensor, so writing it would produce a state dict with unexpected keys.
    """

    var writer: SafeTensorsWriter
    var map: TorchNameMap
    var written: List[String]
    var unmapped: List[String]
    var skipped: Int

    def __init__(out self, var map: TorchNameMap):
        self.writer = SafeTensorsWriter()
        self.map = map^
        self.written = List[String]()
        self.unmapped = List[String]()
        self.skipped = 0

    def __init__(out self, *, deinit move: Self):
        self.writer = move.writer^
        self.map = move.map^
        self.written = move.written^
        self.unmapped = move.unmapped^
        self.skipped = move.skipped

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
        comptime GP = String(Self.GRAPH_PREFIX)
        if not name.startswith(GP):
            self.skipped += 1
            return
        var local = String(name[byte = GP.byte_length() :])
        var i = self.map.index_of_ours(local)
        if i < 0:
            self.unmapped.append(local^)
            return
        if self.map.kind[i] == TN_ZEROS:
            return

        comptime if target == "gpu":
            param.download(ctx.value())

        var shape = self.map.their_shape(i)
        if self.map.numel(i) != N:
            raise Error(
                "SaveTorchNamed: '" + local + "' holds " + String(N)
                + " values but the map declares " + _shape_str(shape) + " = "
                + String(self.map.numel(i))
            )
        var key = String(self.map.theirs[i])
        if self.map.kind[i] == TN_TRANSPOSE:
            # Ours is [C, R]; theirs is [R, C]. One permutation, read the
            # other way round -- the same `transpose_2d` the loader uses, so
            # an export and a re-import cannot disagree.
            var src = List[Float32](unsafe_uninit_length=N)
            for k in range(N):
                src[k] = Float32(param.data[k])
            var t = transpose_2d(src, shape[1], shape[0])
            self.writer.add_f32(key^, shape, t, N)
        else:
            var src = List[Float32](unsafe_uninit_length=N)
            for k in range(N):
                src[k] = Float32(param.data[k])
            self.writer.add_f32(key^, shape, src, N)
        self.written.append(local^)

    def report(self, what: String) raises:
        if len(self.unmapped) > 0:
            raise Error(
                what + ": " + String(len(self.unmapped)) + " parameter(s)"
                " under '" + String(Self.GRAPH_PREFIX) + "' are not in the"
                " name map, first '" + self.unmapped[0] + "' — the export"
                " would silently omit them"
            )
        if len(self.written) == 0:
            raise Error(
                what + ": matched NOTHING under '" + String(Self.GRAPH_PREFIX)
                + "'"
            )


def _same_shape(ref a: List[Int], ref b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _shape_str(ref s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ", "
        out += String(s[i])
    return out + "]"


def _size_err(key: String, local: String, got: Int, want: Int) -> String:
    return (
        "LoadTorchNamed: '" + key + "' -> '" + local + "' has " + String(got)
        + " values but the param holds " + String(want)
    )
