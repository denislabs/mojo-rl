"""Checkpoint — named-section param/state + optimizer-moment save/load.

Walks a Module's Params via `for_each_param` then States via `for_each_state`,
writing one named section per field. Two formats:

v3 (CURRENT — what every save now writes):

    storage-ckpt v3
    P <dotted-name> <size> <has_moments:0|1>\n
    <size raw Scalar[DT] payload bytes>[<m bytes><v bytes>]
    S <dotted-name> <size>\n
    <size raw Scalar[DT] payload bytes>
    ...

v2 (LEGACY — still readable; loaders dispatch on the header line): identical
section headers but one ASCII float per line. v2 hit its ceiling at DreamerV3
size200m: ~5 GB of text per save, silently TRUNCATED at the single-write(2)
syscall cap (0x7FFFF000 ≈ 2 GiB) → corrupt checkpoints, and the per-line
`List[String]` loader could not have held it anyway. v3 payloads are raw
little-endian bytes (3× smaller, no atof/String churn) and ALL file I/O goes
through explicit ≤1 GiB chunks (`_write_file_bytes`/`_read_file_bytes`).

The dotted name (from the A2 name-threading walker, e.g. "0.weight") is VALIDATED
against the in-memory walk order on load — a name/size mismatch raises, catching
topology drift between save and load (the legacy named-section guarantee the
positional v1 format lacked).

Optimizer moments (Adam's per-param `m`/`v`, co-located on the Param) ride the
same param section when populated (`m.n >= N`), enabling exact training resume.
`save_moments=False` writes a model-only checkpoint (moments skipped). GPU params
download on save / upload on load.
"""

from std.ffi import external_call
from max.gpu.host import DeviceContext
from std.memory import unsafe_memcpy
from std.sys.info import size_of

from mojo_rl.nn.constants import DT
from .tensor import Tensor
from .param import ParamVisitor
from .module import Module

comptime _CKPT_CHUNK = 1 << 30  # 1 GiB — below every single-syscall I/O cap


def _write_file_bytes(var path: String, content: List[UInt8]) raises:
    """Chunked, ATOMIC file write. The payload lands in `path + ".tmp"` and is
    renamed over `path` only once fully written, so a crash mid-save can never
    destroy the previous good checkpoint (rename(2) is atomic on the same
    filesystem). Chunking: a single write(2) silently stops at 0x7FFFF000
    (~2 GiB) on Linux — the v2 corruption source — so never issue one call
    for the whole payload."""
    var tmp = path + ".tmp"
    with open(tmp, "w") as f:
        var off = 0
        while off < len(content):
            var take = len(content) - off
            if take > _CKPT_CHUNK:
                take = _CKPT_CHUNK
            # Bounded slice rather than a raw pointer + separate length: the
            # chunk bound is now checked against the buffer, not asserted.
            f.write_bytes(Span(content)[off : off + take])
            off += take
    var rc = external_call["rename", Int32](
        tmp.as_c_string_slice().unsafe_ptr(),
        path.as_c_string_slice().unsafe_ptr(),
    )
    if rc != 0:
        raise Error(
            "checkpoint save: atomic rename failed: " + tmp + " -> " + path
        )


def _read_file_bytes(path: String) raises -> List[UInt8]:
    """Chunked file read (read(2) has the same single-call cap as write)."""
    var out = List[UInt8]()
    with open(path, "r") as f:
        while True:
            var chunk = f.read_bytes(_CKPT_CHUNK)
            if len(chunk) == 0:
                break
            var old = len(out)
            out.resize(old + len(chunk), 0)
            unsafe_memcpy(
                dest=out.unsafe_ptr().unsafe_offset(old),
                src=chunk.unsafe_ptr(),
                count=len(chunk),
            )
    return out^


def _bytes_append_str(mut buf: List[UInt8], s: String):
    var sb = s.as_bytes()
    var old = len(buf)
    buf.resize(old + len(sb), 0)
    unsafe_memcpy(dest=buf.unsafe_ptr().unsafe_offset(old), src=sb.unsafe_ptr(), count=len(sb))


def _bytes_append_vals(mut buf: List[UInt8], t: Tensor, n: Int):
    comptime SB = size_of[Scalar[DT]]()
    var old = len(buf)
    buf.resize(old + n * SB, 0)
    unsafe_memcpy(
        dest=buf.unsafe_ptr().unsafe_offset(old),
        src=t.data.unsafe_ptr().unsafe_bitcast[UInt8](),
        count=n * SB,
    )


def _is_v3_header(bytes: List[UInt8]) -> Bool:
    var tag = String("storage-ckpt v3")
    var tb = tag.as_bytes()
    if len(bytes) < len(tb):
        return False
    for i in range(len(tb)):
        if bytes[i] != tb[i]:
            return False
    return True


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
        # ⚠ RESTORING A WEIGHT IS A WRITE, so it must advance `version` — the
        # same contract the optimizer honours via `ParamVersionBump`. Leaves
        # cache DERIVED copies of the weight gated on this counter (`w_pad`,
        # the K-alignment pad; `w_bf`, the AMP bf16 recast), and without the
        # bump a `make -> forward -> load_state -> forward` sequence keeps
        # serving the PRE-LOAD weight: the checkpoint loads, reports success,
        # and is silently ignored. A viewer switching checkpoints after it has
        # already acted hits exactly that.
        param.version += 1
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


struct BinaryCheckpointWriter(ParamVisitor):
    """V3 twin of `CheckpointWriter`: text section headers, raw-byte payloads.
    `mode`: 0 = Param (P, with optional moments), 1 = State (S)."""
    var content: List[UInt8]
    var mode: Int
    var save_moments: Bool

    def __init__(out self, save_moments: Bool = True):
        self.content = List[UInt8]()
        _bytes_append_str(self.content, String("storage-ckpt v3\n"))
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
            _bytes_append_str(
                self.content, "S " + name + " " + String(N) + "\n"
            )
            _bytes_append_vals(self.content, param, N)
            return
        var has_m = self.save_moments and m.n >= N and v.n >= N
        comptime if target == "gpu":
            if has_m:
                m.download(ctx.value())
                v.download(ctx.value())
        _bytes_append_str(
            self.content,
            "P " + name + " " + String(N) + " "
            + ("1" if has_m else "0") + "\n",
        )
        _bytes_append_vals(self.content, param, N)
        if has_m:
            _bytes_append_vals(self.content, m, N)
            _bytes_append_vals(self.content, v, N)


struct BinaryCheckpointReader(ParamVisitor):
    """V3 twin of `CheckpointReader`: byte-cursor over the whole file, same
    name/size/topology validation as v2."""
    var bytes: List[UInt8]
    var cur: Int
    var mode: Int

    def __init__(out self, var bytes: List[UInt8]):
        self.bytes = bytes^
        self.cur = 0
        self.mode = 0
        # Skip the "storage-ckpt v3" header line.
        while self.cur < len(self.bytes) and self.bytes[self.cur] != UInt8(10):
            self.cur += 1
        if self.cur < len(self.bytes):
            self.cur += 1

    def _next_line(mut self) raises -> String:
        if self.cur >= len(self.bytes):
            raise Error("checkpoint: unexpected end of file")
        var s = String("")
        while self.cur < len(self.bytes) and self.bytes[self.cur] != UInt8(10):
            s += chr(Int(self.bytes[self.cur]))
            self.cur += 1
        if self.cur < len(self.bytes):
            self.cur += 1  # consume '\n'
        return s^

    def _take_vals(mut self, mut t: Tensor, n: Int) raises:
        comptime SB = size_of[Scalar[DT]]()
        if self.cur + n * SB > len(self.bytes):
            raise Error("checkpoint: unexpected end of file")
        unsafe_memcpy(
            dest=t.data.unsafe_ptr().unsafe_bitcast[UInt8](),
            src=self.bytes.unsafe_ptr().unsafe_offset(self.cur),
            count=n * SB,
        )
        self.cur += n * SB

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var hdr = self._next_line()
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
        self._take_vals(param, N)
        # See the note in `CheckpointReader.visit` — restoring a weight must
        # advance `version` or the version-gated derived caches (`w_pad`,
        # `w_bf`) keep serving the pre-load weight.
        param.version += 1
        if self.mode == 0 and len(toks) >= 4 and toks[3] == "1":
            m.ensure(N)
            v.ensure(N)
            self._take_vals(m, N)
            self._take_vals(v, N)
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
    """Write a v3 named checkpoint: Params (+ moments if populated) then States."""
    var w = BinaryCheckpointWriter(save_moments)
    w.mode = 0
    model.for_each_param[target](w, ctx)
    w.mode = 1
    model.for_each_state[target](w, ctx)
    _write_file_bytes(path, w.content)


def load_params[
    target: StaticString, M: Module
](mut model: M, path: String, ctx: Optional[DeviceContext] = None) raises:
    """Load a named checkpoint (v3 binary, or legacy v2 text — dispatched on
    the header line), validating names/sizes against `model`."""
    var bytes = _read_file_bytes(path)
    if _is_v3_header(bytes):
        var r = BinaryCheckpointReader(bytes^)
        r.mode = 0
        model.for_each_param[target](r, ctx)
        r.mode = 1
        model.for_each_state[target](r, ctx)
        return
    # Legacy v2 text checkpoint.
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


def save_params_multi[
    target: StaticString, *Ms: Module
](
    path: String,
    ctx: Optional[DeviceContext],
    save_moments: Bool,
    mut *models: *Ms,
) raises:
    """Write N models into ONE v3 checkpoint file: a single header, then each
    model's Param + State sections, in pack order. `load_params_multi` walks the
    same models in the same order, so each section's dotted name is validated
    against its own model — duplicate names across models never collide. Replaces
    the per-model sidecar layout (plain `save_params` is whole-file-per-model)."""
    var w = BinaryCheckpointWriter(save_moments)

    comptime for i in range(models.__len__()):
        w.mode = 0
        models[i].for_each_param[target](w, ctx)
        w.mode = 1
        models[i].for_each_state[target](w, ctx)
    _write_file_bytes(path, w.content)


def load_params_multi[
    target: StaticString, *Ms: Module
](
    path: String,
    ctx: Optional[DeviceContext],
    mut *models: *Ms,
) raises:
    """Load a single-file multi-model checkpoint written by `save_params_multi`
    (v3 binary, or legacy v2 text), walking the models in the same pack order
    and validating each one's names/sizes against the file's sections."""
    var bytes = _read_file_bytes(path)
    if _is_v3_header(bytes):
        var rb = BinaryCheckpointReader(bytes^)
        comptime for i in range(models.__len__()):
            rb.mode = 0
            models[i].for_each_param[target](rb, ctx)
            rb.mode = 1
            models[i].for_each_state[target](rb, ctx)
        return
    var content: String
    with open(path, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var body = List[String]()
    for li in range(len(lines)):
        if lines[li].startswith("storage-ckpt"):
            continue
        body.append(lines[li])
    var r = CheckpointReader(body^)

    comptime for i in range(models.__len__()):
        r.mode = 0
        models[i].for_each_param[target](r, ctx)
        r.mode = 1
        models[i].for_each_state[target](r, ctx)
