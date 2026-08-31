# +--------------------------------------------------------------------------+ #
# | Our safetensors reader vs the reference implementation — both directions
# +--------------------------------------------------------------------------+ #
"""Gates `mojo_rl/io/safetensors.mojo` against `safetensors` itself.

    pixi run -e act-ref python tools/nn/dump_safetensors_reference.py --out /tmp/st_ref
    pixi run mojo run -I . tests/io/test_safetensors_reference.mojo /tmp/st_ref
    pixi run -e act-ref python tools/nn/dump_safetensors_reference.py --verify /tmp/st_ref

The third command is not optional and is not a formality: this file writes
`ours.safetensors` but CANNOT judge it. A round-trip through our own reader
would pass on any self-consistent format, including a wrong one. Only the
reference library reading our bytes says the file is a safetensors file.

## What is compared

Every float value, as an f32 BIT PATTERN. The values are multiples of 0.25 —
exact in f16, f32 and f64 — so a widening bug cannot be absorbed by rounding,
and comparing integers means no float text is parsed on either side.

⚠ Reading a tensor is a byte reinterpretation, so the default failure is the
right COUNT of plausible numbers. Two guards against a vacuous pass: the
expectation file must declare entries (a missing dump would otherwise be
green with zero comparisons), and the number of values compared is printed
per tensor and totalled.
"""

from std.memory import bitcast
from std.os.path import exists
from std.sys import argv

from mojo_rl.io.safetensors import (
    SafeTensors,
    SafeTensorsWriter,
    ST_F32,
    dtype_name,
    is_float_dtype,
)
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.torch_names import (
    LoadTorchNamed,
    SaveTorchNamed,
    TorchNameMap,
)
from mojo_rl.nn.primitives.linear import Linear

from max.gpu.host import DeviceContext


# A `Linear` is the ONE thing in this framework whose layout differs from
# torch's, and ResNet18 -- the only other `TorchNameMap` instance -- has none
# before `fc`, which the ACT backbone never reaches. So the transpose is
# machinery that nothing else exercises, and the Python side checks it against
# `torch.nn.Linear` rather than against our own reading of the convention.
comptime LIN_IN = 5
comptime LIN_OUT = 3
comptime LIN = Linear[LIN_IN, LIN_OUT]


comptime DEFAULT_REF = "/tmp/st_ref"
comptime GEN = (
    "pixi run -e act-ref python tools/nn/dump_safetensors_reference.py"
    " --out /tmp/st_ref"
)
comptime USAGE = (
    "usage: mojo run -I . tests/io/test_safetensors_reference.mojo <ref_dir>\n"
    "  generate <ref_dir> with:\n    " + GEN
)

# The values `dump_safetensors_reference.py:values()` writes. Restated here
# rather than read from the dump: the file we WRITE has to be checkable by the
# Python side without this gate telling it what to expect.
def ref_value(i: Int) -> Float32:
    return Float32(((i * 37) % 101)) * 0.25 - 12.5


struct FillRef(ParamVisitor):
    """`vals[i] = ref_value(i)`, restarting at every tensor so the Python side
    can reproduce each one without knowing the walk order."""

    def __init__(out self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

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
        param.ensure(N)
        for i in range(N):
            param.data[i] = Scalar[DT](ref_value(i))


struct GrabAll(ParamVisitor):
    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^

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
        for i in range(N):
            self.vals.append(param.data[i])


def linear_map() raises -> TorchNameMap:
    var m = TorchNameMap()
    m.add_linear(String("weight"), String("fc.weight"), LIN_OUT, LIN_IN)
    var bs: List[Int] = [LIN_OUT]
    m.add(String("bias"), String("fc.bias"), bs)
    return m^


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def hex32(s: String) raises -> UInt32:
    var v = UInt32(0)
    var b = s.as_bytes()
    if len(b) != 8:
        raise Error("expected an 8-digit hex word, got '" + s + "'")
    for i in range(8):
        var c = Int(b[i])
        var d: Int
        if c >= ord("0") and c <= ord("9"):
            d = c - ord("0")
        elif c >= ord("a") and c <= ord("f"):
            d = c - ord("a") + 10
        elif c >= ord("A") and c <= ord("F"):
            d = c - ord("A") + 10
        else:
            raise Error("not hex: '" + s + "'")
        v = (v << UInt32(4)) | UInt32(d)
    return v


def bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


@fieldwise_init
struct Expect(Copyable, Movable):
    var name: String
    var dtype: String
    var shape: String
    var bits: List[String]


def parse_expected(path: String) raises -> List[Expect]:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    var out = List[Expect]()
    for line in text.splitlines():
        var s = String(line)
        if s == "":
            continue
        # ⚠ SPACE-separated, and `-` for an empty field. `splitlines()` in
        # Mojo breaks on TAB too, so a tab-separated table would arrive here
        # as one field per line — see the dump script's header.
        var parts = s.split(" ")
        if len(parts) != 4:
            raise Error(
                "malformed expectation line (" + String(len(parts))
                + " fields): " + s
            )
        var bits = List[String]()
        var vs = String(parts[3])
        if vs != "-":
            for w in vs.split(","):
                bits.append(String(w))
        var shape = String(parts[2])
        out.append(
            Expect(
                String(parts[0]),
                String(parts[1]),
                String("") if shape == "-" else shape^,
                bits^,
            )
        )
    return out^


def shape_csv(ref f: SafeTensors, name: String) raises -> String:
    var s = f.shape(name)
    var out = String("")
    for i in range(len(s)):
        if i > 0:
            out += ","
        out += String(s[i])
    return out^


def main() raises:
    var args = argv()
    var ref_dir = String(DEFAULT_REF) if len(args) < 2 else String(args[1])
    if len(args) > 2:
        print(USAGE)
        raise Error("expected at most one argument")
    if not exists(ref_dir + "/expected.txt"):
        raise Error(
            "no reference dump at " + ref_dir + " — generate it with:\n    "
            + GEN
        )

    print("safetensors: ours vs the reference implementation")
    print("  reference dump: " + ref_dir)
    print("")

    var expects = parse_expected(ref_dir + "/expected.txt")
    if len(expects) == 0:
        raise Error(
            "gate: the expectation file declares no tensors — nothing would"
            " have been compared"
        )

    var fails = 0
    var compared = 0

    var basic = SafeTensors(ref_dir + "/ref_basic.safetensors")
    var dtypes = SafeTensors(ref_dir + "/ref_dtypes.safetensors")

    # ── the reference's key order is not ours ─────────────────────────────
    # `enc.weight` is added before `enc.bias` by any sane walk; the reference
    # emits them in its own order. Anything positional reads the right sizes
    # into the wrong tensors on a file that is entirely valid.
    if basic.names[0] == "enc.weight":
        fails += 1
        print(
            "  FAIL  header order: the reference file happens to lead with"
            " 'enc.weight', so this gate is no longer showing that order"
            " differs — pick a different probe"
        )
    else:
        print(
            "  PASS  header order differs from write order (leads with '"
            + basic.names[0] + "')"
        )

    for ei in range(len(expects)):
        ref e = expects[ei]
        var in_basic = basic.has(e.name)
        ref f = basic if in_basic else dtypes
        if not f.has(e.name):
            fails += 1
            print("  FAIL  " + e.name + ": in neither reference file")
            continue

        var dt = f.dtype_of(e.name)
        if dtype_name(dt) != e.dtype:
            fails += 1
            print(
                "  FAIL  " + e.name + ": dtype " + dtype_name(dt) + ", want "
                + e.dtype
            )
            continue
        var got_shape = shape_csv(f, e.name)
        if got_shape != e.shape:
            fails += 1
            print(
                "  FAIL  " + e.name + ": shape [" + got_shape + "], want ["
                + e.shape + "]"
            )
            continue

        if not is_float_dtype(dt):
            # ⚠ These MUST refuse. `num_batches_tracked` is I64 in every real
            # torchvision file; reinterpreting its bits as f32 yields
            # denormals, not an error, and a caller would carry them into a
            # BatchNorm.
            var refused = False
            try:
                var _ = f.read_f32(e.name)
            except:
                refused = True
            if refused:
                print(
                    "  PASS  " + e.name + "  " + e.dtype
                    + " refused by read_f32"
                )
            else:
                fails += 1
                print(
                    "  FAIL  " + e.name + ": read_f32 accepted " + e.dtype
                )
            continue

        var vals = f.read_f32(e.name)
        if len(vals) != len(e.bits):
            fails += 1
            print(
                "  FAIL  " + e.name + ": " + String(len(vals)) + " values,"
                " want " + String(len(e.bits))
            )
            continue
        var bad = 0
        var detail = String("")
        for i in range(len(vals)):
            var want = hex32(e.bits[i])
            if bits_of(vals[i]) != want:
                bad += 1
                if detail == "":
                    detail = (
                        " first at " + String(i) + ": got "
                        + String(vals[i]) + ", want bits 0x" + e.bits[i]
                    )
        compared += len(vals)
        if bad == 0:
            print(
                "  PASS  " + e.name + "  " + e.dtype + " [" + e.shape + "]  "
                + String(len(vals)) + " values compared"
            )
        else:
            fails += 1
            print(
                "  FAIL  " + e.name + ": " + String(bad) + " of "
                + String(len(vals)) + " differ," + detail
            )

    # ── metadata ─────────────────────────────────────────────────────────
    if basic.metadata(String("producer")) != "safetensors-reference":
        fails += 1
        print(
            "  FAIL  __metadata__ producer: '"
            + basic.metadata(String("producer")) + "'"
        )
    else:
        print("  PASS  __metadata__ round-trips from the reference file")

    print("")
    print("  " + String(compared) + " float values compared")
    if compared == 0:
        raise Error("gate: compared nothing")

    # ── now write a file for the reference to read ───────────────────────
    var w = SafeTensorsWriter()
    w.add_metadata(String("format"), String("pt"))
    w.add_metadata(String("producer"), String("mojo-rl"))
    _add(w, String("mojo.weight"), [3, 4])
    _add(w, String("mojo.bias"), [4])
    _add(w, String("mojo.scalar"), [])
    _add(w, String("mojo.empty"), [0])
    var out_path = ref_dir + "/ours.safetensors"
    w.save(out_path)
    print("  wrote " + out_path + " (" + String(w.size()) + " tensors)")

    # ── a Linear in TORCH's layout ───────────────────────────────────────
    var lin = LIN.make["cpu", Deterministic](None)
    var fill = FillRef()
    lin.for_each_param["cpu", FillRef](fill, None)
    var before = GrabAll()
    lin.for_each_param["cpu", GrabAll](before, None)

    var sv = SaveTorchNamed[""](linear_map())
    lin.for_each_param["cpu", SaveTorchNamed[""]](sv, None)
    sv.report(String("linear export"))
    sv.writer.save(ref_dir + "/ours_linear.safetensors")

    var lf = SafeTensors(ref_dir + "/ours_linear.safetensors")
    check(
        fails,
        "the exported Linear has torch's shape [out, in]",
        lf.shape_str(String("fc.weight"))
        == "[" + String(LIN_OUT) + ", " + String(LIN_IN) + "]",
        lf.shape_str(String("fc.weight")),
    )
    # ⚠ Anti-vacuity: a transpose that did nothing would leave the flat order
    # unchanged, and the shape check above would still pass. 5x3 is not square
    # precisely so this can tell the difference.
    var flat = lf.read_f32(String("fc.weight"))
    var moved = 0
    for i in range(len(flat)):
        if bits_of(flat[i]) != bits_of(Float32(before.vals[i])):
            moved += 1
    check(
        fails,
        "the export is actually transposed, not merely reshaped",
        moved > 0,
        String(moved) + " of " + String(len(flat)) + " positions changed",
    )

    # And back: our own map, used in the other direction, must be an identity.
    var lin2 = LIN.make["cpu", Deterministic](None)
    var lv = LoadTorchNamed[""](
        SafeTensors(ref_dir + "/ours_linear.safetensors"), linear_map()
    )
    lin2.for_each_param["cpu", LoadTorchNamed[""]](lv, None)
    lv.report(String("linear import"))
    var after = GrabAll()
    lin2.for_each_param["cpu", GrabAll](after, None)
    var rbad = 0
    for i in range(len(before.vals)):
        if bits_of(Float32(before.vals[i])) != bits_of(Float32(after.vals[i])):
            rbad += 1
    check(
        fails,
        "export then import is the identity",
        rbad == 0 and len(before.vals) == LIN_IN * LIN_OUT + LIN_OUT,
        String(len(before.vals)) + " compared, " + String(rbad) + " differ",
    )
    print("")
    print("  NOW RUN, or this half is untested:")
    print(
        "    pixi run -e act-ref python"
        " tools/nn/dump_safetensors_reference.py --verify " + ref_dir
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("safetensors reference gate failed")


def _add(mut w: SafeTensorsWriter, var name: String, dims: List[Int]) raises:
    var n = 1
    for i in range(len(dims)):
        n *= dims[i]
    var vals = List[Float32]()
    for i in range(n):
        vals.append(ref_value(i))
    w.add_f32(name^, dims, vals, n)
