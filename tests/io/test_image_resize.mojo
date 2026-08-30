# +--------------------------------------------------------------------------+ #
# | The Mojo bilinear resize, against Pillow, byte for byte
# +--------------------------------------------------------------------------+ #
"""Gates `mojo_rl/io/image.mojo` on Pillow's own output.

    pixi run python tools/io/dump_resize_reference.py --out /tmp/resize_ref
    pixi run mojo run -I . tests/io/test_image_resize.mojo /tmp/resize_ref

⚠ **ZERO TOLERANCE IS THE POINT.** A resize that is one LSB off everywhere is
a perfectly good resize and a broken port: the ACT store on disk was written
with Pillow, so anything short of byte-identical means the Mojo importer
produces a different dataset and the store's checksum stops meaning anything.
A tolerance here would pass every version of this code that was ever nearly
right.

The failure report prints the max |delta| and the fraction of bytes that
differ, because those two numbers separate the three ways this goes wrong:
a windowing error moves a FEW bytes a LOT, a rounding error moves MANY bytes
by ONE, and a transposed axis moves nearly everything.
"""

from std.os.path import exists
from std.sys import argv

from mojo_rl.io.image import resize_bilinear_pil


comptime DEFAULT_REF = "/tmp/resize_ref"
comptime SENTINEL = "/cases.txt"
comptime GEN = (
    "pixi run python tools/io/dump_resize_reference.py --out /tmp/resize_ref"
)

comptime USAGE = (
    "usage: mojo run -I . tests/io/test_image_resize.mojo <ref_dir>\n"
    "  generate <ref_dir> with:\n"
    "    pixi run python tools/io/dump_resize_reference.py --out <ref_dir>"
)


def read_raw(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def main() raises:
    var args = argv()
    var ref_dir = String(DEFAULT_REF) if len(args) < 2 else String(args[1])
    if len(args) > 2:
        print(USAGE)
        raise Error("test_image_resize: expected at most one argument")
    if not exists(ref_dir + SENTINEL):
        raise Error(
            "no reference dump at " + ref_dir + " — generate it with:\n    "
            + GEN
        )

    var cf = open(ref_dir + "/cases.txt", "r")
    var cases = cf.read()
    cf.close()

    print("Bilinear resize vs Pillow (BILINEAR)")
    print("  reference dump: " + ref_dir)
    print("")

    var fails = 0
    var n_cases = 0
    var total_bytes = 0
    var scratch = List[UInt8]()

    for line in cases.splitlines():
        var s = String(line)
        if s == "":
            continue
        var parts = s.split(" ")
        if len(parts) != 6:
            raise Error("malformed case line: " + s)
        var tag = String(parts[0])
        var ih = atol(parts[1])
        var iw = atol(parts[2])
        var oh = atol(parts[3])
        var ow = atol(parts[4])
        var ch = atol(parts[5])

        var src = read_raw(ref_dir + "/" + tag + ".src")
        var want = read_raw(ref_dir + "/" + tag + ".dst")
        if len(src) != ih * iw * ch:
            raise Error("case " + tag + ": source is the wrong size")
        if len(want) != oh * ow * ch:
            raise Error("case " + tag + ": reference is the wrong size")

        var got = List[UInt8](unsafe_uninit_length = oh * ow * ch)
        resize_bilinear_pil(
            src.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
            .as_unsafe_any_origin(),
            ih,
            iw,
            got.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
            .as_unsafe_any_origin(),
            oh,
            ow,
            scratch,
            ch,
        )

        var diff = 0
        var worst = 0
        for i in range(len(want)):
            var d = Int(got[i]) - Int(want[i])
            if d != 0:
                diff += 1
                if d < 0:
                    d = -d
                if d > worst:
                    worst = d
        n_cases += 1
        total_bytes += len(want)
        if diff == 0:
            print(
                "  PASS  " + tag + "  " + String(len(want))
                + " bytes identical"
            )
        else:
            fails += 1
            print(
                "  FAIL  " + tag + "  " + String(diff) + "/"
                + String(len(want)) + " bytes differ, max|delta| = "
                + String(worst)
            )

    print("")
    print(
        "  " + String(total_bytes) + " bytes compared over " + String(n_cases)
        + " cases"
    )
    if n_cases == 0 or total_bytes == 0:
        raise Error("gate: no cases compared — this run proved nothing")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("resize gate failed")
