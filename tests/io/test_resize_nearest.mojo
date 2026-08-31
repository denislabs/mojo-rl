# +--------------------------------------------------------------------------+ #
# | NEAREST resize vs Pillow, over a random size sweep
# +--------------------------------------------------------------------------+ #
"""Gate `resize_nearest_pil` in `mojo_rl/io/image.mojo`.

    pixi run python tools/io/dump_resize_nearest_reference.py --out /tmp/nearest_ref
    pixi run mojo run -I . tests/io/test_resize_nearest.mojo [/tmp/nearest_ref]

Both Craftax sprite loaders resize their atlases with `PIL.Image.NEAREST`, so
a one-pixel disagreement is a different sprite sheet — wrong, and not wrong in
a way anything reports.

⚠ A HANDFUL OF HAND-PICKED SIZES WOULD PASS WITH THE WRONG FORMULA. NEAREST
interpolates nothing; the only thing that can differ is which source pixel an
output pixel picks, and that only diverges where the exact coordinate lands on
an integer and double rounding decides the side. `floor((x + 0.5) * in / out)`
agrees with Pillow on most size pairs and disagrees on 93 of 600 random ones;
`floor(x * s + 0.5 * s)` on 104. Pillow keeps a running accumulator instead,
so its floating-point drift is part of the answer. Hence a large random sweep
rather than a few round numbers.
"""

from std.memory import alloc
from std.os.path import exists
from std.sys import argv

from mojo_rl.io.fileio import read_file_bytes
from mojo_rl.io.image import resize_nearest_pil
from mojo_rl.nn.core.ptr import mptr


def main() raises:
    var root = String("/tmp/nearest_ref")
    var args = argv()
    if len(args) > 1:
        root = String(args[1])
    var index_path = root + "/index.tsv"
    if not exists(index_path):
        raise Error(
            "no reference dump at " + root + " — generate it with:\n"
            "    pixi run python tools/io/dump_resize_nearest_reference.py"
            " --out " + root
        )

    print("=== resize_nearest_pil vs Pillow ===")
    var f = open(index_path, "r")
    var text = f.read()
    f.close()

    var cases = 0
    var compared = 0
    var differing = 0
    var first_bad = String("")

    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li]).strip()
        if line.byte_length() == 0:
            continue
        var c = line.split("\t")
        if len(c) != 5:
            raise Error("index.tsv line " + String(li) + " is malformed")
        var key = String(c[0])
        var ih = Int(String(c[1]))
        var iw = Int(String(c[2]))
        var oh = Int(String(c[3]))
        var ow = Int(String(c[4]))

        var src = read_file_bytes(root + "/" + key + "_src.raw")
        var want = read_file_bytes(root + "/" + key + "_dst.raw")
        if len(src) != ih * iw * 4 or len(want) != oh * ow * 4:
            raise Error("case " + key + ": the dumps have the wrong size")

        var dst = List[UInt8]()
        dst.resize(oh * ow * 4, 0)
        resize_nearest_pil(
            mptr(src.unsafe_ptr()), ih, iw, mptr(dst.unsafe_ptr()), oh, ow, 4
        )
        for i in range(len(want)):
            compared += 1
            if dst[i] != want[i]:
                differing += 1
                if first_bad.byte_length() == 0:
                    var px = i // 4
                    first_bad = (
                        "case " + key + " (" + String(ih) + "x" + String(iw)
                        + " -> " + String(oh) + "x" + String(ow) + ") at row "
                        + String(px // ow) + " col " + String(px % ow)
                        + " channel " + String(i % 4) + ": ours "
                        + String(Int(dst[i])) + ", Pillow "
                        + String(Int(want[i]))
                    )
        cases += 1

    print(
        "  " + String(cases) + " size pairs, " + String(compared)
        + " bytes compared, " + String(differing) + " differing"
    )
    if cases < 50:
        raise Error("vacuous: only " + String(cases) + " cases ran")
    if differing != 0:
        raise Error(String(differing) + " bytes differ; the first is " + first_bad)

    print("[PASS] resize_nearest_pil")
