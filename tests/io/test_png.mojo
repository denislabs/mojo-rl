# +--------------------------------------------------------------------------+ #
# | The PNG decoder vs Pillow, on every filter
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/io/png.mojo`.

    pixi run build-http                                    # ONCE (inflate)
    pixi run mojo run -I . tests/io/test_png.mojo [<fixture-dir>]

Self-contained: `tools/io/make_png_fixtures.py` writes both the PNGs and, from
**Pillow**, the pixels they should decode to.

⚠ THIS GATE EXISTS BECAUSE THE REAL DATA DID NOT COVER THE CODE. CIFAR-10's
60,000 images use scanline filters 0, 1, 2 and 4 — measured — and never 3
(`Average`). With the `Average` branch deliberately broken,
`tests/nn/test_cifar10_prep.mojo` stayed green across 30,730,000 compared
bytes. The filter type is the ENCODER's choice, so the only way to cover all
five is to write the fixtures.

Also covered, because each is a separate way to be wrong:

* colour types 0 / 2 / 4 / 6 — the channel count changes `bpp`, which every
  filter's "pixel to the left" depends on,
* a 13x17 image, so neither dimension is a round number,
* a 1x1 image, where every filter's neighbours are all out of bounds,
* a 256x256 image, large enough that Pillow splits `IDAT` across chunks —
  they must be concatenated before inflating, since the split is not aligned
  to anything in the zlib stream,
* palette at depths 8, 4, 2 and 1, with and without `tRNS` — 248 of this
  repo's asset PNGs are palettes and 7 of those are sub-byte, where samples
  pack MSB-first into bytes and a row rounds up to a whole one,
* 16-bit, interlaced and colour-key-`tRNS` images, which must RAISE. Each
  would otherwise decode to a plausible wrong picture that nothing downstream
  can distinguish from the right one.
"""

from std.os.path import exists
from std.sys import argv

from mojo_rl.io.fileio import read_file_bytes
from mojo_rl.io.png import load_png_file
from mojo_rl.io.proc import run_capture


def _compare(dir: String, name: String) raises -> Int:
    """Decode `<name>.png` and compare every byte to `<name>.raw`."""
    var img = load_png_file(dir + "/" + name + ".png")
    var want = read_file_bytes(dir + "/" + name + ".raw")
    ref got = img.pixels
    if len(want) != len(got):
        raise Error(
            name + ": decoded " + String(len(got)) + " bytes, Pillow says "
            + String(len(want)) + " (" + String(img.width) + "x"
            + String(img.height) + "x" + String(img.channels) + ")"
        )
    for i in range(len(want)):
        if want[i] != got[i]:
            var px = i // img.channels
            raise Error(
                name + ": byte " + String(i) + " (row "
                + String(px // img.width) + ", col " + String(px % img.width)
                + ", channel " + String(i % img.channels) + ") is "
                + String(Int(got[i])) + ", Pillow says " + String(Int(want[i]))
            )
    return len(want)


def _must_raise(dir: String, name: String, what: String) raises:
    var raised = False
    try:
        _ = load_png_file(dir + "/" + name + ".png")
    except:
        raised = True
    if not raised:
        raise Error(
            name + ": " + what + " was accepted. It decodes to a plausible"
            " WRONG image, which nothing downstream can detect."
        )


def main() raises:
    var dir = String("/tmp/mojo_rl_png_fixtures")
    var args = argv()
    if len(args) > 1:
        dir = String(args[1])
    if not exists(dir + "/pal1.png"):
        _ = run_capture("python3 tools/io/make_png_fixtures.py " + dir)
    if not exists(dir + "/pal1.png"):
        raise Error(
            "no PNG fixtures at " + dir + " — generate them with:\n"
            "    pixi run python tools/io/make_png_fixtures.py " + dir
        )

    print("=== io/png vs Pillow ===")
    var cases = List[String]()
    for ft in range(5):
        cases.append("filter" + String(ft))
    cases.append(String("filter_mixed"))
    cases.append(String("grey"))
    cases.append(String("grey_alpha"))
    cases.append(String("rgba"))
    cases.append(String("odd_size"))
    cases.append(String("one_pixel"))
    cases.append(String("big_idat"))
    cases.append(String("pal8"))
    cases.append(String("pal4"))
    cases.append(String("pal2"))
    cases.append(String("pal1"))
    cases.append(String("pal8_trns"))
    cases.append(String("pal4_trns"))

    var total = 0
    for i in range(len(cases)):
        var n = _compare(dir, cases[i])
        total += n
        print("  " + cases[i] + ": " + String(n) + " bytes identical")

    # ⚠ Rows-COMPARED beside rows-DIFFERING: a loop that ran zero times also
    # reports zero differences.
    print("  " + String(len(cases)) + " images, " + String(total)
          + " bytes compared, 0 differing")
    if len(cases) != 18:
        raise Error("vacuous: only " + String(len(cases)) + " cases ran")

    _must_raise(dir, String("reject_16bit"), String("a 16-bit image"))
    _must_raise(dir, String("reject_interlaced"), String("an Adam7 image"))
    _must_raise(
        dir, String("reject_trns_rgb"), String("colour-key transparency")
    )
    print("  16-bit, interlaced and colour-key tRNS all refused")

    print("[PASS] io/png")
