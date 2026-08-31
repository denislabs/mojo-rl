# +--------------------------------------------------------------------------+ #
# | The PNG decoder against every asset in the repo
# +--------------------------------------------------------------------------+ #
"""Decode all 1,443 sprite PNGs and compare them to Pillow, byte for byte.

    pixi run build-http                                            # ONCE
    pixi run python tools/io/dump_asset_png_reference.py --out /tmp/png_assets
    pixi run mojo run -I . tests/io/test_png_assets.mojo [/tmp/png_assets]

⚠ TWO GATES, BECAUSE NEITHER COVERS THE OTHER. `tests/io/test_png.mojo` uses
fixtures written to exercise things the real assets never contain — CIFAR-10's
60,000 images never use the `Average` filter, and a sabotaged `Average` branch
left that gate green over 30,730,000 bytes. This one is the converse: real
encoders, real palettes, real chunk layouts, 1,443 files nobody synthesised,
including the 248 palette images and the 7 sub-byte ones that make the
`procgen` and `craftax` sprite loaders work.

Everything is compared as RGBA, which is what all three sprite loaders asked
`PIL.Image.convert("RGBA")` for and what they now ask `io/png.to_rgba` for.
"""

from std.os.path import exists
from std.sys import argv

from mojo_rl.io.fileio import read_file_bytes
from mojo_rl.io.png import load_png_file, to_rgba


def main() raises:
    var root = String("/tmp/png_assets")
    var args = argv()
    if len(args) > 1:
        root = String(args[1])
    var index_path = root + "/index.tsv"
    if not exists(index_path):
        raise Error(
            "no reference dump at " + root + " — generate it with:\n"
            "    pixi run python tools/io/dump_asset_png_reference.py --out "
            + root
        )

    print("=== io/png vs Pillow, on this repo's assets ===")
    var f = open(index_path, "r")
    var text = f.read()
    f.close()

    var images = 0
    var bytes_compared = 0
    var differing = 0
    var first_bad = String("")
    var palettes = 0

    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li]).strip()
        if line.byte_length() == 0:
            continue
        var cols = line.split("\t")
        if len(cols) != 4:
            raise Error("index.tsv line " + String(li) + " has " + String(len(cols)) + " columns")
        var key = String(cols[0])
        var w = Int(String(cols[1]))
        var h = Int(String(cols[2]))
        var path = String(cols[3])

        var img = load_png_file(path)
        if img.width != w or img.height != h:
            raise Error(
                path + ": decoded " + String(img.width) + "x"
                + String(img.height) + ", Pillow says " + String(w) + "x"
                + String(h)
            )
        if img.channels == 4:
            palettes += 1  # RGBA or an expanded palette
        var got = to_rgba(img)
        var want = read_file_bytes(root + "/" + key + ".raw")
        if len(got) != len(want):
            raise Error(
                path + ": " + String(len(got)) + " RGBA bytes, Pillow says "
                + String(len(want))
            )
        for i in range(len(want)):
            bytes_compared += 1
            if got[i] != want[i]:
                differing += 1
                if first_bad.byte_length() == 0:
                    var px = i // 4
                    first_bad = (
                        path + " byte " + String(i) + " (row "
                        + String(px // w) + ", col " + String(px % w)
                        + ", " + String(i % 4) + "): ours "
                        + String(Int(got[i])) + ", Pillow "
                        + String(Int(want[i]))
                    )
        images += 1

    # ⚠ Images-COMPARED beside bytes-DIFFERING: an index that failed to parse
    # would otherwise report a triumphant zero.
    print(
        "  " + String(images) + " images, " + String(bytes_compared)
        + " RGBA bytes compared, " + String(differing) + " differing"
    )
    if images < 100:
        raise Error(
            "vacuous: only " + String(images) + " images were compared — is the"
            " dump stale, or an asset directory missing?"
        )
    if differing != 0:
        raise Error(String(differing) + " bytes differ; the first is " + first_bad)

    print("[PASS] io/png on assets")
