# +--------------------------------------------------------------------------+ #
# | The PNG encoder, read back by Pillow
# +--------------------------------------------------------------------------+ #
"""Gate `encode_png` / `save_png` in `mojo_rl/io/png.mojo`.

    pixi run build-http                                   # ONCE
    pixi run mojo run -I . tests/io/test_png_write.mojo

Writes one image per channel count (grey, grey+alpha, RGB, RGBA), reads each
back with this repo's own decoder, and then hands the whole directory to
`tools/io/verify_png_write.py`, which opens them in **Pillow**.

⚠ THE ROUND TRIP ALONE PROVES NOTHING. Encoder and decoder written together
share their misunderstandings: a CRC over the wrong span, a colour type
written for the wrong channel count, a `IDAT` that is raw deflate rather than
zlib — each of those round-trips perfectly here and is unreadable everywhere
else. Pillow is the third opinion that makes the file real.

Odd dimensions on purpose (23x17): a width that divides nothing catches an
encoder that assumed rows were padded.
"""

from std.os import makedirs
from std.os.path import exists

from mojo_rl.io.fileio import write_file_atomic
from mojo_rl.io.png import load_png_file, save_png
from mojo_rl.io.proc import run_capture


comptime DIR = "/tmp/mojo_rl_png_write"
comptime W = 23
comptime H = 17


def main() raises:
    print("=== io/png encoder ===")
    makedirs(String(DIR), exist_ok=True)

    var checks = 0
    for ch in [1, 2, 3, 4]:
        var px = List[UInt8]()
        for i in range(W * H * ch):
            px.append(UInt8((i * 37 + ch * 11) & 0xFF))

        var path = String(DIR) + "/w" + String(ch) + ".png"
        save_png(path, px, W, H, ch)
        write_file_atomic(String(DIR) + "/expected_" + String(ch) + ".raw", px)

        # ── our own decoder ──────────────────────────────────────────
        var back = load_png_file(path)
        if back.width != W or back.height != H or back.channels != ch:
            raise Error(
                "channels " + String(ch) + ": read back " + String(back.width)
                + "x" + String(back.height) + "x" + String(back.channels)
            )
        for i in range(len(px)):
            if back.pixels[i] != px[i]:
                raise Error(
                    "channels " + String(ch) + ": byte " + String(i)
                    + " round-tripped as " + String(Int(back.pixels[i]))
                    + ", wrote " + String(Int(px[i]))
                )
        checks += 1
        print("  " + String(ch) + " channel(s): " + String(len(px))
              + " bytes round-tripped")

    if checks != 4:
        raise Error("vacuous: only " + String(checks) + " channel counts ran")

    # ── the third opinion ────────────────────────────────────────────
    var out = run_capture(
        "python3 tools/io/verify_png_write.py " + String(DIR) + " 2>&1"
    )
    if "PNG-WRITE-OK" not in out:
        raise Error("Pillow rejected what we wrote:\n" + out)
    print("  Pillow: " + String(out.strip()))

    print("[PASS] io/png encoder")
