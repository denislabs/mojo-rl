"""Generate a printable ArUco marker at an EXACT physical size.

    pixi run python tools/vision/make_printable_marker.py
    pixi run python tools/vision/make_printable_marker.py --id 7 --size-mm 60
    pixi run python tools/vision/make_printable_marker.py --sheet 0,1,2,3

Writes a PNG and a PDF into `scratch/markers/` (gitignored). Print the **PDF**.

⚠⚠ **THE SIZE IS THE BLACK SQUARE, NOT THE PAPER AND NOT THE WHITE BORDER.**
`--size-mm` is the side of the marker's outer BLACK edge, which is what
`solvePnP`'s object points describe. Measuring the white quiet zone instead and
typing that into the studio makes every distance wrong by the same ratio — and
it will look plausible, because a pose from four coplanar points is always
"reasonable", just scaled.

⚠⚠ **PRINT AT 100%, NOT "FIT TO PAGE".** Scale-to-fit is the default in some
dialogs and it silently shrinks the marker by a few percent. Depth scales
linearly with the marker size, so 5% of print scaling is 5% of every distance.
The sheet carries a 100 mm ruler for exactly this reason: after printing,
measure it. If it is not 100 mm, measure the marker itself and type THAT into
the studio's `marker mm` slider — a mismeasured print is recoverable, an
unmeasured one is not.

⚠ MOUNT IT FLAT. A marker taped to a curved or floppy surface is not the planar
target the maths assumes; the corner estimate degrades and the pose wanders.
Card, foam board or a clipboard is enough.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scratch" / "markers"

DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}

DPI = 600
MM_PER_IN = 25.4


def mm_to_px(mm: float) -> int:
    return int(round(mm / MM_PER_IN * DPI))


def draw_text(img, text, x_mm, y_mm, scale=1.0, thickness=2):
    cv2.putText(img, text, (mm_to_px(x_mm), mm_to_px(y_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * DPI / 300.0, 0,
                thickness * 2, cv2.LINE_AA)


def draw_ruler(img, x_mm, y_mm, length_mm=100.0):
    """A ruler on the printed page — the only way to catch print scaling.

    A dialog that silently applied "fit to page" produces a marker that is
    detected perfectly and ranged wrongly, forever, by a constant factor.
    """
    x0, y0 = mm_to_px(x_mm), mm_to_px(y_mm)
    x1 = mm_to_px(x_mm + length_mm)
    cv2.line(img, (x0, y0), (x1, y0), 0, max(2, DPI // 300))
    for t in range(0, int(length_mm) + 1, 10):
        tx = mm_to_px(x_mm + t)
        h = 4.0 if t % 50 == 0 else 2.5
        cv2.line(img, (tx, y0), (tx, y0 - mm_to_px(h)), 0, max(2, DPI // 300))
        if t % 50 == 0:
            draw_text(img, f"{t}", x_mm + t - 2, y_mm + 6, 0.5, 1)
    draw_text(img, "measure me: this line is 100 mm", x_mm, y_mm + 14, 0.5, 1)


def build_page(dict_name: str, ids, size_mm: float, quiet_frac: float = 0.25):
    dictionary = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])
    quiet_mm = size_mm * quiet_frac

    # A4 portrait, which is the safe default for both A4 and US Letter printers
    # as long as nothing is placed within 15 mm of an edge.
    page_w_mm, page_h_mm = 210.0, 297.0
    page = np.full((mm_to_px(page_h_mm), mm_to_px(page_w_mm)), 255, np.uint8)

    tile_mm = size_mm + 2 * quiet_mm
    margin_mm = 20.0
    per_row = max(1, int((page_w_mm - 2 * margin_mm) // (tile_mm + 5.0)))

    y_mm = margin_mm + 10.0
    draw_text(page, f"ArUco {dict_name}   marker side = {size_mm:g} mm",
              margin_mm, y_mm, 0.7, 2)
    y_mm += 8.0
    draw_text(page, "PRINT AT 100% (not 'fit to page'), then measure the ruler",
              margin_mm, y_mm, 0.5, 1)
    y_mm += 10.0

    for k, mid in enumerate(ids):
        col = k % per_row
        row = k // per_row
        tx = margin_mm + col * (tile_mm + 5.0)
        ty = y_mm + row * (tile_mm + 14.0)

        side_px = mm_to_px(size_mm)
        tag = cv2.aruco.generateImageMarker(dictionary, mid, side_px)
        x0 = mm_to_px(tx + quiet_mm)
        y0 = mm_to_px(ty + quiet_mm)
        page[y0:y0 + side_px, x0:x0 + side_px] = tag

        # ⚠ THE CROP MARKS SIT ON THE BLACK SQUARE'S EDGE, not the tile's, so
        # the thing to measure is unambiguous.
        for (mx, my) in ((x0, y0), (x0 + side_px, y0), (x0, y0 + side_px),
                         (x0 + side_px, y0 + side_px)):
            cv2.line(page, (mx - mm_to_px(3), my), (mx - mm_to_px(1), my), 0, 2)
            cv2.line(page, (mx, my - mm_to_px(3)), (mx, my - mm_to_px(1)), 0, 2)
        draw_text(page, f"id {mid}  -  side {size_mm:g} mm (black square only)",
                  tx, ty + tile_mm + 6.0, 0.5, 1)

    rows = (len(ids) + per_row - 1) // per_row
    draw_ruler(page, margin_mm, y_mm + rows * (tile_mm + 14.0) + 15.0)
    return page


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="4x4_50", choices=sorted(DICTS))
    ap.add_argument("--id", type=int, default=7)
    ap.add_argument("--sheet", default="", help="comma-separated ids")
    ap.add_argument("--size-mm", type=float, default=40.0,
                    help="side of the BLACK square, millimetres")
    args = ap.parse_args()

    ids = ([int(v) for v in args.sheet.split(",")] if args.sheet
           else [args.id])
    page = build_page(args.dict, ids, args.size_mm)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"aruco_{args.dict}_{'-'.join(str(i) for i in ids)}_{args.size_mm:g}mm"
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    cv2.imwrite(str(png), page)

    # ⚠ THE PDF IS THE ONE TO PRINT. A PNG carries its DPI only as a hint that
    # print dialogs routinely ignore; PIL writes the PDF with the page geometry
    # baked in, so "100%" means the millimetres above.
    Image.open(png).save(pdf, resolution=DPI)

    # Prove the sheet is detectable before anyone spends paper on it.
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTS[args.dict]))
    small = cv2.resize(page, (page.shape[1] // 8, page.shape[0] // 8),
                       interpolation=cv2.INTER_AREA)
    _, found, _ = det.detectMarkers(small)
    n = 0 if found is None else len(found)
    if n != len(ids):
        print(f"⚠ the sheet detects {n} of {len(ids)} markers at 1/8 scale",
              file=sys.stderr)
        return 1

    print(f"wrote {pdf}")
    print(f"      {png}")
    print(f"  dict {args.dict}, ids {ids}, black square {args.size_mm:g} mm")
    print(f"  detected {n}/{len(ids)} in a 1/8-scale self-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
