"""Generate a printable ArUco marker or ChArUco board at an EXACT physical size.

    pixi run python tools/vision/make_printable_marker.py
    pixi run python tools/vision/make_printable_marker.py --id 7 --size-mm 60
    pixi run python tools/vision/make_printable_marker.py --sheet 0,1,2,3
    pixi run python tools/vision/make_printable_marker.py --charuco

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

⚠ MULTI-PAGE IS AUTOMATIC. Whatever does not fit spills onto another page of
the same PDF. An earlier version packed everything into one page with no bounds
check and crashed on three 60 mm markers; the crash was the GOOD outcome, since
the other way a missing bounds check fails is a marker silently clipped at the
page edge — which still prints, still detects, and is no longer square.

⚠⚠ **THE CHARUCO BOARD'S NUMBERS ARE A CONTRACT, NOT A DESCRIPTION.** A board
is defined by (squares_x, squares_y, square_mm, marker_mm, dict), and the
detector must be built with the SAME five. Give it different ones and it does
not fail — it finds a different set of corners at different board coordinates
and calibrates to a confidently wrong camera. They are printed on the sheet for
that reason, and the studio's calibration panel defaults to them.

⚠ WHY A CHARUCO BOARD RATHER THAN A CHESSBOARD. Its corners are identified by
the ArUco markers in the white squares, so a PARTIALLY VISIBLE board still
contributes — which is what lets you fill the frame corners, where lens
distortion actually lives. A plain chessboard must be wholly visible in every
view, so the views that matter most are the ones it rejects.

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


def build_charuco(dict_name: str, squares_x: int, squares_y: int,
                  square_mm: float, marker_mm: float):
    """One A4 page carrying a ChArUco board at an exact physical size.

    ⚠ THE MARGIN IS NOT COSMETIC. `generateImage`'s board fills its output
    edge to edge, and the outer ArUco markers then sit against the paper's
    white with no defined quiet zone of their own. The margin below is what
    guarantees one.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm / 1000.0, marker_mm / 1000.0,
        dictionary)

    board_w_mm = squares_x * square_mm
    board_h_mm = squares_y * square_mm

    page_w_mm, page_h_mm = 210.0, 297.0
    margin_mm = 12.0
    header_mm = 26.0
    ruler_mm = 26.0
    avail_h = page_h_mm - 2 * margin_mm - header_mm - ruler_mm
    avail_w = page_w_mm - 2 * margin_mm
    if board_w_mm > avail_w or board_h_mm > avail_h:
        raise SystemExit(
            f"a {squares_x}x{squares_y} board of {square_mm:g} mm squares is "
            f"{board_w_mm:.0f}x{board_h_mm:.0f} mm and only "
            f"{avail_w:.0f}x{avail_h:.0f} mm is usable on A4 — reduce "
            f"--square-mm or the square count"
        )

    page = np.full((mm_to_px(page_h_mm), mm_to_px(page_w_mm)), 255, np.uint8)
    img = board.generateImage((mm_to_px(board_w_mm), mm_to_px(board_h_mm)))

    y_mm = margin_mm + 8.0
    draw_text(page, f"ChArUco  {squares_x}x{squares_y} squares  "
                    f"{square_mm:g} mm square  {marker_mm:g} mm marker",
              margin_mm, y_mm, 0.6, 2)
    y_mm += 7.0
    draw_text(page, f"dict {dict_name}   -   PRINT AT 100%, then measure the "
                    "ruler below", margin_mm, y_mm, 0.45, 1)
    y_mm += 9.0

    x0 = mm_to_px((page_w_mm - board_w_mm) / 2.0)
    y0 = mm_to_px(y_mm)
    page[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img

    # Crop marks on the board's own outer edge — the square grid's boundary is
    # what `square_mm * squares_x` measures, and it is easy to mistake for the
    # printed margin.
    for (mx, my) in ((x0, y0), (x0 + img.shape[1], y0),
                     (x0, y0 + img.shape[0]),
                     (x0 + img.shape[1], y0 + img.shape[0])):
        cv2.line(page, (mx - mm_to_px(4), my), (mx - mm_to_px(1.5), my), 0, 3)
        cv2.line(page, (mx, my - mm_to_px(4)), (mx, my - mm_to_px(1.5)), 0, 3)

    draw_ruler(page, margin_mm + 4.0, y_mm + board_h_mm + 16.0)
    return page, board


def build_pages(dict_name: str, ids, size_mm: float):
    """Lay the markers out over as many A4 pages as they need.

    ⚠ THIS PAGINATES BECAUSE THE FIRST VERSION DID NOT, AND CRASHED. It packed
    into one page with no bounds check, so three 60 mm markers (a 90 mm tile
    each, one per row on a 210 mm page) ran off the bottom and numpy refused
    the assignment. A layout without a fit check has exactly two failure modes
    and the crash is the GOOD one — the other is a marker silently clipped at
    the page edge, which still prints, still detects, and is no longer square.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])

    # ⚠ THE QUIET ZONE IS ONE MODULE, WHICH IS WHAT ARUCO ASKS FOR — not a
    # percentage. A marker is `markerSize + 2` modules across (the data grid
    # plus a one-module black border), and the detector finds candidates by
    # contour, so it needs about one module of white outside that border.
    # An earlier arbitrary 25% was both unprincipled and expensive: at 60 mm it
    # made a 90 mm tile, which fits ONE marker per A4 page instead of two.
    modules = dictionary.markerSize + 2
    quiet_mm = size_mm / modules

    # A4 portrait: safe on both A4 and US Letter printers as long as nothing
    # sits within ~15 mm of an edge.
    page_w_mm, page_h_mm = 210.0, 297.0
    margin_mm = 20.0
    gap_mm = 5.0
    caption_mm = 14.0
    header_mm = 28.0
    ruler_mm = 30.0

    tile_mm = size_mm + 2 * quiet_mm
    usable_w = page_w_mm - 2 * margin_mm
    per_row = max(1, int((usable_w + gap_mm) // (tile_mm + gap_mm)))

    row_mm = tile_mm + caption_mm
    usable_h = page_h_mm - margin_mm - header_mm - ruler_mm - margin_mm
    rows_per_page = int(usable_h // row_mm)
    if rows_per_page < 1:
        raise SystemExit(
            f"a {size_mm:g} mm marker needs {row_mm:.0f} mm of page height and "
            f"only {usable_h:.0f} mm is usable — use --size-mm {int(usable_h - caption_mm) // 2} or less"
        )
    per_page = per_row * rows_per_page

    pages = []
    for start in range(0, len(ids), per_page):
        chunk = ids[start:start + per_page]
        page = np.full((mm_to_px(page_h_mm), mm_to_px(page_w_mm)), 255, np.uint8)

        y_mm = margin_mm + 10.0
        n_pages = (len(ids) + per_page - 1) // per_page
        label = f"ArUco {dict_name}   marker side = {size_mm:g} mm"
        if n_pages > 1:
            label += f"   (page {start // per_page + 1} of {n_pages})"
        draw_text(page, label, margin_mm, y_mm, 0.7, 2)
        y_mm += 8.0
        draw_text(page, "PRINT AT 100% (not 'fit to page'), then measure the ruler",
                  margin_mm, y_mm, 0.5, 1)
        y_mm += 10.0

        for k, mid in enumerate(chunk):
            col = k % per_row
            row = k // per_row
            tx = margin_mm + col * (tile_mm + gap_mm)
            ty = y_mm + row * row_mm

            side_px = mm_to_px(size_mm)
            tag = cv2.aruco.generateImageMarker(dictionary, mid, side_px)
            x0 = mm_to_px(tx + quiet_mm)
            y0 = mm_to_px(ty + quiet_mm)
            # A belt-and-braces assert: the arithmetic above should make this
            # unreachable, and it is what turns a future layout change from a
            # numpy broadcast error into a sentence.
            if (y0 + side_px > page.shape[0]) or (x0 + side_px > page.shape[1]):
                raise SystemExit(
                    f"internal: marker {mid} does not fit the page "
                    f"({size_mm:g} mm, {per_row}x{rows_per_page} layout)"
                )
            page[y0:y0 + side_px, x0:x0 + side_px] = tag

            # ⚠ THE CROP MARKS SIT ON THE BLACK SQUARE'S EDGE, not the tile's,
            # so the thing to measure is unambiguous.
            for (mx, my) in ((x0, y0), (x0 + side_px, y0), (x0, y0 + side_px),
                             (x0 + side_px, y0 + side_px)):
                cv2.line(page, (mx - mm_to_px(3), my), (mx - mm_to_px(1), my), 0, 2)
                cv2.line(page, (mx, my - mm_to_px(3)), (mx, my - mm_to_px(1)), 0, 2)
            draw_text(page, f"id {mid}  -  side {size_mm:g} mm (black square only)",
                      tx, ty + tile_mm + 6.0, 0.5, 1)

        rows_used = (len(chunk) + per_row - 1) // per_row
        draw_ruler(page, margin_mm, y_mm + rows_used * row_mm + 12.0)
        pages.append(page)
    return pages


def main_charuco(args) -> int:
    sx, sy = (int(v) for v in args.squares.lower().split("x"))
    if args.marker_mm >= args.square_mm:
        print("--marker-mm must be smaller than --square-mm (the marker sits "
              "INSIDE a white square)", file=sys.stderr)
        return 1
    page, board = build_charuco(args.dict, sx, sy, args.square_mm,
                                args.marker_mm)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = (f"charuco_{args.dict}_{sx}x{sy}_{args.square_mm:g}mm"
            f"_{args.marker_mm:g}mm")
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    cv2.imwrite(str(png), page)
    Image.open(png).save(pdf, resolution=DPI)

    # ⚠ SELF-CHECK AT 1/8 SCALE, and against the FULL corner count. A board
    # that yields only some of its corners on a clean render will yield fewer
    # still through a lens, and calibrating from a handful of corners is how a
    # confidently wrong camera matrix happens.
    det = cv2.aruco.CharucoDetector(board)
    small = cv2.resize(page, (page.shape[1] // 8, page.shape[0] // 8),
                       interpolation=cv2.INTER_AREA)
    corners, ids, _, _ = det.detectBoard(small)
    n = 0 if ids is None else len(ids)
    expect = (sx - 1) * (sy - 1)
    if n != expect:
        print(f"⚠ the board yields {n} of {expect} corners at 1/8 scale",
              file=sys.stderr)
        return 1

    print(f"wrote {pdf}")
    print(f"      {png}")
    print(f"  board {sx}x{sy} squares, {args.square_mm:g} mm square, "
          f"{args.marker_mm:g} mm marker, dict {args.dict}")
    print(f"  outer size {sx * args.square_mm:g} x {sy * args.square_mm:g} mm, "
          f"{expect} chessboard corners")
    print(f"  detected {n}/{expect} corners in a 1/8-scale self-check")
    print("  ⚠ type these five numbers into whatever calibrates from it —")
    print("    a different set does not fail, it calibrates a different camera")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="4x4_50", choices=sorted(DICTS))
    ap.add_argument("--id", type=int, default=7)
    ap.add_argument("--sheet", default="", help="comma-separated ids")
    ap.add_argument("--size-mm", type=float, default=40.0,
                    help="side of the BLACK square, millimetres")
    ap.add_argument("--charuco", action="store_true",
                    help="print a ChArUco calibration board instead")
    ap.add_argument("--squares", default="5x7",
                    help="ChArUco grid, e.g. 5x7")
    ap.add_argument("--square-mm", type=float, default=30.0)
    ap.add_argument("--marker-mm", type=float, default=22.0)
    args = ap.parse_args()

    if args.charuco:
        return main_charuco(args)

    ids = ([int(v) for v in args.sheet.split(",")] if args.sheet
           else [args.id])
    pages = build_pages(args.dict, ids, args.size_mm)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"aruco_{args.dict}_{'-'.join(str(i) for i in ids)}_{args.size_mm:g}mm"
    pngs = []
    for i, page in enumerate(pages):
        png = OUT_DIR / (f"{stem}.png" if len(pages) == 1
                         else f"{stem}_p{i + 1}.png")
        cv2.imwrite(str(png), page)
        pngs.append(png)

    # ⚠ THE PDF IS THE ONE TO PRINT. A PNG carries its DPI only as a hint that
    # print dialogs routinely ignore; PIL writes the PDF with the page geometry
    # baked in, so "100%" means the millimetres above. Multiple pages go into
    # ONE pdf so a sheet cannot be half-printed by accident.
    pdf = OUT_DIR / f"{stem}.pdf"
    imgs = [Image.open(p) for p in pngs]
    imgs[0].save(pdf, resolution=DPI, save_all=True, append_images=imgs[1:])

    # Prove every page is detectable before anyone spends paper on it.
    # ⚠ AT 1/8 SCALE, which is the point: a sheet that only reads at full
    # resolution is a sheet that will not read from across a room either.
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTS[args.dict]))
    found_total = 0
    for page in pages:
        small = cv2.resize(page, (page.shape[1] // 8, page.shape[0] // 8),
                           interpolation=cv2.INTER_AREA)
        _, found, _ = det.detectMarkers(small)
        found_total += 0 if found is None else len(found)
    if found_total != len(ids):
        print(f"⚠ the sheet detects {found_total} of {len(ids)} markers "
              "at 1/8 scale", file=sys.stderr)
        return 1

    print(f"wrote {pdf}  ({len(pages)} page(s))")
    for p in pngs:
        print(f"      {p}")
    print(f"  dict {args.dict}, ids {ids}, black square {args.size_mm:g} mm")
    print(f"  quiet zone {args.size_mm / (cv2.aruco.getPredefinedDictionary(DICTS[args.dict]).markerSize + 2):.1f} mm (one module)")
    print(f"  detected {found_total}/{len(ids)} in a 1/8-scale self-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
