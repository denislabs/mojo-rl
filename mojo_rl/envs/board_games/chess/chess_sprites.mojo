"""Chess piece pixel art sprites (24x24 per piece, 12 pieces).

Sprite sheet layout (288x24 RGBA):
  [wK][wQ][wR][wB][wN][wP][bK][bQ][bR][bB][bN][bP]
  Each piece occupies 24x24 pixels at offset piece_index * 24.

White pieces: RGB(255,255,255) with 1px dark outline RGB(60,60,60).
Black pieces: RGB(40,40,40) with 1px light outline RGB(200,200,200).
Background: transparent (alpha=0).
"""

from std.memory import Pointer, unsafe_memset, alloc


# Sprite sheet dimensions.
comptime PIECE_SIZE: Int = 24
comptime NUM_PIECES: Int = 12
comptime SHEET_WIDTH: Int = PIECE_SIZE * NUM_PIECES  # 288
comptime SHEET_HEIGHT: Int = PIECE_SIZE  # 24
comptime BYTES_PER_PIXEL: Int = 4  # RGBA
comptime SHEET_BYTES: Int = SHEET_WIDTH * SHEET_HEIGHT * BYTES_PER_PIXEL


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------


def _set_pixel(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    x: Int,
    y: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
    a: UInt8,
):
    """Set a single pixel in the sprite sheet."""
    var offset = (y * SHEET_WIDTH + x) * BYTES_PER_PIXEL
    pixels[unsafe_offset=offset] = r
    pixels[unsafe_offset=offset + 1] = g
    pixels[unsafe_offset=offset + 2] = b
    pixels[unsafe_offset=offset + 3] = a


def _fill_rect(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    x0: Int,
    y0: Int,
    x1: Int,
    y1: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
    a: UInt8,
):
    """Fill a rectangle [x0,x1) x [y0,y1) in the sprite sheet."""
    for y in range(y0, y1):
        for x in range(x0, x1):
            _set_pixel(pixels, x, y, r, g, b, a)


def _is_filled(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    x: Int,
    y: Int,
) -> Bool:
    """Check if a pixel has non-zero alpha (i.e. was drawn)."""
    var offset = (y * SHEET_WIDTH + x) * BYTES_PER_PIXEL + 3
    return pixels[unsafe_offset=offset] > 0


def _add_outline(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    piece_idx: Int,
    or_: UInt8,
    og: UInt8,
    ob: UInt8,
):
    """Add a 1px outline around all filled pixels in the given piece slot.

    Scans 4-connected neighbours; if a transparent pixel borders a filled one
    it gets the outline colour.
    """
    var ox = piece_idx * PIECE_SIZE
    # Collect outline coordinates first to avoid modifying while scanning.
    var xs = List[Int]()
    var ys = List[Int]()
    for y in range(PIECE_SIZE):
        for x in range(PIECE_SIZE):
            var sx = ox + x
            if _is_filled(pixels, sx, y):
                continue
            # Check 4 neighbours.
            var has_neighbour = False
            if x > 0 and _is_filled(pixels, sx - 1, y):
                has_neighbour = True
            if x < PIECE_SIZE - 1 and _is_filled(pixels, sx + 1, y):
                has_neighbour = True
            if y > 0 and _is_filled(pixels, sx, y - 1):
                has_neighbour = True
            if y < PIECE_SIZE - 1 and _is_filled(pixels, sx, y + 1):
                has_neighbour = True
            if has_neighbour:
                xs.append(sx)
                ys.append(y)
    for i in range(len(xs)):
        _set_pixel(pixels, xs[i], ys[i], or_, og, ob, 255)


# ---------------------------------------------------------------------------
# Individual piece drawing routines
# ---------------------------------------------------------------------------


def _draw_king(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    ox: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
):
    """King: cross on top, crown head, narrow neck, wide base."""
    # Cross vertical bar: rows 1-5, cols 11-12
    _fill_rect(pixels, ox + 11, 1, ox + 13, 6, r, g, b, 255)
    # Cross horizontal bar: rows 2-3, cols 9-14
    _fill_rect(pixels, ox + 9, 2, ox + 15, 4, r, g, b, 255)

    # Crown / head: rows 6-13
    # Row 6-7: three crown points
    _fill_rect(pixels, ox + 5, 6, ox + 8, 8, r, g, b, 255)  # left point
    _fill_rect(pixels, ox + 10, 6, ox + 14, 8, r, g, b, 255)  # center point
    _fill_rect(pixels, ox + 16, 6, ox + 19, 8, r, g, b, 255)  # right point
    # Rows 8-13: solid crown body
    _fill_rect(pixels, ox + 5, 8, ox + 19, 14, r, g, b, 255)

    # Neck: rows 14-19, cols 8-15
    _fill_rect(pixels, ox + 8, 14, ox + 16, 20, r, g, b, 255)

    # Base: rows 20-23, cols 5-18
    _fill_rect(pixels, ox + 5, 20, ox + 19, 24, r, g, b, 255)


def _draw_queen(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    ox: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
):
    """Queen: five prongs on top, crown body, neck, base."""
    # Five prongs: rows 3-6, small 2px wide blocks
    _fill_rect(pixels, ox + 4, 3, ox + 6, 7, r, g, b, 255)  # prong 1
    _fill_rect(pixels, ox + 8, 3, ox + 10, 7, r, g, b, 255)  # prong 2
    _fill_rect(
        pixels, ox + 11, 2, ox + 13, 7, r, g, b, 255
    )  # prong 3 (center, taller)
    _fill_rect(pixels, ox + 14, 3, ox + 16, 7, r, g, b, 255)  # prong 4
    _fill_rect(pixels, ox + 18, 3, ox + 20, 7, r, g, b, 255)  # prong 5

    # Crown body: rows 7-13
    _fill_rect(pixels, ox + 4, 7, ox + 20, 14, r, g, b, 255)

    # Neck: rows 14-19, cols 8-15
    _fill_rect(pixels, ox + 8, 14, ox + 16, 20, r, g, b, 255)

    # Base: rows 20-23, cols 5-18
    _fill_rect(pixels, ox + 5, 20, ox + 19, 24, r, g, b, 255)


def _draw_rook(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    ox: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
):
    """Rook: crenellated top, tower body, base."""
    # Crenellations: three blocks at rows 3-7 with gaps
    _fill_rect(pixels, ox + 6, 3, ox + 9, 8, r, g, b, 255)  # left merlon
    _fill_rect(pixels, ox + 10, 3, ox + 14, 8, r, g, b, 255)  # center merlon
    _fill_rect(pixels, ox + 15, 3, ox + 18, 8, r, g, b, 255)  # right merlon

    # Tower body: rows 8-19
    _fill_rect(pixels, ox + 6, 8, ox + 18, 20, r, g, b, 255)

    # Base: rows 20-23, cols 4-19 (wider than tower)
    _fill_rect(pixels, ox + 4, 20, ox + 20, 24, r, g, b, 255)


def _draw_bishop(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    ox: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
):
    """Bishop: small nub on top, rounded mitre head, body, base."""
    # Top nub: rows 2-3
    _fill_rect(pixels, ox + 11, 2, ox + 13, 4, r, g, b, 255)

    # Mitre head (rounded by stepping):
    # Row 4: cols 10-13
    _fill_rect(pixels, ox + 10, 4, ox + 14, 5, r, g, b, 255)
    # Row 5: cols 9-14
    _fill_rect(pixels, ox + 9, 5, ox + 15, 6, r, g, b, 255)
    # Rows 6-7: cols 8-15
    _fill_rect(pixels, ox + 8, 6, ox + 16, 8, r, g, b, 255)
    # Rows 8-9: cols 9-14 (narrowing)
    _fill_rect(pixels, ox + 9, 8, ox + 15, 10, r, g, b, 255)

    # Body: rows 10-19, cols 9-14
    _fill_rect(pixels, ox + 9, 10, ox + 15, 20, r, g, b, 255)

    # Collar / flare at rows 17-19
    _fill_rect(pixels, ox + 8, 17, ox + 16, 20, r, g, b, 255)

    # Base: rows 20-23, cols 6-17
    _fill_rect(pixels, ox + 6, 20, ox + 18, 24, r, g, b, 255)


def _draw_knight(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    ox: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
):
    """Knight: horse head profile facing left."""
    # Ear: rows 2-4, cols 8-10
    _fill_rect(pixels, ox + 8, 2, ox + 11, 5, r, g, b, 255)

    # Top of head: rows 4-5, cols 7-14
    _fill_rect(pixels, ox + 7, 4, ox + 15, 6, r, g, b, 255)

    # Upper head: rows 6-7, cols 5-15
    _fill_rect(pixels, ox + 5, 6, ox + 16, 8, r, g, b, 255)

    # Snout: rows 8-9, cols 4-16 (wide, horse face)
    _fill_rect(pixels, ox + 4, 8, ox + 16, 10, r, g, b, 255)

    # Lower snout / jaw: rows 10-11, cols 4-13 (snout narrower, facing left)
    _fill_rect(pixels, ox + 4, 10, ox + 14, 12, r, g, b, 255)

    # Neck: rows 12-15, cols 9-16
    _fill_rect(pixels, ox + 9, 12, ox + 17, 16, r, g, b, 255)

    # Lower neck: rows 16-19, cols 8-17
    _fill_rect(pixels, ox + 8, 16, ox + 17, 20, r, g, b, 255)

    # Base: rows 20-23, cols 5-18
    _fill_rect(pixels, ox + 5, 20, ox + 19, 24, r, g, b, 255)


def _draw_pawn(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    ox: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
):
    """Pawn: round head, tapered body, wide base."""
    # Head (circular-ish):
    # Row 5: cols 10-13
    _fill_rect(pixels, ox + 10, 5, ox + 14, 6, r, g, b, 255)
    # Rows 6-7: cols 9-14
    _fill_rect(pixels, ox + 9, 6, ox + 15, 8, r, g, b, 255)
    # Rows 8-9: cols 8-15
    _fill_rect(pixels, ox + 8, 8, ox + 16, 10, r, g, b, 255)
    # Rows 10-11: cols 9-14
    _fill_rect(pixels, ox + 9, 10, ox + 15, 12, r, g, b, 255)
    # Row 12: cols 10-13
    _fill_rect(pixels, ox + 10, 12, ox + 14, 13, r, g, b, 255)

    # Body (tapered): rows 13-19, cols 9-14
    _fill_rect(pixels, ox + 10, 13, ox + 14, 15, r, g, b, 255)
    _fill_rect(pixels, ox + 9, 15, ox + 15, 18, r, g, b, 255)
    _fill_rect(pixels, ox + 8, 18, ox + 16, 20, r, g, b, 255)

    # Base: rows 20-23, cols 6-17
    _fill_rect(pixels, ox + 6, 20, ox + 18, 24, r, g, b, 255)


# ---------------------------------------------------------------------------
# Piece dispatch
# ---------------------------------------------------------------------------


def _draw_piece(
    mut pixels: Pointer[UInt8, MutAnyOrigin],
    piece_idx: Int,
    r: UInt8,
    g: UInt8,
    b: UInt8,
):
    """Draw piece `piece_idx` (0-11) into the sprite sheet.

    piece_idx % 6 selects: 0=King, 1=Queen, 2=Rook, 3=Bishop, 4=Knight, 5=Pawn.
    """
    var ox = piece_idx * PIECE_SIZE
    var piece_type = piece_idx % 6

    if piece_type == 0:
        _draw_king(pixels, ox, r, g, b)
    elif piece_type == 1:
        _draw_queen(pixels, ox, r, g, b)
    elif piece_type == 2:
        _draw_rook(pixels, ox, r, g, b)
    elif piece_type == 3:
        _draw_bishop(pixels, ox, r, g, b)
    elif piece_type == 4:
        _draw_knight(pixels, ox, r, g, b)
    else:
        _draw_pawn(pixels, ox, r, g, b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_sprite_sheet() -> Pointer[UInt8, MutAnyOrigin]:
    """Create and return a sprite sheet with all 12 chess piece sprites.

    Returns a heap-allocated RGBA pixel buffer (288x24, 27648 bytes).
    Caller must free the returned pointer with ``Pointer.free()``.

    Piece order: wK, wQ, wR, wB, wN, wP, bK, bQ, bR, bB, bN, bP.
    """
    var raw_pixels = alloc[UInt8](SHEET_BYTES)
    unsafe_memset(raw_pixels, 0, SHEET_BYTES)  # fully transparent
    var pixels = rebind[Pointer[UInt8, MutAnyOrigin]](raw_pixels)

    # --- White pieces (indices 0-5) ---
    # Draw fill colour.
    for i in range(6):
        _draw_piece(pixels, i, 255, 255, 255)
    # Add dark outline so white pieces are visible on light backgrounds.
    for i in range(6):
        _add_outline(pixels, i, 60, 60, 60)

    # --- Black pieces (indices 6-11) ---
    for i in range(6, 12):
        _draw_piece(pixels, i, 40, 40, 40)
    # Add light outline so black pieces are visible on dark backgrounds.
    for i in range(6, 12):
        _add_outline(pixels, i, 200, 200, 200)

    return pixels
