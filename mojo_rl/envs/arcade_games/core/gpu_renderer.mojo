"""GPU inline shape renderer — draws to 160×210 grayscale framebuffer.

These functions are @always_inline so they can be called from GPU kernels.
All work on a flat UInt8 buffer of size 160*210 = 33600 bytes.
"""

from .colors import SCREEN_W, SCREEN_H


@always_inline
fn clear_frame(buf: UnsafePointer[UInt8, MutAnyOrigin]):
    """Fill entire framebuffer with black (0)."""
    for i in range(SCREEN_W * SCREEN_H):
        buf[i] = 0


@always_inline
fn draw_filled_rect(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    x: Int,
    y: Int,
    w: Int,
    h: Int,
    color: UInt8,
):
    """Draw a filled rectangle into the 160×210 grayscale buffer.

    Coordinates are clipped to screen bounds.
    """
    var y0 = max(0, y)
    var y1 = min(SCREEN_H, y + h)
    var x0 = max(0, x)
    var x1 = min(SCREEN_W, x + w)

    for row in range(y0, y1):
        for col in range(x0, x1):
            buf[row * SCREEN_W + col] = color


@always_inline
fn draw_hline(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    x: Int,
    y: Int,
    w: Int,
    color: UInt8,
):
    """Draw a horizontal line (1 pixel tall)."""
    if y < 0 or y >= SCREEN_H:
        return
    var x0 = max(0, x)
    var x1 = min(SCREEN_W, x + w)
    for col in range(x0, x1):
        buf[y * SCREEN_W + col] = color


@always_inline
fn draw_dashed_vline(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    x: Int,
    y_start: Int,
    y_end: Int,
    dash_len: Int,
    gap_len: Int,
    color: UInt8,
):
    """Draw a dashed vertical line."""
    if x < 0 or x >= SCREEN_W:
        return
    var y = max(0, y_start)
    var end = min(SCREEN_H, y_end)
    var counter = 0
    while y < end:
        if counter < dash_len:
            buf[y * SCREEN_W + x] = color
        counter += 1
        if counter >= dash_len + gap_len:
            counter = 0
        y += 1


@always_inline
fn draw_digit(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    x: Int,
    y: Int,
    digit: Int,
    color: UInt8,
    scale: Int = 2,
):
    """Draw a single digit (0-9) using a simple 3×5 bitmap font.

    Scale multiplies the pixel size (scale=2 → 6×10 pixels).
    """
    # 3×5 font bitmaps packed as 15-bit integers (MSB = top-left)
    # Row 0: top 3 bits, Row 1: next 3, etc.
    comptime _fonts: InlineArray[UInt16, 10] = [
        0b111_101_101_101_111,  # 0
        0b010_110_010_010_111,  # 1
        0b111_001_111_100_111,  # 2
        0b111_001_111_001_111,  # 3
        0b101_101_111_001_001,  # 4
        0b111_100_111_001_111,  # 5
        0b111_100_111_101_111,  # 6
        0b111_001_001_001_001,  # 7
        0b111_101_111_101_111,  # 8
        0b111_101_111_001_111,  # 9
    ]
    var fonts = materialize[_fonts]()

    var d = digit % 10
    var bitmap = Int(fonts[d])

    for row in range(5):
        for col in range(3):
            var bit_idx = (4 - row) * 3 + (2 - col)
            if (bitmap >> bit_idx) & 1:
                # Draw scaled pixel
                for sy in range(scale):
                    for sx in range(scale):
                        var px = x + col * scale + sx
                        var py = y + row * scale + sy
                        if 0 <= px < SCREEN_W and 0 <= py < SCREEN_H:
                            buf[py * SCREEN_W + px] = color


@always_inline
fn draw_number(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    x: Int,
    y: Int,
    number: Int,
    color: UInt8,
    scale: Int = 2,
):
    """Draw a multi-digit number. Right-aligned at x position."""
    var n = number
    if n < 0:
        n = 0
    if n == 0:
        draw_digit(buf, x, y, 0, color, scale)
        return

    # Count digits
    var digits = 0
    var tmp = n
    while tmp > 0:
        digits += 1
        tmp //= 10

    # Draw from right to left
    var cx = x + (digits - 1) * (3 * scale + scale)
    while n > 0:
        draw_digit(buf, cx, y, n % 10, color, scale)
        n //= 10
        cx -= 3 * scale + scale
