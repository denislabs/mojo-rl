"""TIA frame buffer rendering — generates 160×210 BGRA pixel output.

The Atari 2600 is a "racing the beam" system: the CPU updates TIA registers
mid-frame, so each scanline can have different graphics state. Rendering must
happen scanline-by-scanline DURING frame execution, not after.

Two entry points:
  - render_scanline_bgra(): Render one scanline during emulation (correct)
  - render_frame_bgra(): Render using current TIA state (static snapshot, for tests)

Color priority (standard mode, PF_PRIORITY=0):
  Player/Missile > Playfield/Ball > Background

Color priority (PF_PRIORITY=1):
  Playfield/Ball > Player/Missile > Background

Score mode (PF_SCORE=1):
  Left half playfield uses COLUP0, right half uses COLUP1.

Ported from CuLE (BSD-3): cule/atari/tia.hpp (rendering logic)
"""

from .atari_state import AtariState
from .palette import palette_r, palette_g, palette_b
from .tables import player_mask, missile_mask, ball_mask, playfield_mask
from .flags import (
    FRAME_WIDTH, FRAME_HEIGHT,
    TIA_PF_PRIORITY, TIA_PF_SCORE, TIA_VBLANK,
)

# Frame buffer size: 160x210 pixels, 4 bytes per pixel (BGRA)
comptime FRAME_BUF_SIZE: Int = FRAME_WIDTH * FRAME_HEIGHT * 4


@always_inline
fn _get_pixel_color(state: AtariState, pixel: Int) -> UInt8:
    """Determine the TIA color index for a single pixel position.

    Applies TIA object priority rules to determine which object's
    color register is used at this position.

    Returns: NTSC color index (0-255, even values only).
    """
    var pf = playfield_mask(state, pixel)
    var p0 = player_mask(state, 0, pixel)
    var p1 = player_mask(state, 1, pixel)
    var m0 = missile_mask(state, 0, pixel)
    var m1 = missile_mask(state, 1, pixel)
    var bl = ball_mask(state, pixel)

    var pf_priority = (state.tia_flags & TIA_PF_PRIORITY) != 0
    var pf_score = (state.tia_flags & TIA_PF_SCORE) != 0

    if pf_priority:
        # Playfield/Ball have priority over players/missiles
        if pf or bl:
            if pf_score and pf:
                # Score mode: left half = COLUP0, right half = COLUP1
                if pixel < 80:
                    return state.colup0
                else:
                    return state.colup1
            return state.colupf
        if p0 or m0:
            return state.colup0
        if p1 or m1:
            return state.colup1
    else:
        # Standard priority: players > playfield > background
        if p0 or m0:
            return state.colup0
        if p1 or m1:
            return state.colup1
        if pf or bl:
            if pf_score and pf:
                if pixel < 80:
                    return state.colup0
                else:
                    return state.colup1
            return state.colupf

    return state.colubk


@always_inline
fn render_scanline_bgra(
    state: AtariState,
    scanline: Int,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render one scanline (160 pixels) into the BGRA frame buffer.

    Must be called DURING frame execution, after the CPU has executed
    this scanline's worth of cycles (so TIA registers reflect the game's
    writes for this line).

    Also updates collision detection for this scanline.

    Args:
        state: Current AtariState (TIA registers for this scanline).
        scanline: Visible scanline index (0-209).
        buf: Frame buffer (160×210×4 bytes, BGRA).
    """
    if scanline < 0 or scanline >= FRAME_HEIGHT:
        return

    # During VBLANK, the display is blanked — output background color
    var is_blanked = (state.tia_flags & TIA_VBLANK) != 0

    var row_offset = scanline * FRAME_WIDTH * 4
    for x in range(FRAME_WIDTH):
        var color_idx: UInt8
        if is_blanked:
            color_idx = state.colubk
        else:
            color_idx = _get_pixel_color(state, x)

        var r = palette_r(color_idx)
        var g = palette_g(color_idx)
        var b = palette_b(color_idx)

        var offset = row_offset + x * 4
        buf[offset + 0] = b      # B
        buf[offset + 1] = g      # G
        buf[offset + 2] = r      # R
        buf[offset + 3] = 0xFF   # A


@always_inline
fn render_pixel_range_bgra(
    state: AtariState,
    scanline: Int,
    start_pixel: Int,
    end_pixel: Int,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render a range of pixels [start_pixel, end_pixel) on one scanline.

    Used for incremental rendering during CPU execution, so mid-scanline
    register writes (e.g. PF changes for score digits) take effect at the
    correct pixel position.
    """
    if scanline < 0 or scanline >= FRAME_HEIGHT:
        return

    var is_blanked = (state.tia_flags & TIA_VBLANK) != 0
    var row_offset = scanline * FRAME_WIDTH * 4

    for x in range(start_pixel, end_pixel):
        var color_idx: UInt8
        if is_blanked:
            color_idx = state.colubk
        else:
            color_idx = _get_pixel_color(state, x)

        var r = palette_r(color_idx)
        var g = palette_g(color_idx)
        var b = palette_b(color_idx)

        var offset = row_offset + x * 4
        buf[offset + 0] = b      # B
        buf[offset + 1] = g      # G
        buf[offset + 2] = r      # R
        buf[offset + 3] = 0xFF   # A


fn render_frame_bgra(
    state: AtariState,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render a static frame snapshot (160×210) into a BGRA pixel buffer.

    WARNING: This uses a single TIA state for all scanlines, so it only
    works for static scenes (tests, paused display). For correct rendering
    during gameplay, use run_frame_with_video() which calls
    render_scanline_bgra() per scanline during CPU execution.
    """
    for y in range(FRAME_HEIGHT):
        render_scanline_bgra(state, y, buf)


fn render_frame_rgb(
    state: AtariState,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render one full frame (160×210) into an RGB pixel buffer.

    3 bytes per pixel (R, G, B). Used for RL observations.
    Same static-snapshot caveat as render_frame_bgra().
    """
    for y in range(FRAME_HEIGHT):
        var row_offset = y * FRAME_WIDTH * 3
        for x in range(FRAME_WIDTH):
            var color_idx = _get_pixel_color(state, x)
            var offset = row_offset + x * 3
            buf[offset + 0] = palette_r(color_idx)
            buf[offset + 1] = palette_g(color_idx)
            buf[offset + 2] = palette_b(color_idx)


fn render_frame_grayscale(
    state: AtariState,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render one full frame (160×210) as grayscale.

    1 byte per pixel (Y luminance). Used for preprocessed RL observations.
    Same static-snapshot caveat as render_frame_bgra().
    """
    from .palette import palette_grayscale

    for y in range(FRAME_HEIGHT):
        var row_offset = y * FRAME_WIDTH
        for x in range(FRAME_WIDTH):
            var color_idx = _get_pixel_color(state, x)
            buf[row_offset + x] = palette_grayscale(color_idx)
