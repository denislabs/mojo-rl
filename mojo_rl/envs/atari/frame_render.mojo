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
from .palette import NTSC_PALETTE, palette_r, palette_g, palette_b
from .tables import player_mask, missile_mask, ball_mask, playfield_mask
from .flags import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    TIA_PF_PRIORITY,
    TIA_PF_SCORE,
    TIA_VBLANK,
    CX_M0P1,
    CX_M0P0,
    CX_M1P0,
    CX_M1P1,
    CX_P0PF,
    CX_P0BL,
    CX_P1PF,
    CX_P1BL,
    CX_M0PF,
    CX_M0BL,
    CX_M1PF,
    CX_M1BL,
    CX_BLPF,
    CX_P0P1,
    CX_M0M1,
)

# Frame buffer size: 160x210 pixels, 4 bytes per pixel (BGRA)
comptime FRAME_BUF_SIZE: Int = FRAME_WIDTH * FRAME_HEIGHT * 4

# A/B toggle (temporary): True = Stella-style per-color-clock collision latched
# in pass 1 from live state; False = legacy end-of-line collision in pass 2
# against the rendered frame. Used to measure the laser/phantom impact.
comptime COLLIDE_PER_CLOCK: Bool = False


@always_inline
def _get_pixel_color(state: AtariState, pixel: Int) -> UInt8:
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
def _write_pixel_bgra(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    offset: Int,
    color_idx: UInt8,
    palette: InlineArray[UInt32, 256],
):
    """Write one BGRA pixel using a pre-materialized palette."""
    var rgb = palette[Int(color_idx)]
    buf[offset + 0] = UInt8(rgb & 0xFF)  # B
    buf[offset + 1] = UInt8((rgb >> 8) & 0xFF)  # G
    buf[offset + 2] = UInt8((rgb >> 16) & 0xFF)  # R
    buf[offset + 3] = 0xFF  # A


@always_inline
def render_pf_collide_pixel(
    mut state: AtariState,
    render_row: Int,
    pixel: Int,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    palette: InlineArray[UInt32, 256],
    pf_mask: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Pass 1 (beam-accurate): render playfield + background AND latch all
    collisions for ONE beam pixel, from LIVE object state.

    Called from the scanline loop while the CPU executes, so every register
    (PF, player/missile/ball positions, GRP, enables) holds the value active at
    this exact beam position. This mirrors Stella's TIA, which evaluates each
    object's 1-bit output and ORs the 15 collision pairs once per color clock
    from the very same per-clock bits it renders (TIA::updateCollision). Doing
    collisions here — rather than once at end-of-line — means a fast object
    (Breakout ball, Space Invaders laser) is tested at the position it actually
    occupied when the beam passed, so it can't slip past a target between the
    render and a deferred collision check.

    The playfield is drawn here too (live PF → mid-line rewrites land at the
    right pixel); `pf_mask` records what was rendered so pass 2 can resolve
    sprite-vs-PF draw priority. Sprites themselves are still DRAWN in pass 2
    from the settled end-of-line state, which avoids the edge clipping that
    per-pixel beam positioning causes for sprite-repositioning games.
    """
    if (state.tia_flags & TIA_VBLANK) != 0:
        pf_mask[pixel] = 0
        if render_row >= 0 and render_row < FRAME_HEIGHT:
            _write_pixel_bgra(
                buf,
                (render_row * FRAME_WIDTH + pixel) * 4,
                state.colubk,
                palette,
            )
        return

    # Live per-clock object bits (identical evaluation to the rendered pixel).
    var pf = playfield_mask(state, pixel, live=True)

    @parameter
    if COLLIDE_PER_CLOCK:
        var p0 = player_mask(state, 0, pixel)
        var p1 = player_mask(state, 1, pixel)
        var m0 = missile_mask(state, 0, pixel)
        var m1 = missile_mask(state, 1, pixel)
        var bl = ball_mask(state, pixel)

        # --- Collision latches (per color clock, from live state) ---
        if m0 and p1:
            state.collision = state.collision | CX_M0P1
        if m0 and p0:
            state.collision = state.collision | CX_M0P0
        if m1 and p0:
            state.collision = state.collision | CX_M1P0
        if m1 and p1:
            state.collision = state.collision | CX_M1P1
        if p0 and pf:
            state.collision = state.collision | CX_P0PF
        if p0 and bl:
            state.collision = state.collision | CX_P0BL
        if p1 and pf:
            state.collision = state.collision | CX_P1PF
        if p1 and bl:
            state.collision = state.collision | CX_P1BL
        if m0 and pf:
            state.collision = state.collision | CX_M0PF
        if m0 and bl:
            state.collision = state.collision | CX_M0BL
        if m1 and pf:
            state.collision = state.collision | CX_M1PF
        if m1 and bl:
            state.collision = state.collision | CX_M1BL
        if bl and pf:
            state.collision = state.collision | CX_BLPF
        if p0 and p1:
            state.collision = state.collision | CX_P0P1
        if m0 and m1:
            state.collision = state.collision | CX_M0M1

    # --- Render playfield/background (sprites overlaid in pass 2) ---
    var color_idx: UInt8
    if pf:
        if (state.tia_flags & TIA_PF_SCORE) != 0:
            color_idx = state.colup0 if pixel < 80 else state.colup1
        else:
            color_idx = state.colupf
    else:
        color_idx = state.colubk

    pf_mask[pixel] = 1 if pf else 0
    if render_row >= 0 and render_row < FRAME_HEIGHT:
        _write_pixel_bgra(
            buf, (render_row * FRAME_WIDTH + pixel) * 4, color_idx, palette
        )


@always_inline
def overlay_sprites_pixel(
    mut state: AtariState,
    render_row: Int,
    pixel: Int,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    palette: InlineArray[UInt32, 256],
    pf_mask: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Pass 2 (end-of-line): overlay players/missiles/ball onto the frame.

    Collisions are NOT computed here — they were latched per beam pixel in
    pass 1 (render_pf_collide_pixel) from live state, matching Stella's
    per-color-clock collision model. This pass only resolves the VISIBLE pixel:
    it draws sprites from the settled end-of-line state (which keeps
    sprite-repositioning games like Space Invaders free of edge clipping) using
    the playfield presence recorded in `pf_mask` for priority.
    """
    if (state.tia_flags & TIA_VBLANK) != 0:
        return
    if render_row < 0 or render_row >= FRAME_HEIGHT:
        return

    var pf = pf_mask[pixel] != 0  # what pass 1 actually rendered here
    var p0 = player_mask(state, 0, pixel)
    var p1 = player_mask(state, 1, pixel)
    var m0 = missile_mask(state, 0, pixel)
    var m1 = missile_mask(state, 1, pixel)
    var bl = ball_mask(state, pixel)

    # Legacy A/B path: latch collisions here (end-of-line) instead of per clock.
    @parameter
    if not COLLIDE_PER_CLOCK:
        if m0 and p1:
            state.collision = state.collision | CX_M0P1
        if m0 and p0:
            state.collision = state.collision | CX_M0P0
        if m1 and p0:
            state.collision = state.collision | CX_M1P0
        if m1 and p1:
            state.collision = state.collision | CX_M1P1
        if p0 and pf:
            state.collision = state.collision | CX_P0PF
        if p0 and bl:
            state.collision = state.collision | CX_P0BL
        if p1 and pf:
            state.collision = state.collision | CX_P1PF
        if p1 and bl:
            state.collision = state.collision | CX_P1BL
        if m0 and pf:
            state.collision = state.collision | CX_M0PF
        if m0 and bl:
            state.collision = state.collision | CX_M0BL
        if m1 and pf:
            state.collision = state.collision | CX_M1PF
        if m1 and bl:
            state.collision = state.collision | CX_M1BL
        if bl and pf:
            state.collision = state.collision | CX_BLPF
        if p0 and p1:
            state.collision = state.collision | CX_P0P1
        if m0 and m1:
            state.collision = state.collision | CX_M0M1

    # Decide whether a sprite/ball overwrites the pass-1 (PF/bg) pixel.
    var overwrite = False
    var color_idx: UInt8 = 0
    if (state.tia_flags & TIA_PF_PRIORITY) != 0:
        # PF/BL > P0/M0 > P1/M1 > BG. Where PF is present, pass 1 is correct.
        if pf:
            pass
        elif bl:
            overwrite = True
            color_idx = state.colupf
        elif p0 or m0:
            overwrite = True
            color_idx = state.colup0
        elif p1 or m1:
            overwrite = True
            color_idx = state.colup1
    else:
        # P0/M0 > P1/M1 > PF/BL > BG. Players always win; ball draws at PF
        # level (same color register), so it's safe to set even over PF.
        if p0 or m0:
            overwrite = True
            color_idx = state.colup0
        elif p1 or m1:
            overwrite = True
            color_idx = state.colup1
        elif bl:
            overwrite = True
            color_idx = state.colupf

    if overwrite:
        _write_pixel_bgra(
            buf, (render_row * FRAME_WIDTH + pixel) * 4, color_idx, palette
        )


@always_inline
def render_scanline_bgra(
    state: AtariState,
    scanline: Int,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render one scanline (160 pixels) into the BGRA frame buffer.

    Must be called DURING frame execution, after the CPU has executed
    this scanline's worth of cycles (so TIA registers reflect the game's
    writes for this line).

    Args:
        state: Current AtariState (TIA registers for this scanline).
        scanline: Visible scanline index (0-209).
        buf: Frame buffer (160×210×4 bytes, BGRA).
    """
    if scanline < 0 or scanline >= FRAME_HEIGHT:
        return

    # During VBLANK, the display is blanked — output background color
    var is_blanked = (state.tia_flags & TIA_VBLANK) != 0

    # Materialize palette ONCE per scanline (avoids 3 copies per pixel)
    var palette = materialize[NTSC_PALETTE]()

    var row_offset = scanline * FRAME_WIDTH * 4
    for x in range(FRAME_WIDTH):
        var color_idx: UInt8
        if is_blanked:
            color_idx = state.colubk
        else:
            color_idx = _get_pixel_color(state, x)

        _write_pixel_bgra(buf, row_offset + x * 4, color_idx, palette)


@always_inline
def render_scanline_with_collision_bgra(
    mut state: AtariState,
    scanline: Int,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render one scanline AND update collision in a single pass.

    Computes masks (player, missile, ball, playfield) once per pixel and
    uses them for both rendering and collision detection. This halves the
    total mask computations compared to separate render + collision passes.
    """
    if scanline < 0 or scanline >= FRAME_HEIGHT:
        return

    var is_blanked = (state.tia_flags & TIA_VBLANK) != 0
    var palette = materialize[NTSC_PALETTE]()
    var pf_priority = (state.tia_flags & TIA_PF_PRIORITY) != 0
    var pf_score = (state.tia_flags & TIA_PF_SCORE) != 0
    var row_offset = scanline * FRAME_WIDTH * 4

    for x in range(FRAME_WIDTH):
        # Compute all masks ONCE
        var pf = playfield_mask(state, x)
        var p0 = player_mask(state, 0, x)
        var p1 = player_mask(state, 1, x)
        var m0 = missile_mask(state, 0, x)
        var m1 = missile_mask(state, 1, x)
        var bl = ball_mask(state, x)

        # --- Collision detection (reuse masks) ---
        if m0 and p1:
            state.collision = state.collision | CX_M0P1
        if m0 and p0:
            state.collision = state.collision | CX_M0P0
        if m1 and p0:
            state.collision = state.collision | CX_M1P0
        if m1 and p1:
            state.collision = state.collision | CX_M1P1
        if p0 and pf:
            state.collision = state.collision | CX_P0PF
        if p0 and bl:
            state.collision = state.collision | CX_P0BL
        if p1 and pf:
            state.collision = state.collision | CX_P1PF
        if p1 and bl:
            state.collision = state.collision | CX_P1BL
        if m0 and pf:
            state.collision = state.collision | CX_M0PF
        if m0 and bl:
            state.collision = state.collision | CX_M0BL
        if m1 and pf:
            state.collision = state.collision | CX_M1PF
        if m1 and bl:
            state.collision = state.collision | CX_M1BL
        if bl and pf:
            state.collision = state.collision | CX_BLPF
        if p0 and p1:
            state.collision = state.collision | CX_P0P1
        if m0 and m1:
            state.collision = state.collision | CX_M0M1

        # --- Rendering (reuse masks) ---
        var color_idx: UInt8
        if is_blanked:
            color_idx = state.colubk
        elif pf_priority:
            if pf or bl:
                if pf_score and pf:
                    color_idx = state.colup0 if x < 80 else state.colup1
                else:
                    color_idx = state.colupf
            elif p0 or m0:
                color_idx = state.colup0
            elif p1 or m1:
                color_idx = state.colup1
            else:
                color_idx = state.colubk
        else:
            if p0 or m0:
                color_idx = state.colup0
            elif p1 or m1:
                color_idx = state.colup1
            elif pf or bl:
                if pf_score and pf:
                    color_idx = state.colup0 if x < 80 else state.colup1
                else:
                    color_idx = state.colupf
            else:
                color_idx = state.colubk

        _write_pixel_bgra(buf, row_offset + x * 4, color_idx, palette)


@always_inline
def render_pixel_range_bgra(
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
    var palette = materialize[NTSC_PALETTE]()
    var row_offset = scanline * FRAME_WIDTH * 4

    for x in range(start_pixel, end_pixel):
        var color_idx: UInt8
        if is_blanked:
            color_idx = state.colubk
        else:
            color_idx = _get_pixel_color(state, x)

        _write_pixel_bgra(buf, row_offset + x * 4, color_idx, palette)


def render_frame_bgra(
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


def render_frame_rgb(
    state: AtariState,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render one full frame (160×210) into an RGB pixel buffer.

    3 bytes per pixel (R, G, B). Used for RL observations.
    Same static-snapshot caveat as render_frame_bgra().
    """
    var palette = materialize[NTSC_PALETTE]()
    for y in range(FRAME_HEIGHT):
        var row_offset = y * FRAME_WIDTH * 3
        for x in range(FRAME_WIDTH):
            var color_idx = _get_pixel_color(state, x)
            var rgb = palette[Int(color_idx)]
            var offset = row_offset + x * 3
            buf[offset + 0] = UInt8((rgb >> 16) & 0xFF)
            buf[offset + 1] = UInt8((rgb >> 8) & 0xFF)
            buf[offset + 2] = UInt8(rgb & 0xFF)


def render_frame_grayscale(
    state: AtariState,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Render one full frame (160×210) as grayscale.

    1 byte per pixel (Y luminance). Used for preprocessed RL observations.
    Same static-snapshot caveat as render_frame_bgra().
    """
    var palette = materialize[NTSC_PALETTE]()
    for y in range(FRAME_HEIGHT):
        var row_offset = y * FRAME_WIDTH
        for x in range(FRAME_WIDTH):
            var color_idx = _get_pixel_color(state, x)
            var rgb = palette[Int(color_idx)]
            var r = Int((rgb >> 16) & 0xFF)
            var g = Int((rgb >> 8) & 0xFF)
            var b = Int(rgb & 0xFF)
            buf[row_offset + x] = UInt8((77 * r + 150 * g + 29 * b) >> 8)
