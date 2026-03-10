"""Arcade-style color palette — grayscale values for GPU rendering."""

# Grayscale color constants for the 160×210 framebuffer
comptime COLOR_BLACK: UInt8 = 0
comptime COLOR_WHITE: UInt8 = 255
comptime COLOR_RED: UInt8 = 180
comptime COLOR_ORANGE: UInt8 = 160
comptime COLOR_YELLOW: UInt8 = 200
comptime COLOR_GREEN: UInt8 = 140
comptime COLOR_AQUA: UInt8 = 170
comptime COLOR_BLUE: UInt8 = 100
comptime COLOR_GRAY: UInt8 = 128
comptime COLOR_LIGHT_GRAY: UInt8 = 192
comptime COLOR_DARK_GRAY: UInt8 = 64

# Screen dimensions (Arcade standard)
comptime SCREEN_W: Int = 160
comptime SCREEN_H: Int = 210

# Observation dimensions (after preprocessing)
comptime OBS_W: Int = 84
comptime OBS_H: Int = 84
comptime FRAME_STACK: Int = 4
comptime PIXEL_OBS_DIM: Int = FRAME_STACK * OBS_W * OBS_H  # 4 * 84 * 84 = 28224
