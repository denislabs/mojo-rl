# +--------------------------------------------------------------------------+ #
# | Fixed-width number and text formatting
# +--------------------------------------------------------------------------+ #
"""Column formatting that cannot crash on a short value.

⚠ Written after `String(x)[byte=0:6]` **aborted a running teleop loop** at
t=27 s, mid-motion, because a joint angle happened to print as `3.29` — four
bytes — and the slice asserted. The follower was left holding torque, because
an abort is not an exception and does not run `finally`.

The lesson generalises: **truncating a formatted number by slicing its string
is a latent crash keyed to the VALUE**. It survives every test where the
numbers happen to be wide enough. So format to a width instead of slicing to
one, and never let a display concern reach into a control loop's failure
modes.
"""

from std.math import isnan, isinf


def fixed(value: Float64, decimals: Int = 3) -> String:
    """`value` with exactly `decimals` digits after the point.

    Rounds half away from zero. Never slices, so any magnitude is safe;
    non-finite values print as `nan` / `inf` rather than a wall of digits.
    """
    if isnan(value):
        return String("nan")
    if isinf(value):
        return String("-inf") if value < 0 else String("inf")
    if decimals <= 0:
        return String(Int(value + (0.5 if value >= 0 else -0.5)))

    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = value * Float64(scale)
    var n = Int(scaled + (0.5 if scaled >= 0 else -0.5))
    var neg = n < 0
    if neg:
        n = -n
    var frac = String(n % scale)
    while frac.byte_length() < decimals:
        frac = "0" + frac
    var out = String(n // scale) + "." + frac
    return ("-" + out) if neg else out^


def pad_right(var s: String, width: Int) -> String:
    """Left-aligned in `width`. A value WIDER than the column keeps all its
    digits and pushes the row out — a misaligned table beats a silently
    truncated number."""
    while s.byte_length() < width:
        s += " "
    return s^


def pad_left(var s: String, width: Int) -> String:
    """Right-aligned in `width`, same over-wide policy as `pad_right`."""
    var pad = String("")
    while pad.byte_length() + s.byte_length() < width:
        pad += " "
    return pad + s


def col(value: Float64, width: Int, decimals: Int = 3) -> String:
    """A right-aligned fixed-point column — `fixed` then `pad_left`."""
    return pad_left(fixed(value, decimals), width)
