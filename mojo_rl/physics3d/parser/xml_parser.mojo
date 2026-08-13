"""Minimal MJCF XML parser — extracts model dimensions from a MuJoCo XML string.

This module is designed to run entirely at comptime:

    comptime model = parse_xml(my_xml_string)
    print(model.NBODY)   # 8 for HalfCheetah

It only counts structural elements (bodies, joints, geoms, actuators) and
joint types to compute NQ/NV. It does NOT yet parse kinematics or dynamics
parameters — that is the next step toward a full ModelDef from XML.

Design notes
------------
We deliberately avoid String subscript access (`s[i]`) since its comptime
behaviour in Mojo nightly is unclear. Instead every check is expressed via
`String.find()`, which is known to work at comptime.

A tag `<foo` is considered a real tag occurrence if the character immediately
after the tag-name is one of: SPACE, >, /, NEWLINE, TAB.  We enumerate these
by searching for four explicit suffix patterns: `<foo `, `<foo>`, `<foo/`,
`<foo\n`, `<foo\t`.  This avoids false matches on longer tag names (e.g.
`<worldbody` would NOT match a search for `<body `).
"""

from std.collections import InlineArray

from .flat_model import ACT_KIND_MOTOR, ACT_KIND_POSITION, ACT_KIND_VELOCITY
from ..gpu.constants import MJ_CCD_TOLERANCE, MJ_CCD_ITERATIONS


# =============================================================================
# ParsedModel — result of parsing
# =============================================================================


struct ParsedModel:
    """Dimension counts extracted from a MuJoCo XML string.

    Fields are ordinary Int (not comptime constants), but when the struct is
    bound to a `comptime` variable the values are compile-time known:

        comptime m = parse_xml(xml)
        # m.NBODY is now a comptime Int — can drive type parameters
    """

    var NBODY: Int  # total bodies including worldbody (= counted bodies + 1)
    var NJOINT: Int  # number of joints
    var NQ: Int  # total position DOFs
    var NV: Int  # total velocity DOFs
    var NGEOM: Int  # total geoms (floor + body geoms)
    var NACT: Int  # number of actuators
    var NTEX: Int  # number of <texture> entries in <asset>
    var NMAT: Int  # number of <material> entries in <asset>
    var NLIGHT: Int  # number of <light> entries in <worldbody>
    var NCAM: Int  # number of <camera> entries in <worldbody>
    var NSITE: Int  # number of <site> entries in <worldbody>
    var NEQ: Int  # number of equality constraints (<weld> + <connect> in <equality>)
    var NEXCLUDE: Int  # number of <exclude> entries in <contact>
    var NPAIR: Int  # number of <pair> entries in <contact>
    var NTENDON: Int  # number of <fixed> + <spatial> entries in <tendon>
    var ANGLE_DEG: Bool  # True when <compiler angle="degree"/>
    var TIMESTEP: Float64  # <option timestep="..."/>
    var MAX_CONDIM: Int  # largest `condim=` anywhere in the file (>= 3)
    var NOSLIP_ITER: Int  # <option noslip_iterations="..."/>, 0 = pass off
    var CCD_TOL: Float64  # <option ccd_tolerance="..."/>, MuJoCo default 1e-6
    var CCD_ITER: Int  # <option ccd_iterations="..."/>, MuJoCo default 35

    def __init__(
        out self,
        nbody: Int = 0,
        njoint: Int = 0,
        nq: Int = 0,
        nv: Int = 0,
        ngeom: Int = 0,
        nact: Int = 0,
        ntex: Int = 0,
        nmat: Int = 0,
        nlight: Int = 0,
        ncam: Int = 0,
        nsite: Int = 0,
        neq: Int = 0,
        nexclude: Int = 0,
        npair: Int = 0,
        ntendon: Int = 0,
        angle_deg: Bool = False,
        timestep: Float64 = 0.01,
        max_condim: Int = 3,
        noslip_iter: Int = 0,
        ccd_tol: Float64 = MJ_CCD_TOLERANCE,
        ccd_iter: Int = MJ_CCD_ITERATIONS,
    ):
        self.NBODY = nbody
        self.NJOINT = njoint
        self.NQ = nq
        self.NV = nv
        self.NGEOM = ngeom
        self.NACT = nact
        self.NTEX = ntex
        self.NMAT = nmat
        self.NLIGHT = nlight
        self.NCAM = ncam
        self.NSITE = nsite
        self.NEQ = neq
        self.NEXCLUDE = nexclude
        self.NPAIR = npair
        self.NTENDON = ntendon
        self.ANGLE_DEG = angle_deg
        self.TIMESTEP = timestep
        self.MAX_CONDIM = max_condim
        self.NOSLIP_ITER = noslip_iter
        self.CCD_TOL = ccd_tol
        self.CCD_ITER = ccd_iter

    def __str__(self) -> String:
        return (
            "ParsedModel("
            + "NBODY="
            + String(self.NBODY)
            + ", NJOINT="
            + String(self.NJOINT)
            + ", NQ="
            + String(self.NQ)
            + ", NV="
            + String(self.NV)
            + ", NGEOM="
            + String(self.NGEOM)
            + ", NACT="
            + String(self.NACT)
            + ", NTEX="
            + String(self.NTEX)
            + ", NMAT="
            + String(self.NMAT)
            + ", NLIGHT="
            + String(self.NLIGHT)
            + ", NCAM="
            + String(self.NCAM)
            + ", NSITE="
            + String(self.NSITE)
            + ", TIMESTEP="
            + String(self.TIMESTEP)
            + ")"
        )


# =============================================================================
# Low-level string helpers (comptime-friendly, no subscript access)
# =============================================================================


def _count_exact(xml: String, search: String) -> Int:
    """Count non-overlapping occurrences of `search` in `xml`."""
    var count = 0
    var start = 0
    while True:
        var pos = xml.find(search, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def _strip_xml_comments(s: String) -> String:
    """Strip all XML comments <!-- ... --> from the string.

    Handles multiple comments and nested <!-- in comment text.
    """
    var result = s
    while True:
        var start = result.find("<!--")
        if start == -1:
            break
        var end = result.find("-->", start + 4)
        if end == -1:
            break  # Malformed XML, stop stripping
        # Build into a temporary first: nightly's exclusivity check rejects
        # constructing a `String` from slices of the very string being assigned.
        var stripped = result[byte=:start] + result[byte = end + 3 :]
        result = stripped^
    return result


def _count_tag(xml: String, tag: String) -> Int:
    """Count occurrences of `<tag` followed by SPACE, >, /, NEWLINE, or TAB.

    This intentionally does NOT match longer tag names: `<body ` will NOT
    match `<worldbody>` or `<bthigh>` etc.
    """
    var base = "<" + tag
    return (
        _count_exact(xml, base + " ")
        + _count_exact(xml, base + ">")
        + _count_exact(xml, base + "/")
        + _count_exact(xml, base + "\n")
        + _count_exact(xml, base + "\t")
    )


def _extract_section(xml: String, tag: String) -> String:
    """Return the substring from `<tag` to (including) `</tag>`.

    Returns empty string if the section is not found.
    Handles `<tag>` and `<tag ...>` (with attributes).

    ⚠ SELF-CLOSING `<tag ... />` elements are skipped rather than taken as the
    section opener. MJCF puts per-class defaults in exactly that form, so
    `<default class="coupling"><equality solimp=... solref=.../></default>`
    (dm_control's quadruped) otherwise made this return everything from the
    DEFAULT block to the real `</equality>` — a section starting in the middle
    of `<default>` and containing the whole `<worldbody>`. Also skips a name
    that is a strict prefix of another tag, which the old bare `find` did not:
    `<tendon` matched `<tendonlimited` shaped names.
    """
    var open_marker = "<" + tag
    var close_marker = "</" + tag + ">"
    var n = xml.byte_length()
    var scan = 0
    while scan < n:
        var start = xml.find(open_marker, scan)
        if start == -1:
            return String("")
        # Reject substring matches: the char after the name must end the name.
        var after_pos = start + open_marker.byte_length()
        if after_pos < n:
            var ch = String(xml[byte = after_pos : after_pos + 1])
            if (
                ch != " "
                and ch != ">"
                and ch != "/"
                and ch != "\n"
                and ch != "\t"
                and ch != "\r"
            ):
                scan = after_pos
                continue
        var tag_end = xml.find(">", start)
        if tag_end == -1:
            return String("")
        if _is_self_closing(xml, start, tag_end):
            scan = tag_end + 1
            continue
        var end = xml.find(close_marker, start)
        if end == -1:
            return String("")
        return String(xml[byte = start : end + close_marker.byte_length()])
    return String("")


# =============================================================================
# Attribute extraction helpers (comptime-friendly, no subscript access)
# =============================================================================


def _trim(s: String) -> String:
    """Trim leading/trailing whitespace (space, tab, newline, carriage return).
    """
    var start = 0
    var end = s.byte_length()
    while start < end:
        var c = s[byte = start : start + 1]
        if c == " " or c == "\t" or c == "\n" or c == "\r":
            start += 1
        else:
            break
    while end > start:
        var c = s[byte = end - 1 : end]
        if c == " " or c == "\t" or c == "\n" or c == "\r":
            end -= 1
        else:
            break
    if start >= end:
        return String("")
    return String(s[byte=start:end])


def _extract_opening_tag(xml: String, pos: Int) -> String:
    """From `<tag` at pos, extract everything up to (and including) `>` or `/>`.

    Returns the raw opening-tag string for attribute parsing.
    """
    var end = xml.find(">", pos)
    if end == -1:
        return String("")
    return String(xml[byte = pos : end + 1])


def _extract_attr(tag: String, attr: String) -> String:
    """Extract value from attr="value" or attr='value' in a tag string.

    Returns "" if not found.
    Matches standalone attribute names only (preceded by space/tab/newline),
    avoiding substring matches like "contype" when searching for "type".
    """
    # Try double-quoted form: attr="..."
    var search_dq = attr + '="'
    var search_len = search_dq.byte_length()
    var pos = tag.find(search_dq)
    while pos != -1:
        # Ensure standalone match: char before must be space/tab/newline
        if pos == 0 or _is_attr_separator(String(tag[byte = pos - 1 : pos])):
            var val_start = pos + search_len
            var val_end = tag.find('"', val_start)
            if val_end != -1:
                return String(tag[byte=val_start:val_end])
        pos = tag.find(search_dq, pos + 1)
    # Try single-quoted form: attr='...'
    var search_sq = attr + "='"
    var search_sq_len = search_sq.byte_length()
    pos = tag.find(search_sq)
    while pos != -1:
        if pos == 0 or _is_attr_separator(String(tag[byte = pos - 1 : pos])):
            var val_start = pos + search_sq_len
            var val_end = tag.find("'", val_start)
            if val_end != -1:
                return String(tag[byte=val_start:val_end])
        pos = tag.find(search_sq, pos + 1)
    return String("")


def _is_attr_separator(c: String) -> Bool:
    """Check if character is a valid separator before an attribute name."""
    return c == " " or c == "\t" or c == "\n" or c == "\r"


def _digit_value(c: String) -> Int:
    """Return integer value 0-9 for digit character; -1 if not a digit."""
    var digits = "0123456789"
    return digits.find(c)


def _pow10(k: Int) -> Float64:
    """10^k, exactly, for 0 <= k <= 22.

    Every power of ten up to 10^22 is representable in Float64 (10^22 = 2^22 *
    5^22 and 5^22 < 2^53), and each step of the loop lands on a representable
    value, so the product is exact. Past 22 it is no longer exact — MJCF
    numbers never reach there, and the alternative (repeated *0.1) is far
    worse, being inexact from the very first step.
    """
    var p = Float64(1.0)
    for _ in range(k):
        p *= 10.0
    return p


def _parse_float(s: String) -> Float64:
    """Parse a float string such as "0.7", "-3.14", "1e-3" to Float64.

    Uses slice-based character iteration (s[i:i+1]) — comptime-safe.
    No stdlib float parsing is used.

    All digits go into ONE integer-valued mantissa which is scaled by a single
    power of ten at the end, so the result is the correctly-rounded double
    whenever the mantissa fits in 2^53 and the decimal exponent is within
    +-22 — true of every number in an MJCF file.

    This used to accumulate the fraction as `sum(digit * mul)` with
    `mul *= 0.1`, which is inexact from the first digit: 0.1 is not
    representable, so `<option timestep="0.02"/>` parsed to
    0.020000000000000004, one ULP high. That is a systematic ~1e-16 relative
    error on every float in every model, and it compounds over a rollout —
    which is exactly the regime the dm_control parity tests measure.
    """
    var t = _trim(s)
    if t.byte_length() == 0:
        return Float64(0)

    # Sign
    var neg = False
    var start = 0
    if t[byte=0:1] == "-":
        neg = True
        start = 1
    elif t[byte=0:1] == "+":
        start = 1

    # Find decimal point and exponent marker
    var dot_pos = t.find(".")
    var exp_pos = t.find("e")
    if exp_pos == -1:
        exp_pos = t.find("E")

    # Integer part range: [start, int_end)
    var int_end: Int
    if dot_pos != -1:
        int_end = dot_pos
    elif exp_pos != -1:
        int_end = exp_pos
    else:
        int_end = t.byte_length()

    # One shared mantissa for the integer and fractional digits; `frac_digits`
    # counts how far the decimal point must move back at the end. Digits past
    # the 17th cannot change a Float64, so they are counted (to keep the
    # exponent right) but not accumulated — that also keeps the mantissa
    # under 2^53, where the integer arithmetic below stays exact.
    comptime MAX_MANTISSA_DIGITS = 17
    var mantissa = Float64(0)
    var ndigits = 0

    for i in range(start, int_end):
        var d = _digit_value(String(t[byte = i : i + 1]))
        if d >= 0:
            if ndigits < MAX_MANTISSA_DIGITS:
                mantissa = mantissa * 10.0 + Float64(d)
                ndigits += 1
            else:
                # A dropped INTEGER digit still scales the value.
                mantissa *= 10.0

    # Fractional part
    var frac_digits = 0
    if dot_pos != -1:
        var frac_end: Int
        if exp_pos != -1:
            frac_end = exp_pos
        else:
            frac_end = t.byte_length()
        for i in range(dot_pos + 1, frac_end):
            var d = _digit_value(String(t[byte = i : i + 1]))
            if d >= 0:
                if ndigits < MAX_MANTISSA_DIGITS:
                    mantissa = mantissa * 10.0 + Float64(d)
                    ndigits += 1
                    frac_digits += 1
                # A dropped FRACTIONAL digit is simply below precision.

    var result = mantissa
    if frac_digits > 0:
        result /= _pow10(frac_digits)

    # Exponent part
    if exp_pos != -1:
        var exp_start = exp_pos + 1
        var exp_neg = False
        if exp_start < t.byte_length():
            if t[byte = exp_start : exp_start + 1] == "-":
                exp_neg = True
                exp_start += 1
            elif t[byte = exp_start : exp_start + 1] == "+":
                exp_start += 1
        var exp_val = 0
        for i in range(exp_start, t.byte_length()):
            var d = _digit_value(String(t[byte = i : i + 1]))
            if d >= 0:
                exp_val = exp_val * 10 + d
        # Scale by a single exact power of ten — DIVIDING for a negative
        # exponent rather than multiplying by an inexact 0.1^k.
        var pow10 = _pow10(exp_val)
        if exp_neg:
            result /= pow10
        else:
            result *= pow10

    if neg:
        return -result
    return result


def _parse_int_str(s: String) -> Int:
    """Parse "3", "-1" etc. to Int."""
    var t = _trim(s)
    if t.byte_length() == 0:
        return 0
    var neg = False
    var start = 0
    if t[byte=0:1] == "-":
        neg = True
        start = 1
    var val = 0
    for i in range(start, t.byte_length()):
        var d = _digit_value(String(t[byte = i : i + 1]))
        if d >= 0:
            val = val * 10 + d
    if neg:
        return -val
    return val


def _split_spaces(s: String, mut parts: List[String]):
    """Split string by whitespace runs into parts (in-place fill)."""
    var t = _trim(s)
    var start = 0
    var n = t.byte_length()
    while start < n:
        # Skip whitespace
        while start < n:
            var c = t[byte = start : start + 1]
            if c == " " or c == "\t" or c == "\n" or c == "\r":
                start += 1
            else:
                break
        if start >= n:
            break
        # Find end of token
        var end = start + 1
        while end < n:
            var c = t[byte = end : end + 1]
            if c == " " or c == "\t" or c == "\n" or c == "\r":
                break
            end += 1
        parts.append(String(t[byte=start:end]))
        start = end


def _parse_vec3(s: String) -> Tuple[Float64, Float64, Float64]:
    """Parse "x y z" space-separated string into (x, y, z)."""
    var parts = List[String]()
    _split_spaces(s, parts)
    var x = Float64(0)
    var y = Float64(0)
    var z = Float64(0)
    if len(parts) >= 1:
        x = _parse_float(parts[0])
    if len(parts) >= 2:
        y = _parse_float(parts[1])
    if len(parts) >= 3:
        z = _parse_float(parts[2])
    return (x, y, z)


def _parse_quat(s: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """Parse MuJoCo "w x y z" quaternion string into internal (qx, qy, qz, qw).

    MuJoCo XML stores all quaternion attributes (body quat, geom quat, iquat,
    joint quat) in (w, x, y, z) order. Our internal representation is (x, y, z, w).

    The result is NORMALIZED, as MuJoCo's compiler does to every quat it
    reads (`mju_normalize4` in `user_objects.cc`). Hand-written MJCF is
    routinely a hair off unit length — dm_control's humanoid writes
    `quat="1.000 0 -.002 0"` on `lower_waist`, norm 1.000002 — and an
    unnormalized quat scales every vector it rotates by |q|^2, which leaked
    ~4e-6 of relative error into that body's whole subtree. Normalizing at
    parse time keeps it out of the kinematics rather than papering over it
    downstream. Degenerate (all-zero) input falls back to identity.
    """
    var parts = List[String]()
    _split_spaces(s, parts)
    var qw = Float64(1)
    var qx = Float64(0)
    var qy = Float64(0)
    var qz = Float64(0)
    if len(parts) >= 1:
        qw = _parse_float(parts[0])
    if len(parts) >= 2:
        qx = _parse_float(parts[1])
    if len(parts) >= 3:
        qy = _parse_float(parts[2])
    if len(parts) >= 4:
        qz = _parse_float(parts[3])
    var n = _sqrt_f64(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= Float64(0):
        return (Float64(0), Float64(0), Float64(0), Float64(1))
    return (qx / n, qy / n, qz / n, qw / n)


def _sqrt_f64(x: Float64) -> Float64:
    """Sqrt via Newton–Raphson (comptime-safe, no stdlib)."""
    if x <= Float64(0):
        return Float64(0)
    # Initial guess
    var g = x
    if g > Float64(1):
        g = x * Float64(0.5)
    # 20 Newton steps — converges rapidly
    for _ in range(20):
        g = (g + x / g) * Float64(0.5)
    return g


comptime _PI_F64: Float64 = 3.14159265358979323846
comptime _TWO_PI_F64: Float64 = 6.28318530717958647692


def _sin_cos_f64(x: Float64) -> Tuple[Float64, Float64]:
    """Return (sin x, cos x) for x in radians — comptime-safe, no stdlib math.

    Range-reduces to [-pi, pi], evaluates the Taylor series at x/8 (where
    |x/8| <= pi/8 and the 5/6-term truncation is ~1e-15), then applies the
    double-angle identities three times to climb back. Reducing by 4 instead
    of 8 leaves ~4e-11 at cheetah's `euler="0 -218 0"` — visible against
    MuJoCo's geom_quat.

    The reduction is not cosmetic: an un-reduced series is only accurate near
    zero, and cheetah's `euler="0 -218 0"` geoms need sin/cos at a half-angle
    of ~1.9 rad, where the plain 6-term series is off by ~5e-6 — three orders
    of magnitude above our parity gates.

    The reduction loop is deliberately fixed-trip with no early exit: a
    data-dependent `while` with a `break` is the shape that blows up Mojo
    compile times (see `scripts/audit_while_compile_risk.py`). 64 subtractions
    cover |x| up to ~400 rad, far beyond any angle an MJCF file states.
    """
    var r = x
    for _ in range(64):
        if r > _PI_F64:
            r = r - _TWO_PI_F64
        elif r < -_PI_F64:
            r = r + _TWO_PI_F64

    var t = r * Float64(0.125)
    var t2 = t * t
    # sin(t) = t - t³/6 + t⁵/120 - t⁷/5040 + t⁹/362880
    var s = (
        t
        - t * t2 / Float64(6)
        + t * t2 * t2 / Float64(120)
        - t * t2 * t2 * t2 / Float64(5040)
        + t * t2 * t2 * t2 * t2 / Float64(362880)
    )
    # cos(t) = 1 - t²/2 + t⁴/24 - t⁶/720 + t⁸/40320 - t¹⁰/3628800
    var c = (
        Float64(1)
        - t2 / Float64(2)
        + t2 * t2 / Float64(24)
        - t2 * t2 * t2 / Float64(720)
        + t2 * t2 * t2 * t2 / Float64(40320)
        - t2 * t2 * t2 * t2 * t2 / Float64(3628800)
    )
    # (sin t, cos t) -> 2t -> 4t -> 8t = r
    for _ in range(3):
        var s2 = Float64(2) * s * c
        var c2 = Float64(1) - Float64(2) * s * s
        s = s2
        c = c2
    return (s, c)


def _quat_mul(
    aw: Float64,
    ax: Float64,
    ay: Float64,
    az: Float64,
    bw: Float64,
    bx: Float64,
    by: Float64,
    bz: Float64,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Hamilton product a ⊗ b, both and the result in (w, x, y, z) order.

    Matches MuJoCo's `mjuu_mulquat`; kept in MuJoCo's (w,x,y,z) ordering so the
    euler accumulation below can be read against `ResolveOrientation` directly.
    """
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _z2quat(
    vx: Float64, vy: Float64, vz: Float64
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Minimal rotation taking +Z to `v` → quaternion (qx, qy, qz, qw).

    Mirrors MuJoCo's `mjuu_z2quat`. Used by both `zaxis="..."` and the capsule
    `fromto="..."` shorthand, which both mean "point local +Z along this vector".
    """
    var norm = _sqrt_f64(vx * vx + vy * vy + vz * vz)
    if norm < Float64(1e-10):
        return (Float64(0), Float64(0), Float64(0), Float64(1))
    var nx = vx / norm
    var ny = vy / norm
    var nz = vz / norm

    # Half-angle form of the axis-angle rotation about z × v = (-ny, nx, 0):
    #   qw = sqrt((1+nz)/2),  (qx, qy) = (-ny, nx) / sqrt(2*(1+nz))
    if nz > Float64(1) - Float64(1e-12):
        # Already +Z.
        return (Float64(0), Float64(0), Float64(0), Float64(1))
    if nz < Float64(-1) + Float64(1e-12):
        # Antiparallel: 180° about X (MuJoCo's degenerate-cross fallback).
        return (Float64(1), Float64(0), Float64(0), Float64(0))

    var denom = _sqrt_f64(Float64(2) * (Float64(1) + nz))
    var qx = -ny / denom
    var qy = nx / denom
    var qz = Float64(0)
    var qw = _sqrt_f64((Float64(1) + nz) * Float64(0.5))
    var qlen = _sqrt_f64(qx * qx + qy * qy + qz * qz + qw * qw)
    if qlen > Float64(1e-10):
        qx = qx / qlen
        qy = qy / qlen
        qw = qw / qlen
    return (qx, qy, qz, qw)


def _euler_to_quat(
    ex: Float64, ey: Float64, ez: Float64, seq: String = "xyz"
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Convert MuJoCo `euler` (radians, in `seq` order) → quaternion (qx,qy,qz,qw).

    Follows `ResolveOrientation` in MuJoCo's `user_objects.cc`: accumulate one
    elemental rotation per character of the sequence, post-multiplying for
    lowercase axes (moving/intrinsic) and pre-multiplying for uppercase
    (fixed/extrinsic). `seq` comes from `<compiler eulerseq="...">`, default
    "xyz".
    """
    var angles = [ex, ey, ez]
    var qw = Float64(1)
    var qx = Float64(0)
    var qy = Float64(0)
    var qz = Float64(0)

    for i in range(3):
        var axis = String(seq[byte = i : i + 1]) if seq.byte_length() > i else ""
        var sc = _sin_cos_f64(angles[i] * Float64(0.5))
        var sa = sc[0]
        var rw = sc[1]
        var rx = Float64(0)
        var ry = Float64(0)
        var rz = Float64(0)
        if axis == "x" or axis == "X":
            rx = sa
        elif axis == "y" or axis == "Y":
            ry = sa
        elif axis == "z" or axis == "Z":
            rz = sa

        var out: Tuple[Float64, Float64, Float64, Float64]
        if axis == "x" or axis == "y" or axis == "z":
            # Moving axes: post-multiply.
            out = _quat_mul(qw, qx, qy, qz, rw, rx, ry, rz)
        else:
            # Fixed axes: pre-multiply.
            out = _quat_mul(rw, rx, ry, rz, qw, qx, qy, qz)
        qw = out[0]
        qx = out[1]
        qy = out[2]
        qz = out[3]

    var qlen = _sqrt_f64(qw * qw + qx * qx + qy * qy + qz * qz)
    if qlen > Float64(1e-10):
        qw = qw / qlen
        qx = qx / qlen
        qy = qy / qlen
        qz = qz / qlen
    return (qx, qy, qz, qw)


def _parse_euler_to_quat(
    s: String,
    deg_factor: Float64 = 1.0,
    seq: String = "xyz",
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Parse MuJoCo euler="ax ay az" → quaternion (qx,qy,qz,qw).

    deg_factor: pass pi/180 when the model uses angle="degree", else 1.0.
    """
    var v = _parse_vec3(s)
    return _euler_to_quat(
        v[0] * deg_factor, v[1] * deg_factor, v[2] * deg_factor, seq
    )


def _parse_zaxis_to_quat(
    s: String,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Parse MuJoCo zaxis="x y z" → quaternion (qx,qy,qz,qw)."""
    var v = _parse_vec3(s)
    return _z2quat(v[0], v[1], v[2])


def _axisangle_to_quat(
    ax: Float64, ay: Float64, az: Float64, angle: Float64
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Convert axis-angle (ax,ay,az,angle_rad) to quaternion (qx,qy,qz,qw).

    Normalises the axis before conversion.
    """
    # Normalise axis
    var len2 = ax * ax + ay * ay + az * az
    var norm = _sqrt_f64(len2)
    var nx = ax
    var ny = ay
    var nz = az
    if norm > Float64(1e-10):
        nx = ax / norm
        ny = ay / norm
        nz = az / norm

    var sc = _sin_cos_f64(angle * Float64(0.5))
    var sin_a = sc[0]
    var cos_a = sc[1]
    return (nx * sin_a, ny * sin_a, nz * sin_a, cos_a)


def _parse_axisangle_to_quat(
    s: String,
    deg_factor: Float64 = 1.0,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Parse MuJoCo axisangle="ax ay az angle" → quaternion (qx,qy,qz,qw).

    deg_factor: pass pi/180 when the model uses angle="degree", else 1.0.
    """
    var parts = List[String]()
    _split_spaces(s, parts)
    var ax = Float64(0)
    var ay = Float64(0)
    var az = Float64(0)
    var angle = Float64(0)
    if len(parts) >= 1:
        ax = _parse_float(parts[0])
    if len(parts) >= 2:
        ay = _parse_float(parts[1])
    if len(parts) >= 3:
        az = _parse_float(parts[2])
    if len(parts) >= 4:
        angle = _parse_float(parts[3]) * deg_factor
    return _axisangle_to_quat(ax, ay, az, angle)


def _fromto_to_pos_quat(
    s: String,
) -> Tuple[
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
]:
    """Parse fromto="x1 y1 z1 x2 y2 z2" → (mid_x,mid_y,mid_z, qx,qy,qz,qw, half_len, radius).

    Returns midpoint, quaternion rotating Z-axis to capsule direction,
    and half_length. Radius is extracted separately from the 'size' attr.
    Returned tuple: (pos_x, pos_y, pos_z, qx, qy, qz, qw, half_length, 0.0)
    where the 9th element is a placeholder (radius comes from size attr).
    """
    var parts = List[String]()
    _split_spaces(s, parts)
    var x1 = Float64(0)
    var y1 = Float64(0)
    var z1 = Float64(0)
    var x2 = Float64(0)
    var y2 = Float64(0)
    var z2 = Float64(1)
    if len(parts) >= 1:
        x1 = _parse_float(parts[0])
    if len(parts) >= 2:
        y1 = _parse_float(parts[1])
    if len(parts) >= 3:
        z1 = _parse_float(parts[2])
    if len(parts) >= 4:
        x2 = _parse_float(parts[3])
    if len(parts) >= 5:
        y2 = _parse_float(parts[4])
    if len(parts) >= 6:
        z2 = _parse_float(parts[5])

    # Midpoint
    var mx = (x1 + x2) * Float64(0.5)
    var my = (y1 + y2) * Float64(0.5)
    var mz = (z1 + z2) * Float64(0.5)

    # Direction vector — FROM minus TO, matching MuJoCo's mjCGeom::Compile
    # (`vec = {fromto[0]-fromto[3], ...}` then `mjuu_z2quat(quat, vec)`).
    #
    # We used `to - from` until 2026-07-29, which points local +Z the other
    # way. For a capsule or cylinder that is the SAME SOLID — flipping the
    # long axis end for end changes nothing about the shape, the inertia
    # tensor or the contact geometry — which is why every FK and inertia gate
    # passed either way. It shows up only when the geom quaternion itself is
    # compared, as `tests/dm_control/test_cheetah_vs_dm_control.mojo` does
    # against `model.geom_quat`. Matching MuJoCo exactly is free here, so do
    # it rather than leave a sign trap for whoever next reads a geom's frame.
    var dx = x1 - x2
    var dy = y1 - y2
    var dz = z1 - z2
    var length = _sqrt_f64(dx * dx + dy * dy + dz * dz)
    var half_length = length * Float64(0.5)

    if length < Float64(1e-10):
        return (
            mx,
            my,
            mz,
            Float64(0),
            Float64(0),
            Float64(0),
            Float64(1),
            half_length,
            Float64(0),
        )

    # Quaternion rotating Z=(0,0,1) onto the capsule direction — the same
    # "minimal rotation from +Z" MuJoCo applies for `zaxis`.
    var q = _z2quat(dx, dy, dz)

    return (mx, my, mz, q[0], q[1], q[2], q[3], half_length, Float64(0))


def _find_body_index_by_name(worldbody: String, body_name: String) -> Int:
    """Return 1-based model body index for <body name="body_name">, or 0 (worldbody).
    """
    var search_name = 'name="' + body_name + '"'
    var count = 0
    var scan_pos = 0
    while True:
        var body_pos = worldbody.find("<body", scan_pos)
        if body_pos == -1:
            return 0
        var tag_end = worldbody.find(">", body_pos)
        if tag_end == -1:
            return 0
        var tag = String(worldbody[byte = body_pos : tag_end + 1])
        count += 1
        if tag.find(search_name) != -1:
            return count
        scan_pos = tag_end + 1


def _find_joint_index_by_name(worldbody: String, joint_name: String) -> Int:
    """0-based index of `<joint name="joint_name">` in MuJoCo order, or -1.

    ⚠ WAS A PLAIN TEXT COUNT, WHICH IS THE WRONG ORDER. `_fill_model` ends by
    grouping `result.joints` by body (`_stable_group_by_body_joints`), and this
    lookup runs AFTER that — so a text ordinal indexed a permuted array. See
    `_index_by_name_grouped`.
    """
    return _index_by_name_grouped(worldbody, "<joint", joint_name)


def _find_site_index_by_name(worldbody: String, site_name: String) -> Int:
    """0-based index of `<site name="site_name">` in MuJoCo order, or -1.

    Added for `<spatial>` tendons, whose waypoints are named site references.

    ⚠ THE OLD DOCSTRING'S CLAIM WAS TRUE WHEN WRITTEN AND STOPPED BEING TRUE.
    It said site indices "are assigned by `_fill_model`'s worldbody walk in
    exactly this order, so counting `<site` tags here reproduces them" — and
    then `_stable_group_by_body_sites` was added to the end of that same walk
    and nobody came back to this. A comment asserting agreement with another
    function is a claim with a shelf life.
    """
    return _index_by_name_grouped(worldbody, "<site", site_name)


def _find_geom_index_by_name(worldbody: String, geom_name: String) -> Int:
    """0-based index of `<geom name="geom_name">` in MuJoCo order, or -1.

    Added for `<contact><pair geom1= geom2=>`, whose two references are named
    geoms.

    Body-grouped for the same reason the joint and site resolvers are:
    `_fill_model` ends with `_stable_group_by_body_geoms(result.geoms)`, so a
    raw text ordinal would index a permuted array. The failure would be quiet
    and total — a pair is a geom-index pair and nothing downstream re-checks it,
    so a mis-resolved index collides two unrelated geoms with the pair's
    parameters and drops the one the model asked for.
    """
    return _index_by_name_grouped(worldbody, "<geom", geom_name)


def _count_joints_with_type(xml: String, joint_type: String) -> Int:
    """Count <joint ... type="joint_type" ...> occurrences.

    Scans each `<joint` tag's content and looks for `type="joint_type"`.
    This avoids false matches on `<geom type="...">` etc.
    """
    var count = 0
    var start = 0
    var type_attr = 'type="' + joint_type + '"'
    while True:
        var pos = xml.find("<joint", start)
        if pos == -1:
            break
        # Find end of this opening tag
        var end_pos = xml.find(">", pos)
        if end_pos == -1:
            break
        var tag_content = String(xml[byte = pos : end_pos + 1])
        if tag_content.find(type_attr) != -1:
            count += 1
        start = end_pos + 1
    return count


# =============================================================================
# Main entry point
# =============================================================================


def _compiler_angle_is_deg(xml: String) -> Bool:
    """Return True when the model's angles are in degrees.

    Value-argument twin of `_xml_compiler_angle_is_deg`; see that docstring
    for why the default is DEGREE. Both exist because some call sites have the
    XML as a comptime parameter and some as a value — but the rule must only
    be written once, which is what let the wrong default sit in four separate
    inline copies of this check.
    """
    var t = xml.find("<compiler")
    if t == -1:
        return True
    var tag_end = xml.find(">", t)
    if tag_end == -1:
        return True
    var tag = String(xml[byte = t : tag_end + 1])
    var angle_val = _trim(_extract_attr(tag, "angle"))
    if angle_val.byte_length() == 0:
        return True
    return angle_val == "degree"


def _compiler_deg_factor(xml: String) -> Float64:
    """Radians-per-unit for the model's angle attributes: pi/180 or 1.0."""
    return Float64(
        3.141592653589793 / 180.0
    ) if _compiler_angle_is_deg(xml) else Float64(1.0)


def _xml_compiler_angle_is_deg[xml: String]() -> Bool:
    """Return True when the model's angles are in degrees. Comptime-safe.

    MuJoCo's MJCF default is `angle="degree"` (`user_init.c`:
    `spec->compiler.degree = 1`; only the URDF loader forces radian), so a
    missing `<compiler>` element — or one without the attribute — means
    DEGREE, not radian.

    Fixed 2026-07-29, same shape as the `inertiafromgeom` default bug. It
    stayed hidden because every Gym-derived env XML in the repo states `angle`
    explicitly; dm_control's walker/cheetah/hopper omit it and state their
    joint ranges in degrees, so walker's ankles came out with a +-45 RADIAN
    range — effectively unlimited.
    """
    var t = xml.find("<compiler")
    if t == -1:
        return True
    var tag_end = xml.find(">", t)
    if tag_end == -1:
        return True
    var tag = String(xml[byte = t : tag_end + 1])
    var angle_val = _trim(_extract_attr(tag, "angle"))
    if angle_val.byte_length() == 0:
        return True
    return angle_val == "degree"


def _xml_compiler_inertiafromgeom[xml: String]() -> Int:
    """Return inertiafromgeom mode. 0=false, 1=true, 2=auto. Comptime-safe.

    MuJoCo's default is "auto" (derive a body's mass/inertia from its geoms
    UNLESS the body carries an explicit <inertial>), so a missing <compiler>
    element — or a <compiler> without the attribute — means auto, NOT false.

    Fixed 2026-07-29: both fell through to 0 (=false), which silently gave
    every body a default inertia. It went unnoticed because all Gym-derived
    env XMLs state `inertiafromgeom="true"` explicitly; the dm_control suite
    XMLs state nothing, and pendulum came out with ~1/21 of its true inertia.
    """
    var t = xml.find("<compiler")
    if t == -1:
        return 2
    var tag_end = xml.find(">", t)
    if tag_end == -1:
        return 2
    var tag = String(xml[byte = t : tag_end + 1])
    var val = _trim(_extract_attr(tag, "inertiafromgeom"))
    if val == "true":
        return 1
    elif val == "auto":
        return 2
    elif val == "false":
        return 0
    return 2


def _xml_compiler_settotalmass[xml: String]() -> Float64:
    """Return settotalmass value from <compiler settotalmass="..."/>. Returns -1.0 if absent. Comptime-safe.
    """
    var t = xml.find("<compiler")
    if t == -1:
        return Float64(-1.0)
    var tag_end = xml.find(">", t)
    if tag_end == -1:
        return Float64(-1.0)
    var tag = String(xml[byte = t : tag_end + 1])
    var val = _extract_attr(tag, "settotalmass")
    var trimmed = _trim(val)
    if trimmed.byte_length() == 0:
        return Float64(-1.0)
    return _parse_float(trimmed)


def _xml_compiler_inertiagrouprange[xml: String]() -> Tuple[Int, Int]:
    """Return (group_min, group_max) from <compiler inertiagrouprange="min max"/>.
    Defaults to (0, 5) if absent. Comptime-safe.
    """
    var t = xml.find("<compiler")
    if t == -1:
        return (0, 5)
    var tag_end = xml.find(">", t)
    if tag_end == -1:
        return (0, 5)
    var tag = String(xml[byte = t : tag_end + 1])
    var val = _extract_attr(tag, "inertiagrouprange")
    var trimmed = _trim(val)
    if trimmed.byte_length() == 0:
        return (0, 5)
    var parts = List[String]()
    _split_spaces(trimmed, parts)
    if len(parts) >= 2:
        return (_parse_int_str(parts[0]), _parse_int_str(parts[1]))
    return (0, 5)


def _xml_default_motor_ctrlrange[xml: String]() -> Tuple[Float64, Float64]:
    """Return (ctrl_min, ctrl_max) from <default><motor ctrlrange="lo hi"/>.
    Defaults to (-1.0, 1.0) if absent. Comptime-safe.
    """
    var def_sec = _root_defaults(xml)
    if def_sec.byte_length() == 0:
        return (-1.0, 1.0)
    var t = def_sec.find("<motor")
    if t == -1:
        return (-1.0, 1.0)
    var tag_end = def_sec.find(">", t)
    if tag_end == -1:
        return (-1.0, 1.0)
    var tag = String(def_sec[byte = t : tag_end + 1])
    var cr = _extract_attr(tag, "ctrlrange")
    if cr.byte_length() == 0:
        return (-1.0, 1.0)
    var parts = List[String]()
    _split_spaces(cr, parts)
    if len(parts) >= 2:
        return (_parse_float(parts[0]), _parse_float(parts[1]))
    return (-1.0, 1.0)


def _xml_default_motor_gear[xml: String]() -> Float64:
    """Return `gear` from `<default><motor gear="..."/>`, else MuJoCo's 1.0.

    The twin of `_xml_default_motor_ctrlrange`, which existed from the start —
    `gear` did not, so a model that put its gear in the default class (the
    dm_control `point_mass` does: `<motor gear=".1" .../>`) silently actuated
    at gear 1.0, a 10x force error with no diagnostic. Found 2026-07-29.

    Both twins now route through `_root_defaults`, which strips the named
    `<default class="...">` blocks — without it a `<motor>` inside a class
    would be applied globally, AND a top-level `<motor>` declared after the
    first class block would be missed entirely. swimmer is the second model to
    pay for that, at 2000x; see `_strip_nested_defaults`.
    """
    var def_sec = _root_defaults(xml)
    if def_sec.byte_length() == 0:
        return Float64(1.0)
    var t = def_sec.find("<motor")
    if t == -1:
        return Float64(1.0)
    var tag_end = def_sec.find(">", t)
    if tag_end == -1:
        return Float64(1.0)
    var tag = String(def_sec[byte = t : tag_end + 1])
    var g = _extract_attr(tag, "gear")
    if g.byte_length() == 0:
        return Float64(1.0)
    return _parse_float(g)


def _xml_nth_fixed_tag[xml: String, n: Int]() -> String:
    """Return the XML tag string for the Nth <fixed> tendon, or empty if absent."""
    var sec = _extract_section(xml, "tendon")
    if sec.byte_length() == 0:
        return ""
    var pos = 0
    for i in range(n + 1):
        var t = sec.find("<fixed", pos)
        if t == -1:
            return ""
        if i == n:
            var end = sec.find("</fixed>", t)
            if end == -1:
                end = sec.find("/>", t)
                if end == -1:
                    return ""
                return String(sec[byte = t : end + 2])
            return String(sec[byte = t : end + 8])
        pos = t + 6
    return ""


def _xml_fixed_tendon_njoints[xml: String, n: Int]() -> Int:
    """Return number of joints in the Nth fixed tendon (0 if absent)."""
    var tag = _xml_nth_fixed_tag[xml, n]()
    if tag.byte_length() == 0:
        return 0
    var count = 0
    var pos = 0
    while True:
        var t = tag.find("<joint", pos)
        if t == -1:
            break
        count += 1
        pos = t + 6
    return count


def _xml_fixed_tendon_joint_name[xml: String, n: Int, j: Int]() -> String:
    """Return the joint name of the Jth joint in the Nth fixed tendon."""
    var tag = _xml_nth_fixed_tag[xml, n]()
    if tag.byte_length() == 0:
        return ""
    var pos = 0
    for i in range(j + 1):
        var t = tag.find("<joint", pos)
        if t == -1:
            return ""
        if i == j:
            var end = tag.find(">", t)
            if end == -1:
                return ""
            var jtag = String(tag[byte = t : end + 1])
            return _extract_attr(jtag, "joint")
        pos = t + 6
    return ""


def _xml_fixed_tendon_coef[xml: String, n: Int, j: Int]() -> Float64:
    """Return the coefficient of the Jth joint in the Nth fixed tendon."""
    var tag = _xml_nth_fixed_tag[xml, n]()
    if tag.byte_length() == 0:
        return 0.0
    var pos = 0
    for i in range(j + 1):
        var t = tag.find("<joint", pos)
        if t == -1:
            return 0.0
        if i == j:
            var end = tag.find(">", t)
            if end == -1:
                return 0.0
            var jtag = String(tag[byte = t : end + 1])
            var cs = _extract_attr(jtag, "coef")
            if cs.byte_length() > 0:
                return _parse_float(cs)
            return 0.0
        pos = t + 6
    return 0.0


# =============================================================================
# merge_mjcf — comptime XML merge following MuJoCo <include> semantics
# =============================================================================


def _is_self_closing(xml: String, tag_start: Int, tag_end: Int) -> Bool:
    """True when `xml[tag_start..tag_end]` is a `<tag ... />` element.

    `tag_end` is the index of the closing `>`. Trailing whitespace between the
    `/` and the `>` is tolerated (`<equality ... / >` is legal XML).
    """
    var i = tag_end - 1
    while i > tag_start:
        var ch = String(xml[byte = i : i + 1])
        if ch == " " or ch == "\n" or ch == "\t" or ch == "\r":
            i -= 1
            continue
        return ch == "/"
    return False


def _extract_section_inner(xml: String, tag: String) -> String:
    """Return the inner content of <tag ...>...</tag>, excluding the outermost tags.

    Handles nested same-name tags (e.g., <default><default class="x">...</default></default>)
    by depth-counting. Handles multiple top-level occurrences by concatenating.

    ⚠ SELF-CLOSING tags of the same name are skipped rather than treated as
    section openers, and are not counted as nested opens. Without that, a
    `<default class="coupling"><equality solimp="..." solref="..."/></default>`
    — MJCF's way of putting equality defaults in a class, which dm_control's
    quadruped uses — made `_extract_section_inner(xml, "equality")` return ""
    for the WHOLE FILE: it latched onto the self-closing `<equality/>` as the
    opener, then never found a matching close because the depth counter had
    incremented on a tag that closes itself. `merge_mjcf` then emitted an empty
    `<equality>` section and the four leg-coupling constraints vanished with no
    diagnostic. This is the same shape as the `<tendon>`-dropped-by-merge_mjcf
    bug of 2026-07-30, in the same function, from a different trigger.
    """
    var result = String("")
    var open_marker = "<" + tag
    var close_marker = "</" + tag + ">"
    var scan = 0
    while True:
        var start = xml.find(open_marker, scan)
        if start == -1:
            break
        # Verify it's a real tag (not a substring match)
        var after_pos = start + open_marker.byte_length()
        if after_pos < xml.byte_length():
            var after_ch = String(xml[byte=after_pos : after_pos + 1])
            if after_ch != " " and after_ch != ">" and after_ch != "/" and after_ch != "\n" and after_ch != "\t":
                scan = after_pos
                continue
        # Find end of opening tag
        var tag_end = xml.find(">", start)
        if tag_end == -1:
            break
        # Self-closing `<tag ... />` opens no section — skip it entirely.
        if _is_self_closing(xml, start, tag_end):
            scan = tag_end + 1
            continue
        var inner_start = tag_end + 1
        # Find matching closing tag (depth-counted)
        var depth = 1
        var search_pos = inner_start
        while depth > 0:
            var next_open = xml.find(open_marker, search_pos)
            var next_close = xml.find(close_marker, search_pos)
            if next_close == -1:
                break
            # Check if next_open is a real tag
            if next_open != -1 and next_open < next_close:
                var np = next_open + open_marker.byte_length()
                if np < xml.byte_length():
                    var nc = String(xml[byte=np : np + 1])
                    if nc == " " or nc == ">" or nc == "/" or nc == "\n" or nc == "\t":
                        # A self-closing nested tag needs no matching close,
                        # so counting it would leave depth permanently high
                        # and swallow the real closing tag.
                        var no_end = xml.find(">", next_open)
                        if no_end == -1 or not _is_self_closing(
                            xml, next_open, no_end
                        ):
                            depth += 1
                search_pos = next_open + open_marker.byte_length()
            else:
                depth -= 1
                if depth == 0:
                    result = result + String(xml[byte=inner_start:next_close]) + "\n"
                    scan = next_close + close_marker.byte_length()
                else:
                    search_pos = next_close + close_marker.byte_length()
        if depth > 0:
            break  # Unmatched tags
    return result


def _strip_nested_defaults(sec: String) -> String:
    """Remove nested `<default class="...">...</default>` sub-blocks.

    Comptime twin of `full_parser._strip_nested_defaults`, which the runtime
    parser has had since 2026-07-29. This side did not, so every lookup below
    that scanned the `<default>` section with a bare `find("<tag")` picked up
    the FIRST NAMED CLASS's element whenever the top-level one was declared
    after it — and MJCF puts no ordering constraint on that.

    dm_control's swimmer is the model that exposes it. Its `<default>` is

        <default>
          <default class="swimmer"> <joint ... limited="true" .../> ... </default>
          <default class="free">    <joint limited="false" .../>       </default>
          <motor gear="5e-4" ctrllimited="true" ctrlrange="-1 1"/>
        </default>

    so the top-level `<motor>` comes LAST. `_extract_section` is not depth
    aware either, so it used to hand back a section truncated at the first
    inner `</default>` — with no `<motor>` in it at all. Gear silently fell
    back to MuJoCo's 1.0 against an actual 5e-4: a 2000x actuator force error,
    which is the whole dynamics of the domain. The same truncation made
    `def_limited` read the swimmer class's `limited="true"`, marking the three
    unlimited root DOFs as limited with an empty (0, 0) range.

    Nesting is depth-tracked so a class containing sub-classes is removed
    whole (swimmer's `class="swimmer"` contains `inertial` and `visual`).
    """
    var out = String("")
    var i = 0
    var n = sec.byte_length()
    while i < n:
        var open_t = sec.find("<default", i)
        if open_t == -1:
            out += String(sec[byte=i:n])
            break
        out += String(sec[byte=i:open_t])
        # Walk forward to this block's matching </default>.
        var depth_ = 0
        var j = open_t
        while j < n:
            var next_open = sec.find("<default", j + 1)
            var next_close = sec.find("</default>", j + 1)
            if next_close == -1:
                j = n
                break
            if next_open != -1 and next_open < next_close:
                depth_ += 1
                j = next_open
                continue
            if depth_ == 0:
                j = next_close + 10  # len("</default>")
                break
            depth_ -= 1
            j = next_close
        i = j
    return out


def _root_defaults(xml: String) -> String:
    """The TOP-LEVEL `<default>` content only — named classes stripped.

    Every `<default>` lookup in this file must go through here rather than
    `_extract_section(xml, "default")`; see `_strip_nested_defaults` for the
    2000x actuator error the bare version cost.
    """
    return _strip_nested_defaults(_extract_section_inner(xml, "default"))


def _class_attr(
    xml: String, cls: String, tag_name: String, attr: String
) -> String:
    """`attr` of the first `<tag_name>` directly inside `<default class="cls">`.

    The counterpart to `_root_defaults` for NAMED classes. Until quadruped
    nothing on this side of the parser needed one: `_root_defaults` exists
    precisely to keep class blocks from leaking into the global lookups, and
    every earlier model put its actuator attributes at the top level.

    quadruped does not. Its twelve actuators carry nothing but a name, a
    transmission and `class="yaw_act"` / `"lift_act"` / `"extend_act"`, and
    each of those classes supplies exactly one attribute (`ctrlrange`) on top
    of the top-level `<general>` default that supplies all the rest. Reading
    only root defaults would give all twelve the same ctrlrange of (-1, 1),
    which is right for four of them and wrong for eight.

    ⚠ WHY THIS IS ONE FUNCTION AND NOT THREE COMPOSED ONES. The obvious
    factoring — return the class section, pull the tag out of it, then
    `_extract_attr` that tag — DOES NOT COMPILE. Slicing a `String` that was
    itself built by slicing another `String` defeats the comptime interpreter:
    `String(tag[byte=a:b])` fails with "interpreting memcpy can't get dst
    memory from the interpreter / write clobbers a pointer region". The
    failure is selective in a way that makes it easy to misread — a lookup
    that MISSES in the intermediate string is fine, and only one that HITS
    (and therefore reaches the slice) fails, so seven of eight attribute
    lookups compiled happily. Everything here is therefore index arithmetic
    over the ORIGINAL `xml`, with exactly one slice at the end.

    Elements inside a NESTED `<default class="...">` are skipped, so a class
    that contains sub-classes resolves to its own child rather than theirs.

    An EMPTY `cls` means the top-level `<default>` block — the one with no
    `class` attribute. That is the terminator of the inheritance chain in
    `_class_attr_inherited`, and doing it here rather than via
    `_root_defaults` is not a style choice: `_root_defaults` returns a String
    built by SLICING, and slicing that again is precisely the comptime failure
    this docstring warns about above. Index arithmetic over the original
    `xml`, one slice at the end, is the only shape that survives.
    """
    var n = xml.byte_length()
    var scan = 0
    while scan < n:
        var t = xml.find("<default", scan)
        if t == -1:
            return String("")
        var te = xml.find(">", t)
        if te == -1:
            return String("")
        if _trim(_extract_attr(String(xml[byte = t : te + 1]), "class")) != cls:
            scan = te + 1
            continue

        # This block's inner span, as indices into `xml`.
        var inner = te + 1
        var depth = 0
        var j = inner
        var stop = -1
        while j < n:
            var no = xml.find("<default", j)
            var nc = xml.find("</default>", j)
            if nc == -1:
                break
            if no != -1 and no < nc:
                depth += 1
                j = no + 8  # len("<default")
                continue
            if depth == 0:
                stop = nc
                break
            depth -= 1
            j = nc + 10  # len("</default>")
        if stop < 0:
            return String("")

        # First `<tag_name>` at depth 0 within [inner, stop).
        var marker = "<" + tag_name
        var p = inner
        while p < stop:
            var tt = _find_tag(xml, marker, p)
            if tt == -1 or tt >= stop:
                return String("")
            # Depth of `tt` relative to `inner`: count nested opens before it.
            var d = 0
            var k = inner
            while k < tt:
                var o2 = xml.find("<default", k)
                var c2 = xml.find("</default>", k)
                if o2 != -1 and o2 < tt and (c2 == -1 or o2 < c2):
                    d += 1
                    k = o2 + 8
                    continue
                if c2 != -1 and c2 < tt:
                    d -= 1
                    k = c2 + 10
                    continue
                break
            if d != 0:
                p = tt + 1
                continue
            var tte = xml.find(">", tt)
            if tte == -1 or tte > stop:
                return String("")
            return _extract_attr(String(xml[byte = tt : tte + 1]), attr)
        return String("")
    return String("")


def _class_parent(xml: String, cls: String) -> String:
    """The class enclosing `<default class="cls">`, or "" if it is top level.

    MJCF default classes NEST and INHERIT, which `_class_attr` alone does not
    express — it answers "what does this exact block say", not "what does an
    element in this class end up with". quadruped's legs need the difference:

        <default>
          <default class="body">                     <- type, size, material
            <default class="hip">  <geom fromto=.../>   <- fromto only
            <default class="knee"> ...

    and a leg geom is the bare tag `<geom name="thigh_front_left"/>` under a
    body carrying `childclass="hip"`. Its `type` lives two levels up. Walking
    parents is the only way to reach it.

    ⚠ Index arithmetic over the ORIGINAL `xml`, with single slices, for the
    reason spelled out at length in `_class_attr`: slicing a String that was
    itself produced by slicing another String defeats the comptime
    interpreter, and it fails only on the paths that HIT.
    """
    if cls.byte_length() == 0:
        return String("")
    var n = xml.byte_length()
    # Spans of the currently-open `<default ...>` tags, outermost first.
    var open_start = InlineArray[Int, 32](fill=-1)
    var open_end = InlineArray[Int, 32](fill=-1)
    var depth = 0
    var i = 0
    while i < n:
        var t = xml.find("<default", i)
        var c = xml.find("</default>", i)
        if t == -1 and c == -1:
            break
        if t != -1 and (c == -1 or t < c):
            var te = xml.find(">", t)
            if te == -1:
                break
            if depth < 32:
                open_start[depth] = t
                open_end[depth] = te
            depth += 1
            if (
                _trim(_extract_attr(String(xml[byte = t : te + 1]), "class"))
                == cls
            ):
                # Enclosing block is one level out; the top-level `<default>`
                # carries no class, so its name comes back "" — exactly the
                # terminator the caller wants.
                if depth >= 2 and open_start[depth - 2] >= 0:
                    return _trim(
                        _extract_attr(
                            String(
                                xml[
                                    byte = open_start[depth - 2] : open_end[
                                        depth - 2
                                    ]
                                    + 1
                                ]
                            ),
                            "class",
                        )
                    )
                return String("")
            i = te + 1
        else:
            if depth > 0:
                depth -= 1
            i = c + 10  # len("</default>")
    return String("")


def _class_attr_inherited(
    xml: String, cls: String, tag_name: String, attr: String
) -> String:
    """`attr` for an element of class `cls`, walking the default-class chain.

    Own class first, then each enclosing class, then the top-level `<default>`
    block — MJCF's resolution order. Returns "" if nothing in the chain sets
    it, which lets the caller keep its own fallback.

    This is what `parse_xml_render_data` was missing. It read geom attributes
    only off the geom's own tag, so every geom that inherited its `type` from
    a class fell back to the MJCF default of SPHERE — quadruped's legs became
    default-radius spheres parked at their body origins, i.e. invisible inside
    the torso. Ten of sixteen dm_control domains put geom type/size/fromto in
    a `<default>` block, so this was most of the suite, and no test could see
    it: the PHYSICS reads the runtime parser (`full_parser`), which has
    resolved classes correctly since 2026-07-29.
    """
    var c = cls
    # Bounded: a class chain deeper than 16 is a malformed model, and an
    # unbounded `while` here would be a comptime hazard of its own.
    for _ in range(16):
        if c.byte_length() == 0:
            break
        var v = _class_attr(xml, c, tag_name, attr)
        if v.byte_length() > 0:
            return v
        c = _class_parent(xml, c)
    # Chain exhausted.
    #
    # ⚠ THE TOP-LEVEL `<default>` BLOCK IS DELIBERATELY NOT CONSULTED for the
    # render attributes. Routing the tail through `_class_attr(xml, "", ...)`
    # is correct MJCF and it works for a single model, but it tips
    # `parse_xml_render_data` over the comptime interpreter's budget once a
    # dozen-plus models are instantiated in one binary — that function's own
    # docstring says it exists to dodge exactly such a crash. The named-class
    # chain is where the real defect lived (quadruped's legs inherit `type`
    # from `class="body"`), and a ROOT-level `<geom type=.../>` is rare: no
    # dm_control model sets type, size or fromto there, only solimp/solref,
    # which rendering ignores. Colour still falls back to root via the
    # `def_rgba_*` path in the geom loop, as it always did.
    #
    # If a model ever needs it, the fix is not to re-add the call but to make
    # the resolution cheaper — resolve every class ONCE into a table before
    # the worldbody scan instead of lazily per class. `_DefaultsIndex` below
    # is that table, and `_class_attr_inherited_indexed` is this function over
    # it; wiring `parse_xml_render_data` to them is the prerequisite for
    # reinstating the root-level tail. ⚠ Measured 2026-08-12: routing the
    # render path through the index changes its build time by NOTHING (28 s
    # either way), so it stays on the rescanning path here and the two
    # implementations are diffed against each other in
    # `tests/physics3d/test_defaults_index_equivalence.mojo`.
    return String("")


struct _DefaultsIndex(Copyable, Movable):
    """Every `<default ...>` block in the XML, located in ONE pass.

    `_class_attr` and `_class_parent` each re-scan the whole document per
    lookup, and `_class_attr_inherited` calls both once per link of the
    inheritance chain. dog resolves ~190 distinct (class, kind, attribute)
    triples over a chain two to three deep, so that is on the order of a
    thousand walks of an 80 KB string — in the comptime interpreter, where
    every `<default` tag encountered allocates a String.

    This holds each block as a pair of INDICES INTO THE ORIGINAL `xml` rather
    than as extracted text. That is not a micro-optimisation: `_class_attr`'s
    docstring documents at length that slicing a String which was itself
    produced by slicing another String defeats the comptime interpreter, and
    that it fails only on the paths that HIT — so a table of block TEXT would
    compile clean and then break the first time a class actually resolved.
    Index arithmetic over the original `xml`, one slice at the end, is the
    only shape that survives.

    Blocks appear in the order their opening tags appear in the text, which is
    what makes `find` below reproduce the original "first match wins" rule.
    """

    var cls: List[String]
    """Trimmed `class` attribute; "" for the top-level `<default>` block."""
    var inner: List[Int]
    """First byte after the opening tag's `>`."""
    var stop: List[Int]
    """Index of the matching `</default>`, or -1 if there is none."""
    var parent: List[Int]
    """Index of the enclosing block, -1 when this one is top level."""

    def __init__(out self):
        self.cls = List[String]()
        self.inner = List[Int]()
        self.stop = List[Int]()
        self.parent = List[Int]()

    def find(self, cls: String) -> Int:
        """The FIRST block whose class is `cls`, or -1.

        First-match, not last: `_class_attr` scanned forward from byte 0 and
        took the first `<default>` whose class matched, so a model that
        declares the same class name twice must keep resolving to the earlier
        block.
        """
        for i in range(len(self.cls)):
            if self.cls[i] == cls:
                return i
        return -1

    def innermost(self, at: Int) -> Int:
        """The deepest block containing byte `at`, or -1 if none does.

        Blocks are stored in opening-tag order, so among those that contain
        `at` the deepest is the LAST — a nested block always opens after its
        parent. This replaces the open/close counting `_class_attr` did from
        the block start to each candidate tag.
        """
        var best = -1
        for j in range(len(self.cls)):
            if self.stop[j] >= 0 and self.inner[j] <= at and at < self.stop[j]:
                best = j
        return best


def _build_defaults_index(xml: String) -> _DefaultsIndex:
    """Locate every `<default>` block and its parent in one forward scan.

    ⚠ THE SPAN LOOP IS TRANSCRIBED FROM `_class_attr`, and the parent stack
    from `_class_parent`, deliberately — including where they are loose. Both
    match the bare text `"<default"`, which also fires inside a comment, and
    both give a self-closing `<default class="x"/>` a span running to the next
    `</default>`, swallowing its following siblings. Those quirks decide which
    block a class resolves to, so reproducing them is the difference between a
    memoization and a silent reparse of the model. MJCF class resolution is
    where defect `ab219882` lived (dog's gains came out between 25x and 3000x
    too weak, invisible outside a driven rollout), so this stays faithful and
    the gates stay the judge.
    """
    var out = _DefaultsIndex()
    var n = xml.byte_length()
    # Indices of the blocks currently open at this point in the scan.
    var stack = List[Int]()
    var scan = 0
    while scan < n:
        var t = xml.find("<default", scan)
        if t == -1:
            break
        var te = xml.find(">", t)
        if te == -1:
            break

        # Matching `</default>`, skipping nested pairs — `_class_attr`'s loop.
        var inner = te + 1
        var depth = 0
        var j = inner
        var stop = -1
        while j < n:
            var no = xml.find("<default", j)
            var nc = xml.find("</default>", j)
            if nc == -1:
                break
            if no != -1 and no < nc:
                depth += 1
                j = no + 8  # len("<default")
                continue
            if depth == 0:
                stop = nc
                break
            depth -= 1
            j = nc + 10  # len("</default>")

        # Anything on the stack that ended before this tag is no longer open.
        while len(stack) > 0:
            var top = stack[len(stack) - 1]
            if out.stop[top] >= 0 and out.stop[top] > t:
                break
            _ = stack.pop()

        out.cls.append(
            _trim(_extract_attr(String(xml[byte = t : te + 1]), "class"))
        )
        out.inner.append(inner)
        out.stop.append(stop)
        out.parent.append(stack[len(stack) - 1] if len(stack) > 0 else -1)
        stack.append(len(out.cls) - 1)
        scan = te + 1
    return out^


def _class_attr_indexed(
    xml: String, idx: _DefaultsIndex, ci: Int, tag_name: String, attr: String
) -> String:
    """`_class_attr` against the prebuilt index. Same answer, no rescan."""
    if ci < 0:
        return String("")
    var stop = idx.stop[ci]
    if stop < 0:
        return String("")
    var marker = "<" + tag_name
    var p = idx.inner[ci]
    while p < stop:
        var tt = _find_tag(xml, marker, p)
        if tt == -1 or tt >= stop:
            return String("")
        # `_class_attr` accepted the tag only at depth 0 relative to the block,
        # i.e. not inside a nested `<default>`; that is exactly "the innermost
        # block containing it is this one".
        if idx.innermost(tt) == ci:
            var tte = xml.find(">", tt)
            if tte == -1 or tte > stop:
                return String("")
            return _extract_attr(String(xml[byte = tt : tte + 1]), attr)
        p = tt + 1
    return String("")


def _class_parent_indexed(idx: _DefaultsIndex, cls: String) -> String:
    """`_class_parent` against the prebuilt index."""
    if cls.byte_length() == 0:
        return String("")
    var ci = idx.find(cls)
    if ci < 0:
        return String("")
    var pi = idx.parent[ci]
    if pi < 0:
        return String("")
    return idx.cls[pi]


def _class_attr_inherited_indexed(
    xml: String, idx: _DefaultsIndex, cls: String, tag_name: String, attr: String
) -> String:
    """`_class_attr_inherited` against the prebuilt index.

    The chain is walked BY NAME, not by parent index, because the original
    did: it took `_class_parent`'s answer and fed it back to `_class_attr`,
    which re-resolves the name from the top. With a duplicated class name the
    two disagree, and matching the original is the point.

    The 16-link bound and the deliberate omission of the top-level `<default>`
    block are the original's; see `_class_attr_inherited` for why the tail is
    not routed through the root.
    """
    var c = cls
    for _ in range(16):
        if c.byte_length() == 0:
            break
        var v = _class_attr_indexed(xml, idx, idx.find(c), tag_name, attr)
        if v.byte_length() > 0:
            return v
        c = _class_parent_indexed(idx, c)
    return String("")


def _first_tag(sec: String, tag_name: String) -> String:
    """The first `<tag_name ...>` element in `sec`, or empty."""
    if sec.byte_length() == 0:
        return String("")
    var t = _find_tag(sec, "<" + tag_name, 0)
    if t == -1:
        return String("")
    var te = sec.find(">", t)
    if te == -1:
        return String("")
    return String(sec[byte = t : te + 1])


def _attr_3way(
    xml: String,
    elem: String,
    cls: String,
    tag_name: String,
    root_tag: String,
    name: String,
) -> String:
    """MJCF attribute resolution: element, then its class, then the top level.

    Returns "" when none of the three carries `name`, leaving the caller to
    apply MuJoCo's own documented default. The class level goes through
    `_class_attr_inherited`, which re-walks `xml` per lookup rather than
    caching a class section — see the warning in `_class_attr` for why the
    cached form cannot compile.

    ⚠ THE CLASS LEVEL MUST WALK THE PARENT CHAIN, not just the named class.
    This called `_class_attr`, which reads ONLY the class's own block, so an
    attribute set on an ENCLOSING class was invisible and resolution fell
    straight through to the root default.

    dm_control's dog is the model that exposes it: all 38 actuators are
    `<general dyntype="filter">` whose force IS `gainprm[0] * act`, and their
    gains live in nested `<default>` classes — `lumbar` sets 40, a class nested
    inside it sets 60, `finger` sets 2 — over a root default of 0.02. MuJoCo
    compiles NINE distinct gains, `{0.5, 2, 3, 10, 14, 20, 30, 40, 60}`; we
    compiled 0.02 for every one, i.e. between 25x and 3000x too weak.

    ⚠ WHY IT HID FOR SO LONG. The gain scales the FORCE, not the activation,
    so `act` still matched MuJoCo exactly (`|d(act)| = 0.0`) and every
    zero-actuation comparison was exact — the whole solve at 2.99e-11, an
    applied force on every dof at 3.41e-11, five env steps at 1e-13. Only a
    DRIVEN rollout could see it, and `0.02 * 0 == 40 * 0` is why the staged
    probe never could. Gated by `tests/dm_control/test_dog_actuator_gain.mojo`.

    `_class_attr_inherited` stops before the top-level `<default>` block on
    purpose (see its docstring); that is exactly right here, because
    `root_tag` below IS that block and consulting it twice would be wasted
    work in a comptime-budget-sensitive path.
    """
    var v = _extract_attr(elem, name)
    if v.byte_length() > 0:
        return v
    var c = _class_attr_inherited(xml, cls, tag_name, name)
    if c.byte_length() > 0:
        return c
    return _extract_attr(root_tag, name)


def _nth_float(s: String, n: Int, fallback: Float64) -> Float64:
    """`n`-th whitespace-separated float of `s`, or `fallback`."""
    if s.byte_length() == 0:
        return fallback
    var parts = List[String]()
    _split_spaces(s, parts)
    if n >= len(parts):
        return fallback
    return _parse_float(parts[n])


def _extract_singleton_tag(xml: String, tag: String) -> String:
    """Extract a self-closing singleton tag like <option .../> or <compiler .../>.

    Returns the full tag string (including < and >) or empty if not found.
    """
    var marker = "<" + tag
    var pos = xml.find(marker)
    if pos == -1:
        return String("")
    var end = xml.find(">", pos)
    if end == -1:
        return String("")
    return String(xml[byte=pos : end + 1])


def _merge_singleton_attrs(tags: List[String], tag_name: String) -> String:
    """Merge attributes from multiple singleton tags. Last value wins per attr.

    Input: list of tag strings like ['<option a="1" b="2"/>', '<option b="3"/>']
    Output: '<option a="1" b="3"/>'
    """
    # Collect all unique attribute names and their last values
    var attr_names = List[String]()
    var attr_values = List[String]()

    for t_idx in range(len(tags)):
        var tag = tags[t_idx]
        if tag.byte_length() == 0:
            continue
        # Find the attributes region (after tag name, before > or />)
        var space = tag.find(" ")
        if space == -1:
            continue
        var end = tag.find("/>")
        if end == -1:
            end = tag.find(">")
        if end == -1:
            continue
        var attrs_str = String(tag[byte=space:end])

        # Parse attr="value" pairs
        var scan = 0
        var alen = attrs_str.byte_length()
        while scan < alen:
            var eq = attrs_str.find("=", scan)
            if eq == -1:
                break
            # Find attr name (walk back from = to find start)
            var name_end = eq
            var name_start = name_end - 1
            while name_start >= 0:
                var ch = String(attrs_str[byte=name_start:name_start + 1])
                if ch == " " or ch == "\n" or ch == "\t":
                    break
                name_start -= 1
            name_start += 1
            var attr_name = _trim(String(attrs_str[byte=name_start:name_end]))

            # Find value (between quotes)
            var q1 = attrs_str.find('"', eq + 1)
            if q1 == -1:
                q1 = attrs_str.find("'", eq + 1)
            if q1 == -1:
                break
            var quote_char = String(attrs_str[byte=q1:q1 + 1])
            var q2 = attrs_str.find(quote_char, q1 + 1)
            if q2 == -1:
                break
            var attr_val = String(attrs_str[byte=q1 + 1 : q2])

            # Update or add
            var found = False
            for i in range(len(attr_names)):
                if attr_names[i] == attr_name:
                    attr_values[i] = attr_val
                    found = True
                    break
            if not found:
                attr_names.append(attr_name)
                attr_values.append(attr_val)

            scan = eq + (q2 - eq) + 1

    if len(attr_names) == 0:
        return String("")

    var result = "<" + tag_name
    for i in range(len(attr_names)):
        result = result + ' ' + attr_names[i] + '="' + attr_values[i] + '"'
    result = result + "/>"
    return result


def _strip_wrapper(xml: String) -> String:
    """Strip <mujoco> or <mujocoinclude> wrapper, returning inner content."""
    var result = xml

    # Strip <mujocoinclude>...</mujocoinclude>
    var mci_open = result.find("<mujocoinclude")
    if mci_open != -1:
        var mci_open_end = result.find(">", mci_open)
        if mci_open_end != -1:
            var mci_close = result.find("</mujocoinclude>")
            if mci_close != -1:
                # Temporary first — see `_strip_comments` above.
                var inner = String(result[byte=mci_open_end + 1 : mci_close])
                result = inner^

    # Strip <mujoco>...</mujoco>
    var mj_open = result.find("<mujoco")
    if mj_open != -1:
        var mj_open_end = result.find(">", mj_open)
        if mj_open_end != -1:
            var mj_close = result.find("</mujoco>")
            if mj_close != -1:
                var inner = String(result[byte=mj_open_end + 1 : mj_close])
                result = inner^

    return result


def _normalize_freejoint(xml: String) -> String:
    """Rewrite `<freejoint .../>` as `<joint type="free" .../>`.

    MJCF accepts both spellings for a 6-DOF root; MuJoCo's compiler treats
    `<freejoint>` as sugar. Our scanners look for the literal `"<joint"` in
    roughly twenty places, so supporting the alias at each of them would be
    both invasive and easy to miss one of. Normalizing the TEXT once, before
    anything scans it, covers every site at a stroke.

    This matters because the failure was silent: an unrecognized `<freejoint>`
    is not an error, it simply yields a model with no root joint — the body
    welds to the world and nq/nv come out 7/6 short, which then shows up as a
    dimension mismatch far from the cause. In-scope users are dm_control's
    humanoid and quadruped (dog and humanoid_CMU are descoped).

    EVERY ATTRIBUTE A CLASS COULD SUPPLY IS PINNED, and that is the whole
    point of the distinction MuJoCo draws between the two spellings. Its docs
    say of `<freejoint>`: "The alternative is to set type='free' in a regular
    joint element, but then the joint will inherit any defaults defined for
    joints, which is usually undesirable." The compiler implements that
    literally — `xml_native_reader.cc:3570` calls `mjs_addFreeJoint(body)`,
    whose comment reads "create free joint without defaults", so the joint
    keeps the values `mjs_defaultJoint` memset in (`user_init.c:96`) no matter
    what the enclosing class says. A bare `<joint type="free">` under
    humanoid's `<default class="body"><joint armature=".01" damping=".2"
    stiffness="1" limited="true"/>` would give the ROOT an armature, a damper,
    a spring pulling it toward the origin, and a limit — MuJoCo reports 0 for
    all of them. Writing the defaults out explicitly reproduces that, because
    an attribute on the element beats the class.

    ⚠ THE LIST HAS TO BE COMPLETE, not just the passive scalars. It used to
    stop at armature/damping/stiffness/springref/frictionloss, so quadruped's
    root inherited `solimplimit="0 .99 .01"` from `<default class="body">` and
    reported `jnt_solimp[0] = (0, .99, .01, .5, 2)` where MuJoCo reports the
    global `(.9, .95, .001, .5, 2)`. That one is INERT — a free joint is never
    `limited`, so no limit row is ever built and nothing reads its solimp —
    but `ref` under the same class would not have been, and the omission was
    only ever going to be found by a model-constant diff. The solref/solimp
    numbers are `mj_defaultSolRefImp` (`engine_init.c:32`).

    Other attributes are carried through untouched; the injected ones go
    immediately after the tag name, which is safe because `<freejoint>` admits
    only name/group/align, and `_extract_attr` takes the first match anyway.
    `ref` precedes `springref` for the same reason, though `_extract_attr`
    requires a separator before the name and so would not confuse them.
    """
    var result = String("")
    var scan = 0
    var xlen = xml.byte_length()
    while scan < xlen:
        var fj = xml.find("<freejoint", scan)
        if fj == -1:
            result = result + String(xml[byte=scan:xlen])
            break
        result = (
            result
            + String(xml[byte=scan:fj])
            + '<joint type="free" limited="false" armature="0" damping="0"'
            + ' stiffness="0" ref="0" springref="0" frictionloss="0"'
            + ' range="0 0" margin="0" solreflimit="0.02 1"'
            + ' solimplimit="0.9 0.95 0.001 0.5 2"'
        )
        scan = fj + 10  # len("<freejoint")
    return result


def _strip_include_tags(xml: String) -> String:
    """Remove all <include file="..."/> tags from XML."""
    var result = String("")
    var scan = 0
    var xlen = xml.byte_length()
    while scan < xlen:
        var inc = xml.find("<include", scan)
        if inc == -1:
            result = result + String(xml[byte=scan:xlen])
            break
        result = result + String(xml[byte=scan:inc])
        var inc_end = xml.find(">", inc)
        if inc_end == -1:
            break
        # Check for /> vs >
        scan = inc_end + 1
    return result


def _dedupe_last_wins(inner: String) -> String:
    """Keep only the LAST element of each name in `inner`, in document order.

    `<visual>`'s children (`global`, `quality`, `headlight`, `map`, `scale`,
    `rgba`) are SINGLETONS in the MJCF schema, but `merge_mjcf` treats
    `<visual>` as an accumulator and concatenates every input's children. A
    model that declares its own `<visual><map .../></visual>` on top of the
    shared `common/visual.xml` therefore produced two `<map>` elements, and
    MuJoCo rejects that outright:

        XML Error: Schema violation: unique element 'map' found 2 times

    Nothing in OUR engine reads `<visual>`, so the merged models ran fine and
    this stayed invisible until quadruped — the first merged model with its
    own `<visual>` — needed to be loaded BY MUJOCO for a parity gate. An MJCF
    we cannot round-trip into MuJoCo is an MJCF we cannot gate.

    Last-wins matches `<include>` ordering (the model's own section comes
    after the shared one). It is element replacement, not MuJoCo's
    attribute-wise merge: an attribute set only by the EARLIER element is
    dropped rather than kept. That is exact for every current caller (the
    shared `<map znear>` is a strict subset of quadruped's `<map znear zfar>`)
    and stays cosmetic regardless, since only the renderer reads these.

    Bails out unchanged on anything that is not a flat list of self-closing
    elements — better today's invalid XML than a silently mangled section.
    """
    var starts = List[Int]()
    var ends = List[Int]()
    var name_starts = List[Int]()
    var name_ends = List[Int]()

    var i = 0
    var n = inner.byte_length()
    # BOUNDED, not `while i < n`. An unbounded data-dependent loop inside a
    # nested comptime callee is a known compile-time explosion in this tree
    # (see sensors/subtree.mojo's `walk_to_root` for the reproducer). Each
    # iteration consumes at least one byte, so `n` is an exact bound.
    for _ in range(n):
        if i >= n:
            break
        var t = inner.find("<", i)
        if t == -1:
            break
        if t + 4 <= n and String(inner[byte = t : t + 4]) == "<!--":
            var c = inner.find("-->", t)
            if c == -1:
                return inner
            i = c + 3
            continue
        var te = inner.find(">", t)
        if te == -1:
            return inner
        if not _is_self_closing(inner, t, te):
            return inner
        var ns = t + 1
        var ne = ns
        while ne < te:
            var ch = String(inner[byte = ne : ne + 1])
            if ch == " " or ch == "\n" or ch == "\t" or ch == "\r" or ch == "/":
                break
            ne += 1
        if ne == ns:
            return inner
        starts.append(t)
        ends.append(te + 1)
        name_starts.append(ns)
        name_ends.append(ne)
        i = te + 1

    var out = String("")
    for a in range(len(starts)):
        var la = name_ends[a] - name_starts[a]
        var dup = False
        for b in range(a + 1, len(starts)):
            if name_ends[b] - name_starts[b] != la:
                continue
            var same = True
            for k in range(la):
                var ca = String(
                    inner[byte = name_starts[a] + k : name_starts[a] + k + 1]
                )
                var cb = String(
                    inner[byte = name_starts[b] + k : name_starts[b] + k + 1]
                )
                if ca != cb:
                    same = False
                    break
            if same:
                dup = True
                break
        if not dup:
            out = out + "    " + String(inner[byte = starts[a] : ends[a]]) + "\n"
    return out


def merge_mjcf(*xmls: String) -> String:
    """Merge multiple MJCF XML strings following MuJoCo <include> semantics.

    Singleton tags (<option>, <compiler>): attributes merged, last wins per attr.
    Accumulator tags — inner content concatenated from all inputs:
    <asset>, <default>, <worldbody>, <tendon>, <actuator>, <equality>,
    <visual>, <sensor>, <contact>, <keyframe>, and <option>'s <flag> children.

    ⚠ ANYTHING NOT IN THAT LIST IS SILENTLY DROPPED, with no diagnostic. No
    section is dropped deliberately any more — <contact> was, on the stale
    grounds of "no exclude/pair support yet", until 2026-08-03; see the note
    at `all_contact` below. `<pair>` inside it was likewise unparsed until
    2026-08-12; `full_parser._fill_pairs` reads it now, and carrying the
    section here is what makes that work for an included model — Menagerie
    declares its pairs in `scene.xml` and the geoms they name in the robot
    file it includes.

    <sensor> was in that dropped list until 2026-07-31. Our parser ignores the
    section either way — the ported configs read the underlying fields through
    physics3d/sensors/ — but dropping it made the merged XML UNLOADABLE as a
    reference: MuJoCo built a model with nsensor == 0, so no parity gate could
    ask it what a sensor should read. Accumulating it costs nothing and makes
    the merged text a faithful copy of what the model declares.

    <keyframe> joined on 2026-08-13, for ToddlerBot — which declares
    `<key name="home">` in the INCLUDED robot file while the scene is what
    gets loaded, so the section never survived the merge. Its qpos differs
    from qpos0 in 26 of 51 slots by up to 1.5708 rad, and nothing raised.

    This list used to CLAIM <sensor> was accumulated while omitting <tendon>,
    which was both wrong and the reason a dropped `<tendon>` went unnoticed
    until fish needed one. Keep it honest: if you add an accumulator, add it
    here; if a section is dropped on purpose, say so.

    Each input can be a full <mujoco>...</mujoco> or a <mujocoinclude> fragment.
    <include file="..."/> tags are stripped (already resolved by caller).

    Usage:
        comptime xml = merge_mjcf(basic_scene, xyz_deps, xyz_base, task_xml)
        comptime pm = parse_xml(xml)

    Returns a complete <mujoco>...</mujoco> string ready for parse_xml.
    """
    # Collect singleton tags and accumulator content from all inputs
    var option_tags = List[String]()
    var compiler_tags = List[String]()
    var all_assets = String("")
    var all_defaults = String("")
    var all_worldbody = String("")
    var all_actuator = String("")
    var all_equality = String("")
    var all_visual = String("")
    # <tendon> was missing from this list until 2026-07-30, so the whole
    # section was DROPPED from every merged model. Latent for a long time
    # because the only other merged model with tendons is dm_control's
    # point_mass, which deliberately rewrites its two identity-coefficient
    # fixed tendons as plain joint motors. fish is the first model that needs
    # them for real, and lost BOTH: the `fins_flap` actuator's tendon
    # transmission (which then failed the G3 transmission guard loudly) and
    # the `fins_sym` passive spring (which would have failed NOTHING — a
    # missing passive force is just a slightly different fish).
    var all_tendon = String("")
    # Carried for the MuJoCo side of parity gates (see the docstring); our own
    # parser never looks at it.
    var all_sensor = String("")
    # <keyframe> — ToddlerBot declares `<key name="home">` in the INCLUDED
    # robot file, not the scene, so before this the section was dropped by the
    # merge before any parser could see it. Silent, like every other dropped
    # section: a model with no keyframe is a model that resets to qpos0.
    var all_keyframe = String("")
    # <contact> joined the accumulators on 2026-08-03, for humanoid_CMU. The
    # docstring above had said "no exclude/pair support yet" since the function
    # was written, and by then that was FALSE at both ends: `full_parser`
    # `_fill_excludes` populates the record and `contact_detection` skips the
    # excluded pair (`MODEL_META_IDX_NEXCLUDE`). Only the merge was missing, so
    # a merged model reported `nexclude == 0` against MuJoCo's real count and
    # collided bodies MuJoCo never collides — silently, since a dropped section
    # is not an error. This is the THIRD section dropped this way (<tendon>,
    # then <option>'s <flag> children, now <contact>); the pattern each time is
    # a stale claim in the docstring outliving the limitation that justified it.
    # `<pair>` inside the section was ALSO silently ignored until 2026-08-12,
    # when `_fill_pairs` and the three detection loops landed. Carrying the
    # text here is what makes that work for a merged model, which is the case
    # that matters: Menagerie declares its pairs in `scene.xml` and the geoms
    # they name in the robot file it includes.
    var all_contact = String("")
    # <option> is merged attribute-wise, but MJCF also allows <flag> CHILDREN
    # inside it. Those were silently dropped before 2026-07-29, which quietly
    # disabled `<flag contact="disable"/>` for every merged model — cartpole
    # then launched its cart off the rails it is meant to overlap.
    var all_option_flags = String("")

    for i in range(len(xmls)):
        # ⚠⚠ COMMENTS COME OFF FIRST, AND THAT IS A FIX, NOT HYGIENE.
        # `_extract_section_inner` depth-counts `"<" + tag` over RAW TEXT, so a
        # comment that merely MENTIONS a section tag was counted as an opener,
        # the depth never balanced, and the section was emitted EMPTY. Measured
        # on three fixtures differing by one comment line:
        #
        #   two nested default classes, no comments        -> <default> present
        #   + "<!-- an ordinary remark, no brackets -->"   -> <default> present
        #   + "<!-- ... top-level <default>; ... -->"      -> <default> ABSENT
        #
        # MuJoCo then rejects the merged model with "unknown default class
        # name". ⚠ NESTING IS IRRELEVANT — the bug was filed as "merge_mjcf
        # cannot do nested defaults" and that is false; it handles them fine.
        #
        # This is the THIRD instance of the same shape in this function, after
        # a self-closing `<equality/>` inside a default class emptying
        # `<equality>` for a whole file (quadruped's leg couplings) and
        # `<tendon>` missing from the accumulator list entirely (fish). Both
        # parsers already strip comments at their entry points —
        # `parse_xml_model_data` always has, and `full_parser` was fixed for
        # this exact class after a commented-out `<site>` in Gymnasium's
        # `half_cheetah.xml` was parsed as a REAL site. `merge_mjcf` was the
        # last one reading raw text.
        #
        # ⚠ It runs BEFORE `_strip_wrapper` / `_strip_include_tags` on purpose:
        # a commented-out `<mujoco>` or `<include>` would mislead those two the
        # same way.
        #
        # ⚠ NOT A FULL TOKENISER. A `<` inside a string ATTRIBUTE VALUE would
        # still miscount. No model in the tree has one, and all three recorded
        # instances are comments; widening this to real tokenisation is a
        # separate job with a separate justification.
        #
        # `<freejoint>` -> `<joint type="free">` before ANY scanning, so the
        # ~20 `find("<joint")` sites downstream all see it. See
        # `_normalize_freejoint` for why this is textual rather than per-site.
        var stripped = _normalize_freejoint(
            _strip_include_tags(_strip_wrapper(_strip_xml_comments(xmls[i])))
        )

        # Singleton tags
        var opt = _extract_singleton_tag(stripped, "option")
        if opt.byte_length() > 0:
            option_tags.append(opt)
        # Carry any <flag .../> children of this fragment's <option>.
        all_option_flags = all_option_flags + _extract_section_inner(
            stripped, "option"
        )
        var comp = _extract_singleton_tag(stripped, "compiler")
        if comp.byte_length() > 0:
            compiler_tags.append(comp)

        # Accumulator sections (extract inner content, handle multiple occurrences)
        all_assets = all_assets + _extract_section_inner(stripped, "asset")
        all_defaults = all_defaults + _extract_section_inner(stripped, "default")
        all_worldbody = all_worldbody + _extract_section_inner(stripped, "worldbody")
        all_actuator = all_actuator + _extract_section_inner(stripped, "actuator")
        all_equality = all_equality + _extract_section_inner(stripped, "equality")
        all_tendon = all_tendon + _extract_section_inner(stripped, "tendon")
        all_sensor = all_sensor + _extract_section_inner(stripped, "sensor")
        all_contact = all_contact + _extract_section_inner(stripped, "contact")
        all_keyframe = all_keyframe + _extract_section_inner(
            stripped, "keyframe"
        )
        all_visual = all_visual + _extract_section_inner(stripped, "visual")

    # Build merged XML
    var result = String('<mujoco model="merged">\n')

    # Merged singletons
    var merged_compiler = _merge_singleton_attrs(compiler_tags, "compiler")
    if merged_compiler.byte_length() > 0:
        result = result + "  " + merged_compiler + "\n"

    var merged_option = _merge_singleton_attrs(option_tags, "option")
    if merged_option.byte_length() > 0:
        if _trim(all_option_flags).byte_length() > 0:
            # Re-open the self-closing merged tag so the <flag> children fit.
            var slash = merged_option.rfind("/>")
            var open_tag = (
                String(merged_option[byte=0:slash]) + ">" if slash
                != -1 else merged_option
            )
            result = (
                result
                + "  "
                + open_tag
                + "\n"
                + all_option_flags
                + "\n  </option>\n"
            )
        else:
            result = result + "  " + merged_option + "\n"

    # Visual
    if _trim(all_visual).byte_length() > 0:
        result = (
            result
            + "  <visual>\n"
            + _dedupe_last_wins(all_visual)
            + "  </visual>\n"
        )

    # Defaults
    if _trim(all_defaults).byte_length() > 0:
        result = result + "  <default>\n" + all_defaults + "  </default>\n"

    # Assets
    if _trim(all_assets).byte_length() > 0:
        result = result + "  <asset>\n" + all_assets + "  </asset>\n"

    # Worldbody
    if _trim(all_worldbody).byte_length() > 0:
        result = result + "  <worldbody>\n" + all_worldbody + "  </worldbody>\n"

    # Actuator
    # Emitted BEFORE <actuator> so the merged text reads like a hand-written
    # model; the parser resolves `tendon="..."` by name either way.
    if _trim(all_tendon).byte_length() > 0:
        result = result + "  <tendon>\n" + all_tendon + "  </tendon>\n"

    if _trim(all_actuator).byte_length() > 0:
        result = result + "  <actuator>\n" + all_actuator + "  </actuator>\n"

    # Equality
    if _trim(all_equality).byte_length() > 0:
        result = result + "  <equality>\n" + all_equality + "  </equality>\n"

    # Sensor (reference-only; see the docstring)
    if _trim(all_sensor).byte_length() > 0:
        result = result + "  <sensor>\n" + all_sensor + "  </sensor>\n"

    # Contact — <exclude> and <pair> are both parsed and honoured.
    if _trim(all_contact).byte_length() > 0:
        result = result + "  <contact>\n" + all_contact + "  </contact>\n"

    # Keyframe — emitted LAST, where hand-written models put it. Joined the
    # accumulators on 2026-08-13 for ToddlerBot, whose `<key name="home">`
    # lives in the INCLUDED robot file rather than the scene, so without this
    # the section never reached any parser. Fourth section to be dropped this
    # way after <tendon>, <option>'s <flag> children and <contact>.
    if _trim(all_keyframe).byte_length() > 0:
        result = result + "  <keyframe>\n" + all_keyframe + "  </keyframe>\n"

    result = result + "</mujoco>"
    return result


def _scan_max_condim(xml: String) -> Int:
    """Largest `condim=` in the file, floored at 3.

    Sizes the PYRAMIDAL edge list, which needs `2*(dim-1)` rows per contact.
    ⚠ THIS DELIBERATELY SCANS THE WHOLE FILE, `<default>` blocks included, and
    does not try to work out which classes are actually used. Over-estimating
    is SAFE — the builder zeroes the slots a contact does not need, so the only
    cost is a few unused rows — while under-estimating is SILENT: the extra
    friction rows get built into a workspace nothing reads, and the model spins
    and rolls without resistance. A conservative bound is the whole point.

    (Getting this wrong once already cost a full debugging arc: see
    tests/physics3d/test_rolling_friction_vs_mujoco.mojo.)
    """
    var best = 3
    var pos = 0
    var needle = 'condim="'
    var nlen = needle.byte_length()
    while True:
        var hit = xml.find(needle, pos)
        if hit < 0:
            break
        var vs = hit + nlen
        var ve = xml.find('"', vs)
        if ve < 0:
            break
        var val = 0
        var ok = ve > vs
        for i in range(vs, ve):
            var ch = Int(xml.as_bytes()[i])
            if ch < ord("0") or ch > ord("9"):
                ok = False
                break
            val = val * 10 + (ch - ord("0"))
        if ok and val > best:
            best = val
        pos = ve + 1
    return best


def _scan_noslip_iterations(xml: String) -> Int:
    """`<option noslip_iterations="N">`, or 0 if absent (MuJoCo's default).

    MuJoCo runs `mj_solNoSlip` after the main solver whenever this is > 0. It
    is a friction-only Gauss-Seidel sweep with the normal forces frozen, and
    it is NOT a rounding refinement: on dm_control's dog — the one suite model
    that sets it — turning it off moves MuJoCo's own rollout by `max|d(qvel)|`
    2.9e-2 on the FIRST contacting step.

    ⚠ THE LEDGER CLOSED THIS FEATURE ON A GREP THAT WAS WRONG. `docs/
    DM_CONTROL_PORT.md` decision 4 read "`grep -r noslip references/
    dm_control-main/` returns nothing in the suite" — `dog.xml` line 6 has set
    `noslip_iterations="4"` the whole time. The conclusion was accidentally
    right (dog was descoped) and the evidence was not.

    Unlike `_scan_max_condim`, this reads only the REAL `<option>` element:
    an over-estimate here is not free — it would run a solver pass MuJoCo does
    not run — so a value inside a comment or a `<default>` must not count.
    """
    var opt = _first_tag(xml, "option")
    if opt.byte_length() == 0:
        return 0
    var s = _trim(_extract_attr(opt, "noslip_iterations"))
    if s.byte_length() == 0:
        return 0
    var val = 0
    for i in range(s.byte_length()):
        var ch = Int(s.as_bytes()[i])
        if ch < ord("0") or ch > ord("9"):
            return 0
        val = val * 10 + (ch - ord("0"))
    return val


def _scan_ccd_tolerance(xml: String) -> Float64:
    """`<option ccd_tolerance="X">`, or MuJoCo's 1e-6 default.

    EPA's stopping rule: it breaks when the gap between its lower bound (the
    closest polytope face's distance) and its running upper bound falls below
    this. `mjc_penetration` copies it into `mjCCDConfig.tolerance` and also
    uses it as MPR's, so one number governs both.

    ⚠ TIGHTER IS NOT SAFER HERE. The stopping rule decides WHICH boundary face
    EPA settles on, and the contact NORMAL is that face's, so running past the
    reference does not converge toward it. We hardcoded 1e-8 — tighter than
    MuJoCo's — and a model setting this was ignored outright.

    ⚠ THE DEFAULT IS NOT DOCUMENTED IN THE ENGINE SOURCE, only in the USD
    schema (`src/experimental/usd/mjcPhysics/schema.usda`: `ccd_tolerance =
    1e-06`, `ccd_iterations = 35`). Confirmed against the 3.10.0 runtime on a
    model whose `<option>` sets neither — `m.opt.ccd_tolerance` reads 1e-06 and
    `m.opt.ccd_iterations` reads 35 — because a schema file in an
    `experimental/` directory is not evidence about the runtime by itself, and
    no reference tree here matches that runtime
    (`feedback_reference_tree_version_drift`).

    Like `_scan_noslip_iterations` this reads only the REAL `<option>`
    element: a value inside a comment or a `<default>` must not count.
    """
    var opt = _first_tag(xml, "option")
    if opt.byte_length() == 0:
        return MJ_CCD_TOLERANCE
    var s = _trim(_extract_attr(opt, "ccd_tolerance"))
    if s.byte_length() == 0:
        return MJ_CCD_TOLERANCE
    var v = _parse_float(s)
    # A zero or negative tolerance would make the loop run to its iteration
    # cap on every pair. MuJoCo does not guard this, but MuJoCo also does not
    # have our fixed polytope caps, so the failure mode differs: fall back
    # rather than silently changing what the caps mean.
    if v <= 0.0:
        return MJ_CCD_TOLERANCE
    return v


def _scan_ccd_iterations(xml: String) -> Int:
    """`<option ccd_iterations="N">`, or MuJoCo's 35 default.

    The EPA expansion cap. Ours is additionally bounded by `EPA_V_CAP` /
    `EPA_F_CAP`, which MuJoCo has no equivalent of — it grows the polytope on
    the heap — so a model asking for more iterations than the arrays can hold
    gets the arrays' limit. That is a real difference and it is why `gjk.mojo`
    takes the min explicitly rather than trusting the parsed value.
    """
    var opt = _first_tag(xml, "option")
    if opt.byte_length() == 0:
        return MJ_CCD_ITERATIONS
    var s = _trim(_extract_attr(opt, "ccd_iterations"))
    if s.byte_length() == 0:
        return MJ_CCD_ITERATIONS
    var val = 0
    for i in range(s.byte_length()):
        var ch = Int(s.as_bytes()[i])
        if ch < ord("0") or ch > ord("9"):
            return MJ_CCD_ITERATIONS
        val = val * 10 + (ch - ord("0"))
    if val <= 0:
        return MJ_CCD_ITERATIONS
    return val


def parse_xml(xml: String) -> ParsedModel:
    """Parse a MuJoCo XML string and return dimension counts.

    Designed to be called at comptime:

        comptime model = parse_xml(half_cheetah_xml)
        # model.NBODY == 8, model.NJOINT == 9, ...

    Counting rules
    ~~~~~~~~~~~~~~
    Bodies:  `<body` tags inside `<worldbody>` only (excludes `<default>`).
             NBODY = count + 1 (worldbody always added).
    Joints:  `<joint` tags inside `<worldbody>` only.
             NQ/NV derived from joint types:
               hinge / slide → NQ=1, NV=1
               ball          → NQ=4, NV=3
               free          → NQ=7, NV=6
    Geoms:   `<geom` tags inside `<worldbody>` (includes floor geom).
    Actuators: `<motor`, `<position`, `<velocity`, `<general` in `<actuator>`.
    """

    # ---- Strip XML comments to avoid counting commented-out tags ------------
    #
    # ⚠⚠ `_normalize_freejoint` RUNS HERE, NOT ONLY IN `merge_mjcf`. It used to
    # live only there, so `<freejoint/>` was rewritten for models built by the
    # composer and INVISIBLE to every model handed straight to `parse_xml`.
    # The failure is silent and total: `<freejoint` matches none of the ~20
    # `find("<joint")` sites, so NJOINT/NQ/NV come out 0, the body welds to the
    # world, and `pair_body_filtered`'s first clause (`weld_i == weld_j`)
    # then discards EVERY contact pair it is in. Measured on a free sphere
    # overlapping a static box: MuJoCo 1 contact, ours 0, and the body could
    # not have moved either since it had no dofs.
    #
    # Every model shipped today goes through `merge_mjcf` first and is
    # unaffected — the exposure is single-file MJCF, which is exactly the shape
    # Menagerie / SO-ARM / ToddlerBot ports arrive in.
    #
    # Idempotent: after one pass no `<freejoint` remains, so the composer path
    # normalizing first and this normalizing again is a no-op.
    var xml_clean = _strip_xml_comments(_normalize_freejoint(xml))

    # ---- Isolate sections to avoid counting <default> entries ---------------
    var worldbody = _extract_section(xml_clean, "worldbody")
    var actuator_sec = _extract_section(xml_clean, "actuator")

    # ---- Bodies -------------------------------------------------------------
    # <body tags inside worldbody (does NOT match <worldbody> itself)
    var n_bodies = _count_tag(worldbody, "body")
    var nbody = n_bodies + 1  # +1 for worldbody at index 0

    # ---- Joints & DOFs ------------------------------------------------------
    var njoint = _count_tag(worldbody, "joint")

    # Determine NQ/NV by joint type (search within worldbody section only)
    var n_free = _count_joints_with_type(worldbody, "free")
    var n_ball = _count_joints_with_type(worldbody, "ball")
    var n_other = njoint - n_free - n_ball  # hinge + slide

    var nq = n_free * 7 + n_ball * 4 + n_other * 1
    var nv = n_free * 6 + n_ball * 3 + n_other * 1

    # ---- Geoms --------------------------------------------------------------
    var ngeom = _count_tag(worldbody, "geom")

    # ---- Actuators ----------------------------------------------------------
    var nact = (
        _count_tag(actuator_sec, "motor")
        + _count_tag(actuator_sec, "position")
        + _count_tag(actuator_sec, "velocity")
        + _count_tag(actuator_sec, "general")
    )

    # ---- Assets (<asset> section) -------------------------------------------
    var asset_sec = _extract_section(xml_clean, "asset")
    var ntex = _count_tag(asset_sec, "texture")
    var nmat = _count_tag(asset_sec, "material")

    # ---- Visual elements in worldbody ---------------------------------------
    var nlight = _count_tag(worldbody, "light")
    var ncam = _count_tag(worldbody, "camera")
    var nsite = _count_tag(worldbody, "site")

    # ---- Equality constraints (<equality> section) --------------------------
    var eq_sec = _extract_section(xml_clean, "equality")
    # ⚠ `<joint>` HERE IS `mjEQ_JOINT`, NOT A `<worldbody>` joint — `eq_sec`
    # is the `<equality>` section only, so there is no collision. Omitting it
    # sizes the equality slab too small and `_fill_equality`'s records fall off
    # the end of `MAX_EQUALITY` silently (see the `neq`-vs-`max_equality`
    # trap). `<tendon>` equalities are NOT counted: they live on the tendon
    # record, flagged by `TENDON_IDX_IS_EQUALITY`, not in this slab.
    var neq = (
        _count_tag(eq_sec, "weld")
        + _count_tag(eq_sec, "connect")
        + _count_tag(eq_sec, "joint")
    )

    # ---- Contact exclusions (<contact> section) -----------------------------
    var contact_sec = _extract_section(xml_clean, "contact")
    var nexclude = _count_tag(contact_sec, "exclude")
    var npair = _count_tag(contact_sec, "pair")

    # ---- Tendons (<tendon> section) -----------------------------------------
    var tendon_sec = _extract_section(xml_clean, "tendon")
    var ntendon = _count_tag(tendon_sec, "fixed") + _count_tag(
        tendon_sec, "spatial"
    )

    # ---- Compiler angle units -----------------------------------------------
    var angle_deg = _compiler_angle_is_deg(xml_clean)

    # ---- Timestep (<option timestep="..."/>) --------------------------------
    var timestep = Float64(0.002)  # MuJoCo default
    var option_t = xml_clean.find("<option")
    if option_t != -1:
        var option_end = xml_clean.find(">", option_t)
        if option_end != -1:
            var otag = String(xml_clean[byte = option_t : option_end + 1])
            var ts_val = _extract_attr(otag, "timestep")
            if _trim(ts_val).byte_length() > 0:
                timestep = _parse_float(ts_val)

    return ParsedModel(
        nbody,
        njoint,
        nq,
        nv,
        ngeom,
        nact,
        ntex,
        nmat,
        nlight,
        ncam,
        nsite,
        neq,
        nexclude,
        npair,
        ntendon,
        angle_deg,
        timestep,
        _scan_max_condim(xml),
        _scan_noslip_iterations(xml),
        _scan_ccd_tolerance(xml),
        _scan_ccd_iterations(xml),
    )


# =============================================================================
# ComptimeActData — batch-precomputed XML data for GPU kernels
# =============================================================================


comptime MAX_COMPTIME_TENDONS: Int = 16

# Joint wraps per FIXED tendon, and per actuator transmission (a joint
# transmission is the degenerate one-wrap case).
#
# ⚠ THIS WAS A BARE `4` AND IT SILENTLY TRUNCATED. dm_control's dog wraps 11
# and 10 joints on `caudal_extend` / `caudal_bend` — its tail — so those two
# actuators drove a THIRD of the joints they should, while the parse wrote
# `tendon_trn_n = 4` and every consumer read a complete tendon. Six of dog's
# eight tendons wrap exactly 4, which is why nothing noticed.
#
# The third silent truncation of this shape in the dm_control arc, after
# `MAX_COMPTIME_TENDONS` (8 -> 16) and `MAX_NAMED_DEFAULTS` (16 -> 128). Per
# section 4.3 of `docs/DM_CONTROL_PORT_PHASE2.md`, WIDENING IS THE EASY HALF:
# `ComptimeActData.tendon_wrap_overflow` counts what would not fit and
# `ModelDefFromXML.init_fields` RAISES on it, so the next model to outgrow
# this fails to build instead of running wrong.
comptime MAX_COMPTIME_TENDON_WRAPS: Int = 16
"""Cap on `<fixed>` tendons the comptime parser records (quadruped needs 12)."""

comptime MAX_COMPTIME_MATERIALS: Int = 32
"""Cap on `<material>` records the comptime RENDER parser keeps.

Widened 8 -> 32 on 2026-08-03. Eight was not close: dm_control's shared asset
block (`envs/dm_control/common_xml.mojo`, the port of `common/materials.xml`)
declares THIRTEEN materials, and every ported suite model includes it, so most
of the suite was over the cap. Indices 0-7 cover grid..decoration; `eye`(8),
`target`(9) and `site`(12) are the ones that were past the end.

⚠ THE PAYLOAD WAS AN OUT-OF-BOUNDS READ, NOT A MISSING COLOUR. Material ids
come from `_rcd_find_material_index_by_name`, which returns the material's
ordinal among ALL `<material>` tags and knows nothing about this cap, and
`ModelDefFromXML` bounds-checks them against `Self.nmat` — which is
`ParsedModel.NMAT`, an INDEPENDENT `_count_tag` with no cap either. So `mid=9`
passed a `mid < nmat` guard and then indexed an 8-slot array: point_mass, fish
and reacher aborted on sight. Models whose materials all landed under 8 merely
rendered in the wrong colours, which is why this looked like two separate
complaints.

The `comptime assert` in `model_def_from_xml.mojo` is what keeps the two counts
from drifting apart again."""

comptime MAX_COMPTIME_SPATIAL_TENDONS: Int = 4
"""Cap on `<spatial>` tendons the RENDER parser records (ball_in_cup needs 1)."""

comptime MAX_COMPTIME_SPATIAL_TENDON_SITES: Int = 16
"""Flat cap on site references across all spatial tendons (4 per tendon)."""

comptime MAX_COMPTIME_TEXTURES: Int = 16
"""Cap on `<texture>` records the comptime render parser keeps.

Widened 8 -> 16 alongside the materials. Only two textures come from the shared
block (skybox, grid), so this one was not being exceeded — but it is the same
scan, sized the same way, with the same silent-truncation shape, and leaving it
at 8 next to a 32 just invites the next model to trip it."""

# ── Render-data capacities (`ComptimeRenderData`) ────────────────────────────
#
# ⚠ THESE WERE BARE LITERALS — 64 geoms and 16 sites — with a SILENT fill
# guard (`if geom_count < 64:`) and consumers looping `range(Self.NGEOM)`.
# dm_control's dog has 128 geoms, so half of them were never written and the
# renderer indexed past the end: a `debug_assert` in the viewer build, and an
# OUT-OF-BOUNDS READ in a release one. quadruped (30 sites), humanoid (25) and
# manipulator (20) were over the site cap the same way.
#
# Sized from the measured maxima with headroom (dog 128 geoms, quadruped 30
# sites), and `init_fields` now RAISES when a model exceeds them rather than
# truncating — the same rule as MAX_COMPTIME_TENDON_WRAPS.
# ⚠ 160 -> 448 and 48 -> 192 for dm_control manipulation's BRICK tasks, whose
# models carry a translucent contactless HINT twin of every brick:
#
#     stack_2_bricks   185 geoms   79 sites
#     stack_3_bricks   267        113
#     reassemble_5     431        181
#
# `stack_2_bricks` raised the 160 cap, which did exactly what it was written to
# do — `init_fields` RAISED and named the constant and the file.
#
# ⚠ THE COST WAS MEASURED INTERLEAVED, and the first attempt at measuring it
# was wrong. A single before/after on a 62-geom model def read 8.3 s -> 12.1 s
# and looked like a 46% regression; alternating the two builds gives
# 12.01 / 8.17 for the OLD cap against 7.91 / 8.61 for the NEW one, i.e. the
# widening costs nothing and the machine was simply busy. Report the MIN of an
# interleaved pair, never a single sequential pair. These are `InlineArray`s of
# Int/Float64; the comptime cliff is the STRUCT case
# (`feedback_inlinearray_of_nontrivial_structs_is_a_compile_cliff`).
comptime MAX_COMPTIME_RENDER_GEOMS: Int = 448
comptime MAX_COMPTIME_RENDER_SITES: Int = 192

# ⚠ WAS A BARE LITERAL 16 IN THREE PLACES, WITH A SILENT FILL GUARD
# (`if data.nmesh < 16`) — the exact shape `init_fields` already documents for
# render geoms and sites. SO-ARM100 declares 18 meshes, so two were dropped
# without a word and any geom naming them simply did not draw. 32 covers every
# ported model; the guard below now RAISES instead of truncating.
comptime MAX_COMPTIME_RENDER_MESHES: Int = 32

# Body NAMES are deliberately NOT recorded here.
#
# A `<skin>`'s bones name the bodies they follow, so binding one needs names —
# but every way of writing them into this comptime struct is a compile failure in
# the interpreter, and the failures point at the standard library rather than at
# the cause. Measured, in order:
#
#   · storing a CONSTANT at the DFS site compiles; storing ANY slice there does
#     not, including a fixed `worldbody[byte=p:p+4]`. So it is the slice.
#   · hoisting to a top-level pass does not help, on `worldbody` or on
#     `xml_clean`, with `_extract_attr` or with pure index arithmetic.
#   · shrinking the array from 96 entries to 8 does not help either.
#   · the `<texture>` loop below stores `_extract_attr(tag, "name")` from the
#     same depth and DOES compile, so this is not the slice-depth family in
#     `feedback_comptime_nested_string_slice_fails`. Whatever separates them was
#     not worth more bisecting.
#
# `ModelDefFromXML.body_names()` extracts them from the model's XML AT RUNTIME
# instead, where none of this applies — see its note. The XML is already carried
# as a comptime parameter, so nothing extra is stored to make that possible.

comptime MAX_COMPTIME_KEYFRAMES: Int = 8
"""Cap on `<keyframe><key>` entries the comptime parser records.

⚠ MEASURED with MuJoCo, not grep — a `<key>` inside an XML comment is a real
hazard here: `rethink_robotics_sawyer/sawyer.xml` carries a commented-out
second `<key name="home">` whose qpos is one slot LONGER than nq, and a text
scan reads it as a live over-length keyframe.

Across all of Menagerie the histogram is `{1: 105, 2: 14, 3: 1}` — the maximum
is THREE (`franka_emika_panda/mjx_single_cube.xml`). 8 is headroom over that
rather than a slot above it, per `MAX_COMPTIME_NQ`'s note. Exceeding it sets
`bad_keyframe_code = 1` and `ModelDefFromXML` fails the build, rather than the
silent truncation `MAX_COMPTIME_TENDONS` and `MAX_COMPTIME_ACTUATORS` both
shipped with.

⚠ These arrays are `NKEYS * NQ0` and are materialized by the comptime
interpreter; raising this is not free. Measure the build time if you do."""

comptime MAX_COMPTIME_ACTUATORS: Int = 64
"""Cap on actuators the comptime parser records (humanoid_CMU needs 56).

⚠ COUNT THESE WITH MuJoCo, NOT WITH grep. `grep -c '<motor '` on
humanoid_CMU.xml says 57 and `mjModel.nu` says 56 — the extra match is the
`<motor ctrllimited=... />` inside `<default class="main">`. Every count in
these three docstrings was off by the number of same-named elements sitting in
default blocks until it was checked against a compiled model.

Widened 32 -> 64 on 2026-08-03 for dm_control's humanoid_CMU. The old bound
was a SILENT TRUNCATION of exactly the shape `MAX_COMPTIME_TENDONS` had before
2026-07-31: the scan below is `while act_count < CAP`, so a model with more
actuators than the cap simply stopped recording, while `ParsedModel.nact`
counted the tags INDEPENDENTLY and came out right. The env would therefore
expose the full action space and silently apply zero force through every
actuator past the cap. `ModelDefFromXML` now asserts `nact <=
MAX_COMPTIME_ACTUATORS` so the next model to outgrow this fails to compile.

Measured with MuJoCo 3.10.0: humanoid_CMU `nu` 56, dog `nu` 38 — both fit. If
a later model does not, RAISE THIS AND MEASURE THE BUILD TIME; these arrays are
materialized by the comptime interpreter."""

comptime MAX_COMPTIME_JOINTS: Int = 96
"""Cap on joints the comptime parser records (dog needs 75, humanoid_CMU 57).

(humanoid_CMU: 1 free + 56 hinges. `grep -c '<joint '` says 60: four of those
are the `<joint>` elements of `main`, `stiff_low`, `stiff_medium` and
`stiff_high`.)

Same silent-truncation shape as `MAX_COMPTIME_ACTUATORS` above, and the payload
is worse: `joint_qpos_adr` / `joint_is_limited` / `joint_range_*` feed the JOINT
LIMIT rows, so a joint past the cap keeps its degree of freedom and quietly
loses its stops. Asserted in `ModelDefFromXML` against `njoint`.

Widened 64 -> 96 on 2026-08-03 for dm_control's dog (Phase 4). ⚠ dog is TWO
models, and only one of them was measured when this note first said "75":

    stand / walk / trot / run   njnt 74   nq 80    (`make_model` deletes the
                                                    ball, and with it a free
                                                    joint worth 1 jnt / 7 qpos)
    fetch                       njnt 75   nq 87    (ball kept)

Both were counted with `mjModel`, not grep. 96 leaves headroom over the larger
of the two rather than sitting one slot above it — see `MAX_COMPTIME_NQ`'s note
on why a one-slot margin is not a margin."""

comptime MAX_COMPTIME_NQ: Int = 128
"""Cap on `qpos0` slots the comptime parser records.

⚠ humanoid_CMU does NOT need this widening — its `nq` is 63, ONE SLOT under the
old bound of 64. It was widened on the strength of a miscount and is kept
because dog's `nq` is 80 (stand/walk/trot/run) or 87 (fetch) and genuinely does
not fit, and because the failure mode below is the worst of the three. One slot
is not a margin.

Widened 64 -> 128 on 2026-08-03. ⚠ THIS ONE IS NOT A TRUNCATING SCAN — the
writes are `data.qpos0[qpos_adr] = ...` indexed by the joint's own qpos address,
so a model with `nq` past the cap indexes OUT OF BOUNDS rather than stopping
early. Asserted against `nq` in `ModelDefFromXML`. dog is the model that exceeds 64.

Note the SEPARATE 64-geom cap in `ComptimeRenderData` below: that one is
RENDER data, not physics (`fields.Model` is parameterized by NGEOM and comes
from the runtime `full_parser`), so exceeding it costs geoms in the viewer and
nothing in the dynamics. dog has 290 geoms as authored (296 with the ball) and
blows straight past it; humanoid_CMU has 50 and does not. ⚠ After the Phase 4
mesh-inertia bake the ported dog carries 128 geoms, not 290 — the 162 bone
meshes are non-colliding and are deleted once their inertia is stated
explicitly."""


struct ComptimeActData[NACT: Int, NJNT: Int, NQ0: Int, NTEN: Int, WRAPS: Int](Copyable, Movable):
    """Precomputed actuator/joint data for GPU kernel use.

    Stores results of XML parsing in InlineArrays so that GPU kernels can
    access them via compile-time array indexing (no String operations needed).

    Usage:
        comptime _acd = parse_xml_model_data(Self.xml)
        # In GPU kernel:  Self._acd.motor_gears[i]  (no String ops)
    """

    comptime NKEYS: Int = MAX_COMPTIME_KEYFRAMES

    var motor_gears: InlineArray[Float64, Self.NACT]
    var motor_dof_adr: InlineArray[Int, Self.NACT]
    var motor_ctrl_min: InlineArray[Float64, Self.NACT]
    var motor_ctrl_max: InlineArray[Float64, Self.NACT]
    # `actuator_ctrllimited` — whether the ctrlrange above is APPLIED at all.
    # Exactly the `forcelimited` story below, one field up: `ctrllimited`
    # defaults to "auto", so the absent attribute is NOT "false", it is "true
    # iff a range was defined". MEASURED against the 3.10.0 runtime, all
    # spellings, with `qfrc_actuator` at `ctrl = 5.0` as the observable:
    #   <motor/>                                     -> limited 0, [0, 0], +5.0
    #   <motor ctrlrange="-1 1"/>                    -> limited 1, [-1, 1], +1.0
    #   <motor ctrlrange="-2 3"/>                    -> limited 1, [-2, 3], +3.0
    #   <motor ctrlrange="0 0"/>                     -> limited 0,          +5.0
    #   <motor ctrlrange="-1 1" ctrllimited="false"/>-> limited 0,          +5.0
    #   <motor ctrllimited="false"/>                 -> limited 0,          +5.0
    # and `ctrllimited="true"` with no range is a COMPILE ERROR in MuJoCo
    # ("invalid control range for actuator"), so limited-with-zero-range is
    # unrepresentable and needs no handling — same as `forcelimited`.
    #
    # ⚠⚠ WITHOUT THIS FIELD THE CLAMP RAN UNCONDITIONALLY, against a ctrlrange
    # that FALLS BACK TO (-1, 1) when no level of the model supplies one. So an
    # actuator MuJoCo leaves unclamped was silently squeezed into +-1. Zero of
    # the 254 actuators in the 31 dm_control/Gymnasium reference models are
    # unlimited, which is why nothing here ever saw it — but 423 of Menagerie's
    # 2244 are, and ToddlerBot is 30 of 30 on every variant. Its `<position>`
    # actuators take a target ANGLE as `ctrl`, over joints ranging to 18.5 rad,
    # so that robot could not have been commanded past 1 radian.
    var motor_ctrl_limited: InlineArray[Int, Self.NACT]
    # ── `forcerange` / `forcelimited` (mj_fwdActuation's force clamp) ────────
    #
    # ⚠ THE CLAMP IS ON THE SCALAR ACTUATOR FORCE, BEFORE THE MOMENT. MEASURED:
    # `<motor gear="3" forcerange="-1 1">` at ctrl 5 gives
    # `actuator_force 1`, `actuator_moment 3`, `qfrc_actuator 3` — so the limit
    # bounds `gain*u + bias`, and `gear` multiplies AFTERWARDS. Clamping
    # `qfrc` instead would cap this actuator at 1 N·m rather than 3.
    #
    # `motor_force_limited` is MuJoCo's `actuator_forcelimited`, resolved from
    # `forcelimited="auto"` (the default): limited IFF a `forcerange` other than
    # "0 0" is defined. MEASURED, all four spellings:
    #   <motor/>                                 -> limited 0, range [0, 0]
    #   <motor forcerange="0 0"/>                -> limited 0, range [0, 0]
    #   <motor forcerange="-1 1"/>               -> limited 1, range [-1, 1]
    #   <motor forcerange="-1 1" forcelimited="false"/> -> limited 0
    # and `forcelimited="true"` with no range is a COMPILE ERROR in MuJoCo
    # ("invalid force range for actuator"), so a limited-but-zero-range
    # actuator is unrepresentable and needs no handling here.
    #
    # ⚠ Stored as an explicit 0/1 rather than inferred from `min >= max`. The
    # "undefined" marker really is [0, 0], but leaning on that would make an
    # asymmetric range like [-0.5, 4] and a degenerate one indistinguishable
    # from the storage alone.
    var motor_force_limited: InlineArray[Int, Self.NACT]
    var motor_force_min: InlineArray[Float64, Self.NACT]
    var motor_force_max: InlineArray[Float64, Self.NACT]
    # ── Actuator transmission + gain, as a single flat representation ────────
    #
    # MuJoCo's actuator force is `moment^T * (gain*ctrl + bias)`, and both
    # transmissions we support reduce to the same shape — a list of
    # (qpos address, dof address, coefficient) triples:
    #
    #   mjTRN_JOINT   length = gear * qpos[qadr],          moment = gear
    #   mjTRN_TENDON  length = gear * sum coef_k qpos_k,    moment_k = gear*coef_k
    #
    # so a joint transmission is the degenerate one-triple case with coef 1.
    # `motor_trn_n[i] == 0` means the actuator has no usable transmission
    # (unresolved name) and is skipped.
    #
    # `motor_kind` is ACT_KIND_MOTOR / ACT_KIND_POSITION (flat_model.mojo):
    #   MOTOR     force = kp * u                       (gaintype fixed, NO bias)
    #   POSITION  force = kp*(u - length) - kv*vel      (gaintype fixed +
    #             biastype affine, i.e. the <position> shape)
    #
    # where `u` is `act` for a dyntype actuator and `ctrl` otherwise.
    #
    # ⚠ `motor_kp` IS MuJoCo's `gainprm[0]`, NOT "the position gain". It reads
    # as a servo gain only on the POSITION path. It defaults to 1.0 — MuJoCo's
    # own `gainprm` default — so a plain `<motor>`, which never writes it, is
    # `force = 1 * ctrl` exactly as before.
    #
    # dog is why MOTOR carries a gain at all. Its 38 actuators are
    #
    #     <general ctrllimited="true" ctrlrange="-1 1"
    #              dyntype="filter" dynprm="0.05" gainprm="0.02"/>
    #
    # — no `biastype`, so MuJoCo's default `mjBIAS_NONE` applies and there is
    # no position feedback whatsoever: force = 0.02 * act, where act is a
    # 0.05 s lag of ctrl. Before this, EVERY `<general>` was classified
    # ACT_KIND_POSITION on the strength of the tag name alone, so dog's
    # actuators were refused outright by the `bad_actuator` gate (code 1,
    # biastype != affine). That gate doing its job is the only reason this was
    # a compile error rather than 38 torque motors driven by a phantom
    # position error.
    var motor_kind: InlineArray[Int, Self.NACT]
    var motor_kp: InlineArray[Float64, Self.NACT]
    var motor_kv: InlineArray[Float64, Self.NACT]
    # ── Activation state (`dyntype`), added for dm_control's quadruped ───────
    #
    # `motor_dyn_tau[i] > 0` means actuator i owns ONE activation variable
    # obeying mjDYN_FILTER, `act_dot = (ctrl - act) / tau` (engine_forward.c
    # :340), and its force reads `act` where a dyntype-less actuator reads
    # `ctrl` ("force = gain .* [ctrl/act]", mj_fwdActuation). `motor_act_adr`
    # is that variable's index in `Data.act`, or -1 for the actuators that
    # have none — the same -1 convention as MuJoCo's `actuator_actadr`.
    #
    # `na` is the TOTAL count, i.e. MuJoCo's `m->na`. It is 0 for every model
    # ported before quadruped (`<motor>` and `<position>` both default to
    # dyntype=none), which is why `Data` can carry it as a trailing parameter
    # defaulted to 0 without touching a single existing env config.
    var motor_dyn_tau: InlineArray[Float64, Self.NACT]
    var motor_act_adr: InlineArray[Int, Self.NACT]
    var na: Int
    # ── Unsupported-`<general>` report ──────────────────────────────────────
    #
    # This function runs at comptime and cannot `raise`, so a `<general>` whose
    # gain/bias/dyn shape we do not implement is RECORDED here and turned into
    # a compile error by `ModelDefFromXML`'s `comptime assert`. Silently
    # simulating the wrong actuator law is the failure mode this exists to
    # prevent — it produces a working env for a different robot.
    #
    #   -1  fine
    #    0  gaintype is not `fixed`
    #    1  biastype is neither `none` nor `affine`
    #    2  biasprm[0] != 0        (a constant force offset)
    #    3  biasprm[1] != -gainprm[0]  (not a position servo)
    #    4  dyntype is not `none` or `filter`
    var bad_actuator: Int
    var bad_actuator_code: Int
    var motor_trn_n: InlineArray[Int, Self.NACT]
    var motor_trn_qadr: InlineArray[Int, Self.NACT * Self.WRAPS]
    var motor_trn_dadr: InlineArray[Int, Self.NACT * Self.WRAPS]
    var motor_trn_coef: InlineArray[Float64, Self.NACT * Self.WRAPS]
    # ── Fixed-tendon springs (`<fixed stiffness="..."/>`) ────────────────────
    #
    # MuJoCo's tendon spring (engine_passive.c, "tendon-level spring-dampers")
    # is a DEADBAND on `tendon_lengthspring`:
    #
    #   length > upper -> frc = stiffness*(upper - length)
    #   length < lower -> frc = stiffness*(lower - length)
    #   otherwise      -> 0                       then qfrc += ten_J^T * frc
    #
    # With no `springlength` attribute the compiler collapses the band to the
    # tendon's length at qpos0, which for a fixed tendon over joints at their
    # own qpos0 is where both bounds land. Same triple representation as above.
    # Widened 8 -> MAX_COMPTIME_TENDONS on 2026-07-31 for quadruped, which
    # declares TWELVE fixed tendons (4 coupling + 4 lift + 4 extend). The old
    # bound was a SILENT truncation: `while data.ntendon < 8` simply stopped,
    # and the four dropped tendons' actuators resolved to `motor_trn_n == 0`,
    # which `apply_actions` skips. Four of twelve legs' actuators would have
    # done nothing at all, with no diagnostic anywhere. `ModelDefFromXML` now
    # asserts `MAX_TENDON <= MAX_COMPTIME_TENDONS` so the next model to
    # outgrow this fails to compile instead.
    var tendon_stiffness: InlineArray[Float64, Self.NTEN]
    var tendon_spring_lo: InlineArray[Float64, Self.NTEN]
    var tendon_spring_hi: InlineArray[Float64, Self.NTEN]
    # How many joint wraps did NOT fit in MAX_COMPTIME_TENDON_WRAPS, for the
    # worst tendon. Zero on every well-sized model; `init_fields` RAISES when
    # it is not — the half of a cap that actually prevents bugs.
    var tendon_wrap_overflow: Int
    # How many `<fixed>` tendons did not fit in NTEN. Same job as the field
    # above, one level up: that one guards the WIDTH of a tendon, this one the
    # COUNT of them.
    #
    # ⚠ THIS EXISTS BECAUSE THE CAP WENT LIVE WITHOUT IT. `NTEN` used to be a
    # global `MAX_COMPTIME_TENDONS`, which nobody could under-declare; sizing
    # it from `ModelDefFromXML.max_tendon` (2026-08-11, cc7021d0) turned that
    # parameter — a hand-written `Int = 0` default — into a real bound. fish
    # had never declared it and has two tendons, so it silently ran with one
    # from that commit until 2026-08-12. `while data.ntendon < NTEN` just
    # stops, and a dropped tendon's actuator resolves to `motor_trn_n == 0`,
    # which `apply_actions` skips: no diagnostic anywhere. Same failure the
    # 2026-07-31 comment above describes for quadruped, re-introduced by the
    # fix that made the array small enough to matter again.
    var tendon_count_overflow: Int
    var tendon_trn_n: InlineArray[Int, Self.NTEN]
    var tendon_trn_qadr: InlineArray[Int, Self.NTEN * Self.WRAPS]
    var tendon_trn_dadr: InlineArray[Int, Self.NTEN * Self.WRAPS]
    var tendon_trn_coef: InlineArray[Float64, Self.NTEN * Self.WRAPS]
    var ntendon: Int
    var joint_is_limited: InlineArray[Bool, Self.NJNT]
    var joint_qpos_adr: InlineArray[Int, Self.NJNT]
    var joint_range_min: InlineArray[Float64, Self.NJNT]
    var joint_range_max: InlineArray[Float64, Self.NJNT]
    var inertiafromgeom: Bool
    var settotalmass: Float64
    # Initial qpos values from <custom><numeric name="init_qpos" data="..."/>.
    # nq == 0 means no init_qpos was found; use qpos0 defaults instead.
    var qpos0: InlineArray[Float64, Self.NQ0]
    var nq: Int
    # qpos address of the first free joint (-1 if no free joint present).
    var free_joint_qpos_adr: Int

    # ---- <keyframe><key> ---------------------------------------------------
    #
    # ⚠ A KEYFRAME IS NOT A RESET. Measured on the 3.10.0 runtime: with a
    # keyframe present, `mj_resetData` still writes `qpos0` — only an explicit
    # `mj_resetDataKeyframe(m, d, i)` applies one. So these are RECORDED here
    # and applied only by `ModelDefFromXML.reset_data_keyframe`; `reset_data`
    # is deliberately unchanged. Making the default reset "prefer" a keyframe
    # would silently diverge from MuJoCo on every model that has one, which is
    # the same shape as the `ctrlrange` fallback that became a hard clamp.
    #
    # `key_qvel` is sized by NQ0 rather than by nv, which is never smaller
    # than it needs to be (nv <= nq always) and saves a sixth type parameter.
    var nkey: Int
    var key_time: InlineArray[Float64, Self.NKEYS]
    # Value COUNT supplied per key, 0 when the attribute was absent. MuJoCo
    # fills an omitted `qpos` from qpos0 and an omitted `qvel`/`ctrl` with
    # zeros, so "absent" and "all zeros" are different and must stay so.
    var key_nqpos: InlineArray[Int, Self.NKEYS]
    var key_nqvel: InlineArray[Int, Self.NKEYS]
    var key_nctrl: InlineArray[Int, Self.NKEYS]
    var key_qpos: InlineArray[Float64, Self.NKEYS * Self.NQ0]
    var key_qvel: InlineArray[Float64, Self.NKEYS * Self.NQ0]
    var key_ctrl: InlineArray[Float64, Self.NKEYS * Self.NACT]
    # Non-zero => the model uses a keyframe feature we do not model. Reported
    # by `ModelDefFromXML`'s asserts rather than raised here, mirroring
    # `bad_actuator_code` — the comptime parser has no good way to raise.
    #   1 = more keys than MAX_COMPTIME_KEYFRAMES
    #   2 = `act` / `mpos` / `mquat` present (we model none of them)
    var bad_keyframe_code: Int

    def __init__(out self):
        """Initialize with safe defaults: gears=1.0, dof_adr=-1, all others=0/False.
        """
        self.motor_gears = InlineArray[Float64, Self.NACT](fill=1.0)
        self.motor_dof_adr = InlineArray[Int, Self.NACT](fill=-1)
        self.motor_ctrl_min = InlineArray[Float64, Self.NACT](fill=-1.0)
        self.motor_ctrl_max = InlineArray[Float64, Self.NACT](fill=1.0)
        self.motor_ctrl_limited = InlineArray[Int, Self.NACT](fill=0)
        self.motor_force_limited = InlineArray[Int, Self.NACT](fill=0)
        self.motor_force_min = InlineArray[Float64, Self.NACT](fill=0.0)
        self.motor_force_max = InlineArray[Float64, Self.NACT](fill=0.0)
        self.motor_kind = InlineArray[Int, Self.NACT](fill=0)  # ACT_KIND_MOTOR
        self.motor_kp = InlineArray[Float64, Self.NACT](fill=1.0)
        self.motor_kv = InlineArray[Float64, Self.NACT](fill=0.0)
        self.motor_dyn_tau = InlineArray[Float64, Self.NACT](fill=0.0)
        self.motor_act_adr = InlineArray[Int, Self.NACT](fill=-1)
        self.na = 0
        self.bad_actuator = -1
        self.bad_actuator_code = -1
        self.motor_trn_n = InlineArray[Int, Self.NACT](fill=0)
        self.motor_trn_qadr = InlineArray[Int, Self.NACT * Self.WRAPS](fill=-1)
        self.motor_trn_dadr = InlineArray[Int, Self.NACT * Self.WRAPS](fill=-1)
        self.motor_trn_coef = InlineArray[Float64, Self.NACT * Self.WRAPS](fill=0.0)
        self.tendon_stiffness = InlineArray[Float64, Self.NTEN](
            fill=0.0
        )
        self.tendon_spring_lo = InlineArray[Float64, Self.NTEN](
            fill=0.0
        )
        self.tendon_spring_hi = InlineArray[Float64, Self.NTEN](
            fill=0.0
        )
        self.tendon_wrap_overflow = 0
        self.tendon_count_overflow = 0
        self.tendon_trn_n = InlineArray[Int, Self.NTEN](fill=0)
        self.tendon_trn_qadr = InlineArray[Int, Self.NTEN * Self.WRAPS](
            fill=-1
        )
        self.tendon_trn_dadr = InlineArray[Int, Self.NTEN * Self.WRAPS](
            fill=-1
        )
        self.tendon_trn_coef = InlineArray[Float64, Self.NTEN * Self.WRAPS](
            fill=0.0
        )
        self.ntendon = 0
        self.joint_is_limited = InlineArray[Bool, Self.NJNT](fill=False)
        self.joint_qpos_adr = InlineArray[Int, Self.NJNT](fill=0)
        self.joint_range_min = InlineArray[Float64, Self.NJNT](fill=0.0)
        self.joint_range_max = InlineArray[Float64, Self.NJNT](fill=0.0)
        self.inertiafromgeom = False
        self.settotalmass = Float64(-1.0)
        self.qpos0 = InlineArray[Float64, Self.NQ0](fill=0.0)
        self.nq = 0
        self.free_joint_qpos_adr = -1
        self.nkey = 0
        self.key_time = InlineArray[Float64, Self.NKEYS](fill=0.0)
        self.key_nqpos = InlineArray[Int, Self.NKEYS](fill=0)
        self.key_nqvel = InlineArray[Int, Self.NKEYS](fill=0)
        self.key_nctrl = InlineArray[Int, Self.NKEYS](fill=0)
        self.key_qpos = InlineArray[Float64, Self.NKEYS * Self.NQ0](fill=0.0)
        self.key_qvel = InlineArray[Float64, Self.NKEYS * Self.NQ0](fill=0.0)
        self.key_ctrl = InlineArray[Float64, Self.NKEYS * Self.NACT](fill=0.0)
        self.bad_keyframe_code = 0

    def __init__(out self, *, copy: Self):
        # InlineArray is not ImplicitlyCopyable; copy element-by-element.
        self.motor_gears = InlineArray[Float64, Self.NACT](fill=1.0)
        self.motor_dof_adr = InlineArray[Int, Self.NACT](fill=-1)
        self.motor_ctrl_min = InlineArray[Float64, Self.NACT](fill=-1.0)
        self.motor_ctrl_max = InlineArray[Float64, Self.NACT](fill=1.0)
        self.motor_ctrl_limited = InlineArray[Int, Self.NACT](fill=0)
        self.motor_force_limited = InlineArray[Int, Self.NACT](fill=0)
        self.motor_force_min = InlineArray[Float64, Self.NACT](fill=0.0)
        self.motor_force_max = InlineArray[Float64, Self.NACT](fill=0.0)
        self.motor_kind = InlineArray[Int, Self.NACT](fill=0)
        self.motor_kp = InlineArray[Float64, Self.NACT](fill=1.0)
        self.motor_kv = InlineArray[Float64, Self.NACT](fill=0.0)
        self.motor_dyn_tau = InlineArray[Float64, Self.NACT](fill=0.0)
        self.motor_act_adr = InlineArray[Int, Self.NACT](fill=-1)
        self.na = copy.na
        self.bad_actuator = copy.bad_actuator
        self.bad_actuator_code = copy.bad_actuator_code
        self.motor_trn_n = InlineArray[Int, Self.NACT](fill=0)
        self.motor_trn_qadr = InlineArray[Int, Self.NACT * Self.WRAPS](fill=-1)
        self.motor_trn_dadr = InlineArray[Int, Self.NACT * Self.WRAPS](fill=-1)
        self.motor_trn_coef = InlineArray[Float64, Self.NACT * Self.WRAPS](fill=0.0)
        self.tendon_stiffness = InlineArray[Float64, Self.NTEN](
            fill=0.0
        )
        self.tendon_spring_lo = InlineArray[Float64, Self.NTEN](
            fill=0.0
        )
        self.tendon_spring_hi = InlineArray[Float64, Self.NTEN](
            fill=0.0
        )
        # ⚠ CARRY IT. Resetting to 0 in the copy ctor would lose the overflow
        # precisely when the data is copied, which is how a cap diagnostic dies.
        self.tendon_wrap_overflow = copy.tendon_wrap_overflow
        self.tendon_count_overflow = copy.tendon_count_overflow
        self.tendon_trn_n = InlineArray[Int, Self.NTEN](fill=0)
        self.tendon_trn_qadr = InlineArray[Int, Self.NTEN * Self.WRAPS](
            fill=-1
        )
        self.tendon_trn_dadr = InlineArray[Int, Self.NTEN * Self.WRAPS](
            fill=-1
        )
        self.tendon_trn_coef = InlineArray[Float64, Self.NTEN * Self.WRAPS](
            fill=0.0
        )
        self.ntendon = copy.ntendon
        self.joint_is_limited = InlineArray[Bool, Self.NJNT](fill=False)
        self.joint_qpos_adr = InlineArray[Int, Self.NJNT](fill=0)
        self.joint_range_min = InlineArray[Float64, Self.NJNT](fill=0.0)
        self.joint_range_max = InlineArray[Float64, Self.NJNT](fill=0.0)
        self.inertiafromgeom = copy.inertiafromgeom
        self.settotalmass = copy.settotalmass
        self.qpos0 = InlineArray[Float64, Self.NQ0](fill=0.0)
        self.nq = copy.nq
        self.free_joint_qpos_adr = copy.free_joint_qpos_adr
        self.nkey = copy.nkey
        self.bad_keyframe_code = copy.bad_keyframe_code
        self.key_time = InlineArray[Float64, Self.NKEYS](fill=0.0)
        self.key_nqpos = InlineArray[Int, Self.NKEYS](fill=0)
        self.key_nqvel = InlineArray[Int, Self.NKEYS](fill=0)
        self.key_nctrl = InlineArray[Int, Self.NKEYS](fill=0)
        self.key_qpos = InlineArray[Float64, Self.NKEYS * Self.NQ0](fill=0.0)
        self.key_qvel = InlineArray[Float64, Self.NKEYS * Self.NQ0](fill=0.0)
        self.key_ctrl = InlineArray[Float64, Self.NKEYS * Self.NACT](fill=0.0)
        # Each of these is copied over ITS OWN length, matching how every
        # other array in this constructor is handled.
        for k in range(Self.NKEYS):
            self.key_time[k] = copy.key_time[k]
            self.key_nqpos[k] = copy.key_nqpos[k]
            self.key_nqvel[k] = copy.key_nqvel[k]
            self.key_nctrl[k] = copy.key_nctrl[k]
        for i in range(Self.NKEYS * Self.NQ0):
            self.key_qpos[i] = copy.key_qpos[i]
            self.key_qvel[i] = copy.key_qvel[i]
        for i in range(Self.NKEYS * Self.NACT):
            self.key_ctrl[i] = copy.key_ctrl[i]
        for i in range(Self.NACT):
            self.motor_gears[i] = copy.motor_gears[i]
            self.motor_dof_adr[i] = copy.motor_dof_adr[i]
            self.motor_ctrl_min[i] = copy.motor_ctrl_min[i]
            self.motor_ctrl_max[i] = copy.motor_ctrl_max[i]
            self.motor_ctrl_limited[i] = copy.motor_ctrl_limited[i]
            self.motor_force_limited[i] = copy.motor_force_limited[i]
            self.motor_force_min[i] = copy.motor_force_min[i]
            self.motor_force_max[i] = copy.motor_force_max[i]
            self.motor_kind[i] = copy.motor_kind[i]
            self.motor_kp[i] = copy.motor_kp[i]
            self.motor_kv[i] = copy.motor_kv[i]
            self.motor_dyn_tau[i] = copy.motor_dyn_tau[i]
            self.motor_act_adr[i] = copy.motor_act_adr[i]
            self.motor_trn_n[i] = copy.motor_trn_n[i]
        # Joints have their OWN cap — the two were one loop while both were 32,
        # which is a bug the moment they diverge (dog: 58 actuators, 147 joints).
        for i in range(Self.NJNT):
            self.joint_is_limited[i] = copy.joint_is_limited[i]
            self.joint_qpos_adr[i] = copy.joint_qpos_adr[i]
            self.joint_range_min[i] = copy.joint_range_min[i]
            self.joint_range_max[i] = copy.joint_range_max[i]
        for i in range(Self.NACT * Self.WRAPS):
            self.motor_trn_qadr[i] = copy.motor_trn_qadr[i]
            self.motor_trn_dadr[i] = copy.motor_trn_dadr[i]
            self.motor_trn_coef[i] = copy.motor_trn_coef[i]
        for i in range(Self.NTEN):
            self.tendon_stiffness[i] = copy.tendon_stiffness[i]
            self.tendon_spring_lo[i] = copy.tendon_spring_lo[i]
            self.tendon_spring_hi[i] = copy.tendon_spring_hi[i]
            self.tendon_trn_n[i] = copy.tendon_trn_n[i]
        for i in range(Self.NTEN * Self.WRAPS):
            self.tendon_trn_qadr[i] = copy.tendon_trn_qadr[i]
            self.tendon_trn_dadr[i] = copy.tendon_trn_dadr[i]
            self.tendon_trn_coef[i] = copy.tendon_trn_coef[i]
        for i in range(Self.NQ0):
            self.qpos0[i] = copy.qpos0[i]

    def __init__(out self, *, deinit move: Self):
        self.motor_gears = move.motor_gears^
        self.motor_dof_adr = move.motor_dof_adr^
        self.motor_ctrl_min = move.motor_ctrl_min^
        self.motor_ctrl_max = move.motor_ctrl_max^
        self.motor_ctrl_limited = move.motor_ctrl_limited^
        self.motor_force_limited = move.motor_force_limited^
        self.motor_force_min = move.motor_force_min^
        self.motor_force_max = move.motor_force_max^
        self.motor_kind = move.motor_kind^
        self.motor_kp = move.motor_kp^
        self.motor_kv = move.motor_kv^
        self.motor_dyn_tau = move.motor_dyn_tau^
        self.motor_act_adr = move.motor_act_adr^
        self.na = move.na
        self.bad_actuator = move.bad_actuator
        self.bad_actuator_code = move.bad_actuator_code
        self.motor_trn_n = move.motor_trn_n^
        self.motor_trn_qadr = move.motor_trn_qadr^
        self.motor_trn_dadr = move.motor_trn_dadr^
        self.motor_trn_coef = move.motor_trn_coef^
        self.tendon_stiffness = move.tendon_stiffness^
        self.tendon_spring_lo = move.tendon_spring_lo^
        self.tendon_spring_hi = move.tendon_spring_hi^
        self.tendon_wrap_overflow = move.tendon_wrap_overflow
        self.tendon_count_overflow = move.tendon_count_overflow
        self.tendon_trn_n = move.tendon_trn_n^
        self.tendon_trn_qadr = move.tendon_trn_qadr^
        self.tendon_trn_dadr = move.tendon_trn_dadr^
        self.tendon_trn_coef = move.tendon_trn_coef^
        self.ntendon = move.ntendon
        self.joint_is_limited = move.joint_is_limited^
        self.joint_qpos_adr = move.joint_qpos_adr^
        self.joint_range_min = move.joint_range_min^
        self.joint_range_max = move.joint_range_max^
        self.inertiafromgeom = move.inertiafromgeom
        self.settotalmass = move.settotalmass
        self.qpos0 = move.qpos0^
        self.nq = move.nq
        self.free_joint_qpos_adr = move.free_joint_qpos_adr
        self.nkey = move.nkey
        self.bad_keyframe_code = move.bad_keyframe_code
        self.key_time = move.key_time^
        self.key_nqpos = move.key_nqpos^
        self.key_nqvel = move.key_nqvel^
        self.key_nctrl = move.key_nctrl^
        self.key_qpos = move.key_qpos^
        self.key_qvel = move.key_qvel^
        self.key_ctrl = move.key_ctrl^


def _xml_find_joint_dof_adr(xml: String, jname: String) -> Int:
    """A named joint's DOF address in MuJoCo's element order, or -1.

    Delegates to `_xml_joint_adr_grouped`, which explains why the obvious
    linear text scan this used to be is WRONG: MuJoCo groups joints by body,
    and dog is the model where that stops coinciding with text order.
    """
    return _xml_joint_adr_grouped(xml, jname, False)


def _find_tag(sec: String, marker: String, start: Int) -> Int:
    """Index of the next REAL `marker` tag at or after `start`, else -1.

    "Real" means the character after the marker ends the element name, so
    `<position` does not match `<positionfoo` (and, historically, `<motor`
    must not match a longer name either).
    """
    var pos = start
    var n = sec.byte_length()
    var mlen = marker.byte_length()
    while pos < n:
        var t = sec.find(marker, pos)
        if t == -1:
            return -1
        var after_pos = t + mlen
        if after_pos >= n:
            return t
        var after = String(sec[byte=after_pos : after_pos + 1])
        if (
            after == " "
            or after == ">"
            or after == "/"
            or after == "\n"
            or after == "\t"
        ):
            return t
        pos = after_pos
    return -1


def _xml_joint_adr_grouped(xml: String, jname: String, want_qpos: Bool) -> Int:
    """A named joint's qpos/dof address in MuJoCo's element order.

    ⚠ MuJoCo'S ORDER IS NOT XML TEXT ORDER. It emits joints GROUPED BY BODY —
    all of body 0's, then body 1's, declaration order preserved inside each —
    and the two coincide only when every body declares its own joints BEFORE
    its nested `<body>` children. dm_control's dog does not: its `skull`
    declares 42 teeth after its child bodies, and its spine bodies nest before
    they joint. `full_parser` already reorders for exactly this reason
    (`_stable_group_by_body_joints`, defect 7) — this is the same correction on
    the COMPTIME side, which resolves actuator and tendon transmissions.

    THE DEFECT THIS FIXES. The comptime scanners walked the text linearly, so
    every one of dog's 38 actuators wrote its force at the wrong dof: `hip_L_
    supinate` drove dof 8 where MuJoCo drives 17, and the tail tendons drove a
    descending run of the wrong joints entirely. It is invisible at `ctrl = 0`
    — which is why the whole step measured exact and only a DRIVEN rollout
    diverged.

    Body ids are assigned in DFS order at each `<body` open, which is MuJoCo's
    body order, so accumulating widths over `(body_id, text_index)` reproduces
    the compiled layout. `want_qpos` picks NQ widths (free 7, ball 4) over NV
    ones (6, 3); they differ only for those two types.
    """
    var wb = _extract_section(xml, "worldbody")
    var n = wb.byte_length()
    var search_name = 'name="' + jname + '"'

    # Pass 1: every joint in text order, tagged with the body it belongs to.
    var jbody = List[Int]()
    var jwidth = List[Int]()
    var target = -1
    var pos = 0
    var next_body = 0
    var cur = 0  # the world body, which cannot carry a joint
    var stack = List[Int]()
    while pos < n:
        var t_open = _find_tag(wb, "<body", pos)
        var t_joint = _find_tag(wb, "<joint", pos)
        var t_close = wb.find("</body", pos)
        var t = _min_valid_pos(_min_valid_pos(t_open, t_joint), t_close)
        if t == -1:
            break
        var tag_end = wb.find(">", t)
        if tag_end == -1:
            break
        if t == t_close:
            if len(stack) > 0:
                cur = stack.pop()
            else:
                cur = 0
        elif t == t_open:
            # The id is consumed even by a childless body, or later siblings
            # would be numbered as though it had never existed.
            next_body += 1
            var self_closed = (
                tag_end >= 1
                and String(wb[byte = tag_end - 1 : tag_end]) == "/"
            )
            if not self_closed:
                stack.append(cur)
                cur = next_body
        else:
            var tag = String(wb[byte = t : tag_end + 1])
            var jtype = _trim(_extract_attr(tag, "type"))
            var w = 1
            if jtype == "ball":
                w = 4 if want_qpos else 3
            elif jtype == "free":
                w = 7 if want_qpos else 6
            if target < 0 and tag.find(search_name) != -1:
                target = len(jbody)
            jbody.append(cur)
            jwidth.append(w)
        pos = tag_end + 1

    if target < 0:
        return -1

    # Pass 2: sum the widths of every joint that MuJoCo emits before this one.
    var adr = 0
    var tbody = jbody[target]
    for i in range(len(jbody)):
        if jbody[i] < tbody or (jbody[i] == tbody and i < target):
            adr += jwidth[i]
    return adr


def _index_by_name_grouped(worldbody: String, marker: String, name: String) -> Int:
    """Ordinal of the named element in MuJoCo's ELEMENT order, or -1.

    The index twin of `_xml_joint_adr_grouped`: same body-grouping rule, but it
    returns a position in the element array rather than a qpos/dof address.
    MuJoCo emits `<joint>`s and `<site>`s grouped by body — all of body 0's,
    then body 1's, declaration order preserved inside each — so counting tags
    in raw text order is only right when every body declares its own elements
    BEFORE its nested `<body>` children.

    ⚠ `<worldbody>`'s OWN sites belong to body 0 and therefore come FIRST,
    ahead of every site declared inside a body, however early in the text those
    world-level sites appear. That is the whole of the finger / manipulator /
    stacker divergence: their `target` and `palm_touch` sites move.

    ⚠ SCANS `marker` ONLY, mirroring what the array builder scans. `_fill_model`
    looks for `"<joint"` and nothing else, so this must too — `<freejoint>` is
    already rewritten to `<joint type="free">` by `merge_mjcf` before either is
    reached, and adding a second marker here would number joints DIFFERENTLY
    from the array being indexed. A resolver has to mirror its builder, not
    MuJoCo.
    """
    var n = worldbody.byte_length()
    var ebody = List[Int]()
    var target = -1
    var pos = 0
    var next_body = 0
    var cur = 0  # the world body
    var stack = List[Int]()
    while pos < n:
        var t_open = _find_tag(worldbody, "<body", pos)
        var t_elem = _find_tag(worldbody, marker, pos)
        var t_close = worldbody.find("</body", pos)
        var t = _min_valid_pos(_min_valid_pos(t_open, t_elem), t_close)
        if t == -1:
            break
        var tag_end = worldbody.find(">", t)
        if tag_end == -1:
            break
        if t == t_close:
            if len(stack) > 0:
                cur = stack.pop()
            else:
                cur = 0
        elif t == t_open:
            # The id is consumed even by a childless body, or later siblings
            # would be numbered as though it had never existed.
            next_body += 1
            var self_closed = (
                tag_end >= 1
                and String(worldbody[byte = tag_end - 1 : tag_end]) == "/"
            )
            if not self_closed:
                stack.append(cur)
                cur = next_body
        else:
            var tag = String(worldbody[byte = t : tag_end + 1])
            if target < 0 and _trim(_extract_attr(tag, "name")) == name:
                target = len(ebody)
            ebody.append(cur)
        pos = tag_end + 1

    if target < 0:
        return -1

    var idx = 0
    var tbody = ebody[target]
    for i in range(len(ebody)):
        if ebody[i] < tbody or (ebody[i] == tbody and i < target):
            idx += 1
    return idx


def _min_valid_pos(a: Int, b: Int) -> Int:
    """The smaller of two find() results, ignoring -1."""
    if a == -1:
        return b
    if b == -1:
        return a
    return a if a < b else b


def _xml_find_joint_ref(xml: String, jname: String, deg_factor: Float64) -> Float64:
    """A named joint's `ref` (MuJoCo `qpos0`), in radians for angular joints.

    Only hinge/ball ranges and refs get the deg->rad conversion, matching
    `mjCJoint::Compile`; a slide `ref` is in metres. Returns 0 when the joint
    or the attribute is absent, which IS MuJoCo's default reference pose.
    """
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var search_name = 'name="' + jname + '"'
    while True:
        var t = _find_tag(wb, "<joint", scan_pos)
        if t == -1:
            return 0.0
        var tag_end = wb.find(">", t)
        if tag_end == -1:
            return 0.0
        var tag = String(wb[byte = t : tag_end + 1])
        if tag.find(search_name) != -1:
            var rs = _extract_attr(tag, "ref")
            if rs.byte_length() == 0:
                return 0.0
            var ts = _trim(_extract_attr(tag, "type"))
            var angular = ts == "" or ts == "hinge" or ts == "ball"
            return _parse_float(rs) * (deg_factor if angular else 1.0)
        scan_pos = tag_end + 1


def _xml_find_joint_qpos_adr(xml: String, jname: String) -> Int:
    """Return the QPOS address of a named joint, in worldbody DFS order.

    The twin of `_xml_find_joint_dof_adr`; they differ only for `free` (7 vs 6)
    and `ball` (4 vs 3) joints. A position servo needs BOTH — its `length` is a
    qpos read and its force lands on a dof — and fish is the first model where
    they diverge, since its root is a free joint ahead of every actuated hinge.
    """
    return _xml_joint_adr_grouped(xml, jname, True)


def _xml_find_joint_index(xml: String, jname: String) -> Int:
    """Return joint INDEX (0-based) of joint with the given name.

    Unlike _xml_find_joint_dof_adr which returns the DOF address,
    this returns the joint's position in the joints array.
    Returns -1 if not found.
    """
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var joint_idx = 0
    var search_name = 'name="' + jname + '"'
    while True:
        var t = wb.find("<joint", scan_pos)
        if t == -1:
            break
        if wb.byte_length() > t + 6:
            var after = String(wb[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                scan_pos = t + 6
                continue
        var tag_end = wb.find(">", t)
        if tag_end == -1:
            break
        var tag = String(wb[byte = t : tag_end + 1])
        if tag.find(search_name) != -1:
            return joint_idx
        joint_idx += 1
        scan_pos = tag_end + 1
    return -1


struct _JointAdrTable(Copyable, Movable):
    """Every joint's qpos/dof address and `ref`, from ONE worldbody walk.

    Replaces ~340 O(n) re-walks per model in `parse_xml_model_data` (each of
    which copied the whole <worldbody> and allocated a String per tag) with one
    walk plus linear lookups over ~75 entries.
    """

    var names: List[String]
    var qadr: List[Int]
    var dadr: List[Int]
    var refs: List[Float64]

    def __init__(out self):
        self.names = List[String]()
        self.qadr = List[Int]()
        self.dadr = List[Int]()
        self.refs = List[Float64]()

    def qpos_adr(self, jname: String) -> Int:
        """-1 when absent, matching `_xml_find_joint_qpos_adr`."""
        for i in range(len(self.names)):
            if self.names[i] == jname:
                return self.qadr[i]
        return -1

    def dof_adr(self, jname: String) -> Int:
        """-1 when absent, matching `_xml_find_joint_dof_adr`."""
        for i in range(len(self.names)):
            if self.names[i] == jname:
                return self.dadr[i]
        return -1

    def ref(self, jname: String) -> Float64:
        """0.0 when absent, which IS MuJoCo's default reference pose."""
        for i in range(len(self.names)):
            if self.names[i] == jname:
                return self.refs[i]
        return 0.0


def _build_joint_adr_table(xml: String, deg_factor: Float64) -> _JointAdrTable:
    """One walk of <worldbody>, emitting every joint's addresses.

    ⚠ TRANSCRIBED FROM `_xml_joint_adr_grouped`, PASS FOR PASS. MuJoCo emits
    joints GROUPED BY BODY -- all of body 0's, then body 1's, declaration order
    preserved inside each -- and that coincides with text order only when every
    body declares its joints before its nested <body> children. dm_control's
    dog does not. Changing the rule here changes every actuator and tendon
    transmission in the tree; the gate that catches it is
    `test_dog_actuator_transmission`'s `max|d(moment)|`.
    """
    var out = _JointAdrTable()
    var wb = _extract_section(xml, "worldbody")
    var n = wb.byte_length()

    # Pass 1: every joint in text order, tagged with the body it belongs to.
    var jbody = List[Int]()
    var jqw = List[Int]()
    var jdw = List[Int]()
    var pos = 0
    var next_body = 0
    var cur = 0  # the world body, which cannot carry a joint
    var stack = List[Int]()
    while pos < n:
        var t_open = _find_tag(wb, "<body", pos)
        var t_joint = _find_tag(wb, "<joint", pos)
        var t_close = wb.find("</body", pos)
        var t = _min_valid_pos(_min_valid_pos(t_open, t_joint), t_close)
        if t == -1:
            break
        var tag_end = wb.find(">", t)
        if tag_end == -1:
            break
        if t == t_close:
            if len(stack) > 0:
                cur = stack.pop()
            else:
                cur = 0
        elif t == t_open:
            # The id is consumed even by a childless body, or later siblings
            # would be numbered as though it had never existed.
            next_body += 1
            var self_closed = (
                tag_end >= 1
                and String(wb[byte = tag_end - 1 : tag_end]) == "/"
            )
            if not self_closed:
                stack.append(cur)
                cur = next_body
        else:
            var tag = String(wb[byte = t : tag_end + 1])
            var jtype = _trim(_extract_attr(tag, "type"))
            var qw = 1
            var dw = 1
            if jtype == "ball":
                qw = 4
                dw = 3
            elif jtype == "free":
                qw = 7
                dw = 6
            out.names.append(_trim(_extract_attr(tag, "name")))
            # `ref`, by `_xml_find_joint_ref`'s rule: deg->rad for angular
            # joints only, because a slide ref is in metres.
            var rs = _extract_attr(tag, "ref")
            var rv = Float64(0.0)
            if rs.byte_length() > 0:
                var angular = jtype == "" or jtype == "hinge" or jtype == "ball"
                rv = _parse_float(rs) * (deg_factor if angular else 1.0)
            out.refs.append(rv)
            jbody.append(cur)
            jqw.append(qw)
            jdw.append(dw)
        pos = tag_end + 1

    # Pass 2: for each joint, sum the widths of every joint MuJoCo emits first.
    for i in range(len(jbody)):
        var qa = 0
        var da = 0
        for j in range(len(jbody)):
            if jbody[j] < jbody[i] or (jbody[j] == jbody[i] and j < i):
                qa += jqw[j]
                da += jdw[j]
        out.qadr.append(qa)
        out.dadr.append(da)
    return out^


struct _ClassAttrCache(Copyable, Movable):
    """Class-attribute resolution for `parse_xml_model_data`'s actuator loop.

    Two layers, because the loop repeats itself in two different ways. It asks
    for 8 attributes per actuator, and many actuators share a class, so the
    (class, kind, attribute) memo below kills the repeats — dog's 38 actuators
    make 304 requests over ~190 distinct triples. What remains is one
    resolution per distinct triple, and each of those used to walk the whole
    document once per link of the inheritance chain; `_DefaultsIndex` locates
    every `<default>` block once instead.

    The memo is keyed on all three parts because the same attribute resolves
    differently for a `<motor>` than for a `<general>` of the same class.
    """

    var idx: _DefaultsIndex
    var keys: List[String]
    var vals: List[String]

    def __init__(out self, xml: String):
        self.idx = _build_defaults_index(xml)
        self.keys = List[String]()
        self.vals = List[String]()

    def get(
        mut self, xml: String, cls: String, tag_name: String, attr: String
    ) -> String:
        """`_class_attr_inherited`, memoized and index-backed.

        Equivalence with the rescanning original is not assumed: every
        (class, kind, attribute) triple these models use is diffed against it
        in `tests/physics3d/test_defaults_index_equivalence.mojo`.
        """
        var key = cls + "|" + tag_name + "|" + attr
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.vals[i]
        var v = _class_attr_inherited_indexed(xml, self.idx, cls, tag_name, attr)
        self.keys.append(key)
        self.vals.append(v)
        return v


def _attr_3way_cached(
    xml: String,
    elem: String,
    cls: String,
    tag_name: String,
    root_tag: String,
    name: String,
    mut cache: _ClassAttrCache,
) -> String:
    """`_attr_3way` with the class level memoized.

    Same three-level order — element, then class, then the root `<default>`
    block — so this is a drop-in. Only the middle lookup changes, from a fresh
    XML walk to a cache hit.
    """
    var v = _extract_attr(elem, name)
    if v.byte_length() > 0:
        return v
    var c = cache.get(xml, cls, tag_name, name)
    if c.byte_length() > 0:
        return c
    return _extract_attr(root_tag, name)


def parse_xml_model_data[NACT: Int, NJNT: Int, NQ0: Int, NTEN: Int, WRAPS: Int](xml: String) -> ComptimeActData[NACT, NJNT, NQ0, NTEN, WRAPS]:
    """Parse XML and return actuator/joint data as InlineArrays.

    Designed to be called at struct-level comptime:

        comptime _acd = parse_xml_model_data(Self.xml)

    GPU kernels then access Self._acd.motor_gears[i] etc. without String ops,
    bypassing the GPU kernel compiler's limitation with String operations.
    """
    var data = ComptimeActData[NACT, NJNT, NQ0, NTEN, WRAPS]()

    var xml_clean = _strip_xml_comments(xml)

    # ---- Compiler flags -------------------------------------------------------
    var angle_deg = _compiler_angle_is_deg(xml_clean)
    var compiler_t = xml_clean.find("<compiler")
    if compiler_t != -1:
        var compiler_end = xml_clean.find(">", compiler_t)
        if compiler_end != -1:
            var ctag = String(xml_clean[byte = compiler_t : compiler_end + 1])
            var ifg = _extract_attr(ctag, "inertiafromgeom")
            if _trim(ifg) == "true":
                data.inertiafromgeom = True
            var stm = _extract_attr(ctag, "settotalmass")
            var stm_trimmed = _trim(stm)
            if stm_trimmed.byte_length() > 0:
                data.settotalmass = _parse_float(stm_trimmed)

    var deg_factor = Float64(
        3.141592653589793 / 180.0
    ) if angle_deg else Float64(1.0)

    # ⚠ BUILT ONCE. The tendon and actuator loops below resolve joint names to
    # qpos/dof addresses; doing that per lookup re-walked the whole <worldbody>
    # (~340 times on dog, each copying ~75 KB). Measured at ~286 s + ~300 s of
    # dog's ~543 s comptime cost.
    var jtab = _build_joint_adr_table(xml_clean, deg_factor)

    # ---- Default motor gear / ctrlrange (fallbacks for per-motor values) -----
    var def_ctrl_min = Float64(-1.0)
    var def_ctrl_max = Float64(1.0)
    var def_gear = Float64(1.0)
    var def_sec_motor = _root_defaults(xml_clean)
    if def_sec_motor.byte_length() > 0:
        var mt = def_sec_motor.find("<motor")
        if mt != -1:
            var mte = def_sec_motor.find(">", mt)
            if mte != -1:
                var mtag = String(def_sec_motor[byte = mt : mte + 1])
                var mcr = _extract_attr(mtag, "ctrlrange")
                if mcr.byte_length() > 0:
                    var mparts = List[String]()
                    _split_spaces(mcr, mparts)
                    if len(mparts) >= 2:
                        def_ctrl_min = _parse_float(mparts[0])
                        def_ctrl_max = _parse_float(mparts[1])
                # `gear` was missing here until 2026-07-29 — see
                # `_xml_default_motor_gear` for what that silently cost.
                var mg = _extract_attr(mtag, "gear")
                if mg.byte_length() > 0:
                    def_gear = _parse_float(mg)

    # ---- Fixed tendons: joint/coef lists + the passive spring ----------------
    #
    # Parsed BEFORE the actuators because a `<position tendon="..."/>`
    # transmission resolves through here.
    var ten_sec = _extract_section(xml_clean, "tendon")
    var ten_names = List[String]()
    if ten_sec.byte_length() > 0:
        # ⚠ COUNTED BEFORE THE LOOP, because the loop's own bound is what hides
        # the problem: `while data.ntendon < NTEN` stops without a word.
        var n_fixed = _count_tag(ten_sec, "fixed")
        if n_fixed > NTEN:
            data.tendon_count_overflow = n_fixed - NTEN
        var tpos = 0
        while data.ntendon < NTEN:
            var ft = ten_sec.find("<fixed", tpos)
            if ft == -1:
                break
            var open_end = ten_sec.find(">", ft)
            if open_end == -1:
                break
            var fend = ten_sec.find("</fixed>", ft)
            if fend == -1:
                fend = ten_sec.byte_length()
            var ftag = String(ten_sec[byte = ft : open_end + 1])
            var ti = data.ntendon
            ten_names.append(_trim(_extract_attr(ftag, "name")))

            var st = _extract_attr(ftag, "stiffness")
            if st.byte_length() > 0:
                data.tendon_stiffness[ti] = _parse_float(st)

            # Joint/coef children.
            var body = String(ten_sec[byte = open_end + 1 : fend])
            var jscan = 0
            var n = 0
            var length0 = Float64(0)
            # ⚠ COUNT WHAT DOES NOT FIT. The bound used to be a bare `4` and
            # the loop simply stopped, writing `tendon_trn_n = 4` so every
            # consumer read a complete tendon. Overflow is now recorded and
            # `init_fields` raises on it — see `MAX_COMPTIME_TENDON_WRAPS`.
            while n < WRAPS:
                var jt = body.find("<joint", jscan)
                if jt == -1:
                    break
                var jte = body.find(">", jt)
                if jte == -1:
                    break
                var jtag = String(body[byte = jt : jte + 1])
                var jn = _trim(_extract_attr(jtag, "joint"))
                var cf = _extract_attr(jtag, "coef")
                var coef = _parse_float(cf) if cf.byte_length() > 0 else 1.0
                data.tendon_trn_qadr[ti * WRAPS + n] = jtab.qpos_adr(jn)
                data.tendon_trn_dadr[ti * WRAPS + n] = jtab.dof_adr(jn)
                data.tendon_trn_coef[ti * WRAPS + n] = coef
                length0 += coef * jtab.ref(jn)
                n += 1
                jscan = jte + 1
            data.tendon_trn_n[ti] = n
            # Keep scanning past the cap purely to size the overflow, so the
            # diagnostic can say HOW MANY were dropped rather than just that
            # some were.
            var extra = 0
            while True:
                var jt2 = body.find("<joint", jscan)
                if jt2 == -1:
                    break
                var jte2 = body.find(">", jt2)
                if jte2 == -1:
                    break
                extra += 1
                jscan = jte2 + 1
            if extra > data.tendon_wrap_overflow:
                data.tendon_wrap_overflow = extra

            # `springlength`: one value sets both bounds, two set the band.
            # ABSENT is the common case and is NOT zero — MuJoCo's compiler
            # collapses the band onto the tendon's length at qpos0, which for a
            # fixed tendon is `sum coef * jnt_ref`. Defaulting to 0 instead
            # would be right only for a model whose joints all have ref="0",
            # and silently wrong for any other.
            var sl = _extract_attr(ftag, "springlength")
            if sl.byte_length() > 0:
                var sparts = List[String]()
                _split_spaces(sl, sparts)
                if len(sparts) >= 2:
                    data.tendon_spring_lo[ti] = _parse_float(sparts[0])
                    data.tendon_spring_hi[ti] = _parse_float(sparts[1])
                elif len(sparts) == 1:
                    data.tendon_spring_lo[ti] = _parse_float(sparts[0])
                    data.tendon_spring_hi[ti] = _parse_float(sparts[0])
            else:
                data.tendon_spring_lo[ti] = length0
                data.tendon_spring_hi[ti] = length0

            data.ntendon = ti + 1
            tpos = fend + 1

    # ---- Actuators ------------------------------------------------------------
    #
    # `<motor>`, `<position>`, `<velocity>` and `<general>` are scanned TOGETHER
    # in document order, which is the order MuJoCo indexes actuators in —
    # scanning one tag type and then the other would permute `ctrl` on any model
    # that mixes them.
    #
    # ⚠⚠ `<velocity` WAS MISSING FROM THIS SCAN while `_count_model_elements`
    # (:2525) counted it, so NACT was right and the loop found nothing: those
    # slots kept `motor_trn_n == 0`, and both `apply_actions` paths skip a
    # zero-transmission actuator. A `<velocity>` model would have run with the
    # actuator applying NO FORCE AT ALL rather than the wrong one. The runtime
    # `full_parser` refuses `ACT_KIND_VELOCITY` outright, which is the only
    # reason this was never reachable — an env passing
    # `allow_unsupported_actuators=True` would have hit it silently.
    var act_sec = _extract_section(xml_clean, "actuator")
    var act_pos = 0
    var act_count = 0
    # ⚠ HOISTED OUT OF THE LOOP. `_root_defaults` is a pure function of the XML
    # that extracts and strips the whole <default> section; it was being rebuilt
    # once per actuator. The cache below does the same for the class level,
    # which `_class_attr_inherited` otherwise re-walks per lookup.
    var rootdef = _root_defaults(xml_clean)
    var cacache = _ClassAttrCache(xml_clean)
    while act_count < NACT:
        var nm = _find_tag(act_sec, "<motor", act_pos)
        var npos = _find_tag(act_sec, "<position", act_pos)
        var nvel = _find_tag(act_sec, "<velocity", act_pos)
        var ngen = _find_tag(act_sec, "<general", act_pos)
        var t = _min_valid_pos(
            _min_valid_pos(nm, npos), _min_valid_pos(nvel, ngen)
        )
        if t == -1:
            break
        var is_position = t == npos
        var is_velocity = t == nvel
        var is_general = t == ngen
        var tag_end = act_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(act_sec[byte = t : tag_end + 1])

        # Attributes resolve element -> `class="..."` -> top-level default.
        # `<general>` is the only kind that reads the two outer levels today;
        # `<motor>`/`<position>` keep the root-only lookup they had, so this
        # cannot change any previously parsed model.
        var elem_cls = _trim(_extract_attr(tag, "class"))
        # Single-assignment throughout: reassigning a `String` var inside this
        # loop makes the COMPTIME INTERPRETER fail outright ("write clobbers a
        # pointer region" out of `String._iadd`'s memcpy), not merely slowly.
        var tag_name = String("general") if is_general else (
            String("position") if is_position else (
                String("velocity") if is_velocity else String("motor")
            )
        )
        var root_tag = _first_tag(rootdef, tag_name)

        data.motor_kind[act_count] = ACT_KIND_VELOCITY if is_velocity else (
            ACT_KIND_POSITION if (is_position or is_general) else ACT_KIND_MOTOR
        )

        var g = _attr_3way_cached(xml_clean, tag, elem_cls, tag_name, root_tag, "gear", cacache)
        if g.byte_length() > 0:
            data.motor_gears[act_count] = _parse_float(g)
        else:
            data.motor_gears[act_count] = def_gear

        # ctrlrange: element, then `class="..."`, then the top-level default.
        # The class level is where quadruped keeps all three of its ranges.
        #
        # ⚠ Resolved AND CONSUMED here rather than at the bottom of the loop
        # where it used to live. A `String` that stays live across the
        # transmission block below makes the COMPTIME INTERPRETER give up —
        # "interpreting memcpy can't get dst memory from the interpreter /
        # write clobbers a pointer region" out of `_extract_attr`, which is a
        # hard failure to compile, not a slow build. Seven other `_attr_3way`
        # calls in this same loop are fine because each is consumed on the
        # spot. Keep every string lookup here short-lived.
        var cr = _attr_3way_cached(xml_clean, tag, elem_cls, tag_name, root_tag, "ctrlrange", cacache)
        var used_default = True
        if cr.byte_length() > 0:
            var parts = List[String]()
            _split_spaces(cr, parts)
            if len(parts) >= 2:
                data.motor_ctrl_min[act_count] = _parse_float(parts[0])
                data.motor_ctrl_max[act_count] = _parse_float(parts[1])
                used_default = False
        if used_default:
            data.motor_ctrl_min[act_count] = def_ctrl_min
            data.motor_ctrl_max[act_count] = def_ctrl_max

        # `ctrllimited` — see the field comment on
        # `ComptimeActData.motor_ctrl_limited` for the measured semantics.
        # Identical shape to `forcelimited` below: "auto" means limited iff a
        # range was DEFINED, "0 0" is the undefined marker, and an explicit
        # true/false overrides.
        #
        # ⚠ `used_default` IS THE "no range was defined" TEST, and it has to
        # be. `_attr_3way_cached` already searched element -> class chain ->
        # root `<default>`, so `used_default` means no level supplied one and
        # `motor_ctrl_min/max` now hold OUR (-1, 1) fallback rather than
        # anything from the model. Deciding limitedness from those values
        # instead would read the fallback as a real range and re-introduce
        # exactly the unconditional +-1 clamp this field exists to remove.
        var c_lo = data.motor_ctrl_min[act_count]
        var c_hi = data.motor_ctrl_max[act_count]
        var cl = _trim(
            _attr_3way_cached(
                xml_clean, tag, elem_cls, tag_name, root_tag, "ctrllimited",
                cacache,
            )
        )
        var c_limited = (not used_default) and (c_lo != 0.0 or c_hi != 0.0)
        if cl == "true" or cl == "1":
            c_limited = True
        elif cl == "false" or cl == "0":
            c_limited = False
        data.motor_ctrl_limited[act_count] = 1 if c_limited else 0

        # forcerange / forcelimited — see the field comments on
        # `ComptimeActData.motor_force_limited` for the measured semantics.
        # Same 3-way resolution and the same keep-it-short-lived rule as
        # ctrlrange above.
        var fr = _attr_3way_cached(
            xml_clean, tag, elem_cls, tag_name, root_tag, "forcerange", cacache
        )
        var f_lo = Float64(0)
        var f_hi = Float64(0)
        if fr.byte_length() > 0:
            var fparts = List[String]()
            _split_spaces(fr, fparts)
            if len(fparts) >= 2:
                f_lo = _parse_float(fparts[0])
                f_hi = _parse_float(fparts[1])
        data.motor_force_min[act_count] = f_lo
        data.motor_force_max[act_count] = f_hi
        # `forcelimited` defaults to "auto" = limited iff the range is defined,
        # and "0 0" IS the undefined marker (measured — an explicit
        # `forcerange="0 0"` still reports forcelimited 0). An explicit
        # true/false overrides. MuJoCo REFUSES `forcelimited="true"` with no
        # range, so the true-with-zero-range combination cannot reach us.
        var fl = _trim(
            _attr_3way_cached(
                xml_clean, tag, elem_cls, tag_name, root_tag, "forcelimited",
                cacache,
            )
        )
        var limited = f_lo != 0.0 or f_hi != 0.0
        if fl == "true" or fl == "1":
            limited = True
        elif fl == "false" or fl == "0":
            limited = False
        data.motor_force_limited[act_count] = 1 if limited else 0

        # `<position>` is `<general>` with gaintype=fixed, biastype=affine:
        # gainprm = [kp, 0, 0] and biasprm = [0, -kp, -kv]. MuJoCo's kp
        # default is 1 and kv's is 0.
        if is_position:
            # ⚠⚠ RESOLVED 3-WAY (element -> class chain -> root `<default>`).
            # These two read `_extract_attr(tag, ...)` — the ELEMENT ONLY —
            # until 2026-08-13, while EVERY neighbouring attribute (`gear`,
            # `ctrlrange`, `forcerange`, `forcelimited`, `gaintype`,
            # `biastype`, `gainprm`, `biasprm`, and `<velocity>`'s own `kv`)
            # already went through `_attr_3way_cached`. A gain declared in a
            # `<default>` class was therefore MISSED and MuJoCo's defaults took
            # over — kp 1, kv 0.
            #
            # ⚠ IT WAS NOT A CLASS-CHAIN DEFECT, and it was filed as one first.
            # `_class_attr_inherited` walks the chain correctly. The
            # discriminating fixture is ONE class, no nesting, `kp` in the
            # class: it still parsed 1.0. See
            # `tests/physics3d/test_position_gain_defaults.mojo`.
            #
            # ⚠ WHY IT HID: the fallback is a PLAUSIBLE number. Measured on the
            # two SO-ARM ports, which is where it surfaced —
            #
            #     SO-100  kp 50     -> 1.0   servo 50x weak; the arm still
            #                                moved TOWARD its target, just far
            #                                too slowly, i.e. "bad tuning"
            #     SO-101  kp 998.22 -> 1.0   torque ~1 N.m short of the gravity
            #             kv 2.731  -> 0.0   load, so the arm FELL to its limits
            #
            # Nothing raises either way, and every static gate passed. Only a
            # lockstep rollout against MuJoCo could see it.
            #
            # ⚠ BLAST RADIUS AT THE TIME OF THE FIX: nil. Every model in the
            # tree with `<position>` actuators (fish, manipulator, sawyer, both
            # SO-ARMs) writes its gains on the ELEMENT, so all five compared
            # EXACT against `mjModel` both before and after. This change can
            # only make previously-broken input work; it cannot move a model
            # that was already right.
            var kp_s = _attr_3way_cached(
                xml_clean, tag, elem_cls, tag_name, root_tag, "kp", cacache
            )
            data.motor_kp[act_count] = (
                _parse_float(kp_s) if kp_s.byte_length() > 0 else 1.0
            )
            var kv_s = _attr_3way_cached(
                xml_clean, tag, elem_cls, tag_name, root_tag, "kv", cacache
            )
            data.motor_kv[act_count] = (
                _parse_float(kv_s) if kv_s.byte_length() > 0 else 0.0
            )
        elif is_velocity:
            # `<velocity kv=K>` is `<general>` with gaintype=fixed,
            # biastype=affine, gainprm = [K, 0, 0] and biasprm = [0, 0, -K] —
            # i.e. `force = K*(ctrl - vel)`. MEASURED on MuJoCo 3.10.0, all four
            # spellings (bare, kv=, gear=, and the equivalent `<general>`):
            #
            #     <velocity kv="3"/>  -> gainprm [3 0 0]  biasprm [0 0 -3]
            #     <velocity/>         -> gainprm [1 0 0]  biasprm [0 0 -1]
            #
            # ⚠ The kv DEFAULT IS 1, not 0. `<position>`'s kv defaults to 0
            # because there the damping term is optional; here it IS the
            # actuator, and a 0 default would give a dead motor.
            #
            # ⚠ Resolved 3-way (element -> class -> root default) because the
            # class level is live: a `<default><velocity kv="7"/></default>`
            # reaches an attribute-less `<velocity/>`, measured above.
            # `<position>` above used to read the element only — this note
            # recorded that as "a separate, pre-existing gap, not widened
            # here", and it stayed unfixed until the SO-ARM ports gave it two
            # consumers. Both branches resolve the same way now.
            var vkv_s = _attr_3way_cached(
                xml_clean, tag, elem_cls, tag_name, root_tag, "kv", cacache
            )
            var vkv = _parse_float(vkv_s) if vkv_s.byte_length() > 0 else 1.0
            # gainprm[0] AND -biasprm[2] are both K, so both slots carry it.
            data.motor_kp[act_count] = vkv
            data.motor_kv[act_count] = vkv
        elif is_general:
            # Spelled-out form of the same law. MuJoCo (mj_fwdActuation):
            #     force = gain * [ctrl | act] + bias
            #     gain  = gainprm[0]                          (gaintype fixed)
            #     bias  = biasprm[0] + biasprm[1]*len + biasprm[2]*vel (affine)
            # which is our POSITION path exactly when biasprm[0] == 0 and
            # biasprm[1] == -gainprm[0]; then kp = gain and kv = -biasprm[2].
            #
            # With NO bias at all (`biastype` absent or "none", MuJoCo's
            # default `mjBIAS_NONE`) the same formula degenerates to
            # `force = gain * u`, which is our MOTOR path carrying a gain.
            # That is dog: 38 `<general dyntype="filter" gainprm="0.02"/>`
            # with no biastype. Classifying by TAG NAME alone called those
            # position servos and refused them.
            #
            # Anything else is a different actuator and is refused below.
            var gaintype = _trim(_attr_3way_cached(xml_clean, tag, elem_cls, tag_name, root_tag, "gaintype", cacache))
            var biastype = _trim(_attr_3way_cached(xml_clean, tag, elem_cls, tag_name, root_tag, "biastype", cacache))
            var gainprm = _attr_3way_cached(xml_clean, tag, elem_cls, tag_name, root_tag, "gainprm", cacache)
            var biasprm = _attr_3way_cached(xml_clean, tag, elem_cls, tag_name, root_tag, "biasprm", cacache)
            var dyntype = _trim(_attr_3way_cached(xml_clean, tag, elem_cls, tag_name, root_tag, "dyntype", cacache))
            var dynprm = _attr_3way_cached(xml_clean, tag, elem_cls, tag_name, root_tag, "dynprm", cacache)

            var gain = _nth_float(gainprm, 0, 1.0)  # MuJoCo gainprm default 1
            var b0 = _nth_float(biasprm, 0, 0.0)
            var b1 = _nth_float(biasprm, 1, 0.0)
            var b2 = _nth_float(biasprm, 2, 0.0)

            var no_bias = biastype.byte_length() == 0 or biastype == "none"

            if gaintype.byte_length() > 0 and gaintype != "fixed":
                if data.bad_actuator < 0:
                    data.bad_actuator = act_count
                    data.bad_actuator_code = 0
            elif not (no_bias or biastype == "affine"):
                if data.bad_actuator < 0:
                    data.bad_actuator = act_count
                    data.bad_actuator_code = 1
            elif (not no_bias) and b0 != 0.0:
                if data.bad_actuator < 0:
                    data.bad_actuator = act_count
                    data.bad_actuator_code = 2
            elif (not no_bias) and b1 != -gain and b1 != 0.0:
                if data.bad_actuator < 0:
                    data.bad_actuator = act_count
                    data.bad_actuator_code = 3

            # A bias-free `<general>` is a gained torque motor, not a servo.
            # The unconditional ACT_KIND_POSITION written above is corrected
            # here, once `biastype` has actually been read.
            #
            # `biasprm[1]` is what separates the two servo laws, and it is the
            # ONLY thing that does — MuJoCo writes the same gaintype/biastype
            # for both:
            #     b1 == -gain  ->  force = gain*(u - length) - kv*vel  POSITION
            #     b1 == 0      ->  force = gain*u            - kv*vel  VELOCITY
            # Code 3 used to reject `b1 == 0` outright, so a `<general>` spelled
            # the way MuJoCo itself expands `<velocity>` was refused while the
            # `<velocity>` tag was silently ignored. Both spellings compile to
            # an identical mjModel (measured), so both land here.
            #
            # ⚠ gain and -b2 are INDEPENDENT. `<velocity>` happens to set both
            # to K, but `gainprm="5 0 0" biasprm="0 0 -3"` is legal and means
            # `force = 5*u - 3*vel`; kp/kv carry them separately for that
            # reason. Do not collapse them.
            if no_bias:
                data.motor_kind[act_count] = ACT_KIND_MOTOR
            elif b1 == 0.0 and gain != 0.0:
                # `gain != 0` only to keep the two branches disjoint: at gain 0
                # both laws collapse to `force = -kv*vel`, so which one is
                # picked cannot matter, and POSITION keeps it.
                data.motor_kind[act_count] = ACT_KIND_VELOCITY

            data.motor_kp[act_count] = gain
            data.motor_kv[act_count] = 0.0 if no_bias else -b2

            # mjDYN_FILTER: act_dot = (ctrl - act) / dynprm[0], one activation
            # variable per actuator, integrated by the same Euler step as qvel.
            if dyntype.byte_length() == 0 or dyntype == "none":
                data.motor_dyn_tau[act_count] = 0.0
                data.motor_act_adr[act_count] = -1
            elif dyntype == "filter":
                data.motor_dyn_tau[act_count] = _nth_float(dynprm, 0, 1.0)
                data.motor_act_adr[act_count] = data.na
                data.na += 1
            else:
                if data.bad_actuator < 0:
                    data.bad_actuator = act_count
                    data.bad_actuator_code = 4

        # Transmission: a joint is one (qadr, dadr, 1) triple; a tendon is the
        # tendon's own joint/coef list.
        var jname = _trim(_extract_attr(tag, "joint"))
        var tname = _trim(_extract_attr(tag, "tendon"))
        if jname.byte_length() > 0:
            var dadr = jtab.dof_adr(jname)
            var qadr = jtab.qpos_adr(jname)
            data.motor_dof_adr[act_count] = dadr
            if dadr >= 0:
                data.motor_trn_qadr[act_count * WRAPS] = qadr
                data.motor_trn_dadr[act_count * WRAPS] = dadr
                data.motor_trn_coef[act_count * WRAPS] = 1.0
                data.motor_trn_n[act_count] = 1
        elif tname.byte_length() > 0:
            for ti in range(len(ten_names)):
                if ten_names[ti] != tname:
                    continue
                var n = data.tendon_trn_n[ti]
                for k in range(n):
                    data.motor_trn_qadr[act_count * WRAPS + k] = (
                        data.tendon_trn_qadr[ti * WRAPS + k]
                    )
                    data.motor_trn_dadr[act_count * WRAPS + k] = (
                        data.tendon_trn_dadr[ti * WRAPS + k]
                    )
                    data.motor_trn_coef[act_count * WRAPS + k] = (
                        data.tendon_trn_coef[ti * WRAPS + k]
                    )
                data.motor_trn_n[act_count] = n
                if n > 0:
                    data.motor_dof_adr[act_count] = data.tendon_trn_dadr[ti * WRAPS]
                break

        act_count += 1
        # Past the whole opening tag. `t + 6` was enough while the scanned tags
        # were `<motor`/`<position`, but `<general` is longer than the shortest
        # marker and re-scanning from inside it would rematch the same element.
        act_pos = tag_end + 1

    # ---- Default joint limited from <default> section -------------------------
    var def_limited = False
    var def_sec = _root_defaults(xml_clean)
    if def_sec.byte_length() > 0:
        var jpos = def_sec.find("<joint")
        if jpos != -1:
            var tag_end = def_sec.find(">", jpos)
            if tag_end != -1:
                var tag = String(def_sec[byte = jpos : tag_end + 1])
                var lim = _extract_attr(tag, "limited")
                if lim == "true" or lim == "1":
                    def_limited = True

    # ---- Joint data -----------------------------------------------------------
    var wb = _extract_section(xml_clean, "worldbody")
    var jnt_pos = 0
    var jnt_count = 0
    var qpos_adr = 0
    while jnt_count < NJNT:
        var t = wb.find("<joint", jnt_pos)
        if t == -1:
            break
        if wb.byte_length() > t + 6:
            var after = String(wb[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                jnt_pos = t + 6
                continue
        data.joint_qpos_adr[jnt_count] = qpos_adr
        var tag_end = wb.find(">", t)
        if tag_end != -1:
            var tag = String(wb[byte = t : tag_end + 1])
            # Limited. Explicit attribute wins; otherwise MuJoCo's
            # `compiler/autolimits` (default "true" since 2.2.2) makes a joint
            # limited exactly when it declares a `range`. Deriving it from the
            # range is what lets this class-blind scan agree with the compiler:
            # swimmer puts `limited` on TWO default classes (`swimmer` true,
            # `free` false), neither of which this path can resolve, yet every
            # joint's range presence already encodes the same answer. The root
            # `<default>` value is the last resort, for a joint with neither.
            var lim = _extract_attr(tag, "limited")
            var rng_attr = _extract_attr(tag, "range")
            if lim == "true" or lim == "1":
                data.joint_is_limited[jnt_count] = True
            elif lim == "false" or lim == "0":
                data.joint_is_limited[jnt_count] = False
            elif rng_attr.byte_length() > 0:
                data.joint_is_limited[jnt_count] = True
            else:
                data.joint_is_limited[jnt_count] = def_limited
            # Range. deg→rad applies to ANGULAR joints only (MuJoCo's
            # mjCJoint::Compile gates on HINGE/BALL) — a slide range is in
            # metres. An absent `type` means hinge, MuJoCo's default.
            #
            # Note this path reads `type` off the element and does not resolve
            # default classes, so a class that sets `type="slide"` on an
            # element that omits it would be misread as angular. No model in
            # the repo does that; `full_parser` resolves the class properly.
            var range_str = _extract_attr(tag, "range")
            if range_str.byte_length() > 0:
                var ts = _trim(_extract_attr(tag, "type"))
                var angular = ts == "" or ts == "hinge" or ts == "ball"
                var rf = deg_factor if angular else Float64(1.0)
                var parts = List[String]()
                _split_spaces(range_str, parts)
                if len(parts) >= 1:
                    data.joint_range_min[jnt_count] = (
                        _parse_float(parts[0]) * rf
                    )
                if len(parts) >= 2:
                    data.joint_range_max[jnt_count] = (
                        _parse_float(parts[1]) * rf
                    )
            # Extract ref value (MuJoCo joint reference → qpos0 for slide/hinge).
            # deg→rad applies exactly as it does to `range` above: `ref` is an
            # ANGLE for hinge/ball and a LENGTH for slide, and MuJoCo converts
            # it with the same compiler angle unit. Missing this made finger's
            # `ref="-90"` land in qpos0 as -90 rad instead of -pi/2.
            var ref_str = _extract_attr(tag, "ref")
            if ref_str.byte_length() > 0:
                var rts = _trim(_extract_attr(tag, "type"))
                var r_angular = rts == "" or rts == "hinge" or rts == "ball"
                var rrf = deg_factor if r_angular else Float64(1.0)
                data.qpos0[qpos_adr] = _parse_float(ref_str) * rrf
            # Advance qpos_adr, track free joint
            var jtype = _extract_attr(tag, "type")
            if jtype == "free":
                if data.free_joint_qpos_adr == -1:
                    data.free_joint_qpos_adr = qpos_adr
                # Extract enclosing body's pos for free joint initial translation.
                # Find the last <body before position t by scanning forward.
                var last_body_start = -1
                var bscan = 0
                while True:
                    var bp = wb.find("<body", bscan)
                    if bp == -1 or bp >= t:
                        break
                    last_body_start = bp
                    bscan = bp + 5
                if last_body_start >= 0:
                    var be = wb.find(">", last_body_start)
                    if be != -1:
                        var btag = String(
                            wb[byte = last_body_start : be + 1]
                        )
                        var bpos = _extract_attr(btag, "pos")
                        if bpos.byte_length() > 0:
                            var bparts = List[String]()
                            _split_spaces(bpos, bparts)
                            if len(bparts) >= 3:
                                data.qpos0[qpos_adr + 0] = _parse_float(
                                    bparts[0]
                                )
                                data.qpos0[qpos_adr + 1] = _parse_float(
                                    bparts[1]
                                )
                                data.qpos0[qpos_adr + 2] = _parse_float(
                                    bparts[2]
                                )
                # qw=1 (identity quaternion) set later in fallback block
                qpos_adr += 7
            elif jtype == "ball":
                qpos_adr += 4
            else:
                qpos_adr += 1
        jnt_count += 1
        jnt_pos = t + 6

    # ---- init_qpos from <custom><numeric name="init_qpos" data="..."/> -------
    var custom_sec = _extract_section(xml_clean, "custom")
    if custom_sec.byte_length() > 0:
        var num_pos = 0
        while True:
            var t = custom_sec.find("<numeric", num_pos)
            if t == -1:
                break
            var tag_end = custom_sec.find(">", t)
            if tag_end == -1:
                break
            var tag = String(custom_sec[byte = t : tag_end + 1])
            var nname = _extract_attr(tag, "name")
            if _trim(nname) == "init_qpos":
                var ndata = _extract_attr(tag, "data")
                var parts = List[String]()
                _split_spaces(ndata, parts)
                var count = len(parts)
                if count > 64:
                    count = 64
                for i in range(count):
                    data.qpos0[i] = _parse_float(parts[i])
                data.nq = count
                break
            num_pos = t + 7

    # If no explicit init_qpos was found, use qpos0 values from joint ref
    # attributes (already stored above).  Set nq so reset_data applies them.
    if data.nq == 0 and qpos_adr > 0:
        data.nq = qpos_adr
        # For free joints, ensure qw=1 (identity quaternion)
        if data.free_joint_qpos_adr >= 0:
            data.qpos0[data.free_joint_qpos_adr + 3] = 1.0

    # ---- <keyframe><key qpos= qvel= ctrl= time=> ---------------------------
    #
    # ⚠ RECORDED, NOT APPLIED. See the field block on `ComptimeActData`:
    # `mj_resetData` ignores keyframes, so `reset_data` must too.
    #
    # ⚠ A WRONG-LENGTH ATTRIBUTE IS REJECTED RATHER THAN PADDED. MuJoCo pads a
    # SHORT one, but measurably not from `qpos0`: for a model whose
    # `qpos0[7]` is 0.00436332 (a `ref="0.25"` in degrees), a short `qpos`
    # comes back with 0.25 in that slot — the RAW attribute value, before unit
    # conversion. That is spec-level default state leaking through, it differs
    # from anything else in the model, and it is not worth reproducing.
    # Nothing real depends on it: across Menagerie's 66 keyframed models,
    # 145 of 145 attributes are EXACTLY full length and none is short.
    var kf_sec = _extract_section(xml_clean, "keyframe")
    if kf_sec.byte_length() > 0:
        var kpos = 0
        var kcount = 0
        while True:
            var t = _find_tag(kf_sec, "<key", kpos)
            if t == -1:
                break
            var tag_end = kf_sec.find(">", t)
            if tag_end == -1:
                break
            var ktag = String(kf_sec[byte = t : tag_end + 1])
            kpos = tag_end + 1

            if kcount >= MAX_COMPTIME_KEYFRAMES:
                data.bad_keyframe_code = 1
                break

            # `act` / `mpos` / `mquat` are REJECTED, not dropped. Zero of the
            # 147 keyframe attributes in Menagerie use any of them (na == 0 in
            # all 66 models), so this refuses nothing that exists — but a
            # silently ignored `act` would be a wrong actuator state at reset
            # with nothing to notice it. Same call as `<pair gap=>`.
            if (
                _trim(_extract_attr(ktag, "act")).byte_length() > 0
                or _trim(_extract_attr(ktag, "mpos")).byte_length() > 0
                or _trim(_extract_attr(ktag, "mquat")).byte_length() > 0
            ):
                data.bad_keyframe_code = 2
                break

            var ktime = _trim(_extract_attr(ktag, "time"))
            if ktime.byte_length() > 0:
                data.key_time[kcount] = _parse_float(ktime)

            var kq = _trim(_extract_attr(ktag, "qpos"))
            if kq.byte_length() > 0:
                var parts = List[String]()
                _split_spaces(kq, parts)
                var n = len(parts)
                if n > NQ0:
                    n = NQ0
                for i in range(n):
                    data.key_qpos[kcount * NQ0 + i] = _parse_float(
                        parts[i]
                    )
                data.key_nqpos[kcount] = len(parts)

            var kv = _trim(_extract_attr(ktag, "qvel"))
            if kv.byte_length() > 0:
                var parts_v = List[String]()
                _split_spaces(kv, parts_v)
                var nv_ = len(parts_v)
                if nv_ > NQ0:
                    nv_ = NQ0
                for i in range(nv_):
                    data.key_qvel[kcount * NQ0 + i] = _parse_float(
                        parts_v[i]
                    )
                data.key_nqvel[kcount] = len(parts_v)

            var kc = _trim(_extract_attr(ktag, "ctrl"))
            if kc.byte_length() > 0:
                var parts_c = List[String]()
                _split_spaces(kc, parts_c)
                var nc = len(parts_c)
                if nc > NACT:
                    nc = NACT
                for i in range(nc):
                    data.key_ctrl[kcount * NACT + i] = _parse_float(
                        parts_c[i]
                    )
                data.key_nctrl[kcount] = len(parts_c)

            kcount += 1
        data.nkey = kcount

    return data^


# =============================================================================
# ComptimeRenderData — pre-computed rendering data from XML
# =============================================================================


struct ComptimeRenderData(Copyable, Movable):
    """Precomputed rendering data for ModelRenderer use.

    Stores results of lightweight XML parsing in InlineArrays so that
    rendering functions can access them without re-parsing the full XML.
    Avoids the comptime interpreter crash caused by calling parse_xml_full
    multiple times for large models (25+ bodies).

    Usage:
        comptime _rcd = parse_xml_render_data(Self.xml)
        # In rendering functions:  Self._rcd.geom_type[i]  (no re-parse)
    """

    # Counts
    var ngeom: Int
    var nlight: Int
    var ncam: Int
    var ntex: Int
    var nmat: Int
    var nsite: Int

    # Geoms (max 64)
    var geom_body_id: InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS]
    var geom_type: InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS]
    var geom_pos_x: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_pos_y: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_pos_z: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_quat_x: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_quat_y: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_quat_z: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_quat_w: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_radius: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_half_length: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_half_x: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_half_y: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_half_z: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_rgba_r: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_rgba_g: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_rgba_b: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_rgba_a: InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS]
    var geom_material_id: InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS]
    var geom_mesh_id: InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS]  # index into mesh_names[], -1 if not mesh
    var geom_group: InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS]
    """MuJoCo's `geom group`, default 0.

    ⚠ THIS IS VISIBILITY, and ignoring it is why dog rendered as a skeleton.
    MuJoCo's viewer draws groups 0-2 and hides the rest; dm_control's dog puts
    its collision capsules in group 3 (`rgba="0 0.5 0.5"`, the teal you see)
    and its 162 bone meshes in group 5. Drawing every group means drawing the
    collision proxy as if it were the model."""


# ⚠ NOTHING ABOUT `<skin>` IS RECORDED HERE EITHER, for the same reason as body
# names above — and dog is the model that proved it. Resolving
# `<skin material=>` -> `<material texture=>` -> `<texture file=>` needs an
# attribute read off a tag sliced out of the asset section, and that read is a
# comptime compile failure the moment it HITS. `ModelDefFromXML.render_skin`
# does the whole resolution at RUNTIME from `Self.xml`; the only comptime
# question left is "does this model have a skin at all", which is a `find` and
# never a slice.

    # Mesh assets (max 16) — name and file path for STL loading
    var nmesh: Int
    var mesh_names: InlineArray[String, MAX_COMPTIME_RENDER_MESHES]
    var mesh_files: InlineArray[String, MAX_COMPTIME_RENDER_MESHES]

    # Lights (max 8)
    var light_dir_x: InlineArray[Float64, 8]
    var light_dir_y: InlineArray[Float64, 8]
    var light_dir_z: InlineArray[Float64, 8]
    var light_diffuse_r: InlineArray[Float64, 8]
    var light_diffuse_g: InlineArray[Float64, 8]
    var light_diffuse_b: InlineArray[Float64, 8]
    var light_specular_r: InlineArray[Float64, 8]
    var light_specular_g: InlineArray[Float64, 8]
    var light_specular_b: InlineArray[Float64, 8]
    var light_ambient_r: InlineArray[Float64, 8]
    var light_ambient_g: InlineArray[Float64, 8]
    var light_ambient_b: InlineArray[Float64, 8]
    var light_directional: InlineArray[Bool, 8]
    var light_castshadow: InlineArray[Bool, 8]
    var light_exponent: InlineArray[Float64, 8]

    # Cameras (max 8)
    var cam_pos_x: InlineArray[Float64, 8]
    var cam_pos_y: InlineArray[Float64, 8]
    var cam_pos_z: InlineArray[Float64, 8]
    var cam_quat_x: InlineArray[Float64, 8]
    var cam_quat_y: InlineArray[Float64, 8]
    var cam_quat_z: InlineArray[Float64, 8]
    var cam_quat_w: InlineArray[Float64, 8]
    var cam_fovy: InlineArray[Float64, 8]
    var cam_mode: InlineArray[Int, 8]
    var cam_body_id: InlineArray[Int, 8]
    var cam_target_body: InlineArray[Int, 8]
    """Body index a `mode="targetbody"` camera aims at; -1 when it has none.

    Resolved at parse time from `target="..."`, because the render-time re-aim
    happens every frame and must not do string work."""

    # Textures (max MAX_COMPTIME_TEXTURES)
    var tex_type: InlineArray[Int, MAX_COMPTIME_TEXTURES]
    var tex_builtin: InlineArray[Int, MAX_COMPTIME_TEXTURES]
    var tex_rgb1_r: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_rgb1_g: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_rgb1_b: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_rgb2_r: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_rgb2_g: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_rgb2_b: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_names: InlineArray[String, MAX_COMPTIME_TEXTURES]  # texture name (for material lookup)
    var tex_files: InlineArray[String, MAX_COMPTIME_TEXTURES]  # texture file path (PNG)
    # `mark`/`markrgb`/`random`: MuJoCo sprinkles marks INTO the generated
    # texture. On a `builtin="gradient"` skybox with `mark="random"` that is a
    # starfield — which is exactly what dm_control's common/skybox.xml asks for
    # and what a two-colour gradient cannot reproduce. 0=none 1=edge 2=cross
    # 3=random; only `random` is rendered, and only on the skybox.
    var tex_mark: InlineArray[Int, MAX_COMPTIME_TEXTURES]
    var tex_markrgb_r: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_markrgb_g: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_markrgb_b: InlineArray[Float64, MAX_COMPTIME_TEXTURES]
    var tex_random: InlineArray[Float64, MAX_COMPTIME_TEXTURES]  # mark density

    # Materials (max MAX_COMPTIME_MATERIALS)
    var mat_rgba_r: InlineArray[Float64, MAX_COMPTIME_MATERIALS]
    var mat_rgba_g: InlineArray[Float64, MAX_COMPTIME_MATERIALS]
    var mat_rgba_b: InlineArray[Float64, MAX_COMPTIME_MATERIALS]
    var mat_rgba_a: InlineArray[Float64, MAX_COMPTIME_MATERIALS]
    var mat_shininess: InlineArray[Float64, MAX_COMPTIME_MATERIALS]
    var mat_specular: InlineArray[Float64, MAX_COMPTIME_MATERIALS]
    var mat_reflectance: InlineArray[Float64, MAX_COMPTIME_MATERIALS]
    var mat_tex_id: InlineArray[Int, MAX_COMPTIME_MATERIALS]  # index into tex_names[], -1 if no texture
    var mat_texrepeat_u: InlineArray[Float64, MAX_COMPTIME_MATERIALS]  # texture repeat U (default 1.0)
    var mat_texrepeat_v: InlineArray[Float64, MAX_COMPTIME_MATERIALS]  # texture repeat V (default 1.0)

    # Sites (max 16)
    var site_body_id: InlineArray[Int, MAX_COMPTIME_RENDER_SITES]
    var site_pos_x: InlineArray[Float64, MAX_COMPTIME_RENDER_SITES]
    var site_pos_y: InlineArray[Float64, MAX_COMPTIME_RENDER_SITES]
    var site_pos_z: InlineArray[Float64, MAX_COMPTIME_RENDER_SITES]
    var site_size_0: InlineArray[Float64, MAX_COMPTIME_RENDER_SITES]
    # Spatial tendons, for DRAWING ONLY. The physics reads its tendon records
    # from the runtime parser; this records the site chain so the renderer can
    # draw the segments, without which ball_in_cup's string is simply absent.
    # ⚠ Wrapping geoms are NOT modelled: a <geom> child inside a <spatial> is
    # ignored, so a tendon that wraps would be drawn as the straight chord it
    # is not.
    var nsten: Int
    var sten_nsite: InlineArray[Int, MAX_COMPTIME_SPATIAL_TENDONS]
    var sten_sites: InlineArray[Int, MAX_COMPTIME_SPATIAL_TENDON_SITES]
    var sten_width: InlineArray[Float64, MAX_COMPTIME_SPATIAL_TENDONS]
    var sten_rgba_r: InlineArray[Float64, MAX_COMPTIME_SPATIAL_TENDONS]
    var sten_rgba_g: InlineArray[Float64, MAX_COMPTIME_SPATIAL_TENDONS]
    var sten_rgba_b: InlineArray[Float64, MAX_COMPTIME_SPATIAL_TENDONS]

    # Visual settings from <visual> section
    var vis_znear: Float64  # <map znear="..."/>  (camera near plane)
    var vis_fogstart: Float64  # <map fogstart="..."/>
    var vis_fogend: Float64  # <map fogend="..."/>
    var vis_shadowsize: Int  # <quality shadowsize="..."/>
    var vis_headlight_ambient_r: Float64  # <headlight ambient="r g b"/>
    var vis_headlight_ambient_g: Float64
    var vis_headlight_ambient_b: Float64
    var vis_has_headlight: Bool  # True if <headlight> was found

    def __init__(out self):
        """Initialize with safe defaults."""
        self.ngeom = 0
        self.nlight = 0
        self.ncam = 0
        self.ntex = 0
        self.nmat = 0
        self.nsite = 0

        self.geom_body_id = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=0)
        self.geom_type = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=1)  # SPHERE default
        self.geom_pos_x = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_pos_y = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_pos_z = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_quat_x = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_quat_y = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_quat_z = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_quat_w = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=1.0)
        self.geom_radius = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_half_length = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_half_x = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_half_y = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_half_z = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_rgba_r = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.7)
        self.geom_rgba_g = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.7)
        self.geom_rgba_b = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.7)
        self.geom_rgba_a = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=1.0)
        self.geom_material_id = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=-1)
        self.geom_mesh_id = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=-1)
        self.geom_group = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=0)
        self.nmesh = 0
        self.mesh_names = InlineArray[String, MAX_COMPTIME_RENDER_MESHES](fill=String(""))
        self.mesh_files = InlineArray[String, MAX_COMPTIME_RENDER_MESHES](fill=String(""))

        self.light_dir_x = InlineArray[Float64, 8](fill=0.0)
        self.light_dir_y = InlineArray[Float64, 8](fill=0.0)
        self.light_dir_z = InlineArray[Float64, 8](fill=-1.0)
        self.light_diffuse_r = InlineArray[Float64, 8](fill=0.7)
        self.light_diffuse_g = InlineArray[Float64, 8](fill=0.7)
        self.light_diffuse_b = InlineArray[Float64, 8](fill=0.7)
        self.light_specular_r = InlineArray[Float64, 8](fill=0.3)
        self.light_specular_g = InlineArray[Float64, 8](fill=0.3)
        self.light_specular_b = InlineArray[Float64, 8](fill=0.3)
        self.light_ambient_r = InlineArray[Float64, 8](fill=0.0)
        self.light_ambient_g = InlineArray[Float64, 8](fill=0.0)
        self.light_ambient_b = InlineArray[Float64, 8](fill=0.0)
        self.light_directional = InlineArray[Bool, 8](fill=False)
        self.light_castshadow = InlineArray[Bool, 8](fill=True)
        self.light_exponent = InlineArray[Float64, 8](fill=10.0)

        self.cam_pos_x = InlineArray[Float64, 8](fill=0.0)
        self.cam_pos_y = InlineArray[Float64, 8](fill=0.0)
        self.cam_pos_z = InlineArray[Float64, 8](fill=0.0)
        self.cam_quat_x = InlineArray[Float64, 8](fill=0.0)
        self.cam_quat_y = InlineArray[Float64, 8](fill=0.0)
        self.cam_quat_z = InlineArray[Float64, 8](fill=0.0)
        self.cam_quat_w = InlineArray[Float64, 8](fill=1.0)
        self.cam_fovy = InlineArray[Float64, 8](fill=45.0)
        self.cam_mode = InlineArray[Int, 8](fill=0)
        self.cam_body_id = InlineArray[Int, 8](fill=0)
        self.cam_target_body = InlineArray[Int, 8](fill=-1)

        self.tex_type = InlineArray[Int, MAX_COMPTIME_TEXTURES](fill=0)
        self.tex_builtin = InlineArray[Int, MAX_COMPTIME_TEXTURES](fill=0)
        self.tex_rgb1_r = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.8)
        self.tex_rgb1_g = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.8)
        self.tex_rgb1_b = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.8)
        self.tex_rgb2_r = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.5)
        self.tex_rgb2_g = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.5)
        self.tex_rgb2_b = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.5)
        self.tex_names = InlineArray[String, MAX_COMPTIME_TEXTURES](fill=String(""))
        self.tex_files = InlineArray[String, MAX_COMPTIME_TEXTURES](fill=String(""))
        self.tex_mark = InlineArray[Int, MAX_COMPTIME_TEXTURES](fill=0)
        self.tex_markrgb_r = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=1.0)
        self.tex_markrgb_g = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=1.0)
        self.tex_markrgb_b = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=1.0)
        self.tex_random = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.01)

        self.mat_rgba_r = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_rgba_g = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_rgba_b = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_rgba_a = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_shininess = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=0.5)
        self.mat_specular = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=0.5)
        self.mat_reflectance = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=0.0)
        self.mat_tex_id = InlineArray[Int, MAX_COMPTIME_MATERIALS](fill=-1)
        self.mat_texrepeat_u = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_texrepeat_v = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)

        self.site_body_id = InlineArray[Int, MAX_COMPTIME_RENDER_SITES](fill=0)
        self.site_pos_x = InlineArray[Float64, MAX_COMPTIME_RENDER_SITES](fill=0.0)
        self.site_pos_y = InlineArray[Float64, MAX_COMPTIME_RENDER_SITES](fill=0.0)
        self.site_pos_z = InlineArray[Float64, MAX_COMPTIME_RENDER_SITES](fill=0.0)
        self.site_size_0 = InlineArray[Float64, MAX_COMPTIME_RENDER_SITES](fill=0.005)
        self.nsten = 0
        self.sten_nsite = InlineArray[Int, MAX_COMPTIME_SPATIAL_TENDONS](fill=0)
        self.sten_sites = InlineArray[
            Int, MAX_COMPTIME_SPATIAL_TENDON_SITES
        ](fill=-1)
        self.sten_width = InlineArray[
            Float64, MAX_COMPTIME_SPATIAL_TENDONS
        ](fill=0.003)
        self.sten_rgba_r = InlineArray[
            Float64, MAX_COMPTIME_SPATIAL_TENDONS
        ](fill=0.5)
        self.sten_rgba_g = InlineArray[
            Float64, MAX_COMPTIME_SPATIAL_TENDONS
        ](fill=0.5)
        self.sten_rgba_b = InlineArray[
            Float64, MAX_COMPTIME_SPATIAL_TENDONS
        ](fill=0.5)

        # Visual defaults (MuJoCo defaults)
        self.vis_znear = 0.01  # MuJoCo default
        self.vis_fogstart = 3.0
        self.vis_fogend = 10.0
        self.vis_shadowsize = 4096  # MuJoCo default
        self.vis_headlight_ambient_r = 0.1
        self.vis_headlight_ambient_g = 0.1
        self.vis_headlight_ambient_b = 0.1
        self.vis_has_headlight = False

    def __init__(out self, *, copy: Self):
        """Copy constructor — element-by-element InlineArray copy."""
        self.ngeom = copy.ngeom
        self.nlight = copy.nlight
        self.ncam = copy.ncam
        self.ntex = copy.ntex
        self.nmat = copy.nmat
        self.nsite = copy.nsite

        self.geom_body_id = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=0)
        self.geom_type = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=1)
        self.geom_pos_x = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_pos_y = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_pos_z = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_quat_x = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_quat_y = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_quat_z = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_quat_w = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=1.0)
        self.geom_radius = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_half_length = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_half_x = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_half_y = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_half_z = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.0)
        self.geom_rgba_r = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.7)
        self.geom_rgba_g = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.7)
        self.geom_rgba_b = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=0.7)
        self.geom_rgba_a = InlineArray[Float64, MAX_COMPTIME_RENDER_GEOMS](fill=1.0)
        self.geom_material_id = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=-1)
        self.geom_mesh_id = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=-1)
        self.geom_group = InlineArray[Int, MAX_COMPTIME_RENDER_GEOMS](fill=0)
        # ⚠ WAS `range(64)` AGAINST A 160-SLOT ARRAY. The write guard further
        # down was widened from 64 when dog overflowed it; this copy was not,
        # so every geom past 63 silently kept its fill value here — the same
        # defect, in the constructor rather than the parser. Bound both to the
        # cap so widening it again cannot leave one behind.
        for i in range(MAX_COMPTIME_RENDER_GEOMS):
            self.geom_body_id[i] = copy.geom_body_id[i]
            self.geom_type[i] = copy.geom_type[i]
            self.geom_pos_x[i] = copy.geom_pos_x[i]
            self.geom_pos_y[i] = copy.geom_pos_y[i]
            self.geom_pos_z[i] = copy.geom_pos_z[i]
            self.geom_quat_x[i] = copy.geom_quat_x[i]
            self.geom_quat_y[i] = copy.geom_quat_y[i]
            self.geom_quat_z[i] = copy.geom_quat_z[i]
            self.geom_quat_w[i] = copy.geom_quat_w[i]
            self.geom_radius[i] = copy.geom_radius[i]
            self.geom_half_length[i] = copy.geom_half_length[i]
            self.geom_half_x[i] = copy.geom_half_x[i]
            self.geom_half_y[i] = copy.geom_half_y[i]
            self.geom_half_z[i] = copy.geom_half_z[i]
            self.geom_rgba_r[i] = copy.geom_rgba_r[i]
            self.geom_rgba_g[i] = copy.geom_rgba_g[i]
            self.geom_rgba_b[i] = copy.geom_rgba_b[i]
            self.geom_rgba_a[i] = copy.geom_rgba_a[i]
            self.geom_material_id[i] = copy.geom_material_id[i]
            self.geom_mesh_id[i] = copy.geom_mesh_id[i]
            self.geom_group[i] = copy.geom_group[i]
        self.nmesh = copy.nmesh
        self.mesh_names = InlineArray[String, MAX_COMPTIME_RENDER_MESHES](fill=String(""))
        self.mesh_files = InlineArray[String, MAX_COMPTIME_RENDER_MESHES](fill=String(""))
        # ⚠ BOUNDED BY THE CAP, NOT A LITERAL. This loop said 16 while the
        # arrays grew to MAX_COMPTIME_RENDER_MESHES, which would have dropped
        # the extra meshes on every COPY — a fresh instance of the same silent
        # truncation the cap itself was raised to fix.
        for i in range(MAX_COMPTIME_RENDER_MESHES):
            self.mesh_names[i] = copy.mesh_names[i]
            self.mesh_files[i] = copy.mesh_files[i]

        self.light_dir_x = InlineArray[Float64, 8](fill=0.0)
        self.light_dir_y = InlineArray[Float64, 8](fill=0.0)
        self.light_dir_z = InlineArray[Float64, 8](fill=-1.0)
        self.light_diffuse_r = InlineArray[Float64, 8](fill=0.7)
        self.light_diffuse_g = InlineArray[Float64, 8](fill=0.7)
        self.light_diffuse_b = InlineArray[Float64, 8](fill=0.7)
        self.light_specular_r = InlineArray[Float64, 8](fill=0.3)
        self.light_specular_g = InlineArray[Float64, 8](fill=0.3)
        self.light_specular_b = InlineArray[Float64, 8](fill=0.3)
        self.light_ambient_r = InlineArray[Float64, 8](fill=0.0)
        self.light_ambient_g = InlineArray[Float64, 8](fill=0.0)
        self.light_ambient_b = InlineArray[Float64, 8](fill=0.0)
        self.light_directional = InlineArray[Bool, 8](fill=False)
        self.light_castshadow = InlineArray[Bool, 8](fill=True)
        self.light_exponent = InlineArray[Float64, 8](fill=10.0)
        for i in range(8):
            self.light_dir_x[i] = copy.light_dir_x[i]
            self.light_dir_y[i] = copy.light_dir_y[i]
            self.light_dir_z[i] = copy.light_dir_z[i]
            self.light_diffuse_r[i] = copy.light_diffuse_r[i]
            self.light_diffuse_g[i] = copy.light_diffuse_g[i]
            self.light_diffuse_b[i] = copy.light_diffuse_b[i]
            self.light_specular_r[i] = copy.light_specular_r[i]
            self.light_specular_g[i] = copy.light_specular_g[i]
            self.light_specular_b[i] = copy.light_specular_b[i]
            self.light_ambient_r[i] = copy.light_ambient_r[i]
            self.light_ambient_g[i] = copy.light_ambient_g[i]
            self.light_ambient_b[i] = copy.light_ambient_b[i]
            self.light_directional[i] = copy.light_directional[i]
            self.light_castshadow[i] = copy.light_castshadow[i]
            self.light_exponent[i] = copy.light_exponent[i]

        self.cam_pos_x = InlineArray[Float64, 8](fill=0.0)
        self.cam_pos_y = InlineArray[Float64, 8](fill=0.0)
        self.cam_pos_z = InlineArray[Float64, 8](fill=0.0)
        self.cam_quat_x = InlineArray[Float64, 8](fill=0.0)
        self.cam_quat_y = InlineArray[Float64, 8](fill=0.0)
        self.cam_quat_z = InlineArray[Float64, 8](fill=0.0)
        self.cam_quat_w = InlineArray[Float64, 8](fill=1.0)
        self.cam_fovy = InlineArray[Float64, 8](fill=45.0)
        self.cam_mode = InlineArray[Int, 8](fill=0)
        self.cam_body_id = InlineArray[Int, 8](fill=0)
        self.cam_target_body = InlineArray[Int, 8](fill=-1)
        for i in range(8):
            self.cam_pos_x[i] = copy.cam_pos_x[i]
            self.cam_pos_y[i] = copy.cam_pos_y[i]
            self.cam_pos_z[i] = copy.cam_pos_z[i]
            self.cam_quat_x[i] = copy.cam_quat_x[i]
            self.cam_quat_y[i] = copy.cam_quat_y[i]
            self.cam_quat_z[i] = copy.cam_quat_z[i]
            self.cam_quat_w[i] = copy.cam_quat_w[i]
            self.cam_fovy[i] = copy.cam_fovy[i]
            self.cam_mode[i] = copy.cam_mode[i]
            self.cam_body_id[i] = copy.cam_body_id[i]
            self.cam_target_body[i] = copy.cam_target_body[i]

        self.tex_type = InlineArray[Int, MAX_COMPTIME_TEXTURES](fill=0)
        self.tex_builtin = InlineArray[Int, MAX_COMPTIME_TEXTURES](fill=0)
        self.tex_rgb1_r = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.8)
        self.tex_rgb1_g = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.8)
        self.tex_rgb1_b = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.8)
        self.tex_rgb2_r = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.5)
        self.tex_rgb2_g = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.5)
        self.tex_rgb2_b = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.5)
        self.tex_names = InlineArray[String, MAX_COMPTIME_TEXTURES](fill=String(""))
        self.tex_files = InlineArray[String, MAX_COMPTIME_TEXTURES](fill=String(""))
        self.tex_mark = InlineArray[Int, MAX_COMPTIME_TEXTURES](fill=0)
        self.tex_markrgb_r = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=1.0)
        self.tex_markrgb_g = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=1.0)
        self.tex_markrgb_b = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=1.0)
        self.tex_random = InlineArray[Float64, MAX_COMPTIME_TEXTURES](fill=0.01)
        for i in range(MAX_COMPTIME_TEXTURES):
            self.tex_type[i] = copy.tex_type[i]
            self.tex_builtin[i] = copy.tex_builtin[i]
            self.tex_rgb1_r[i] = copy.tex_rgb1_r[i]
            self.tex_rgb1_g[i] = copy.tex_rgb1_g[i]
            self.tex_rgb1_b[i] = copy.tex_rgb1_b[i]
            self.tex_rgb2_r[i] = copy.tex_rgb2_r[i]
            self.tex_rgb2_g[i] = copy.tex_rgb2_g[i]
            self.tex_rgb2_b[i] = copy.tex_rgb2_b[i]
            self.tex_names[i] = copy.tex_names[i]
            self.tex_files[i] = copy.tex_files[i]
            self.tex_mark[i] = copy.tex_mark[i]
            self.tex_markrgb_r[i] = copy.tex_markrgb_r[i]
            self.tex_markrgb_g[i] = copy.tex_markrgb_g[i]
            self.tex_markrgb_b[i] = copy.tex_markrgb_b[i]
            self.tex_random[i] = copy.tex_random[i]

        self.mat_rgba_r = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_rgba_g = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_rgba_b = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_rgba_a = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_shininess = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=0.5)
        self.mat_specular = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=0.5)
        self.mat_reflectance = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=0.0)
        self.mat_tex_id = InlineArray[Int, MAX_COMPTIME_MATERIALS](fill=-1)
        self.mat_texrepeat_u = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        self.mat_texrepeat_v = InlineArray[Float64, MAX_COMPTIME_MATERIALS](fill=1.0)
        for i in range(MAX_COMPTIME_MATERIALS):
            self.mat_rgba_r[i] = copy.mat_rgba_r[i]
            self.mat_rgba_g[i] = copy.mat_rgba_g[i]
            self.mat_rgba_b[i] = copy.mat_rgba_b[i]
            self.mat_rgba_a[i] = copy.mat_rgba_a[i]
            self.mat_shininess[i] = copy.mat_shininess[i]
            self.mat_specular[i] = copy.mat_specular[i]
            self.mat_reflectance[i] = copy.mat_reflectance[i]
            self.mat_tex_id[i] = copy.mat_tex_id[i]
            self.mat_texrepeat_u[i] = copy.mat_texrepeat_u[i]
            self.mat_texrepeat_v[i] = copy.mat_texrepeat_v[i]

        self.site_body_id = InlineArray[Int, MAX_COMPTIME_RENDER_SITES](fill=0)
        self.site_pos_x = InlineArray[Float64, MAX_COMPTIME_RENDER_SITES](fill=0.0)
        self.site_pos_y = InlineArray[Float64, MAX_COMPTIME_RENDER_SITES](fill=0.0)
        self.site_pos_z = InlineArray[Float64, MAX_COMPTIME_RENDER_SITES](fill=0.0)
        self.site_size_0 = InlineArray[Float64, MAX_COMPTIME_RENDER_SITES](fill=0.005)
        self.nsten = copy.nsten
        self.sten_nsite = InlineArray[Int, MAX_COMPTIME_SPATIAL_TENDONS](fill=0)
        self.sten_sites = InlineArray[
            Int, MAX_COMPTIME_SPATIAL_TENDON_SITES
        ](fill=-1)
        self.sten_width = InlineArray[
            Float64, MAX_COMPTIME_SPATIAL_TENDONS
        ](fill=0.003)
        self.sten_rgba_r = InlineArray[
            Float64, MAX_COMPTIME_SPATIAL_TENDONS
        ](fill=0.5)
        self.sten_rgba_g = InlineArray[
            Float64, MAX_COMPTIME_SPATIAL_TENDONS
        ](fill=0.5)
        self.sten_rgba_b = InlineArray[
            Float64, MAX_COMPTIME_SPATIAL_TENDONS
        ](fill=0.5)
        # ⚠ WAS `range(16)` AGAINST A 48-SLOT ARRAY — same stale bound as the
        # geoms above (quadruped has 30 sites, humanoid 25).
        for i in range(MAX_COMPTIME_RENDER_SITES):
            self.site_body_id[i] = copy.site_body_id[i]
            self.site_pos_x[i] = copy.site_pos_x[i]
            self.site_pos_y[i] = copy.site_pos_y[i]
            self.site_pos_z[i] = copy.site_pos_z[i]
            self.site_size_0[i] = copy.site_size_0[i]
        for i in range(MAX_COMPTIME_SPATIAL_TENDONS):
            self.sten_nsite[i] = copy.sten_nsite[i]
            self.sten_width[i] = copy.sten_width[i]
            self.sten_rgba_r[i] = copy.sten_rgba_r[i]
            self.sten_rgba_g[i] = copy.sten_rgba_g[i]
            self.sten_rgba_b[i] = copy.sten_rgba_b[i]
        for i in range(MAX_COMPTIME_SPATIAL_TENDON_SITES):
            self.sten_sites[i] = copy.sten_sites[i]

        # Visual settings
        self.vis_znear = copy.vis_znear
        self.vis_fogstart = copy.vis_fogstart
        self.vis_fogend = copy.vis_fogend
        self.vis_shadowsize = copy.vis_shadowsize
        self.vis_headlight_ambient_r = copy.vis_headlight_ambient_r
        self.vis_headlight_ambient_g = copy.vis_headlight_ambient_g
        self.vis_headlight_ambient_b = copy.vis_headlight_ambient_b
        self.vis_has_headlight = copy.vis_has_headlight

    def __init__(out self, *, deinit move: Self):
        self.ngeom = move.ngeom
        self.nlight = move.nlight
        self.ncam = move.ncam
        self.ntex = move.ntex
        self.nmat = move.nmat
        self.nsite = move.nsite
        self.geom_body_id = move.geom_body_id^
        self.geom_type = move.geom_type^
        self.geom_pos_x = move.geom_pos_x^
        self.geom_pos_y = move.geom_pos_y^
        self.geom_pos_z = move.geom_pos_z^
        self.geom_quat_x = move.geom_quat_x^
        self.geom_quat_y = move.geom_quat_y^
        self.geom_quat_z = move.geom_quat_z^
        self.geom_quat_w = move.geom_quat_w^
        self.geom_radius = move.geom_radius^
        self.geom_half_length = move.geom_half_length^
        self.geom_half_x = move.geom_half_x^
        self.geom_half_y = move.geom_half_y^
        self.geom_half_z = move.geom_half_z^
        self.geom_rgba_r = move.geom_rgba_r^
        self.geom_rgba_g = move.geom_rgba_g^
        self.geom_rgba_b = move.geom_rgba_b^
        self.geom_rgba_a = move.geom_rgba_a^
        self.geom_material_id = move.geom_material_id^
        self.geom_mesh_id = move.geom_mesh_id^
        self.geom_group = move.geom_group^
        self.nmesh = move.nmesh
        self.mesh_names = move.mesh_names^
        self.mesh_files = move.mesh_files^
        self.light_dir_x = move.light_dir_x^
        self.light_dir_y = move.light_dir_y^
        self.light_dir_z = move.light_dir_z^
        self.light_diffuse_r = move.light_diffuse_r^
        self.light_diffuse_g = move.light_diffuse_g^
        self.light_diffuse_b = move.light_diffuse_b^
        self.light_specular_r = move.light_specular_r^
        self.light_specular_g = move.light_specular_g^
        self.light_specular_b = move.light_specular_b^
        self.light_ambient_r = move.light_ambient_r^
        self.light_ambient_g = move.light_ambient_g^
        self.light_ambient_b = move.light_ambient_b^
        self.light_directional = move.light_directional^
        self.light_castshadow = move.light_castshadow^
        self.light_exponent = move.light_exponent^
        self.cam_pos_x = move.cam_pos_x^
        self.cam_pos_y = move.cam_pos_y^
        self.cam_pos_z = move.cam_pos_z^
        self.cam_quat_x = move.cam_quat_x^
        self.cam_quat_y = move.cam_quat_y^
        self.cam_quat_z = move.cam_quat_z^
        self.cam_quat_w = move.cam_quat_w^
        self.cam_fovy = move.cam_fovy^
        self.cam_mode = move.cam_mode^
        self.cam_body_id = move.cam_body_id^
        self.cam_target_body = move.cam_target_body^
        self.tex_type = move.tex_type^
        self.tex_builtin = move.tex_builtin^
        self.tex_rgb1_r = move.tex_rgb1_r^
        self.tex_rgb1_g = move.tex_rgb1_g^
        self.tex_rgb1_b = move.tex_rgb1_b^
        self.tex_rgb2_r = move.tex_rgb2_r^
        self.tex_rgb2_g = move.tex_rgb2_g^
        self.tex_rgb2_b = move.tex_rgb2_b^
        self.tex_names = move.tex_names^
        self.tex_files = move.tex_files^
        self.tex_mark = move.tex_mark^
        self.tex_markrgb_r = move.tex_markrgb_r^
        self.tex_markrgb_g = move.tex_markrgb_g^
        self.tex_markrgb_b = move.tex_markrgb_b^
        self.tex_random = move.tex_random^
        self.mat_rgba_r = move.mat_rgba_r^
        self.mat_rgba_g = move.mat_rgba_g^
        self.mat_rgba_b = move.mat_rgba_b^
        self.mat_rgba_a = move.mat_rgba_a^
        self.mat_shininess = move.mat_shininess^
        self.mat_specular = move.mat_specular^
        self.mat_reflectance = move.mat_reflectance^
        self.mat_tex_id = move.mat_tex_id^
        self.mat_texrepeat_u = move.mat_texrepeat_u^
        self.mat_texrepeat_v = move.mat_texrepeat_v^
        self.site_body_id = move.site_body_id^
        self.site_pos_x = move.site_pos_x^
        self.site_pos_y = move.site_pos_y^
        self.site_pos_z = move.site_pos_z^
        self.site_size_0 = move.site_size_0^
        self.nsten = move.nsten
        self.sten_nsite = move.sten_nsite^
        self.sten_sites = move.sten_sites^
        self.sten_width = move.sten_width^
        self.sten_rgba_r = move.sten_rgba_r^
        self.sten_rgba_g = move.sten_rgba_g^
        self.sten_rgba_b = move.sten_rgba_b^

        # Visual settings
        self.vis_znear = move.vis_znear
        self.vis_fogstart = move.vis_fogstart
        self.vis_fogend = move.vis_fogend
        self.vis_shadowsize = move.vis_shadowsize
        self.vis_headlight_ambient_r = move.vis_headlight_ambient_r
        self.vis_headlight_ambient_g = move.vis_headlight_ambient_g
        self.vis_headlight_ambient_b = move.vis_headlight_ambient_b
        self.vis_has_headlight = move.vis_has_headlight


# =============================================================================
# Render data helper functions (copied from full_parser.mojo for independence)
# =============================================================================


def _rcd_geom_type_from_str(s: String) -> Int:
    """Convert geom type string to integer constant.
    PLANE=0, SPHERE=1, CAPSULE=2, BOX=3, CYLINDER=4, MESH=5, ELLIPSOID=6.

    The comptime twin of `full_parser._geom_type_from_str`; both must know the
    same set, because this table feeds the RENDERER while that one feeds the
    inertia. `ellipsoid` was in neither until fish (bug 26) — see
    `physics3d/constants.GEOM_ELLIPSOID`.
    """
    var t = _trim(s)
    if t == "plane":
        return 0
    elif t == "sphere":
        return 1
    elif t == "capsule":
        return 2
    elif t == "box":
        return 3
    elif t == "cylinder":
        return 4
    elif t == "mesh":
        return 5
    elif t == "ellipsoid":
        return 6
    return 1  # default = sphere (a SILENT substitution — see the twin)


def _rcd_parse_rgba4(s: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """Parse "r g b a" string into four Float64 values."""
    var parts = List[String]()
    _split_spaces(s, parts)
    var r = Float64(1)
    var g = Float64(1)
    var b = Float64(1)
    var a = Float64(1)
    if len(parts) >= 1:
        r = _parse_float(parts[0])
    if len(parts) >= 2:
        g = _parse_float(parts[1])
    if len(parts) >= 3:
        b = _parse_float(parts[2])
    if len(parts) >= 4:
        a = _parse_float(parts[3])
    return (r, g, b, a)


def _rcd_parse_rgb3(s: String) -> Tuple[Float64, Float64, Float64]:
    """Parse "r g b" string into three Float64 values."""
    var parts = List[String]()
    _split_spaces(s, parts)
    var r = Float64(0)
    var g = Float64(0)
    var b = Float64(0)
    if len(parts) >= 1:
        r = _parse_float(parts[0])
    if len(parts) >= 2:
        g = _parse_float(parts[1])
    if len(parts) >= 3:
        b = _parse_float(parts[2])
    return (r, g, b)


def _rcd_xyaxes_to_quat(s: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """Convert xyaxes="x1 x2 x3 y1 y2 y3" to quaternion (qx, qy, qz, qw)."""
    var parts = List[String]()
    _split_spaces(s, parts)
    if len(parts) < 6:
        return (Float64(0), Float64(0), Float64(0), Float64(1))
    var xx = _parse_float(parts[0])
    var xy = _parse_float(parts[1])
    var xz = _parse_float(parts[2])
    var yx = _parse_float(parts[3])
    var yy = _parse_float(parts[4])
    var yz = _parse_float(parts[5])
    var xn = _sqrt_f64(xx * xx + xy * xy + xz * xz)
    if xn > 0.0:
        xx /= xn
        xy /= xn
        xz /= xn
    var zx = xy * yz - xz * yy
    var zy = xz * yx - xx * yz
    var zz = xx * yy - xy * yx
    var zn = _sqrt_f64(zx * zx + zy * zy + zz * zz)
    if zn > 0.0:
        zx /= zn
        zy /= zn
        zz /= zn
    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx
    # ⚠ THE VECTOR PART WAS NEGATED — this returned the CONJUGATE, i.e. the
    # inverse rotation. The frame's axes are the COLUMNS of R, so
    # R[i][j] = (axis_j)_i and therefore R[2][1] = y_z (`yz`), R[1][2] = z_y
    # (`zy`); the standard qx = (R[2][1] - R[1][2]) is `yz - zy`, and every
    # branch here had the operands the other way round. Consistently so, which
    # is why it produced a unit quaternion that looked plausible and simply
    # rotated the wrong way.
    #
    # It stayed latent because the only caller was the camera `targetbody`
    # branch, which no ported model takes. The moment `setup_cameras` started
    # deriving look/up from the orientation for EVERY camera, it surfaced as
    # cameras aimed 180 degrees away from the model — ball_in_cup's cam0 at
    # (0,-1,.8) targeting (0,-1.89,.35), i.e. facing out of the scene.
    var trace = xx + yy + zz
    var qx: Float64
    var qy: Float64
    var qz: Float64
    var qw: Float64
    if trace > 0.0:
        var s2 = _sqrt_f64(trace + 1.0) * 2.0
        qw = 0.25 * s2
        qx = (yz - zy) / s2
        qy = (zx - xz) / s2
        qz = (xy - yx) / s2
    elif xx > yy and xx > zz:
        var s2 = _sqrt_f64(1.0 + xx - yy - zz) * 2.0
        qw = (yz - zy) / s2
        qx = 0.25 * s2
        qy = (xy + yx) / s2
        qz = (xz + zx) / s2
    elif yy > zz:
        var s2 = _sqrt_f64(1.0 + yy - xx - zz) * 2.0
        qw = (zx - xz) / s2
        qx = (xy + yx) / s2
        qy = 0.25 * s2
        qz = (yz + zy) / s2
    else:
        var s2 = _sqrt_f64(1.0 + zz - xx - yy) * 2.0
        qw = (xy - yx) / s2
        qx = (xz + zx) / s2
        qy = (yz + zy) / s2
        qz = 0.25 * s2
    return (qx, qy, qz, qw)


def _rcd_find_material_index_by_name(asset_sec: String, name: String) -> Int:
    """Return 0-based index of <material name="name"> in asset_sec, or -1."""
    var search = 'name="' + name + '"'
    var scan_pos = 0
    var count = 0
    while True:
        var t = asset_sec.find("<material", scan_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(asset_sec[byte = t : tag_end + 1])
        if tag.find(search) != -1:
            return count
        count += 1
        scan_pos = tag_end + 1
    return -1


def _rcd_tex_type_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "skybox":
        return 1  # TEX_SKYBOX
    elif t == "cube":
        return 3  # TEX_CUBE
    return 0  # TEX_2D


def _rcd_tex_builtin_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "gradient":
        return 1  # TEX_BUILTIN_GRADIENT
    elif t == "checker":
        return 2  # TEX_BUILTIN_CHECKER
    elif t == "flat":
        return 3  # TEX_BUILTIN_FLAT
    return 0  # TEX_BUILTIN_NONE


def _rcd_cam_mode_from_str(s: String) -> Int:
    var t = _trim(s)
    if t == "track":
        return 1
    elif t == "trackcom":
        return 2
    elif t == "targetbody":
        return 3
    elif t == "targetbodycom":
        return 4
    return 0


def _rcd_find_site_index_by_name(worldbody: String, name: String) -> Int:
    """Ordinal of the `<site name="...">` matching `name`, or -1.

    ⚠ SCANS ON DEMAND rather than keeping a name table. Two earlier shapes of
    this both tripped the comptime interpreter with "interpreting memcpy can't
    get dst memory": an `InlineArray[String, 16]` field on
    `ComptimeRenderData`, and the same array as a parse-local filled for EVERY
    site. This runs only for the handful of `site="..."` references a
    `<spatial>` tendon actually makes — two, for ball_in_cup — so models
    without tendons pay nothing at all.

    ⚠ BODY-GROUPED, NOT TEXT ORDER. `sten_sites` indexes the site array that
    `full_parser` builds and then groups by body, so this had the same defect
    as its `full_parser` twin. Inert until now only because ball_in_cup is the
    one model with a `<spatial>` tendon and its site order happens to match;
    finger, manipulator and stacker have the permutation and no spatial tendon.
    """
    return _index_by_name_grouped(worldbody, "<site", name)


def _rcd_tex_mark_from_str(s: String) -> Int:
    """MuJoCo `mark` -> 0 none, 1 edge, 2 cross, 3 random (mjtMark order)."""
    var t = _trim(s)
    if t == "edge":
        return 1
    elif t == "cross":
        return 2
    elif t == "random":
        return 3
    return 0


def _render_meshdir(xml_clean: String) -> String:
    """`<compiler meshdir>` (or `assetdir`) with a trailing slash, or "".

    The render twin of `full_parser`'s resolution, and it must stay in step
    with it: `meshdir` WINS over `assetdir`, and an absolute `file=` ignores
    both. Measured on the 3.10.0 runtime — see that function's note for the
    four cases.
    """
    var t = xml_clean.find("<compiler")
    if t == -1:
        return String("")
    var e = xml_clean.find(">", t)
    if e == -1:
        return String("")
    var ctag = String(xml_clean[byte = t : e + 1])
    var d = _trim(_extract_attr(ctag, "meshdir"))
    if d.byte_length() == 0:
        d = _trim(_extract_attr(ctag, "assetdir"))
    if d.byte_length() == 0:
        return String("")
    if not d.endswith("/"):
        d = d + "/"
    return d


def _rcd_min_valid(a: Int, b: Int) -> Int:
    """Return the smaller of a and b, treating -1 as +infinity."""
    if a == -1:
        return b
    if b == -1:
        return a
    if a < b:
        return a
    return b


# =============================================================================
# parse_xml_render_data — lightweight rendering-only XML parser
# =============================================================================


def parse_xml_render_data(xml: String) -> ComptimeRenderData:
    """Parse XML and return rendering data as InlineArrays.

    Designed to be called at struct-level comptime:

        comptime _rcd = parse_xml_render_data(Self.xml)

    Extracts ONLY rendering-relevant data (geoms, lights, cameras, textures,
    materials, sites) in a single pass, avoiding the comptime interpreter crash
    caused by multiple parse_xml_full calls for large models.
    """
    var data = ComptimeRenderData()
    var xml_clean = _strip_xml_comments(xml)

    # ---- Compiler angle units ------------------------------------------------
    var deg_factor = _compiler_deg_factor(xml_clean)

    # ---- Default geom rgba from <default> section ----------------------------
    var def_rgba_r = Float64(-1.0)
    var def_rgba_g = Float64(-1.0)
    var def_rgba_b = Float64(-1.0)
    var def_rgba_a = Float64(-1.0)
    var def_sec = _extract_section(xml_clean, "default")
    if def_sec.byte_length() > 0:
        var gpos = def_sec.find("<geom")
        if gpos != -1:
            var tag_end = def_sec.find(">", gpos)
            if tag_end != -1:
                var gtag = String(def_sec[byte = gpos : tag_end + 1])
                var rgba_s = _extract_attr(gtag, "rgba")
                if rgba_s.byte_length() > 0:
                    var cv = _rcd_parse_rgba4(rgba_s)
                    def_rgba_r = cv[0]
                    def_rgba_g = cv[1]
                    def_rgba_b = cv[2]
                    def_rgba_a = cv[3]

    # ---- Top-level <default><geom> type/size/fromto, resolved ONCE ----------
    #
    # acrobot needs this: `<default><geom type="capsule" mass="1"/></default>`
    # with no named class, and its geoms carry only a name and `fromto`.
    #
    # ⚠ GOES THROUGH `_class_attr`, NOT THROUGH A LOCAL SLICE OF `def_sec`.
    # The obvious version — slice the `<geom>` tag out of the section and
    # `_extract_attr` it, exactly as the `rgba` lookup above does — CRASHES the
    # comptime interpreter here, and the reason is the subtlety spelled out on
    # `_class_attr`: the failure happens only on a lookup that HITS, because
    # only a hit reaches `String(tag[byte=a:b])`. The `rgba` lookup above
    # always MISSES (no in-tree model puts rgba on the root geom) so it never
    # reaches the slice and looks like proof the shape is safe. It is not.
    # acrobot's root geom DOES carry `type`, so that one hit, and the viewer
    # died with "interpreting memcpy can't get dst memory" — after this shipped
    # in 5b62d09f, because no test forces `_rcd` evaluation and only rendering
    # does.
    #
    # `_class_attr` with an empty class means the top-level block and is index
    # arithmetic over `xml_clean` with a single slice, which survives a hit.
    # THREE calls, hoisted out of every loop: doing this per geom class was
    # what blew the interpreter's budget in the first place.
    var def_geom_type = _class_attr(xml_clean, String(""), "geom", "type")
    var def_geom_size = _class_attr(xml_clean, String(""), "geom", "size")
    var def_geom_fromto = _class_attr(xml_clean, String(""), "geom", "fromto")

    # ---- Parse <asset> section: textures and materials -----------------------
    var asset_sec = _extract_section(xml_clean, "asset")

    # Textures
    var tex_pos = 0
    var tex_count = 0
    while tex_count < MAX_COMPTIME_TEXTURES:
        var t = asset_sec.find("<texture", tex_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(asset_sec[byte = t : tag_end + 1])
        data.tex_type[tex_count] = _rcd_tex_type_from_str(
            _extract_attr(tag, "type")
        )
        data.tex_builtin[tex_count] = _rcd_tex_builtin_from_str(
            _extract_attr(tag, "builtin")
        )
        var rgb1_s = _extract_attr(tag, "rgb1")
        if rgb1_s.byte_length() > 0:
            var c = _rcd_parse_rgb3(rgb1_s)
            data.tex_rgb1_r[tex_count] = c[0]
            data.tex_rgb1_g[tex_count] = c[1]
            data.tex_rgb1_b[tex_count] = c[2]
        var rgb2_s = _extract_attr(tag, "rgb2")
        if rgb2_s.byte_length() > 0:
            var c = _rcd_parse_rgb3(rgb2_s)
            data.tex_rgb2_r[tex_count] = c[0]
            data.tex_rgb2_g[tex_count] = c[1]
            data.tex_rgb2_b[tex_count] = c[2]
        var tex_name_s = _extract_attr(tag, "name")
        if tex_name_s.byte_length() > 0:
            data.tex_names[tex_count] = tex_name_s
        var tex_file_s = _extract_attr(tag, "file")
        if tex_file_s.byte_length() > 0:
            data.tex_files[tex_count] = tex_file_s
        data.tex_mark[tex_count] = _rcd_tex_mark_from_str(
            _extract_attr(tag, "mark")
        )
        var markrgb_s = _extract_attr(tag, "markrgb")
        if markrgb_s.byte_length() > 0:
            var mc = _rcd_parse_rgb3(markrgb_s)
            data.tex_markrgb_r[tex_count] = mc[0]
            data.tex_markrgb_g[tex_count] = mc[1]
            data.tex_markrgb_b[tex_count] = mc[2]
        var rand_s = _extract_attr(tag, "random")
        if rand_s.byte_length() > 0:
            data.tex_random[tex_count] = _parse_float(rand_s)
        tex_count += 1
        tex_pos = tag_end + 1
    data.ntex = tex_count

    # Materials
    var mat_pos = 0
    var mat_count = 0
    while mat_count < MAX_COMPTIME_MATERIALS:
        var t = asset_sec.find("<material", mat_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(asset_sec[byte = t : tag_end + 1])
        var rgba_s = _extract_attr(tag, "rgba")
        if rgba_s.byte_length() > 0:
            var c = _rcd_parse_rgba4(rgba_s)
            data.mat_rgba_r[mat_count] = c[0]
            data.mat_rgba_g[mat_count] = c[1]
            data.mat_rgba_b[mat_count] = c[2]
            data.mat_rgba_a[mat_count] = c[3]
        var shin_s = _extract_attr(tag, "shininess")
        if shin_s.byte_length() > 0:
            data.mat_shininess[mat_count] = _parse_float(shin_s)
        var spec_s = _extract_attr(tag, "specular")
        if spec_s.byte_length() > 0:
            data.mat_specular[mat_count] = _parse_float(spec_s)
        var refl_s = _extract_attr(tag, "reflectance")
        if refl_s.byte_length() > 0:
            data.mat_reflectance[mat_count] = _parse_float(refl_s)
        # Resolve material → texture reference
        var mat_tex_name = _extract_attr(tag, "texture")
        if mat_tex_name.byte_length() > 0:
            for ti in range(data.ntex):
                if data.tex_names[ti] == mat_tex_name:
                    data.mat_tex_id[mat_count] = ti
                    break
        # Parse texrepeat (default 1 1)
        var tr_s = _extract_attr(tag, "texrepeat")
        if tr_s.byte_length() > 0:
            var tv = _parse_vec3(tr_s)  # reuse vec3 parser, only need first 2
            data.mat_texrepeat_u[mat_count] = tv[0]
            data.mat_texrepeat_v[mat_count] = tv[1]
        mat_count += 1
        mat_pos = tag_end + 1
    data.nmat = mat_count

    # Meshes (name → file path)
    var mesh_pos = 0
    while True:
        var t = asset_sec.find("<mesh", mesh_pos)
        if t == -1:
            break
        var tag_end = asset_sec.find(">", t)
        if tag_end == -1:
            break
        var tag = String(asset_sec[byte=t : tag_end + 1])
        var mesh_name = _extract_attr(tag, "name")
        var mesh_file = _extract_attr(tag, "file")
        # ⚠⚠ `meshdir` APPLIES HERE TOO, AND FIXING ONLY `full_parser` LEFT THE
        # ROBOT INVISIBLE. `renderer.draw_mesh(file_path=...)` opens this string
        # directly, so a bare stem ("Base.stl") silently draws nothing. On the
        # SO-ARM viewer that showed as a floor with NO ARM ON IT — the physics
        # was correct (the runtime parser had already been fixed), only the
        # picture was empty. Textbook `feedback_physics3d_two_parser_paths`:
        # the same attribute has to be honoured on BOTH paths or the halves
        # disagree about where the model's geometry lives.
        var _mdir = _render_meshdir(xml_clean)
        if _mdir.byte_length() > 0 and not mesh_file.startswith("/"):
            mesh_file = _mdir + mesh_file
        if mesh_name.byte_length() > 0 and data.nmesh < MAX_COMPTIME_RENDER_MESHES:
            data.mesh_names[data.nmesh] = mesh_name
            data.mesh_files[data.nmesh] = mesh_file
            data.nmesh += 1
        mesh_pos = tag_end + 1

    # ---- DFS scan <worldbody>: geoms, lights, cameras, sites -----------------
    var worldbody = _extract_section(xml_clean, "worldbody")

    var body_id_stack = InlineArray[Int, 65](fill=0)
    # MJCF `childclass` applies to every DESCENDANT of the body that declares
    # it, so it has to ride the body stack rather than be read per element.
    # Without this a geom that inherits its type from a class is unresolvable.
    var childclass_stack = InlineArray[String, 65](fill=String(""))
    # ⚠ RESOLVE EACH CLASS ONCE. `_class_attr_inherited` re-scans the whole
    # XML per lookup, and doing that per geom per attribute blows the comptime
    # interpreter's budget on this function — which is already the one whose
    # docstring says it exists to avoid a comptime crash on large models. A
    # model has a handful of distinct geom classes and dozens of geoms, so
    # caching by class name turns O(geoms x attrs) scans into O(classes x
    # attrs). quadruped: 64 walks becomes about 20.
    var cls_names = InlineArray[String, 24](fill=String(""))
    var cls_group = InlineArray[String, 24](fill=String(""))
    var cls_type = InlineArray[String, 24](fill=String(""))
    var cls_size = InlineArray[String, 24](fill=String(""))
    var cls_fromto = InlineArray[String, 24](fill=String(""))
    var cls_rgba = InlineArray[String, 24](fill=String(""))
    var cls_material = InlineArray[String, 24](fill=String(""))
    # Effective material name per geom, resolved WITH the class chain. The
    # post-pass below used to re-scan the worldbody for `<geom` tags and read
    # `material=` straight off each one, which cannot see a `<default>` — and
    # dm_control puts it there almost everywhere (`<default class="body">
    # <geom material="self">`). Every such geom came back with material id -1
    # and fell through to the default rgba, which is why the suite rendered
    # white/grey. Resolving it here, where `ci` exists, is the whole fix; the
    # post-pass then has nothing left to look up.
    var geom_mat_name = InlineArray[String, MAX_COMPTIME_RENDER_GEOMS](fill=String(""))
    var geom_has_rgba = InlineArray[Bool, MAX_COMPTIME_RENDER_GEOMS](fill=False)
    var n_cls = 0
    var depth = 0
    var body_count = 0
    var geom_count = 0
    var light_count = 0
    var cam_count = 0
    var site_count = 0
    var scan_pos = 0
    var wlen = worldbody.byte_length()

    while scan_pos < wlen:
        var next_body_open = worldbody.find("<body", scan_pos)
        var next_body_close = worldbody.find("</body>", scan_pos)
        var next_geom = worldbody.find("<geom", scan_pos)
        var next_light = worldbody.find("<light", scan_pos)
        var next_cam = worldbody.find("<camera", scan_pos)
        var next_site = worldbody.find("<site", scan_pos)
        if (
            next_body_open == -1
            and next_body_close == -1
            and next_geom == -1
            and next_light == -1
            and next_cam == -1
            and next_site == -1
        ):
            break
        var earliest = _rcd_min_valid(
            _rcd_min_valid(
                _rcd_min_valid(next_body_open, next_body_close),
                _rcd_min_valid(next_geom, next_light),
            ),
            _rcd_min_valid(next_cam, next_site),
        )
        if earliest == next_body_open:
            var btag = _extract_opening_tag(worldbody, next_body_open)
            var inherited_cc = childclass_stack[depth]
            depth += 1
            body_count += 1
            body_id_stack[depth] = body_count
            # `childclass` on this body replaces the inherited one for its
            # whole subtree; otherwise the parent's carries down.
            var own_cc = _trim(_extract_attr(btag, "childclass"))
            childclass_stack[depth] = (
                own_cc if own_cc.byte_length() > 0 else inherited_cc
            )
            var tag_end = worldbody.find(">", next_body_open)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen
        elif earliest == next_body_close:
            if depth > 0:
                depth -= 1
            scan_pos = next_body_close + 7
        elif earliest == next_geom:
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_geom)
            # Effective class: the geom's own `class=` wins, else the nearest
            # enclosing body's `childclass`.
            var own_cls = _trim(_extract_attr(tag, "class"))
            var geom_cls = (
                own_cls if own_cls.byte_length() > 0
                else childclass_stack[depth]
            )
            # Cache slot for this class ("" is a real key — it means "root
            # defaults only", which most geoms use).
            var ci = -1
            for k in range(n_cls):
                if cls_names[k] == geom_cls:
                    ci = k
                    break
            if ci < 0 and n_cls < 24:
                cls_names[n_cls] = geom_cls
                cls_type[n_cls] = _class_attr_inherited(
                    xml_clean, geom_cls, "geom", "type"
                )
                cls_size[n_cls] = _class_attr_inherited(
                    xml_clean, geom_cls, "geom", "size"
                )
                cls_fromto[n_cls] = _class_attr_inherited(
                    xml_clean, geom_cls, "geom", "fromto"
                )
                cls_rgba[n_cls] = _class_attr_inherited(
                    xml_clean, geom_cls, "geom", "rgba"
                )
                cls_material[n_cls] = _class_attr_inherited(
                    xml_clean, geom_cls, "geom", "material"
                )
                # ⚠ INHERITED, NOT READ OFF THE GEOM. dog declares
                # `group="3"` once on `<default class="collision_primitive">`
                # and never repeats it on a single geom, so a per-element read
                # would see nothing and default every capsule to the visible
                # group 0 — exactly the bug this field exists to fix.
                cls_group[n_cls] = _class_attr_inherited(
                    xml_clean, geom_cls, "geom", "group"
                )
                ci = n_cls
                n_cls += 1
            if geom_count < MAX_COMPTIME_RENDER_GEOMS:
                data.geom_body_id[geom_count] = current_body
                var type_s = _extract_attr(tag, "type")
                if type_s.byte_length() == 0:
                    type_s = cls_type[ci] if ci >= 0 else String("")
                if type_s.byte_length() == 0:
                    type_s = def_geom_type
                data.geom_type[geom_count] = _rcd_geom_type_from_str(type_s)
                # Mesh reference: mesh="name" → lookup in mesh assets
                if data.geom_type[geom_count] == 5:  # GEOM_MESH
                    var mesh_ref = _extract_attr(tag, "mesh")
                    if mesh_ref.byte_length() > 0:
                        for mi in range(data.nmesh):
                            if data.mesh_names[mi] == mesh_ref:
                                data.geom_mesh_id[geom_count] = mi
                                break
                var group_s = _extract_attr(tag, "group")
                if group_s.byte_length() == 0:
                    group_s = cls_group[ci] if ci >= 0 else String("")
                if group_s.byte_length() > 0:
                    data.geom_group[geom_count] = Int(_parse_float(group_s))
                var fromto_s = _extract_attr(tag, "fromto")
                if fromto_s.byte_length() == 0:
                    fromto_s = cls_fromto[ci] if ci >= 0 else String("")
                if fromto_s.byte_length() == 0:
                    fromto_s = def_geom_fromto
                if fromto_s.byte_length() > 0:
                    var ft = _fromto_to_pos_quat(fromto_s)
                    data.geom_pos_x[geom_count] = ft[0]
                    data.geom_pos_y[geom_count] = ft[1]
                    data.geom_pos_z[geom_count] = ft[2]
                    data.geom_quat_x[geom_count] = ft[3]
                    data.geom_quat_y[geom_count] = ft[4]
                    data.geom_quat_z[geom_count] = ft[5]
                    data.geom_quat_w[geom_count] = ft[6]
                    data.geom_half_length[geom_count] = ft[7]
                else:
                    var pos_s = _extract_attr(tag, "pos")
                    if pos_s.byte_length() > 0:
                        var pv = _parse_vec3(pos_s)
                        data.geom_pos_x[geom_count] = pv[0]
                        data.geom_pos_y[geom_count] = pv[1]
                        data.geom_pos_z[geom_count] = pv[2]
                    # ⚠ ALL FIVE MJCF ORIENTATION FORMS, not just two. Until
                    # 2026-08-03 this chain stopped after `quat` and
                    # `axisangle`, so `zaxis` and `euler` were silently DROPPED
                    # and the geom kept its identity rotation. Both helpers
                    # already existed a few hundred lines up in this same file
                    # — they simply had no caller, which is why nothing flagged
                    # it. What it cost, all of it visible only by eye:
                    #   cartpole  — rails are `zaxis="1 0 0"` capsules, drawn
                    #               standing on end instead of lying flat;
                    #   cheetah   — `euler="0 -218 0"` geoms drawn unrotated
                    #               (this file's own docstring cites that very
                    #               attribute as a precision case, while the
                    #               render path was throwing it away);
                    #   swimmer   — same, and point_mass's walls likewise.
                    # MuJoCo permits only ONE of these per element, so the
                    # order here is a fallback chain, not a precedence rule.
                    var quat_s = _extract_attr(tag, "quat")
                    var aa_s = _extract_attr(tag, "axisangle")
                    var zax_s = _extract_attr(tag, "zaxis")
                    var eul_s = _extract_attr(tag, "euler")
                    var xy_s2 = _extract_attr(tag, "xyaxes")
                    if quat_s.byte_length() > 0:
                        var qv = _parse_quat(quat_s)
                        data.geom_quat_x[geom_count] = qv[0]
                        data.geom_quat_y[geom_count] = qv[1]
                        data.geom_quat_z[geom_count] = qv[2]
                        data.geom_quat_w[geom_count] = qv[3]
                    elif aa_s.byte_length() > 0:
                        var aq = _parse_axisangle_to_quat(aa_s, deg_factor)
                        data.geom_quat_x[geom_count] = aq[0]
                        data.geom_quat_y[geom_count] = aq[1]
                        data.geom_quat_z[geom_count] = aq[2]
                        data.geom_quat_w[geom_count] = aq[3]
                    elif zax_s.byte_length() > 0:
                        var zq = _parse_zaxis_to_quat(zax_s)
                        data.geom_quat_x[geom_count] = zq[0]
                        data.geom_quat_y[geom_count] = zq[1]
                        data.geom_quat_z[geom_count] = zq[2]
                        data.geom_quat_w[geom_count] = zq[3]
                    elif eul_s.byte_length() > 0:
                        var eq = _parse_euler_to_quat(eul_s, deg_factor)
                        data.geom_quat_x[geom_count] = eq[0]
                        data.geom_quat_y[geom_count] = eq[1]
                        data.geom_quat_z[geom_count] = eq[2]
                        data.geom_quat_w[geom_count] = eq[3]
                    elif xy_s2.byte_length() > 0:
                        var xq2 = _rcd_xyaxes_to_quat(xy_s2)
                        data.geom_quat_x[geom_count] = xq2[0]
                        data.geom_quat_y[geom_count] = xq2[1]
                        data.geom_quat_z[geom_count] = xq2[2]
                        data.geom_quat_w[geom_count] = xq2[3]
                var size_s = _extract_attr(tag, "size")
                if size_s.byte_length() == 0:
                    size_s = cls_size[ci] if ci >= 0 else String("")
                if size_s.byte_length() == 0:
                    size_s = def_geom_size
                if size_s.byte_length() > 0:
                    var size_parts = List[String]()
                    _split_spaces(size_s, size_parts)
                    var s0 = Float64(0)
                    var s1 = Float64(0)
                    var s2 = Float64(0)
                    if len(size_parts) >= 1:
                        s0 = _parse_float(size_parts[0])
                    if len(size_parts) >= 2:
                        s1 = _parse_float(size_parts[1])
                    if len(size_parts) >= 3:
                        s2 = _parse_float(size_parts[2])
                    var gt = data.geom_type[geom_count]
                    if gt == 1:  # SPHERE
                        data.geom_radius[geom_count] = s0
                        data.geom_half_x[geom_count] = s0
                        data.geom_half_y[geom_count] = s0
                        data.geom_half_z[geom_count] = s0
                    elif gt == 2:  # CAPSULE
                        data.geom_radius[geom_count] = s0
                        # Only use size[1] as half-length if no fromto was
                        # specified (fromto already set the correct value).
                        if len(size_parts) >= 2 and fromto_s.byte_length() == 0:
                            data.geom_half_length[geom_count] = s1
                    elif gt == 3:  # BOX
                        data.geom_half_x[geom_count] = s0
                        data.geom_half_y[geom_count] = s1
                        data.geom_half_z[geom_count] = s2
                        data.geom_radius[geom_count] = _sqrt_f64(
                            s0 * s0 + s1 * s1 + s2 * s2
                        )
                    elif gt == 4:  # CYLINDER
                        data.geom_radius[geom_count] = s0
                        if fromto_s.byte_length() == 0:
                            data.geom_half_length[geom_count] = s1
                    elif gt == 6:  # ELLIPSOID — `size` is three semi-axes
                        data.geom_half_x[geom_count] = s0
                        data.geom_half_y[geom_count] = s1
                        data.geom_half_z[geom_count] = s2
                        data.geom_radius[geom_count] = s0
                    elif gt == 0:  # PLANE
                        data.geom_half_x[geom_count] = s0
                        data.geom_half_y[geom_count] = s1
                    else:
                        data.geom_radius[geom_count] = s0
                var mat_s = _extract_attr(tag, "material")
                if mat_s.byte_length() == 0:
                    mat_s = cls_material[ci] if ci >= 0 else String("")
                geom_mat_name[geom_count] = mat_s
                var rgba_s = _extract_attr(tag, "rgba")
                if rgba_s.byte_length() == 0:
                    rgba_s = cls_rgba[ci] if ci >= 0 else String("")
                geom_has_rgba[geom_count] = rgba_s.byte_length() > 0
                if rgba_s.byte_length() > 0:
                    var cv = _rcd_parse_rgba4(rgba_s)
                    data.geom_rgba_r[geom_count] = cv[0]
                    data.geom_rgba_g[geom_count] = cv[1]
                    data.geom_rgba_b[geom_count] = cv[2]
                    data.geom_rgba_a[geom_count] = cv[3]
                elif def_rgba_r >= Float64(0):
                    data.geom_rgba_r[geom_count] = def_rgba_r
                    data.geom_rgba_g[geom_count] = def_rgba_g
                    data.geom_rgba_b[geom_count] = def_rgba_b
                    data.geom_rgba_a[geom_count] = def_rgba_a
            geom_count += 1
            var tag_end = worldbody.find(">", next_geom)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen
        elif earliest == next_light:
            var tag = _extract_opening_tag(worldbody, next_light)
            if light_count < 8:
                var dir_s = _extract_attr(tag, "dir")
                if dir_s.byte_length() > 0:
                    var dv = _parse_vec3(dir_s)
                    data.light_dir_x[light_count] = dv[0]
                    data.light_dir_y[light_count] = dv[1]
                    data.light_dir_z[light_count] = dv[2]
                var diff_s = _extract_attr(tag, "diffuse")
                if diff_s.byte_length() > 0:
                    var c = _rcd_parse_rgb3(diff_s)
                    data.light_diffuse_r[light_count] = c[0]
                    data.light_diffuse_g[light_count] = c[1]
                    data.light_diffuse_b[light_count] = c[2]
                var spec_s = _extract_attr(tag, "specular")
                if spec_s.byte_length() > 0:
                    var c = _rcd_parse_rgb3(spec_s)
                    data.light_specular_r[light_count] = c[0]
                    data.light_specular_g[light_count] = c[1]
                    data.light_specular_b[light_count] = c[2]
                var amb_s = _extract_attr(tag, "ambient")
                if amb_s.byte_length() > 0:
                    var c = _rcd_parse_rgb3(amb_s)
                    data.light_ambient_r[light_count] = c[0]
                    data.light_ambient_g[light_count] = c[1]
                    data.light_ambient_b[light_count] = c[2]
                data.light_directional[light_count] = (
                    _extract_attr(tag, "directional") == "true"
                )
                if _extract_attr(tag, "castshadow") == "false":
                    data.light_castshadow[light_count] = False
                var exp_s = _extract_attr(tag, "exponent")
                if exp_s.byte_length() > 0:
                    data.light_exponent[light_count] = _parse_float(exp_s)
            light_count += 1
            var tag_end = worldbody.find(">", next_light)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen
        elif earliest == next_cam:
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_cam)
            if cam_count < 8:
                data.cam_body_id[cam_count] = current_body
                var pos_s = _extract_attr(tag, "pos")
                if pos_s.byte_length() > 0:
                    var pv = _parse_vec3(pos_s)
                    data.cam_pos_x[cam_count] = pv[0]
                    data.cam_pos_y[cam_count] = pv[1]
                    data.cam_pos_z[cam_count] = pv[2]
                # ⚠ ALL FIVE ORIENTATION FORMS, same as geoms. This chain
                # stopped after quat/axisangle/xyaxes, so `zaxis` and `euler`
                # fell through to the identity — which for a camera means
                # looking straight DOWN its own -Z. Once `setup_cameras` began
                # deriving the look direction from the orientation, that turned
                # into a top-down view of the floor for every model whose
                # camera uses one of the two missing forms:
                #   acrobot, cartpole  — `zaxis="0 -1 0"`
                #   walker (side)      — `euler="60 0 0"`
                # Models that genuinely ARE top-down (point_mass and reacher
                # declare `quat="1 0 0 0"`, fish defaults to `tracking_top`)
                # were correct before and stay correct.
                var quat_s = _extract_attr(tag, "quat")
                var aa_s = _extract_attr(tag, "axisangle")
                var zax_s = _extract_attr(tag, "zaxis")
                var eul_s = _extract_attr(tag, "euler")
                var xy_s = _extract_attr(tag, "xyaxes")
                if quat_s.byte_length() > 0:
                    var qv = _parse_quat(quat_s)
                    data.cam_quat_x[cam_count] = qv[0]
                    data.cam_quat_y[cam_count] = qv[1]
                    data.cam_quat_z[cam_count] = qv[2]
                    data.cam_quat_w[cam_count] = qv[3]
                elif aa_s.byte_length() > 0:
                    var aq = _parse_axisangle_to_quat(aa_s, deg_factor)
                    data.cam_quat_x[cam_count] = aq[0]
                    data.cam_quat_y[cam_count] = aq[1]
                    data.cam_quat_z[cam_count] = aq[2]
                    data.cam_quat_w[cam_count] = aq[3]
                elif zax_s.byte_length() > 0:
                    var zq = _parse_zaxis_to_quat(zax_s)
                    data.cam_quat_x[cam_count] = zq[0]
                    data.cam_quat_y[cam_count] = zq[1]
                    data.cam_quat_z[cam_count] = zq[2]
                    data.cam_quat_w[cam_count] = zq[3]
                elif eul_s.byte_length() > 0:
                    var eq = _parse_euler_to_quat(eul_s, deg_factor)
                    data.cam_quat_x[cam_count] = eq[0]
                    data.cam_quat_y[cam_count] = eq[1]
                    data.cam_quat_z[cam_count] = eq[2]
                    data.cam_quat_w[cam_count] = eq[3]
                elif xy_s.byte_length() > 0:
                    var xq = _rcd_xyaxes_to_quat(xy_s)
                    data.cam_quat_x[cam_count] = xq[0]
                    data.cam_quat_y[cam_count] = xq[1]
                    data.cam_quat_z[cam_count] = xq[2]
                    data.cam_quat_w[cam_count] = xq[3]
                var fovy_s = _extract_attr(tag, "fovy")
                if fovy_s.byte_length() > 0:
                    data.cam_fovy[cam_count] = _parse_float(fovy_s)
                var mode_s = _extract_attr(tag, "mode")
                if mode_s.byte_length() > 0:
                    data.cam_mode[cam_count] = _rcd_cam_mode_from_str(mode_s)
                # `target="body"` — only meaningful for targetbody(com), and
                # resolved here so the per-frame re-aim is pure arithmetic.
                var tgt_s = _trim(_extract_attr(tag, "target"))
                if tgt_s.byte_length() > 0:
                    data.cam_target_body[cam_count] = (
                        _find_body_index_by_name(worldbody, tgt_s)
                    )
            cam_count += 1
            var tag_end = worldbody.find(">", next_cam)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen
        elif earliest == next_site:
            var current_body = body_id_stack[depth]
            var tag = _extract_opening_tag(worldbody, next_site)
            if site_count < MAX_COMPTIME_RENDER_SITES:
                data.site_body_id[site_count] = current_body
                # `fromto` supersedes `pos` on a site exactly as it does on a
                # geom (user_objects.cc:3841). This record has no quaternion
                # field, so only the midpoint lands here — enough for the
                # render path, which is all this parser feeds. The runtime
                # parser carries the orientation and the size remap.
                var site_ft_s = _extract_attr(tag, "fromto")
                if site_ft_s.byte_length() > 0:
                    var sft = _fromto_to_pos_quat(site_ft_s)
                    data.site_pos_x[site_count] = sft[0]
                    data.site_pos_y[site_count] = sft[1]
                    data.site_pos_z[site_count] = sft[2]
                else:
                    var pos_s = _extract_attr(tag, "pos")
                    if pos_s.byte_length() > 0:
                        var pv = _parse_vec3(pos_s)
                        data.site_pos_x[site_count] = pv[0]
                        data.site_pos_y[site_count] = pv[1]
                        data.site_pos_z[site_count] = pv[2]
                var size_s = _extract_attr(tag, "size")
                if size_s.byte_length() > 0:
                    var sparts = List[String]()
                    _split_spaces(size_s, sparts)
                    if len(sparts) >= 1:
                        data.site_size_0[site_count] = _parse_float(sparts[0])
            site_count += 1
            var tag_end = worldbody.find(">", next_site)
            scan_pos = tag_end + 1 if tag_end != -1 else wlen
        else:
            scan_pos = earliest + 1

    data.ngeom = geom_count
    data.nlight = light_count
    data.ncam = cam_count
    data.nsite = site_count

    # ---- Parse <tendon>: <spatial> polylines, for DRAWING only ---------------
    # The physics gets its tendons from the runtime parser. This records the
    # site chain so the renderer can draw the segments — without it
    # ball_in_cup's string simply is not there, which is how it was reported.
    var tendon_sec = _extract_section(xml_clean, "tendon")
    var sten_pos = 0
    var sten_count = 0
    var sten_site_cursor = 0
    while sten_count < MAX_COMPTIME_SPATIAL_TENDONS:
        var sp = tendon_sec.find("<spatial", sten_pos)
        if sp == -1:
            break
        var sp_close = tendon_sec.find("</spatial", sp)
        if sp_close == -1:
            break
        var sp_head_end = tendon_sec.find(">", sp)
        if sp_head_end == -1:
            break
        var head = String(tendon_sec[byte = sp : sp_head_end + 1])
        var w_s = _extract_attr(head, "width")
        if w_s.byte_length() > 0:
            data.sten_width[sten_count] = _parse_float(w_s)
        var trgba_s = _extract_attr(head, "rgba")
        if trgba_s.byte_length() > 0:
            var tc = _rcd_parse_rgba4(trgba_s)
            data.sten_rgba_r[sten_count] = tc[0]
            data.sten_rgba_g[sten_count] = tc[1]
            data.sten_rgba_b[sten_count] = tc[2]
        # Site chain, in document order — that order IS the routing.
        var body = String(tendon_sec[byte = sp_head_end + 1 : sp_close])
        var child_pos = 0
        var n_in_this = 0
        while True:
            var cs = body.find("<site", child_pos)
            if cs == -1:
                break
            var cs_end = body.find(">", cs)
            if cs_end == -1:
                break
            var ctag = String(body[byte = cs : cs_end + 1])
            var sref = _trim(_extract_attr(ctag, "site"))
            if (
                sref.byte_length() > 0
                and sten_site_cursor < MAX_COMPTIME_SPATIAL_TENDON_SITES
            ):
                data.sten_sites[sten_site_cursor] = (
                    _rcd_find_site_index_by_name(worldbody, sref)
                )
                sten_site_cursor += 1
                n_in_this += 1
            child_pos = cs_end + 1
        data.sten_nsite[sten_count] = n_in_this
        sten_count += 1
        sten_pos = sp_close + 1
    data.nsten = sten_count

    # ---- Parse <visual> section: map, quality, headlight ----------------------
    var visual_sec = _extract_section(xml_clean, "visual")
    if visual_sec.byte_length() > 0:
        # <map znear="..." fogstart="..." fogend="..."/>
        var map_pos = visual_sec.find("<map")
        if map_pos != -1:
            var map_end = visual_sec.find(">", map_pos)
            if map_end != -1:
                var map_tag = String(visual_sec[byte = map_pos : map_end + 1])
                var znear_s = _extract_attr(map_tag, "znear")
                if znear_s.byte_length() > 0:
                    data.vis_znear = _parse_float(znear_s)
                var fogstart_s = _extract_attr(map_tag, "fogstart")
                if fogstart_s.byte_length() > 0:
                    data.vis_fogstart = _parse_float(fogstart_s)
                var fogend_s = _extract_attr(map_tag, "fogend")
                if fogend_s.byte_length() > 0:
                    data.vis_fogend = _parse_float(fogend_s)
        # <quality shadowsize="..."/>
        var qual_pos = visual_sec.find("<quality")
        if qual_pos != -1:
            var qual_end = visual_sec.find(">", qual_pos)
            if qual_end != -1:
                var qual_tag = String(visual_sec[byte = qual_pos : qual_end + 1])
                var ss_s = _extract_attr(qual_tag, "shadowsize")
                if ss_s.byte_length() > 0:
                    data.vis_shadowsize = Int(_parse_float(ss_s))
        # <headlight ambient="r g b"/>
        var hl_pos = visual_sec.find("<headlight")
        if hl_pos != -1:
            var hl_end = visual_sec.find(">", hl_pos)
            if hl_end != -1:
                var hl_tag = String(visual_sec[byte = hl_pos : hl_end + 1])
                var amb_s = _extract_attr(hl_tag, "ambient")
                if amb_s.byte_length() > 0:
                    var c = _rcd_parse_rgb3(amb_s)
                    data.vis_headlight_ambient_r = c[0]
                    data.vis_headlight_ambient_g = c[1]
                    data.vis_headlight_ambient_b = c[2]
                    data.vis_has_headlight = True

    # ---- Resolve geom material="name" -> index -------------------------------
    # Names were gathered in the geom loop above, WITH the class chain applied;
    # this pass only turns each into an index and applies the material's colour
    # where the geom did not state one of its own.
    var n_resolved = (
        geom_count if geom_count < MAX_COMPTIME_RENDER_GEOMS
        else MAX_COMPTIME_RENDER_GEOMS
    )
    for geom_idx in range(n_resolved):
        var mat_name = geom_mat_name[geom_idx]
        if mat_name.byte_length() > 0:
            var mid = _rcd_find_material_index_by_name(asset_sec, mat_name)
            data.geom_material_id[geom_idx] = mid
            if not geom_has_rgba[geom_idx] and mid >= 0 and mid < mat_count:
                data.geom_rgba_r[geom_idx] = data.mat_rgba_r[mid]
                data.geom_rgba_g[geom_idx] = data.mat_rgba_g[mid]
                data.geom_rgba_b[geom_idx] = data.mat_rgba_b[mid]
                data.geom_rgba_a[geom_idx] = data.mat_rgba_a[mid]

    return data^


# =============================================================================
# Comptime scalar helpers for GPU kernels in ModelDefFromXML
# =============================================================================


def _xml_nth_motor_gear[xml: String, n: Int]() -> Float64:
    """Return gear ratio for the n-th <motor> in <actuator> section.

    Falls back to `<default><motor gear="..."/>` and then to MuJoCo's 1.0.
    Comptime-safe.
    """
    comptime def_gear = _xml_default_motor_gear[xml]()
    var sec = _extract_section(xml, "actuator")
    var pos = 0
    var count = 0
    while True:
        var t = sec.find("<motor", pos)
        if t == -1:
            break
        # Verify valid tag (next char must be space, >, /, newline, tab)
        if sec.byte_length() > t + 6:
            var after = String(sec[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                pos = t + 6
                continue
        if count == n:
            var tag_end = sec.find(">", t)
            if tag_end == -1:
                return def_gear
            var tag = String(sec[byte = t : tag_end + 1])
            var g = _extract_attr(tag, "gear")
            if g.byte_length() == 0:
                return def_gear
            return _parse_float(g)
        count += 1
        pos = t + 6
    return def_gear


def _xml_nth_motor_dof_adr[xml: String, n: Int]() -> Int:
    """Return DOF address for the n-th motor's actuated joint.

    DOF address = sum of NV for all joints before the target joint in DFS order.
    Returns -1 if not found. Comptime-safe.
    """
    # Step 1: Extract the joint name for the n-th motor
    var sec = _extract_section(xml, "actuator")
    var pos = 0
    var count = 0
    var jname = String("")
    while True:
        var t = sec.find("<motor", pos)
        if t == -1:
            break
        if sec.byte_length() > t + 6:
            var after = String(sec[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                pos = t + 6
                continue
        if count == n:
            var tag_end = sec.find(">", t)
            if tag_end != -1:
                var tag = String(sec[byte = t : tag_end + 1])
                jname = _extract_attr(tag, "joint")
            break
        count += 1
        pos = t + 6
    if jname.byte_length() == 0:
        return -1

    # Step 2: Scan worldbody joints in DFS order to find DOF address
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var dof_adr = 0
    var search_name = 'name="' + jname + '"'
    while True:
        var t = wb.find("<joint", scan_pos)
        if t == -1:
            break
        if wb.byte_length() > t + 6:
            var after = String(wb[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                scan_pos = t + 6
                continue
        var tag_end = wb.find(">", t)
        if tag_end == -1:
            break
        var tag = String(wb[byte = t : tag_end + 1])
        if tag.find(search_name) != -1:
            return dof_adr  # Found the target joint
        # Accumulate DOF count for this joint
        var jtype = _extract_attr(tag, "type")
        if jtype == "ball":
            dof_adr += 3
        elif jtype == "free":
            dof_adr += 6
        else:  # hinge, slide, or empty (default=hinge)
            dof_adr += 1
        scan_pos = tag_end + 1
    return -1


def _xml_nth_joint_qpos_adr[xml: String, n: Int]() -> Int:
    """Return qpos address for the n-th joint in worldbody DFS order.

    qpos address = sum of NQ for all joints before joint n. Comptime-safe.
    """
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var count = 0
    var qpos_adr = 0
    while True:
        var t = wb.find("<joint", scan_pos)
        if t == -1:
            break
        if wb.byte_length() > t + 6:
            var after = String(wb[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                scan_pos = t + 6
                continue
        if count == n:
            return qpos_adr
        var tag_end = wb.find(">", t)
        if tag_end == -1:
            break
        var tag = String(wb[byte = t : tag_end + 1])
        var jtype = _extract_attr(tag, "type")
        if jtype == "free":
            qpos_adr += 7
        elif jtype == "ball":
            qpos_adr += 4
        else:
            qpos_adr += 1
        count += 1
        scan_pos = tag_end + 1
    return qpos_adr


def _xml_nth_joint_limited[xml: String, n: Int]() -> Bool:
    """Return whether the n-th joint in worldbody DFS order has limits.

    Checks per-joint limited attr first; falls back to <default><joint limited=.../>.
    Comptime-safe.
    """
    # Read default from <default> section
    var def_limited = False
    var def_sec = _root_defaults(xml)
    if def_sec.byte_length() > 0:
        var jpos = def_sec.find("<joint")
        if jpos != -1:
            var tag_end = def_sec.find(">", jpos)
            if tag_end != -1:
                var tag = String(def_sec[byte = jpos : tag_end + 1])
                var lim = _extract_attr(tag, "limited")
                if lim == "true" or lim == "1":
                    def_limited = True

    # Scan worldbody for n-th joint
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var count = 0
    while True:
        var t = wb.find("<joint", scan_pos)
        if t == -1:
            break
        if wb.byte_length() > t + 6:
            var after = String(wb[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                scan_pos = t + 6
                continue
        if count == n:
            var tag_end = wb.find(">", t)
            if tag_end == -1:
                return def_limited
            var tag = String(wb[byte = t : tag_end + 1])
            var lim = _extract_attr(tag, "limited")
            if lim == "true" or lim == "1":
                return True
            elif lim == "false" or lim == "0":
                return False
            # `compiler/autolimits` — see the twin in `parse_xml_model_data`.
            if _extract_attr(tag, "range").byte_length() > 0:
                return True
            return def_limited
        count += 1
        scan_pos = t + 6
    return False


def _xml_nth_joint_range_min[xml: String, n: Int]() -> Float64:
    """Return range_min for the n-th joint in worldbody DFS order (radians).

    Converts from degrees for ANGULAR joints when the model is in degree mode
    (MuJoCo's default) — a slide range stays in metres, matching
    mjCJoint::Compile. Returns 0.0 if no range attribute. Comptime-safe.
    """
    comptime deg_factor = (
        3.141592653589793 / 180.0 if _xml_compiler_angle_is_deg[xml]() else 1.0
    )
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var count = 0
    while True:
        var t = wb.find("<joint", scan_pos)
        if t == -1:
            break
        if wb.byte_length() > t + 6:
            var after = String(wb[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                scan_pos = t + 6
                continue
        if count == n:
            var tag_end = wb.find(">", t)
            if tag_end == -1:
                return Float64(0.0)
            var tag = String(wb[byte = t : tag_end + 1])
            var range_str = _extract_attr(tag, "range")
            if range_str.byte_length() == 0:
                return Float64(0.0)
            var ts = _trim(_extract_attr(tag, "type"))
            var angular = ts == "" or ts == "hinge" or ts == "ball"
            var rf = deg_factor if angular else 1.0
            var parts = List[String]()
            _split_spaces(range_str, parts)
            if len(parts) >= 1:
                return _parse_float(parts[0]) * rf
            return Float64(0.0)
        count += 1
        scan_pos = t + 6
    return Float64(0.0)


def _xml_nth_joint_range_max[xml: String, n: Int]() -> Float64:
    """Return range_max for the n-th joint in worldbody DFS order (radians).

    Converts from degrees for ANGULAR joints when the model is in degree mode
    (MuJoCo's default) — a slide range stays in metres, matching
    mjCJoint::Compile. Returns 0.0 if no range attribute. Comptime-safe.
    """
    comptime deg_factor = (
        3.141592653589793 / 180.0 if _xml_compiler_angle_is_deg[xml]() else 1.0
    )
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var count = 0
    while True:
        var t = wb.find("<joint", scan_pos)
        if t == -1:
            break
        if wb.byte_length() > t + 6:
            var after = String(wb[byte = t + 6 : t + 7])
            if (
                after != " "
                and after != ">"
                and after != "/"
                and after != "\n"
                and after != "\t"
            ):
                scan_pos = t + 6
                continue
        if count == n:
            var tag_end = wb.find(">", t)
            if tag_end == -1:
                return Float64(0.0)
            var tag = String(wb[byte = t : tag_end + 1])
            var range_str = _extract_attr(tag, "range")
            if range_str.byte_length() == 0:
                return Float64(0.0)
            var ts = _trim(_extract_attr(tag, "type"))
            var angular = ts == "" or ts == "hinge" or ts == "ball"
            var rf = deg_factor if angular else 1.0
            var parts = List[String]()
            _split_spaces(range_str, parts)
            if len(parts) >= 2:
                return _parse_float(parts[1]) * rf
            return Float64(0.0)
        count += 1
        scan_pos = t + 6
    return Float64(0.0)
