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
    # ⚠ NO `ANGLE_DEG`. It was carried here and READ NOWHERE, and it is the
    # one field `mjModel` does not retain — MuJoCo's compiler converts angles
    # to radians and discards the unit. Phase 1b generates this struct from
    # MuJoCo, so a field only our own scanner can produce would have had to be
    # invented by the generator. Angle units are still resolved where they are
    # actually needed, by `_compiler_angle_is_deg` at the point of use.
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


# ═══ THE `<default>` INDEX CLUSTER — DELETED (phase 1a.5c) ═════════════════
#
# `_DefaultsIndex`, `_build_defaults_index`, `_class_attr_indexed`,
# `_class_parent_indexed`, `_class_attr_inherited_indexed`, `_AttrCache`,
# `_attr_3way_cached`, plus the rescanning `_class_attr_inherited` and
# `_attr_3way` they replaced.
#
# Written FOR `parse_xml_model_data`'s actuator loop and
# `parse_xml_render_data`'s geom loop, where they killed ~340 O(n) document
# re-walks per model. Phase 1a.4e deleted the first caller and 1a.5c the
# second, leaving the whole cluster reachable only from its own equivalence
# test — a gate proving an optimisation still matched a scan that no longer
# exists. Both are gone.
#
# ⚠ The `<default>` chain itself is NOT gone and is not optional: it is what
# `full_parser` resolves at runtime, and every time a parser skipped it the
# result was a silent wrong value rather than a missing one — geom `type`
# (quadruped's legs), actuator classes, joint limits, geom `material`. The
# index was an optimisation; the semantics live in `full_parser`.


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

    ⚠ NOTHING IN THE TREE CALLS THIS ANY MORE. It composed the 34 models that
    were built from shared fragments, and phase 1b.5 replaced those with flat
    `.xml` assets on disk. It is kept, rather than deleted, because it is the
    natural implementation of a real `<include file=...>`: read the included
    file and merge it, which reproduces today's composition EXACTLY and is the
    obvious follow-up to the flat assets. If that never happens, delete it —
    do not let it rot as unreferenced code with a bug history.


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

# ⚠ SIX RENDER CAPS WERE DELETED HERE (phase 1a.5c), with the fixed arrays
# they bounded: MAX_COMPTIME_RENDER_GEOMS (448), _SITES (192), _MESHES (32),
# MAX_COMPTIME_TEXTURES (16), MAX_COMPTIME_MATERIALS (32) and
# MAX_COMPTIME_SPATIAL_TENDONS (4) / _TENDON_SITES (16).
#
# Their notes are worth keeping as a class of bug rather than as constants.
# Every one of them was widened at least once AFTER a model outgrew it, and
# the failure was never a missing element:
#
#   * materials 8 -> 32. dm_control's shared asset block declares THIRTEEN
#     materials and every suite model includes it. Ids came from a scan that
#     knew nothing about the cap and were bounds-checked against `nmat`, an
#     INDEPENDENT uncapped count — so `mid = 9` passed `mid < nmat` and then
#     indexed an 8-slot array. point_mass, fish and reacher aborted on sight;
#     models whose materials happened to land under 8 merely rendered wrong.
#   * geoms 64 -> 160 -> 448 and sites 16 -> 48 -> 192, for dog and then for
#     manipulation's brick tasks (reassemble_5: 431 geoms, 181 sites).
#   * meshes 16 -> 32. SO-ARM100 declares 18, so two were dropped silently
#     and any geom naming them did not draw.
#
# `RenderFields` is `List`-backed and sized by what the model has, so none of
# this is expressible any more. The remaining MAX_COMPTIME_* constants below
# belong to `parse_xml`, which still runs.

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


# ═══ `ComptimeActData` / `parse_xml_model_data` DELETED (phase 1a.4e) ═══════
#
# 1269 lines: a struct of ~20 `InlineArray`s and the ~800-line scan that
# filled it, both INTERPRETED AT COMPTIME for every model that named a
# `ModelDefFromXML`. It was the actuator tables, the reference pose, the
# keyframes and the joint limit tables — everything `SpecFields` now holds,
# built at runtime by `full_parser` + `fields_build.build_spec_fields`.
#
# ⚠ THE HELPERS BELOW IT SURVIVE because `parse_xml` and
# `parse_xml_render_data` still use them. `_attr_3way_cached` and
# `_DefaultsIndex` were written FOR this scan (they replaced ~340 O(n)
# re-walks per model); if `_rcd` goes the same way in 1a.5, check whether
# anything still calls them before assuming they are load-bearing.
#
# ⚠ Two known defects died with it rather than being fixed, because the code
# that had them is gone and the runtime path does not: its joint scan read
# `limited`/`range` off the ELEMENT TAG only (so quadruped reported 0 of 17
# limited joints, and so_arm100 0 of 6), and `_normalize_freejoint` was
# missing from its entry point (so ten single-file manipulation models got
# `free_joint_qpos_adr = -1` and a zero reset quaternion). Both are recorded
# in the assessment doc, §10.13 and §10.15, so the shapes stay findable.

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
