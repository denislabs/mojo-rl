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

    var NBODY: Int   # total bodies including worldbody (= counted bodies + 1)
    var NJOINT: Int  # number of joints
    var NQ: Int      # total position DOFs
    var NV: Int      # total velocity DOFs
    var NGEOM: Int   # total geoms (floor + body geoms)
    var NACT: Int    # number of actuators
    var NTEX: Int    # number of <texture> entries in <asset>
    var NMAT: Int    # number of <material> entries in <asset>
    var NLIGHT: Int  # number of <light> entries in <worldbody>
    var NCAM: Int    # number of <camera> entries in <worldbody>
    var NSITE: Int   # number of <site> entries in <worldbody>

    fn __init__(
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

    fn __str__(self) -> String:
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
            + ")"
        )


# =============================================================================
# Low-level string helpers (comptime-friendly, no subscript access)
# =============================================================================


fn _count_exact(xml: String, search: String) -> Int:
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


fn _count_tag(xml: String, tag: String) -> Int:
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


fn _extract_section(xml: String, tag: String) -> String:
    """Return the substring from `<tag` to (including) `</tag>`.

    Returns empty string if the section is not found.
    Handles `<tag>` and `<tag ...>` (with attributes).
    """
    var open_marker = "<" + tag
    var close_marker = "</" + tag + ">"
    var start = xml.find(open_marker)
    if start == -1:
        return String("")
    var end = xml.find(close_marker, start)
    if end == -1:
        return String("")
    return String(xml[start : end + len(close_marker)])


# =============================================================================
# Attribute extraction helpers (comptime-friendly, no subscript access)
# =============================================================================


fn _trim(s: String) -> String:
    """Trim leading/trailing whitespace (space, tab, newline, carriage return)."""
    var start = 0
    var end = len(s)
    while start < end:
        var c = s[start : start + 1]
        if c == " " or c == "\t" or c == "\n" or c == "\r":
            start += 1
        else:
            break
    while end > start:
        var c = s[end - 1 : end]
        if c == " " or c == "\t" or c == "\n" or c == "\r":
            end -= 1
        else:
            break
    if start >= end:
        return String("")
    return String(s[start:end])


fn _extract_opening_tag(xml: String, pos: Int) -> String:
    """From `<tag` at pos, extract everything up to (and including) `>` or `/>`.

    Returns the raw opening-tag string for attribute parsing.
    """
    var end = xml.find(">", pos)
    if end == -1:
        return String("")
    return String(xml[pos : end + 1])


fn _extract_attr(tag: String, attr: String) -> String:
    """Extract value from attr="value" or attr='value' in a tag string.

    Returns "" if not found.
    """
    # Try double-quoted form: attr="..."
    var search_dq = attr + '="'
    var pos = tag.find(search_dq)
    if pos != -1:
        var val_start = pos + len(search_dq)
        var val_end = tag.find('"', val_start)
        if val_end != -1:
            return String(tag[val_start:val_end])
    # Try single-quoted form: attr='...'
    var search_sq = attr + "='"
    pos = tag.find(search_sq)
    if pos != -1:
        var val_start = pos + len(search_sq)
        var val_end = tag.find("'", val_start)
        if val_end != -1:
            return String(tag[val_start:val_end])
    return String("")


fn _digit_value(c: String) -> Int:
    """Return integer value 0-9 for digit character; -1 if not a digit."""
    var digits = "0123456789"
    return digits.find(c)


fn _parse_float(s: String) -> Float64:
    """Parse a float string such as "0.7", "-3.14", "1e-3" to Float64.

    Uses slice-based character iteration (s[i:i+1]) — comptime-safe.
    No stdlib float parsing is used.
    """
    var t = _trim(s)
    if len(t) == 0:
        return Float64(0)

    # Sign
    var neg = False
    var start = 0
    if t[0:1] == "-":
        neg = True
        start = 1
    elif t[0:1] == "+":
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
        int_end = len(t)

    var int_part = Float64(0)
    for i in range(start, int_end):
        var d = _digit_value(String(t[i : i + 1]))
        if d >= 0:
            int_part = int_part * 10.0 + Float64(d)

    # Fractional part
    var frac_part = Float64(0)
    if dot_pos != -1:
        var frac_end: Int
        if exp_pos != -1:
            frac_end = exp_pos
        else:
            frac_end = len(t)
        var frac_mul = Float64(0.1)
        for i in range(dot_pos + 1, frac_end):
            var d = _digit_value(String(t[i : i + 1]))
            if d >= 0:
                frac_part += Float64(d) * frac_mul
                frac_mul *= 0.1

    var result = int_part + frac_part

    # Exponent part
    if exp_pos != -1:
        var exp_start = exp_pos + 1
        var exp_neg = False
        if exp_start < len(t):
            if t[exp_start : exp_start + 1] == "-":
                exp_neg = True
                exp_start += 1
            elif t[exp_start : exp_start + 1] == "+":
                exp_start += 1
        var exp_val = 0
        for i in range(exp_start, len(t)):
            var d = _digit_value(String(t[i : i + 1]))
            if d >= 0:
                exp_val = exp_val * 10 + d
        var pow10 = Float64(1.0)
        for _ in range(exp_val):
            if exp_neg:
                pow10 *= 0.1
            else:
                pow10 *= 10.0
        result *= pow10

    if neg:
        return -result
    return result


fn _parse_int_str(s: String) -> Int:
    """Parse "3", "-1" etc. to Int."""
    var t = _trim(s)
    if len(t) == 0:
        return 0
    var neg = False
    var start = 0
    if t[0:1] == "-":
        neg = True
        start = 1
    var val = 0
    for i in range(start, len(t)):
        var d = _digit_value(String(t[i : i + 1]))
        if d >= 0:
            val = val * 10 + d
    if neg:
        return -val
    return val


fn _split_spaces(s: String, mut parts: List[String]):
    """Split string by whitespace runs into parts (in-place fill)."""
    var t = _trim(s)
    var start = 0
    var n = len(t)
    while start < n:
        # Skip whitespace
        while start < n:
            var c = t[start : start + 1]
            if c == " " or c == "\t" or c == "\n" or c == "\r":
                start += 1
            else:
                break
        if start >= n:
            break
        # Find end of token
        var end = start + 1
        while end < n:
            var c = t[end : end + 1]
            if c == " " or c == "\t" or c == "\n" or c == "\r":
                break
            end += 1
        parts.append(String(t[start:end]))
        start = end


fn _parse_vec3(s: String) -> Tuple[Float64, Float64, Float64]:
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


fn _parse_quat(s: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """Parse "x y z w" space-separated string into (qx, qy, qz, qw)."""
    var parts = List[String]()
    _split_spaces(s, parts)
    var qx = Float64(0)
    var qy = Float64(0)
    var qz = Float64(0)
    var qw = Float64(1)
    if len(parts) >= 1:
        qx = _parse_float(parts[0])
    if len(parts) >= 2:
        qy = _parse_float(parts[1])
    if len(parts) >= 3:
        qz = _parse_float(parts[2])
    if len(parts) >= 4:
        qw = _parse_float(parts[3])
    return (qx, qy, qz, qw)


fn _sqrt_f64(x: Float64) -> Float64:
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


fn _axisangle_to_quat(
    ax: Float64, ay: Float64, az: Float64, angle: Float64
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Convert axis-angle (ax,ay,az,angle_rad) to quaternion (qx,qy,qz,qw).

    Uses Taylor series for sin/cos — comptime-safe without math stdlib.
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

    # sin(angle/2) and cos(angle/2) via Taylor series
    var a = angle * Float64(0.5)
    # cos(a) = 1 - a²/2 + a⁴/24 - a⁶/720 + a⁸/40320 - a¹⁰/3628800
    var a2 = a * a
    var cos_a = (
        Float64(1)
        - a2 / Float64(2)
        + a2 * a2 / Float64(24)
        - a2 * a2 * a2 / Float64(720)
        + a2 * a2 * a2 * a2 / Float64(40320)
        - a2 * a2 * a2 * a2 * a2 / Float64(3628800)
    )
    # sin(a) = a - a³/6 + a⁵/120 - a⁷/5040 + a⁹/362880
    var sin_a = (
        a
        - a * a2 / Float64(6)
        + a * a2 * a2 / Float64(120)
        - a * a2 * a2 * a2 / Float64(5040)
        + a * a2 * a2 * a2 * a2 / Float64(362880)
    )
    return (nx * sin_a, ny * sin_a, nz * sin_a, cos_a)


fn _parse_axisangle_to_quat(
    s: String,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Parse MuJoCo axisangle="ax ay az angle_rad" → quaternion (qx,qy,qz,qw)."""
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
        angle = _parse_float(parts[3])
    return _axisangle_to_quat(ax, ay, az, angle)


fn _fromto_to_pos_quat(
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

    # Direction vector
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var length = _sqrt_f64(dx * dx + dy * dy + dz * dz)
    var half_length = length * Float64(0.5)

    if length < Float64(1e-10):
        return (mx, my, mz, Float64(0), Float64(0), Float64(0), Float64(1), half_length, Float64(0))

    var ndx = dx / length
    var ndy = dy / length
    var ndz = dz / length

    # Quaternion to rotate Z=(0,0,1) to direction (ndx,ndy,ndz)
    # Using half-angle formulas (sqrt only, no trig):
    #   cos(θ/2) = sqrt((1+ndz)/2),  rotation axis = (-ndy, ndx, 0) / sin(θ)
    #   qx = -ndy / sqrt(2*(1+ndz)),  qy = ndx / sqrt(2*(1+ndz)),  qz=0, qw=sqrt((1+ndz)/2)
    var qx: Float64
    var qy: Float64
    var qz: Float64
    var qw: Float64

    if ndz > Float64(0.9999):
        # Already pointing in +Z direction
        qx = Float64(0)
        qy = Float64(0)
        qz = Float64(0)
        qw = Float64(1)
    elif ndz < Float64(-0.9999):
        # Pointing in -Z direction: 180° rotation around X
        qx = Float64(1)
        qy = Float64(0)
        qz = Float64(0)
        qw = Float64(0)
    else:
        var denom = _sqrt_f64(Float64(2) * (Float64(1) + ndz))
        qx = -ndy / denom
        qy = ndx / denom
        qz = Float64(0)
        qw = _sqrt_f64((Float64(1) + ndz) * Float64(0.5))
        # Normalise
        var qlen = _sqrt_f64(qx * qx + qy * qy + qz * qz + qw * qw)
        if qlen > Float64(1e-10):
            qx = qx / qlen
            qy = qy / qlen
            qw = qw / qlen

    return (mx, my, mz, qx, qy, qz, qw, half_length, Float64(0))


fn _find_joint_index_by_name(worldbody: String, joint_name: String) -> Int:
    """Return 0-based index of first <joint name="joint_name"> in DFS order, or -1."""
    var search_name = 'name="' + joint_name + '"'
    var count = 0
    var scan_pos = 0
    var searching = True
    while searching:
        var joint_pos = worldbody.find("<joint", scan_pos)
        if joint_pos == -1:
            return -1
        var tag_end = worldbody.find(">", joint_pos)
        if tag_end == -1:
            return -1
        var tag = String(worldbody[joint_pos : tag_end + 1])
        if tag.find(search_name) != -1:
            return count
        count += 1
        scan_pos = tag_end + 1
    return -1


fn _count_joints_with_type(xml: String, joint_type: String) -> Int:
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
        var tag_content = String(xml[pos : end_pos + 1])
        if tag_content.find(type_attr) != -1:
            count += 1
        start = end_pos + 1
    return count


# =============================================================================
# Main entry point
# =============================================================================


fn parse_xml(xml: String) -> ParsedModel:
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

    # ---- Isolate sections to avoid counting <default> entries ---------------
    var worldbody = _extract_section(xml, "worldbody")
    var actuator_sec = _extract_section(xml, "actuator")

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
    var asset_sec = _extract_section(xml, "asset")
    var ntex = _count_tag(asset_sec, "texture")
    var nmat = _count_tag(asset_sec, "material")

    # ---- Visual elements in worldbody ---------------------------------------
    var nlight = _count_tag(worldbody, "light")
    var ncam = _count_tag(worldbody, "camera")
    var nsite = _count_tag(worldbody, "site")

    return ParsedModel(nbody, njoint, nq, nv, ngeom, nact, ntex, nmat, nlight, ncam, nsite)


# =============================================================================
# Comptime scalar helpers for GPU kernels in ModelDefFromXML
# =============================================================================


fn _xml_nth_motor_gear[xml: String, n: Int]() -> Float64:
    """Return gear ratio for the n-th <motor> in <actuator> section.

    Returns 1.0 if not found or no gear attribute. Comptime-safe.
    """
    var sec = _extract_section(xml, "actuator")
    var pos = 0
    var count = 0
    while True:
        var t = sec.find("<motor", pos)
        if t == -1:
            break
        # Verify valid tag (next char must be space, >, /, newline, tab)
        if len(sec) > t + 6:
            var after = String(sec[t + 6 : t + 7])
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
                return Float64(1.0)
            var tag = String(sec[t : tag_end + 1])
            var g = _extract_attr(tag, "gear")
            if len(g) == 0:
                return Float64(1.0)
            return _parse_float(g)
        count += 1
        pos = t + 6
    return Float64(1.0)


fn _xml_nth_motor_dof_adr[xml: String, n: Int]() -> Int:
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
        if len(sec) > t + 6:
            var after = String(sec[t + 6 : t + 7])
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
                var tag = String(sec[t : tag_end + 1])
                jname = _extract_attr(tag, "joint")
            break
        count += 1
        pos = t + 6
    if len(jname) == 0:
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
        if len(wb) > t + 6:
            var after = String(wb[t + 6 : t + 7])
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
        var tag = String(wb[t : tag_end + 1])
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


fn _xml_nth_joint_qpos_adr[xml: String, n: Int]() -> Int:
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
        if len(wb) > t + 6:
            var after = String(wb[t + 6 : t + 7])
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
        var tag = String(wb[t : tag_end + 1])
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


fn _xml_nth_joint_limited[xml: String, n: Int]() -> Bool:
    """Return whether the n-th joint in worldbody DFS order has limits.

    Checks per-joint limited attr first; falls back to <default><joint limited=.../>.
    Comptime-safe.
    """
    # Read default from <default> section
    var def_limited = False
    var def_sec = _extract_section(xml, "default")
    if len(def_sec) > 0:
        var jpos = def_sec.find("<joint")
        if jpos != -1:
            var tag_end = def_sec.find(">", jpos)
            if tag_end != -1:
                var tag = String(def_sec[jpos : tag_end + 1])
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
        if len(wb) > t + 6:
            var after = String(wb[t + 6 : t + 7])
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
            var tag = String(wb[t : tag_end + 1])
            var lim = _extract_attr(tag, "limited")
            if lim == "true" or lim == "1":
                return True
            elif lim == "false" or lim == "0":
                return False
            return def_limited
        count += 1
        scan_pos = t + 6
    return False


fn _xml_nth_joint_range_min[xml: String, n: Int]() -> Float64:
    """Return range_min for the n-th joint in worldbody DFS order.

    Returns 0.0 if no range attribute. Comptime-safe.
    """
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var count = 0
    while True:
        var t = wb.find("<joint", scan_pos)
        if t == -1:
            break
        if len(wb) > t + 6:
            var after = String(wb[t + 6 : t + 7])
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
            var tag = String(wb[t : tag_end + 1])
            var range_str = _extract_attr(tag, "range")
            if len(range_str) == 0:
                return Float64(0.0)
            var parts = List[String]()
            _split_spaces(range_str, parts)
            if len(parts) >= 1:
                return _parse_float(parts[0])
            return Float64(0.0)
        count += 1
        scan_pos = t + 6
    return Float64(0.0)


fn _xml_nth_joint_range_max[xml: String, n: Int]() -> Float64:
    """Return range_max for the n-th joint in worldbody DFS order.

    Returns 0.0 if no range attribute. Comptime-safe.
    """
    var wb = _extract_section(xml, "worldbody")
    var scan_pos = 0
    var count = 0
    while True:
        var t = wb.find("<joint", scan_pos)
        if t == -1:
            break
        if len(wb) > t + 6:
            var after = String(wb[t + 6 : t + 7])
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
            var tag = String(wb[t : tag_end + 1])
            var range_str = _extract_attr(tag, "range")
            if len(range_str) == 0:
                return Float64(0.0)
            var parts = List[String]()
            _split_spaces(range_str, parts)
            if len(parts) >= 2:
                return _parse_float(parts[1])
            return Float64(0.0)
        count += 1
        scan_pos = t + 6
    return Float64(0.0)
