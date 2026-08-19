"""`<attach>` and `<frame>` — MJCF composition, expanded to flat text — S2.

## What this is for

The physics3d studio's scene file IS MJCF. MuJoCo already standardises
composition, so a scene is:

    <asset>
      <model name="cube" file="cube.xml"/>
    </asset>
    <worldbody>
      <frame pos="0.3 0 0.5"><attach model="cube" prefix="cube1_"/></frame>
    </worldbody>

⇒ **MuJoCo can load the scene file unchanged, so `mjModel` is the oracle for
the whole composer** — and `mj_saveLastXML` emits the flattened form, which
makes MuJoCo's own flattening the golden for this file. That is a far stronger
gate than "it parses", and it is the reason the format was chosen (see
`docs/PHYSICS3D_STUDIO_PLAN.md` §3).

`full_parser` knows neither element. Rather than teach it, this expands them
into flat text BEFORE it runs, so there is exactly one parser and the expander
produces text the existing one already reads. §10 risk 2: do not let the
studio become a second model path.

## The three steps

1. **splice** — `<attach>` is replaced by the sub-model's bodies;
2. **prefix** — every `name=` and every NAME REFERENCE in the sub-model gains
   the instance prefix, which is what lets two copies of `cube.xml` coexist;
3. **accumulate** — `<frame pos quat>` folds into its direct children's
   `pos`/`quat` and disappears, so no wrapper bodies appear and `nbody`
   matches the asset.

## ⚠⚠ The traps, each of which this tree has already paid for once

* **A dropped section is silent.** Handled by `merge_mjcf`'s completeness plus
  `resolve_includes`' post-check; see those.
* **A missed name reference is silent.** `full_parser` resolves some
  references to `-1` WITHOUT raising — an actuator with an unresolved `joint=`
  applies ZERO FORCE and `-1` is a legal sentinel, so nothing downstream can
  tell it from a tendon transmission. The whitelist below is therefore
  followed by `_check_resolved`, which fails on a reference naming nothing.
* **`<option>`/`<compiler>` are singletons, last-wins.** A prop carrying
  `<option timestep=…>` would silently retune the robot's physics, so they are
  taken from the BASE instance and an attached one raises.
* **Keyframes multiply, and each moves EVERY instance.** N instances of a
  keyframed asset give N keyframes each sized to the whole model, so loading
  `c1_home` also moves c2. Verified on MuJoCo 3.10.0. Keyframes come from the
  base only; an attached one raises.
* **A FREE JOINT CAN ONLY BE ATTACHED AT TOP LEVEL** — a hard MuJoCo error
  otherwise. A free-jointed prop must go in a world-level `<frame>`.

## ⚠ Asset paths are re-based, not copied

A sub-model's `<mesh file="assets/x.stl">` is relative to ITS directory, and
after splicing the host's directory is what `parse_xml_full` resolves against.
Every `file=` in a spliced `<asset>` is therefore rewritten to the sub-model's
directory first. Getting this wrong gives a mesh geom with no mesh — invisible
and uncollidable, exactly the failure the nameless-`<mesh>` bug produced.
"""

from .xml_parser import (
    _trim, _extract_attr, _extract_section_inner, _extract_section,
    _strip_xml_comments, _strip_wrapper, _normalize_freejoint,
    _parse_float, _find_tag, resolve_includes, merge_mjcf,
)


def _ref_attrs() -> List[String]:
    """Attributes whose VALUE is a name that must gain the prefix.

    ⚠⚠ A WHITELIST, AND A MISS IS SILENT. An unprefixed reference either
    dangles (some paths raise, several do not — see the module header) or, far
    worse, resolves against ANOTHER INSTANCE's element of the same name, which
    is exactly what prefixing exists to prevent and produces a model that
    loads and is wrong.

    ⚠ `file` IS DELIBERATELY ABSENT. It is a PATH, not a name; prefixing it
    would break every mesh and texture. See `_rebase_files`.
    """
    var a = List[String]()
    for n in [
        "name", "class", "childclass",
        "joint", "joint1", "joint2", "jointinparent",
        "body", "body1", "body2",
        "site", "site1", "site2", "refsite", "sidesite",
        "cranksite", "slidersite",
        "geom", "geom1", "geom2",
        "mesh", "material", "texture", "hfield", "skin",
        "tendon", "tendon1", "tendon2",
        "actuator", "target", "objname", "sensor",
    ]:
        a.append(String(n))
    return a^


def _prefix_all(xml: String, prefix: String) -> String:
    """Prefix every name and name reference. Text-level, attribute by attribute.

    ⚠ THE VALUE IS REWRITTEN IN PLACE, so this walks occurrences of
    `attr="..."` rather than re-serialising tags — a re-serialiser would have
    to understand every element, and any attribute it did not know about would
    be dropped. Rewriting in place cannot lose anything it does not recognise.

    ⚠ AN EMPTY VALUE IS LEFT ALONE. `class=""` means "no class", and
    `class="cube1_"` would name a class that does not exist.
    """
    var out = xml
    for attr in _ref_attrs():
        var needle = attr + '="'
        var res = String("")
        var scan = 0
        while True:
            var at = out.find(needle, scan)
            if at == -1:
                res += String(out[byte=scan : out.byte_length()])
                break
            # ⚠ THE CHARACTER BEFORE MUST BE A SEPARATOR, or `body=` also
            # matches the tail of `refbody=` and `joint=` the tail of
            # `sidejoint=`. Substring matching on attribute names is exactly
            # how a prefixer corrupts a model it was meant to namespace.
            var ok = at == 0
            if not ok:
                var prev = String(out[byte = at - 1 : at])
                ok = prev == " " or prev == "\t" or prev == "\n" or prev == "\r"
            var vs = at + needle.byte_length()
            var ve = out.find('"', vs)
            if not ok or ve == -1:
                res += String(out[byte=scan : vs])
                scan = vs
                continue
            var val = String(out[byte=vs:ve])
            res += String(out[byte=scan:vs])
            if val.byte_length() > 0:
                res += prefix
            res += val + '"'
            scan = ve + 1
        out = res^
    return out^


def _rebase_files(xml: String, dir: String) -> String:
    """Make every `file=` in a spliced fragment resolve against `dir`.

    ⚠ WITHOUT THIS A SPLICED MESH SILENTLY VANISHES. `parse_xml_full` gets ONE
    base directory — the HOST's — while a sub-model's paths are relative to
    its own. A mesh geom whose asset fails to resolve draws nothing and
    collides with nothing, and raises neither.
    """
    if dir.byte_length() == 0:
        return xml
    var out = String("")
    var scan = 0
    while True:
        var at = xml.find('file="', scan)
        if at == -1:
            out += String(xml[byte=scan : xml.byte_length()])
            break
        var vs = at + 6
        var ve = xml.find('"', vs)
        if ve == -1:
            out += String(xml[byte=scan : xml.byte_length()])
            break
        var val = String(xml[byte=vs:ve])
        out += String(xml[byte=scan:vs])
        # Absolute paths escape, as everywhere else in this parser.
        if val.byte_length() > 0 and not val.startswith("/"):
            out += dir + "/" + val
        else:
            out += val
        out += '"'
        scan = ve + 1
    return out^


# ═══════════════════════════════════════════════════════════════════════════
# quaternion helpers — local, to keep `parser` free of a `math3d` edge
# ═══════════════════════════════════════════════════════════════════════════


def _qmul(
    aw: Float64, ax: Float64, ay: Float64, az: Float64,
    bw: Float64, bx: Float64, by: Float64, bz: Float64,
) -> List[Float64]:
    var r = List[Float64]()
    r.append(aw * bw - ax * bx - ay * by - az * bz)
    r.append(aw * bx + ax * bw + ay * bz - az * by)
    r.append(aw * by - ax * bz + ay * bw + az * bx)
    r.append(aw * bz + ax * by - ay * bx + az * bw)
    return r^


def _qrot(
    qw: Float64, qx: Float64, qy: Float64, qz: Float64,
    vx: Float64, vy: Float64, vz: Float64,
) -> List[Float64]:
    var tx = 2.0 * (qy * vz - qz * vy)
    var ty = 2.0 * (qz * vx - qx * vz)
    var tz = 2.0 * (qx * vy - qy * vx)
    var r = List[Float64]()
    r.append(vx + qw * tx + qy * tz - qz * ty)
    r.append(vy + qw * ty + qz * tx - qx * tz)
    r.append(vz + qw * tz + qx * ty - qy * tx)
    return r^


def _vec3(s: String, d0: Float64, d1: Float64, d2: Float64) -> List[Float64]:
    var out = List[Float64]()
    out.append(d0)
    out.append(d1)
    out.append(d2)
    if s.byte_length() == 0:
        return out^
    var parts = List[String]()
    var cur = String("")
    for i in range(s.byte_length()):
        var ch = String(s[byte = i : i + 1])
        if ch == " " or ch == "\t" or ch == "\n":
            if cur.byte_length() > 0:
                parts.append(cur)
                cur = String("")
        else:
            cur += ch
    if cur.byte_length() > 0:
        parts.append(cur)
    for i in range(3):
        if i < len(parts):
            out[i] = _parse_float(parts[i])
    return out^


def _quat4(s: String) -> List[Float64]:
    var out = List[Float64]()
    out.append(1.0)
    out.append(0.0)
    out.append(0.0)
    out.append(0.0)
    if s.byte_length() == 0:
        return out^
    var parts = List[String]()
    var cur = String("")
    for i in range(s.byte_length()):
        var ch = String(s[byte = i : i + 1])
        if ch == " " or ch == "\t" or ch == "\n":
            if cur.byte_length() > 0:
                parts.append(cur)
                cur = String("")
        else:
            cur += ch
    if cur.byte_length() > 0:
        parts.append(cur)
    for i in range(4):
        if i < len(parts):
            out[i] = _parse_float(parts[i])
    return out^


def _f(v: Float64) -> String:
    """Enough digits that a round trip through text is exact for a pose.

    ⚠ SIX DECIMALS IS NOT ENOUGH and this is not a style choice: a quaternion
    written at MuJoCo's own print precision drifts the attached body by ~1e-7
    per level of nesting, which a record-for-record gate against `mjModel`
    sees immediately.
    """
    return String(v)


# ═══════════════════════════════════════════════════════════════════════════
# <frame> — accumulate into the direct children, then vanish
# ═══════════════════════════════════════════════════════════════════════════


def _child_tag_end(xml: String, open_at: String, start: Int) -> Int:
    """Byte just past the matching close of the element starting at `start`.

    Handles self-closing tags and nesting of the SAME tag name, which
    `<body>` inside `<body>` needs.
    """
    var tag_end = xml.find(">", start)
    if tag_end == -1:
        return xml.byte_length()
    if tag_end >= 1 and String(xml[byte = tag_end - 1 : tag_end]) == "/":
        return tag_end + 1
    var depth = 1
    var pos = tag_end + 1
    var open_needle = "<" + open_at
    var close_needle = "</" + open_at
    while depth > 0 and pos < xml.byte_length():
        var no = _find_tag(xml, open_needle, pos)
        var nc = xml.find(close_needle, pos)
        if nc == -1:
            return xml.byte_length()
        if no != -1 and no < nc:
            var oe = xml.find(">", no)
            if oe == -1:
                return xml.byte_length()
            if not (oe >= 1 and String(xml[byte = oe - 1 : oe]) == "/"):
                depth += 1
            pos = oe + 1
        else:
            depth -= 1
            var ce = xml.find(">", nc)
            pos = ce + 1 if ce != -1 else xml.byte_length()
    return pos


def _apply_frame(inner: String, fpos: List[Float64], fq: List[Float64]) -> String:
    """Fold a frame's transform into each DIRECT CHILD element of `inner`.

    MuJoCo: "a pure coordinate transformation that can wrap any group of
    elements in the kinematic tree… after compilation, frame elements
    disappear and their transformation is accumulated in their direct
    children" (XMLreference, `frame`).

    ⚠ DIRECT CHILDREN ONLY. A grandchild's pose is already relative to its
    parent, so touching it would apply the frame twice — the classic
    double-transform, which looks like a scaling error rather than a
    duplicated rotation.
    """
    var out = String("")
    var scan = 0
    var n = inner.byte_length()
    while scan < n:
        var lt = inner.find("<", scan)
        if lt == -1:
            out += String(inner[byte=scan:n])
            break
        out += String(inner[byte=scan:lt])
        # element name
        var ne = lt + 1
        while ne < n:
            var c = String(inner[byte = ne : ne + 1])
            if c == " " or c == ">" or c == "/" or c == "\n" or c == "\t":
                break
            ne += 1
        var ename = String(inner[byte = lt + 1 : ne])
        var elem_end = _child_tag_end(inner, ename, lt)
        var elem = String(inner[byte=lt:elem_end])
        var tag_end = elem.find(">")
        var tag = String(elem[byte = 0 : tag_end + 1]) if tag_end != -1 else elem

        var cp = _vec3(_trim(_extract_attr(tag, "pos")), 0.0, 0.0, 0.0)
        var cq = _quat4(_trim(_extract_attr(tag, "quat")))
        var rp = _qrot(fq[0], fq[1], fq[2], fq[3], cp[0], cp[1], cp[2])
        var np0 = fpos[0] + rp[0]
        var np1 = fpos[1] + rp[1]
        var np2 = fpos[2] + rp[2]
        var nq = _qmul(fq[0], fq[1], fq[2], fq[3], cq[0], cq[1], cq[2], cq[3])

        var new_tag = _set_attr(tag, "pos",
                                _f(np0) + " " + _f(np1) + " " + _f(np2))
        new_tag = _set_attr(new_tag, "quat",
                            _f(nq[0]) + " " + _f(nq[1]) + " " + _f(nq[2])
                            + " " + _f(nq[3]))
        out += new_tag
        if tag_end != -1:
            out += String(elem[byte = tag_end + 1 : elem.byte_length()])
        scan = elem_end
    return out^


def _set_attr(tag: String, attr: String, value: String) -> String:
    """Replace or insert `attr="value"` in an opening tag."""
    var needle = attr + '="'
    var at = -1
    var scan = 0
    while True:
        var f = tag.find(needle, scan)
        if f == -1:
            break
        var ok = f == 0
        if not ok:
            var prev = String(tag[byte = f - 1 : f])
            ok = prev == " " or prev == "\t" or prev == "\n"
        if ok:
            at = f
            break
        scan = f + 1
    if at >= 0:
        var vs = at + needle.byte_length()
        var ve = tag.find('"', vs)
        if ve != -1:
            return (
                String(tag[byte=0:vs]) + value
                + String(tag[byte = ve : tag.byte_length()])
            )
    # insert just before the closing `>` / `/>`
    var end = tag.byte_length() - 1
    while end > 0 and String(tag[byte = end : end + 1]) != ">":
        end -= 1
    var ins = end
    if ins >= 1 and String(tag[byte = ins - 1 : ins]) == "/":
        ins -= 1
    return (
        String(tag[byte=0:ins]) + " " + attr + '="' + value + '"'
        + String(tag[byte = ins : tag.byte_length()])
    )


def expand_frames(xml: String) -> String:
    """Replace every `<frame>` with its transformed children. Innermost first.

    ⚠ INNERMOST FIRST, because a nested frame's transform must be folded into
    its own children BEFORE the outer one folds into it — otherwise the outer
    frame lands on a `<frame>` tag, which carries no pose of its own to
    accumulate into, and the inner transform is lost entirely.
    """
    var out = xml
    var guard = 0
    while True:
        guard += 1
        if guard > 64:
            return out^
        # innermost = the LAST `<frame` whose content has no further `<frame`
        var at = -1
        var scan = 0
        while True:
            var f = _find_tag(out, "<frame", scan)
            if f == -1:
                break
            var e = _child_tag_end(out, String("frame"), f)
            var body = String(out[byte=f:e])
            if _find_tag(body, "<frame", 1) == -1:
                at = f
            scan = f + 1
            if at >= 0 and _find_tag(out, "<frame", scan) == -1:
                break
        if at == -1:
            return out^
        var end = _child_tag_end(out, String("frame"), at)
        var elem = String(out[byte=at:end])
        var tag_end = elem.find(">")
        var tag = String(elem[byte = 0 : tag_end + 1]) if tag_end != -1 else elem
        var fpos = _vec3(_trim(_extract_attr(tag, "pos")), 0.0, 0.0, 0.0)
        var fq = _quat4(_trim(_extract_attr(tag, "quat")))
        var inner = String("")
        if tag_end != -1 and not (
            tag_end >= 1 and String(elem[byte = tag_end - 1 : tag_end]) == "/"
        ):
            var close = elem.rfind("</frame")
            if close != -1:
                inner = String(elem[byte = tag_end + 1 : close])
        out = (
            String(out[byte=0:at]) + _apply_frame(inner, fpos, fq)
            + String(out[byte = end : out.byte_length()])
        )


# ═══════════════════════════════════════════════════════════════════════════
# <attach> — splice a sub-model in, prefixed
# ═══════════════════════════════════════════════════════════════════════════


def _compiler_attr(xml: String, attr: String) -> String:
    """One attribute of the model's `<compiler>` tag, or "" if absent."""
    var t = _find_tag(xml, "<compiler", 0)
    if t == -1:
        return String("")
    var e = xml.find(">", t)
    if e == -1:
        return String("")
    return _trim(_extract_attr(String(xml[byte = t : e + 1]), attr))


def _read(path: String) raises -> String:
    var f = open(path, "r")
    var t = f.read()
    f.close()
    return t^


def _dirname(p: String) -> String:
    var cut = p.rfind("/")
    return String(p[byte=0:cut]) if cut > 0 else String("")


def _named_body(worldbody: String, name: String) raises -> String:
    """The `<body name=...>` element, whole. `<attach body=...>`'s target."""
    var scan = 0
    while True:
        var b = _find_tag(worldbody, "<body", scan)
        if b == -1:
            raise Error(
                "physics3d: <attach body='" + name + "'> names no body in the"
                " sub-model."
            )
        var e = _child_tag_end(worldbody, String("body"), b)
        var tag_end = worldbody.find(">", b)
        var tag = String(worldbody[byte = b : tag_end + 1])
        if _trim(_extract_attr(tag, "name")) == name:
            return String(worldbody[byte=b:e])
        scan = b + 1


def expand_attach(xml: String, base_dir: String, depth: Int = 0) raises -> String:
    """Splice every `<attach>`; returns flat MJCF the existing parser reads.

    ⚠ THE SUB-MODEL'S NON-WORLDBODY SECTIONS COME TOO, prefixed and re-based:
    its `<asset>`, `<default>`, `<actuator>`, `<tendon>`, `<equality>`,
    `<contact>` and `<sensor>`. Splicing only the bodies is the obvious
    mistake and gives a model that LOADS — the geoms are all there — with no
    actuators, no materials and no default classes, so the prop is grey and
    limp rather than absent. MuJoCo's own flattening carries all of them; see
    `mj_saveLastXML` output in the fixtures.
    """
    if _find_tag(xml, "<attach", 0) == -1:
        return xml
    if depth > 6:
        raise Error("physics3d: <attach> nested more than 6 deep — a cycle?")

    # ── the asset table `attach model=` resolves against ──────────────────
    var asset_sec = _extract_section_inner(xml, "asset")
    var mnames = List[String]()
    var mfiles = List[String]()
    var scan = 0
    while True:
        var t = _find_tag(asset_sec, "<model", scan)
        if t == -1:
            break
        var te = asset_sec.find(">", t)
        if te == -1:
            break
        var tag = String(asset_sec[byte = t : te + 1])
        mnames.append(_trim(_extract_attr(tag, "name")))
        mfiles.append(_trim(_extract_attr(tag, "file")))
        scan = te + 1

    var out = xml
    var extra = List[String]()
    var guard = 0
    while True:
        guard += 1
        if guard > 256:
            raise Error("physics3d: too many <attach> elements (>256).")
        var at = _find_tag(out, "<attach", 0)
        if at == -1:
            break
        var te = out.find(">", at)
        if te == -1:
            raise Error("physics3d: unterminated <attach> tag.")
        var end = _child_tag_end(out, String("attach"), at)
        var tag = String(out[byte = at : te + 1])
        var mdl = _trim(_extract_attr(tag, "model"))
        var body = _trim(_extract_attr(tag, "body"))
        var prefix = _trim(_extract_attr(tag, "prefix"))

        var file = String("")
        for i in range(len(mnames)):
            if mnames[i] == mdl:
                file = mfiles[i]
        if file.byte_length() == 0:
            raise Error(
                "physics3d: <attach model='" + mdl + "'> — no <asset><model"
                " name='" + mdl + "' file=...> declares it."
            )
        var path = file
        if not file.startswith("/") and base_dir.byte_length() > 0:
            path = base_dir + "/" + file
        var sub_dir = _dirname(path)

        # ⚠ THE SUB-MODEL IS EXPANDED FIRST — it may itself attach or include.
        var sub = _normalize_freejoint(
            _strip_wrapper(_strip_xml_comments(
                resolve_includes(_read(path), sub_dir)
            ))
        )
        sub = expand_attach(sub, sub_dir, depth + 1)
        sub = expand_frames(sub)

        # ⚠⚠ `<option>` / `<compiler>` / `<keyframe>` FROM THE BASE ONLY.
        # A prop carrying `<option timestep=...>` would silently retune the
        # ROBOT's physics, and N instances of a keyframed asset give N
        # keyframes each sized to the whole model — loading `c1_home` moves
        # c2. Both verified on MuJoCo 3.10.0. Raising beats a diagnostic
        # nobody reads, because the wrong outcome is a model that runs.
        for sec in ["option", "keyframe"]:
            if _find_tag(sub, "<" + String(sec), 0) != -1:
                raise Error(
                    "physics3d: attached model '" + mdl + "' declares <"
                    + String(sec) + ">. Singletons and keyframes come from the"
                    " BASE model only — an attached <option> retunes the whole"
                    " scene's physics, and an attached keyframe is sized to"
                    " the whole model and moves every other instance."
                )

        # ⚠⚠ `<compiler>` IS NOT DROPPED, IT IS CHECKED. Nearly every model
        # declares one, and MuJoCo keeps a merged `<compiler>` in its own
        # flattened output — but the attribute that matters is `angle`, and it
        # governs how the SUB-MODEL's OWN numbers are read. Splicing a
        # `degree` model's text into a `radian` host silently reinterprets
        # every joint range and euler in it: a 90 becomes 90 radians. MuJoCo
        # cannot hit this because it compiles each model separately and
        # attaches the RESULT; a text splice can, so it has to refuse.
        var sub_angle = _compiler_attr(sub, "angle")
        var host_angle = _compiler_attr(xml, "angle")
        if sub_angle.byte_length() > 0 and host_angle.byte_length() > 0 \
                and sub_angle != host_angle:
            raise Error(
                "physics3d: attached model '" + mdl + "' uses <compiler"
                " angle='" + sub_angle + "'> while the scene uses '"
                + host_angle + "'. A text splice would read its angles in the"
                " scene's units. Convert the asset, or set both to the same."
            )
        # ⚠ `meshdir` / `assetdir` FOLD INTO THE REBASE, and must: after the
        # splice the HOST's compiler applies, so a sub-model whose paths are
        # `meshdir`-relative would resolve against the wrong directory and its
        # meshes would silently vanish.
        var sub_assets_dir = sub_dir
        var md = _compiler_attr(sub, "meshdir")
        if md.byte_length() == 0:
            md = _compiler_attr(sub, "assetdir")
        if md.byte_length() > 0 and not md.startswith("/"):
            sub_assets_dir = sub_dir + "/" + md if sub_dir.byte_length() > 0 else md
        elif md.byte_length() > 0:
            sub_assets_dir = md

        var sub_prefixed = _rebase_files(
            _prefix_all(sub, prefix), sub_assets_dir
        )
        var sub_world = _extract_section_inner(sub_prefixed, "worldbody")
        var spliced = sub_world
        if body.byte_length() > 0:
            spliced = _named_body(sub_world, prefix + body)

        # Everything that is not the worldbody rides along.
        for sec in ["asset", "default", "actuator", "tendon", "equality",
                    "contact", "sensor"]:
            var inner = _extract_section_inner(sub_prefixed, String(sec))
            if _trim(inner).byte_length() > 0:
                extra.append(
                    "<" + String(sec) + ">" + inner + "</" + String(sec) + ">"
                )

        out = (
            String(out[byte=0:at]) + spliced
            + String(out[byte = end : out.byte_length()])
        )

    if len(extra) == 0:
        return out^
    # ⚠ MERGED, NOT CONCATENATED. `merge_mjcf` is what folds two `<asset>`s
    # into one and keeps the singleton rules; pasting a second `<asset>` block
    # into the host would be a second section the parser reads once.
    var host = "<mujoco>" + out + "</mujoco>"
    var add = String("<mujoco>")
    for e in extra:
        add += e
    add += "</mujoco>"
    return merge_mjcf(host, add)


def expand_mjcf(xml: String, base_dir: String) raises -> String:
    """`<include>` -> `<attach>` -> `<frame>`, in that order. The entry point.

    ⚠ THE ORDER IS FORCED. An include may bring an `<attach>`; an attach
    splices bodies that a `<frame>` must then transform. Running frames first
    would fold a transform into an `<attach/>` tag, which has no pose to
    accumulate into and vanishes with it.
    """
    var flat = expand_frames(
        expand_attach(resolve_includes(xml, base_dir), base_dir)
    )
    # ⚠ ONLY WHEN SOMETHING WAS ACTUALLY COMPOSED. A plain single-file model
    # goes through untouched, and validating it here would turn this into a
    # second opinion on `full_parser`'s own name resolution — a different job,
    # with its own false positives on the attributes MuJoCo lets dangle.
    if xml.find("<attach") != -1 or xml.find("<include") != -1:
        check_references(flat)
    return flat^


# ═══════════════════════════════════════════════════════════════════════════
# post-expand validation — the other half of the prefixing contract
# ═══════════════════════════════════════════════════════════════════════════


def _attr_values(xml: String, attr: String) -> List[String]:
    """Every value of `attr` in the document, in order.

    Same separator rule as `_prefix_all`: the character before the attribute
    must be whitespace or the start of the text, or `body=` also matches the
    tail of `refbody=`.
    """
    var out = List[String]()
    var needle = attr + '="'
    var scan = 0
    while True:
        var at = xml.find(needle, scan)
        if at == -1:
            return out^
        var ok = at == 0
        if not ok:
            var prev = String(xml[byte = at - 1 : at])
            ok = prev == " " or prev == "\t" or prev == "\n" or prev == "\r"
        var vs = at + needle.byte_length()
        var ve = xml.find('"', vs)
        if ve == -1:
            return out^
        if ok:
            out.append(String(xml[byte=vs:ve]))
        scan = ve + 1


def check_references(xml: String) raises:
    """Every name REFERENCE in `xml` must name something declared in it.

    ⚠⚠ THIS EXISTS BECAUSE `full_parser`'s BEHAVIOUR IS MIXED, and the silent
    paths are exactly the ones a prefixer breaks. Measured (§3.2 of the plan):

        <contact><pair> geom1/geom2   RAISES, naming the geom
        <equality> joint names        RAISES
        actuator joint=               **SILENT** — joint_id = -1, and
                                      `_fill_actuator_transmission` has no
                                      else, so trn_n = 0: an actuator that
                                      applies ZERO FORCE, with no diagnostic
        <equality> body1/body2        **SILENT** -1 into the record
        <contact><exclude>            **SILENTLY SKIPPED**

    and the actuator case is unfixable downstream, because `-1` is a LEGAL
    sentinel there ("no joint transmission"), so nothing after the parser can
    tell an unresolved name from a tendon transmission. A prefixer that misses
    one attribute therefore produces a limp robot and no error.

    ⇒ validate HERE, where the rewrite happened and the name is still in hand.

    ⚠ THE DECLARED SET INCLUDES `<default class="X">`, which declares with
    `class=` rather than `name=` — the one element in MJCF that does. Omitting
    it would make every `class=` reference look dangling.

    ⚠ `target=` ON A CAMERA IS EXEMPT. `full_parser` resolves it to -1 with a
    documented, deliberate degradation, so a model that names a missing
    target is one MuJoCo also accepts.
    """
    # ── everything the document declares ─────────────────────────────────
    var declared = List[String]()
    for v in _attr_values(xml, String("name")):
        declared.append(v)
    # `<default class="X">` DECLARES; `class=` elsewhere REFERENCES.
    var scan = 0
    while True:
        var d = _find_tag(xml, "<default", scan)
        if d == -1:
            break
        var e = xml.find(">", d)
        if e == -1:
            break
        var c = _trim(_extract_attr(String(xml[byte = d : e + 1]), "class"))
        if c.byte_length() > 0:
            declared.append(c)
        scan = e + 1

    var refs = List[String]()
    for a in _ref_attrs():
        # `name` DECLARES, and `class` is handled above for the declaring
        # case; `target` is the documented exemption.
        if a == "name" or a == "target":
            continue
        refs.append(a)

    var bad = List[String]()
    for attr in refs:
        for v in _attr_values(xml, attr):
            if v.byte_length() == 0:
                continue
            var found = False
            for d in declared:
                if d == v:
                    found = True
            if not found:
                bad.append(attr + "='" + v + "'")

    if len(bad) > 0:
        var msg = String(
            "physics3d: expansion left "
        ) + String(len(bad)) + " DANGLING name reference(s). A prefixer that"
        msg += " misses an attribute produces a model that LOADS and is wrong"
        msg += " — an actuator with an unresolved joint= applies zero force"
        msg += " and raises nothing. Offenders:"
        for i in range(len(bad)):
            if i >= 8:
                msg += " ... and " + String(len(bad) - 8) + " more"
                break
            msg += " " + bad[i]
        raise Error(msg)
