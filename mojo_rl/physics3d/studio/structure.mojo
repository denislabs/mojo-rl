"""Structural edits — V2.1. Delete a body, and everything that named it.

## ⚠⚠ THE EDIT IS ON THE **TEXT**, AND THAT IS THE DESIGN, NOT A SHORTCUT

The obvious implementation is to mutate the `FlatModelDef`: drop
`bodies[k]`, drop the joints and geoms whose `body_id` is `k`, and renumber.
It is also the worst thing this codebase could do. `FlatModelDef` stores
INDICES, so deleting body `k` invalidates every index at or above it in

    BodyData.parent   JointData.body_id   GeomData.body_id   SiteData.body_id
    EqualityData.body_a/body_b   ExcludeData.body1/body2
    PairData.geom1/geom2   ActuatorData.joint_id/tendon_id
    TendonData.wrap_objs/joint_ids   body_names / joint_names / geom_names /
    site_names / actuator_names   qpos0's address layout

— fifteen places, each of which fails SILENTLY and none of which raises. This
tree's whole history is one index quietly meaning something else
(`meta[26]`, the `range(-1)` invweight, the `xyaxes` conjugate that crossed to
one of two parsers). A structural editor built on index surgery would be a
machine for producing that bug.

So the slow path is what the plan specified (§4): **regenerate the MJCF, and
re-parse it.** The parser assigns every index, exactly once, the way it always
has. The tool gains no second model path (§10 risk 2), and — the part that
makes it gateable — **MuJoCo can load the very same text**, so an external
oracle judges the result of every edit.

## ⚠⚠ A DELETION IS A GRAPH OPERATION, NOT A SPAN REMOVAL

Cutting `<body name="thigh">…</body>` out of the text leaves behind the
actuator that drives its joint, the `<contact><exclude>` that names it, and
the tendon routed through its sites. Every one of those is now a DANGLING
reference — which MuJoCo refuses and, for `joint=` on an actuator, this parser
used to accept as a zero-force actuator with no diagnostic. So the delete
must prune, and

    ⚠ IT MUST SAY WHAT IT PRUNED.

Silently removing an actuator because the user deleted a link is how an editor
loses work. Every prune lands in `EditResult.notes`, which the studio shows.

⚠ AND THE KEYFRAMES GO. Deleting a joint changes `nq`, and every
`<key qpos="…">` is then the wrong length — MuJoCo refuses the model outright
("keyframe 'k': invalid qpos size, expected 1, got 2"). There is no honest way
to resize one, so a joint-removing edit DROPS the keyframe section and says so.

⚠ A NAME IS ONLY GONE IF NOTHING ELSE DECLARES IT. Names are scoped per
element kind, so a body and a geom may share one (measured). Deleting body
`x` while geom `x` remains must not prune the pair that names `geom1="x"` —
so the gone-set is filtered against what the EDITED document still declares,
not against what the deleted span contained.
"""

from ..parser.expander import (
    element_end, _find_tag, _ref_attrs, _attr_values, dangling_references,
    name_table, tag_name_at, decl_kind, attr_kind, NameTable,
)
from ..parser.xml_parser import _extract_attr, _trim


struct EditResult(Movable):
    """The new document, plus what had to go with it."""

    var xml: String
    var ok: Bool
    """False when the target was not found — the text comes back unchanged."""
    var notes: List[String]
    """Everything removed BEYOND what was asked for, in the user's words.

    ⚠ NOT A LOG LEVEL. These are the edits the user did not make and must be
    told about; the studio surfaces them beside the Problems list.
    """

    def __init__(out self, var xml: String, ok: Bool, var notes: List[String]):
        self.xml = xml^
        self.ok = ok
        self.notes = notes^


# =============================================================================
# Finding elements
# =============================================================================


def _in_comment(xml: String, at: Int) -> Bool:
    """Is byte `at` inside an XML comment?

    ⚠⚠ THE GATE'S FIRST RUN FOUND THIS, and it deleted almost the whole
    model. The zoo fixture's header comment contained the words
    `<body name="arm">`; `find_named` matched INSIDE the comment, and
    `element_end` then ran from there to the first `</body>` several elements
    later — taking the `<compiler>`, the floor and the trunk with it. nbody
    went 5 -> 1 and nothing raised.

    A commented-out body is completely ordinary in a hand-written MJCF, so
    this is not a fixture artefact. `expand_mjcf` strips comments on the
    include path, but `structure` must not depend on having been handed
    stripped text.
    """
    var open_at = String(xml[byte=0:at]).rfind("<!--")
    if open_at == -1:
        return False
    var close_at = xml.find("-->", open_at)
    return close_at == -1 or close_at > at


def find_named(xml: String, tag: String, name: String) -> Int:
    """Byte offset of `<tag name="name"`, or -1.

    ⚠ THE NAME IS THE KEY, NOT THE INDEX, and that is the whole reason
    `FlatModelDef` was taught to carry names. An index-keyed edit cannot
    survive the edit before it.
    """
    var scan = 0
    while True:
        var at = _find_tag(xml, "<" + tag, scan)
        if at == -1:
            return -1
        var e = xml.find(">", at)
        if e == -1:
            return -1
        var got = _trim(_extract_attr(String(xml[byte = at : e + 1]), "name"))
        if got == name and not _in_comment(xml, at):
            return at
        scan = e + 1


struct Gone(Movable):
    """The names a deletion removed, EACH WITH ITS NAMESPACE.

    ⚠⚠ THE KIND IS NOT DECORATION. `half_cheetah.xml` names a body, a joint,
    a geom and a motor all `bthigh` — legal, because MuJoCo scopes names per
    element kind. Deleting the body removes three of the four; asking "is
    `bthigh` still declared?" gets a YES from the orphaned motor's own
    `name=`, so nothing is pruned and the document is left with an actuator
    driving a joint that no longer exists. The kind is what makes the
    question answerable.
    """

    var names: List[String]
    var kinds: List[String]

    def __init__(out self):
        self.names = List[String]()
        self.kinds = List[String]()

    def add(mut self, name: String, kind: String):
        if not self.has(name, kind):
            self.names.append(name)
            self.kinds.append(kind)

    def has(self, name: String, kind: String) -> Bool:
        """⚠ "" MATCHES ANYTHING, ON EITHER SIDE — it is the "namespace not
        known" answer, and it has to stay permissive in both directions or an
        unmapped attribute silently stops pruning."""
        for i in range(len(self.names)):
            if self.names[i] != name:
                continue
            if (kind.byte_length() == 0 or self.kinds[i].byte_length() == 0
                    or self.kinds[i] == kind):
                return True
        return False

    def count(self) -> Int:
        return len(self.names)


def _contains(names: List[String], n: String) -> Bool:
    for x in names:
        if x == n:
            return True
    return False


# =============================================================================
# Pruning what a deletion orphaned
# =============================================================================


def _prune_attrs() -> List[String]:
    """Reference attributes a deletion can orphan.

    `_ref_attrs()` minus the three that do NOT name an element a structural
    edit removes: `name` DECLARES, and `class`/`childclass` name a
    `<default>` block, which nothing here deletes.
    """
    var out = List[String]()
    for a in _ref_attrs():
        if a == "name" or a == "class" or a == "childclass":
            continue
        out.append(a)
    return out^


def _first_gone_ref(span: String, gone: Gone) -> String:
    """`joint='hip'` for the first reference in `span` naming a gone element.

    ⚠ THE ATTRIBUTE'S NAMESPACE DECIDES. `joint="bthigh"` asks about joints,
    not about whatever else happens to be called `bthigh`.
    """
    for attr in _prune_attrs():
        var kind = attr_kind(attr)
        for v in _attr_values(span, attr):
            if gone.has(v, kind):
                return attr + "='" + v + "'"
    return String("")


def _ref_attr(pair: String) -> String:
    """`joint` out of `joint='hip'`.

    ⚠ THE `=` IS NOT PART OF THE NAME. It was, for one run: `attr_kind`
    then answered "" for every attribute (an unmapped attribute means "any
    kind"), the cascade recorded the orphan under no namespace, and
    `_first_gone_ref` — which asks by namespace — never matched it. The
    tendon stayed dangling and every other arm was green.
    """
    var a = pair.find("=")
    return String(pair[byte=0:a]) if a > 0 else String("")


def _ref_value(pair: String) -> String:
    """`hip` out of `joint='hip'` — `dangling_references`' report format."""
    var a = pair.find("'")
    if a == -1:
        return String("")
    var b = pair.rfind("'")
    if b <= a:
        return String("")
    return String(pair[byte = a + 1 : b])


def _prunable_sections() -> List[String]:
    """Sections whose DIRECT CHILDREN reference elements by name.

    ⚠ THE WHOLE CHILD GOES, NOT THE ATTRIBUTE. A `<spatial>` tendon routed
    through a deleted site would otherwise come back one waypoint shorter —
    a tendon that loads, takes a different path, and reports nothing. Same for
    an `<equality><connect>` that lost one of its two bodies.
    """
    var out = List[String]()
    for s in ["actuator", "contact", "equality", "tendon", "sensor"]:
        out.append(String(s))
    return out^


def _prune_referencing(
    xml: String, gone: Gone, mut notes: List[String]
) -> String:
    """Remove every top-level element of a section that names a gone element.

    ⚠ SPANS ARE COLLECTED FIRST AND CUT IN DESCENDING ORDER. Splicing as we
    go would invalidate every offset already found — the classic way a
    multi-element edit removes the wrong bytes on its second hit.
    """
    var starts = List[Int]()
    var ends = List[Int]()

    for sec in _prunable_sections():
        var scan = 0
        while True:
            var s_at = _find_tag(xml, "<" + sec, scan)
            if s_at == -1:
                break
            var s_end = element_end(xml, sec, s_at)
            var open_end = xml.find(">", s_at)
            if open_end == -1:
                break
            scan = s_end
            # A self-closing `<contact/>` has no children.
            if open_end + 1 >= s_end:
                continue
            var pos = open_end + 1
            while pos < s_end:
                var lt = xml.find("<", pos)
                if lt == -1 or lt >= s_end:
                    break
                # ⚠ A COMMENT IS NOT A CHILD. Without this the walk treats
                # `<!-- <motor .../> -->` as an element and can splice the
                # comment's bytes out from under the next cut.
                if lt + 4 <= xml.byte_length() \
                        and String(xml[byte = lt : lt + 4]) == "<!--":
                    var ce = xml.find("-->", lt)
                    pos = (ce + 3) if ce != -1 else s_end
                    continue
                var child = tag_name_at(xml, lt)
                if child.byte_length() == 0:
                    pos = lt + 1
                    continue
                var c_end = element_end(xml, child, lt)
                if c_end > s_end:
                    break
                var span = String(xml[byte=lt:c_end])
                var hit = _first_gone_ref(span, gone)
                if hit.byte_length() > 0:
                    starts.append(lt)
                    ends.append(c_end)
                    notes.append(
                        "removed <" + child + "> in <" + sec + ">, which"
                        " referenced " + hit
                    )
                pos = c_end

    if len(starts) == 0:
        return xml

    # Descending by start, so each cut leaves earlier offsets valid.
    var order = List[Int]()
    for i in range(len(starts)):
        order.append(i)
    for i in range(len(order)):
        for k in range(i + 1, len(order)):
            if starts[order[k]] > starts[order[i]]:
                var tmp = order[i]
                order[i] = order[k]
                order[k] = tmp

    var out = xml
    for oi in order:
        out = String(out[byte = 0 : starts[oi]]) + String(
            out[byte = ends[oi] : out.byte_length()]
        )
    return out^


def _drop_sections(
    xml: String, tag: String, mut notes: List[String], why: String
) -> String:
    """Remove every `<tag>…</tag>` section, noting it once per occurrence."""
    var out = xml
    while True:
        var at = _find_tag(out, "<" + tag, 0)
        if at == -1:
            return out^
        var end = element_end(out, tag, at)
        out = String(out[byte=0:at]) + String(
            out[byte = end : out.byte_length()]
        )
        notes.append("dropped <" + tag + "> — " + why)


def _span_has_joint(span: String) -> Bool:
    return (
        _find_tag(span, "<joint", 0) != -1
        or _find_tag(span, "<freejoint", 0) != -1
    )


# =============================================================================
# The operations
# =============================================================================


def delete_element(xml: String, tag: String, name: String) -> EditResult:
    """Delete one named element and its subtree, then repair the document.

    Works for `body` (which takes its whole subtree), `geom`, `joint` and
    `site`. Returns the new text and everything that had to go with it.
    """
    var notes = List[String]()
    var at = find_named(xml, tag, name)
    if at == -1:
        notes.append("no <" + tag + " name=\"" + name + "\"> in this model")
        return EditResult(xml, False, notes^)

    var end = element_end(xml, tag, at)
    var span = String(xml[byte=at:end])
    var removes_dof = tag == "joint" or _span_has_joint(span)
    var inside = name_table(span)

    var out = String(xml[byte=0:at]) + String(
        xml[byte = end : xml.byte_length()]
    )

    # ⚠ FILTERED AGAINST THE EDITED DOCUMENT, PER NAMESPACE. Deleting body
    # `bthigh` while a MOTOR named `bthigh` survives must still count the
    # BODY (and its joint, and its geom) as gone.
    var survivors = name_table(out)
    var gone = Gone()
    for i in range(len(inside.names)):
        if not survivors.has(inside.names[i], inside.kinds[i]):
            gone.add(inside.names[i], inside.kinds[i])

    if gone.count() > 0:
        out = _prune_referencing(out, gone, notes)

    # ⚠⚠ THE PRUNE CASCADES, AND ONE PASS IS NOT ENOUGH. Deleting the body
    # that held a tendon's site removes the TENDON — and an actuator may drive
    # that tendon. `cable` was never in the deleted span, so the first pass
    # cannot know it is gone; only re-reading the edited document can. This
    # loop is what the zoo fixture in the gate exists to force: without it the
    # document comes back with `tendon='cable'` dangling and MuJoCo refuses a
    # file the editor called done.
    var guard = 0
    while guard < 8:
        guard += 1
        var dang = dangling_references(out)
        if len(dang) == 0:
            break
        var more = Gone()
        for d in dang:
            var v = _ref_value(d)
            var k = attr_kind(_ref_attr(d))
            if v.byte_length() > 0 and not gone.has(v, k):
                more.add(v, k)
                gone.add(v, k)
        if more.count() == 0:
            break
        out = _prune_referencing(out, more, notes)

    # ⚠ A KEYFRAME IS SIZED TO THE WHOLE MODEL. Removing a dof makes every
    # `qpos` row the wrong length and MuJoCo refuses the file; there is no
    # honest resize, so the section goes and the user is told.
    if removes_dof:
        out = _drop_sections(
            out, String("keyframe"), notes,
            String(
                "removing a joint changes nq, and a keyframe's qpos is sized"
                " to the whole model"
            ),
        )

    return EditResult(out^, True, notes^)


def delete_body(xml: String, name: String) -> EditResult:
    return delete_element(xml, String("body"), name)


def delete_geom(xml: String, name: String) -> EditResult:
    return delete_element(xml, String("geom"), name)


def delete_joint(xml: String, name: String) -> EditResult:
    return delete_element(xml, String("joint"), name)


def delete_site(xml: String, name: String) -> EditResult:
    return delete_element(xml, String("site"), name)


def leftover_dangling(xml: String) -> List[String]:
    """Any reference the prune missed — the studio's own check on its work.

    ⚠ THIS IS NOT THE VALIDATOR BEING CALLED TWICE. `validate_document` tells
    the USER what is wrong with a model; this tells the EDITOR whether the
    edit it just performed left the document loadable. An empty list here is
    the post-condition of every operation in this file.
    """
    return dangling_references(xml)
