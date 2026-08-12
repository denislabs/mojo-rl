"""`_DefaultsIndex` must answer exactly what the rescanning helpers answered.

MJCF `<default>` class resolution is the single most defect-prone corner of
this parser. It has produced, among others, defect `ab219882` — dog's actuator
gains resolving to the root default instead of their nested class, making every
one of its 38 actuators between 25x and 3000x too weak, and visible ONLY in a
driven rollout because `0.02 * 0 == 40 * 0`. Replacing the resolution with a
prebuilt index is therefore a change that must be proven equal, not argued
equal.

This is a pure DIFFERENTIAL test: for every class in every model it compares

    _class_attr            vs  _class_attr_indexed
    _class_parent          vs  _class_parent_indexed
    _class_attr_inherited  vs  _class_attr_inherited_indexed

and fails on the first disagreement. It never asserts what the right answer IS
— the dm_control parity gates do that. What it asserts is that the two
implementations cannot be told apart, including on the inputs where the old one
is loose: a `<default` inside a comment, a duplicated class name, a
self-closing block. Those quirks decide which block a class resolves to, so a
"cleaner" index that quietly disagreed would be a reparse of every model.

The real dm_control XMLs are read from `references/` rather than the merged
comptime strings in `mojo_rl/envs/dm_control/`: importing those would drag in
their `comptime parse_xml(...)` aliases and hand this test a multi-minute
build, which is precisely what the index exists to avoid.
"""

from mojo_rl.physics3d.parser.xml_parser import (
    _build_defaults_index,
    _class_attr,
    _class_attr_indexed,
    _class_attr_inherited,
    _class_attr_inherited_indexed,
    _class_parent,
    _class_parent_indexed,
    _extract_attr,
    _trim,
)
from std.testing import assert_equal, assert_true, TestSuite


# Exactly the element kinds and attributes the parser resolves through a class
# — `parse_xml_model_data`'s actuator loop and `parse_xml_render_data`'s geom
# loop — plus one tag and one attribute that no model sets. The misses earn
# their place: a lookup that MISSES leaves `_class_attr` by a different route
# than one that HITS, and the hitting route is the one that used to fail.
#
# ⚠ THE GRID IS DELIBERATELY SMALL. Every cell runs the OLD implementation
# once, and that is an O(len(xml)) walk allocating a String per `<default>`
# tag; on dog's 82 KB the full cross product cost 250 s of wall clock. Widen it
# only for a one-off investigation.
def _tags() -> List[String]:
    var out: List[String] = [
        "geom",
        "joint",
        "general",
        "motor",
        "nosuchtag",
    ]
    return out^


def _attrs() -> List[String]:
    var out: List[String] = [
        "type",
        "size",
        "fromto",
        "rgba",
        "material",
        "group",
        "gear",
        "ctrlrange",
        "gaintype",
        "biastype",
        "gainprm",
        "biasprm",
        "dyntype",
        "dynprm",
        "nosuchattr",
    ]
    return out^


def _class_names(xml: String) raises -> List[String]:
    """Every `class=` on a `<default>` tag, in text order, plus "" and a miss.

    Deliberately enumerated by a scan of its own rather than from the index —
    an index that dropped a block would otherwise never be asked about it.
    """
    var out = List[String]()
    out.append(String(""))  # the top-level block
    out.append(String("definitely_not_a_class"))  # the not-found path
    var n = xml.byte_length()
    var scan = 0
    while scan < n:
        var t = xml.find("<default", scan)
        if t == -1:
            break
        var te = xml.find(">", t)
        if te == -1:
            break
        out.append(_trim(_extract_attr(String(xml[byte = t : te + 1]), "class")))
        scan = te + 1
    return out^


def _compare(xml: String, label: String) raises -> Int:
    """Cross-check both implementations over every (class, tag, attr).

    Returns the number of comparisons made, so a model that silently supplied
    no classes cannot pass by doing nothing.
    """
    var idx = _build_defaults_index(xml)
    var names = _class_names(xml)
    var tags = _tags()
    var attrs = _attrs()
    var checks = 0

    for ci in range(len(names)):
        var cls = names[ci]

        var want_parent = _class_parent(xml, cls)
        var got_parent = _class_parent_indexed(idx, cls)
        assert_equal(
            got_parent,
            want_parent,
            label + ": _class_parent disagrees for class '" + cls + "'",
        )
        checks += 1

        for ti in range(len(tags)):
            var tag = tags[ti]
            for ai in range(len(attrs)):
                var attr = attrs[ai]

                var want_own = _class_attr(xml, cls, tag, attr)
                var got_own = _class_attr_indexed(xml, idx, idx.find(cls), tag, attr)
                assert_equal(
                    got_own,
                    want_own,
                    label
                    + ": _class_attr disagrees for ("
                    + cls
                    + ", "
                    + tag
                    + ", "
                    + attr
                    + ")",
                )

                var want_inh = _class_attr_inherited(xml, cls, tag, attr)
                var got_inh = _class_attr_inherited_indexed(xml, idx, cls, tag, attr)
                assert_equal(
                    got_inh,
                    want_inh,
                    label
                    + ": _class_attr_inherited disagrees for ("
                    + cls
                    + ", "
                    + tag
                    + ", "
                    + attr
                    + ")",
                )
                checks += 2
    return checks


def test_real_dm_control_models() raises:
    """The models the port actually compiles, dog first.

    dog is the one that matters: nested classes three deep, 38 actuators
    resolving gains through them, and the model whose misresolution shipped.
    """
    var paths: List[String] = [
        "references/dm_control-main/dm_control/suite/dog.xml",
        "references/dm_control-main/dm_control/suite/quadruped.xml",
        "references/dm_control-main/dm_control/suite/humanoid_CMU.xml",
        "references/dm_control-main/dm_control/suite/manipulator.xml",
        "references/dm_control-main/dm_control/suite/finger.xml",
        "references/dm_control-main/dm_control/suite/fish.xml",
        "references/dm_control-main/dm_control/suite/swimmer.xml",
        "references/dm_control-main/dm_control/suite/cheetah.xml",
        "references/dm_control-main/dm_control/suite/walker.xml",
        "references/dm_control-main/dm_control/suite/cartpole.xml",
    ]
    var total = 0
    for i in range(len(paths)):
        var path = paths[i]
        var xml: String
        with open(path, "r") as f:
            xml = f.read()
        assert_true(
            xml.byte_length() > 0, "empty or unreadable reference model: " + path
        )
        var n = _compare(xml, path)
        total += n
        print("  ", path, "->", n, "comparisons agree")
    # A collapse toward zero would mean the enumeration stopped, not that
    # the two implementations agreed.
    assert_true(total > 2000, "far too few comparisons ran: " + String(total))
    print("real models:", total, "comparisons, no disagreement")


def test_pathological_shapes() raises:
    """The inputs where the OLD code is loose, and the index must be too."""

    # A duplicated class name. `_class_attr` scanned from byte 0 and took the
    # FIRST match, so `dup` must resolve to the sphere, never the capsule.
    var duplicate = String(
        "<mujoco><default>"
        + '<default class="dup"><geom type="sphere"/></default>'
        + '<default class="dup"><geom type="capsule"/></default>'
        + "</default></mujoco>"
    )
    _ = _compare(duplicate, String("duplicate-class"))
    var d_idx = _build_defaults_index(duplicate)
    assert_equal(
        _class_attr_inherited_indexed(duplicate, d_idx, "dup", "geom", "type"),
        "sphere",
        "a duplicated class must keep resolving to the FIRST block",
    )

    # `<default` inside a comment. Both implementations match bare text and so
    # both see it; the index must not get clever.
    var commented = String(
        "<mujoco><default>"
        + '<!-- <default class="ghost"><geom type="box"/></default> -->'
        + '<default class="real"><geom type="capsule"/></default>'
        + "</default></mujoco>"
    )
    _ = _compare(commented, String("commented-default"))

    # A self-closing block, which the span loop lets swallow its siblings.
    var selfclosed = String(
        "<mujoco><default>"
        + '<default class="empty"/>'
        + '<default class="after"><geom type="ellipsoid"/></default>'
        + "</default></mujoco>"
    )
    _ = _compare(selfclosed, String("self-closing-default"))

    # Deep nesting: the chain walk, and an attribute that only the outermost
    # named class sets — quadruped's legs and dog's `lumbar` in miniature.
    var nested = String(
        "<mujoco><default>"
        + '<geom rgba="1 1 1 1"/>'
        + '<default class="body"><geom type="capsule" material="self"/>'
        + '<default class="hip"><geom fromto="0 0 0 1 0 0"/>'
        + '<default class="knee"><geom size="0.01"/></default>'
        + "</default>"
        + "</default>"
        + "</default></mujoco>"
    )
    _ = _compare(nested, String("nested-chain"))
    var n_idx = _build_defaults_index(nested)
    assert_equal(
        _class_attr_inherited_indexed(nested, n_idx, "knee", "geom", "type"),
        "capsule",
        "`type` must be reachable two links up the chain",
    )
    assert_equal(
        _class_attr_inherited_indexed(nested, n_idx, "knee", "geom", "rgba"),
        "",
        "the TOP-LEVEL block is deliberately not consulted by the chain walk",
    )

    # No `<default>` at all, and an unterminated one.
    _ = _compare(String("<mujoco><worldbody/></mujoco>"), String("no-defaults"))
    _ = _compare(
        String('<mujoco><default><geom type="box"/>'), String("unterminated")
    )

    print("pathological shapes: no disagreement")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
