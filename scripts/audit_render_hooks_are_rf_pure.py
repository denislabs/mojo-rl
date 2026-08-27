#!/usr/bin/env python3
"""The physics3d render hooks must be pure functions of `rf: RenderFields`.

WHY THIS EXISTS
===============
`ModelRenderer[MODEL_DEF]` uses its parameter as nothing but a namespace of
static hooks, and every hook takes `rf`. That is what lets ONE instantiation
draw ANY MJCF, which is what the physics3d studio is built on — see
`RfOnlyModelDef` in `parser/model_def_from_xml.mojo`.

The property is fragile in a way no compiler catches. `Self.NGEOM` is a legal
expression inside a hook; on `RfOnlyModelDef` it is 0, so reintroducing one
makes every runtime-loaded model draw ZERO geoms — silently, with no error and
no crash, while every comptime-model gate stays green. That is the "two model
paths" failure this tree has paid for repeatedly (`_acd`/`_rcd`, the `xyaxes`
conjugate fix that crossed to one of two parsers).

So: a hook body may not mention `Self.` at all. Counts come from
`len(rf.<list>)`, the source text from `rf.xml_text` / `rf.asset_base_dir`,
and anything else has to be a module-level function.

⚠ THE ALLOWED SET IS EMPTY ON PURPOSE. A hook calling a sibling hook through
`Self.` would be pure in fact and unlintable in practice; make it a
module-level function and have both call that.

    python3 scripts/audit_render_hooks_are_rf_pure.py       # 0 = clean
"""

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / (
    "mojo_rl/physics3d/parser/model_def_from_xml.mojo"
)

# Every static method that takes `rf` is a render hook by definition — the
# list is DERIVED, not hand-written, so a hook added later is covered without
# anyone remembering to add it here.
HOOK_ARG = re.compile(r"^    def (\w+)\(\s*$|^    def (\w+)\(rf: RenderFields")
MEMBER = re.compile(r"^    (def |@staticmethod|@always_inline)")
SELF = re.compile(r"Self\.(\w+)")


def hook_ranges(lines):
    """(name, start, end) for each `def` whose signature names `rf`.

    ⚠ The range stops at the next member OR at the first line that is not
    indented — otherwise the LAST hook in the struct swallows every
    module-level docstring after it, and the audit fails on its own prose.
    """
    out = []
    for i, line in enumerate(lines):
        m = re.match(r"^    def (\w+)\(", line)
        if not m:
            continue
        j = i + 1
        while j < len(lines):
            if MEMBER.match(lines[j]):
                break
            if lines[j].strip() and not lines[j].startswith(" "):
                break
            j += 1
        head = lines[i:j]
        sig_end = 0
        for k, h in enumerate(head):
            if re.match(r"^    \)", h) or h.rstrip().endswith("):"):
                sig_end = k
                break
        if "rf: RenderFields" in "\n".join(head[: sig_end + 1]):
            out.append((m.group(1), i, j))
    return out


def code_lines(lines, i, j):
    """The hook's CODE — comments and docstrings removed.

    ⚠ A `Self.x` inside a docstring is documentation, and this file's
    docstrings discuss the very members the audit forbids (that is how the
    reader learns why). Stripping them is what keeps the audit from failing on
    its own explanation.
    """
    out = []
    in_doc = False
    for k in range(i, j):
        line = lines[k]
        ticks = line.count('"""')
        if in_doc:
            if ticks:
                in_doc = False
            continue
        if ticks == 1:
            in_doc = True
            continue
        if ticks >= 2:
            continue
        if line.lstrip().startswith("#"):
            continue
        out.append((k, line))
    return out


def main() -> int:
    lines = SRC.read_text(encoding="utf-8").split("\n")
    hooks = hook_ranges(lines)
    if not hooks:
        # ⚠ NON-VACUITY. An audit that finds no hooks reports success while
        # checking nothing — this tree's single most common gate failure.
        print("FAIL: found NO render hooks; the signature match is stale.")
        return 2

    bad = 0
    scanned = 0
    for name, i, j in hooks:
        for k, code in code_lines(lines, i, j):
            scanned += 1
            for hit in SELF.finditer(code):
                bad += 1
                print(
                    f"FAIL {SRC.name}:{k + 1}: hook `{name}` reads "
                    f"`Self.{hit.group(1)}` — see this script's header."
                )
    # ⚠ PRINT WHAT WAS COMPARED BESIDE WHAT DIFFERED. "0 impure" reads the
    # same whether the audit examined 900 lines or none.
    print(
        f"checked {len(hooks)} render hooks / {scanned} code lines; "
        f"{bad} impure reference(s)"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
