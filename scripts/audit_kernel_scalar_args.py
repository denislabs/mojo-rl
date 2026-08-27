#!/usr/bin/env python3
"""Flag GPU kernel parameters that are not `DevicePassable`.

Since Mojo 1.0.0rc2 `Int` and `UInt` do NOT conform to `DevicePassable`
(`Bool` never did). A kernel that takes one compiles fine on its own; it fails
where it is LAUNCHED, as a constraint error raised deep inside `SIMD`:

    simd.mojo:612: constraint failed: Int and UInt do not conform to
    DevicePassable; use a fixed-width type such as Int32 or Int64 instead

**`pixi run build` CANNOT SEE THIS.** `mojo precompile` stops at elaboration
and never instantiates a generic kernel, so the whole class is invisible to
the package build and surfaces only when some test or example instantiates
that particular kernel with those particular parameters. That is why this
audit exists: it is the build-time check the compiler cannot give us.

FIX: declare the parameter `Int32` (or `Int64` if the value can exceed 2^31),
re-widen with `Int(...)` at its uses inside the kernel body, and cast at every
launch site — a bare `Int` argument re-introduces the failure through implicit
conversion even when the signature is right. See
`planners/trajectory/mppi_kernels.mojo::mppi_sample_actions_batched_kernel`.

Usage:
    python3 scripts/audit_kernel_scalar_args.py [roots...]
        # default roots: mojo_rl examples tests

Exit code is the number of findings, so it can gate CI.

CAVEATS:

  * A "kernel" here is any symbol reaching `enqueue_function[...]`, resolved
    through one level of `comptime <alias> = [Self.]<kernel>[`. An alias name
    is mapped to the SET of every RHS it is bound to in that file, NOT the last
    one — reusing `comptime k = ...` for several kernels in one file is common
    here, and collapsing that to a dict silently hid two kernels the first time
    this was written.
  * UNRESOLVED launches are reported and counted. A launch whose symbol cannot
    be traced to a `def` is a blind spot, not a pass.
  * Only scalar parameters are checked. A struct passed by value would need to
    be `DevicePassable` as a whole; none is passed that way today.
"""

import os
import re
import sys

BAD = re.compile(r":\s*(Int|UInt|Bool)\s*(=|$)")
LAUNCH = re.compile(r"enqueue_function(?:_checked|_unchecked)?\s*\[\s*([A-Za-z_]\w*)")
ALIAS = re.compile(
    r"comptime\s+([A-Za-z_]\w*)\s*=\s*\(?\s*(?:Self\.)?([A-Za-z_]\w*)\s*[\[(]"
)


def _match(text, i, opener, closer):
    """Index of the bracket closing the one at `i`, or -1."""
    depth = 0
    while i < len(text):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top(params):
    """Split a parameter list on top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in params:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


# `examples/archive/README.md` states these "do NOT compile and are excluded
# from the `examples-compile` CI manifest" — they reference removed APIs on
# purpose. Scanning them only ever produces unresolvable imports.
SKIP = ("examples/archive",)


def _sources(roots):
    for root in roots:
        for d, _, files in os.walk(root):
            if any(d.startswith(s) for s in SKIP):
                continue
            for f in files:
                if f.endswith(".mojo"):
                    yield os.path.join(d, f)


def scan(roots):
    defs, launched = {}, set()
    for path in _sources(roots):
        text = open(path).read()

        for m in re.finditer(r"^\s*def\s+([A-Za-z_]\w*)\s*", text, re.M):
            name, j = m.group(1), m.end()
            if j < len(text) and text[j] == "[":
                k = _match(text, j, "[", "]")
                if k < 0:
                    continue
                j = k + 1
            if j >= len(text) or text[j] != "(":
                continue
            k = _match(text, j, "(", ")")
            if k < 0:
                continue
            line = text.count("\n", 0, m.start()) + 1
            # Strip comments BEFORE the comma split: a `#` note inside a
            # parameter list often contains commas of its own, and splitting
            # first tears it into fragments that read like parameters.
            params = "\n".join(
                l.split("#")[0] for l in text[j + 1 : k].split("\n")
            )
            defs.setdefault(name, []).append((path, line, params))

        aliases = {}
        for alias, target in ALIAS.findall(text):
            aliases.setdefault(alias, set()).add(target)
        for sym in LAUNCH.findall(text):
            launched.update(aliases.get(sym, {sym}))

    findings, unresolved = [], sorted(n for n in launched if n not in defs)
    for name in sorted(launched):
        for path, line, params in defs.get(name, []):
            for p in _split_top(params):
                p = p.strip()
                if p and BAD.search(p + "\n"):
                    findings.append((path, line, name, " ".join(p.split())))
    return findings, unresolved, len(launched)


def main():
    # Defaults span all three roots on purpose: `pixi run build` only covers
    # `mojo_rl`, and tests/examples define kernels of their own that nothing
    # else compiles. Three of the first four findings here were in an example.
    roots = sys.argv[1:] or ["mojo_rl", "examples", "tests"]
    findings, unresolved, n_launched = scan(roots)
    resolved = n_launched - len(unresolved)
    print(f"kernels launched: {n_launched}   resolved to a def: {resolved}")
    print(f"kernel params that are not DevicePassable: {len(findings)}")
    for path, line, name, param in findings:
        print(f"  {path}:{line}  {name}(… {param} …)")
    if unresolved:
        print(f"\nUNRESOLVED launches (blind spots, counted as findings): "
              f"{len(unresolved)}")
        for name in unresolved:
            print(f"  {name}")
    return len(findings) + len(unresolved)


if __name__ == "__main__":
    sys.exit(main())
