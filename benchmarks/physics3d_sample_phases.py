"""Attribute `sample` call-graph time to physics phases, EXCLUSIVELY.

Three things this has to get right, each of which silently corrupted an
earlier version of this script:

1. ⚠ MAIN THREAD ONLY. The report contains one subtree per thread, and the
   idle runtime workers each carry a full window of samples — summing over the
   file gives ~9x the real total.
2. ⚠ MATCH THE SYMBOL HEAD, NOT THE WHOLE LINE. Mojo mangles the entire
   comptime XML and every generic parameter into the symbol, so a substring
   test against the full line matches modules the frame has nothing to do with.
   Only the part before the first '(' or '[' is the function's own path.
3. ⚠ DO NOT RESTRICT TO THE `step` SUBTREE. Inlining means much of the physics
   (solver, mass matrix, cdof) is symbolicated as a SIBLING of `step` rather
   than a child. The whole main thread is the step loop once sampling starts
   after model load.

Every sample is charged to the DEEPEST phase label on its stack, so the phases
partition the thread exactly and nothing is double counted.
"""
import re
import sys
from collections import defaultdict

PHASES = [
    ("broadphase",    "collision::broadphase_sap"),
    ("narrow/gjk",    "collision::gjk"),
    ("narrow/prim",   "collision::contact_detection"),
    ("convex_hull",   "collision::convex_hull"),
    ("collision/oth", "physics3d::collision"),
    ("solver",        "physics3d::solver"),
    ("constraint",    "physics3d::constraints"),
    ("mass_matrix",   "dynamics::mass_matrix"),
    ("ldl",           "dynamics::ldl"),
    ("cdof",          "dynamics::cdof"),
    ("subtree_com",   "dynamics::subtree_com"),
    ("rne/bias",      "dynamics::rne"),
    ("jacobian",      "dynamics::jacobian"),
    ("dyn/other",     "physics3d::dynamics"),
    ("kinematics",    "physics3d::kinematics"),
    ("integrator",    "physics3d::integrator"),
    ("sensor",        "physics3d::sensor"),
    ("parser",        "physics3d::parser"),
    ("physics/other", "physics3d::"),
    ("env/other",     "envs::"),
]

rows = []
for ln in open(sys.argv[1], errors="replace").read().split("\n"):
    m = re.match(r"^([ +!:|]*)(\d+) (.*)$", ln)
    if not m:
        continue
    head = re.split(r"[\(\[]", m.group(3), maxsplit=1)[0]
    rows.append((len(m.group(1)), int(m.group(2)), head, m.group(3)))

# (1) main thread subtree only
mi = next(i for i, r in enumerate(rows) if "main-thread" in r[3])
md = rows[mi][0]
hi = len(rows)
for j in range(mi + 1, len(rows)):
    if rows[j][0] <= md:
        hi = j
        break
base = rows[mi][1]


def label(head):
    for name, key in PHASES:
        if key in head:
            return name
    return None


child = defaultdict(int)
stack = []
for i in range(mi, hi):
    while stack and rows[stack[-1]][0] >= rows[i][0]:
        stack.pop()
    if stack:
        child[stack[-1]] += rows[i][1]
    stack.append(i)

excl = defaultdict(int)
stack = []
for i in range(mi, hi):
    depth, cnt, head, _ = rows[i]
    while stack and rows[stack[-1][0]][0] >= depth:
        stack.pop()
    lab = label(head) or (stack[-1][1] if stack else None)
    stack.append((i, lab))
    excl[lab] += cnt - child[i]

print(f"main thread {base} samples (1 ms each)")
print(f"\n  phase             samples   % of thread")
tot = 0
for name, _ in PHASES:
    if excl[name]:
        tot += excl[name]
        print(f"  {name:<16} {excl[name]:7d}   {100*excl[name]/base:8.2f}%")
print(f"  {'(unattributed)':<16} {base-tot:7d}   {100*(base-tot)/base:8.2f}%")
