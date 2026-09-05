"""Top EXCLUSIVE symbols of a `sample` report, main thread only.

    python3 benchmarks/physics3d_sample_top.py /tmp/s.txt [N]

Companion to `physics3d_sample_phases.py`, which folds the same tree into
phase buckets; this one answers "which FUNCTION" inside a bucket. Same three
traps (main thread only; symbol HEAD, not the mangled line; no `step`-subtree
restriction) -- see that script's header.
"""
import re
import sys
from collections import defaultdict

rows = []
for ln in open(sys.argv[1], errors="replace").read().split("\n"):
    m = re.match(r"^([ +!:|]*)(\d+) (.*)$", ln)
    if not m:
        continue
    head = re.split(r"[\(\[]", m.group(3), maxsplit=1)[0]
    rows.append((len(m.group(1)), int(m.group(2)), head))

mi = next(i for i, r in enumerate(rows) if "main-thread" in r[2])
md = rows[mi][0]
hi = len(rows)
for j in range(mi + 1, len(rows)):
    if rows[j][0] <= md:
        hi = j
        break
base = rows[mi][1]

child = defaultdict(int)
stack = []
for i in range(mi, hi):
    while stack and rows[stack[-1]][0] >= rows[i][0]:
        stack.pop()
    if stack:
        child[stack[-1]] += rows[i][1]
    stack.append(i)

excl = defaultdict(int)
for i in range(mi, hi):
    excl[rows[i][2]] += rows[i][1] - child[i]

n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
print(f"main thread {base} samples")
for head, cnt in sorted(excl.items(), key=lambda kv: -kv[1])[:n]:
    if cnt:
        print(f"  {cnt:7d} {100*cnt/base:6.2f}%  {head[-110:]}")
