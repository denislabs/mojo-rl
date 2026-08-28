#!/usr/bin/env python3
"""Which call sites are doing the device allocation? — nsys sqlite -> ranked sites.

`nsys stats` can tell you that `cuMemFree_v2` is 45% of CUDA API time. It cannot
tell you WHO called it. This can, and it is the method that already worked once
on TD-MPC2 (`docs/GPU_STEP_PERF.md`).

Capture (backtraces are only collected for MEMORY APIs under this flag, which is
the point — a full CPU sampling profile of a Mojo JIT process is noise):

    nsys profile --trace=cuda --cudabacktrace=memory:10000 \\
        --sample=process-tree --backtrace=dwarf --force-overwrite=true \\
        -o act_alloc  pixi run -e nvidia mojo run -I . \\
        examples/so101/act_so101_profile_gpu.mojo
    nsys export --type sqlite --force-overwrite=true act_alloc.nsys-rep
    python3 tools/nsys_alloc_sites.py act_alloc.sqlite

⚠ Symbols come back UNRESOLVED (raw IPs, module `[Unknown]`) for JIT-compiled
Mojo, and nsys assigns a FRESH `callchainId` per sample — so the grouping key
here is the ADDRESS TUPLE, not the id. Unresolved was enough last time: 97% of
allocations came from one site, and the cutlass kernel NAMES around it
identified the caller.

⚠ The timings in a backtrace-enabled run are NOT comparable to a plain profile.
Read counts and call-site shares from this one, wall clock from the other.

## If the callchain table comes back empty

nsys ACCEPTS `--cudabacktrace` and then silently collects nothing when the
unwinder cannot walk the stack. In rough order of likelihood:

  1. The threshold was above every call. Lower it: `--cudabacktrace=memory:1000`.
  2. Sampling was not actually on. `--cudabacktrace` needs it to unwind;
     `--sample=process-tree` on current nsys, `--sample=cpu` on older ones.
     nsys does not always warn when it downgrades this.
  3. The unwinder cannot walk a JIT frame. Try `--backtrace=lbr` (Intel
     last-branch, cheap and needs no frame pointers) or `--backtrace=fp`
     instead of `dwarf`.
  4. The nsys shipped inside `nsight-compute/.../target-linux-x64/` is a cut-down
     target-side binary. A standalone `nsys` from a Nsight Systems install
     (`/opt/nvidia/nsight-systems/*/bin/nsys`) collects backtraces the embedded
     one may not — check `nsys --version` and `nsys profile --help | grep -A3
     cudabacktrace`.

Run with `--list` to see what an export DOES contain before debugging further.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict

MEMORY_APIS = ("cuMemAlloc", "cuMemFree", "cuMemCreate", "cuMemAllocHost",
               "cuMemAllocManaged", "cuMemMap", "cuMemRelease")


def table_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def columns(cur, table: str) -> set[str]:
    return {r[1] for r in cur.execute(f"PRAGMA table_info({table})")}


def die(msg: str, cur=None) -> None:
    """⚠ STDOUT, not stderr. The caller pipes this through `tee`, and a
    diagnosis that goes to stderr lands in the terminal while the log file
    comes out EMPTY — which is exactly how the first run of this script was
    lost."""
    print(f"\nerror: {msg}")
    if cur is not None:
        inventory(cur)
    sys.exit(1)


def inventory(cur) -> None:
    """Every non-empty table, with its row count. Printed BEFORE any analysis,
    so a schema that does not match this script is legible instead of fatal."""
    names = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    rows = []
    for n in names:
        try:
            c = cur.execute(f"SELECT count(*) FROM [{n}]").fetchone()[0]
        except sqlite3.Error:
            c = -1
        if c:
            rows.append((n, c))
    print(f"\n=== non-empty tables ({len(rows)} of {len(names)}) ===")
    for n, c in rows:
        star = "  <== callchains" if "CALLCHAIN" in n.upper() else ""
        print(f"  {c:>12}  {n}{star}")
    # The columns that decide whether this export can be attributed at all.
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUDA_CALLCHAINS",
              "SAMPLING_CALLCHAINS"):
        if table_exists(cur, t):
            print(f"\n  {t} columns: {sorted(columns(cur, t))}")


def find_callchain_table(cur) -> tuple[str, str, str] | None:
    """(table, id column, ip column) for whichever callchain table this nsys
    version wrote. The name and the column spellings both drift between
    versions, so probe instead of hardcoding `CUDA_CALLCHAINS.originalIP`."""
    for t in ("CUDA_CALLCHAINS", "SAMPLING_CALLCHAINS", "CUDA_CALLCHAIN"):
        if not table_exists(cur, t):
            continue
        cols = columns(cur, t)
        idc = next((c for c in ("id", "callchainId", "chainId") if c in cols), None)
        ipc = next((c for c in ("originalIP", "ip", "originalIp", "address")
                    if c in cols), None)
        if idc and ipc and cur.execute(f"SELECT count(*) FROM [{t}]").fetchone()[0]:
            return t, idc, ipc
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite", help="the .sqlite from `nsys export --type sqlite`")
    ap.add_argument("--top", type=int, default=12, help="call sites to print")
    ap.add_argument("--depth", type=int, default=14, help="frames per site")
    ap.add_argument("--api", default=None,
                    help="only this API (substring, e.g. cuMemAlloc). "
                         "Default: every memory API found.")
    ap.add_argument("--list", action="store_true",
                    help="print the table inventory and stop")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()

    # ⚠ ALWAYS first, and always on stdout. A schema this script cannot read is
    # a fact about the capture, not a crash, and it has to survive `tee`.
    inventory(cur)
    if args.list:
        return

    rt = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if not table_exists(cur, rt):
        die(f"no {rt} table — was the run captured with --trace=cuda?")

    cc = find_callchain_table(cur)
    if cc is None:
        die("no non-empty callchain table (looked for CUDA_CALLCHAINS, "
            "SAMPLING_CALLCHAINS). nsys accepted --cudabacktrace but collected "
            "nothing, which means the UNWINDER produced no frames — see the "
            "'if this comes back empty' notes in the header.")
    cc_table, cc_id, cc_ip = cc
    print(f"\nusing callchains from {cc_table}({cc_id}, {cc_ip})")

    rtcols = columns(cur, rt)
    if "callchainId" not in rtcols:
        die(f"{rt} has no callchainId column (columns: {sorted(rtcols)})")

    # ── which API is each runtime row? ───────────────────────────────────
    names = dict(cur.execute("SELECT id, value FROM StringIds"))

    rows = list(cur.execute(
        f"SELECT nameId, callchainId, start, end FROM {rt} "
        f"WHERE callchainId IS NOT NULL AND callchainId != 0"))
    if not rows:
        n_rt = cur.execute(f"SELECT count(*) FROM {rt}").fetchone()[0]
        die(f"{n_rt} runtime rows, NONE with a callchainId. Only MEMORY APIs "
            "carry one under --cudabacktrace=memory, so this is usually the "
            "threshold sitting above every call: lower the `:<ns>` to :1000.")

    # ── the address tuple IS the identity (see the header) ───────────────
    order = "stackDepth" if "stackDepth" in columns(cur, cc_table) else cc_id
    chains: dict[int, list[int]] = defaultdict(list)
    for cid, ip in cur.execute(
            f"SELECT [{cc_id}], [{cc_ip}] FROM [{cc_table}] "
            f"ORDER BY [{cc_id}], [{order}]"):
        chains[cid].append(ip)

    # optional symbol/module resolution, when nsys managed any
    sym = {}
    ccols = columns(cur, cc_table)
    if {cc_ip, "symbol", "module"} <= ccols:
        for ip, sy, m in cur.execute(
                f"SELECT [{cc_ip}], symbol, module FROM [{cc_table}]"):
            if sy or m:
                sym.setdefault(ip, (names.get(sy, ""), names.get(m, "")))

    per_api: dict[str, Counter] = defaultdict(Counter)
    per_api_ns: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    totals_ns: Counter = Counter()

    for nameid, cid, start, end in rows:
        api = names.get(nameid, f"<name {nameid}>")
        if args.api and args.api not in api:
            continue
        if not any(api.startswith(p) for p in MEMORY_APIS):
            continue
        key = tuple(chains.get(cid, ()))
        dur = (end or 0) - (start or 0)
        per_api[api][key] += 1
        per_api_ns[api][key] += dur
        totals[api] += 1
        totals_ns[api] += dur

    if not totals:
        die("no memory-API rows with backtraces. Widen with --api or lower the "
            "--cudabacktrace threshold.")

    print(f"=== memory APIs with backtraces in {args.sqlite} ===\n")
    print(f"  {'API':<24}{'calls':>10}{'total ms':>12}{'mean us':>11}")
    for api, n in totals.most_common():
        ms = totals_ns[api] / 1e6
        print(f"  {api:<24}{n:>10}{ms:>12.1f}{totals_ns[api]/n/1e3:>11.1f}")

    for api, n in totals.most_common():
        sites = per_api[api]
        print(f"\n\n=== {api}: {len(sites)} distinct call sites, "
              f"{n} calls ===")
        for rank, (key, count) in enumerate(sites.most_common(args.top), 1):
            ms = per_api_ns[api][key] / 1e6
            share = 100.0 * count / n
            print(f"\n  #{rank}  {count} calls ({share:.1f}%)  {ms:.1f} ms"
                  f"  [{len(key)} frames]")
            if not key:
                print("       <empty callchain>")
                continue
            for ip in key[:args.depth]:
                s, m = sym.get(ip, ("", ""))
                label = s or "[unresolved]"
                mod = f"  in {m}" if m else ""
                print(f"       0x{ip:016x}  {label}{mod}")
            if len(key) > args.depth:
                print(f"       ... {len(key) - args.depth} more frames "
                      f"(raise --depth)")

    print("\n\nReading this: identical stacks mean ONE site. If the top site is "
          "a large share,\nname it from the kernels around it — a cutlass "
          "`..._tn_align1` next to it means a\nvjp GEMM, `_align4` next to a "
          "workspace memset means cuBLAS picked cutlass.\n"
          "⚠ `cuMemAlloc` count is NOT 1:1 with `splitKreduce`: cuBLAS "
          "allocates whenever it\npicks a cutlass kernel, and only about half "
          "of those also split-K. Deriving 'a\nsecond allocation source' from "
          "that mismatch was wrong once already.")


if __name__ == "__main__":
    main()
