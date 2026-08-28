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
    print(f"error: {msg}", file=sys.stderr)
    if cur is not None:
        names = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print("\ntables present in this export:", file=sys.stderr)
        for n in names:
            print(f"    {n}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite", help="the .sqlite from `nsys export --type sqlite`")
    ap.add_argument("--top", type=int, default=12, help="call sites to print")
    ap.add_argument("--depth", type=int, default=14, help="frames per site")
    ap.add_argument("--api", default=None,
                    help="only this API (substring, e.g. cuMemAlloc). "
                         "Default: every memory API found.")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()

    rt = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if not table_exists(cur, rt):
        die(f"no {rt} table — was the run captured with --trace=cuda?", cur)
    if not table_exists(cur, "CUDA_CALLCHAINS"):
        die("no CUDA_CALLCHAINS table — the run had no --cudabacktrace=memory:<ns>. "
            "Re-capture with the command in this file's docstring.", cur)

    rtcols = columns(cur, rt)
    if "callchainId" not in rtcols:
        die(f"{rt} has no callchainId column (columns: {sorted(rtcols)})")

    # ── which API is each runtime row? ───────────────────────────────────
    names = dict(cur.execute("SELECT id, value FROM StringIds"))

    rows = list(cur.execute(
        f"SELECT nameId, callchainId, start, end FROM {rt} "
        f"WHERE callchainId IS NOT NULL AND callchainId != 0"))
    if not rows:
        die("no runtime rows carry a callchainId. Only MEMORY APIs do under "
            "--cudabacktrace=memory, so this usually means the threshold was "
            "higher than every call: lower the `:<ns>` (try :1000).")

    # ── the address tuple IS the identity (see the header) ───────────────
    chains: dict[int, list[int]] = defaultdict(list)
    for cid, ip in cur.execute(
            "SELECT id, originalIP FROM CUDA_CALLCHAINS ORDER BY id, stackDepth"):
        chains[cid].append(ip)

    # optional symbol/module resolution, when nsys managed any
    sym = {}
    if table_exists(cur, "CUDA_CALLCHAINS"):
        ccols = columns(cur, "CUDA_CALLCHAINS")
        if {"originalIP", "symbol", "module"} <= ccols:
            for ip, s, m in cur.execute(
                    "SELECT originalIP, symbol, module FROM CUDA_CALLCHAINS"):
                if s or m:
                    sym.setdefault(ip, (names.get(s, ""), names.get(m, "")))

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
