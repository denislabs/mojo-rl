#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | Reference .parquet files + expected values, for the native Mojo reader
# +--------------------------------------------------------------------------+ #
"""Writes a small corpus of Parquet files and the values they should decode to.

    pixi run python tools/io/dump_parquet_reference.py --out /tmp/pq_ref
    pixi run mojo run -I . tests/io/test_parquet.mojo /tmp/pq_ref

⚠ THE LEROBOT FILES EXERCISE ONE CORNER OF THE FORMAT. Everything Arrow wrote
for `DenisLabs/record-test_*` is SNAPPY + `RLE_DICTIONARY` + data page v1, so a
reader that handles only that decodes the whole dataset and still has three
untested paths. Each case below turns one knob away from that default:

  plain_v1       dictionaries OFF  -> PLAIN values
  plain_v2       ... and data page v2, where the levels are UNCOMPRESSED and
                 sit in front of the compressed values, with byte counts in
                 the header instead of a 4-byte prefix
  raw_v1         compression OFF   -> the aliasing path that never calls snappy
  dict_v1        the LeRobot shape, reproduced small
  wide           a 40k-row column, so a chunk spans MANY pages and the
                 page loop's `seen`/`num_values` accounting is actually load
                 bearing rather than trivially satisfied on page 1
  lists          fixed_size_list<float> and list<int64>: max_def 3, max_rep 1,
                 the level streams the flat read has to consume and discard

`repeat` values are deliberately low-cardinality so the RLE branch of the
hybrid fires; `noise` values are deliberately distinct so the BIT-PACKED
branch does. Both branches in one file, which is how they appear in practice.

Expected values are written as text, one column per line:

    <name> <physical> <count> <v0> <v1> ...

floats with `repr()` so a float32 round-trips exactly through the text.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_case(out: Path, name: str, table: pa.Table, **kw):
    path = out / f"{name}.parquet"
    pq.write_table(table, path, **kw)
    return path


def expectations(table: pa.Table) -> list[str]:
    """One line per LEAF column, using the reader's dotted leaf paths."""
    lines = []
    for field in table.schema:
        col = table.column(field.name).combine_chunks()
        t = field.type
        if pa.types.is_fixed_size_list(t) or pa.types.is_list(t):
            leaf = f"{field.name}.list.element"
            vals = col.flatten()
            et = vals.type
        else:
            leaf = field.name
            vals = col
            et = t
        arr = np.asarray(vals)
        if pa.types.is_float32(et):
            phys, body = "FLOAT", [repr(float(np.float32(v))) for v in arr]
        elif pa.types.is_float64(et):
            phys, body = "DOUBLE", [repr(float(v)) for v in arr]
        elif pa.types.is_int64(et):
            phys, body = "INT64", [str(int(v)) for v in arr]
        elif pa.types.is_int32(et):
            phys, body = "INT32", [str(int(v)) for v in arr]
        elif pa.types.is_boolean(et):
            phys, body = "BOOLEAN", [str(int(bool(v))) for v in arr]
        else:
            raise SystemExit(f"unhandled element type {et}")
        lines.append(" ".join([leaf, phys, str(len(body))] + body))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/pq_ref")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)

    # ── the flat case, in four encoding/compression combinations ────────
    n = 528
    flat = pa.table(
        {
            # low cardinality -> RLE runs in the hybrid
            "repeat": pa.array(np.repeat(np.arange(8), n // 8), pa.int64()),
            # all distinct -> bit-packed runs
            "noise": pa.array(rng.integers(-2**40, 2**40, n), pa.int64()),
            "f32": pa.array(rng.standard_normal(n).astype(np.float32)),
            "f64": pa.array(rng.standard_normal(n)),
            "i32": pa.array(rng.integers(-2**30, 2**30, n), pa.int32()),
            "flag": pa.array(rng.integers(0, 2, n).astype(bool)),
        }
    )
    cases = {
        "dict_v1": dict(compression="snappy", use_dictionary=True,
                        data_page_version="1.0"),
        "plain_v1": dict(compression="snappy", use_dictionary=False,
                         data_page_version="1.0"),
        "plain_v2": dict(compression="snappy", use_dictionary=False,
                         data_page_version="2.0"),
        "raw_v1": dict(compression="none", use_dictionary=True,
                       data_page_version="1.0"),
        "dict_v2": dict(compression="snappy", use_dictionary=True,
                        data_page_version="2.0"),
    }
    manifest = []
    for name, kw in cases.items():
        write_case(out, name, flat, **kw)
        manifest.append((name, flat))

    # ── many pages in one chunk ─────────────────────────────────────────
    m = 40000
    wide = pa.table(
        {
            "seq": pa.array(np.arange(m, dtype=np.int64)),
            "val": pa.array((np.arange(m) * 0.25).astype(np.float32)),
        }
    )
    write_case(out, "wide", wide, compression="snappy",
               data_page_size=4096, row_group_size=9000)
    manifest.append(("wide", wide))

    # ── nested: the levels the flat read must consume ───────────────────
    k = 300
    lists = pa.table(
        {
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(rng.standard_normal(k * 6).astype(np.float32)), 6
            ),
            "ids": pa.ListArray.from_arrays(
                pa.array(np.arange(0, 3 * k + 1, 3, dtype=np.int32)),
                pa.array(rng.integers(0, 1000, 3 * k), pa.int64()),
            ),
        }
    )
    write_case(out, "lists", lists, compression="snappy")
    manifest.append(("lists", lists))

    with open(out / "expected.txt", "w") as f:
        for name, table in manifest:
            f.write(f"# {name} rows={table.num_rows}\n")
            for line in expectations(table):
                f.write(f"{name} {line}\n")

    print(f"wrote {len(manifest)} cases + expected.txt to {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:20s} {p.stat().st_size:>9d} bytes")


if __name__ == "__main__":
    main()
