#!/usr/bin/env python3
"""Generate the committed golden Parquet fixture for the WRITER's gate.

    pixi run python tools/io/make_parquet_golden.py

Writes `tests/fixtures/parquet/golden_v3_shapes.parquet` — a small file, in
every column shape a LeRobot v3 dataset uses, produced by Arrow's own writer.

## Why a committed file rather than a test-time comparison

`tests/io/test_parquet_write.mojo` has to answer "is what we wrote a real
Parquet file", and a round trip through our own reader cannot: the pair would
share any misunderstanding and agree with itself while nothing else could read
the result. That is the same reasoning `tests/io/test_png_write.mojo` records.

The independent party here is Arrow, and it is used ONCE — now, at authoring
time — exactly as `test_sha256.mojo`'s digests came from `hashlib` once and
were pinned. The gate then reads committed bytes that this repo did not
produce, and needs no `pyarrow` and no network to run.

## What the shapes are for

Measured from a real recording, not invented:

    scalar INT64 / FLOAT              episode_index, timestamp
    list<float32>[6]                  action                        def 3 rep 1
    list<string>, VARIABLE length     tasks                         def 3 rep 1
    list<double>[6] / list<int64>[1]  stats/<feature>/{mean,count}  def 3 rep 1
    list<list<list<double>>> [3,1,1]  stats/<camera>/mean           def 7 rep 3
    list<list<list<double>>> [2,3,2]  nested_2x3x2                  def 7 rep 3

⚠ THE DEPTH-3 COLUMN IS THE POINT. A writer bug that made the schema element
count `3*depth` instead of `2*depth+1` is INVISIBLE at depth 1 — the two agree
there — and produced a footer neither this repo nor pyarrow could deserialize
at depth 3. A fixture with only `list<T>` columns passes a broken writer.

⚠ ROW GROUPS HAVE DIFFERENT ROW COUNTS, on purpose. LeRobot writes one row
group per episode, so they are never uniform; a fixture with equal groups
cannot catch an offset computed from a fixed stride.

⚠ TASK LISTS HAVE DIFFERENT LENGTHS, on purpose. A fixed width per row is the
case where repetition levels are predictable from the row index, which is
exactly the assumption a writer must not make.
"""

from __future__ import annotations

import argparse
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

# Row counts per row group — deliberately unequal, like real episodes.
GROUPS = [7, 3, 11, 5]
N = sum(GROUPS)

# Elements per row for `tasks` — deliberately variable.
TASK_LENS = [1, 2, 1, 3, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1,
             1, 2, 1, 1]


def build() -> pa.Table:
    assert len(TASK_LENS) == N, f"{len(TASK_LENS)} task lengths for {N} rows"

    episode_index = [i // 4 for i in range(N)]
    timestamp = [round(i * 0.03333, 5) for i in range(N)]

    # Distinctive values: a shift of one row or one element is visible by eye
    # in a failure message, which a column of zeros would not be.
    action = [[float(i * 10 + j) for j in range(6)] for i in range(N)]
    stats_mean = [[float(i) + j / 8.0 for j in range(6)] for i in range(N)]
    stats_count = [[i + 1] for i in range(N)]

    tasks = []
    for i in range(N):
        tasks.append([f"t{i}.{k}" for k in range(TASK_LENS[i])])

    # [3, 1, 1] — the per-channel image statistic LeRobot stores.
    img_mean = [[[[float(i) + c / 4.0]] for c in range(3)] for i in range(N)]

    # ⚠ [2, 3, 2] EXISTS BECAUSE [3, 1, 1] CANNOT COVER THE CODE PATH.
    # With both inner dimensions equal to 1 there is never a second element
    # at depth 2 or 3, so repetition levels 2 and 3 are NEVER EMITTED — the
    # branches that produce them are dead. Corrupting them left the whole gate
    # green until this column existed. LeRobot's real data is [3, 1, 1] and so
    # can never exercise this; the fixture has to.
    nested = [
        [[[float(i * 100 + a * 10 + b * 2 + c) for c in range(2)]
          for b in range(3)] for a in range(2)]
        for i in range(N)
    ]

    table = pa.table(
        {
            "episode_index": pa.array(episode_index, pa.int64()),
            "timestamp": pa.array(timestamp, pa.float32()),
            "action": pa.array(action, pa.list_(pa.float32(), 6)),
            "tasks": pa.array(tasks, pa.list_(pa.string())),
            "stats/action/mean": pa.array(stats_mean, pa.list_(pa.float64())),
            "stats/action/count": pa.array(stats_count, pa.list_(pa.int64())),
            "stats/observation.images.front/mean": pa.array(
                img_mean, pa.list_(pa.list_(pa.list_(pa.float64())))
            ),
            "nested_2x3x2": pa.array(
                nested, pa.list_(pa.list_(pa.list_(pa.float64())))
            ),
        }
    )
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="tests/fixtures/parquet/golden_v3_shapes.parquet",
    )
    args = ap.parse_args()

    table = build()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(out, table.schema, compression="snappy")
    start = 0
    for n in GROUPS:
        writer.write_table(table.slice(start, n), row_group_size=n)
        start += n
    writer.close()

    # Read it back and report what a gate will see, so the numbers in the
    # test's header can be checked against this output rather than guessed.
    f = pq.ParquetFile(out)
    print(f"wrote {out}  {out.stat().st_size} bytes")
    print(f"  rows={f.metadata.num_rows}  row_groups={f.metadata.num_row_groups}")
    print(f"  group rows={[f.metadata.row_group(i).num_rows for i in range(f.metadata.num_row_groups)]}")
    print(f"  total task strings={sum(TASK_LENS)}")
    print("  leaves:")
    for i in range(f.metadata.row_group(0).num_columns):
        c = f.metadata.row_group(0).column(i)
        print(f"    {c.path_in_schema}  {c.physical_type}")


if __name__ == "__main__":
    main()
