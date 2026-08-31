# +--------------------------------------------------------------------------+ #
# | The Parquet WRITER, against a file Arrow wrote
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/io/parquet/writer.mojo`.

    pixi run mojo run -I . tests/io/test_parquet_write.mojo

⚠ A ROUND TRIP THROUGH OUR OWN PAIR PROVES NOTHING ON ITS OWN. Reader and
writer would share any misunderstanding about level encoding, the LIST
annotation or the schema tree, agree with each other perfectly, and produce a
file nothing else can open. `tests/io/test_png_write.mojo` records the same
reasoning for PNG; Parquet has strictly more places to get this wrong.

So the reference is `tests/fixtures/parquet/golden_v3_shapes.parquet` — 10 KB
of bytes THIS REPO DID NOT PRODUCE, written by Arrow
(`parquet-cpp-arrow 24.0.0`) via `tools/io/make_parquet_golden.py`. The
generator ran ONCE, at authoring time, the way `test_sha256.mojo`'s digests
came out of `hashlib` once and were pinned. Running this gate needs no
`pyarrow`, no network and no dataset.

## The three legs

**Structure.** Every leaf of the file we write must match the golden's leaf
for leaf: path, physical type, max definition level, max repetition level —
plus the row-group split and each column chunk's value count. Those numbers
are DERIVED FROM ARROW'S BYTES by `reader.mojo`, so agreeing with them is a
statement about the format, not about us.

**Values.** Every value read back out of our file must equal the value read
out of the golden — 26 rows across 8 columns, including 38 variable-length
strings and two three-level nested columns.

**Grouping.** The repetition level stream of every list column, compared
value for value against the golden's.

⚠ **THE THIRD LEG IS NOT OPTIONAL, AND THIS GATE PROVED IT.** With only the
first two, deliberately corrupting the depth-3 repetition levels — emitting
`1` where the format requires `2` and `3` — left this gate FULLY GREEN:
structure identical, every value identical. A wrong repetition stream puts
every value in the right order and the WRONG ROW, and the value readers cannot
see it because `reader.mojo` used to discard those levels outright. That is
why `ParquetFile.read_rep_levels` exists; it is a gate's tool and nothing on
the import path calls it.

⚠ **AND THE GROUPING LEG ALONE WAS STILL NOT ENOUGH.** LeRobot's nested column
is `[3,1,1]`, whose inner dimensions are BOTH 1 — so a second element at depth
2 or 3 never occurs and repetition levels 2 and 3 are never emitted at all.
The same sabotage survived the grouping leg until `nested_2x3x2` was added to
the fixture. Real data cannot cover that path; the fixture has to. Compare
`_the_real_data_did_not_cover_the_decoder` — CIFAR-10 never emits PNG's
Average filter either.
"""

from mojo_rl.io.parquet import ParquetFile
from mojo_rl.io.parquet.metadata import PT_BYTE_ARRAY, PT_INT64, physical_type_name
from mojo_rl.io.parquet.writer import (
    ParquetWriter, PQ_F32, PQ_F64, PQ_I64, PQ_STR, PqColumn, pq_list,
    pq_list3, pq_scalar,
)


comptime GOLDEN = "tests/fixtures/parquet/golden_v3_shapes.parquet"
comptime OUT = "/tmp/mojo_rl_parquet_write_gate.parquet"

comptime N_ROWS = 26
comptime N_GROUPS = 4

comptime GROUP_ROWS: InlineArray[Int, N_GROUPS] = [7, 3, 11, 5]

# ⚠ PINNED FROM THE GENERATOR, not recovered from the golden — see the header.
comptime TASK_LENS: InlineArray[Int, N_ROWS] = [
    1, 2, 1, 3, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1,
    1, 2, 1, 1,
]

comptime N_TASK_STRINGS = 38


def _columns() -> List[PqColumn]:
    """The golden's schema, in its column order."""
    var c = List[PqColumn]()
    c.append(pq_scalar(String("episode_index"), PQ_I64))
    c.append(pq_scalar(String("timestamp"), PQ_F32))
    c.append(pq_list(String("action"), PQ_F32, 6))
    c.append(pq_list(String("tasks"), PQ_STR))  # variable width
    c.append(pq_list(String("stats/action/mean"), PQ_F64, 6))
    c.append(pq_list(String("stats/action/count"), PQ_I64, 1))
    c.append(
        pq_list3(
            String("stats/observation.images.front/mean"), PQ_F64, 3, 1, 1
        )
    )
    # ⚠ [2,3,2] is not a LeRobot shape. It is here because [3,1,1] has both
    # inner dimensions equal to 1 and therefore NEVER emits repetition level
    # 2 or 3 — see the header.
    c.append(pq_list3(String("nested_2x3x2"), PQ_F64, 2, 3, 2))
    return c^


def main() raises:
    print("[parquet-write] gate")

    var golden = ParquetFile(String(GOLDEN))
    if golden.num_rows() != N_ROWS:
        raise Error(
            "the golden fixture has " + String(golden.num_rows())
            + " rows, this gate is written for " + String(N_ROWS)
            + " — regenerate it with tools/io/make_parquet_golden.py"
        )

    # ── read every column out of the golden ───────────────────────────
    var g_ep = golden.read_i64(String("episode_index"))
    var g_ts = golden.read_f64(String("timestamp"))
    var g_act = golden.read_f64(String("action.list.element"))
    var g_mean = golden.read_f64(String("stats/action/mean.list.element"))
    var g_cnt = golden.read_i64(String("stats/action/count.list.element"))
    var g_img = golden.read_f64(
        String("stats/observation.images.front/mean.list.element.list.element"
               ".list.element")
    )
    var g_nest = golden.read_f64(
        String("nested_2x3x2.list.element.list.element.list.element")
    )
    var g_tb = List[UInt8]()
    var g_to = List[Int]()
    var n_tasks = golden.read_byte_arrays(
        String("tasks.list.element"), g_tb, g_to
    )
    if n_tasks != N_TASK_STRINGS:
        raise Error(
            "the golden holds " + String(n_tasks) + " task strings, expected "
            + String(N_TASK_STRINGS)
        )

    # ── write the same table back out ─────────────────────────────────
    var w = ParquetWriter(_columns())
    var task_lens = materialize[TASK_LENS]()
    var group_rows = materialize[GROUP_ROWS]()
    var row = 0
    var task_i = 0
    for g in range(N_GROUPS):
        var vals = w.new_values()
        for _ in range(group_rows[g]):
            vals[0].push_i64(Int(g_ep[row]))
            vals[1].push_f64(g_ts[row])
            for j in range(6):
                vals[2].push_f64(g_act[row * 6 + j])
            var k = task_lens[row]
            for _ in range(k):
                var s = String("")
                for b in range(g_to[task_i], g_to[task_i + 1]):
                    s += chr(Int(g_tb[b]))
                vals[3].push_str(s)
                task_i += 1
            vals[3].push_count(k)
            for j in range(6):
                vals[4].push_f64(g_mean[row * 6 + j])
            vals[5].push_i64(Int(g_cnt[row]))
            for j in range(3):
                vals[6].push_f64(g_img[row * 3 + j])
            for j in range(12):
                vals[7].push_f64(g_nest[row * 12 + j])
            row += 1
        w.write_row_group(vals, group_rows[g])
    var nbytes = w.close(String(OUT))
    if row != N_ROWS or task_i != N_TASK_STRINGS:
        raise Error(
            "the writer loop staged " + String(row) + " rows and "
            + String(task_i) + " strings"
        )
    print(
        "  wrote " + String(nbytes) + " bytes, " + String(N_ROWS) + " rows in "
        + String(N_GROUPS) + " row groups"
    )

    # ── leg 1: the STRUCTURE must match Arrow's ───────────────────────
    var ours = ParquetFile(String(OUT))
    if len(ours.meta.leaves) != len(golden.meta.leaves):
        raise Error(
            "we wrote " + String(len(ours.meta.leaves)) + " leaves, the golden"
            " has " + String(len(golden.meta.leaves))
        )
    var leaves_checked = 0
    for i in range(len(golden.meta.leaves)):
        ref a = golden.meta.leaves[i]
        ref b = ours.meta.leaves[i]
        if a.path != b.path:
            raise Error(
                "leaf " + String(i) + ": golden path '" + a.path
                + "', ours '" + b.path + "'"
            )
        if a.physical_type != b.physical_type:
            raise Error(
                "leaf '" + a.path + "': golden type "
                + physical_type_name(a.physical_type) + ", ours "
                + physical_type_name(b.physical_type)
            )
        if a.max_def != b.max_def or a.max_rep != b.max_rep:
            raise Error(
                "leaf '" + a.path + "': golden levels def=" + String(a.max_def)
                + "/rep=" + String(a.max_rep) + ", ours def="
                + String(b.max_def) + "/rep=" + String(b.max_rep)
            )
        leaves_checked += 1
    if leaves_checked != 8:
        raise Error(
            "compared " + String(leaves_checked) + " leaves, expected 8 —"
            " the fixture changed shape"
        )
    print(
        "  structure: " + String(leaves_checked)
        + "/8 leaves match Arrow's paths, types and levels"
    )

    if len(ours.meta.row_groups) != len(golden.meta.row_groups):
        raise Error(
            "we wrote " + String(len(ours.meta.row_groups)) + " row groups,"
            " the golden has " + String(len(golden.meta.row_groups))
        )
    var chunks_checked = 0
    for g in range(len(golden.meta.row_groups)):
        ref ga = golden.meta.row_groups[g]
        ref gb = ours.meta.row_groups[g]
        if ga.num_rows != gb.num_rows:
            raise Error(
                "row group " + String(g) + ": golden has "
                + String(ga.num_rows) + " rows, ours " + String(gb.num_rows)
            )
        for c in range(len(ga.columns)):
            if ga.columns[c].num_values != gb.columns[c].num_values:
                raise Error(
                    "row group " + String(g) + " column '"
                    + ga.columns[c].path + "': golden declares "
                    + String(ga.columns[c].num_values) + " values, ours "
                    + String(gb.columns[c].num_values)
                )
            chunks_checked += 1
    print(
        "  structure: " + String(len(golden.meta.row_groups))
        + " row groups, " + String(chunks_checked)
        + " column chunks agree on row and value counts"
    )
    if chunks_checked != 32:
        raise Error(
            "compared " + String(chunks_checked) + " chunks, expected 32"
        )

    # ── leg 2: every VALUE must survive ───────────────────────────────
    var o_ep = ours.read_i64(String("episode_index"))
    var o_ts = ours.read_f64(String("timestamp"))
    var o_act = ours.read_f64(String("action.list.element"))
    var o_mean = ours.read_f64(String("stats/action/mean.list.element"))
    var o_cnt = ours.read_i64(String("stats/action/count.list.element"))
    var o_img = ours.read_f64(
        String("stats/observation.images.front/mean.list.element.list.element"
               ".list.element")
    )
    var o_nest = ours.read_f64(
        String("nested_2x3x2.list.element.list.element.list.element")
    )
    var o_tb = List[UInt8]()
    var o_to = List[Int]()
    var o_ntasks = ours.read_byte_arrays(
        String("tasks.list.element"), o_tb, o_to
    )

    var compared = 0
    compared += _same_i64(String("episode_index"), g_ep, o_ep)
    compared += _same_f64(String("timestamp"), g_ts, o_ts)
    compared += _same_f64(String("action"), g_act, o_act)
    compared += _same_f64(String("stats/action/mean"), g_mean, o_mean)
    compared += _same_i64(String("stats/action/count"), g_cnt, o_cnt)
    compared += _same_f64(String("stats/<cam>/mean"), g_img, o_img)
    compared += _same_f64(String("nested_2x3x2"), g_nest, o_nest)

    if o_ntasks != n_tasks:
        raise Error(
            "we wrote " + String(o_ntasks) + " task strings, the golden has "
            + String(n_tasks)
        )
    if len(o_tb) != len(g_tb):
        raise Error(
            "task bytes: " + String(len(o_tb)) + " vs " + String(len(g_tb))
        )
    for i in range(len(g_tb)):
        if g_tb[i] != o_tb[i]:
            raise Error("task byte " + String(i) + " differs")
    for i in range(len(g_to)):
        if g_to[i] != o_to[i]:
            raise Error(
                "task string offset " + String(i) + ": golden "
                + String(g_to[i]) + ", ours " + String(o_to[i])
            )
    compared += len(g_tb)

    # ⚠ VACUITY. Print what was actually compared; "0 mismatches" over an
    # empty comparison is the default failure mode of a gate like this.
    print(
        "  values: " + String(compared) + " compared, 0 differ  ("
        + String(len(g_ep)) + " i64 + " + String(len(g_act) + len(g_mean)
        + len(g_img) + len(g_ts)) + " float + " + String(n_tasks)
        + " strings / " + String(len(g_tb)) + " bytes)"
    )
    if compared < 500:
        raise Error(
            "only " + String(compared) + " values compared — the fixture is"
            " not exercising what this gate claims to cover"
        )

    # ── leg 3: the GROUPING must match ────────────────────────────────
    # ⚠ Without this leg, sabotaging the depth-3 repetition levels left the
    # two legs above completely green. Values in the right order, wrong rows.
    var list_cols = List[String]()
    list_cols.append(String("action.list.element"))
    list_cols.append(String("tasks.list.element"))
    list_cols.append(String("stats/action/mean.list.element"))
    list_cols.append(String("stats/action/count.list.element"))
    list_cols.append(
        String("stats/observation.images.front/mean.list.element.list.element"
               ".list.element")
    )
    list_cols.append(
        String("nested_2x3x2.list.element.list.element.list.element")
    )

    var levels_compared = 0
    for i in range(len(list_cols)):
        var ga = golden.read_rep_levels(list_cols[i])
        var gb = ours.read_rep_levels(list_cols[i])
        if len(ga) != len(gb):
            raise Error(
                list_cols[i] + ": golden has " + String(len(ga))
                + " repetition levels, ours " + String(len(gb))
            )
        if len(ga) == 0:
            raise Error(list_cols[i] + ": no repetition levels to compare")
        for k in range(len(ga)):
            if ga[k] != gb[k]:
                raise Error(
                    list_cols[i] + ": repetition level " + String(k)
                    + " is " + String(Int(gb[k])) + ", Arrow wrote "
                    + String(Int(ga[k]))
                    + " — the values are grouped into the wrong rows"
                )
        levels_compared += len(ga)

        # And the derived per-row counts, which is what a caller would see.
        var ca = golden.read_list_counts(list_cols[i])
        var cb = ours.read_list_counts(list_cols[i])
        if len(ca) != N_ROWS or len(cb) != N_ROWS:
            raise Error(
                list_cols[i] + ": counts cover " + String(len(ca)) + " / "
                + String(len(cb)) + " rows, expected " + String(N_ROWS)
            )
        for k in range(N_ROWS):
            if ca[k] != cb[k]:
                raise Error(
                    list_cols[i] + " row " + String(k) + ": we grouped "
                    + String(cb[k]) + " values, Arrow grouped " + String(ca[k])
                )

    # The variable-width column is the one whose grouping is not implied by a
    # constant, so say out loud that it really varies.
    var tl = golden.read_list_counts(String("tasks.list.element"))
    var widths_seen = 0
    for k in range(len(tl)):
        if tl[k] != tl[0]:
            widths_seen = 1
            break
    if widths_seen == 0:
        raise Error(
            "the `tasks` column has a constant width in the fixture — the"
            " variable-width path is not being exercised"
        )
    print(
        "  grouping: " + String(levels_compared) + " repetition levels across "
        + String(len(list_cols)) + " list columns match Arrow's, and the"
        " variable-width column really varies"
    )

    # ── the refusals ──────────────────────────────────────────────────
    var refused = 0

    # A column staged with the wrong number of values.
    var raised = False
    try:
        var w2 = ParquetWriter(_columns())
        var v2 = w2.new_values()
        v2[0].push_i64(1)
        w2.write_row_group(v2, 2)  # 1 value for 2 rows
    except:
        raised = True
    if not raised:
        raise Error("a short column was ACCEPTED")
    refused += 1

    # A file with no row groups at all.
    raised = False
    try:
        var w3 = ParquetWriter(_columns())
        _ = w3.close(String("/tmp/mojo_rl_parquet_empty.parquet"))
    except:
        raised = True
    if not raised:
        raise Error("an empty file was ACCEPTED")
    refused += 1

    print("  refusals: " + String(refused) + "/2 raised")
    print("[PASS] parquet-write")


def _same_f64(
    name: String, ref a: List[Float64], ref b: List[Float64]
) raises -> Int:
    if len(a) != len(b):
        raise Error(
            name + ": golden has " + String(len(a)) + " values, ours "
            + String(len(b))
        )
    if len(a) == 0:
        raise Error(name + ": nothing to compare")
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error(
                name + "[" + String(i) + "]: golden " + String(a[i])
                + ", ours " + String(b[i])
            )
    return len(a)


def _same_i64(
    name: String, ref a: List[Int64], ref b: List[Int64]
) raises -> Int:
    if len(a) != len(b):
        raise Error(
            name + ": golden has " + String(len(a)) + " values, ours "
            + String(len(b))
        )
    if len(a) == 0:
        raise Error(name + ": nothing to compare")
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error(
                name + "[" + String(i) + "]: golden " + String(a[i])
                + ", ours " + String(b[i])
            )
    return len(a)
