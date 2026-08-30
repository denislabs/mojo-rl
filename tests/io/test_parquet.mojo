# +--------------------------------------------------------------------------+ #
# | The native Parquet reader, against pyarrow
# +--------------------------------------------------------------------------+ #
"""Gates `mojo_rl/io/parquet` value-for-value on a corpus pyarrow wrote.

    pixi run python tools/io/dump_parquet_reference.py --out /tmp/pq_ref
    pixi run mojo run -I . tests/io/test_parquet.mojo /tmp/pq_ref

Every value of every column of every case is compared — floats BIT-EXACTLY,
since decoding is a byte reinterpretation and any tolerance here would hide a
real defect rather than absorb rounding.

⚠ A DECODER'S DEFAULT FAILURE IS SILENCE. A mis-sliced level stream, an
off-by-one in a bit-packed run, an overlapping Snappy copy done with a block
move — none of them raise, they all return the right COUNT of plausible
numbers. So this file checks two things beyond equality: that the case list
was actually found (a missing dump directory would otherwise pass with zero
comparisons), and that the number of values compared is non-zero and matches
the expectation line. `values compared` is printed for every column so a
vacuous run is visible rather than green.

See `tools/io/dump_parquet_reference.py` for what each case turns off.
"""

from std.os.path import exists
from std.sys import argv

from mojo_rl.io.parquet import ParquetFile


comptime DEFAULT_REF = "/tmp/pq_ref"
comptime SENTINEL = "/expected.txt"
comptime GEN = (
    "pixi run python tools/io/dump_parquet_reference.py --out /tmp/pq_ref"
)

comptime USAGE = (
    "usage: mojo run -I . tests/io/test_parquet.mojo <ref_dir>\n"
    "  generate <ref_dir> with:\n"
    "    pixi run python tools/io/dump_parquet_reference.py --out <ref_dir>"
)


@fieldwise_init
struct Expect(Copyable, Movable):
    var case_name: String
    var column: String
    var physical: String
    var values: List[String]


def parse_expected(path: String) raises -> List[Expect]:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    var out = List[Expect]()
    for line in text.splitlines():
        var s = String(line)
        if s == "" or s.startswith("#"):
            continue
        var parts = s.split(" ")
        if len(parts) < 3:
            raise Error("malformed expectation line: " + s)
        var count = atol(parts[3])
        var vals = List[String]()
        for i in range(4, 4 + count):
            vals.append(String(parts[i]))
        out.append(
            Expect(
                String(parts[0]), String(parts[1]), String(parts[2]), vals^
            )
        )
    return out^


def main() raises:
    var args = argv()
    var ref_dir = String(DEFAULT_REF) if len(args) < 2 else String(args[1])
    if len(args) > 2:
        print(USAGE)
        raise Error("test_parquet: expected at most one argument")
    if not exists(ref_dir + SENTINEL):
        raise Error(
            "no reference dump at " + ref_dir + " — generate it with:\n    "
            + GEN
        )

    print("Native Parquet reader vs pyarrow")
    print("  reference dump: " + ref_dir)
    print("")

    var expects = parse_expected(ref_dir + "/expected.txt")
    if len(expects) == 0:
        raise Error(
            "gate: the expectation file declares no columns — nothing would"
            " have been compared"
        )

    var fails = 0
    var total_values = 0
    var open_case = String("")
    var pf = ParquetFile(ref_dir + "/" + expects[0].case_name + ".parquet")

    for ei in range(len(expects)):
        ref e = expects[ei]
        if e.case_name != open_case:
            pf = ParquetFile(ref_dir + "/" + e.case_name + ".parquet")
            open_case = String(e.case_name)
            print("  " + e.case_name + ".parquet  (" + String(pf.num_rows())
                  + " rows, " + String(len(pf.column_names())) + " columns)")

        var n_expect = len(e.values)
        var mismatch = 0
        var first_bad = -1
        var detail = String("")

        if e.physical == "FLOAT" or e.physical == "DOUBLE":
            var got = pf.read_f64(e.column)
            if len(got) != n_expect:
                fails += 1
                print(
                    "    FAIL  " + e.column + ": " + String(len(got))
                    + " values, expected " + String(n_expect)
                )
                continue
            for i in range(n_expect):
                var want = atof(e.values[i])
                if got[i] != want:
                    mismatch += 1
                    if first_bad < 0:
                        first_bad = i
                        detail = (
                            " at " + String(i) + ": got " + String(got[i])
                            + ", want " + String(want)
                        )
        else:
            var got = pf.read_i64(e.column)
            if len(got) != n_expect:
                fails += 1
                print(
                    "    FAIL  " + e.column + ": " + String(len(got))
                    + " values, expected " + String(n_expect)
                )
                continue
            for i in range(n_expect):
                var want = Int64(atol(e.values[i]))
                if got[i] != want:
                    mismatch += 1
                    if first_bad < 0:
                        first_bad = i
                        detail = (
                            " at " + String(i) + ": got " + String(got[i])
                            + ", want " + String(want)
                        )

        total_values += n_expect
        if mismatch == 0:
            print(
                "    PASS  " + e.column + " [" + e.physical + "]  "
                + String(n_expect) + " values compared"
            )
        else:
            fails += 1
            print(
                "    FAIL  " + e.column + " [" + e.physical + "]  "
                + String(mismatch) + " of " + String(n_expect)
                + " differ" + detail
            )

    print("")
    print("  " + String(total_values) + " values compared over "
          + String(len(expects)) + " columns")
    if total_values == 0:
        raise Error("gate: zero values compared — this run proved nothing")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("parquet reader gate failed")
