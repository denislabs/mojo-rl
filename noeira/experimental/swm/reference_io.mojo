"""Reader for the pinned Phase 0 oracle (`tests/experimental/swm/reference/`).

The oracle is a whitespace table: one record per line, a string key followed by
numbers. Deliberately dumb — the point is that this parser cannot be where a
gate goes wrong.

Why the oracle exists at all: numpy's PCG64 draws are not reproducible in Mojo,
so a gate that only re-ran the same *generative process* here could compare
statistics and nothing more. Instead the oracle carries the exact observation
pairs numpy saw, so the Mojo fit is required to land on numpy's answer. numpy
gets there by SVD, this tree gets there by Newton polar decomposition
(`procrustes.mojo`) — two implementations, one answer, so a shared bug cannot
make the gate pass.
"""

from std.math import abs


@fieldwise_init
struct RefRow(Copyable, Movable):
    """One record: a key and its numeric tail."""

    var key: String
    var nums: List[Float64]


def load_reference(path: String) raises -> List[RefRow]:
    """Parse a reference table. Blank lines and `#` comments are skipped."""
    var text: String
    with open(path, "r") as f:
        text = f.read()

    var rows = List[RefRow]()
    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li]).strip()
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var fields = line.split(" ")
        if len(fields) == 0:
            continue
        var key = String(fields[0])
        var nums = List[Float64]()
        for fi in range(1, len(fields)):
            var tok = String(fields[fi]).strip()
            if tok.byte_length() == 0:
                continue
            nums.append(atof(tok))
        rows.append(RefRow(key, nums^))
    if len(rows) == 0:
        raise Error("load_reference: no records in " + path)
    return rows^


def ref_scalar(rows: List[RefRow], key: String) raises -> Float64:
    """The single number on the row named `key`."""
    for i in range(len(rows)):
        if rows[i].key == key and len(rows[i].nums) >= 1:
            return rows[i].nums[0]
    raise Error("ref_scalar: key not found: " + key)


def ref_int(rows: List[RefRow], key: String) raises -> Int:
    return Int(ref_scalar(rows, key))


def ref_indexed(
    rows: List[RefRow], key: String, index: Int
) raises -> List[Float64]:
    """Numbers after the leading index on `key <index> ...`."""
    for i in range(len(rows)):
        if rows[i].key != key or len(rows[i].nums) < 1:
            continue
        if Int(rows[i].nums[0]) != index:
            continue
        var out = List[Float64]()
        for j in range(1, len(rows[i].nums)):
            out.append(rows[i].nums[j])
        return out^
    raise Error("ref_indexed: not found: " + key + " " + String(index))


def ref_vector(rows: List[RefRow], key: String) raises -> List[Float64]:
    """All numbers on the row named `key` (e.g. a flattened matrix)."""
    for i in range(len(rows)):
        if rows[i].key == key:
            return rows[i].nums.copy()
    raise Error("ref_vector: key not found: " + key)


def ref_count(rows: List[RefRow], key: String) -> Int:
    """How many records carry this key — the vacuity guard for a gate loop."""
    var n = 0
    for i in range(len(rows)):
        if rows[i].key == key:
            n += 1
    return n
