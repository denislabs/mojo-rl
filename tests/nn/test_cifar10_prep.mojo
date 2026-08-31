# +--------------------------------------------------------------------------+ #
# | CIFAR-10 prep vs pyarrow + Pillow
# +--------------------------------------------------------------------------+ #
"""Gate the native CIFAR-10 preparation chain, byte for byte.

    pixi run build-http                                          # ONCE
    pixi run python tools/io/dump_cifar_reference.py --out /tmp/cifar_ref
    pixi run mojo run -I . tests/nn/test_cifar10_prep.mojo [/tmp/cifar_ref]

`nn/datasets/cifar10.mojo` used to need FOUR Python packages —
`huggingface_hub`, `pyarrow`, `PIL` and `numpy`, the last three in a
subprocess because libarrow cannot safely load inside Mojo's embedded
interpreter. This gate is what makes removing them checkable: the same
parquet, read by `io/parquet`'s BYTE_ARRAY path, decoded by `io/png.mojo`,
laid out by the loader's own rule, against a `.bin` that `pyarrow` + `PIL`
produced.

⚠ ONE COMPARISON COVERS THREE IMPLEMENTATIONS. A byte difference could come
from the parquet reader (a dictionary-encoded BYTE_ARRAY column decoded
wrong), the PNG decoder (any of the five scanline filters), or the layout (HWC
written where channel-major belongs). All three produce a file of exactly the
right SIZE, which is why the size check alone is worth nothing and every byte
is compared.

⚠ Reported as bytes-COMPARED beside bytes-DIFFERING. "0 mismatches" over a
file that failed to load is the failure mode this arrangement exists to make
visible.
"""

from std.os.path import exists
from std.sys import argv

from mojo_rl.io.fileio import file_size, read_file_bytes
from mojo_rl.io.parquet import ParquetFile
from mojo_rl.io.png import decode_png


comptime N = 10000
comptime BYTES_PER_SAMPLE = 1 + 3 * 32 * 32
comptime CHAN = 32 * 32


def main() raises:
    var root = String("/tmp/cifar_ref")
    var args = argv()
    if len(args) > 1:
        root = String(args[1])

    var pq_path = root + "/test.parquet"
    var ref_path = root + "/test_batch_reference.bin"
    if not exists(pq_path) or not exists(ref_path):
        raise Error(
            "no reference dump at " + root + " — generate it with:\n"
            "    pixi run python tools/io/dump_cifar_reference.py --out " + root
        )

    print("=== CIFAR-10 prep vs pyarrow + Pillow ===")

    # ── ours: parquet -> PNG -> the canonical layout ─────────────────
    var f = ParquetFile(pq_path)
    var pngs = List[UInt8]()
    var offs = List[Int]()
    var n = f.read_byte_arrays(String("img.bytes"), pngs, offs)
    var labels = f.read_i64(String("label"))
    print("  read " + String(n) + " PNG blobs (" + String(len(pngs))
          + " bytes) and " + String(len(labels)) + " labels")
    if n != N:
        raise Error(
            "the parquet holds " + String(n) + " images, expected " + String(N)
        )
    if len(labels) != N:
        raise Error("label count " + String(len(labels)) + " != " + String(N))

    var mine = List[UInt8]()
    mine.resize(N * BYTES_PER_SAMPLE, 0)
    for k in range(N):
        var one = List[UInt8]()
        for j in range(offs[k], offs[k + 1]):
            one.append(pngs[j])
        var img = decode_png(one)
        if img.width != 32 or img.height != 32 or img.channels != 3:
            raise Error(
                "image " + String(k) + " decoded to " + String(img.width) + "x"
                + String(img.height) + "x" + String(img.channels)
            )
        var base = k * BYTES_PER_SAMPLE
        mine[base] = UInt8(Int(labels[k]) & 0xFF)
        for p in range(CHAN):
            mine[base + 1 + p] = img.pixels[p * 3]
            mine[base + 1 + CHAN + p] = img.pixels[p * 3 + 1]
            mine[base + 1 + 2 * CHAN + p] = img.pixels[p * 3 + 2]

    # ── theirs ───────────────────────────────────────────────────────
    var want = read_file_bytes(ref_path)
    if len(want) != len(mine):
        raise Error(
            "the reference is " + String(len(want)) + " bytes, ours is "
            + String(len(mine))
        )

    var compared = 0
    var differing = 0
    var first_bad = -1
    for i in range(len(want)):
        compared += 1
        if want[i] != mine[i]:
            differing += 1
            if first_bad < 0:
                first_bad = i
    print("  " + String(compared) + " bytes compared, " + String(differing)
          + " differing")
    if compared != N * BYTES_PER_SAMPLE:
        raise Error("vacuous: compared " + String(compared) + " bytes")
    if differing != 0:
        var k = first_bad // BYTES_PER_SAMPLE
        var off = first_bad % BYTES_PER_SAMPLE
        raise Error(
            String(differing) + " bytes differ; the first is sample "
            + String(k) + " byte " + String(off) + " ("
            + ("the label" if off == 0 else "pixel " + String(off - 1))
            + "): ours " + String(Int(mine[first_bad])) + ", reference "
            + String(Int(want[first_bad]))
        )

    # ── the labels, separately, so a layout bug cannot hide one ──────
    var label_hist = List[Int]()
    label_hist.resize(10, 0)
    for k in range(N):
        var lab = Int(mine[k * BYTES_PER_SAMPLE])
        if lab < 0 or lab > 9:
            raise Error("sample " + String(k) + " has label " + String(lab))
        label_hist[lab] += 1
    for c in range(10):
        if label_hist[c] != 1000:
            raise Error(
                "class " + String(c) + " has " + String(label_hist[c])
                + " samples; the CIFAR-10 test split is 1000 of each"
            )
    print("  10 classes x 1000 samples")

    print("[PASS] CIFAR-10 prep")
