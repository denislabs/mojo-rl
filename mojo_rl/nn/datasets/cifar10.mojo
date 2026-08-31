"""CIFAR-10 loader — no Python anywhere.

First run materializes the 6 canonical binary batch files
(data_batch_1..5.bin + test_batch.bin) under
~/.cache/mojo_rl/cifar10/cifar-10-batches-bin/, then Mojo reads them natively
via open(path, "r").read_bytes().

Source (in priority order):
  1. Already-present .bin files → used as-is.
  2. A binary tarball — if ~/.cache/.../cifar-10-binary.tar.gz already exists,
     or env `CIFAR10_BIN_URL` is set (a mirror of cifar-10-binary.tar.gz).
     Fetched resumably by `io/fetch.mojo`, inflated by `io/http.gunzip_file`
     and unpacked by `io/tar.mojo`.
  3. DEFAULT: the HuggingFace parquet dataset `uoft-cs/cifar10` (reliable CDN;
     the toronto.edu origin is slow — measured ~60 KB/s, so the 162 MB tarball
     is a 45-minute first run). Downloaded by `io/hf.mojo`, read by
     `io/parquet`, its PNG-encoded images decoded by `io/png.mojo`, and written
     out as the 6 .bin files in the exact binary layout below — so the Mojo
     parser below is unchanged whichever path produced them.

⚠ THIS LOADER USED FOUR PYTHON PACKAGES: `huggingface_hub`, `pyarrow`, `PIL`
and `numpy`, the last three inside a SUBPROCESS because libarrow cannot safely
load in Mojo's embedded interpreter. All four are gone. `pixi run build-http`
is the one prerequisite (the shim carries libcurl and zlib).

Binary format (per file, 10000 samples):
    sample_i = [1 byte label][1024 R][1024 G][1024 B]  = 3073 bytes
    R/G/B are row-major 32×32 per channel.

The on-disk channel-major layout (R-then-G-then-B) already matches the
(channels, h, w) flat layout Conv2D expects for IN_DIM = 3 * 32 * 32, so
we copy pixels as-is after normalizing to Float32 / 255.
"""

from std.os import getenv, makedirs
from std.os.path import exists

from mojo_rl.io.fetch import fetch_to_cache
from mojo_rl.io.fileio import remove_file
from mojo_rl.io.hf import HF_DATASET, hf_download_file, mojo_rl_cache
from mojo_rl.io.http import gunzip_file
from mojo_rl.io.parquet import ParquetFile
from mojo_rl.io.png import decode_png
from mojo_rl.io.tar import untar
from mojo_rl.nn.core.ptr import mptr


comptime _CIFAR_URL = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
)
comptime _EXTRACTED_DIR = "cifar-10-batches-bin"
comptime _HF_REPO = "uoft-cs/cifar10"
comptime _HF_TRAIN = "plain_text/train-00000-of-00001.parquet"
comptime _HF_TEST = "plain_text/test-00000-of-00001.parquet"
comptime _TEST_FILE = "test_batch.bin"

comptime _SAMPLES_PER_FILE = 10000
comptime _BYTES_PER_SAMPLE = 1 + 3 * 32 * 32  # 1 label + 3072 pixels

# Canonical CIFAR-10 per-channel normalization constants (PyTorch convention).
# Applied during load: normalized = (raw / 255 - mean[c]) / std[c]
# Keeps values roughly in [-2, 2], stabilizes training + BN.
comptime _MEAN_R: Float32 = 0.4914
comptime _MEAN_G: Float32 = 0.4822
comptime _MEAN_B: Float32 = 0.4465
comptime _STD_R: Float32 = 0.2470
comptime _STD_G: Float32 = 0.2435
comptime _STD_B: Float32 = 0.2616


def _train_filename(idx: Int) -> String:
    """data_batch_1.bin .. data_batch_5.bin indexed 0..4."""
    return "data_batch_" + String(idx + 1) + ".bin"


def _cache_dir() raises -> String:
    var path = mojo_rl_cache() + "/cifar10"
    makedirs(path, exist_ok=True)
    return path^


def _write_bins_from_parquet(
    parquet_path: String, out_dir: String, names: List[String]
) raises -> Int:
    """Decode a parquet of PNG images into the canonical `.bin` layout.

    `names` receives `_SAMPLES_PER_FILE` samples each, in file order — five
    names for the training parquet, one for the test parquet.

    ⚠ THE LAYOUT IS CHANNEL-MAJOR AND THE PNG IS NOT. A decoded PNG is HWC
    (`R G B R G B ...`); the `.bin` format is one label byte then 1024 R, 1024
    G, 1024 B. Writing the HWC bytes straight out produces a file of exactly
    the right size, full of colour noise, that every size check accepts.
    """
    var f = ParquetFile(parquet_path)
    var pngs = List[UInt8]()
    var offs = List[Int]()
    var n = f.read_byte_arrays(String("img.bytes"), pngs, offs)
    var labels = f.read_i64(String("label"))
    if len(labels) != n:
        raise Error(
            "cifar10: " + parquet_path + " has " + String(n) + " images but "
            + String(len(labels)) + " labels"
        )
    if n != len(names) * _SAMPLES_PER_FILE:
        raise Error(
            "cifar10: " + parquet_path + " holds " + String(n) + " images, the"
            " canonical layout needs " + String(len(names) * _SAMPLES_PER_FILE)
        )

    comptime CHAN = 32 * 32
    var written = 0
    for b in range(len(names)):
        var buf = List[UInt8]()
        buf.resize(_SAMPLES_PER_FILE * _BYTES_PER_SAMPLE, 0)
        for k in range(_SAMPLES_PER_FILE):
            var idx = b * _SAMPLES_PER_FILE + k
            var one = List[UInt8]()
            for j in range(offs[idx], offs[idx + 1]):
                one.append(pngs[j])
            var img = decode_png(one)
            if img.width != 32 or img.height != 32 or img.channels != 3:
                raise Error(
                    "cifar10: image " + String(idx) + " is " + String(img.width)
                    + "x" + String(img.height) + "x" + String(img.channels)
                    + ", expected 32x32x3"
                )
            var base = k * _BYTES_PER_SAMPLE
            buf[base] = UInt8(Int(labels[idx]) & 0xFF)
            for p in range(CHAN):
                buf[base + 1 + p] = img.pixels[p * 3]
                buf[base + 1 + CHAN + p] = img.pixels[p * 3 + 1]
                buf[base + 1 + 2 * CHAN + p] = img.pixels[p * 3 + 2]
        var out_path = out_dir + "/" + names[b]
        with open(out_path, "w") as o:
            o.write_bytes(Span(buf))
        written += _SAMPLES_PER_FILE
        print("  [cifar10] wrote " + names[b])
    return written


def _ensure_extracted() raises -> String:
    """Ensure the 6 .bin files exist. Returns the dir containing them.

    Priority: (1) already-present .bin files; (2) a binary tarball if one is
    already cached or `CIFAR10_BIN_URL` is set; (3) DEFAULT — the HuggingFace
    parquet dataset, decoded to the 6 .bin files.
    """
    var cache = _cache_dir()
    var extracted_path = cache + "/" + String(_EXTRACTED_DIR)

    # 1. already materialized?
    if exists(extracted_path + "/" + String(_TEST_FILE)):
        return extracted_path^

    makedirs(extracted_path, exist_ok=True)

    # 2. the binary tarball — only when one is already cached or a mirror URL
    #    is given, because the canonical origin is slow (see the module note).
    var tar_gz = cache + "/cifar-10-binary.tar.gz"
    var url_env = getenv("CIFAR10_BIN_URL")
    if exists(tar_gz) or url_env != "":
        if not exists(tar_gz):
            var url = url_env if url_env != "" else String(_CIFAR_URL)
            print("  [cifar10] downloading " + url + " (~162 MB)")
            _ = fetch_to_cache(url, tar_gz, String(""), 0, String("cifar10"))
        print("  [cifar10] extracting " + tar_gz)
        var tar_path = cache + "/cifar-10-binary.tar"
        _ = gunzip_file(tar_gz, tar_path)
        # The archive carries its own `cifar-10-batches-bin/` directory, so it
        # unpacks into `cache`, not into `extracted_path`.
        _ = untar(tar_path, cache)
        remove_file(tar_path)
        if not exists(extracted_path + "/" + String(_TEST_FILE)):
            raise Error(
                "cifar10: " + tar_gz + " did not contain "
                + String(_EXTRACTED_DIR) + "/" + String(_TEST_FILE)
            )
        print("  [cifar10] extracted -> " + extracted_path)
        return extracted_path^

    # 3. default: the HuggingFace parquet, decoded to the 6 .bin files.
    print("  [cifar10] preparing from HuggingFace parquet (" + String(_HF_REPO) + ")")
    var train_pq = hf_download_file(
        String(_HF_REPO), String(_HF_TRAIN), HF_DATASET,
        cache + "/train.parquet",
    )
    var test_pq = hf_download_file(
        String(_HF_REPO), String(_HF_TEST), HF_DATASET,
        cache + "/test.parquet",
    )

    var train_names = List[String]()
    for i in range(5):
        train_names.append(_train_filename(i))
    _ = _write_bins_from_parquet(train_pq, extracted_path, train_names)

    var test_names = List[String]()
    test_names.append(String(_TEST_FILE))
    _ = _write_bins_from_parquet(test_pq, extracted_path, test_names)

    print("  [cifar10] ready -> " + extracted_path)
    return extracted_path^


def _load_batch(
    path: String,
    dst_images: Pointer[Scalar[DType.float32], MutAnyOrigin],
    dst_labels: Pointer[Int32, MutAnyOrigin],
    offset_samples: Int,
) raises:
    """Read one .bin file (10000 samples) into dst_images/dst_labels at
    offset_samples.
    """
    var bytes = open(path, "r").read_bytes()
    var expected = _SAMPLES_PER_FILE * _BYTES_PER_SAMPLE
    if len(bytes) != expected:
        raise Error(
            "CIFAR file wrong size: "
            + path
            + " expected "
            + String(expected)
            + " got "
            + String(len(bytes))
        )
    var inv255 = Scalar[DType.float32](1.0) / Scalar[DType.float32](255.0)
    var inv_std_r = Scalar[DType.float32](1.0) / _STD_R
    var inv_std_g = Scalar[DType.float32](1.0) / _STD_G
    var inv_std_b = Scalar[DType.float32](1.0) / _STD_B
    comptime CHAN_SZ = 32 * 32
    for i in range(_SAMPLES_PER_FILE):
        var base = i * _BYTES_PER_SAMPLE
        dst_labels[unsafe_offset=offset_samples + i] = Int32(Int(bytes[base]))
        var img_dst = (offset_samples + i) * (3 * 32 * 32)
        # Channel R (first 1024 bytes)
        for p in range(CHAN_SZ):
            var raw = Scalar[DType.float32](Int(bytes[base + 1 + p])) * inv255
            dst_images[unsafe_offset=img_dst + p] = (raw - _MEAN_R) * inv_std_r
        # Channel G (next 1024 bytes)
        for p in range(CHAN_SZ):
            var raw = (
                Scalar[DType.float32](Int(bytes[base + 1 + CHAN_SZ + p]))
                * inv255
            )
            dst_images[unsafe_offset=img_dst + CHAN_SZ + p] = (raw - _MEAN_G) * inv_std_g
        # Channel B (last 1024 bytes)
        for p in range(CHAN_SZ):
            var raw = (
                Scalar[DType.float32](
                    Int(bytes[base + 1 + 2 * CHAN_SZ + p])
                )
                * inv255
            )
            dst_images[unsafe_offset=img_dst + 2 * CHAN_SZ + p] = (
                (raw - _MEAN_B) * inv_std_b
            )


struct CIFAR10(Movable):
    """Loaded CIFAR-10 dataset. Images flat [N, 3*32*32] channel-major
    (R-G-B blocks of 1024 each) normalized to [0, 1]."""

    comptime IMG_C: Int = 3
    comptime IMG_H: Int = 32
    comptime IMG_W: Int = 32
    comptime IMG_SIZE: Int = 3 * 32 * 32
    comptime NUM_CLASSES: Int = 10
    comptime N_TRAIN: Int = 50000
    comptime N_TEST: Int = 10000

    var train_images: List[Scalar[DType.float32]]
    var train_labels: List[Int32]
    var test_images: List[Scalar[DType.float32]]
    var test_labels: List[Int32]
    var num_train: Int
    var num_test: Int

    def __init__(out self) raises:
        var extracted = _ensure_extracted()

        # Pre-allocate
        var train_images = List[Scalar[DType.float32]](
            length=Self.N_TRAIN * Self.IMG_SIZE,
            fill=Scalar[DType.float32](0.0),
        )
        var train_labels = List[Int32](length=Self.N_TRAIN, fill=Int32(0))
        var test_images = List[Scalar[DType.float32]](
            length=Self.N_TEST * Self.IMG_SIZE,
            fill=Scalar[DType.float32](0.0),
        )
        var test_labels = List[Int32](length=Self.N_TEST, fill=Int32(0))

        # Load 5 training batches (10k samples each)
        for i in range(5):
            var path = extracted + "/" + _train_filename(i)
            _load_batch(
                path,
                mptr(train_images.unsafe_ptr()),
                mptr(train_labels.unsafe_ptr()),
                i * _SAMPLES_PER_FILE,
            )

        # Load test batch
        var test_path = extracted + "/" + _TEST_FILE
        _load_batch(
            test_path,
            mptr(test_images.unsafe_ptr()),
            mptr(test_labels.unsafe_ptr()),
            0,
        )

        self.train_images = train_images^
        self.train_labels = train_labels^
        self.test_images = test_images^
        self.test_labels = test_labels^
        self.num_train = Self.N_TRAIN
        self.num_test = Self.N_TEST

    def __init__(out self, *, deinit move: Self):
        self.train_images = move.train_images^
        self.train_labels = move.train_labels^
        self.test_images = move.test_images^
        self.test_labels = move.test_labels^
        self.num_train = move.num_train
        self.num_test = move.num_test
