"""MNIST loader — no Python anywhere.

First run downloads the 4 MNIST IDX files over `io/http.mojo` (libcurl),
inflates them with the same shim's zlib entry point, and caches the raw bytes
to ~/.cache/mojo_rl/mnist/*.ubyte. Subsequent runs read the cached
uncompressed files directly through Mojo's binary file I/O.

⚠ THE `.gz` IS NOT A `Content-Encoding`. libcurl decodes the header form on
its own and cannot decode this one, where the compression is announced by the
file extension — hence the explicit `gunzip`.

IDX format: 4-byte magic, 4-byte big-endian num_items, then either
  - images: 4-byte rows, 4-byte cols, num*rows*cols uint8 pixels
  - labels: num uint8 class ids

Pixels are normalized to [0, 1] by dividing by 255.
"""

from std.os import makedirs
from std.os.path import exists

from mojo_rl.io.fileio import write_file_atomic
from mojo_rl.io.hf import mojo_rl_cache
from mojo_rl.io.http import gunzip, http_get_bytes


comptime _MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
comptime _FN_TRAIN_IMG = "train-images-idx3-ubyte"
comptime _FN_TRAIN_LBL = "train-labels-idx1-ubyte"
comptime _FN_TEST_IMG = "t10k-images-idx3-ubyte"
comptime _FN_TEST_LBL = "t10k-labels-idx1-ubyte"


def _be_u32(bytes: List[UInt8], offset: Int) -> Int:
    """Read a 4-byte big-endian unsigned int from a byte list."""
    return (
        (Int(bytes[offset]) << 24)
        | (Int(bytes[offset + 1]) << 16)
        | (Int(bytes[offset + 2]) << 8)
        | Int(bytes[offset + 3])
    )


def _cache_dir() raises -> String:
    """Resolve ~/.cache/mojo_rl/mnist and ensure it exists."""
    var path = mojo_rl_cache() + "/mnist"
    makedirs(path, exist_ok=True)
    return path^


def _ensure_downloaded(filename: String) raises -> String:
    """Ensure decompressed MNIST file is present in cache. Returns its path."""
    var cache = _cache_dir()
    var dst_path = cache + "/" + filename

    if exists(dst_path):
        return dst_path^

    var url = String(_MNIST_MIRROR) + filename + ".gz"
    print("  [mnist] downloading " + url)

    var compressed = http_get_bytes(url)
    var raw = gunzip(compressed)
    # Atomic: a crash mid-write must not leave a short .ubyte that the next
    # run reads as a complete one.
    write_file_atomic(dst_path, raw)
    print("  [mnist] cached -> " + dst_path)
    return dst_path^


struct MNIST(Movable):
    """Loaded MNIST dataset. Images flattened [N, 784], normalized to [0, 1]."""

    comptime IMG_H: Int = 28
    comptime IMG_W: Int = 28
    comptime IMG_SIZE: Int = 28 * 28
    comptime NUM_CLASSES: Int = 10
    comptime N_TRAIN: Int = 60000
    comptime N_TEST: Int = 10000

    var train_images: List[Scalar[DType.float32]]
    var train_labels: List[Int32]
    var test_images: List[Scalar[DType.float32]]
    var test_labels: List[Int32]
    var num_train: Int
    var num_test: Int

    def __init__(out self) raises:
        var train_img_path = _ensure_downloaded(_FN_TRAIN_IMG)
        var train_lbl_path = _ensure_downloaded(_FN_TRAIN_LBL)
        var test_img_path = _ensure_downloaded(_FN_TEST_IMG)
        var test_lbl_path = _ensure_downloaded(_FN_TEST_LBL)

        # ── train images ──
        var train_img_bytes = open(train_img_path, "r").read_bytes()
        if len(train_img_bytes) < 16:
            raise Error("train image file too short")
        var ti_magic = _be_u32(train_img_bytes, 0)
        if ti_magic != 0x00000803:
            raise Error("train image bad magic: " + String(ti_magic))
        var n_train = _be_u32(train_img_bytes, 4)
        var ti_rows = _be_u32(train_img_bytes, 8)
        var ti_cols = _be_u32(train_img_bytes, 12)
        if ti_rows != Self.IMG_H or ti_cols != Self.IMG_W:
            raise Error("train images not 28x28")
        var train_px = n_train * Self.IMG_SIZE
        if len(train_img_bytes) < 16 + train_px:
            raise Error("train image file truncated")
        var train_images = List[Scalar[DType.float32]](capacity=train_px)
        var inv = Scalar[DType.float32](1.0) / Scalar[DType.float32](255.0)
        for i in range(train_px):
            train_images.append(
                Scalar[DType.float32](Int(train_img_bytes[16 + i])) * inv
            )

        # ── train labels ──
        var train_lbl_bytes = open(train_lbl_path, "r").read_bytes()
        if len(train_lbl_bytes) < 8:
            raise Error("train label file too short")
        var tl_magic = _be_u32(train_lbl_bytes, 0)
        if tl_magic != 0x00000801:
            raise Error("train label bad magic: " + String(tl_magic))
        var n_train_lbl = _be_u32(train_lbl_bytes, 4)
        if n_train_lbl != n_train:
            raise Error("train image/label count mismatch")
        var train_labels = List[Int32](capacity=n_train_lbl)
        for i in range(n_train_lbl):
            train_labels.append(Int32(Int(train_lbl_bytes[8 + i])))

        # ── test images ──
        var test_img_bytes = open(test_img_path, "r").read_bytes()
        if len(test_img_bytes) < 16:
            raise Error("test image file too short")
        var xi_magic = _be_u32(test_img_bytes, 0)
        if xi_magic != 0x00000803:
            raise Error("test image bad magic: " + String(xi_magic))
        var n_test = _be_u32(test_img_bytes, 4)
        var xi_rows = _be_u32(test_img_bytes, 8)
        var xi_cols = _be_u32(test_img_bytes, 12)
        if xi_rows != Self.IMG_H or xi_cols != Self.IMG_W:
            raise Error("test images not 28x28")
        var test_px = n_test * Self.IMG_SIZE
        if len(test_img_bytes) < 16 + test_px:
            raise Error("test image file truncated")
        var test_images = List[Scalar[DType.float32]](capacity=test_px)
        for i in range(test_px):
            test_images.append(
                Scalar[DType.float32](Int(test_img_bytes[16 + i])) * inv
            )

        # ── test labels ──
        var test_lbl_bytes = open(test_lbl_path, "r").read_bytes()
        if len(test_lbl_bytes) < 8:
            raise Error("test label file too short")
        var xl_magic = _be_u32(test_lbl_bytes, 0)
        if xl_magic != 0x00000801:
            raise Error("test label bad magic: " + String(xl_magic))
        var n_test_lbl = _be_u32(test_lbl_bytes, 4)
        if n_test_lbl != n_test:
            raise Error("test image/label count mismatch")
        var test_labels = List[Int32](capacity=n_test_lbl)
        for i in range(n_test_lbl):
            test_labels.append(Int32(Int(test_lbl_bytes[8 + i])))

        self.train_images = train_images^
        self.train_labels = train_labels^
        self.test_images = test_images^
        self.test_labels = test_labels^
        self.num_train = n_train
        self.num_test = n_test

    def __init__(out self, *, deinit move: Self):
        self.train_images = move.train_images^
        self.train_labels = move.train_labels^
        self.test_images = move.test_images^
        self.test_labels = move.test_labels^
        self.num_train = move.num_train
        self.num_test = move.num_test
