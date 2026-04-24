"""CIFAR-10 loader — Python urllib+tarfile for download, Mojo for binary parse.

First run downloads https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
(~162 MB), extracts to ~/.cache/mojo_rl/cifar10/cifar-10-batches-bin/, and
the 6 raw .bin files (5 train batches + 1 test) are then read natively by
Mojo via open(path, "r").read_bytes().

Binary format (per file, 10000 samples):
    sample_i = [1 byte label][1024 R][1024 G][1024 B]  = 3073 bytes
    R/G/B are row-major 32×32 per channel.

The on-disk channel-major layout (R-then-G-then-B) already matches the
(channels, h, w) flat layout Conv2D expects for IN_DIM = 3 * 32 * 32, so
we copy pixels as-is after normalizing to Float32 / 255.
"""

from std.python import Python, PythonObject


comptime _CIFAR_URL = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
)
comptime _EXTRACTED_DIR = "cifar-10-batches-bin"
comptime _TEST_FILE = "test_batch.bin"
comptime _SAMPLES_PER_FILE = 10000
comptime _BYTES_PER_SAMPLE = 1 + 3 * 32 * 32  # 1 label + 3072 pixels


def _train_filename(idx: Int) -> String:
    """data_batch_1.bin .. data_batch_5.bin indexed 0..4."""
    return "data_batch_" + String(idx + 1) + ".bin"


def _cache_dir() raises -> String:
    var os = Python.import_module("os")
    var home = String(os.path.expanduser(PythonObject("~")))
    var path = home + "/.cache/mojo_rl/cifar10"
    _ = os.makedirs(PythonObject(path), exist_ok=True)
    return path


def _ensure_extracted() raises -> String:
    """Ensure the 6 .bin files are extracted. Returns the dir containing them.
    """
    var os = Python.import_module("os")
    var cache = _cache_dir()
    var extracted_path = cache + "/" + _EXTRACTED_DIR

    # Consider extraction complete if test_batch.bin exists at expected size
    var test_path = extracted_path + "/" + _TEST_FILE
    if Bool(os.path.exists(PythonObject(test_path))):
        return extracted_path

    var tar_path = cache + "/cifar-10-binary.tar.gz"
    if not Bool(os.path.exists(PythonObject(tar_path))):
        print("  [cifar10] downloading " + _CIFAR_URL + " (~162 MB)")
        var urllib_request = Python.import_module("urllib.request")
        var builtins = Python.import_module("builtins")
        var resp = urllib_request.urlopen(
            PythonObject(_CIFAR_URL), timeout=120
        )
        var data = resp.read()
        _ = resp.close()
        var f = builtins.open(PythonObject(tar_path), PythonObject("wb"))
        _ = f.write(data)
        _ = f.close()
        print("  [cifar10] download complete -> " + tar_path)

    print("  [cifar10] extracting " + tar_path)
    var tarfile_mod = Python.import_module("tarfile")
    var tar = tarfile_mod.open(PythonObject(tar_path))
    _ = tar.extractall(PythonObject(cache))
    _ = tar.close()
    print("  [cifar10] extracted -> " + extracted_path)
    return extracted_path


def _load_batch(
    path: String,
    dst_images: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    dst_labels: UnsafePointer[Int32, MutAnyOrigin],
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
    var inv = Scalar[DType.float32](1.0) / Scalar[DType.float32](255.0)
    for i in range(_SAMPLES_PER_FILE):
        var base = i * _BYTES_PER_SAMPLE
        dst_labels[offset_samples + i] = Int32(Int(bytes[base]))
        var img_dst = (offset_samples + i) * (3 * 32 * 32)
        for p in range(3 * 32 * 32):
            dst_images[img_dst + p] = (
                Scalar[DType.float32](Int(bytes[base + 1 + p])) * inv
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
                train_images.unsafe_ptr(),
                train_labels.unsafe_ptr(),
                i * _SAMPLES_PER_FILE,
            )

        # Load test batch
        var test_path = extracted + "/" + _TEST_FILE
        _load_batch(
            test_path,
            test_images.unsafe_ptr(),
            test_labels.unsafe_ptr(),
            0,
        )

        self.train_images = train_images^
        self.train_labels = train_labels^
        self.test_images = test_images^
        self.test_labels = test_labels^
        self.num_train = Self.N_TRAIN
        self.num_test = Self.N_TEST

    def __init__(out self, *, deinit take: Self):
        self.train_images = take.train_images^
        self.train_labels = take.train_labels^
        self.test_images = take.test_images^
        self.test_labels = take.test_labels^
        self.num_train = take.num_train
        self.num_test = take.num_test
