"""CIFAR-10 loader — Python for download/prep, Mojo for binary parse.

First run materializes the 6 canonical binary batch files
(data_batch_1..5.bin + test_batch.bin) under
~/.cache/mojo_rl/cifar10/cifar-10-batches-bin/, then Mojo reads them natively
via open(path, "r").read_bytes().

Source (in priority order):
  1. Already-present .bin files → used as-is.
  2. A binary tarball — if ~/.cache/.../cifar-10-binary.tar.gz already exists, or
     env `CIFAR10_BIN_URL` is set (a mirror of cifar-10-binary.tar.gz). Extracted
     with tarfile.
  3. DEFAULT: the HuggingFace parquet dataset `uoft-cs/cifar10` (reliable CDN;
     the toronto.edu origin is slow/flaky). Downloaded via huggingface_hub, the
     PNG-encoded images decoded with PIL, and written out as the 6 .bin files in
     the exact binary layout below — so the Mojo parser is unchanged.

Binary format (per file, 10000 samples):
    sample_i = [1 byte label][1024 R][1024 G][1024 B]  = 3073 bytes
    R/G/B are row-major 32×32 per channel.

The on-disk channel-major layout (R-then-G-then-B) already matches the
(channels, h, w) flat layout Conv2D expects for IN_DIM = 3 * 32 * 32, so
we copy pixels as-is after normalizing to Float32 / 255.
"""

from std.python import Python, PythonObject

from mojo_rl.nn.core.ptr import mptr


comptime _CIFAR_URL = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
)
comptime _EXTRACTED_DIR = "cifar-10-batches-bin"
comptime _TEST_FILE = "test_batch.bin"

# Python prep: HuggingFace parquet (uoft-cs/cifar10) → the 6 canonical .bin files.
# Run in a SUBPROCESS (via sys.executable -c) — pyarrow's libarrow crashes at
# teardown inside Mojo's embedded interpreter, so it must load in a child process
# only; the child reads `EXTRACTED_DIR` from the env, writes the .bin files, and
# exits. Images are written channel-major [label][1024 R][1024 G][1024 B],
# matching the Mojo binary parser.
comptime _PY_PREP_HF = """
import os, io
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image
import numpy as np

EXTRACTED_DIR = os.environ['CIFAR10_EXTRACTED_DIR']
os.makedirs(EXTRACTED_DIR, exist_ok=True)

# Resolve an HF token from the env, else from a .env file in the CWD (the same
# .env used for the logger, e.g. RL_MONITOR_API_KEY) — huggingface_hub reads
# HF_TOKEN from os.environ. Optional (the dataset is public); a token only lifts
# rate limits / speeds the CDN.
if not os.environ.get('HF_TOKEN') and not os.environ.get('HUGGING_FACE_HUB_TOKEN'):
    if os.path.exists('.env'):
        for _line in open('.env'):
            _line = _line.strip()
            if _line.startswith('export '):
                _line = _line[7:]
            if not _line or _line.startswith('#') or '=' not in _line:
                continue
            _k, _v = _line.split('=', 1)
            _v = _v.strip().strip('"').strip("'")
            if _k.strip() in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN') and _v:
                os.environ['HF_TOKEN'] = _v
                break

def _write_split(parquet_name, out_files, per_file):
    print('  [cifar10] fetching ' + parquet_name + ' from HuggingFace (uoft-cs/cifar10)')
    path = hf_hub_download('uoft-cs/cifar10', parquet_name, repo_type='dataset')
    tbl = pq.read_table(path, columns=['img', 'label'])
    imgs = tbl.column('img').to_pylist()
    labels = tbl.column('label').to_pylist()
    idx = 0
    for out in out_files:
        buf = bytearray()
        for _ in range(per_file):
            arr = np.asarray(
                Image.open(io.BytesIO(imgs[idx]['bytes'])).convert('RGB'),
                dtype=np.uint8,
            )                                   # (32, 32, 3) HWC
            chw = np.transpose(arr, (2, 0, 1)).reshape(-1)   # R,G,B planes
            buf.append(int(labels[idx]) & 0xFF)
            buf.extend(chw.tobytes())
            idx += 1
        with open(os.path.join(EXTRACTED_DIR, out), 'wb') as f:
            f.write(buf)
        print('  [cifar10] wrote ' + out)

_write_split('plain_text/train-00000-of-00001.parquet',
             ['data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin',
              'data_batch_4.bin', 'data_batch_5.bin'], 10000)
_write_split('plain_text/test-00000-of-00001.parquet', ['test_batch.bin'], 10000)
"""
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
    var os = Python.import_module("os")
    var home = String(os.path.expanduser(PythonObject("~")))
    var path = home + "/.cache/mojo_rl/cifar10"
    _ = os.makedirs(PythonObject(path), exist_ok=True)
    return path


def _ensure_extracted() raises -> String:
    """Ensure the 6 .bin files exist. Returns the dir containing them.

    Priority: (1) already-present .bin files; (2) a binary tarball if one is
    already cached or `CIFAR10_BIN_URL` is set; (3) DEFAULT — the HuggingFace
    parquet dataset, decoded to the 6 .bin files.
    """
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var cache = _cache_dir()
    var extracted_path = cache + "/" + _EXTRACTED_DIR

    # 1. already materialized?
    var test_path = extracted_path + "/" + _TEST_FILE
    if Bool(os.path.exists(PythonObject(test_path))):
        return extracted_path

    # 2. binary tarball path — only if a tar is already cached or a mirror URL is
    #    given via CIFAR10_BIN_URL (avoids the flaky toronto.edu default).
    var tar_path = cache + "/cifar-10-binary.tar.gz"
    var url_env = String(
        os.environ.get(PythonObject("CIFAR10_BIN_URL"), PythonObject(""))
    )
    var have_tar = Bool(os.path.exists(PythonObject(tar_path)))
    if have_tar or url_env.byte_length() > 0:
        if not have_tar:
            var url = url_env if url_env.byte_length() > 0 else String(_CIFAR_URL)
            print("  [cifar10] downloading " + url + " (~162 MB)")
            var urllib_request = Python.import_module("urllib.request")
            var resp = urllib_request.urlopen(PythonObject(url), timeout=120)
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

    # 3. default: HuggingFace parquet → write the 6 .bin files, in a SUBPROCESS
    #    (pyarrow's libarrow can't safely load in Mojo's embedded interpreter).
    print("  [cifar10] preparing from HuggingFace parquet (uoft-cs/cifar10)")
    var sys = Python.import_module("sys")
    var subprocess = Python.import_module("subprocess")
    var child_env = os.environ.copy()
    child_env["CIFAR10_EXTRACTED_DIR"] = PythonObject(extracted_path)
    var argv = builtins.list()
    _ = argv.append(sys.executable)
    _ = argv.append(PythonObject("-c"))
    _ = argv.append(PythonObject(_PY_PREP_HF))
    _ = subprocess.run(argv, check=True, env=child_env)
    print("  [cifar10] ready -> " + extracted_path)
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
    var inv255 = Scalar[DType.float32](1.0) / Scalar[DType.float32](255.0)
    var inv_std_r = Scalar[DType.float32](1.0) / _STD_R
    var inv_std_g = Scalar[DType.float32](1.0) / _STD_G
    var inv_std_b = Scalar[DType.float32](1.0) / _STD_B
    comptime CHAN_SZ = 32 * 32
    for i in range(_SAMPLES_PER_FILE):
        var base = i * _BYTES_PER_SAMPLE
        dst_labels[offset_samples + i] = Int32(Int(bytes[base]))
        var img_dst = (offset_samples + i) * (3 * 32 * 32)
        # Channel R (first 1024 bytes)
        for p in range(CHAN_SZ):
            var raw = Scalar[DType.float32](Int(bytes[base + 1 + p])) * inv255
            dst_images[img_dst + p] = (raw - _MEAN_R) * inv_std_r
        # Channel G (next 1024 bytes)
        for p in range(CHAN_SZ):
            var raw = (
                Scalar[DType.float32](Int(bytes[base + 1 + CHAN_SZ + p]))
                * inv255
            )
            dst_images[img_dst + CHAN_SZ + p] = (raw - _MEAN_G) * inv_std_g
        # Channel B (last 1024 bytes)
        for p in range(CHAN_SZ):
            var raw = (
                Scalar[DType.float32](
                    Int(bytes[base + 1 + 2 * CHAN_SZ + p])
                )
                * inv255
            )
            dst_images[img_dst + 2 * CHAN_SZ + p] = (
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
