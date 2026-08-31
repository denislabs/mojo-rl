"""TinyShakespeare loader — char-level text dataset for GPT validation.

First call downloads ~1MB of Shakespeare from
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
into ~/.cache/mojo_rl/tinyshakespeare/input.txt and parses it natively in Mojo.

Pipeline:
    text = load_text()                     # raw String
    tok  = CharTokenizer(text)             # build byte-level vocab
    ids  = tok.encode(text)                # List[Int] of length N
    train_ids, val_ids = train_val_split(ids, val_frac=0.1)
    inp, tgt = make_batch(train_ids, batch=64, seq_len=256, seed=k)
    one_hot  = to_one_hot(inp, tok.vocab_size, batch=64, seq_len=256)
    # one_hot has shape (BATCH, seq_len * vocab_size) — feed straight into GPT.

The text is pure ASCII so byte-level tokenization is identical to char-level.
Vocab size is typically ~65 (printable ASCII + newline).
"""

from std.os import makedirs
from std.os.path import exists

from mojo_rl.io.fileio import write_file_atomic
from mojo_rl.io.hf import mojo_rl_cache
from mojo_rl.io.http import http_get_bytes
from std.random import random_si64
from mojo_rl.nn.constants import DT


comptime _SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
comptime _CACHE_FILE = "input.txt"


def _cache_dir() raises -> String:
    var path = mojo_rl_cache() + "/tinyshakespeare"
    makedirs(path, exist_ok=True)
    return path^


def _ensure_downloaded() raises -> String:
    """Returns the path to the cached input.txt, downloading it if missing."""
    var cache = _cache_dir()
    var txt_path = cache + "/" + String(_CACHE_FILE)

    if exists(txt_path):
        return txt_path^

    print("  [tinyshakespeare] downloading " + String(_SHAKESPEARE_URL) + " (~1 MB)")
    var data = http_get_bytes(String(_SHAKESPEARE_URL))
    write_file_atomic(txt_path, data)
    print("  [tinyshakespeare] download complete -> " + txt_path)
    return txt_path^


def load_text() raises -> String:
    """Load the raw TinyShakespeare text as a String. Downloads on first call."""
    var path = _ensure_downloaded()
    with open(path, "r") as f:
        return f.read()


# =============================================================================
# Char-level tokenizer
# =============================================================================
struct CharTokenizer(Movable):
    """Byte-level tokenizer for ASCII text.

    Builds the vocabulary from the unique byte values present in the source
    text. For TinyShakespeare this gives ~65 tokens (printable ASCII +
    newline), identical to a "char-level" tokenizer over ASCII.

    Attributes:
        vocab_size: Number of unique tokens.
        id_to_byte: List mapping token id (0..vocab_size) -> byte value.
        byte_to_id: 256-element list mapping byte -> token id, or -1 if the
                    byte is not in the vocabulary.
    """

    var vocab_size: Int
    var id_to_byte: List[Int]
    var byte_to_id: List[Int]

    def __init__(out self, text: String):
        # Mark which bytes appear in the text.
        var seen = List[Bool](capacity=256)
        for _ in range(256):
            seen.append(False)
        var bytes = text.as_bytes()
        for i in range(len(bytes)):
            seen[Int(bytes[i])] = True

        # Build sorted vocab.
        self.id_to_byte = List[Int]()
        self.byte_to_id = List[Int](capacity=256)
        for _ in range(256):
            self.byte_to_id.append(-1)
        for b in range(256):
            if seen[b]:
                self.byte_to_id[b] = len(self.id_to_byte)
                self.id_to_byte.append(b)
        self.vocab_size = len(self.id_to_byte)

    def __init__(out self, *, deinit move: Self):
        self.vocab_size = move.vocab_size
        self.id_to_byte = move.id_to_byte^
        self.byte_to_id = move.byte_to_id^

    def encode(self, text: String) -> List[Int]:
        var bytes = text.as_bytes()
        var ids = List[Int](capacity=len(bytes))
        for i in range(len(bytes)):
            var tid = self.byte_to_id[Int(bytes[i])]
            # Unknown bytes (not in training vocab) are silently skipped —
            # callers should encode against text they built the vocab from.
            if tid >= 0:
                ids.append(tid)
        return ids^

    def decode(self, ids: List[Int]) -> String:
        var s = String("")
        for i in range(len(ids)):
            var tid = ids[i]
            if tid < 0 or tid >= self.vocab_size:
                continue
            s += chr(self.id_to_byte[tid])
        return s^


# =============================================================================
# Train/val split + minibatch sampling
# =============================================================================
struct DatasetSplit(Movable):
    """Contiguous (train, val) token-id split."""
    var train: List[Int]
    var val: List[Int]

    def __init__(out self, var train: List[Int], var val: List[Int]):
        self.train = train^
        self.val = val^

    def __init__(out self, *, deinit move: Self):
        self.train = move.train^
        self.val = move.val^


struct Minibatch(Movable):
    """A single (inputs, targets) minibatch of token ids.

    `inputs[b * seq_len + t]`  = source token id at window position t
    `targets[b * seq_len + t]` = inputs[b * seq_len + t + 1] (next-token target)
    Both are flat List[Int] of length BATCH * seq_len.
    """
    var inputs: List[Int]
    var targets: List[Int]

    def __init__(out self, var inputs: List[Int], var targets: List[Int]):
        self.inputs = inputs^
        self.targets = targets^

    def __init__(out self, *, deinit move: Self):
        self.inputs = move.inputs^
        self.targets = move.targets^


def train_val_split(
    ids: List[Int], val_frac: Float64 = 0.1
) -> DatasetSplit:
    """Contiguous train/val split — last `val_frac` of tokens go to val.

    A contiguous split (rather than shuffled) is correct for autoregressive
    LMs: shuffling token-level training data destroys the sequential
    structure the model is trying to learn.
    """
    var n = len(ids)
    var n_train = Int(Float64(n) * (1.0 - val_frac))
    var train = List[Int](capacity=n_train)
    var val = List[Int](capacity=n - n_train)
    for i in range(n_train):
        train.append(ids[i])
    for i in range(n_train, n):
        val.append(ids[i])
    return DatasetSplit(train^, val^)


def make_batch(
    ids: List[Int], batch_size: Int, seq_len: Int
) raises -> Minibatch:
    """Sample `batch_size` random windows of length `seq_len + 1` from `ids`.

    Uses std.random.random_si64; seed externally with std.random.seed for
    reproducibility.
    """
    var n = len(ids)
    if n < seq_len + 1:
        raise Error(
            "make_batch: ids has "
            + String(n)
            + " tokens but seq_len+1 = "
            + String(seq_len + 1)
            + " required"
        )

    var inp = List[Int](capacity=batch_size * seq_len)
    var tgt = List[Int](capacity=batch_size * seq_len)
    for _ in range(batch_size * seq_len):
        inp.append(0)
        tgt.append(0)

    var max_start = Int64(n - seq_len - 1)
    for b in range(batch_size):
        # Inclusive [0, max_start]
        var start = Int(random_si64(Int64(0), max_start))
        for t in range(seq_len):
            inp[b * seq_len + t] = ids[start + t]
            tgt[b * seq_len + t] = ids[start + t + 1]
    return Minibatch(inp^, tgt^)


# =============================================================================
# Token ids -> one-hot tensor
# =============================================================================
def to_one_hot(
    token_ids: List[Int], vocab_size: Int, batch_size: Int, seq_len: Int
) raises -> List[Scalar[DT]]:
    """Convert a flat (BATCH * seq_len) list of token ids into a flat
    (BATCH * seq_len * vocab_size) one-hot tensor laid out row-major
    outer-batch-then-token-then-vocab — exactly what GPT.IN_DIM expects.
    """
    if len(token_ids) != batch_size * seq_len:
        raise Error(
            "to_one_hot: expected "
            + String(batch_size * seq_len)
            + " token ids, got "
            + String(len(token_ids))
        )
    var total = batch_size * seq_len * vocab_size
    var oh = List[Scalar[DT]](capacity=total)
    for _ in range(total):
        oh.append(0)
    for b in range(batch_size):
        for t in range(seq_len):
            var tid = token_ids[b * seq_len + t]
            if tid < 0 or tid >= vocab_size:
                continue
            var off = (b * seq_len + t) * vocab_size + tid
            oh[off] = Scalar[DT](1.0)
    return oh^
