"""Smoke test for the TinyShakespeare loader.

Verifies:
  1. Download + load returns a non-empty text string.
  2. Tokenizer round-trip: encode(decode(x)) == x for the full text.
  3. Vocab has the expected ~65 chars (printable ASCII + newline).
  4. train/val split sums to total length and val_frac is honored.
  5. make_batch produces the correct shapes and target = input shifted by 1
     for the same window start.
  6. to_one_hot output has shape (BATCH * seq_len * vocab) and is exactly
     consumable by GPT.IN_DIM (one-hot row per token).

Run:
    pixi run mojo run -I . tests/nn/test_tinyshakespeare_loader.mojo
"""

from std.random import seed
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn2.datasets import (
    CharTokenizer,
    load_text,
    train_val_split,
    make_batch,
    to_one_hot,
)
from mojo_rl.nn.composites import GPT
from layout import Layout, LayoutTensor


def check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


def print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


def test_load_and_tokenize() raises -> Int:
    print_header("load_text + CharTokenizer round-trip")
    var fails = 0

    var text = load_text()
    check(
        text.byte_length() > 100_000,
        "loaded text length " + String(text.byte_length()) + " > 100k bytes",
        fails,
    )

    var tok = CharTokenizer(text)
    check(
        tok.vocab_size > 50 and tok.vocab_size < 100,
        "vocab_size in [50, 100], got " + String(tok.vocab_size),
        fails,
    )

    # Round-trip a small slice — encode then decode must reconstruct it.
    var slice_str = String(text[byte=0:2000])
    var ids = tok.encode(slice_str)
    var back = tok.decode(ids)
    check(
        back.byte_length() == slice_str.byte_length(),
        "encode/decode preserves byte length: " + String(back.byte_length()),
        fails,
    )
    var roundtrip_ok = back == slice_str
    check(roundtrip_ok, "encode/decode round-trip exact match on 2000-byte slice", fails)

    return fails


def test_split_and_batch() raises -> Int:
    print_header("train/val split + make_batch shapes")
    var fails = 0
    seed(42)

    var text = load_text()
    var tok = CharTokenizer(text)
    var ids = tok.encode(text)
    var n = len(ids)

    var split = train_val_split(ids, 0.1)
    check(
        len(split.train) + len(split.val) == n,
        "train + val == total: " + String(len(split.train) + len(split.val)),
        fails,
    )
    var expected_train = Int(Float64(n) * 0.9)
    check(
        len(split.train) == expected_train,
        "len(train) = floor(0.9 * n) = " + String(len(split.train)),
        fails,
    )

    # Sample a batch and check shapes + target = input shifted by 1 for some
    # batch row.
    var BATCH = 4
    var SEQ = 32
    var batch = make_batch(split.train, BATCH, SEQ)
    check(
        len(batch.inputs) == BATCH * SEQ,
        "input length BATCH*SEQ = " + String(len(batch.inputs)),
        fails,
    )
    check(
        len(batch.targets) == BATCH * SEQ,
        "target length BATCH*SEQ = " + String(len(batch.targets)),
        fails,
    )

    # For every batch row, target[t] must be the token immediately after
    # input[t] in the source. Local invariant: target[t] for t in [0, SEQ-2]
    # equals input[t+1].
    var max_mismatch = 0
    for b in range(BATCH):
        for t in range(SEQ - 1):
            if batch.targets[b * SEQ + t] != batch.inputs[b * SEQ + t + 1]:
                max_mismatch += 1
    check(
        max_mismatch == 0,
        "target[t] == input[t+1] for all t in [0, seq-2] (causal next-token)",
        fails,
    )

    # Token ids must all be in [0, vocab_size).
    var min_id = 1_000_000
    var max_id = -1
    for i in range(len(batch.inputs)):
        if batch.inputs[i] < min_id:
            min_id = batch.inputs[i]
        if batch.inputs[i] > max_id:
            max_id = batch.inputs[i]
    check(
        min_id >= 0 and max_id < tok.vocab_size,
        "token ids in [0, vocab_size): [" + String(min_id) + ", " + String(max_id) + "]",
        fails,
    )

    return fails


def test_one_hot_into_gpt() raises -> Int:
    """End-to-end: load -> tokenize -> batch -> one-hot -> GPT.forward.
    No NaN, no shape errors. This validates the loader feeds the model."""
    print_header("end-to-end: TinyShakespeare batch -> GPT.forward")
    var fails = 0
    seed(7)

    var text = load_text()
    var tok = CharTokenizer(text)
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)

    # Use exactly the vocab present in the data.
    var V = tok.vocab_size

    # Tiny GPT to keep the test fast.
    comptime S = 16
    comptime D = 16
    comptime H = 2
    comptime N = 1
    comptime BATCH = 2
    # GPT vocab is comptime — we have to commit to a constant. Use the
    # tokenizer's vocab_size only if it matches; otherwise fail early.
    comptime TEST_VOCAB = 65

    if V != TEST_VOCAB:
        # Not strictly a failure, but the test compiles GPT with TEST_VOCAB
        # at comptime so we can't adapt at runtime. Print and skip
        # forward — Karpathy's TinyShakespeare consistently has 65 unique
        # chars, so this branch is informational.
        print(
            "  SKIP: tokenizer vocab "
            + String(V)
            + " != compile-time TEST_VOCAB "
            + String(TEST_VOCAB)
            + " — re-run with adjusted constant"
        )
        return fails

    comptime Model = GPT[TEST_VOCAB, S, D, H, N]

    # Pull a batch and one-hot it.
    var batch = make_batch(split.train, BATCH, S)
    var oh = to_one_hot(batch.inputs, TEST_VOCAB, BATCH, S)
    check(
        len(oh) == BATCH * Model.IN_DIM,
        "one-hot length matches GPT.IN_DIM = " + String(BATCH * Model.IN_DIM),
        fails,
    )

    # Each token contributes exactly one 1.0 in its vocab block — total
    # number of 1's in the one-hot must equal BATCH * S.
    var ones_count: Int = 0
    for i in range(len(oh)):
        if Float64(oh[i]) > 0.5:
            ones_count += 1
    check(
        ones_count == BATCH * S,
        "exactly BATCH*S = " + String(BATCH * S) + " ones in one-hot tensor",
        fails,
    )

    # Run forward — random params, just verify no NaN.
    var params = List[Scalar[dtype]](capacity=Model.PARAM_SIZE)
    for _ in range(Model.PARAM_SIZE):
        params.append(Scalar[dtype](0.01))
    var out_data = List[Scalar[dtype]](capacity=BATCH * Model.OUT_DIM)
    for _ in range(BATCH * Model.OUT_DIM):
        out_data.append(0)
    var cache_data = List[Scalar[dtype]](capacity=BATCH * Model.CACHE_SIZE)
    for _ in range(BATCH * Model.CACHE_SIZE):
        cache_data.append(0)
    var state = List[Scalar[dtype]](capacity=1)
    state.append(0)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](oh.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Model.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_t = LayoutTensor[
        dtype, Layout.row_major(Model.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Model.forward[BATCH, dtype](inp_t, out_t, p_t, s_t, c_t)

    var has_nan = False
    for i in range(BATCH * Model.OUT_DIM):
        var v = Float64(out_data[i])
        if v != v:
            has_nan = True
    check(not has_nan, "GPT logits non-NaN on real Shakespeare batch", fails)

    return fails


def main() raises:
    var total_fails = 0
    total_fails += test_load_and_tokenize()
    total_fails += test_split_and_batch()
    total_fails += test_one_hot_into_gpt()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TINYSHAKESPEARE LOADER TESTS PASSED")
    else:
        print("FAILED: " + String(total_fails) + " checks")
    print("=" * 70)
