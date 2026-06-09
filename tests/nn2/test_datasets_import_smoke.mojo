"""nn2.datasets move smoke — import-surface compile check.

Validates that `mojo_rl/nn2/datasets/` parses + type-checks end to end after
the move from `mojo_rl/nn/datasets/`: all four loaders (MNIST, CIFAR10,
TinyShakespeare, LeWM-PushT) plus the `CIFAR10CropFlipAugmenter` re-export
(which actually lives in `nn2.training.augmenter`). No data is loaded — the
import + a trivial `to_one_hot` instantiation is enough to force compilation of
the package without needing MNIST/CIFAR assets on disk.

Run:  pixi run mojo run -I . tests/nn2/test_datasets_import_smoke.mojo
"""

from mojo_rl.nn2.datasets import (
    MNIST,
    CIFAR10,
    CIFAR10CropFlipAugmenter,
    CharTokenizer,
    DatasetSplit,
    Minibatch,
    load_text,
    train_val_split,
    make_batch,
    to_one_hot,
    LewmPushTExpert,
    LewmPushTWindow,
)


def main() raises:
    print("nn2.datasets import smoke")
    # Exercise a pure (data-free) helper so a generic instantiation is forced.
    var ids = List[Int]()
    ids.append(0)
    ids.append(2)
    ids.append(1)
    var oh = to_one_hot(ids, vocab_size=3, batch_size=1, seq_len=3)
    # one-hot of [0,2,1] (1 batch, seq 3) over vocab 3 → 9 entries, three 1.0s.
    var ones = 0
    for v in oh:
        if v == 1.0:
            ones += 1
    if len(oh) != 9 or ones != 3:
        raise Error("to_one_hot produced wrong shape/content")
    print("  to_one_hot ok (len=", len(oh), " ones=", ones, ")")
    print("OK — nn2.datasets compiles and the export surface is intact")
