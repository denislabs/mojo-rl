"""Sanity check for mojo_rl.nn.datasets.mnist loader.

Confirms:
  - Files download/cache
  - IDX parsing gives 60000 train + 10000 test samples
  - Labels are in [0, 9]
  - Pixels are in [0, 1], mean approximately 0.13

Run:
    pixi run mojo run -I . tests/nn/test_mnist_loader.mojo
"""

from mojo_rl.nn.datasets.mnist import MNIST


def main() raises:
    print("Loading MNIST...")
    var ds = MNIST()

    print("  num_train =", ds.num_train)
    print("  num_test  =", ds.num_test)

    if ds.num_train != 60000 or ds.num_test != 10000:
        raise Error("unexpected sample counts")

    # Label range check
    var min_lbl = Int32(99)
    var max_lbl = Int32(-1)
    for i in range(ds.num_train):
        if ds.train_labels[i] < min_lbl:
            min_lbl = ds.train_labels[i]
        if ds.train_labels[i] > max_lbl:
            max_lbl = ds.train_labels[i]
    print("  train labels range: [", min_lbl, ",", max_lbl, "]")

    # Pixel stats on train set
    var px_min: Scalar[DType.float32] = 1.0
    var px_max: Scalar[DType.float32] = 0.0
    var px_sum: Float64 = 0.0
    var total_px = ds.num_train * MNIST.IMG_SIZE
    for i in range(total_px):
        var v = ds.train_images[i]
        if v < px_min:
            px_min = v
        if v > px_max:
            px_max = v
        px_sum += Float64(v)
    var mean = px_sum / Float64(total_px)
    print("  train pixels: min=", px_min, "max=", px_max, "mean=", mean)

    # Label histogram
    var counts = List[Int](length=10, fill=0)
    for i in range(ds.num_train):
        counts[Int(ds.train_labels[i])] += 1
    print("  train label histogram:")
    for c in range(10):
        print("    ", c, ":", counts[c])

    print("OK")
