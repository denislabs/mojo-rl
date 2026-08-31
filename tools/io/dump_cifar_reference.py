"""Reference `.bin` for `tests/nn/test_cifar10_prep.mojo`.

    pixi run python tools/io/dump_cifar_reference.py --out /tmp/cifar_ref

Downloads CIFAR-10's TEST parquet from `uoft-cs/cifar10` and writes, beside
it, the canonical `test_batch.bin` built the way `nn/datasets/cifar10.mojo`
now builds it — but with `pyarrow` reading the parquet and `PIL` decoding the
PNGs, which is exactly what that loader stopped using.

⚠ THE ORACLE MUST NOT SHARE CODE WITH WHAT IT GATES. The Mojo side reads the
same parquet with its own reader, decodes with its own PNG decoder, and lays
the bytes out itself; a disagreement anywhere in that chain shows up as a byte
difference here. Two implementations of one wrong assumption is the failure
this arrangement exists to prevent.

Only the 10,000-image test split: it is a quarter of the download and covers
every code path the 50,000-image train split does.
"""

import argparse
import io
import os
import urllib.request

URL = (
    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
    "plain_text/test-00000-of-00001.parquet"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/cifar_ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pq_path = os.path.join(args.out, "test.parquet")
    if not os.path.exists(pq_path):
        print("downloading %s" % URL)
        req = urllib.request.Request(URL)
        with urllib.request.urlopen(req, timeout=300) as r, \
                open(pq_path + ".part", "wb") as f:
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(pq_path + ".part", pq_path)
    print("parquet: %s (%d bytes)" % (pq_path, os.path.getsize(pq_path)))

    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    t = pq.read_table(pq_path)
    imgs = t.column("img").to_pylist()
    labels = t.column("label").to_pylist()
    assert len(imgs) == 10000, len(imgs)

    out = bytearray()
    for d, lab in zip(imgs, labels):
        a = np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
        assert a.shape == (32, 32, 3), a.shape
        out.append(lab & 0xFF)
        # channel-major: 1024 R, then G, then B
        for c in range(3):
            out.extend(a[:, :, c].tobytes())

    bin_path = os.path.join(args.out, "test_batch_reference.bin")
    with open(bin_path, "wb") as f:
        f.write(out)
    print("reference: %s (%d bytes, %d samples)"
          % (bin_path, len(out), len(imgs)))


if __name__ == "__main__":
    main()
