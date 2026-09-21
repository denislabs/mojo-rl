"""Which LIBERO demo files are readable, and which demos inside them.

    pixi run python tools/tasks/check_libero_demos.py                 # libero_goal, 26 demos
    pixi run python tools/tasks/check_libero_demos.py libero_object 50

A `libero_demo_batched` run that dies inside libhdf5 with "message not
aligned" / "can't deserialize object header chunk" is reading a file whose
BYTES are wrong on disk — a truncated or corrupted download — and libhdf5's
diagnostic names neither the file nor the demo. This walks every
`*_demo.hdf5` of a family, opens `data/demo_i/actions` and `/states` for
`i < n`, and prints one line per file: `ok` with the demo count and the
file size, or `BAD` with the exception. Compare a BAD file's size against
the HF listing (`yifengzhu-hf/LIBERO-datasets`) and check `df -h`: a full
disk truncates writes silently.
"""
import glob
import os
import sys

import h5py


def main():
    family = sys.argv[1] if len(sys.argv) > 1 else "libero_goal"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 26
    files = sorted(glob.glob(f"references/libero_demos/{family}/*_demo.hdf5"))
    if not files:
        sys.exit(f"no *_demo.hdf5 under references/libero_demos/{family}")
    bad = 0
    for f in files:
        size = os.path.getsize(f)
        try:
            with h5py.File(f, "r") as h:
                have = len(h["data"])
                for i in range(min(n, have)):
                    h[f"data/demo_{i}/actions"][()]
                    h[f"data/demo_{i}/states"][0]
            print(f"ok   {size:>12,} B  {have:>3} demos  {f}")
        except Exception as e:  # noqa: BLE001 - the point is to name it
            bad += 1
            print(f"BAD  {size:>12,} B  {f}\n     {type(e).__name__}: {e}")
    print(f"{len(files) - bad} readable, {bad} damaged")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
