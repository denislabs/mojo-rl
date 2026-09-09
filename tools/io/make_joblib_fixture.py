"""Write `tests/fixtures/joblib_arrays.pkl` — the fixture `tests/io/test_pickle_joblib.mojo`
reads with `mojo_rl/io/pickle.mojo`. Needs joblib + numpy (any env that has them):

    python tools/io/make_joblib_fixture.py

Deterministic content (a hashed ramp), every dtype and rank the reader
claims, nested dicts, an int / bool / str / float / None, and one array
long enough to span several pickle FRAMEs. The expected values the test
pins are printed so they can be pasted, and are also derivable from the
ramp formula in the test itself.
"""
import joblib, numpy as np
from pathlib import Path

def ramp(n, seed, dtype):
    k = (np.arange(n) * 7919 + seed * 104729) % 1000
    v = (k / 999.0 * 2.0 - 1.0) * 3.0
    return v.astype(dtype)

d = {
    "clipA": {
        "f4_2d": ramp(15, 1, np.float32).reshape(5, 3),
        "f4_3d": ramp(24, 2, np.float32).reshape(2, 4, 3),
        "f8_2d": ramp(4, 3, np.float64).reshape(2, 2),
        "i8_1d": np.array([3, -7, 11], dtype=np.int64),
        "i4_1d": np.array([1, 2, 3, 4], dtype=np.int32),
        "big": ramp(20000, 4, np.float32).reshape(2000, 10),
        "fps": 30,
        "flag": True,
        "name": "clip-A",
        "scale": 0.25,
        "nothing": None,
    },
    "clipB": {"f4_2d": ramp(6, 5, np.float32).reshape(2, 3), "fps": 60},
}
out = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "joblib_arrays.pkl"
joblib.dump(d, out)
print("wrote", out, out.stat().st_size, "bytes")
print("clipA.f4_2d[0]", d["clipA"]["f4_2d"][0].tolist(), "sum", float(d["clipA"]["f4_2d"].astype(np.float64).sum()))
print("clipA.f4_3d sum", float(d["clipA"]["f4_3d"].astype(np.float64).sum()), "[1,3,2]", float(d["clipA"]["f4_3d"][1, 3, 2]))
print("clipA.f8_2d", d["clipA"]["f8_2d"].tolist())
print("clipA.big sum", float(d["clipA"]["big"].astype(np.float64).sum()), "[1999,9]", float(d["clipA"]["big"][1999, 9]))
print("clipB.f4_2d sum", float(d["clipB"]["f4_2d"].astype(np.float64).sum()))
