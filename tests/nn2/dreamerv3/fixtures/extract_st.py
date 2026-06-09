"""Fixture: straight-through one-hot sample backward (jax.vjp).

The DreamerV3 stoch sample is `value = sg(onehot(idx)) + (probs - sg(probs))`
with `probs = (1-u)·softmax(z) + u/C` (unimix). The gradient is
`d value/d z = d probs/d z = (1-u)·softmax_jacobian(z)` — INDEPENDENT of the
random index (it's stop-grad'd). So we validate the backward with any seed.

Run: /tmp/dreamer_fixtures_venv/bin/python3 tests/nn2/dreamerv3/fixtures/extract_st.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "references", "dreamerv3-main"))

import numpy as np
import jax
import jax.numpy as jnp
from embodied.jax import outs

rng = np.random.default_rng(23)
B, STOCH, CLASSES = 2, 3, 5
UNIMIX = 0.01

z = jnp.asarray(rng.standard_normal((B, STOCH, CLASSES)), jnp.float32)
cot = jnp.asarray(rng.standard_normal((B, STOCH, CLASSES)), jnp.float32)


def st_value(z):
    return outs.OneHot(z, UNIMIX).sample(jax.random.PRNGKey(0))


val, vjp = jax.vjp(st_value, z)
(g_z,) = vjp(cot)

lines = []


def emit(name, arr):
    a = np.asarray(arr, np.float64).reshape(-1)
    lines.append(f"{name}#size={a.size}")
    for v in a:
        lines.append(f"{v:.9g}")


lines.append(f"unimix={UNIMIX:.9g}")
emit("st.z", z)
emit("st.cot", cot)
emit("st.g_z", g_z)

path = os.path.join(_HERE, "st_fixture.txt")
with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", path)
print("g_z[:5]", np.asarray(g_z).reshape(-1)[:5])
