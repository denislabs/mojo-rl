"""PR-1 primitive fixtures: GELU, SiLU, RMSNorm, BlockLinear.

Ground truth from real jax (forward + jax.vjp gradients) so the Mojo nn2
primitives can be validated to ≤1e-4 against the exact reference math.

- GELU  = `jax.nn.gelu` (approximate=True, the tanh approximation) — this
  is what `references/dreamerv3-main/embodied/jax/nets.py:act('gelu')`
  resolves to (`getattr(jax.nn, 'gelu')`).
- SiLU  = `jax.nn.silu` (== x·sigmoid(x); reference `act('silu')`).
- RMSNorm = reference `nets.py:Norm` impl='rms', VERBATIM forward:
      mean2 = square(x).mean(-1, keepdims=True)
      y     = x * (rsqrt(mean2 + eps) * scale)     # scale = gamma, eps=1e-4
- BlockLinear = reference `nets.py:BlockLinear` einsum '...ki,kio->...ko'
  with a [blocks, in/blocks, out/blocks] kernel + [out] bias.

Run from the throwaway venv (project pixi has no jax):
    /tmp/dreamer_fixtures_venv/bin/python \\
        tests/nn2/dreamerv3/fixtures/extract_prims.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

f32 = jnp.float32
RMS_EPS = 1e-4


def main():
    rng = np.random.default_rng(20260529)
    lines = []

    def emit(s):
        lines.append(s)

    def emit_flat(name, arr):
        flat = np.asarray(arr).ravel()
        emit(f"{name}#size={flat.size}")
        for v in flat:
            emit(repr(float(v)))

    emit(f"# jax={jax.__version__} np={np.__version__}")

    # ── GELU (tanh approx) ──
    x = jnp.asarray(rng.standard_normal(32).astype(np.float32) * 2.0)
    go = jnp.asarray(rng.standard_normal(32).astype(np.float32))
    y, vjp = jax.vjp(lambda z: jax.nn.gelu(z, approximate=True), x)
    (gx,) = vjp(go)
    emit_flat("gelu.x", x)
    emit_flat("gelu.go", go)
    emit_flat("gelu.y", y)
    emit_flat("gelu.gx", gx)

    # ── SiLU ──
    xs = jnp.asarray(rng.standard_normal(32).astype(np.float32) * 2.0)
    gos = jnp.asarray(rng.standard_normal(32).astype(np.float32))
    ys, vjps = jax.vjp(jax.nn.silu, xs)
    (gxs,) = vjps(gos)
    emit_flat("silu.x", xs)
    emit_flat("silu.go", gos)
    emit_flat("silu.y", ys)
    emit_flat("silu.gx", gxs)

    # ── RMSNorm (reference Norm 'rms') ──
    BATCH, DIM = 3, 8

    def rms_forward(x, scale):
        x = f32(x)
        mean2 = jnp.square(x).mean(-1, keepdims=True)
        return x * (jax.lax.rsqrt(mean2 + RMS_EPS) * scale)

    xr = jnp.asarray(rng.standard_normal((BATCH, DIM)).astype(np.float32))
    gamma = jnp.asarray(
        (1.0 + 0.3 * rng.standard_normal(DIM)).astype(np.float32)
    )
    gor = jnp.asarray(rng.standard_normal((BATCH, DIM)).astype(np.float32))
    yr, vjpr = jax.vjp(rms_forward, xr, gamma)
    gxr, ggamma = vjpr(gor)
    emit(f"rms.batch={BATCH}")
    emit(f"rms.dim={DIM}")
    emit(f"rms.eps={RMS_EPS!r}")
    emit_flat("rms.x", xr)
    emit_flat("rms.gamma", gamma)
    emit_flat("rms.go", gor)
    emit_flat("rms.y", yr)
    emit_flat("rms.gx", gxr)
    emit_flat("rms.ggamma", ggamma)

    # ── BlockLinear (reference BlockLinear) ──
    # kernel shape: [blocks, in/blocks, out/blocks]; einsum '...ki,kio->...ko'
    # reshape x [B, IN] -> [B, blocks, in/blocks]; out -> [B, OUT]; + bias[OUT].
    BB, BIN, BOUT, BLK = 4, 12, 16, 4

    def bl_forward(x, kernel, bias):
        insize = x.shape[-1]
        xr2 = x.reshape((*x.shape[:-1], BLK, insize // BLK))
        o = jnp.einsum("...ki,kio->...ko", xr2, kernel)
        o = o.reshape((*o.shape[:-2], BOUT))
        return o + bias

    xb = jnp.asarray(rng.standard_normal((BB, BIN)).astype(np.float32))
    kernel = jnp.asarray(
        (0.1 * rng.standard_normal((BLK, BIN // BLK, BOUT // BLK))).astype(
            np.float32
        )
    )
    biasb = jnp.asarray(
        (0.05 * rng.standard_normal(BOUT)).astype(np.float32)
    )
    gob = jnp.asarray(rng.standard_normal((BB, BOUT)).astype(np.float32))
    yb, vjpb = jax.vjp(bl_forward, xb, kernel, biasb)
    gxb, gkernel, gbias = vjpb(gob)
    emit(f"bl.batch={BB}")
    emit(f"bl.in={BIN}")
    emit(f"bl.out={BOUT}")
    emit(f"bl.blocks={BLK}")
    emit_flat("bl.x", xb)
    emit_flat("bl.kernel", kernel)   # [BLK, IN/BLK, OUT/BLK] row-major
    emit_flat("bl.bias", biasb)
    emit_flat("bl.go", gob)
    emit_flat("bl.y", yb)
    emit_flat("bl.gx", gxb)
    emit_flat("bl.gkernel", gkernel)
    emit_flat("bl.gbias", gbias)

    out_path = os.path.join(os.path.dirname(__file__), "prims_fixture.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
