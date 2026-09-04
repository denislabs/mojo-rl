"""The content channel: everything the frame channel deliberately does not carry.

Hypothesis 4.0 splits the state in two. The FRAME `u` is transported by an
orthogonal connection indexed by the edge, and carries what is topologically
relevant — orientation, feuillet, parity. Everything else is CONTENT `h`, moved
by an ordinary learned transition. Phase 5 measured the cost of running without
the second half: nine of ten planner failures were the RIGHT PARITY IN THE WRONG
CELL, because the frame alone is a weak place code (adjacent cells differ by
~0.3 rad). This file is that missing half.

Two jobs, and they have to be given carefully:

**Reconstruction grounds both channels.** `decoder(u, h) -> obs` is the anchoring
term of §4.5. Without it `h` has no reason to encode anything at all.

**The transition must not be trained on its own.** `h_{t+1} = f(h_t, a)` with a
free `f` is satisfied exactly by `h = constant` — the same trap that made the
frame channel's transport loss vacuous in Phase 3, where a place-indexed
constant fit every edge at zero residual and turned `det H` into a coin. Here
the reconstruction is what stops it: a constant `h` cannot reconstruct twelve
different textures.

**The content channel is RECURRENT and could in principle learn the parity**,
since `f(h_t, a)` carries state across the seam. It has no incentive to — the
texture it reconstructs is identical at both parities — and whether it does is
the sharpest available test of whether the split is doing real work. If `h`
learned the parity, the frame channel would be redundant and hypothesis 4.0's
split would be unmotivated. That is gated, not assumed.
"""

from std.math import abs

from .mlp import Mlp
from .rng import Rng


struct ContentChannel[
    OBS_DIM: Int,
    LAT: Int,
    HID: Int,
    CONTENT_DIM: Int,
    N_ACTIONS: Int,
    dtype: DType = DType.float64,
](Copyable, Movable):
    comptime TRANS_IN: Int = Self.CONTENT_DIM + Self.N_ACTIONS

    var dec: Mlp[Self.LAT, Self.HID, Self.OBS_DIM, Self.dtype]
    """`(u, h) -> obs`. Anchors both channels; without it `h` encodes nothing."""
    var trans: Mlp[Self.TRANS_IN, Self.HID, Self.CONTENT_DIM, Self.dtype]
    """`(h, action) -> h'`. The ordinary dynamics the frame channel is not."""

    def __init__(out self, mut rng: Rng):
        self.dec = Mlp[Self.LAT, Self.HID, Self.OBS_DIM, Self.dtype](rng)
        self.trans = Mlp[
            Self.TRANS_IN, Self.HID, Self.CONTENT_DIM, Self.dtype
        ](rng)

    def __init__(out self, *, copy: Self):
        self.dec = copy.dec.copy()
        self.trans = copy.trans.copy()

    def __init__(out self, *, deinit move: Self):
        self.dec = move.dec^
        self.trans = move.trans^

    def zero_grad(mut self):
        self.dec.zero_grad()
        self.trans.zero_grad()

    def adam_step(mut self, lr: Float64):
        self.dec.adam_step(lr)
        self.trans.adam_step(lr)

    def reconstruction(
        mut self,
        lat: List[Scalar[Self.dtype]],
        obs: List[Scalar[Self.dtype]],
        mut d_lat: List[Scalar[Self.dtype]],
        weight: Float64,
    ) -> Float64:
        """MSE of `decoder(lat)` against the observation. Accumulates grads."""
        var hid = List[Scalar[Self.dtype]](length=Self.HID, fill=0)
        var out = List[Scalar[Self.dtype]](length=Self.OBS_DIM, fill=0)
        self.dec.forward(lat, hid, out)
        var loss = Float64(0)
        var d_out = List[Scalar[Self.dtype]](length=Self.OBS_DIM, fill=0)
        for i in range(Self.OBS_DIM):
            var e = Float64(out[i] - obs[i])
            loss += e * e
            d_out[i] = Scalar[Self.dtype](2.0 * e * weight)
        var d_in = List[Scalar[Self.dtype]](length=Self.LAT, fill=0)
        self.dec.backward(lat, hid, d_out, d_in)
        for i in range(Self.LAT):
            d_lat[i] += d_in[i]
        return loss

    def _pack(
        self, h: List[Scalar[Self.dtype]], action: Int
    ) -> List[Scalar[Self.dtype]]:
        var x = List[Scalar[Self.dtype]](length=Self.TRANS_IN, fill=0)
        for i in range(Self.CONTENT_DIM):
            x[i] = h[i]
        if action >= 0 and action < Self.N_ACTIONS:
            x[Self.CONTENT_DIM + action] = 1
        return x^

    def predict_next(
        self, h: List[Scalar[Self.dtype]], action: Int
    ) -> List[Scalar[Self.dtype]]:
        """One imagined content step. Used by the planner; no gradients."""
        var x = self._pack(h, action)
        var hid = List[Scalar[Self.dtype]](length=Self.HID, fill=0)
        var out = List[Scalar[Self.dtype]](length=Self.CONTENT_DIM, fill=0)
        self.trans.forward(x, hid, out)
        return out^

    def transition(
        mut self,
        h_src: List[Scalar[Self.dtype]],
        action: Int,
        h_dst: List[Scalar[Self.dtype]],
        mut d_lat_src: List[Scalar[Self.dtype]],
        frame_dim: Int,
        weight: Float64,
    ) -> Float64:
        """Predict `h_dst` from `(h_src, action)`; target carries no gradient.

        Pre-consensus targets here too, for the same reason as the frame
        channel: a target that had already been reconciled would train the
        dynamics to reduce their own disagreement.
        """
        var x = self._pack(h_src, action)
        var hid = List[Scalar[Self.dtype]](length=Self.HID, fill=0)
        var out = List[Scalar[Self.dtype]](length=Self.CONTENT_DIM, fill=0)
        self.trans.forward(x, hid, out)
        var loss = Float64(0)
        var d_out = List[Scalar[Self.dtype]](length=Self.CONTENT_DIM, fill=0)
        for i in range(Self.CONTENT_DIM):
            var e = Float64(out[i] - h_dst[i])
            loss += e * e
            d_out[i] = Scalar[Self.dtype](2.0 * e * weight)
        var d_in = List[Scalar[Self.dtype]](length=Self.TRANS_IN, fill=0)
        self.trans.backward(x, hid, d_out, d_in)
        # Gradient reaches the encoder only through the SOURCE content latent.
        for i in range(Self.CONTENT_DIM):
            d_lat_src[frame_dim + i] += d_in[i]
        return loss
