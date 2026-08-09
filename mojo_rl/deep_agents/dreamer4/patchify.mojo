"""Frame ↔ patch conversion (model.py:temporal_patchify / temporal_unpatchify).

CPU data-prep utilities, NOT in the differentiable path: the tokenizer trains
in patch space (input = patches, masked-MSE loss in patch space), so patchify
turns rendered frames into the encoder's input and unpatchify is only needed
to reconstruct images for PSNR / visualization.

Layout matches PyTorch `F.unfold(kernel=stride=patch)`: per frame the grid is
`NP = (H/P)·(W/P)` patches (row-major `pr·(W/P)+pc`), each of width
`DP = C·P·P` ordered `c·P² + ky·P + kx`. Buffers are flat at nn-BATCH = B·T:

    video   : [BT, C, H, W]   (frame `bt`, channel c, row h, col w)
    patches : [BT, NP, DP]

`temporal_patchify` and `temporal_unpatchify` are exact inverses.
"""

from mojo_rl.nn.constants import DT


def downscale_box[
    H_IN: Int, W_IN: Int, H_OUT: Int, W_OUT: Int
](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """Box-filter downscale of a single H_IN×W_IN grayscale image to
    H_OUT×W_OUT (averages the source pixels covering each output pixel).
    Same approach as the Pong env's 160×210→84×84 resize."""
    for dy in range(H_OUT):
        var sy0 = dy * H_IN // H_OUT
        var sy1 = (dy + 1) * H_IN // H_OUT
        if sy1 <= sy0:
            sy1 = sy0 + 1
        for dx in range(W_OUT):
            var sx0 = dx * W_IN // W_OUT
            var sx1 = (dx + 1) * W_IN // W_OUT
            if sx1 <= sx0:
                sx1 = sx0 + 1
            var total: Float64 = 0.0
            var count: Int = 0
            for sy in range(sy0, sy1):
                for sx in range(sx0, sx1):
                    total += Float64(src[unsafe_offset=sy * W_IN + sx])
                    count += 1
            dst[unsafe_offset=dy * W_OUT + dx] = Scalar[DT](total / Float64(count))


def temporal_patchify[
    BT: Int, C: Int, H: Int, W: Int, PATCH: Int
](
    video: Pointer[Scalar[DT], MutAnyOrigin],
    patches: Pointer[Scalar[DT], MutAnyOrigin],
):
    comptime assert H % PATCH == 0 and W % PATCH == 0, (
        "temporal_patchify: H, W must be divisible by PATCH"
    )
    comptime PH = H // PATCH
    comptime PW = W // PATCH
    comptime NP = PH * PW
    comptime DP = C * PATCH * PATCH
    for bt in range(BT):
        var vbase = bt * C * H * W
        var pbase = bt * NP * DP
        for pr in range(PH):
            for pc in range(PW):
                var np_off = pbase + (pr * PW + pc) * DP
                for c in range(C):
                    for ky in range(PATCH):
                        var h = pr * PATCH + ky
                        for kx in range(PATCH):
                            var w = pc * PATCH + kx
                            var dp = c * PATCH * PATCH + ky * PATCH + kx
                            patches[unsafe_offset=np_off + dp] = video[unsafe_offset=
                                vbase + c * H * W + h * W + w
                            ]


def temporal_unpatchify[
    BT: Int, C: Int, H: Int, W: Int, PATCH: Int
](
    patches: Pointer[Scalar[DT], MutAnyOrigin],
    video: Pointer[Scalar[DT], MutAnyOrigin],
):
    comptime assert H % PATCH == 0 and W % PATCH == 0, (
        "temporal_unpatchify: H, W must be divisible by PATCH"
    )
    comptime PH = H // PATCH
    comptime PW = W // PATCH
    comptime NP = PH * PW
    comptime DP = C * PATCH * PATCH
    for bt in range(BT):
        var vbase = bt * C * H * W
        var pbase = bt * NP * DP
        for pr in range(PH):
            for pc in range(PW):
                var np_off = pbase + (pr * PW + pc) * DP
                for c in range(C):
                    for ky in range(PATCH):
                        var h = pr * PATCH + ky
                        for kx in range(PATCH):
                            var w = pc * PATCH + kx
                            var dp = c * PATCH * PATCH + ky * PATCH + kx
                            video[unsafe_offset=vbase + c * H * W + h * W + w] = patches[unsafe_offset=
                                np_off + dp
                            ]
