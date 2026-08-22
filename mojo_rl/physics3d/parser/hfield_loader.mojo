"""Load a `<hfield file=...>` elevation grid — `mjCHField::Compile`.

Ported from `references/mujoco-3.11.0/src/user/user_objects.cc:4700-4885`:
`LoadPNG` (:4750), `LoadCustom` (:4708) and the normalisation tail of
`Compile` (:4870).

WHAT MuJoCo DOES, IN ORDER:

    PNG   -> lodepng with LCT_GREY, then rows REVERSED
    .bin  -> two int32 (nrow, ncol) then nrow*ncol float32, verbatim
    both  -> min-max rescale to [0, 1]

⚠⚠ THE ROWS ARE REVERSED FOR PNG AND NOT FOR THE BINARY FORM. `LoadPNG` writes
`data[r*ncol + c] = image[c + (nrow-1-r)*ncol]`, because a PNG's first row is
its TOP and a heightfield's first row is its `-y` edge. Reading the image
straight through mirrors the terrain about y, which on a symmetric field is
invisible and on any other one puts every slope on the wrong side.

⚠ THE GREY CONVERSION IS "TAKE RED", NOT A LUMA. lodepng's `rgba8ToPixel` for
`LCT_GREY` writes `out[i] = r` and discards g/b — so a colour PNG becomes its
RED channel, not its brightness. Taking a luma instead would differ on every
non-grey image. (barkour's field has R == G == B, so the two agree there; the
rule is the reference's, not the fixture's.)

⚠ THE NORMALISATION IS PER FIELD AND UNCONDITIONAL. Whatever the file holds is
rescaled so its own min becomes 0 and its own max becomes 1; the PHYSICAL
height is `data * size[2]`, applied at collision time. A loader that keeps the
file's units produces a field `size[2]` times too tall.
"""

from std.memory import bitcast

from mojo_rl.render.png_loader import load_png


def load_hfield_file(path: String) raises -> Tuple[Int, Int, List[Float64]]:
    """Returns `(nrow, ncol, data)` with `data` normalised to [0, 1].

    Row-major, `data[r * ncol + c]`, matching `mjModel.hfield_data`.
    """
    if path.endswith(".png") or path.endswith(".PNG"):
        return _load_png_hfield(path)
    return _load_custom_hfield(path)


def _normalize(mut d: List[Float64]) raises:
    """`mjCHField::Compile`'s tail — min-max to [0, 1], in place."""
    if len(d) == 0:
        raise Error("physics3d: <hfield> has no elevation data")
    var emin = d[0]
    var emax = d[0]
    for i in range(len(d)):
        if d[i] < emin:
            emin = d[i]
        if d[i] > emax:
            emax = d[i]
    if emin > emax:
        raise Error("physics3d: invalid data range in <hfield>")
    var span = emax - emin
    for i in range(len(d)):
        d[i] = d[i] - emin
        # `mjEPS`, as the reference guards it: a flat field stays flat rather
        # than dividing by zero.
        if span > 1e-14:
            d[i] = d[i] / span


def _load_png_hfield(path: String) raises -> Tuple[Int, Int, List[Float64]]:
    var tex = load_png(path)
    var ncol = tex.width
    var nrow = tex.height
    if nrow < 1 or ncol < 1:
        raise Error("physics3d: <hfield> PNG '" + path + "' is empty")
    var d = List[Float64](capacity=nrow * ncol)
    for r in range(nrow):
        var src_row = nrow - 1 - r
        for c in range(ncol):
            # RED, and rows reversed — see the header.
            d.append(Float64(Int(tex.pixels[(src_row * ncol + c) * 4])))
    _normalize(d)
    return (nrow, ncol, d^)


def _load_custom_hfield(path: String) raises -> Tuple[Int, Int, List[Float64]]:
    """`LoadCustom`: `int32 nrow, int32 ncol, float32 data[nrow*ncol]`."""
    var f = open(path, "r")
    var raw = f.read_bytes()
    f.close()
    if len(raw) < 8:
        raise Error(
            "physics3d: <hfield file='" + path + "'> is missing its header"
        )

    @parameter
    def _u32(o: Int) -> Int:
        return (
            Int(raw[o + 0])
            | (Int(raw[o + 1]) << 8)
            | (Int(raw[o + 2]) << 16)
            | (Int(raw[o + 3]) << 24)
        )

    var nrow = _u32(0)
    var ncol = _u32(4)
    if nrow < 1 or ncol < 1:
        raise Error(
            "physics3d: non-positive <hfield> dimensions in '" + path + "'"
        )
    if len(raw) != nrow * ncol * 4 + 8:
        raise Error(
            "physics3d: unexpected file size in <hfield file='" + path + "'>"
        )
    var d = List[Float64](capacity=nrow * ncol)
    for i in range(nrow * ncol):
        var o = 8 + i * 4
        var bits = UInt32(_u32(o))
        d.append(Float64(bitcast[DType.float32, 1](bits)))
    _normalize(d)
    return (nrow, ncol, d^)
