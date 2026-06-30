"""PPM image writer for visualizing encoder/decoder reconstructions.

Pure Mojo, zero dependencies. Writes PPM P6 (binary RGB) files that macOS
Preview, GIMP, and most image viewers open natively.

Usage — pixel observations (e.g., 84x84 grayscale frame stacks):

    from mojo_rl.render.image_writer import save_reconstruction_grid

    # originals, reconstructions: UnsafePointer[Scalar[float32]]
    # laid out as [N, C, H, W] in CHW order, values in ~[0, 1] or [-1, 1]
    save_reconstruction_grid(
        "recon.ppm", originals, reconstructions,
        n=8, height=84, width=84, channels=1,
    )

Usage — flat observation vectors (e.g., Pendulum OBS_DIM=3):

    from mojo_rl.render.image_writer import save_vector_heatmap

    # data: UnsafePointer[Scalar[float32]], shape [N, DIM]
    save_vector_heatmap("obs_vs_recon.ppm", data, n_rows=16, dim=17)
"""

from std.math import min, max


# ─────────────────────────────────────────────────────────────────────────────
# save_ppm — single image
# ─────────────────────────────────────────────────────────────────────────────


def save_ppm(
    path: String,
    data: UnsafePointer[Scalar[DType.float32], _],
    height: Int,
    width: Int,
    channels: Int = 1,
    vmin: Float32 = 0.0,
    vmax: Float32 = 1.0,
) raises:
    """Write a single float32 image as PPM P6 (binary RGB).

    Args:
        path: Output file path (should end in .ppm).
        data: Pixel values in CHW order (channels-first), contiguous.
        height: Image height.
        width: Image width.
        channels: 1 (grayscale) or 3 (RGB). Default 1.
        vmin: Value mapped to 0. Default 0.0.
        vmax: Value mapped to 255. Default 1.0.
    """
    var n_pixels = height * width
    var buf = List[UInt8](capacity=n_pixels * 3 + 64)

    # PPM P6 header
    var header = "P6\n" + String(width) + " " + String(height) + "\n255\n"
    var hb = header.as_bytes()
    for i in range(len(hb)):
        buf.append(hb[i])

    var scale = 255.0 / max(vmax - vmin, Float32(1e-8))
    if channels == 1:
        for i in range(n_pixels):
            var v = Float32((data + i)[])
            var byte = UInt8(min(max((v - vmin) * scale, Float32(0.0)), Float32(255.0)))
            buf.append(byte)
            buf.append(byte)
            buf.append(byte)
    elif channels == 3:
        for i in range(n_pixels):
            var r = Float32((data + i)[])
            var g = Float32((data + n_pixels + i)[])
            var b = Float32((data + 2 * n_pixels + i)[])
            buf.append(UInt8(min(max((r - vmin) * scale, Float32(0.0)), Float32(255.0))))
            buf.append(UInt8(min(max((g - vmin) * scale, Float32(0.0)), Float32(255.0))))
            buf.append(UInt8(min(max((b - vmin) * scale, Float32(0.0)), Float32(255.0))))
    else:
        raise Error("channels must be 1 or 3, got " + String(channels))

    with open(path, "w") as f:
        f.write_bytes(buf)


def _write_pixel(
    mut buf: List[UInt8],
    src: UnsafePointer[Scalar[DType.float32], _],
    img_idx: Int,
    local_y: Int,
    local_x: Int,
    img_size: Int,
    hw: Int,
    width: Int,
    channels: Int,
    vmin: Float32,
    scale: Float32,
):
    var base = img_idx * img_size
    if channels == 1:
        var v = Float32((src + base + local_y * width + local_x)[])
        var byte = UInt8(min(max((v - vmin) * scale, Float32(0.0)), Float32(255.0)))
        buf.append(byte)
        buf.append(byte)
        buf.append(byte)
    else:
        var rv = Float32((src + base + local_y * width + local_x)[])
        var gv = Float32((src + base + hw + local_y * width + local_x)[])
        var bv = Float32((src + base + 2 * hw + local_y * width + local_x)[])
        buf.append(UInt8(min(max((rv - vmin) * scale, Float32(0.0)), Float32(255.0))))
        buf.append(UInt8(min(max((gv - vmin) * scale, Float32(0.0)), Float32(255.0))))
        buf.append(UInt8(min(max((bv - vmin) * scale, Float32(0.0)), Float32(255.0))))


# ─────────────────────────────────────────────────────────────────────────────
# save_reconstruction_grid — original vs reconstructed side-by-side
# ─────────────────────────────────────────────────────────────────────────────


def save_reconstruction_grid(
    path: String,
    originals: UnsafePointer[Scalar[DType.float32], _],
    reconstructions: UnsafePointer[Scalar[DType.float32], _],
    n: Int,
    height: Int,
    width: Int,
    channels: Int = 1,
    vmin: Float32 = 0.0,
    vmax: Float32 = 1.0,
    pad: Int = 2,
) raises:
    """Write an N-row grid: each row has [original | reconstructed].

    Top label row is gray (originals side) vs white (reconstructions side).

    Args:
        path: Output file path.
        originals: [N, C, H, W] float32, CHW per image.
        reconstructions: [N, C, H, W] float32, same layout.
        n: Number of image pairs.
        height: Per-image height.
        width: Per-image width.
        channels: 1 or 3.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.
        pad: Pixels of padding between images. Default 2.
    """
    var img_size = channels * height * width
    var grid_w = 2 * width + 3 * pad  # pad | orig | pad | recon | pad
    var grid_h = n * height + (n + 1) * pad
    var total = grid_h * grid_w * 3
    var buf = List[UInt8](capacity=total + 64)

    # PPM header
    var header = "P6\n" + String(grid_w) + " " + String(grid_h) + "\n255\n"
    var hb = header.as_bytes()
    for i in range(len(hb)):
        buf.append(hb[i])

    var scale = 255.0 / max(vmax - vmin, Float32(1e-8))
    var pad_color = UInt8(40)  # dark gray padding
    var left_x0 = pad
    var right_x0 = 2 * pad + width
    var hw = height * width

    for row in range(grid_h):
        var img_row = (row - pad) // (height + pad)
        var local_y = (row - pad) % (height + pad)

        for col in range(grid_w):
            var is_content = img_row >= 0 and img_row < n and local_y < height
            var is_left = col >= left_x0 and col < left_x0 + width
            var is_right = col >= right_x0 and col < right_x0 + width

            if is_content and is_left:
                var local_x = col - left_x0
                _write_pixel(buf, originals, img_row, local_y, local_x,
                             img_size, hw, width, channels, vmin, scale)
            elif is_content and is_right:
                var local_x = col - right_x0
                _write_pixel(buf, reconstructions, img_row, local_y, local_x,
                             img_size, hw, width, channels, vmin, scale)
            else:
                buf.append(pad_color)
                buf.append(pad_color)
                buf.append(pad_color)

    with open(path, "w") as f:
        f.write_bytes(buf)
    print(
        "Reconstruction grid saved: " + path + " (" + String(n) + " pairs, "
        + String(grid_w) + "x" + String(grid_h) + ")"
    )


# ─────────────────────────────────────────────────────────────────────────────
# save_image_row — horizontal strip of N images (like plt.subplots(1, N))
# ─────────────────────────────────────────────────────────────────────────────


def save_image_row(
    path: String,
    data: UnsafePointer[Scalar[DType.float32], _],
    n: Int,
    height: Int,
    width: Int,
    channels: Int = 1,
    vmin: Float32 = 0.0,
    vmax: Float32 = 1.0,
    pad: Int = 2,
    pixel_scale: Int = 1,
    labels: List[String] = List[String](),
) raises:
    """Write N images side-by-side in a single horizontal strip.

    Matches the common notebook pattern: plt.subplots(1, N). Useful for
    visualizing generated digits, frame sequences, or any image set.

    Args:
        path: Output file path.
        data: [N, C, H, W] float32, CHW per image, contiguous.
        n: Number of images.
        height: Per-image height.
        width: Per-image width.
        channels: 1 or 3.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.
        pad: Pixels of padding between and around images. Default 2.
        pixel_scale: Nearest-neighbor upscale factor. Default 1 (no scaling).
        labels: Optional list of N short labels rendered below each image.
    """
    var img_size = channels * height * width
    var hw = height * width
    var has_labels = len(labels) == n
    var s = pixel_scale
    var scaled_w = width * s
    var scaled_h = height * s
    var label_h = 10 * s if has_labels else 0
    var scaled_pad = pad * s
    var grid_w = n * scaled_w + (n + 1) * scaled_pad
    var grid_h = scaled_h + 2 * scaled_pad + label_h
    var scale = 255.0 / max(vmax - vmin, Float32(1e-8))
    var pad_color = UInt8(40)

    var total = grid_w * grid_h * 3
    var buf = List[UInt8](capacity=total + 64)

    var header = "P6\n" + String(grid_w) + " " + String(grid_h) + "\n255\n"
    var hb = header.as_bytes()
    for i in range(len(hb)):
        buf.append(hb[i])

    for row in range(grid_h):
        for col in range(grid_w):
            # Map scaled pixel back to logical coordinates
            var img_idx = (col - scaled_pad) // (scaled_w + scaled_pad)
            var local_x = ((col - scaled_pad) % (scaled_w + scaled_pad)) // s
            var local_y = (row - scaled_pad) // s
            var in_col = (col - scaled_pad) % (scaled_w + scaled_pad) < scaled_w
            var in_image = (
                img_idx >= 0 and img_idx < n
                and in_col
                and row >= scaled_pad and local_y < height
            )

            if in_image:
                _write_pixel(buf, data, img_idx, local_y, local_x,
                             img_size, hw, width, channels, vmin, scale)
            elif has_labels and row >= scaled_h + 2 * scaled_pad and img_idx >= 0 and img_idx < n and in_col:
                var char_row = (row - (scaled_h + 2 * scaled_pad)) // s
                var char_col = local_x
                var label = labels[img_idx]
                if _font_pixel(label, char_row, char_col, width):
                    buf.append(UInt8(220))
                    buf.append(UInt8(220))
                    buf.append(UInt8(220))
                else:
                    buf.append(pad_color)
                    buf.append(pad_color)
                    buf.append(pad_color)
            else:
                buf.append(pad_color)
                buf.append(pad_color)
                buf.append(pad_color)

    with open(path, "w") as f:
        f.write_bytes(buf)
    print(
        "Image row saved: " + path + " (" + String(n) + " images, "
        + String(grid_w) + "x" + String(grid_h) + ")"
    )


def _font_pixel(label: String, char_row: Int, char_col: Int, cell_width: Int) -> Bool:
    """Render a centered label using a minimal 3x5 digit font. Returns True if pixel is lit."""
    if char_row >= 7:
        return False

    # 3x5 bitmaps for digits 0-9, packed as 5 rows of 3 bits each (MSB left)
    var glyphs: InlineArray[UInt16, 10] = [
        UInt16(0b111_101_101_101_111),  # 0
        UInt16(0b010_110_010_010_111),  # 1
        UInt16(0b111_001_111_100_111),  # 2
        UInt16(0b111_001_111_001_111),  # 3
        UInt16(0b101_101_111_001_001),  # 4
        UInt16(0b111_100_111_001_111),  # 5
        UInt16(0b111_100_111_101_111),  # 6
        UInt16(0b111_001_001_001_001),  # 7
        UInt16(0b111_101_111_101_111),  # 8
        UInt16(0b111_101_111_001_111),  # 9
    ]

    # Center the label horizontally; one extra pixel gap between chars
    var label_len = label.byte_length()
    var char_w = 4  # 3 pixels + 1 gap
    var total_w = label_len * char_w - 1
    var x_offset = (cell_width - total_w) // 2
    var lx = char_col - x_offset
    if lx < 0 or lx >= total_w:
        return False

    # Map to row 1..5 (skip row 0 = top padding)
    var gy = char_row - 1
    if gy < 0 or gy >= 5:
        return False

    var char_idx = lx // char_w
    var cx = lx % char_w
    if cx >= 3:  # gap between characters
        return False
    if char_idx >= label_len:
        return False

    # Get digit value from ASCII
    var digit = Int(label.as_bytes()[char_idx]) - 48
    if digit < 0 or digit > 9:
        return False

    var bit_pos = (4 - gy) * 3 + (2 - cx)
    return Bool((Int(glyphs[digit]) >> bit_pos) & 1)


# ─────────────────────────────────────────────────────────────────────────────
# save_vector_heatmap — flat obs vectors as color-mapped rows
# ─────────────────────────────────────────────────────────────────────────────


def _viridis_rgb(t: Float32) -> Tuple[UInt8, UInt8, UInt8]:
    """Approximate viridis colormap at t in [0, 1]."""
    # Piecewise linear approximation (5 control points)
    var r: Float32
    var g: Float32
    var b: Float32
    if t < 0.25:
        var s = t * 4.0
        r = 68.0 + s * (49.0 - 68.0)
        g = 1.0 + s * (104.0 - 1.0)
        b = 84.0 + s * (142.0 - 84.0)
    elif t < 0.5:
        var s = (t - 0.25) * 4.0
        r = 49.0 + s * (33.0 - 49.0)
        g = 104.0 + s * (165.0 - 104.0)
        b = 142.0 + s * (133.0 - 142.0)
    elif t < 0.75:
        var s = (t - 0.5) * 4.0
        r = 33.0 + s * (144.0 - 33.0)
        g = 165.0 + s * (206.0 - 165.0)
        b = 133.0 + s * (62.0 - 133.0)
    else:
        var s = (t - 0.75) * 4.0
        r = 144.0 + s * (253.0 - 144.0)
        g = 206.0 + s * (231.0 - 206.0)
        b = 62.0 + s * (37.0 - 62.0)
    return (
        UInt8(min(max(r, Float32(0.0)), Float32(255.0))),
        UInt8(min(max(g, Float32(0.0)), Float32(255.0))),
        UInt8(min(max(b, Float32(0.0)), Float32(255.0))),
    )


def save_vector_heatmap(
    path: String,
    data: UnsafePointer[Scalar[DType.float32], _],
    n_rows: Int,
    dim: Int,
    cell_w: Int = 16,
    cell_h: Int = 16,
    vmin: Float32 = -1.0,
    vmax: Float32 = 1.0,
) raises:
    """Render flat observation vectors as a viridis heatmap grid.

    Each row is one sample, each column is one dimension. Useful for comparing
    original vs reconstructed vectors (stack them vertically or interleave).

    Args:
        path: Output file path.
        data: [n_rows, dim] float32, row-major.
        n_rows: Number of vectors.
        dim: Dimensionality per vector.
        cell_w: Pixel width per cell. Default 16.
        cell_h: Pixel height per cell. Default 16.
        vmin: Value mapped to lowest color. Default -1.0.
        vmax: Value mapped to highest color. Default 1.0.
    """
    var img_w = dim * cell_w
    var img_h = n_rows * cell_h
    var total = img_w * img_h * 3
    var buf = List[UInt8](capacity=total + 64)

    var header = "P6\n" + String(img_w) + " " + String(img_h) + "\n255\n"
    var hb = header.as_bytes()
    for i in range(len(hb)):
        buf.append(hb[i])

    var inv_range = 1.0 / max(vmax - vmin, Float32(1e-8))
    for py in range(img_h):
        var row = py // cell_h
        for px in range(img_w):
            var col = px // cell_w
            var v = Float32((data + row * dim + col)[])
            var t = min(max((v - vmin) * inv_range, Float32(0.0)), Float32(1.0))
            var rgb = _viridis_rgb(t)
            buf.append(rgb[0])
            buf.append(rgb[1])
            buf.append(rgb[2])

    with open(path, "w") as f:
        f.write_bytes(buf)
    print(
        "Heatmap saved: " + path + " (" + String(n_rows) + "x" + String(dim)
        + " → " + String(img_w) + "x" + String(img_h) + "px)"
    )


def save_vector_comparison(
    path: String,
    originals: UnsafePointer[Scalar[DType.float32], _],
    reconstructions: UnsafePointer[Scalar[DType.float32], _],
    n: Int,
    dim: Int,
    cell_w: Int = 16,
    cell_h: Int = 16,
    vmin: Float32 = -1.0,
    vmax: Float32 = 1.0,
) raises:
    """Interleave original/reconstructed vectors as paired heatmap rows.

    Row pattern: orig_0, recon_0, (separator), orig_1, recon_1, ...
    Originals get a blue left-edge marker, reconstructions get orange.

    Args:
        path: Output file path.
        originals: [n, dim] float32, row-major.
        reconstructions: [n, dim] float32, row-major.
        n: Number of pairs.
        dim: Dimensionality per vector.
        cell_w: Pixel width per cell. Default 16.
        cell_h: Pixel height per cell. Default 16.
        vmin: Value mapped to lowest color. Default -1.0.
        vmax: Value mapped to highest color. Default 1.0.
    """
    var marker_w = 4
    var sep_h = 2
    var img_w = marker_w + dim * cell_w
    var pair_h = 2 * cell_h
    var img_h = n * pair_h + (n - 1) * sep_h
    var total = img_w * img_h * 3
    var buf = List[UInt8](capacity=total + 64)

    var header = "P6\n" + String(img_w) + " " + String(img_h) + "\n255\n"
    var hb = header.as_bytes()
    for i in range(len(hb)):
        buf.append(hb[i])

    var inv_range = 1.0 / max(vmax - vmin, Float32(1e-8))
    var global_y = 0
    for pair_idx in range(n):
        # Separator between pairs (not before first)
        if pair_idx > 0:
            for _ in range(sep_h):
                for _ in range(img_w):
                    buf.append(UInt8(40))
                    buf.append(UInt8(40))
                    buf.append(UInt8(40))
                global_y += 1

        # Original row (blue marker)
        for _ in range(cell_h):
            for px in range(img_w):
                if px < marker_w:
                    buf.append(UInt8(70))
                    buf.append(UInt8(130))
                    buf.append(UInt8(230))
                else:
                    var col = (px - marker_w) // cell_w
                    var v = Float32((originals + pair_idx * dim + col)[])
                    var t = min(max((v - vmin) * inv_range, Float32(0.0)), Float32(1.0))
                    var rgb = _viridis_rgb(t)
                    buf.append(rgb[0])
                    buf.append(rgb[1])
                    buf.append(rgb[2])
            global_y += 1

        # Reconstruction row (orange marker)
        for _ in range(cell_h):
            for px in range(img_w):
                if px < marker_w:
                    buf.append(UInt8(230))
                    buf.append(UInt8(150))
                    buf.append(UInt8(50))
                else:
                    var col = (px - marker_w) // cell_w
                    var v = Float32((reconstructions + pair_idx * dim + col)[])
                    var t = min(max((v - vmin) * inv_range, Float32(0.0)), Float32(1.0))
                    var rgb = _viridis_rgb(t)
                    buf.append(rgb[0])
                    buf.append(rgb[1])
                    buf.append(rgb[2])
            global_y += 1

    with open(path, "w") as f:
        f.write_bytes(buf)
    print(
        "Vector comparison saved: " + path + " (" + String(n)
        + " pairs, dim=" + String(dim) + ")"
    )


# ─────────────────────────────────────────────────────────────────────────────
# save_frame_sequence_gif — animated GIF89a from a grayscale frame sequence
#
# Pure Mojo, zero dependencies (no Python imageio, no SDL). GIF is palette-
# indexed; grayscale frames map 1:1 onto a fixed 256-entry gray global palette
# so there is NO color quantization — only LZW compression. Useful for
# visualizing world-model imagination rollouts, reconstruction sequences, or any
# grayscale frame animation. Compose multi-panel frames (e.g. real|recon|imag)
# host-side into one wide grayscale image per step, then pass the sequence here.
# ─────────────────────────────────────────────────────────────────────────────


def _put_u16_le(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))


struct _GifBitWriter(Movable):
    """LSB-first variable-width bit packer for GIF LZW code streams."""

    var out: List[UInt8]
    var bitbuf: Int
    var bitcnt: Int

    def __init__(out self):
        self.out = List[UInt8]()
        self.bitbuf = 0
        self.bitcnt = 0

    def write(mut self, code: Int, size: Int):
        self.bitbuf |= code << self.bitcnt
        self.bitcnt += size
        while self.bitcnt >= 8:
            self.out.append(UInt8(self.bitbuf & 0xFF))
            self.bitbuf >>= 8
            self.bitcnt -= 8

    def flush(mut self):
        if self.bitcnt > 0:
            self.out.append(UInt8(self.bitbuf & 0xFF))
            self.bitbuf = 0
            self.bitcnt = 0

    def take(deinit self) -> List[UInt8]:
        return self.out^


def _lzw_encode_gif(indices: List[UInt8], min_code_size: Int) -> List[UInt8]:
    """GIF-variant LZW. Returns the raw code stream (pre sub-block packing).

    Dictionary keyed by (prefix_code << 8 | next_byte) → code. Codes 0..2^k-1 are
    the literal bytes; `clear = 2^k`, `end = clear + 1`. Width grows from k+1 up
    to 12 bits, resetting (clear code) when the table fills (4096 entries).
    """
    var clear = 1 << min_code_size
    var end = clear + 1
    var code_size = min_code_size + 1
    var next_code = end + 1
    var dict = Dict[Int, Int]()

    var bw = _GifBitWriter()
    bw.write(clear, code_size)
    if len(indices) == 0:
        bw.write(end, code_size)
        bw.flush()
        return bw^.take()

    var prev = Int(indices[0])
    for i in range(1, len(indices)):
        var c = Int(indices[i])
        var key = (prev << 8) | c
        var found = dict.get(key)
        if found:
            prev = found.value()
        else:
            bw.write(prev, code_size)
            if next_code < 4096:
                dict[key] = next_code
                next_code += 1
                # GIF LZW: the decoder adds its first table entry one code later
                # than the encoder (its first data code only outputs), so it lags
                # by one entry → bump the code width at 2^code_size + 1, not
                # 2^code_size, or the widths desync ("broken data stream").
                if next_code == (1 << code_size) + 1 and code_size < 12:
                    code_size += 1
            else:
                # table full → emit clear, reset to the literal table
                bw.write(clear, code_size)
                dict = Dict[Int, Int]()
                code_size = min_code_size + 1
                next_code = end + 1
            prev = c
    bw.write(prev, code_size)
    bw.write(end, code_size)
    bw.flush()
    return bw^.take()


def _append_subblocks(mut buf: List[UInt8], data: List[UInt8]):
    """Write `data` as GIF sub-blocks (≤255 bytes each), terminated by 0x00."""
    var off = 0
    var n = len(data)
    while off < n:
        var chunk = min(255, n - off)
        buf.append(UInt8(chunk))
        for i in range(chunk):
            buf.append(data[off + i])
        off += chunk
    buf.append(UInt8(0))  # block terminator


def save_frame_sequence_gif(
    path: String,
    frames: UnsafePointer[Scalar[DType.float32], _],
    n_frames: Int,
    height: Int,
    width: Int,
    channels: Int = 1,
    fps: Int = 10,
    loop: Bool = True,
    vmin: Float32 = 0.0,
    vmax: Float32 = 1.0,
) raises:
    """Write a grayscale float frame sequence as an animated GIF89a.

    Args:
        path: Output path (should end in .gif).
        frames: [n_frames, H, W] float32, row-major (one grayscale image per
            frame, contiguous). Values mapped through [vmin, vmax] → [0, 255].
        n_frames: Number of frames.
        height: Per-frame height.
        width: Per-frame width.
        channels: Must be 1 (grayscale). RGB GIF needs color quantization and is
            not supported here — composite/convert to grayscale first.
        fps: Playback speed (frames per second). Delay = round(100/fps) cs.
        loop: True → loop forever (NETSCAPE2.0 ext); False → play once.
        vmin: Value mapped to palette index 0.
        vmax: Value mapped to palette index 255.
    """
    if channels != 1:
        raise Error(
            "save_frame_sequence_gif is grayscale-only (channels=1), got "
            + String(channels)
        )
    var hw = height * width
    var scale = 255.0 / max(vmax - vmin, Float32(1e-8))
    var delay_cs = 100 // fps
    if delay_cs < 1:
        delay_cs = 1

    var buf = List[UInt8]()
    # ── Header ──
    var sig = String("GIF89a").as_bytes()
    for i in range(len(sig)):
        buf.append(sig[i])
    # ── Logical Screen Descriptor ──
    _put_u16_le(buf, width)
    _put_u16_le(buf, height)
    buf.append(UInt8(0xF7))  # GCT present, color res 8b, GCT size 256 (2^(7+1))
    buf.append(UInt8(0))     # background color index
    buf.append(UInt8(0))     # pixel aspect ratio
    # ── Global Color Table: 256 grayscale entries (i,i,i) ──
    for i in range(256):
        buf.append(UInt8(i))
        buf.append(UInt8(i))
        buf.append(UInt8(i))
    # ── NETSCAPE2.0 looping extension ──
    if loop:
        buf.append(UInt8(0x21))
        buf.append(UInt8(0xFF))
        buf.append(UInt8(0x0B))
        var ns = String("NETSCAPE2.0").as_bytes()
        for i in range(len(ns)):
            buf.append(ns[i])
        buf.append(UInt8(0x03))
        buf.append(UInt8(0x01))
        _put_u16_le(buf, 0)  # loop count 0 = forever
        buf.append(UInt8(0))

    # ── per-frame: GCE + image descriptor + LZW data ──
    var indices = List[UInt8](capacity=hw)
    for f in range(n_frames):
        # Graphic Control Extension (per-frame delay; disposal = leave in place)
        buf.append(UInt8(0x21))
        buf.append(UInt8(0xF9))
        buf.append(UInt8(0x04))
        buf.append(UInt8(0x04))  # packed: disposal method 1, no transparency
        _put_u16_le(buf, delay_cs)
        buf.append(UInt8(0))     # transparent color index (unused)
        buf.append(UInt8(0))     # block terminator
        # Image Descriptor (full frame, no local color table)
        buf.append(UInt8(0x2C))
        _put_u16_le(buf, 0)
        _put_u16_le(buf, 0)
        _put_u16_le(buf, width)
        _put_u16_le(buf, height)
        buf.append(UInt8(0))
        # float → grayscale index
        indices.clear()
        var base = f * hw
        for p in range(hw):
            var v = Float32((frames + base + p)[])
            var idx = min(max((v - vmin) * scale, Float32(0.0)), Float32(255.0))
            indices.append(UInt8(idx))
        # LZW image data: min code size byte + packed sub-blocks
        buf.append(UInt8(8))  # LZW minimum code size (8-bit indices)
        var codes = _lzw_encode_gif(indices, 8)
        _append_subblocks(buf, codes)

    buf.append(UInt8(0x3B))  # trailer

    with open(path, "w") as f:
        f.write_bytes(buf)
    print(
        "Animated GIF saved: " + path + " (" + String(n_frames) + " frames, "
        + String(width) + "x" + String(height) + " @ " + String(fps) + " fps)"
    )
