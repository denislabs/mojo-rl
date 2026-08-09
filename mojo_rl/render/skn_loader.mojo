"""MuJoCo `.skn` skin loader — the deformable envelope, not a rigid mesh.

    var skin = load_skn("dog_skin.skn")     # 24065 verts, 33900 faces, 57 bones

WHAT A SKIN IS, AND WHY IT IS NOT A MESH GEOM. A mesh geom rides exactly ONE
body: the renderer builds one model matrix and the GPU does the rest, which is
why `load_stl` + `draw_mesh` need nothing per frame. A skin vertex belongs to
SEVERAL bodies at once — 2.62 of them on average in dog's case — with a weight
per (bone, vertex) pair. A weighted blend of transforms is not a matrix, so the
mesh has to be re-deformed on the CPU every frame and re-uploaded. That is the
entire reason this file exists next to `stl_loader.mojo` rather than inside it.

⚠ THE FORMAT IS TRANSCRIBED FROM `mjCSkin::LoadSKN`
(`references/mujoco-3.6.0/src/user/user_mesh.cc:3170`), not guessed:

    int32   nvert, ntexcoord, nface, nbone
    float32 vert[3*nvert]
    float32 texcoord[2*ntexcoord]
    int32   face[3*nface]
    per bone:
        char    bodyname[40]      ⚠ 40 BYTES READ, 10 FLOATS ADVANCED
        float32 bindpos[3]
        float32 bindquat[4]       (w, x, y, z — MuJoCo's order)
        int32   vertid_count
        int32   vertid[count]
        float32 vertweight[count]

⚠ THE NAME FIELD IS 40 BYTES BUT THE CURSOR MOVES 10 FLOATS — which is 40
bytes, so they agree, but MuJoCo writes it as `strncpy(txt, ..., 39); cnt += 10`
in two different units and it reads like an off-by-one waiting to happen. It is
not: 10 floats * 4 = 40. Getting this wrong desynchronises every bone after the
first, and the failure is silent — you get a plausible-looking skin whose limbs
follow the wrong bodies.

The gate for all of it is that the cursor must land EXACTLY on the end of the
file. `load_skn` raises when it does not, and `tests/render/test_skn_loader.mojo`
pins the counts a Python transcription of the same format produced.
"""

from std.memory import Pointer

comptime SKN_BONE_NAME_BYTES: Int = 40
"""Bytes in a bone's `bodyname` field. See the header — MuJoCo advances its
cursor by 10 floats, which is this many bytes."""


struct SkinBone(Movable):
    """One bone: the body it follows, its bind pose, and its vertex weights."""

    var body_name: String
    """Name of the BODY this bone tracks. Resolved to an index by the caller,
    which is the only part of a skin that needs to know about a model."""

    var bind_pos_x: Float32
    var bind_pos_y: Float32
    var bind_pos_z: Float32

    var bind_quat_w: Float32
    var bind_quat_x: Float32
    var bind_quat_y: Float32
    var bind_quat_z: Float32
    """The bone's pose when the rest vertices were authored. Deformation is
    relative to it: `rotate = xquat[body] * conj(bindquat)`."""

    var vert_ids: List[Int32]
    var weights: List[Float32]
    """Parallel arrays, one entry per vertex this bone influences. A vertex
    appears in several bones' lists; its weights across them sum to 1."""

    def __init__(out self):
        self.body_name = String("")
        self.bind_pos_x = 0
        self.bind_pos_y = 0
        self.bind_pos_z = 0
        self.bind_quat_w = 1
        self.bind_quat_x = 0
        self.bind_quat_y = 0
        self.bind_quat_z = 0
        self.vert_ids = List[Int32]()
        self.weights = List[Float32]()

    def __init__(out self, *, deinit move: Self):
        self.body_name = move.body_name^
        self.bind_pos_x = move.bind_pos_x
        self.bind_pos_y = move.bind_pos_y
        self.bind_pos_z = move.bind_pos_z
        self.bind_quat_w = move.bind_quat_w
        self.bind_quat_x = move.bind_quat_x
        self.bind_quat_y = move.bind_quat_y
        self.bind_quat_z = move.bind_quat_z
        self.vert_ids = move.vert_ids^
        self.weights = move.weights^


struct SkinData(Movable):
    """A loaded `.skn`: rest geometry plus the bones that deform it."""

    var nvert: Int
    var nface: Int
    var vert: List[Float32]
    """Rest positions, 3 per vertex. NEVER mutated — every frame re-derives the
    posed vertices from these, so a drifting accumulation is impossible."""
    var texcoord: List[Float32]
    """2 per vertex, or empty when the file carries none."""
    var face: List[Int32]
    """Triangle indices, 3 per face."""
    var bones: List[SkinBone]

    def __init__(out self):
        self.nvert = 0
        self.nface = 0
        self.vert = List[Float32]()
        self.texcoord = List[Float32]()
        self.face = List[Int32]()
        self.bones = List[SkinBone]()

    def __init__(out self, *, deinit move: Self):
        self.nvert = move.nvert
        self.nface = move.nface
        self.vert = move.vert^
        self.texcoord = move.texcoord^
        self.face = move.face^
        self.bones = move.bones^

    def has_texcoords(self) -> Bool:
        return len(self.texcoord) == 2 * self.nvert

    def total_weights(self) -> Int:
        """Sum of every bone's vertex count — the per-frame skinning work."""
        var n = 0
        for i in range(len(self.bones)):
            n += len(self.bones[i].vert_ids)
        return n


def load_skn(path: String) raises -> SkinData:
    """Parse a binary MuJoCo `.skn`. Raises on anything that does not add up.

    ⚠ EVERY SIZE IS CHECKED BEFORE IT IS TRUSTED, and the cursor must finish on
    the last byte. A skin that parses "successfully" from a misread header does
    not crash — it renders a mangled envelope that looks like a physics bug.
    """
    var f = open(path, "r")
    var content = f.read_bytes()
    f.close()

    var n_bytes = len(content)
    if n_bytes < 16:
        raise Error(
            "SKN '" + path + "' is too small to hold a header: "
            + String(n_bytes) + " bytes"
        )

    var base = content.unsafe_ptr()

    @parameter
    def i32_at(byte_off: Int) -> Int:
        return Int(
            (base.unsafe_offset(byte_off)).unsafe_bitcast[Int32]()[]
        )

    @parameter
    def f32_at(byte_off: Int) -> Float32:
        return (base.unsafe_offset(byte_off)).unsafe_bitcast[Float32]()[]

    var nvert = i32_at(0)
    var ntexcoord = i32_at(4)
    var nface = i32_at(8)
    var nbone = i32_at(12)

    if nvert < 0 or ntexcoord < 0 or nface < 0 or nbone < 0:
        raise Error("SKN '" + path + "' has a negative count in its header")

    var geom_end = 16 + 12 * nvert + 8 * ntexcoord + 12 * nface
    if n_bytes < geom_end:
        raise Error(
            "SKN '" + path + "' truncated: header wants " + String(geom_end)
            + " bytes of geometry, file is " + String(n_bytes)
        )

    var skin = SkinData()
    skin.nvert = nvert
    skin.nface = nface

    var off = 16
    skin.vert.reserve(3 * nvert)
    for i in range(3 * nvert):
        skin.vert.append(f32_at(off + 4 * i))
    off += 12 * nvert

    skin.texcoord.reserve(2 * ntexcoord)
    for i in range(2 * ntexcoord):
        skin.texcoord.append(f32_at(off + 4 * i))
    off += 8 * ntexcoord

    skin.face.reserve(3 * nface)
    for i in range(3 * nface):
        skin.face.append(Int32(i32_at(off + 4 * i)))
    off += 12 * nface

    for b in range(nbone):
        # 40 (name) + 12 (bindpos) + 16 (bindquat) + 4 (count) = 72 bytes
        # before the variable-length arrays.
        if off + 72 > n_bytes:
            raise Error(
                "SKN '" + path + "' truncated in bone " + String(b)
            )

        var bone = SkinBone()

        # The field is a fixed 40 bytes, NUL-padded; the name is whatever
        # precedes the first NUL.
        var name_len = 0
        while (
            name_len < SKN_BONE_NAME_BYTES
            and content[off + name_len] != UInt8(0)
        ):
            name_len += 1
        var name_bytes = List[UInt8]()
        for i in range(name_len):
            name_bytes.append(content[off + i])
        name_bytes.append(UInt8(0))
        bone.body_name = String(unsafe_from_utf8_ptr=name_bytes.unsafe_ptr())
        off += SKN_BONE_NAME_BYTES

        bone.bind_pos_x = f32_at(off)
        bone.bind_pos_y = f32_at(off + 4)
        bone.bind_pos_z = f32_at(off + 8)
        off += 12

        bone.bind_quat_w = f32_at(off)
        bone.bind_quat_x = f32_at(off + 4)
        bone.bind_quat_y = f32_at(off + 8)
        bone.bind_quat_z = f32_at(off + 12)
        off += 16

        var vcount = i32_at(off)
        off += 4
        if vcount < 1:
            raise Error(
                "SKN '" + path + "' bone " + String(b)
                + " claims " + String(vcount) + " vertices"
            )
        if off + 8 * vcount > n_bytes:
            raise Error(
                "SKN '" + path + "' truncated in bone " + String(b)
                + "'s vertex arrays"
            )

        bone.vert_ids.reserve(vcount)
        for i in range(vcount):
            var vid = i32_at(off + 4 * i)
            if vid < 0 or vid >= nvert:
                raise Error(
                    "SKN '" + path + "' bone " + String(b)
                    + " references vertex " + String(vid)
                    + ", outside [0, " + String(nvert) + ")"
                )
            bone.vert_ids.append(Int32(vid))
        off += 4 * vcount

        bone.weights.reserve(vcount)
        for i in range(vcount):
            bone.weights.append(f32_at(off + 4 * i))
        off += 4 * vcount

        skin.bones.append(bone^)

    # ⚠ THE REAL GATE. Every check above catches a file that is too SHORT; only
    # this one catches a misread STRIDE, which is the failure that renders
    # rather than raises.
    if off != n_bytes:
        raise Error(
            "SKN '" + path + "' parsed to byte " + String(off)
            + " but the file is " + String(n_bytes)
            + " — the bone stride is being read wrong"
        )

    return skin^
