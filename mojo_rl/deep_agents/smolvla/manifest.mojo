"""Reader for `tools/vla/smolvla_base_manifest.tsv` — the checkpoint checklist.

The manifest is the 500-tensor index of `lerobot/smolvla_base`, fetched with two
HTTP Range requests (73 KB) instead of the 906,712,520-byte weight file, so a
name-map gate needs neither the network nor a download.

Lives here rather than inside one test because two gates read it (vision, text)
and a third will (the expert). A parser copied per caller is the shape that
drifts — one copy learns to skip a comment line and the other does not, and the
two gates then disagree about what the checkpoint contains.

Format, one tensor per line, `#` comments skipped:

    <name>\\t<dtype>\\t<d0,d1,...>
"""

from mojo_rl.io.fileio import read_file_bytes

comptime SMOLVLA_MANIFEST = String("tools/vla/smolvla_base_manifest.tsv")


struct Manifest(Movable):
    var names: List[String]
    var dtypes: List[String]
    var shapes: List[List[Int]]

    def __init__(out self, path: String = SMOLVLA_MANIFEST) raises:
        self.names = List[String]()
        self.dtypes = List[String]()
        self.shapes = List[List[Int]]()
        var raw = read_file_bytes(path)
        var text = String(from_utf8=Span(raw))
        for line in text.split(String("\n")):
            if line.byte_length() == 0 or line.startswith(String("#")):
                continue
            var parts = line.split(String("\t"))
            if len(parts) < 3:
                continue
            var dims = List[Int]()
            for d in parts[2].split(String(",")):
                dims.append(Int(d))
            self.names.append(String(parts[0]))
            self.dtypes.append(String(parts[1]))
            self.shapes.append(dims^)

    def size(self) -> Int:
        return len(self.names)

    def index_of(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def same_shape(self, i: Int, ref want: List[Int]) -> Bool:
        ref got = self.shapes[i]
        if len(want) != len(got):
            return False
        for k in range(len(want)):
            if want[k] != got[k]:
                return False
        return True


def shape_str(ref s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ", "
        out += String(s[i])
    return out + "]"
