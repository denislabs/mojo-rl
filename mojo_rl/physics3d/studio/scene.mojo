"""The scene DOCUMENT — a base model plus a list of placed instances — S2.

## ⚠⚠ THE DOCUMENT IS MJCF. THERE IS NO SIDECAR AND NO CUSTOM FORMAT.

    <mujoco model="my_scene">
      <option timestep="0.002"/>            <!-- base instance only -->
      <asset>
        <model name="cube" file="props/cube.xml"/>
      </asset>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 .1"/>
        <frame pos="0.3 0 0.75"><attach model="cube" prefix="cube1_"/></frame>
      </worldbody>
    </mujoco>

The payoff is that **MuJoCo can load the scene file unchanged**, so `mjModel`
is the oracle for the entire composer — the same record-for-record gate style
that caught `meta[26]` and the `range(-1)` invweight bug. A proprietary format
would forfeit that on day one, and would also be the thing that blocks V2:
a scene format that cannot express a kinematic tree cannot later grow into a
robot editor. See `docs/PHYSICS3D_STUDIO_PLAN.md` §3 and §11.

## What an instance is

A `<frame>` carrying the pose, wrapping an `<attach>` carrying the identity:

* **the frame vanishes at compile time**, folding into its children, so an
  instance costs NO extra body and `nbody` matches the asset;
* **the prefix IS the instance identity**, and doubles as the key for state
  remap across a rebuild (plan §4).

⚠ PLACEMENT IS DELIBERATELY NOT AN OVERRIDE. `pos`/`quat` live in the
`<frame>`, so moving, rotating and duplicating — the overwhelmingly common
edits — never unlink an instance from its asset. Only editing a PROPERTY
does, and that is S3's "materialize on override".

⚠ A FREE-JOINTED PROP MAY ONLY BE INSTANTIATED AT WORLD LEVEL. Attaching one
inside a `<body>` is a hard MuJoCo error ("free joint can only be used on top
level"), verified on 3.10.0. This writer only emits world-level frames, which
keeps that unrepresentable rather than merely unadvised.
"""

def _fmt(v: Float64) -> String:
    """FULL precision. ⚠ NOT the panel's display formatter.

    A pose written at four decimals moves the instance by up to 5e-5, and a
    quaternion so written is not a unit quaternion — MuJoCo renormalises it
    and the body ends up rotated by a slightly different amount than the
    document says. The scene file is DATA, so it round-trips exactly; the
    panel's `_f` is for a human reading a number off a screen and belongs
    nowhere near a serialiser.
    """
    return String(v)


@fieldwise_init
struct Instance(Copyable, Movable):
    """One placed asset. `prefix` is its identity."""

    var asset: String
    """Key into the scene's `<asset><model>` table."""
    var prefix: String
    var body: String
    """`<attach body=>` — attach ONE body of a multi-root asset, or "" for
    the whole worldbody. Whole-worldbody attach brings every root plus the
    asset's actuators, sensors and excludes; verified on MuJoCo 3.10.0."""
    var px: Float64
    var py: Float64
    var pz: Float64
    var qw: Float64
    var qx: Float64
    var qy: Float64
    var qz: Float64

    def __init__(out self, asset: String, prefix: String,
                 px: Float64 = 0.0, py: Float64 = 0.0, pz: Float64 = 0.0):
        self.asset = asset
        self.prefix = prefix
        self.body = String("")
        self.px = px
        self.py = py
        self.pz = pz
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0


struct SceneDoc(Movable):
    """A scene: the base model's own text, an asset table, and instances.

    ⚠ THE BASE IS TEXT, NOT A PARSE. A scene composes assets around a robot
    the user did not author here, and round-tripping that robot through our
    own writer would silently drop whatever our parser does not model. Keeping
    it verbatim means the scene file can only ever LOSE something the studio
    itself added.
    """

    var base_xml: String
    """The scene's own `<mujoco>` body, minus the asset table and instances —
    the floor, lights, cameras, `<option>`, and any robot declared inline."""
    var asset_names: List[String]
    var asset_files: List[String]
    var instances: List[Instance]

    def __init__(out self):
        self.base_xml = String("")
        self.asset_names = List[String]()
        self.asset_files = List[String]()
        self.instances = List[Instance]()

    def add_asset(mut self, name: String, file: String):
        for i in range(len(self.asset_names)):
            if self.asset_names[i] == name:
                return
        self.asset_names.append(name)
        self.asset_files.append(file)

    def unique_prefix(self, asset: String) -> String:
        """`cube1_`, `cube2_`, … — never colliding with a live instance.

        ⚠ THE TRAILING UNDERSCORE IS PART OF IT. MuJoCo concatenates the
        prefix with the sub-model's names verbatim, so `cube1` + `root` is
        `cube1root`. Every Menagerie example writes the underscore, and
        omitting it produces names that load and read as typos.
        """
        var n = 1
        while True:
            var p = asset + String(n) + "_"
            var taken = False
            for i in range(len(self.instances)):
                if self.instances[i].prefix == p:
                    taken = True
            if not taken:
                return p^
            n += 1

    def place(mut self, asset: String, px: Float64, py: Float64,
              pz: Float64) -> String:
        """Add an instance of `asset`; returns its prefix."""
        var p = self.unique_prefix(asset)
        self.instances.append(Instance(asset, p, px, py, pz))
        return p^

    def remove(mut self, prefix: String):
        var keep = List[Instance]()
        for i in range(len(self.instances)):
            if self.instances[i].prefix != prefix:
                keep.append(self.instances[i].copy())
        self.instances = keep^

    def find(self, prefix: String) -> Int:
        for i in range(len(self.instances)):
            if self.instances[i].prefix == prefix:
                return i
        return -1

    def to_mjcf(self, model_name: String = String("scene")) -> String:
        """Serialise. The result is a file MuJoCo loads and the expander reads.

        ⚠ THE ASSET TABLE IS DECLARED UP FRONT AND THAT IS NOT A STYLE
        CHOICE: `<attach model=>` resolves against `<asset><model name=>`, and
        declaring per instance would stop MuJoCo loading the file — which
        forfeits the oracle this whole design exists for.
        """
        var s = String('<mujoco model="') + model_name + '">\n'
        if _trim_local(self.base_xml).byte_length() > 0:
            s += self.base_xml + "\n"
        if len(self.asset_names) > 0:
            s += "  <asset>\n"
            for i in range(len(self.asset_names)):
                s += (
                    '    <model name="' + self.asset_names[i]
                    + '" file="' + self.asset_files[i] + '"/>\n'
                )
            s += "  </asset>\n"
        if len(self.instances) > 0:
            s += "  <worldbody>\n"
            for i in range(len(self.instances)):
                ref it = self.instances[i]
                s += (
                    '    <frame pos="' + _fmt(it.px) + " " + _fmt(it.py)
                    + " " + _fmt(it.pz) + '" quat="' + _fmt(it.qw) + " "
                    + _fmt(it.qx) + " " + _fmt(it.qy) + " " + _fmt(it.qz)
                    + '">\n      <attach model="' + it.asset + '"'
                )
                if it.body.byte_length() > 0:
                    s += ' body="' + it.body + '"'
                s += ' prefix="' + it.prefix + '"/>\n    </frame>\n'
            s += "  </worldbody>\n"
        s += "</mujoco>\n"
        return s^


def _trim_local(s: String) -> String:
    var a = 0
    var b = s.byte_length()
    while a < b:
        var c = String(s[byte = a : a + 1])
        if c != " " and c != "\n" and c != "\t" and c != "\r":
            break
        a += 1
    while b > a:
        var c = String(s[byte = b - 1 : b])
        if c != " " and c != "\n" and c != "\t" and c != "\r":
            break
        b -= 1
    return String(s[byte=a:b])


def scene_from_base(
    base_path: String, floor: Bool = True
) raises -> SceneDoc:
    """A scene whose base is a floor, ready for props.

    The studio's "new scene" — deliberately NOT "load a robot and call it a
    scene", because the base is the one part a user did not author here.
    """
    var d = SceneDoc()
    if floor:
        d.base_xml = String(
            '  <worldbody>\n'
            '    <light pos="0 0 3" dir="0 0 -1" directional="true"/>\n'
            '    <geom name="floor" type="plane" size="5 5 0.1"/>\n'
            "  </worldbody>"
        )
    if base_path.byte_length() > 0:
        # The robot is an ASSET like any other, so it can be moved and
        # duplicated with the same machinery as a prop.
        var nm = _stem(base_path)
        d.add_asset(nm, base_path)
        _ = d.place(nm, 0.0, 0.0, 0.0)
    return d^


def _stem(path: String) -> String:
    var base = path
    var cut = path.rfind("/")
    if cut >= 0:
        base = String(path[byte = cut + 1 : path.byte_length()])
    var dot = base.rfind(".")
    if dot <= 0:
        return base^
    return String(base[byte=0:dot])
