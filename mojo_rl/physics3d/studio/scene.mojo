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


comptime PROP_BOX: Int = 0
comptime PROP_SPHERE: Int = 1
comptime PROP_CAPSULE: Int = 2
comptime PROP_CYLINDER: Int = 3

def _prop_mjcf_type(kind: Int) -> String:
    """MJCF `type=` for a prop kind.

    ⚠ A FUNCTION, NOT A COMPTIME ARRAY. `Array[String, 4]` is not
    `ImplicitlyCopyable`, so a comptime table cannot be indexed at runtime —
    the error names materialisation, not the lookup, which is the sort of
    thing worth writing down once.
    """
    if kind == PROP_SPHERE:
        return String("sphere")
    if kind == PROP_CAPSULE:
        return String("capsule")
    if kind == PROP_CYLINDER:
        return String("cylinder")
    return String("box")


def _prop_stem(kind: Int) -> String:
    if kind == PROP_SPHERE:
        return String("sphere")
    if kind == PROP_CAPSULE:
        return String("capsule")
    if kind == PROP_CYLINDER:
        return String("cylinder")
    return String("box")


@fieldwise_init
struct Prop(Copyable, ImplicitlyCopyable, Movable):
    """A PRIMITIVE dropped into the scene — box, sphere, capsule, cylinder.

    ⚠⚠ INLINE, NOT AN `<attach>`, AND THAT IS THE HONEST REPRESENTATION.
    `<attach model=X>` resolves against `<asset><model file=...>`, so an
    attached prop must exist as a FILE. A primitive the user just created has
    no file, and inventing one on disk to describe six numbers would make the
    scene depend on a directory the studio wrote behind the user's back.
    Inlining is also what §11.1 calls "materialize on override": a prop with
    no asset to unlink from is already materialised.

    ⇒ a prop is a `<body>` written straight into the scene's `<worldbody>`.
    Reusable assets still go through `<attach>`; the two coexist because both
    are just MJCF.

    ⚠ MASS AND INERTIA ARE NOT WRITTEN. MuJoCo derives both from the geom's
    shape and density when a body has no `<inertial>` — and so does our
    builder (`model/inertia_from_geom.mojo`). Emitting a `mass` we computed
    ourselves would be a SECOND implementation of that derivation, checkable
    against MuJoCo only by accident. `density` is the knob; the compiler does
    the rest.
    """

    var name: String
    """Unique in the scene. The body takes it; the geom takes `<name>_geom`."""
    var kind: Int
    var s0: Float64
    var s1: Float64
    var s2: Float64
    """MJCF `size`, per type: box = 3 half-extents, sphere = radius,
    capsule/cylinder = radius + half-length."""
    var px: Float64
    var py: Float64
    var pz: Float64
    var qw: Float64
    var qx: Float64
    var qy: Float64
    var qz: Float64
    var r: Float64
    var g: Float64
    var b: Float64
    var a: Float64
    var free: Bool
    """A free joint (it falls and collides) or welded to the world (a table,
    a wall, an obstacle).

    ⚠ A FREE JOINT IS ONLY LEGAL AT TOP LEVEL — a hard MuJoCo error nested
    inside a body, verified on 3.10.0. Props are written at world level, which
    keeps that unrepresentable rather than merely discouraged."""
    var density: Float64

    def __init__(out self, name: String, kind: Int, s0: Float64,
                 s1: Float64 = 0.0, s2: Float64 = 0.0):
        self.name = name
        self.kind = kind
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.r = 0.8
        self.g = 0.4
        self.b = 0.2
        self.a = 1.0
        self.free = True
        self.density = 1000.0

    def size_attr(self) -> String:
        """MJCF `size`, with only the components this type reads.

        ⚠ THE COUNT IS PER TYPE and MuJoCo checks it: a sphere with three
        numbers is an error, not a rounding. Emitting all three always would
        make every sphere and capsule fail to load.
        """
        if self.kind == PROP_SPHERE:
            return _fmt(self.s0)
        if self.kind == PROP_BOX:
            return _fmt(self.s0) + " " + _fmt(self.s1) + " " + _fmt(self.s2)
        return _fmt(self.s0) + " " + _fmt(self.s1)

    def to_mjcf(self) -> String:
        var s = String('    <body name="') + self.name + '" pos="'
        s += _fmt(self.px) + " " + _fmt(self.py) + " " + _fmt(self.pz)
        s += '" quat="' + _fmt(self.qw) + " " + _fmt(self.qx) + " "
        s += _fmt(self.qy) + " " + _fmt(self.qz) + '">\n'
        if self.free:
            s += '      <freejoint name="' + self.name + '_free"/>\n'
        s += '      <geom name="' + self.name + '_geom" type="'
        s += _prop_mjcf_type(self.kind) + '" size="' + self.size_attr()
        s += '" density="' + _fmt(self.density) + '" rgba="'
        s += _fmt(self.r) + " " + _fmt(self.g) + " " + _fmt(self.b) + " "
        s += _fmt(self.a) + '"/>\n    </body>\n'
        return s^


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
    """The scene's own sections OTHER than `<worldbody>` — `<option>`,
    `<compiler>`, `<visual>`, `<default>`, `<asset>` content it wrote itself."""

    var base_world: String
    """The INNER content of the scene's `<worldbody>`: floor, lights, cameras.

    ⚠⚠ SEPARATE FROM `base_xml` SO THE WRITER EMITS EXACTLY ONE
    `<worldbody>`. The first version appended a second one for the props, and
    MuJoCo accepts that — it merges repeated sections — while **our parser
    reads only the first**: a five-prop scene loaded as a bare floor, nbody 1.
    Caught by `test_props_vs_mujoco` on its first run.

    ⚠ THE PARSER GAP IS REAL AND STILL OPEN: a HAND-WRITTEN MJCF with two
    `<worldbody>` sections silently loses one here. `merge_mjcf` concatenates
    them correctly, so composed models are safe; a single file is not. Worth
    fixing in `full_parser` on its own justification — this writer simply
    stops relying on it."""
    var asset_names: List[String]
    var asset_files: List[String]
    var instances: List[Instance]
    var props: List[Prop]

    def __init__(out self):
        self.base_xml = String("")
        self.base_world = String("")
        self.asset_names = List[String]()
        self.asset_files = List[String]()
        self.instances = List[Instance]()
        self.props = List[Prop]()

    def add_asset(mut self, name: String, file: String):
        for i in range(len(self.asset_names)):
            if self.asset_names[i] == name:
                return
        self.asset_names.append(name)
        self.asset_files.append(file)

    def retarget_asset(mut self, file_from: String, file_to: String) -> Bool:
        """Point an asset entry at a different file. Returns False if absent.

        ⚠⚠ THIS IS §11.1's "MATERIALIZE ON OVERRIDE", at ASSET granularity.
        `<attach>` has no way to express a per-instance change, so a scene that
        merely REFERENCES a robot cannot carry an edit made to that robot's
        tree — reopening the scene would silently load the original. Writing
        the edited model beside the scene and re-pointing the entry at it is
        the honest answer: the scene stays MuJoCo-loadable, stays a
        composition, and names the file it actually means.

        ⚠ ASSET granularity, not INSTANCE. Every instance of an edited asset
        follows it, which is right for the one case the studio can currently
        reach (a single opened robot) and is the thing to revisit when a scene
        instantiates one asset twice and only one copy is edited.
        """
        for i in range(len(self.asset_files)):
            if self.asset_files[i] == file_from:
                self.asset_files[i] = file_to
                return True
        return False

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

    def unique_prop_name(self, stem: String) -> String:
        """`box1`, `box2`, … — unique against props AND instance prefixes.

        ⚠ AGAINST BOTH. A prop named `cube1` beside an instance prefixed
        `cube1_` is fine, but a prop named `cube1_root` would collide with the
        body that instance splices in — and MJCF duplicate names are a load
        error only sometimes, so the collision can survive as the WRONG
        element being referenced.
        """
        var n = 1
        while True:
            var nm = stem + String(n)
            var taken = False
            for i in range(len(self.props)):
                if self.props[i].name == nm:
                    taken = True
            for i in range(len(self.instances)):
                if self.instances[i].prefix.startswith(nm):
                    taken = True
            if not taken:
                return nm^
            n += 1

    def add_prop(mut self, kind: Int, s0: Float64, s1: Float64, s2: Float64,
                 px: Float64, py: Float64, pz: Float64) -> String:
        var p = Prop(self.unique_prop_name(_prop_stem(kind)), kind,
                     s0, s1, s2)
        p.px = px
        p.py = py
        p.pz = pz
        self.props.append(p)
        return self.props[len(self.props) - 1].name

    def duplicate_prop(mut self, name: String, dx: Float64 = 0.15) -> String:
        """Copy a prop, offset so the copy is VISIBLE rather than coincident.

        ⚠ THE OFFSET IS NOT COSMETIC. Two free-jointed bodies at identical
        poses start interpenetrating, and the solver pushes them apart with a
        large impulse on the first step — the copy shoots away and reads as a
        physics bug rather than as a duplicate placed on top of its original.
        """
        for i in range(len(self.props)):
            if self.props[i].name == name:
                var c = self.props[i].copy()
                c.name = self.unique_prop_name(_stem_of(name))
                c.px += dx
                self.props.append(c)
                return self.props[len(self.props) - 1].name
        return String("")

    def remove_prop(mut self, name: String):
        var keep = List[Prop]()
        for i in range(len(self.props)):
            if self.props[i].name != name:
                keep.append(self.props[i].copy())
        self.props = keep^

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
        # ⚠ ONE `<worldbody>`, ALWAYS. See `base_world`.
        var have_world = (
            _trim_local(self.base_world).byte_length() > 0
            or len(self.instances) > 0 or len(self.props) > 0
        )
        if have_world:
            s += "  <worldbody>\n"
            if _trim_local(self.base_world).byte_length() > 0:
                s += self.base_world + "\n"
            for i in range(len(self.props)):
                s += self.props[i].to_mjcf()
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
        if have_world:
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
        d.base_world = String(
            '    <light pos="0 0 3" dir="0 0 -1" directional="true"/>\n'
            '    <geom name="floor" type="plane" size="5 5 0.1"/>'
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


def _stem_of(name: String) -> String:
    """`box3` -> `box`. Trailing digits only, so `so_arm100` keeps its name."""
    var e = name.byte_length()
    while e > 0:
        var c = String(name[byte = e - 1 : e])
        if c < "0" or c > "9":
            break
        e -= 1
    return String(name[byte=0:e]) if e > 0 else name
