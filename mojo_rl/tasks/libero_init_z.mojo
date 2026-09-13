"""LIBERO's own start height per (prop, region) — `init_z_<family>.kv`.

    from mojo_rl.tasks.libero_init_z import load_init_z
    var z = load_init_z(String("libero_spatial"))
    var h = z.height(String("akita_black_bowl_1"),
                     String("wooden_cabinet_1_top_region"))   # 1.15063

## ⚠⚠ WHY THIS IS DATA AND NOT A FORMULA

`sampler.sample_placements` computes a resting height from the region's site
and the asset's `bottom_site`, and a gate that recomputed that formula would be
checking the sampler against itself. These numbers come from LIBERO's
`.pruned_init` files — the fifty frozen states the benchmark restores at reset
— extracted by `tools/tasks/libero_init_z.py`, which validates its own column
layout structurally (a free joint's quaternion must be unit norm) and never
reads anything of ours.

They are what caught `spec.TABLE_Z_OFFSET`: the sampler had the FIXTURE
sampler's `z_offset` (0.0) where a table or floor region takes
`TableRegionSampler`'s 0.01, so every prop of every family started a centimetre
low.

## ⚠ KEYED BY THE `.bddl`'s OWN NAMES

The region is the `.bddl`'s composed name (`wooden_cabinet_1_top_region`), or
another PROP's name when the `:init` stacks one thing on another. It is NOT the
family's region name, because a union family renames a region whose role moves
between rectangles (`libero_import.RegionAlias`) — so a caller resolves the key
from the corpus, and both sides stay in LIBERO's vocabulary.

## ⚠ ONE HEIGHT PER (PROP, REGION), NOT PER PROP

`libero_spatial`'s `akita_black_bowl_1` starts at 0.9700 on the table, 1.0100
on the stove's cook region, 1.0800 stacked on the cookie box, 1.1506 inside the
cabinet's top drawer and 1.2315 on its roof. robosuite's height is
`z_offset + base_offset[2] - bottom_offset[-1]` and `base_offset` is the
REGION's reference, so the pair is the key and the prop alone is not.
"""

from std.os.path import exists

from mojo_rl.core.kv import split_on


comptime INIT_Z_DIR: String = "mojo_rl/tasks/libero"
comptime INIT_Z_SCHEMA: Int = 1


struct InitZTable(Movable & Deinitable):
    """The frozen heights of one family, by (prop, region)."""

    var family: String
    var prop: List[String]
    var region: List[String]
    var z: List[Float64]

    def __init__(out self):
        self.family = String("")
        self.prop = List[String]()
        self.region = List[String]()
        self.z = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.family = move.family^
        self.prop = move.prop^
        self.region = move.region^
        self.z = move.z^

    def index(self, prop: String, region: String) -> Int:
        for i in range(len(self.prop)):
            if self.prop[i] == prop and self.region[i] == region:
                return i
        return -1

    def height(self, prop: String, region: String) raises -> Float64:
        """⚠ RAISES when the pair is absent rather than returning a default.
        A missing entry means the corpus places that prop somewhere this file
        was not generated from, and a defaulted height would make the
        comparison pass for the wrong reason."""
        var i = self.index(prop, region)
        if i < 0:
            raise Error(
                "libero init_z: family '" + self.family + "' has no frozen"
                " height for '" + prop + "' on '" + region + "'. The key is"
                " the `.bddl`'s own names; re-run `pixi run libero-init-z`"
                " for this family."
            )
        return self.z[i]


def init_z_path(family: String) -> String:
    return String(INIT_Z_DIR) + "/init_z_" + family + ".kv"


def has_init_z(family: String) -> Bool:
    return exists(init_z_path(family))


def load_init_z(family: String) raises -> InitZTable:
    """Parse `init_z_<family>.kv`. RAISES on anything unexpected."""
    var path = init_z_path(family)
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    var out = InitZTable()
    var saw_version = False
    var lineno = 0
    for line in text.splitlines():
        lineno += 1
        var l = String(line).strip()
        if l.byte_length() == 0 or l.startswith("#"):
            continue
        var eq = l.find("=")
        if eq <= 0:
            raise Error(
                path + ":" + String(lineno) + ": '" + l + "' is not key=value"
            )
        var key = String(l[byte=0:eq])
        var val = String(l[byte = eq + 1 : l.byte_length()])
        if key == "schema_version":
            saw_version = True
            if Int(val) > INIT_Z_SCHEMA:
                raise Error(
                    path + ": schema_version " + val + " is newer than this"
                    " build supports (" + String(INIT_Z_SCHEMA) + ")"
                )
        elif key == "family":
            out.family = val^
        elif key == "z":
            # `<prop>@<region>:<float>`
            var colon = val.rfind(":")
            var at = val.find("@")
            if colon <= 0 or at <= 0 or at > colon:
                raise Error(
                    path + ":" + String(lineno) + ": expected"
                    " `z=<prop>@<region>:<float>`, got '" + val + "'"
                )
            out.prop.append(String(val[byte=0:at]))
            out.region.append(String(val[byte = at + 1 : colon]))
            out.z.append(Float64(String(val[byte = colon + 1 : val.byte_length()])))
        else:
            raise Error(
                path + ":" + String(lineno) + ": unknown key '" + key
                + "' (known: schema_version, family, z)"
            )
    if not saw_version:
        raise Error(path + ": no schema_version line")
    if out.family != family:
        raise Error(
            path + ": says family=" + out.family + " but was loaded as "
            + family
        )
    if len(out.prop) == 0:
        raise Error(path + ": no z= lines")
    return out^
